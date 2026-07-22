"""
Consolidated utilities for masked-model experiments.

Provides default configs (matching CART gold-standard), training wrappers
that delegate to pxl_utils.train_model / get_model_latents, imputation
evaluation, and synthetic data generation.
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata
import anndata as ad

sys.path.insert(0, "..")
# Cached models were saved with 'PixelGen.*' module paths in their pickle;
# adding the grandparent directory lets torch.load resolve them.
_grandparent = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _grandparent not in sys.path:
    sys.path.insert(0, _grandparent)

from multimodalvi import MultiModalSCVI
from enums import D, AggMethod
from pxl_utils import train_model, get_model_latents
from utils import mask_adata


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _align_obsm_to_model(adata, save_path):
    """Align obsm DataFrame columns to match what a saved model expects.

    Reads the model's registry to discover expected column names for each
    obsm field, then reindexes the corresponding adata.obsm DataFrames
    so that scvi-tools' transfer validation passes.
    """
    import torch

    model_pt = os.path.join(save_path, "model.pt")
    if not os.path.isfile(model_pt):
        return

    state = torch.load(model_pt, map_location="cpu", weights_only=False)
    registries = state.get("attr_dict", {}).get("registry_", {}).get("field_registries", {})

    for field_key, field_reg in registries.items():
        state_reg = field_reg.get("state_registry", {})
        expected_cols = state_reg.get("column_names", None)
        attr_key = field_reg.get("data_registry", {}).get("attr_key", None)

        if expected_cols is None or attr_key is None:
            continue

        expected_cols = list(expected_cols)

        # obsm fields
        if attr_key in adata.obsm and hasattr(adata.obsm[attr_key], "columns"):
            current_cols = list(adata.obsm[attr_key].columns)
            if current_cols != expected_cols:
                # Try to reindex from a wider source (e.g. full HOTSPOT)
                # or from the obsm itself if columns are a subset
                available = set(adata.obsm[attr_key].columns)
                # Also check if a parent obsm has all columns
                # (e.g. HOTSPOT_top500_var comes from HOTSPOT)
                base_key = attr_key.replace("_top500_var", "")
                source_df = None
                if base_key != attr_key and base_key in adata.obsm and hasattr(adata.obsm[base_key], "columns"):
                    source_df = adata.obsm[base_key]
                elif all(c in available for c in expected_cols):
                    source_df = adata.obsm[attr_key]

                if source_df is not None and all(c in source_df.columns for c in expected_cols):
                    print(f"  Aligning obsm['{attr_key}'] columns to match saved model")
                    adata.obsm[attr_key] = source_df[expected_cols]

        # layer fields
        if attr_key in (adata.layers if hasattr(adata, "layers") else {}):
            pass  # layers don't have column name issues


def load_or_train(save_path, train_fn, adata, force_train=False):
    """Load a cached model from *save_path* if it exists, otherwise train.

    Parameters
    ----------
    save_path : str or None
        Directory where the model is (or will be) saved.
        If None, always trains without saving.
    train_fn : callable
        ``train_fn(adata)`` → trained model.
    adata : AnnData
        Data passed to ``train_fn`` and to ``MultiModalSCVI.load``.
    force_train : bool
        If True, always retrain even when a cache exists.

    Returns
    -------
    model : MultiModalSCVI
    """
    if save_path and not force_train and os.path.isdir(save_path):
        print(f"Loading cached model from {save_path}")
        # Align obsm columns to match what the saved model expects
        _align_obsm_to_model(adata, save_path)
        # Try loading with provided adata first; if var count mismatches
        # (e.g. spatial-only models saved with spatial X), fall back to
        # the saved adata bundled inside the model directory.
        try:
            model = MultiModalSCVI.load(save_path, adata=adata)
        except ValueError:
            saved_adata = os.path.join(save_path, "adata.h5ad")
            if os.path.isfile(saved_adata):
                print("  var mismatch — loading with saved adata instead")
                model = MultiModalSCVI.load(save_path, adata=None)
            else:
                raise
        return model

    print(f"Training model{f' (will save to {save_path})' if save_path else ''} ...")
    model = train_fn(adata)

    if save_path:
        model.save(save_path, overwrite=True, save_anndata=True)
        print(f"Saved to {save_path}")

    return model


# ---------------------------------------------------------------------------
# Default configs — match CART notebook (carT_PNA_experiment.ipynb) exactly
# ---------------------------------------------------------------------------

def get_default_model_kwargs(n_modalities=1, agg_method=None):
    """Return CART-standard model_kwargs.

    Parameters
    ----------
    n_modalities : int
        1 for unimodal, 2 for joint.
    agg_method : AggMethod or None
        If None and n_modalities==2, defaults to AOE_GLOBAL_WEIGHTS
        (separate encoders, weighted PoE).

    Returns
    -------
    dict
        Ready to unpack into ``MultiModalSCVI(...)``.
    """
    distrs = [D.Normal] * n_modalities

    kwargs = dict(
        n_latent=30,
        n_hidden=128,
        n_layers=2 if n_modalities >= 2 else 1,
        dropout_rate=0.1,
        distrs=distrs,
        loss_weights="auto",
        external_kl_weight=1,
        decoder_kwargs=dict(decoder_param_eps=1e-2, decoder_activation="exp"),
    )

    if n_modalities >= 2:
        if agg_method is AggMethod.SHARED_ENCODER:
            kwargs.update(
                agg_method=AggMethod.SHARED_ENCODER,
                joint_kl=True,
                unimodal_kl=False,
            )
        else:
            # Default: weighted PoE (separate encoders)
            kwargs.update(
                joint_kl=False,
                unimodal_kl=True,
            )

    return kwargs


def get_default_train_kwargs(max_epochs=10000):
    """Return CART-standard train_kwargs."""
    return dict(
        train_size=0.8,
        check_val_every_n_epoch=1,
        early_stopping=True,
        early_stopping_patience=200,
        batch_size=2000,
        max_epochs=max_epochs,
        enable_checkpointing=True,
        plan_kwargs=dict(lr=1e-4, optimizer="Adam", n_epochs_kl_warmup=400),
    )


# ---------------------------------------------------------------------------
# Training wrappers — delegate to pxl_utils.train_model
# ---------------------------------------------------------------------------

def run_abundance_model(
    adata,
    ab_layer,
    model_name,
    batch_key=None,
    max_epochs=10000,
    save_path=None,
    force_train=False,
    **overrides,
):
    """Train (or load cached) abundance-only (unimodal) model.

    Returns the trained model and stores latents in
    ``adata.obsm[f'z_{model_name}']``.
    """
    setup_kwargs = dict(layer=ab_layer, n_modalities=1, batch_key=batch_key)
    model_kwargs = get_default_model_kwargs(n_modalities=1)
    train_kwargs = get_default_train_kwargs(max_epochs)

    model_kwargs.update(overrides.pop("model_kwargs", {}))
    train_kwargs.update(overrides.pop("train_kwargs", {}))

    def _train(ad):
        return train_model(
            ad, model_cls=MultiModalSCVI,
            setup_kwargs=setup_kwargs, model_kwargs=model_kwargs,
            train_kwargs=train_kwargs,
        )

    model = load_or_train(save_path, _train, adata, force_train=force_train)

    latent_name = f"z_{model_name}"
    get_model_latents(adata, model, [("joint", latent_name)])
    return model


def _make_spatial_adata(adata, sp_layer, batch_key=None):
    """Create a standalone AnnData with spatial features as X.

    Spatial-only models need X to be the spatial matrix (e.g. 500 features),
    not the protein abundance matrix (159 features).
    """
    X_spatial = adata.obsm[sp_layer]
    if hasattr(X_spatial, "columns"):
        features = X_spatial.columns.tolist()
        X_vals = X_spatial.values
    else:
        features = [f"spatial_{i}" for i in range(X_spatial.shape[1])]
        X_vals = np.asarray(X_spatial)

    obs = adata.obs.copy()
    spatial_adata = anndata.AnnData(
        X=pd.DataFrame(X_vals, index=adata.obs_names, columns=features),
        obs=obs,
        var=pd.DataFrame(index=features),
        layers={sp_layer: pd.DataFrame(X_vals, index=adata.obs_names, columns=features)},
    )
    return spatial_adata


def run_spatial_model(
    adata,
    sp_layer,
    model_name,
    batch_key=None,
    max_epochs=10000,
    save_path=None,
    force_train=False,
    **overrides,
):
    """Train (or load cached) spatial-only (unimodal) model.

    A separate AnnData is built with the spatial obsm as X, since
    spatial-only models expect n_vars == n_spatial_features.
    """
    setup_kwargs = dict(layer=sp_layer, n_modalities=1, batch_key=batch_key)
    model_kwargs = get_default_model_kwargs(n_modalities=1)
    train_kwargs = get_default_train_kwargs(max_epochs)

    model_kwargs.update(overrides.pop("model_kwargs", {}))
    train_kwargs.update(overrides.pop("train_kwargs", {}))

    spatial_adata = _make_spatial_adata(adata, sp_layer, batch_key)

    def _train(ad):
        return train_model(
            ad, model_cls=MultiModalSCVI,
            setup_kwargs=setup_kwargs, model_kwargs=model_kwargs,
            train_kwargs=train_kwargs,
        )

    model = load_or_train(save_path, _train, spatial_adata, force_train=force_train)

    latent_name = f"z_{model_name}"
    # Extract latents into the *original* adata for downstream use
    adata.obsm[latent_name] = model.get_latent_representation(modality="joint")
    return model


def run_joint_model(
    adata,
    ab_layer,
    sp_layer,
    model_name,
    method="weighted",
    batch_key=None,
    max_epochs=10000,
    spatial_mask_key=None,
    save_path=None,
    force_train=False,
    **overrides,
):
    """Train (or load cached) joint (bimodal) model.

    Parameters
    ----------
    method : str
        ``'weighted'``  → AOE_GLOBAL_WEIGHTS, joint_kl=False, unimodal_kl=True
        ``'shared_encoder'`` → SHARED_ENCODER, joint_kl=True, unimodal_kl=False
    spatial_mask_key : str or None
        obs column with boolean mask (from ``mask_adata``).
    save_path : str or None
        Directory for caching. Loaded if exists, saved after training.
    force_train : bool
        If True, retrain even when cache exists.
    """
    agg = AggMethod.SHARED_ENCODER if method == "shared_encoder" else None

    setup_kwargs = dict(
        layer=ab_layer,
        extra_modality_keys=[sp_layer],
        n_modalities=2,
        batch_key=batch_key,
        spatial_mask_key=spatial_mask_key,
    )
    model_kwargs = get_default_model_kwargs(n_modalities=2, agg_method=agg)
    train_kwargs = get_default_train_kwargs(max_epochs)

    model_kwargs.update(overrides.pop("model_kwargs", {}))
    train_kwargs.update(overrides.pop("train_kwargs", {}))

    def _train(ad):
        return train_model(
            ad, model_cls=MultiModalSCVI,
            setup_kwargs=setup_kwargs, model_kwargs=model_kwargs,
            train_kwargs=train_kwargs,
        )

    model = load_or_train(save_path, _train, adata, force_train=force_train)

    # Extract latents — shared encoder only has joint, not per-modality
    latents = [("joint", f"z_{model_name}_joint")]
    if method != "shared_encoder":
        latents.extend([
            (ab_layer, f"z_{model_name}_ab"),
            (sp_layer, f"z_{model_name}_sp"),
        ])
    get_model_latents(adata, model, latents)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_imputation(adata_original, adata_masked, model, spatial_key, mask_col="spatial_masked"):
    """Compare reconstruction quality on masked vs. unmasked cells.

    Parameters
    ----------
    adata_original : AnnData
        Original (unmasked) data with ground-truth spatial in ``obsm[spatial_key]``.
    adata_masked : AnnData
        Masked data the model was trained on.
    model : MultiModalSCVI
        Trained model.
    spatial_key : str
        obsm key for the spatial modality (the *masked* version, e.g. ``'spatial_asinh_masked'``).
    mask_col : str
        Boolean obs column indicating masked cells.

    Returns
    -------
    dict with keys ``'l2_masked'``, ``'l2_unmasked'``, ``'ratio'``.
    """
    imputed = model.get_normalized_expression(
        adata=adata_masked,
        return_mean_expression=True,
        return_l2_error=True,
        return_px_distrs=False,
        return_numpy=True,
    )

    mask = adata_masked.obs[mask_col].values.astype(bool)

    # Ground truth from the *original* (unmasked) spatial key
    original_spatial_key = spatial_key.replace("_masked", "")
    if original_spatial_key in adata_original.obsm:
        true_spatial = adata_original.obsm[original_spatial_key]
    else:
        true_spatial = adata_original.obsm[spatial_key]
    true_spatial = true_spatial.values if hasattr(true_spatial, "values") else true_spatial

    imputed_spatial = imputed["exprs"][spatial_key]

    l2_masked = float(np.mean((true_spatial[mask] - imputed_spatial[mask]) ** 2))
    l2_unmasked = float(np.mean((true_spatial[~mask] - imputed_spatial[~mask]) ** 2))
    ratio = l2_masked / (l2_unmasked + 1e-12)

    print(f"Mean L2 reconstruction error (masked):     {l2_masked:.6f}")
    print(f"Mean L2 reconstruction error (non-masked): {l2_unmasked:.6f}")
    print(f"Ratio (masked / non-masked):               {ratio:.3f}")

    return dict(l2_masked=l2_masked, l2_unmasked=l2_unmasked, ratio=ratio)


# ---------------------------------------------------------------------------
# Synthetic data (from test.ipynb)
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_cells=1000, n_features_per_mod=50, seed=0):
    """Create a two-modality AnnData with known cluster structure.

    Modality 1 (``layers['normal']``): two Gaussian clusters.
    Modality 2 (``obsm['beta_std']``): standardised beta-distributed features.

    The returned AnnData has ``obs['sample']`` (4 groups from the cross of
    the two modality clusters) and ``obs['normal_group']``,
    ``obs['beta_group']``.
    """
    rng = np.random.default_rng(seed=seed)

    constant_beta_frac = 0.25
    n_constant_beta = int(n_cells * constant_beta_frac)
    n_real_beta = n_cells - n_constant_beta
    n_normal_1 = n_cells // 2
    n_normal_2 = n_cells - n_normal_1

    beta_1 = rng.beta(a=50, b=50, size=(n_real_beta, n_features_per_mod))
    beta_2 = np.ones((n_constant_beta, n_features_per_mod)) * 0.5
    normal_1 = rng.normal(loc=1, scale=0.3, size=(n_normal_1, n_features_per_mod))
    normal_2 = rng.normal(loc=-1, scale=0.6, size=(n_normal_2, n_features_per_mod))

    beta = np.concatenate((beta_1, beta_2), axis=0)
    normal = np.concatenate((normal_1, normal_2), axis=0)

    obs_beta_group = np.array(["beta_1"] * n_real_beta + ["beta_2"] * n_constant_beta)
    obs_normal_group = np.array(["normal_1"] * n_normal_1 + ["normal_2"] * n_normal_2)

    beta_perm = rng.permutation(n_cells)
    normal_perm = rng.permutation(n_cells)
    beta = beta[beta_perm]
    obs_beta_group = obs_beta_group[beta_perm]
    normal = normal[normal_perm]
    obs_normal_group = obs_normal_group[normal_perm]

    obs_names = [f"obs_{i}" for i in range(n_cells)]
    normal_features = [f"{i}_normal" for i in range(n_features_per_mod)]
    beta_features = [f"{i}_beta" for i in range(n_features_per_mod)]

    normal_df = pd.DataFrame(normal, columns=normal_features, index=obs_names)
    beta_df = pd.DataFrame(beta, columns=beta_features, index=obs_names)

    adata = anndata.AnnData(
        X=normal_df,
        obs=pd.DataFrame(
            {"normal_group": obs_normal_group, "beta_group": obs_beta_group},
            index=obs_names,
        ),
        obsm={
            "beta": beta_df,
            "beta_std": (beta_df - beta_df.mean(axis=0)) / beta_df.std(axis=0),
        },
        layers={"normal": normal_df},
    )

    sample_map = {
        ("beta_1", "normal_1"): "1",
        ("beta_1", "normal_2"): "2",
        ("beta_2", "normal_1"): "3",
        ("beta_2", "normal_2"): "4",
    }
    adata.obs["sample"] = (
        adata.obs[["beta_group", "normal_group"]].apply(tuple, axis=1).map(sample_map)
    )

    return adata
