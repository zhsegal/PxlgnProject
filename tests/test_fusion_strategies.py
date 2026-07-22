"""Tests for the four anti-modality-collapse fusion strategies in MultiModalSCVI.

Run on a compute/GPU node (imports torch/scvi/lightning, trains tiny models):

    pytest tests/test_fusion_strategies.py -q

All synthetic, CPU-friendly (200 cells, max_epochs=2). The regression test is the
backward-compatibility guarantee: with every new flag at its default, the loss equals the
original summed-ELBO objective recomputed from first principles.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from multimodalvi import MultiModalSCVI  # noqa: E402
from enums import D, AggMethod  # noqa: E402


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def make_adata(n=200, d_ab=20, d_sp=30, n_batch=2, seed=0, planted=False):
    """Tiny 2-modality AnnData. If ``planted``, the spatial modality carries a 2-cluster
    signal that the abundance modality does NOT (so spatial is the only way to recover it)."""
    rng = np.random.default_rng(seed)
    ab = rng.normal(size=(n, d_ab)).astype("float32")
    sp = rng.normal(size=(n, d_sp)).astype("float32")
    obs = pd.DataFrame(
        {"batch": rng.integers(0, n_batch, size=n).astype(str)},
        index=[f"c{i}" for i in range(n)],
    )
    if planted:
        label = rng.integers(0, 2, size=n)
        sp[:, :5] += (label[:, None] * 6.0)  # strong cluster shift only in spatial
        obs["planted"] = label.astype(str)
    adata = AnnData(X=ab, obs=obs)
    adata.layers["arcsinh"] = ab
    adata.obsm["spatial"] = pd.DataFrame(sp, index=obs.index, columns=[f"s{j}" for j in range(d_sp)])
    return adata


def build_model(adata, n_latent=8, **model_kwargs):
    MultiModalSCVI.setup_anndata(
        adata, layer="arcsinh", batch_key="batch", n_modalities=2, extra_modality_keys=["spatial"]
    )
    return MultiModalSCVI(
        adata, n_latent=n_latent, n_hidden=32, n_layers=1,
        distrs=[D.Normal, D.Normal], **model_kwargs,
    )


def _forward(model, adata):
    """Run inference + generative on the full adata in one batch; return (tensors, inf, gen)."""
    module = model.module
    dl = model._make_data_loader(adata=model._validate_anndata(adata), batch_size=adata.n_obs)
    tensors = next(iter(dl))
    inf = module.inference(**module._get_inference_input(tensors))
    gen = module.generative(**module._get_generative_input(tensors, inf))
    return tensors, inf, gen


# --------------------------------------------------------------------------------------
# 1. regression / backward-compat — defaults reproduce the original summed-ELBO objective
# --------------------------------------------------------------------------------------
def test_defaults_reproduce_elbo_objective():
    torch.manual_seed(0)
    adata = make_adata(seed=1)
    model = build_model(adata, loss_weights="auto")  # all new flags at defaults
    module = model.module

    tensors, inf, gen = _forward(model, adata)
    out = module.loss(tensors, inf, gen, kl_weight=1.0)

    # Recompute the original objective from first principles (joint_kl=False, unimodal_kl=True,
    # free_bits=0, no spatial mask, rec_weight=1, no cross-recon, no entropy penalty).
    from torch.distributions import Normal, kl_divergence as kl

    kl_div = 0
    for d in inf["modalities"]:
        post = Normal(d["qzm"], d["qzv"].sqrt())
        prior = Normal(torch.zeros_like(d["qzm"]), torch.ones_like(d["qzm"]))
        kl_div = kl_div + kl(post, prior).sum(-1)

    log_lik = 0
    for inp, distr, w in zip(inf["inputs"], gen["px_distrs"], module._loss_weights):
        log_lik = log_lik + w * distr.log_prob(inp).sum(-1)

    expected = torch.mean(-(log_lik - kl_div))
    assert torch.allclose(out.loss, expected, atol=1e-5), (out.loss.item(), expected.item())
    assert torch.isfinite(out.loss)
    # No extra metrics emitted on the default path.
    assert not getattr(out, "extra_metrics", None)


def test_inference_default_keys_present():
    """New return-dict keys exist but default path still exposes the original ones."""
    adata = make_adata(seed=2)
    model = build_model(adata)
    _, inf, _ = _forward(model, adata)
    for k in ("qzm", "qzv", "z", "qz", "modalities", "inputs", "weights", "modality_z"):
        assert k in inf
    assert inf["shared_zs"] is None  # only populated when cross_reconstruction=True


# --------------------------------------------------------------------------------------
# 2. smoke test per strategy
# --------------------------------------------------------------------------------------
STRATEGY_KWARGS = {
    "baseline": dict(experts_method="POE"),
    "S1_floor": dict(experts_method="MOE", weight_floor=0.2, weight_entropy_reg=0.1),
    "S1_moe_plain": dict(experts_method="MOE"),
    "S2_temperature": dict(experts_method="POE", poe_calibration="temperature"),
    "S2_balanced": dict(experts_method="POE", poe_calibration="balanced"),
    "S3_mvtcae": dict(experts_method="POE", objective="mvtcae", tc_alpha=0.5, tc_beta=1.0),
    "S4_cross": dict(experts_method="POE", n_private=[4, 4], cross_reconstruction=True, free_bits=0.5),
    "S3+S4": dict(experts_method="POE", objective="mvtcae", n_private=[4, 4], cross_reconstruction=True),
}


@pytest.mark.parametrize("name", list(STRATEGY_KWARGS))
def test_smoke_each_strategy(name):
    torch.manual_seed(0)
    adata = make_adata(seed=3)
    model = build_model(adata, n_latent=8, loss_weights="auto", **STRATEGY_KWARGS[name])
    model.train(max_epochs=2, batch_size=64, accelerator="cpu", enable_progress_bar=False)

    z = model.get_latent_representation(adata, modality="joint")
    assert z.shape == (adata.n_obs, 8)
    assert np.isfinite(z).all()

    # private-block accessors for S4 configs
    if STRATEGY_KWARGS[name].get("cross_reconstruction"):
        z_jp = model.get_latent_representation(adata, representation="joint_private")
        assert z_jp.shape == (adata.n_obs, 8 + 4 + 4)
        z_priv = model.get_latent_representation(adata, representation="private", modality="spatial")
        assert z_priv.shape == (adata.n_obs, 4)
        assert np.isfinite(z_jp).all() and np.isfinite(z_priv).all()


# --------------------------------------------------------------------------------------
# 3. Strategy-1 behavior — floor and entropy regularizer
# --------------------------------------------------------------------------------------
def test_weight_floor_respected():
    adata = make_adata(seed=4)
    model = build_model(adata, agg_method=AggMethod.AOE_GLOBAL_WEIGHTS, experts_method="MOE",
                        weight_floor=0.2)
    # Push learned_weights to an extreme so softmax alone would drive one weight ~0.
    with torch.no_grad():
        model.module.learned_weights.copy_(torch.tensor([10.0, -10.0]))
    w = model.module.get_global_weights(numpy=True)
    assert w.min() >= 0.2 - 1e-5, w
    assert np.isclose(w.sum(), 1.0, atol=1e-5)


def test_weight_entropy_pushes_uniform():
    torch.manual_seed(0)
    adata = make_adata(seed=5)
    model = build_model(adata, agg_method=AggMethod.AOE_GLOBAL_WEIGHTS, experts_method="MOE",
                        weight_entropy_reg=50.0)
    with torch.no_grad():
        model.module.learned_weights.copy_(torch.tensor([5.0, -5.0]))  # start far from uniform
    model.train(max_epochs=20, batch_size=64, accelerator="cpu", enable_progress_bar=False)
    w = model.module.get_global_weights(numpy=True)
    assert abs(w[0] - w[1]) < 0.4, w  # strong entropy reg should pull toward 0.5/0.5


def test_entropy_penalty_in_loss():
    adata = make_adata(seed=6)
    model = build_model(adata, experts_method="MOE", weight_entropy_reg=1.0)
    tensors, inf, gen = _forward(model, adata)
    out = model.module.loss(tensors, inf, gen)
    assert out.extra_metrics is not None and "weight_entropy_penalty" in out.extra_metrics
    assert torch.isfinite(out.loss)


# --------------------------------------------------------------------------------------
# 4. Strategy-3 sanity — MVTCAE differs from ELBO and uses a spatial-only signal more
# --------------------------------------------------------------------------------------
def _kmeans_nmi(z, labels, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score
    km = KMeans(n_clusters=len(np.unique(labels)), n_init=10, random_state=seed).fit(z)
    return normalized_mutual_info_score(labels, km.labels_)


def test_mvtcae_uses_planted_spatial_signal():
    adata = make_adata(seed=7, planted=True)
    labels = adata.obs["planted"].to_numpy()

    torch.manual_seed(0)
    m_elbo = build_model(adata, n_latent=8, loss_weights="auto", experts_method="POE")
    m_elbo.train(max_epochs=60, batch_size=64, accelerator="cpu", enable_progress_bar=False)
    z_elbo = m_elbo.get_latent_representation(adata, modality="joint")

    torch.manual_seed(0)
    m_tc = build_model(adata, n_latent=8, loss_weights="auto", experts_method="POE",
                       objective="mvtcae", tc_alpha=0.8, tc_beta=1.0)
    m_tc.train(max_epochs=60, batch_size=64, accelerator="cpu", enable_progress_bar=False)
    z_tc = m_tc.get_latent_representation(adata, modality="joint")

    assert not np.allclose(z_elbo, z_tc)  # the objective actually changes the latent
    # MVTCAE should recover the spatial-only cluster at least as well as plain ELBO.
    nmi_tc = _kmeans_nmi(z_tc, labels)
    nmi_elbo = _kmeans_nmi(z_elbo, labels)
    assert nmi_tc >= nmi_elbo - 0.1, (nmi_tc, nmi_elbo)
