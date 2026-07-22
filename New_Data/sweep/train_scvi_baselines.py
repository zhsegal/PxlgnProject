#!/usr/bin/env python3
"""Train the two single-modality scVI specialist baselines used as benchmark anchors.

These are the per-modality "best/worst envelope" for the final sweep:
  - z_scvi_abundance : vanilla scVI on the arcsinh abundance layer (159 feat)
  - z_scvi_spatial   : vanilla scVI on spatial_asinh5_top500var (500 feat)
n_latent=20 -> this is the reference latent size the sweep's per-modality latents must match.
Saves cache/final_sweep/scvi_baselines.npz (keys: z_scvi_abundance, z_scvi_spatial).
Refactored from spatial_pca_benchmark.ipynb cell `4e04145b`.
"""
import sys
from pathlib import Path

PIXELGEN_ROOT = '/home/projects/nyosef/zvise/PixelGen/'
if PIXELGEN_ROOT not in sys.path:
    sys.path.append(PIXELGEN_ROOT)

ROOT = Path('/home/projects/nyosef/zvise/PixelGen/PixelGen')
CACHE = ROOT / 'New_Data' / 'cache'
ADATA_PATH = CACHE / 'adata_cytovi_annotated_compat.h5ad'

ABUNDANCE_LAYER = 'arcsinh'
SPATIAL_KEY = 'spatial_asinh5_top500var'
BATCH_KEY = None   # no batch correction — matches spatial_pca_benchmark.ipynb scVI baselines

SCVI_KWARGS = dict(n_latent=20, n_hidden=128, n_layers=2, dropout_rate=0.1,
                   gene_likelihood='normal')   # continuous features
TRAIN_KWARGS = dict(max_epochs=200, batch_size=512, early_stopping=True)


def main():
    import numpy as np
    import anndata as ad
    import scanpy as sc
    import scvi
    from PixelGen.utils import get_dense

    scvi.settings.seed = 0
    adata = sc.read_h5ad(ADATA_PATH)

    def train_scvi_baseline(X, var_names, name):
        """Single-modality scvi.model.SCVI baseline; returns the latent representation.

        scVI assumes non-negative, count-like input. Spatial colocalization features are
        signed, so shift each feature to >= 0 before fitting (no-op for arcsinh abundance).
        Only the encoder latent z is used downstream, so the shift is harmless.
        """
        Xd = np.asarray(get_dense(X), dtype='float32')
        Xd = np.clip(Xd - Xd.min(axis=0, keepdims=True), 0.0, None)
        a = ad.AnnData(X=Xd, obs=adata.obs.copy())
        a.var_names = [str(v) for v in var_names]
        scvi.model.SCVI.setup_anndata(a, batch_key=BATCH_KEY)
        m = scvi.model.SCVI(a, **SCVI_KWARGS)
        print(f'>>> training vanilla scVI {name}: n_vars={a.n_vars}')
        m.train(**TRAIN_KWARGS)
        return m.get_latent_representation()

    z_scvi_abundance = train_scvi_baseline(adata.layers[ABUNDANCE_LAYER], adata.var_names, 'abundance')
    z_scvi_spatial = train_scvi_baseline(
        adata.obsm[SPATIAL_KEY], adata.obsm[SPATIAL_KEY].columns, 'spatial')

    out = CACHE / 'final_sweep'
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'scvi_baselines.npz',
             z_scvi_abundance=z_scvi_abundance, z_scvi_spatial=z_scvi_spatial)
    print(f'[done] saved {out / "scvi_baselines.npz"}  '
          f'abundance={z_scvi_abundance.shape} spatial={z_scvi_spatial.shape}')


if __name__ == '__main__':
    main()
