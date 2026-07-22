#!/usr/bin/env python3
"""Train one MultiModalSCVI *final-sweep* config and cache its artifacts.

Final sweep = fusion architecture (baseline / S1-S4 from fusion_benchmark.ipynb)
crossed with the ref-notebook params (spatial x n_latent x n_layers x batch_mask).
`experts_method` is folded into the architecture, so there is no separate experts dim.
n_latent is restricted to {20, 30} so every per-modality latent is >= the scVI
baseline n_latent (20). Idempotent: skips if <out-root>/<run_id>/DONE exists.
"""
import argparse
import json
import sys
from pathlib import Path

PIXELGEN_ROOT = '/home/projects/nyosef/zvise/PixelGen/'
if PIXELGEN_ROOT not in sys.path:
    sys.path.append(PIXELGEN_ROOT)

ROOT = Path('/home/projects/nyosef/zvise/PixelGen/PixelGen')
CACHE = ROOT / 'New_Data' / 'cache'
ADATA_PATH = CACHE / 'adata_cytovi_annotated_compat.h5ad'

SPATIAL = {'top500': 'spatial_asinh5_top500var', 'ct100': 'spatial_asinh5_ct_top100',
           'triplet': 'coloc_triplet_asinh5'}
CHUNGLU = ROOT / 'New_Data' / 'results' / 'chunglu'
BATCH_MASK = {'ff': [False, False], 'ft': [False, True]}

# Architecture -> extra MultiModalVAE kwargs (from fusion_benchmark.ipynb cell `cfg`).
ARCH = {
    'baseline':     dict(experts_method='POE'),
    'S1_moe':       dict(experts_method='MOE', weight_floor=0.2, weight_entropy_reg=0.1),
    'S2_poetemp':   dict(experts_method='POE', poe_calibration='temperature'),
    'S3_mvtcae':    dict(experts_method='POE', objective='mvtcae', tc_alpha=0.5, tc_beta=1.0),
    'S4_mmvaeplus': dict(experts_method='POE', n_private=[10, 10],
                         cross_reconstruction=True, cross_recon_weight=1.0, free_bits=0.5),
}

ABUNDANCE_LAYER = 'arcsinh'
BATCH_KEY = 'cell_system'
BIO_KEY = 'cell_type_annot'


def make_run_id(a):
    return f'arch-{a.arch}__sp-{a.spatial}__nl{a.n_latent}__L{a.n_layers}__bm-{a.batch_mask}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=ARCH, required=True)
    p.add_argument('--spatial', choices=SPATIAL, required=True)
    p.add_argument('--n-latent', type=int, choices=[20, 30], required=True)
    p.add_argument('--n-layers', type=int, required=True)
    p.add_argument('--batch-mask', choices=BATCH_MASK, required=True)
    p.add_argument('--out-root', default=str(CACHE / 'final_sweep'))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-epochs', type=int, default=10000)  # lower for a dry run
    args = p.parse_args()

    run_id = make_run_id(args)
    outdir = Path(args.out_root) / run_id
    if (outdir / 'DONE').exists():
        print(f'[skip] {run_id} already done')
        return
    outdir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import pandas as pd
    import scanpy as sc
    import torch
    import scvi
    from PixelGen.multimodalvi import MultiModalSCVI
    from PixelGen.enums import D
    from PixelGen.pxl_utils import train_model
    from PixelGen.utils import (build_spatial_celltype_obsm, build_triplet_asinh5_obsm,
                                calculate_metrics, get_dense)

    scvi.settings.seed = args.seed
    torch.set_float32_matmul_precision('high')

    adata = sc.read_h5ad(ADATA_PATH)

    if args.spatial == 'ct100':
        spatial_key = build_spatial_celltype_obsm(
            adata, celltype_key=BIO_KEY, score_key='spatial_raw', value_key='spatial_asinh5',
            z_thresh=2.04, top_x=100, use_abs=True,
        )
    elif args.spatial == 'triplet':
        spatial_key = build_triplet_asinh5_obsm(adata, CHUNGLU)
    else:
        spatial_key = SPATIAL[args.spatial]
    assert spatial_key in adata.obsm, f'{spatial_key} missing from obsm'

    setup_kwargs = dict(
        layer=ABUNDANCE_LAYER, extra_modality_keys=[spatial_key], n_modalities=2,
        batch_key=BATCH_KEY, spatial_mask_key=None,
    )
    model_kwargs = dict(
        n_latent=args.n_latent, n_hidden=128, n_layers=args.n_layers, dropout_rate=0.1,
        distrs=[D.Normal, D.Normal],
        loss_weights='auto',
        joint_kl=False, unimodal_kl=True,
        batch_mask=BATCH_MASK[args.batch_mask],
        **ARCH[args.arch],
    )  # default decoder -> softplus, eps=1e-12
    base_train_kwargs = dict(
        train_size=0.8,
        check_val_every_n_epoch=1,
        early_stopping=True,
        early_stopping_patience=200,
        batch_size=2000,
        max_epochs=args.max_epochs,
        enable_checkpointing=True,
        plan_kwargs=dict(lr=1e-4, optimizer='Adam', n_epochs_kl_warmup=400),
    )

    print(f'[train] {run_id}  arch={args.arch}  spatial_key={spatial_key}  '
          f'mask={BATCH_MASK[args.batch_mask]}  extra={ARCH[args.arch]}')
    model = train_model(adata, MultiModalSCVI, setup_kwargs, model_kwargs, base_train_kwargs,
                        generate_loss_plots=False)

    model.save(str(outdir / 'model'), overwrite=True)

    z_joint = model.get_latent_representation(adata, modality='joint')
    z_abn = model.get_latent_representation(adata, modality=ABUNDANCE_LAYER)
    z_spt = model.get_latent_representation(adata, modality=spatial_key)
    np.savez(outdir / 'latents.npz', z_joint=z_joint, z_abn=z_abn, z_spt=z_spt)

    out = model.get_normalized_expression(
        adata=adata, return_mean_expression=True, return_l2_error=False,
        return_px_distrs=False, return_numpy=True,
    )
    rows = []
    rows += calculate_metrics(get_dense(adata.layers[ABUNDANCE_LAYER]),
                              get_dense(out['exprs'][ABUNDANCE_LAYER]), run_id, 'Abundance')
    rows += calculate_metrics(get_dense(adata.obsm[spatial_key]),
                              get_dense(out['exprs'][spatial_key]), run_id, 'Spatial')
    pd.DataFrame(rows).to_parquet(outdir / 'ppc_metrics.parquet')

    weights = model.get_weights()
    config = dict(
        run_id=run_id, arch=args.arch, arch_kwargs=ARCH[args.arch],
        spatial=args.spatial, spatial_key=spatial_key,
        n_latent=args.n_latent, n_layers=args.n_layers,
        batch_mask=args.batch_mask, batch_mask_vals=BATCH_MASK[args.batch_mask],
        seed=args.seed,
        weights=np.asarray(weights).tolist(),
    )
    (outdir / 'config.json').write_text(json.dumps(config, indent=2))
    (outdir / 'DONE').touch()
    print(f'[done] {run_id}  weights={config["weights"]}')


if __name__ == '__main__':
    main()
