# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

⚠️ Login node is fragile. Heavy jobs (training, full-notebook exec, large fits) AND big/expensive Bash commands (recursive find/rg over the whole tree, huge reads) can crash the login kernel and lock the user out for a long time. Keep commands cheap+scoped; edit code and hand heavy runs to the user's GPU kernel.

Progress: log significant changes/additions to Notion per the protocol in root `.claude/CLAUDE.md`.

## Project Overview

Single-cell variational inference framework for PixelGen Technologies proteomics data, following the scVI paradigm. The project implements multi-modal VAEs that integrate protein abundance data with spatial colocalization/polarization features computed from MPX (Molecular Pixelation) graph data.

## Environment Setup

```bash
# Create conda environment from spec
conda env create -f pixelgen-scvi.yml
conda activate pixelgen-scvi
```

Key dependencies: PyTorch (CUDA 11.8), scvi-tools 1.2.2, scanpy, pixelator, hotspot

## Architecture

### Core Model (`multimodalvi.py` + `multimodalvae.py`)

- **MultiModalSCVI**: High-level model class (scVI paradigm) that handles data registration, training, and inference
- **MultiModalVAE**: PyTorch module implementing encoder-decoder architecture with multi-modality support

**Modality Aggregation Methods** (see `enums.AggMethod`):
- `SHARED_ENCODER`: Single encoder for concatenated inputs
- `AOE_FIXED_WEIGHTS`: Product-of-Experts with fixed weights
- `AOE_GLOBAL_WEIGHTS`: Learned global modality weights
- `AOE_PER_CELL_WEIGHTS`: Per-cell learned weights

**Decoder Distributions** (see `enums.D`):
- `Normal`: For continuous unbounded data
- `Beta`: For bounded [0,1] data (supports mixture with constant values)

### Data Flow

1. Load MPX data via `pixelator.read()`
2. Compute spatial features with `pxl_utils.compute_hotspot_pol_and_coloc()` (uses Hotspot on A-pixel graph)
3. Transform spatial data with `utils.build_spatial_obsms()` (raw/tanh/asinh/quantile)
4. Register AnnData with `MultiModalSCVI.setup_anndata()` - main modality via `layer`, extra modalities via `extra_modality_keys` pointing to `obsm` DataFrames
5. Train model, extract latents with `get_latent_representation(modality='joint'|'X'|obsm_key)`

### Utility Modules

- `pxl_utils.py`: PixelGen data loading, pixel graph neighborhood computation, Hotspot-based polarization/colocalization, 3D cell visualization
- `utils.py`: Data transformations (`build_spatial_obsms`), posterior predictive checks (`plot_composite_ppc`), latent visualization (`plot_model_latents`)
- `metrics.py`: `MultiModalVIMetrics` class for comprehensive model evaluation (scib-metrics, reconstruction error, latent quality)
- `scvi_utils.py`: PCA, UMAP pipelines, loss plotting, distribution utilities

## Common Patterns

### Setting up multi-modal model
```python
MultiModalSCVI.setup_anndata(
    adata,
    layer='clr',                           # Main modality layer
    batch_key='sample',
    n_modalities=2,
    extra_modality_keys=['spatial_tanh4'], # obsm keys for additional modalities
)
model = MultiModalSCVI(adata, n_latent=20, n_hidden=128, agg_method=AggMethod.AOE_GLOBAL_WEIGHTS)
model.train(max_epochs=200)
```

### Extracting representations
```python
z_joint = model.get_latent_representation(modality='joint')
z_counts = model.get_latent_representation(modality='clr')  # Main modality
z_spatial = model.get_latent_representation(modality='spatial_tanh4')
weights = model.get_weights()  # Modality weights (global or per-cell)
```

## Notebooks

Ordered by complexity:
- `test.ipynb` - Synthetic data tests
- `figure_2/` - Model analysis and latent annotation
- `run_hotspot.ipynb`, `run_hotspot_pbmsc.ipynb` - Spatial feature computation
- `hotspot_anaylisis.ipynb` - Hotspot results analysis
