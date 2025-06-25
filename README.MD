Single-cell variational inference of PixelGen Technologies proteomics data. Follows scvi paradigm.

Contents:

Notebooks (in order of increasing complexity):
- test.ipynb - Model testing on artifical data
- CD20_Rituximab.ipynb - Model application on a relatively simple dataset of CD20 polarization
- CD3_Capping.ipynb - Model application on a dataset of T cells undergone CD3 clustering
- uropod.ipynb - Model application on a dataset of uropod formation
- carT.ipynb - Model application on a dataset of Raji and car-T cells

The last two use polarization & colocalization calculated with Hotspot.

Code:
- multimodalvi.py - Probabilistic Model (as in scvi format)
- multimodalvae.py - Inner Neural Network Model (as in scvi format)
- metrics.py - metric utilities
- pxl_utils.py - utils for handling pixelgen data
- scvi_utils.py - general utils for scvi, scanpy etc.
- other source files
