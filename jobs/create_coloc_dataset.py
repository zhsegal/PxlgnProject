import pandas as pd
from .. import pxl_utils
import pixelator

# Run from parent directory of PixelGen with python -m PixelGen.jobs.create_coloc_dataset

DATASET = 'PixelGen/datasets/combined_resting_PHA_data_PROCESSED.pxl'
NEW_DATASET = 'PixelGen/datasets/combined_resting_PHA_data_PROCESSED_WITH_ASYMM_COLOC_NO_OVERLAP.h5ad'

pg_data = pixelator.read(DATASET)

# Test
# components = pg_data.adata.obs_names[:10]
# verbose = 1
# pg_data.adata = pg_data.adata[pg_data.adata.obs.index.isin(components)]

# All components
components = None
verbose = 100

# ungrouped_marker_coloc = pxl_utils.convert_edgelist_to_protein_pair_colocalization(pg_data=pg_data, nbhd_radius=2, group_markers=False, verbose=verbose, components=components)

# pg_data.adata.obsm['grouped_marker_coloc'] = grouped_marker_coloc
# pg_data.adata.obsm['ungrouped_marker_coloc'] = ungrouped_marker_coloc

# pg_data.adata.uns['coloc_datasets'] = {
#     'grouped_marker_coloc': dict(nbhd_size=2, group_markers=True),
#     'ungrouped_marker_coloc': dict(nbhd_size=2, group_markers=False),
# }

marker_pair_coloc = pxl_utils.convert_edgelist_to_protein_pair_colocalization(pg_data=pg_data, nbhd_radius=1, group_markers=True, verbose=verbose, components=components, detailed_info=False)
pg_data.adata.obsm['marker_pair_coloc'] = marker_pair_coloc.layers['marker_pair_coloc']
pg_data.adata.obsm['marker_counts_and_pair_coloc'] = pd.concat((pg_data.adata.to_df(layer='counts'), pg_data.adata.obsm['marker_pair_coloc']), axis=1)
pg_data.adata.uns['coloc_info'] = marker_pair_coloc.info

# pg_data.adata.obsm['marker_pair_intersection'] = grouped_marker_coloc.marker_pair_intersection
# pg_data.adata.obsm['marker_pair_union'] = grouped_marker_coloc.marker_pair_union
# pg_data.adata.uns['marker_pair_names_tuples'] = grouped_marker_coloc.marker_pair_names_tuples
pg_data.adata.write_h5ad(NEW_DATASET)


# Not saving adata with the changes, probably pixelator bug
# pg_data.save(NEW_DATASET, force_overwrite=True)
