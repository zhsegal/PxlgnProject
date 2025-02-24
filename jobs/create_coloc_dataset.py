import pandas as pd
from PixelGen.pxl_utils import convert_edgelist_to_protein_pair_colocalization, print_w_time
import pixelator
from pixelator.statistics import clr_transformation
from concurrent.futures import ProcessPoolExecutor
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Run from parent directory of PixelGen with python -m PixelGen.jobs.create_coloc_dataset

DATASET = 'PixelGen/datasets/combined_resting_PHA_data_PROCESSED.pxl'
NEW_DATASET = 'PixelGen/datasets/combined_data_coloc_autocorr_with_norm.h5ad'
N_THREADS = 16

pg_data = pixelator.read(DATASET)
adata = pg_data.adata

# Test
# components = adata.obs_names[:100]
# verbose = 1
# adata = adata[adata.obs.index.isin(components)]

# All components
components = adata.obs_names
verbose = 100

# ungrouped_marker_coloc = pxl_utils.convert_edgelist_to_protein_pair_colocalization(pg_data=pg_data, nbhd_radius=2, group_markers=False, verbose=verbose, components=components)

# adata.obsm['grouped_marker_coloc'] = grouped_marker_coloc
# adata.obsm['ungrouped_marker_coloc'] = ungrouped_marker_coloc

# adata.uns['coloc_datasets'] = {
#     'grouped_marker_coloc': dict(nbhd_size=2, group_markers=True),
#     'ungrouped_marker_coloc': dict(nbhd_size=2, group_markers=False),
# }

logger.info('Loading edgelist into memory...')

pg_edgelist = pg_data.edgelist
# Do some op to load into memory
_ = pg_edgelist[pg_edgelist['component'] == pg_edgelist['component'].iloc[0]]


def process_component(component, idx):
    result = convert_edgelist_to_protein_pair_colocalization(pg_edgelist=pg_edgelist, adata=adata, nbhd_radius=1, pxl_type='b',
                                                             verbose=False, components=[component], 
                                                            count_layer='counts', detailed_info=False,
                                                            score_types=('autocorr', 'coloc'))
    if idx % verbose == 0:
        logger.info(f'Finished {idx}')
    
    return result


results = []
logger.info('Beginning conversion...')

with ProcessPoolExecutor(max_workers=N_THREADS) as executor:
    results = list(executor.map(process_component, components, range(len(components))))
    logger.info('Finished conversion!')

    for score_type, score_dict in results[0]['results'].items():
        for layer_name, layer in score_dict.items():
            adata.obsm[f'{score_type}_{layer_name}'] = pd.concat(
                [res['results'][score_type][layer_name] for res in results], axis=0
            )
    
    # adata.obsm['counts_and_coloc'] = pd.concat((adata.to_df(layer='counts'), adata.obsm['coloc']), axis=1)
    # adata.obsm['counts_and_z_coloc'] = pd.concat((adata.to_df(layer='counts'), adata.obsm['z_coloc']), axis=1)

    adata.uns['coloc_info'] = results[0].info

    logger.info('Writing...')
    adata.write_h5ad(NEW_DATASET)

    logger.info('Finished, exiting!')



# marker_pair_coloc = pxl_utils.

# adata.obsm['coloc_raw'] = marker_pair_coloc.layers['coloc']
# adata.obsm['expected_coloc'] = marker_pair_coloc.layers['expected_coloc']
# adata.obsm['variance_coloc'] = marker_pair_coloc.layers['variance_coloc']
# adata.obsm['z_coloc'] = marker_pair_coloc.layers['z_coloc']

# adata.obsm['counts_and_coloc_raw'] = pd.concat((adata.to_df(layer='counts'), adata.obsm['coloc_raw']), axis=1)
# adata.obsm['counts_and_z_coloc'] = pd.concat((adata.to_df(layer='counts'), adata.obsm['z_coloc']), axis=1)
# # adata.obsm['clr_by_ab_and_z_coloc'] = pd.concat((clr_transformation(adata.to_df(layer='counts'), axis=0), adata.obsm['z_coloc']), axis=1)
# # adata.obsm['clr_by_cell_and_z_coloc'] = pd.concat((clr_transformation(adata.to_df(layer='counts'), axis=1), adata.obsm['z_coloc']), axis=1)

# adata.uns['coloc_info'] = marker_pair_coloc.info

# # adata.obsm['marker_pair_intersection'] = grouped_marker_coloc.marker_pair_intersection
# # adata.obsm['marker_pair_union'] = grouped_marker_coloc.marker_pair_union
# # adata.uns['marker_pair_names_tuples'] = grouped_marker_coloc.marker_pair_names_tuples
# adata.write_h5ad(NEW_DATASET)


# Not saving adata with the changes, probably pixelator bug
# pg_data.save(NEW_DATASET, force_overwrite=True)
