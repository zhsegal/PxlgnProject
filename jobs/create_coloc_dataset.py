import pandas as pd
from PixelGen.pxl_utils import compute_hotspot_pol_and_coloc_single_component
import pixelator
from pixelator.statistics import clr_transformation
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path
import os
import contextlib

# Run from parent directory of PixelGen with python -m PixelGen.jobs.create_coloc_dataset

DATASET_DIR = Path('PixelGen/datasets/technote-cart-fmc63-v2.0')
DATASET = DATASET_DIR / 'carT_combined.pxl'
NEW_DATASET = DATASET_DIR / 'carT_combined_with_hs.h5ad'
N_THREADS = 16

TEST_MODE = False
TOY_EDGELIST_DIR = DATASET_DIR / 'toy_edgelist.csv'
TEST_NEW_DATASET = DATASET_DIR / 'TEST_carT_combined_with_hs.h5ad'


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    pg_data = pixelator.read(DATASET)
    adata = pg_data.adata

    adata.layers['counts'] = adata.to_df()

    # Test
    # components = adata.obs_names[:100]
    # verbose = 1
    # adata = adata[adata.obs.index.isin(components)]

    # All components
    components = adata.obs_names
    logger.info(f'Total # Components: {len(components)}')


    isotype_controls=['mIgG1', 'mIgG2a', 'mIgG2b']
    non_isotype_vars = [var for var in adata.var_names if var not in isotype_controls]

    if TEST_MODE:
        logger.info('TEST_MODE = True')
        components = components[:100]
        if not os.path.exists(TOY_EDGELIST_DIR):
            logger.info('Toy edgelist does not exist. Creating...')

            logger.info('Loading full edgelist into memory...')
            edgelist = pg_data.edgelist

            toy_edgelist = edgelist.iloc[:100]
            toy_edgelist.to_csv(TOY_EDGELIST_DIR)
        else:
            toy_edgelist = pd.read_csv(TOY_EDGELIST_DIR)
        
        edgelist = toy_edgelist
    
    else:
        logger.info('Loading full edgelist into memory...')
        edgelist = pg_data.edgelist

        # Do some op to load into memory for sure
        _ = edgelist[edgelist['component'] == edgelist['component'].iloc[0]]

    kwargs = dict(edgelist=edgelist, adata=adata, vars=non_isotype_vars, marker_count_threshold=10, knn_neighbors=30)

    def _process_component(component, idx):
        logger.info(f'Calc {idx}')
        try:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    with contextlib.redirect_stderr(devnull):
                        pol, coloc = compute_hotspot_pol_and_coloc_single_component(
                            component=component, 
                            **kwargs,
                        )
                        return pol, coloc
        except Exception as e:
            logger.info(f'ERROR IN COMPONENT {component}:\n{e}')
            return pd.DataFrame(), pd.DataFrame()


    results = []
    logger.info('Beginning conversion...')

    with ProcessPoolExecutor(max_workers=N_THREADS) as executor:
        results = list(executor.map(_process_component, components, range(len(components))))
        logger.info('Finished conversion!')

        all_pol, all_coloc = zip(*results)
            
        pol = pd.concat(all_pol, axis=0, ignore_index=True)
        coloc = pd.concat(all_coloc, axis=0, ignore_index=True)

        adata.uns['pol_hs'] = pol
        adata.uns['coloc_hs'] = coloc

        save_dir = NEW_DATASET if not TEST_MODE else TEST_NEW_DATASET
        adata.write_h5ad(save_dir)

# with ProcessPoolExecutor(max_workers=njobs) as executor:
#     futures = [
#         executor.submit(
#             _process_component, component, i, logger,
#             edgelist=edgelist,
#             adata=adata,
#             vars=vars,
#             marker_count_threshold=marker_count_threshold, 
#             knn_neighbors=knn_neighbors,
#         ) for i, component in enumerate(components)
#     ]
#     results = [future.result() for future in futures]
#     _log('Finished conversion!', logger)


# pol, coloc = compute_hotspot_pol_and_coloc(
#     edgelist=edgelist, 
#     adata=adata, 
#     components=components, 
#     vars=non_isotype_vars, 
#     marker_count_threshold=10, 
#     knn_neighbors=30, 
#     njobs=16, 
#     logger=logger,
# )

# adata.uns['pol_hs'] = pol
# adata.uns['coloc_hs'] = coloc

# adata.write_h5ad(NEW_DATASET)

# ungrouped_marker_coloc = pxl_utils.convert_edgelist_to_protein_pair_colocalization(pg_data=pg_data, nbhd_radius=2, group_markers=False, verbose=verbose, components=components)

# adata.obsm['grouped_marker_coloc'] = grouped_marker_coloc
# adata.obsm['ungrouped_marker_coloc'] = ungrouped_marker_coloc

# adata.uns['coloc_datasets'] = {
#     'grouped_marker_coloc': dict(nbhd_size=2, group_markers=True),
#     'ungrouped_marker_coloc': dict(nbhd_size=2, group_markers=False),
# }

# logger.info('Loading edgelist into memory...')

# pg_edgelist = pg_data.edgelist
# # Do some op to load into memory
# _ = pg_edgelist[pg_edgelist['component'] == pg_edgelist['component'].iloc[0]]


# def process_component(component, idx):
#     result = convert_edgelist_to_protein_pair_colocalization(pg_edgelist=pg_edgelist, adata=adata, nbhd_radius=1, pxl_type='b',
#                                                              verbose=False, components=[component], 
#                                                             count_layer='counts', detailed_info=False,
#                                                             score_types=('autocorr', 'coloc'))
#     if idx % verbose == 0:
#         logger.info(f'Finished {idx}')
    
#     return result


# results = []
# logger.info('Beginning conversion...')

# with ProcessPoolExecutor(max_workers=N_THREADS) as executor:
#     results = list(executor.map(process_component, components, range(len(components))))
#     logger.info('Finished conversion!')

#     for score_type, score_dict in results[0]['results'].items():
#         for layer_name, layer in score_dict.items():
#             adata.obsm[f'{score_type}_{layer_name}'] = pd.concat(
#                 [res['results'][score_type][layer_name] for res in results], axis=0
#             )
    
#     # adata.obsm['counts_and_coloc'] = pd.concat((adata.to_df(layer='counts'), adata.obsm['coloc']), axis=1)
#     # adata.obsm['counts_and_z_coloc'] = pd.concat((adata.to_df(layer='counts'), adata.obsm['z_coloc']), axis=1)

#     adata.uns['coloc_info'] = results[0].info

#     logger.info('Writing...')
#     adata.write_h5ad(NEW_DATASET)

#     logger.info('Finished, exiting!')



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
