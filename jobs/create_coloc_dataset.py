import pandas as pd
from PixelGen.pxl_utils import compute_hotspot_pol_and_coloc_single_component
import pixelator
from pixelator.statistics import clr_transformation
from concurrent.futures import ProcessPoolExecutor
import logging
import argparse
from pathlib import Path
import os
import contextlib
import traceback

ISOTYPE_CONTROLS=['mIgG1', 'mIgG2a', 'mIgG2b']
MARKER_COUNT_THRESHOLD = 10
KNN_NEIGHBORS = 30

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to compute Hotspot polarization & colocalization from a Pixelgen MPX dataset. Run from parent directory of PixelGen with python -m PixelGen.jobs.create_coloc_dataset <args>"
    )
    
    parser.add_argument(
        '--pxl-file',
        type=Path,
        help="Path to pxl file inside a dataset directory",
        required=True,
    )

    parser.add_argument(
        '--output-name',
        type=str,
        help='New dataset will be saved at <dataset_dir>/<output_name>.h5ad where <dataset_dir> is inferred from pxl_file',
        required=True,
    )
    
    parser.add_argument(
        '--n-threads',
        type=int,
        default=16,
    )
    
    parser.add_argument(
        '--test-mode',
        type=bool,
        default=False,
        help="For debugging"
    )
    
    return parser.parse_args()

# DATASET_DIR = Path('PixelGen/datasets/technote-cart-fmc63-v2.0')
# DATASET = DATASET_DIR / 'carT_combined.pxl'
# NEW_DATASET = DATASET_DIR / 'carT_combined_with_hs.h5ad'
# N_THREADS = 16

# TEST_MODE = False



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    args = parse_args()
    DATASET = args.pxl_file
    DATASET_DIR = args.pxl_file.parent
    TEST_MODE = args.test_mode
    NEW_DATASET = DATASET_DIR / f'{args.output_name}.h5ad'
    N_THREADS = args.n_threads
    TEST_EDGELIST_DIR = DATASET_DIR / 'toy_edgelist.csv'
    TEST_NEW_DATASET = DATASET_DIR / f'TEST_{args.output_name}.h5ad'


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


    non_isotype_vars = [var for var in adata.var_names if var not in ISOTYPE_CONTROLS]

    if TEST_MODE:
        logger.info('TEST_MODE = True')
        if not os.path.exists(TEST_EDGELIST_DIR):
            logger.info('Toy edgelist does not exist. Creating...')

            logger.info('Loading full edgelist into memory (this could take a few minutes)...')
            edgelist = pg_data.edgelist

            components = edgelist['component'].unique()[:10]

            toy_edgelist = edgelist[edgelist['component'].isin(components)]
            toy_edgelist.to_csv(TEST_EDGELIST_DIR)
        else:
            toy_edgelist = pd.read_csv(TEST_EDGELIST_DIR)
        
        edgelist = toy_edgelist
        components = edgelist['component'].unique()
    
    else:
        logger.info('Loading full edgelist into memory (this could take a few minutes)...')
        edgelist = pg_data.edgelist

        # Do some op to load into memory for sure
        _ = edgelist[edgelist['component'] == edgelist['component'].iloc[0]]

    kwargs = dict(edgelist=edgelist, adata=adata, vars=non_isotype_vars, marker_count_threshold=MARKER_COUNT_THRESHOLD, knn_neighbors=KNN_NEIGHBORS)

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
            logger.info(f'ERROR IN COMPONENT {component}:\n{traceback.format_exc()}')
            return pd.DataFrame(), pd.DataFrame()


    results = []
    logger.info('Beginning conversion...')

    with ProcessPoolExecutor(max_workers=N_THREADS) as executor:
        results = list(executor.map(_process_component, components, range(len(components))))
        logger.info('Finished conversion!')

        all_pol, all_coloc = zip(*results)
            
        pol = pd.concat(all_pol, axis=0, ignore_index=True)
        coloc = pd.concat(all_coloc, axis=0, ignore_index=True)

        adata.uns['pol_hs_longform'] = pol
        adata.uns['coloc_hs_longform'] = coloc

        adata.obsm['pol_hs_c'] = pol.pivot_table(
            index='component', columns='marker', values='pol_hs_c', fill_value=0, observed=True).reindex(adata.obs.index)
        adata.obsm['pol_hs_z'] = pol.pivot_table(
            index='component', columns='marker', values='pol_hs_z', fill_value=0, observed=True).reindex(adata.obs.index)
        
        adata.obsm['coloc_hs_c'] = coloc.pivot_table(
            index='component', columns='pair_name', values='coloc_hs_c', fill_value=0, observed=True).reindex(adata.obs.index)
        adata.obsm['coloc_hs_z'] = coloc.pivot_table(
            index='component', columns='pair_name', values='coloc_hs_z', fill_value=0, observed=True).reindex(adata.obs.index)

        save_dir = NEW_DATASET if not TEST_MODE else TEST_NEW_DATASET
        adata.write_h5ad(save_dir)