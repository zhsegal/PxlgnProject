import pixelator
import anndata
import datetime

import os
from pathlib import Path
import itertools
from tqdm import tqdm


from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import polars as pl


import scanpy as sc
from scipy.sparse import csr_matrix


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def print_w_time(s):
    print(f'[{datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}]: {s}', flush=True)

def pd_unique_values(df, col_names, observed, **kwargs):
    return df.groupby(col_names, as_index=False, observed=observed, **kwargs)[col_names].last()
    # return df.drop_duplicates(subset=col_names, inplace=False, ignore_index=False)[col_names]

def pair2str(marker_1, marker_2):
    return f'({marker_1},{marker_2})'

def convert_edgelist_to_protein_pair_colocalization(pg_data, nbhd_radius=1, pxl_type='a', components=None, group_markers=True, verbose='tqdm', detailed_info=False):
    '''
    Converts the edgelist from pg_data.edgelist into a table that describes colocalization of pairs of proteins in small neighborhoods in the graph.
    components (optional): subset of components on which to calculate
    group_markers: If group_markers=True, markers of the same type inside a single pixel are treated as one for counting purposes.
    verbose: 'tqdm' or True for a progress bar, an integer v for a checkpoint every v components, False for nothing
    Returns pandas DataFrame with index and columns (marker_1, marker_2) for each pair.
    The values are calculated conceptually as follows for every component:
    close_pixel_pairs = set of pairs of pixels with distance <= nbhd_size from each other
    for every two markers m1, m2:
    score[(m1, m2)] = \sum_{pair \in close_pixel_pairs} 1{m1 in pixel1 and m2 in pixel2}
    '''

    if not group_markers:
        raise NotImplementedError

    assert pxl_type in ('a', 'b')

    adata = pg_data.adata

    if components is None:
        components = adata.obs_names

    # marker_pair_coloc_by_component_list = []
    marker_pair_intrsct_list = []
    marker_pair_union_list = []

    marker_pair_coloc_list = []

    if detailed_info:
        upi_connectivity_by_component_list = []
        nbhd_marker_count_by_component_list = []
        upia_to_upia_int_list = []

    marker_names = adata.var_names
    marker_diff_pair_names = sorted([pair for pair in itertools.product(marker_names, marker_names) if pair[0] != pair[1]])
    marker_pairs_to_sorted_pairs = {
        tuple(pair): tuple(sorted(pair)) for pair in itertools.product(marker_names, marker_names)
    }
    sorted_marker_pair_names = sorted(list(set(marker_pairs_to_sorted_pairs.values())))

    if verbose is not False:
        print_w_time('Starting conversion...')

    iter = tqdm(components) if verbose in (True, 'tqdm') else components
    for i_comp, component in enumerate(iter):
        # component_edgelist_orig = pg_data.edgelist_lazy.filter(pl.col('component') == component).select(['upia', 'upib', 'marker']).collect().to_pandas()
        component_edgelist_orig = pg_data.edgelist[pg_data.edgelist['component'] == component]

        upi_s = 'upia_int' if pxl_type == 'a' else 'upib_int'
        upi_s_other = 'upib_int' if pxl_type == 'a' else 'upia_int'

        upia_unique = pd.DataFrame(component_edgelist_orig['upia'].unique()).reset_index().rename(columns={0:'upia'}).set_index('upia').rename(columns={'index':'upia_int'})
        upib_unique = pd.DataFrame(component_edgelist_orig['upib'].unique()).reset_index().rename(columns={0:'upib'}).set_index('upib').rename(columns={'index': 'upib_int'})

        component_edgelist = component_edgelist_orig.join(upia_unique['upia_int'], on='upia').join(upib_unique['upib_int'], on='upib')
        upia_upib_int_unique = pd_unique_values(component_edgelist, ['upia_int', 'upib_int'], observed=True)

        upi_to_ungrouped_marker_counts = component_edgelist[[upi_s, 'marker']].groupby([upi_s, 'marker'], observed=True).size().reset_index().rename(columns={0: 'Count'})
        if group_markers:
            upi_to_grouped_marker_counts = upi_to_ungrouped_marker_counts.copy()
            upi_to_grouped_marker_counts['Count'] = 1

        upi_connectivity = upia_unique if upi_s == 'upia_int' else upib_unique
        upi_connectivity = upi_connectivity.reset_index()[[upi_s]].rename(columns={upi_s: f'{upi_s}_0'})

        for i in range(nbhd_radius):
            upi_connectivity = upi_connectivity.join(upia_upib_int_unique.set_index(upi_s)[upi_s_other], on=f'{upi_s}_{i}')
            upi_connectivity = upi_connectivity.join(upia_upib_int_unique.set_index(upi_s_other)[upi_s], on=upi_s_other).rename(columns={upi_s:f'{upi_s}_{i+1}'})

            col_names = [f'{upi_s}_{j}' for j in range(i+2)]
            upi_connectivity = pd_unique_values(upi_connectivity, col_names=col_names, observed=True)

        pxl_0 = f'{upi_s}_0'
        pxl_n = f'{upi_s}_{nbhd_radius}'
        center_pxl = f'nbhd_center_{upi_s}'

        center_marker = 'marker_1'
        nbhd_marker = 'marker_2'

        center_count = 'Count_1'
        nbhd_count = 'Count_2'

        if detailed_info:
            upi_connectivity['component'] = component
            upi_connectivity_by_component_list.append(upi_connectivity)

        # print(upi_connectivity)
        upi_connectivity = upi_connectivity[[pxl_0, pxl_n]]

        # Count each pair of pixels only once
        # upi_connectivity = upi_connectivity[upi_connectivity[center_pxl] <= upi_connectivity[nbhd_pxl]]

        # upi_to_marker_counts = upi_to_grouped_marker_counts if group_markers else upi_to_ungrouped_marker_counts
        upi_to_marker_counts = upi_to_ungrouped_marker_counts

        nbhds_markers = upi_connectivity.join(upi_to_marker_counts.set_index(upi_s)[['marker', 'Count']], on=pxl_n)[[pxl_0, 'marker', 'Count']].rename(columns={pxl_0: center_pxl})
        # nbhds_markers = nbhds_markers.groupby([center_pxl, 'marker'], observed=True)['Count'].sum().reset_index().rename(columns={0: 'Count'})
        # if group_markers:
        #     nbhds_markers['Count'] = 1
        # nbhds_markers['Count'] = nbhds_markers['Count'].astype(int)

        upi_markers_pivoted = pd.pivot_table(upi_to_marker_counts, values='Count', index=upi_s, columns='marker', observed=True, fill_value=0).astype(int)
        nbhds_markers_pivoted = pd.pivot_table(nbhds_markers, values='Count', index=center_pxl, columns='marker', observed=True, fill_value=0, aggfunc='sum').astype(int)

        for marker in marker_names:
            for tbl in (upi_markers_pivoted, nbhds_markers_pivoted):
                if marker not in tbl.columns:
                    tbl[marker] = 0

        marker_pair_coloc = pd.Series(
            [
                (upi_markers_pivoted[pair[0]]*(nbhds_markers_pivoted[pair[1]].astype(bool).astype(int))).sum() for pair in marker_diff_pair_names
            ], index=marker_diff_pair_names
        )
        
        # marker_pair_coloc = pd.Series(
        #     [
        #         (nbhds_markers_pivoted[pair[0]]*(nbhds_markers_pivoted[pair[1]].astype(bool).astype(int))).sum() for pair in marker_diff_pair_names
        #     ], index=marker_diff_pair_names,
        # )
        marker_pair_coloc_list.append(marker_pair_coloc)

        if detailed_info:
            nbhds_markers = nbhds_markers_pivoted.melt(var_name='marker', value_name='Count', ignore_index=False).reset_index().rename(columns={'index': 'upia_int'})
            nbhds_markers['component'] = component
            nbhd_marker_count_by_component_list.append(nbhds_markers)
            upia_unique['component'] = component
            upia_to_upia_int_list.append(upia_unique)

        # if group_markers:
        #     nbhds_markers_pivoted = nbhds_markers_pivoted.astype(bool).astype(int)



        # print(nbhds_markers_pivoted)

        # marker_pairs_intersection = pd.Series(
        #     [(nbhds_markers_pivoted[sorted_marker_pair[0]] & nbhds_markers_pivoted[sorted_marker_pair[1]]).sum() for sorted_marker_pair in sorted_marker_pair_names],
        #     index=sorted_marker_pair_names
        # )
        # marker_pairs_union = pd.Series(
        #     [(nbhds_markers_pivoted[sorted_marker_pair[0]] | nbhds_markers_pivoted[sorted_marker_pair[1]]).sum() for sorted_marker_pair in sorted_marker_pair_names],
        #     index=sorted_marker_pair_names,
        # )

        # marker_pair_intrsct_list.append(marker_pairs_intersection)
        # marker_pair_union_list.append(marker_pairs_union)

        # print(marker_pairs_intersection)
        # print(marker_pairs_union)

        # marker_pairs_pixel_pairs = upi_connectivity.join(upi_to_marker_counts.set_index(upi_s)[['marker', 'Count']], on=center_pxl).rename(columns={'marker': center_marker, 'Count': center_count}).join(
        #                     upi_to_marker_counts.set_index(upi_s)[['marker', 'Count']], on=nbhd_pxl
        #                 ).rename(columns={'marker': nbhd_marker, 'Count': nbhd_count})
        
        # # Remove same-marker pairs
        # marker_pairs_pixel_pairs = marker_pairs_pixel_pairs[marker_pairs_pixel_pairs[center_marker] != marker_pairs_pixel_pairs[nbhd_marker]]

        # marker_pairs_pixel_pairs['marker_pair'] = [marker_pairs_to_sorted_pairs[(marker_1, marker_2)] for marker_1, marker_2 in zip(marker_pairs_pixel_pairs[center_marker], marker_pairs_pixel_pairs[nbhd_marker])]

        # marker_pairs_pixel_pairs['Product'] = marker_pairs_pixel_pairs[center_count]*marker_pairs_pixel_pairs[nbhd_count]

        # marker_pair_counts = marker_pairs_pixel_pairs.groupby('marker_pair')['Product'].sum().rename('Count')

        # marker_pair_coloc_by_component_list.append(marker_pair_counts.astype(int))

        if type(verbose) == int:
            if i_comp % verbose == 0:
                print_w_time(f'Component {i_comp} finished')

    if verbose is not False:
        print_w_time('Finished conversion!')

    # marker_pair_coloc_by_component = pd.DataFrame(data=marker_pair_coloc_by_component_list, index=components).fillna(0).astype(int)
    # col_names = sorted(list(marker_pair_coloc_by_component.columns))
    # marker_pair_coloc_by_component = marker_pair_coloc_by_component[col_names]
    
    # marker_pair_intrsct_df = pd.DataFrame(marker_pair_intrsct_list, index=components, dtype=int)
    # marker_pair_union_df = pd.DataFrame(marker_pair_union_list, index=components, dtype=int)

    # marker_pair_names_tuples = list(marker_pair_intrsct_df.columns.values)

    marker_pair_coloc_df = pd.DataFrame(marker_pair_coloc_list, index=components, dtype=int)

    layers = {'marker_pair_coloc': marker_pair_coloc_df,}

    for df in layers.values():
        # Rename columns to strings instead of tuples
        df.rename(
            columns={(m1, m2): pair2str(m1,m2) for m1,m2 in df.columns.values},
            inplace=True
        )
    
    ret = dotdict({
        'layers': layers,
        'info': dict(nbhd_radius=nbhd_radius, pxl_type=pxl_type, marker_diff_pair_tuples=marker_diff_pair_names)
    })

    # return_dict = {
    #     'marker_pair_intersection': marker_pair_intrsct_df,
    #     'marker_pair_union': marker_pair_union_df,
    #     'marker_pair_names_tuples': marker_pair_names_tuples,
    # }

    if detailed_info:
        ret.info['upi_connectivity'] = pd.concat(upi_connectivity_by_component_list, axis=0)
        ret.info['nbhds_marker_counts'] = pd.concat(nbhd_marker_count_by_component_list, axis=0)
        ret.info['upia_to_upia_int'] = pd.concat(upia_to_upia_int_list, axis=0)
    
    return ret


def apply_func_to_neighbors(adata, neighbors_key, func):
    '''
    Receives adata with a obsp connectivities sparse matrix.
    Applies func between each pair of neighbors, and returns a matrix M with the same structure but M_ij = func(obs_i, obs_j) for neighboring observations.
    Func receives two observation indices and returns a scalar.
    '''
    neighbors_row, neighbors_col = adata.obsp[f'{neighbors_key}_distances'].nonzero()
    func_data = []
    for (obs1, obs2) in tqdm(zip(neighbors_row, neighbors_col)):
        func_data.append(func(obs1, obs2))
    return csr_matrix(func_data, (neighbors_row, neighbors_col))
    
