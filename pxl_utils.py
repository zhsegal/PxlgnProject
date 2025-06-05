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
from pygsp.graphs import Graph


import scanpy as sc
import scipy
from scipy.sparse import csr_matrix, coo_array
from scipy.interpolate import griddata


import plotly.graph_objects as go
import plotly.io as pio
from plotly import subplots
import nbformat

from PixelGen.scvi_utils import plot_losses
from IPython.utils import io

from scipy.sparse.csgraph import dijkstra
import hotspot
from concurrent.futures import ProcessPoolExecutor
import logging
import multiprocessing


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def print_w_time(s):
    print(f'[{datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}]: {s}', flush=True)

def pd_unique_values(df, col_names, observed, **kwargs):
    # return df.groupby(col_names, as_index=False, observed=observed, **kwargs)[col_names].last()
    return df.drop_duplicates(subset=col_names, keep='first', inplace=False, ignore_index=True)[col_names]

def pair2str(marker_1, marker_2, sort=False):
    if sort:
        marker_1, marker_2 = sorted([marker_1, marker_2])
    return f'({marker_1},{marker_2})'

def str2pair(p):
    return p.strip('()').split(',')

def get_marker_pair_tuples(marker_names, only_diff=True, only_sorted_pairs=True):
    diff = (lambda m1, m2 : m1 != m2) if only_diff else (lambda m1, m2: True)
    ordered = (lambda m1, m2: m1 < m2) if only_sorted_pairs else (lambda m1, m2: True)
    return sorted([(m1, m2) for (m1, m2) in itertools.product(marker_names, marker_names) if diff(m1, m2) and ordered(m1, m2)])


def get_marker_pair_names(marker_names, only_diff=True, only_sorted_pairs=True):
    return [pair2str(*p) for p in get_marker_pair_tuples(marker_names, only_diff=only_diff, only_sorted_pairs=only_sorted_pairs)]


def get_component_nbhds(component_edgelist, upia_unique, nbhd_radius, upia_to_marker_counts, upib_to_marker_counts, vars, upia_s='upia_int', upib_s='upib_int'):
    '''
    Assumes neihborhoods are built around a pixels.
    '''
    upia_upib_int_unique = pd_unique_values(component_edgelist, [upia_s, upib_s], observed=True)
    upi_connectivity = upia_unique.reset_index()[[upia_s]].rename(columns={upia_s: f'{upia_s}_0'})
    for i in range(nbhd_radius):
        if i % 2 == 0:
            upi_connectivity = upi_connectivity.join(upia_upib_int_unique.set_index(upia_s)[upib_s], on=f'{upia_s}_{i}').rename(columns={upib_s: f'{upib_s}_{i+1}'})
        else:
            upi_connectivity = upi_connectivity.join(upia_upib_int_unique.set_index(upib_s)[upia_s], on=f'{upib_s}_{i}').rename(columns={upia_s:f'{upia_s}_{i+1}'})

        upi_connectivity = pd_unique_values(upi_connectivity, col_names=upi_connectivity.columns.values, observed=True)

    last_pixel_type = upia_s if nbhd_radius % 2 == 0 else upib_s
    last_to_marker_counts = upia_to_marker_counts if nbhd_radius % 2 == 0 else upib_to_marker_counts
    
    pxl_0 = f'{upia_s}_0'
    pxl_n = f'{last_pixel_type}_{nbhd_radius}'
    center_pxl = f'nbhd_center_upi_int'
    # surr_pxl = f'nbhd_surr_upi_int'


    upi_connectivity_short = pd_unique_values(upi_connectivity, col_names=[pxl_0, pxl_n], observed=True)

    nbhds_markers = upi_connectivity_short.join(last_to_marker_counts.set_index('upi_int')[['marker', 'Count']], on=pxl_n)[[pxl_0, 'marker', 'Count']].rename(
        columns={pxl_0: center_pxl}).groupby([center_pxl, 'marker'], as_index=False, observed=True)['Count'].sum()
    
    # nbhds_total_counts = nbhds_markers.groupby(center_pxl, observed=True)['Count'].sum()

    # upi_markers_pivoted = pd.pivot_table(base_upi_to_marker_counts, values='Count', index='upi_int', columns='marker', observed=True, fill_value=0).astype(int)

    # Table showing marker counts per neighborhood (each nbhd is defined by a center pixel; neighborhoods overlap)
    # nbhds_markers_pivoted = pd.pivot_table(nbhds_markers, values='Count', index=center_pxl, columns='marker', observed=True, fill_value=0,).astype(int)

    # for marker in vars:
    #     if marker not in nbhds_markers_pivoted.columns:
    #         nbhds_markers_pivoted[marker] = 0
    return {
        'connectivity': upi_connectivity_short,
        'nbhd_marker_counts': nbhds_markers,
    }

    


def get_pixel_nbhd_counts(pg_edgelist, adata, pixel_type='both', nbhd_radius=0, components=None, vars=None, compute_pixel_graph=False, verbose=True):
    if components is None:
        components = adata.obs_names
    if vars is None:
        vars = adata.var_names    
    # counts_df = adata.to_df(count_layer)

    assert pixel_type in ('a', 'b', 'both')

    if nbhd_radius > 0:
        assert pixel_type == 'a'           

    if verbose:
        print_w_time('Starting conversion...')
    iter = tqdm(components) if verbose else components

    pixel_counts_per_component = []
    nbhd_counts_per_component = []
    graph_per_component = []

    for i_comp, component in enumerate(iter):
        component_edgelist_orig = pg_edgelist[pg_edgelist['component'] == component]
        component_edgelist_orig = component_edgelist_orig[component_edgelist_orig['marker'].isin(vars)]

        n_upia_unique = component_edgelist_orig['upia'].nunique()
        n_upib_unique = component_edgelist_orig['upib'].nunique()

        upia_unique = pd.DataFrame({'upi': component_edgelist_orig['upia'].unique(), 'upi_type': 'a', 'upia_int': range(0, n_upia_unique)}).set_index('upi')
        upib_unique = pd.DataFrame({'upi': component_edgelist_orig['upib'].unique(), 'upi_type': 'b', 'upib_int': range(n_upia_unique, n_upia_unique + n_upib_unique)}).set_index('upi')

        upi_unique = pd.concat((upia_unique, upib_unique), axis=0)

        component_edgelist = component_edgelist_orig.join(upia_unique['upia_int'], on='upia').join(upib_unique['upib_int'], on='upib')

        if pixel_type in ('a', 'both') or nbhd_radius > 0:
            upia_to_marker_counts = component_edgelist.groupby(['upia_int', 'marker'], observed=True, as_index=False).size().rename(columns={'size': 'Count'}).rename(columns={'upia_int': 'upi_int'})
            upia_to_marker_counts['type'] = 'a'
        
        if pixel_type in ('b', 'both') or nbhd_radius > 0:
            upib_to_marker_counts = component_edgelist.groupby(['upib_int', 'marker'], observed=True, as_index=False).size().rename(columns={'size': 'Count'}).rename(columns={'upib_int': 'upi_int'})
            upib_to_marker_counts['type'] = 'b'
        
        if pixel_type == 'a':
            upi_to_marker_counts = upia_to_marker_counts
        elif pixel_type == 'b':
            upi_to_marker_counts = upib_to_marker_counts
        else:
            upi_to_marker_counts = pd.concat((upia_to_marker_counts, upib_to_marker_counts), axis=0)
        
        upi_to_marker_counts['component'] = component
        pixel_counts_per_component.append(upi_to_marker_counts)

        if nbhd_radius > 0:
            nbhd_counts = get_component_nbhds(component_edgelist, upia_unique=upia_unique, nbhd_radius=nbhd_radius, vars=vars,
                                                 upia_to_marker_counts=upia_to_marker_counts, upib_to_marker_counts=upib_to_marker_counts
                                                 )['nbhd_marker_counts']
            nbhd_counts['component'] = component
            nbhd_counts_per_component.append(nbhd_counts)
        
        if compute_pixel_graph:
            connectivity = get_component_nbhds(component_edgelist, upia_unique=upia_unique, nbhd_radius=2, vars=vars,
                                                 upia_to_marker_counts=upia_to_marker_counts, upib_to_marker_counts=upib_to_marker_counts
                                                 )['connectivity']
            edges = connectivity.rename(columns={'upia_int_0': 'a1', 'upia_int_2': 'a2'})[['a1', 'a2']]
            edges = edges[edges['a1'] != edges['a2']]
            n_a_pixels = len(upia_unique)
            data = np.ones((len(edges),))
            coords = (edges['a1'], edges['a2'])
            edges_sparse = coo_array((data, coords), shape=(n_a_pixels, n_a_pixels))
            graph = Graph(edges_sparse, )
            graph_per_component.append(graph)



    
    return {
        'pixel_counts': pd.concat(pixel_counts_per_component, axis=0, ignore_index=True),
        'nbhd_counts': pd.concat(nbhd_counts_per_component, axis=0, ignore_index=True) if nbhd_radius > 0 else None,
        'graphs': graph_per_component if compute_pixel_graph else None
    }


    
    # marker_pair_tuples = get_marker_pair_tuples(vars, only_diff=True, only_sorted_pairs=True)
    # marker_pair_strs = get_marker_pair_names(vars, only_diff=True, only_sorted_pairs=True) 


def convert_edgelist_to_protein_pair_colocalization(pg_edgelist, adata, nbhd_radius=1, pxl_type='a', components=None, verbose='tqdm', 
                                                    detailed_info=False, count_layer='counts',
                                                    score_types=['autocorr', 'coloc']):
    '''
    Converts the edgelist from pg_data.edgelist into a table that describes colocalization of pairs of proteins in small neighborhoods in the graph.
    components (optional): subset of components on which to calculate
    pg_edgelist: edgelist containing at least all of the components detailed in "components"
    verbose: 'tqdm' or True for a progress bar, an integer v for a checkpoint every v components, False for nothing
    Returns pandas DataFrame with index and columns (marker_1, marker_2) for each pair.
    count_layer: the layer from pg_data.adata from which to take raw marker counts for frequency estimation.
    The values are calculated conceptually as follows for every component:
    close_pixel_pairs = set of pairs of pixels with distance <= nbhd_size from each other
    for every two markers m1, m2:
    score[(m1, m2)] = \sum_{pair \in close_pixel_pairs} 1{m1 in pixel1 and m2 in pixel2}
    '''

    assert pxl_type in ('a', 'b')

    if components is None:
        components = adata.obs_names

    # marker_pair_coloc_by_component_list = []
    marker_pair_intrsct_list = []
    marker_pair_union_list = []

    assert all([t in ('autocorr', 'coloc') for t in score_types])

    layers = {'actual': int, 'expected': float, 'variance': float, 'z': float, 'normalization': int}

    results = {score_type: {layer: [] for layer in layers.keys()} for score_type in score_types}

    if detailed_info:
        detailed_info_dict = {
            'upi_connectivity': [],
            'nbhd_marker_counts': [],
            'upia_marker_counts': [],
            'upib_marker_counts': [],
            'upia_to_upia_int': [],
        }

    marker_names = adata.var_names
    marker_pair_tuples = get_marker_pair_tuples(marker_names, only_diff=True, ordered_pairs=False) 

    counts_df = adata.to_df(count_layer)

    if verbose is not False:
        print_w_time('Starting conversion...')

    iter = tqdm(components) if verbose in (True, 'tqdm') else components
    for i_comp, component in enumerate(iter):
        # component_edgelist_orig = pg_data.edgelist_lazy.filter(pl.col('component') == component).select(['upia', 'upib', 'marker']).collect().to_pandas()
        component_edgelist_orig = pg_edgelist[pg_edgelist['component'] == component]

        component_counts = counts_df.loc[component]
        N = component_counts.sum()
        freqs = component_counts / component_counts.sum()

        upi_s = 'upia_int' if pxl_type == 'a' else 'upib_int'
        upi_s_other = 'upib_int' if pxl_type == 'a' else 'upia_int'

        n_upia_unique = component_edgelist_orig['upia'].nunique()
        n_upib_unique = component_edgelist_orig['upib'].nunique()

        upia_unique = pd.DataFrame({'upi': component_edgelist_orig['upia'].unique(), 'upi_type': 'a', 'upia_int': range(0, n_upia_unique)}).set_index('upi')
        upib_unique = pd.DataFrame({'upi': component_edgelist_orig['upib'].unique(), 'upi_type': 'b', 'upib_int': range(n_upia_unique, n_upia_unique + n_upib_unique)}).set_index('upi')

        # upi_unique = pd.concat((upia_unique, upib_unique), axis=0)

        # upia_unique = pd.DataFrame(component_edgelist_orig['upia'].unique()).reset_index().rename(columns={0:'upia'}).set_index('upia').rename(columns={'index':'upia_int'})
        # upib_unique = pd.DataFrame(component_edgelist_orig['upib'].unique()).reset_index().rename(columns={0:'upib'}).set_index('upib').rename(columns={'index': 'upib_int'})

        component_edgelist = component_edgelist_orig.join(upia_unique['upia_int'], on='upia').join(upib_unique['upib_int'], on='upib')
        upia_upib_int_unique = pd_unique_values(component_edgelist, ['upia_int', 'upib_int'], observed=True)

        upia_to_marker_counts = component_edgelist.groupby(['upia_int', 'marker'], observed=True, as_index=False).size().rename(columns={'size': 'Count'}).rename(columns={'upia_int': 'upi_int'})
        upia_to_marker_counts['type'] = 'a'
        upib_to_marker_counts = component_edgelist.groupby(['upib_int', 'marker'], observed=True, as_index=False).size().rename(columns={'size': 'Count'}).rename(columns={'upib_int': 'upi_int'})
        upib_to_marker_counts['type'] = 'b'

        base_upi_to_marker_counts = upia_to_marker_counts if pxl_type == 'a' else upib_to_marker_counts
        other_upi_to_marker_counts = upib_to_marker_counts if pxl_type == 'a' else upia_to_marker_counts

        # if group_markers:
        #     upi_to_grouped_marker_counts = upi_to_ungrouped_marker_counts.copy()
        #     upi_to_grouped_marker_counts['Count'] = 1

        upi_connectivity = upia_unique if pxl_type =='a' else upib_unique
        upi_connectivity = upi_connectivity.reset_index()[[upi_s]].rename(columns={upi_s: f'{upi_s}_0'})

        for i in range(nbhd_radius):
            if i % 2 == 0:
                upi_connectivity = upi_connectivity.join(upia_upib_int_unique.set_index(upi_s)[upi_s_other], on=f'{upi_s}_{i}').rename(columns={upi_s_other: f'{upi_s_other}_{i+1}'})
            else:
                upi_connectivity = upi_connectivity.join(upia_upib_int_unique.set_index(upi_s_other)[upi_s], on=f'{upi_s_other}_{i}').rename(columns={upi_s:f'{upi_s}_{i+1}'})

            upi_connectivity = pd_unique_values(upi_connectivity, col_names=upi_connectivity.columns.values, observed=True)

        last_is_other = nbhd_radius % 2 != 0
        last_pixel_type = upi_s_other if last_is_other else upi_s
        last_pixel_type_marker_counts = other_upi_to_marker_counts if last_is_other else base_upi_to_marker_counts

        pxl_0 = f'{upi_s}_0'
        pxl_n = f'{last_pixel_type}_{nbhd_radius}'
        center_pxl = f'nbhd_center_upi_int'

        # center_marker = 'marker_1'
        # nbhd_marker = 'marker_2'

        # center_count = 'Count_1'
        # nbhd_count = 'Count_2'            

        # print(upi_connectivity)
        upi_connectivity_short = pd_unique_values(upi_connectivity, col_names=[pxl_0, pxl_n], observed=True)

        # Count each pair of pixels only once
        # upi_connectivity = upi_connectivity[upi_connectivity[center_pxl] <= upi_connectivity[nbhd_pxl]]

        upi_total_counts = base_upi_to_marker_counts.groupby('upi_int', observed=True)['Count'].sum()

        nbhds_markers = upi_connectivity_short.join(last_pixel_type_marker_counts.set_index('upi_int')[['marker', 'Count']], on=pxl_n)[[pxl_0, 'marker', 'Count']].rename(
            columns={pxl_0: center_pxl}).groupby([center_pxl, 'marker'], as_index=False, observed=True)['Count'].sum()
        nbhds_total_counts = nbhds_markers.groupby(center_pxl, observed=True)['Count'].sum()

        # nbhds_markers = nbhds_markers.groupby([center_pxl, 'marker'], observed=True)['Count'].sum().reset_index().rename(columns={0: 'Count'})
        # if group_markers:
        #     nbhds_markers['Count'] = 1
        # nbhds_markers['Count'] = nbhds_markers['Count'].astype(int)

        # Table showing marker counts per pixel

        upi_markers_pivoted = pd.pivot_table(base_upi_to_marker_counts, values='Count', index='upi_int', columns='marker', observed=True, fill_value=0).astype(int)

        # Table showing marker counts per neighborhood (each nbhd is defined by a center pixel; neighborhoods overlap)
        nbhds_markers_pivoted = pd.pivot_table(nbhds_markers, values='Count', index=center_pxl, columns='marker', observed=True, fill_value=0,).astype(int)

        for marker in marker_names:
            for tbl in (upi_markers_pivoted, nbhds_markers_pivoted):
                if marker not in tbl.columns:
                    tbl[marker] = 0
        


        def f(p):
            return p*(1-p)
        
        def g(p, n):
            return np.power(1-p, n)
        
        if results.get('autocorr'):

            autocorr = pd.Series(
                [((upi_total_counts)*(nbhds_markers_pivoted[m] > 0).astype(int)).sum(axis=0) for m in marker_names],
                index=marker_names
            )

            expected_autocorr = pd.Series(
                [N - ((upi_total_counts)*g(freqs[m], nbhds_total_counts)).sum(axis=0) for m in marker_names],
                index=marker_names
            )

            variance_autocorr = pd.Series(
                [(np.square(upi_total_counts)*f(g(freqs[m], nbhds_total_counts))).sum(axis=0) for m in marker_names],
                index=marker_names
            )

            results['autocorr']['actual'].append(autocorr)
            results['autocorr']['expected'].append(expected_autocorr)
            results['autocorr']['variance'].append(variance_autocorr)
            results['autocorr']['z'].append(((autocorr - expected_autocorr) / np.sqrt(variance_autocorr)).replace([np.inf, -np.inf, np.nan], 0))


        def h(p1, p2, n):
            return np.power(1-p1, n) + np.power(1-p2, n) - np.power(1-p1-p2, n)
        
        if results.get('coloc'):


            coloc = pd.Series(
                [((upi_total_counts)*((nbhds_markers_pivoted[m1] > 0).astype(int) & (nbhds_markers_pivoted[m2] > 0).astype(int))).sum(axis=0) for m1, m2 in marker_pair_tuples],
                index=marker_pair_tuples,
            )

            normalization_coloc = pd.Series(
                [((upi_total_counts)*((nbhds_markers_pivoted[m1] > 0).astype(int) | (nbhds_markers_pivoted[m2] > 0).astype(int))).sum(axis=0) for m1, m2 in marker_pair_tuples],
                index=marker_pair_tuples,
            )

            expected_coloc = pd.Series(
                [N - ((upi_total_counts)*(h(freqs[m1], freqs[m2], nbhds_total_counts))).sum(axis=0) for m1, m2 in marker_pair_tuples],
                index=marker_pair_tuples,
            )

            variance_coloc = pd.Series(
                [(np.square(upi_total_counts)*(f(h(freqs[m1], freqs[m2], nbhds_total_counts)))).sum(axis=0) for m1, m2 in marker_pair_tuples],
                index=marker_pair_tuples,
            )


            results['coloc']['actual'].append(coloc)
            results['coloc']['expected'].append(expected_coloc)
            results['coloc']['normalization'].append(normalization_coloc)
            results['coloc']['variance'].append(variance_coloc)
            results['coloc']['z'].append(((coloc - expected_coloc) / np.sqrt(variance_coloc)).replace([np.inf, -np.inf, np.nan], 0))


        # nbhds_m2_binary = []
        # upi_m1 = []
        # upi_total_times_binary = []
        # m1_freqs = []
        # m2_freqs = []
        # m1_total = []
        # prob_no_m2_in_nbhds = []

        # pixel_sum_proportion = {}


        # for m1, m2 in marker_pair_tuples:
        #     if m1 != m2:
        #         cur_m2_binary = (nbhds_markers_pivoted[m2] > 0).astype(int)
        #     else:
        #         cur_m2_binary = (nbhds_markers_pivoted[m2] > 1).astype(int)
        #     nbhds_m2_binary.append(cur_m2_binary)
        #     upi_m1.append(upi_markers_pivoted[m1])
        #     upi_total_times_binary.append(upi_total_counts*cur_m2_binary)
        #     # prob_no_m2_in_nbhds.append(np.power(1 - component_freqs[m2], nbhds_total_counts))

        #     m1_freqs.append(component_freqs[m1])
        #     m1_total.append(component_counts[m1])
        #     m2_freqs.append(component_freqs[m2])



        # # nbhds_m2_binary, upi_m1, upi_total_times_binary, prob_no_m2_in_nbhds  = [pd.DataFrame(l, index=marker_pair_tuples).T for l in  
        # #         (nbhds_m2_binary, upi_m1, upi_total_times_binary, prob_no_m2_in_nbhds)]
        
        # nbhds_m2_binary, upi_m1, upi_total_times_binary = [pd.DataFrame(l, index=marker_pair_tuples).T for l in  
        #         (nbhds_m2_binary, upi_m1, upi_total_times_binary)]
        
        # m1_freqs, m2_freqs, m1_total = [pd.Series(l, index=marker_pair_tuples) for l in (m1_freqs, m2_freqs, m1_total)]
                
        # coloc = pd.Series((upi_m1*nbhds_m2_binary).sum(axis=0), index=marker_pair_tuples)
        # expected_coloc = pd.Series(m1_freqs*(upi_total_times_binary).sum(axis=0), index=marker_pair_tuples)
        # variance_coloc = pd.Series((m1_freqs)*(1-m1_freqs)*(upi_total_times_binary).sum(axis=0), index=marker_pair_tuples)
        # # experimental_exp = pd.Series(
        # #     m1_total - (m1_freqs / (1 - m2_freqs))*(prob_no_m2_in_nbhds.mul(upi_total_counts, axis=0)).sum(axis=0),
        # #     index=marker_pair_tuples
        # # )
        # # experimental_variance = pd.Series(
        # #     (m1_freqs / (1 - m2_freqs)).pow(2) * (prob_no_m2_in_nbhds*(1 - prob_no_m2_in_nbhds).mul(upi_total_counts.pow(2), axis=0)).sum(axis=0) + \
        # #     + m1_freqs*(1 - m1_freqs - m2_freqs) / (1 - m2_freqs).pow(2) * (prob_no_m2_in_nbhds.mul(upi_total_counts, axis=0)).sum(axis=0),
        # #     index=marker_pair_tuples,
        # # )

                # coloc_list.append(coloc)
        # expected_coloc_list.append(expected_coloc)
        # variance_coloc_list.append(variance_coloc)


        '''
        # For markers m1 and m2, the contribution of every pixel k to coloc(m1,m2) is the count of m1 in k, if the neighborhood of k includes at least one marker m2
        coloc = pd.Series(
            [
                (upi_markers_pivoted[m1]*(nbhds_markers_pivoted[m2].astype(bool).astype(int))).sum() for m1, m2 in marker_pair_tuples
            ], index=marker_pair_tuples
        )


        # Expected value of coloc(m1,m2) under null hypothesis is the cell-wide frequency of m1 times the sum of molecules in the vicinity of m2 molecules
        expected_coloc = pd.Series(
            [
                component_freqs[m1]*(
                    upi_total_counts*(nbhds_markers_pivoted[m2].astype(bool).astype(int))
                ).sum() for m1, m2 in marker_pair_tuples
            ], index=marker_pair_tuples
        )

        # Variance is same but with another factor (1-freq(m1))
        variance_coloc = pd.Series(
            [
                (1-component_freqs[m1])*component_freqs[m1]*(
                    upi_total_counts*(nbhds_markers_pivoted[m2].astype(bool).astype(int))
                ).sum() for m1, m2 in marker_pair_tuples
            ], index=marker_pair_tuples
        )
        '''

        
        # marker_pair_coloc = pd.Series(
        #     [
        #         (nbhds_markers_pivoted[pair[0]]*(nbhds_markers_pivoted[pair[1]].astype(bool).astype(int))).sum() for pair in marker_diff_pair_names
        #     ], index=marker_diff_pair_names,
        # )

        # experimental_exp_list.append(experimental_exp)
        # experimental_variance_list.append(experimental_variance)

        if detailed_info:
            for (df, name) in zip((nbhds_markers, upia_unique, upia_to_marker_counts, upib_to_marker_counts, upi_connectivity), 
                                  ('nbhd_marker_counts', 'upia_to_upia_int', 'upia_marker_counts', 'upib_marker_counts', 'upi_connectivity')):
                df['component'] = component
                detailed_info_dict[name].append(df)

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

    for score_type in score_types:
        for layer, dtype in layers.items():
            results[score_type][layer] = pd.DataFrame(results[score_type][layer], index=components, dtype=dtype)


    # coloc_layers = {}

    # # for res_name, res_list, res_dtype in zip(('coloc', 'expected_coloc', 'variance_coloc', 'experimental_exp', 'experimental_variance'), 
    # #                                          (coloc_list, expected_coloc_list, variance_coloc_list, experimental_exp_list, experimental_variance_list), 
    # #                                          (int, float, float, float, float)):
    # for res_name, res_list, res_dtype in zip(('coloc', 'expected_coloc', 'variance_coloc'), 
    #                                         (coloc_list, expected_coloc_list, variance_coloc_list), 
    #                                         (int, float, float)):
    #     coloc_layers[res_name] = pd.DataFrame(res_list, index=components, dtype=res_dtype)
    
    # coloc_layers['z_coloc'] = (coloc_layers['coloc'] - coloc_layers['expected_coloc']) / (coloc_layers['variance_coloc']).map(np.sqrt)
    # coloc_layers['z_coloc'].replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    # autocorr_layers = {}
    # for res_name, res_list, res_dtype in zip(('autocorr', 'expected_autocorr', 'variance_autocorr'), 
    #                                     (autocorr_list, expected_autocorr_list, variance_autocorr_list), 
    #                                     (int, float, float)):
    #     autocorr_layers[res_name] = pd.DataFrame(res_list, index=components, dtype=res_dtype)
    
    # autocorr_layers['z_coloc'] = (coloc_layers['coloc'] - coloc_layers['expected_coloc']) / (coloc_layers['variance_coloc']).map(np.sqrt)
    # autocorr_layers['z_coloc'].replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    # layers['experimental_z_coloc'] = (layers['coloc'] - layers['experimental_exp']) / (layers['experimental_variance']).map(np.sqrt)
    # layers['experimental_z_coloc'].replace([np.inf, -np.inf], 0, inplace=True)

    if results.get('coloc'):
        for df in results['coloc'].values():
            # Rename columns to strings instead of tuples
            df.rename(
                columns={(m1, m2): pair2str(m1,m2) for m1,m2 in df.columns.values},
                inplace=True
            )
    
    ret = dotdict({
        'results': results,
        'info': dict(nbhd_radius=nbhd_radius, pxl_type=pxl_type, marker_pair_tuples=marker_pair_tuples)
    })

    # return_dict = {
    #     'marker_pair_intersection': marker_pair_intrsct_df,
    #     'marker_pair_union': marker_pair_union_df,
    #     'marker_pair_names_tuples': marker_pair_names_tuples,
    # }

    if detailed_info:
        for k, v in detailed_info_dict.items():
            ret.info[k] = pd.concat(v, axis=0)
    
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


def latent_to_distr_correlation(adata, latent_key_lst, names, distr_key, vars=None, n_pcs_lst=None, num_pairs=1000, seed=0):
    '''
    Pass n_pcs to select a number of PCs if latent_key is a PCA representation
    '''
    rng = np.random.default_rng(seed=seed)
    obs_pairs = [rng.choice(len(adata.obs), size=(2,)) for _ in range(num_pairs)]

    if n_pcs_lst is None:
        n_pcs_lst = [None]*len(names)

    ret = []

    for (latent_key, n_pcs, name) in zip(latent_key_lst, n_pcs_lst, names):

        distr = adata.obsm[distr_key]
        latent_dists = []
        distr_dists = []
        latent = np.array(adata.obsm[latent_key])
        if n_pcs is not None:
            latent = latent[:, :n_pcs]
        if vars is None:
            vars = distr.columns.values
        distr = distr.loc[:, vars]

        distr = np.array(distr)

        for obs1, obs2 in obs_pairs:
            latent_dists.append(np.linalg.norm(latent[obs1, :] - latent[obs2, :]))
            distr_dists.append(np.linalg.norm(distr[obs1, :] - distr[obs2, :]))
        
        ret.append(
            {
                'Latent': name,
                'Corr': scipy.stats.pearsonr(latent_dists, distr_dists)[0],
            }
        )
    
    return pd.DataFrame(ret)
         


def add_one_hot_encoding_obsm(adata, obs_column, key_added=None):
    if not key_added:
        key_added = obs_column
    adata.obsm[obs_column] = pd.get_dummies(adata.obs[[obs_column]], dtype=int)

def get_rep(adata, rep_name):
    '''
    Retrieves a representation from adata, checking first in layers and then in obsm.
    If rep_name is None returns main layer.
    '''
    if rep_name is None:
        rep = adata.to_df()
    elif rep_name in adata.layers:
        rep = adata.to_df(rep_name)
    elif rep_name in adata.obsm:
        rep = adata.obsm[rep_name]
    else:
        raise RuntimeError(f'{rep_name} not in layers or obsm')
    return rep


def distr_autocorrelation_in_latent(adata, latent_neighbors_connectivities_key_lst, names, distr_key, vars=None):
    '''
    type = morans or gearys
    vars: list of vars to include in computation in distr_key
    '''
    ret = []


    for (neighbors_key, name) in zip(latent_neighbors_connectivities_key_lst, names):

        # Need to work around the fact that use_graph is not supported in anndata
        s_conn = 'connectivities'
        requested_conn = adata.obsp[neighbors_key]
        existing_conn = adata.obsp.get(s_conn)
        adata.obsp[s_conn] = requested_conn

        distr = get_rep(adata, distr_key)
        if vars is None:
            vars = distr.columns.values
        distr = distr.loc[:, vars]
        distr = distr.to_numpy().T

        autocorr = pd.DataFrame(
            {
                'morans': sc.metrics.morans_i(adata, vals=distr),
                'gearys': sc.metrics.gearys_c(adata, vals=distr),
                'latent': name,
            },
            index=vars,
        )

        if existing_conn is not None:
            adata.obsp[s_conn] = existing_conn
        
        ret.append(autocorr)

    return pd.concat(ret, axis=0)


def significant_hits_by_cluster_size(adata, neighbors_key_lst, latent_name_lst, distr_key):
    '''
    neighbors_key_lst: neighbor keys to compare
    latent_name_lst: name of latent space for each neighbor key
    adata: adata including neighbors_key with which to cluster
    distr_key: obsm key to calculate the rank genes group on
    '''
    adata = adata.copy()
    distr_adata = anndata.AnnData(X=adata.obsm[distr_key], obs=adata.obs)
    distr_adata.var_names = adata.obsm[distr_key].columns.values
    resolution_range = np.arange(0.2, 2, 0.1)
    de = []
    for latent_name, neighbors_key in zip(latent_name_lst, neighbors_key_lst):
        print(f'Computing for {latent_name} for a range of resolutions...')
        for r in tqdm(resolution_range):
            leiden_key = f'leiden_{latent_name}_{r:.1f}'
            de_key = f'de_{latent_name}_{r:.1f}'
            sc.tl.leiden(adata, resolution=r, random_state=0, n_iterations=-1, 
                        key_added=leiden_key, flavor='leidenalg', neighbors_key=neighbors_key, copy=False)


            n_clusters = adata.obs[leiden_key].nunique()
            distr_adata.obs = distr_adata.obs.join(adata.obs[leiden_key])
            sc.tl.rank_genes_groups(distr_adata, groupby=leiden_key, method='wilcoxon', key_added=de_key)
            significant_hits_df = sc.get.rank_genes_groups_df(distr_adata, group=None, key=de_key, pval_cutoff=0.05)
            de.append(
                {
                    'Model': latent_name,
                    'n_clusters': n_clusters,
                    'resolution': r,
                    'n_significant_hits' : len(significant_hits_df)
                }
            )

    de_df = pd.DataFrame(de)
    return de_df


def multiple_correlation(df, vars, target):
    '''
    Calculates the R^2 score of https://en.wikipedia.org/wiki/Coefficient_of_multiple_correlation
    df: Dataframe containing source and target variables
    vars: list of source variables
    target: name of target variable
    '''
    corrs = df[vars + [target]].corr()
    R_xx = corrs.loc[vars, vars].to_numpy()
    c = corrs.loc[vars, target].to_numpy().reshape((2,1))
    return (c.T @ np.linalg.solve(R_xx, c)).item()


def pairs_to_marginals_correlation(counts_and_coloc_df, marker_names):
    '''
    Based on the R^2 score of https://en.wikipedia.org/wiki/Coefficient_of_multiple_correlation
    Ranks the pairs by the extent to which they are a linear function of the two marginal counts.

    expects df of (n_cells, n_features) where features includes both count and coloc. 
    The values should be in raw count space
    '''
    pair_names = get_marker_pair_names(marker_names)
    means = counts_and_coloc_df.mean(axis=0)
    coeff = []
    m1_means = []
    m2_means = []
    for pair_str in pair_names:
        m1, m2 = str2pair(pair_str)
        coeff.append(multiple_correlation(counts_and_coloc_df, vars=[m1, m2], target=pair_str))
        m1_means.append(means[m1])
        m2_means.append(means[m2])
    return pd.DataFrame({'R2': coeff, 'm1_mean_counts': m1_means, 'm2_mean_counts': m2_means}, index=pair_names)
        


def calc_PCA(adata, key_added, rep=None, layer=None, obsm=None, **kwargs):
    '''
    Extends sc.pp.PCA to allow arbitrary representation for PCA and arbitrary key name to add
    Can pass any of the PCA kwargs (except copy / return_info)
    Can pass rep instead of layer/obsm and then they will be searched (in that order)
    Note, varm['PCs'] will not be registered
    adata will have new obsm field called {key_added}_pca and new uns field called {key_added}_pca_info
    Returns None (changes adata)
    '''
    if all((not rep, not layer, not obsm)):
        X = adata.X
    if rep:
        X = adata.layers.get(rep)
        if X is None:
            X = adata.obsm.get(rep)
        if X is None:
            raise RuntimeError('rep is not in layers or obsm')
    else:
        if all((layer, obsm)):
            raise RuntimeError('Cannot pass both layer and obsm')
        if layer:
            X = adata.layers[layer]
        else:
            X = adata.obsm[obsm]
    new_adata = anndata.AnnData(X=X)
    pca_data = sc.pp.pca(new_adata, copy=True, **kwargs)
    adata.obsm[f'{key_added}_pca'] = pca_data.obsm['X_pca']
    adata.uns[f'{key_added}_pca_info'] = pca_data.uns['pca']


def calc_hvg(adata=None, rep=None, layer=None, obsm=None, var_names=None, **kwargs):
    '''
    Extends sc.pp.highly_variable_genes to allow arbitrary representation (obsm)
    Can pass any of the original kwargs (except inplace, which will always be false)
    For the representation, can pass either adata + layer/obsm name, or adata=None and rep = array / df
    If the representation is not a df, must pass var_names
    Returns metric df
    '''
    if adata is None:
        assert all((rep is not None, layer is None, obsm is None))
        final_rep = rep
    else:
        assert rep is None
        if obsm is not None:
            assert layer is None
            final_rep = adata.obsm[obsm]
        else:
            assert obsm is None
            final_rep = adata.to_df(layer)
    if type(final_rep) != pd.DataFrame:
        if var_names is None:
            raise ValueError('If representation is not a dataframe, must pass var names')
        new_adata.var_names = var_names
    new_adata = anndata.AnnData(X=final_rep)
    metrics = sc.pp.highly_variable_genes(new_adata, inplace=False, **kwargs)
    return metrics

def fill_with_mean(arr):
    '''
    Fills NaN and inf values with mean of columns
    arr: df or np array
    Returns copy of arr
    '''
    arr = arr.copy()
    if type(arr) == np.ndarray:
        nan_inds = np.where(np.isnan(arr) | np.isinf(arr))
        arr[nan_inds] = np.take(np.nanmean(arr, axis=0), nan_inds[1])
    elif type(arr) == pd.DataFrame:
        arr.replace([np.inf, -np.inf], [np.nan, np.nan], inplace=True)
        arr.replace(np.nan, arr.mean(axis=0), inplace=True)
    return arr

    
def plot_cumulative_variance(adata, pca_info_key='pca', ax=None, n_pcs=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    var_ratio_accum = np.cumsum(adata.uns[pca_info_key]['variance_ratio'])
    if n_pcs is None:
        n_pcs = len(var_ratio_accum)
    sns.scatterplot(x=list(range(1, n_pcs + 1)), y=var_ratio_accum, ax=ax)
    title = title if title is not None else 'Cumulative Variance Explained'
    ax.set_title(title)
    return ax


def compute_graph_layout(pg_data, component):
    return pg_data.precomputed_layouts.filter(component_ids=[component]).to_df()


def plot_cell_with_markers(graph_layout_data, marker_1, marker_2=None, norm=True, **marker_kwargs):
    '''
    marker_1 will be drawn as red, marker_2 as blue
    '''

    pio.renderers.default = "plotly_mimetype+notebook_connected"

    # --- Create the sphere surface ---
    # Generate a mesh in spherical coordinates.
    theta = np.linspace(0, np.pi, 40)      # polar angle
    phi = np.linspace(0, 2 * np.pi, 40)      # azimuthal angle
    theta, phi = np.meshgrid(theta, phi)

    # Parametric equations for a unit sphere.
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Create the surface trace with a light gray color.
    sphere = go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, 'lightgray'], [1, 'lightgray']],  # light gray surface
        showscale=False,  # hides the color scale bar
        opacity=1,
        lighting=dict(
            ambient=1,   # maximize ambient light
            diffuse=0,   # remove diffuse reflections
            specular=0   # remove specular reflections
        ),
        lightposition=dict(
            x=0,
            y=0,
            z=1000  # position the light source far away to further reduce shading nuances
        )
    )

    if marker_2 is None:
        graph_layout_data['color'] = ['firebrick' if graph_layout_data[marker_1] > 0 else 'gray']
    else:
        graph_layout_data['color'] =['darkmagenta' if (expr_1 > 0 and expr_2 > 0) else ('firebrick' if expr_1 > 0 else ('royalblue' if expr_2 > 0 else 'gray')) for 
                                    (expr_1, expr_2) in zip(graph_layout_data[marker_1], graph_layout_data[marker_2])]
    
    if marker_kwargs is None:
        marker_kwargs = dict(size=3)
    else:
        if 'size' not in marker_kwargs:
            marker_kwargs['size'] = 3

    push_out = 1.05

    markers = go.Scatter3d(
        x=graph_layout_data['x_norm']*push_out,
        y=graph_layout_data['y_norm']*push_out,
        z=graph_layout_data['z_norm']*push_out,
        mode="markers",
        marker=marker_kwargs,
        marker_color=graph_layout_data['color'],
    )


    fig = go.Figure(data=[sphere, markers])
    fig.update_layout(scene=dict(aspectmode='data'))

    return fig


def plot_cell_smooth(graph_layout_data, marker_1, marker_2=None):

    # --- Create a spherical grid ---
    n_theta = 100  # number of points along polar angle (theta)
    n_phi   = 100  # number of points along azimuth (phi)

    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    # Create a meshgrid where theta varies along rows and phi along columns
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    # Parametric equations for a unit sphere:
    x_sphere = np.sin(theta_grid) * np.cos(phi_grid)
    y_sphere = np.sin(theta_grid) * np.sin(phi_grid)
    z_sphere = np.cos(theta_grid)

    if marker_2 is not None:
        data = np.array(graph_layout_data[['x_norm', 'y_norm', 'z_norm', marker_1, marker_2]])
    else:
        graph_layout_data['tmp'] = 0
        data = np.array(graph_layout_data[['x_norm', 'y_norm', 'z_norm', marker_1, 'tmp']])
        graph_layout_data.drop(columns='tmp', inplace=True)

    points = data[:, :3]  # (x, y, z) coordinates
    A_values = data[:, 3].astype(bool).astype(float)
    B_values = data[:, 4].astype(bool).astype(float)

    # Flatten the sphere coordinates for interpolation
    pts_sphere = np.column_stack((x_sphere.flatten(), y_sphere.flatten(), z_sphere.flatten()))

    # --- Interpolate A and B on the sphere surface ---
    # For points where data is missing, fill with 0 (black)
    A_interp = griddata(points, A_values, pts_sphere, method='nearest', fill_value=0)
    B_interp = griddata(points, B_values, pts_sphere, method='nearest', fill_value=0)


    # Reshape interpolated values back to the grid shape
    A_interp = A_interp.reshape(x_sphere.shape)
    B_interp = B_interp.reshape(x_sphere.shape)

    # --- Compute per-vertex colors ---
    # We want each vertex to have an RGB color of (int(255*A), 0, int(255*B))
    # where missing data (A=B=0) gives black.
    colors = np.empty(x_sphere.shape, dtype=object)
    for i in range(x_sphere.shape[0]):
        for j in range(x_sphere.shape[1]):
            r = int(np.clip(A_interp[i, j] * 255, 0, 255))
            b = int(np.clip(B_interp[i, j] * 255, 0, 255))
            colors[i, j] = f'rgb({r},0,{b})'

    # --- Build the mesh for go.Mesh3d ---
    # Flatten the grid arrays (each vertex)
    x_flat = x_sphere.flatten()
    y_flat = y_sphere.flatten()
    z_flat = z_sphere.flatten()
    vertex_colors = colors.flatten()

    # Create the triangular faces.
    # The grid is (n_theta x n_phi) so each cell forms two triangles.
    triangles = []
    for i in range(n_theta - 1):
        for j in range(n_phi - 1):
            idx = i * n_phi + j
            idx_right = idx + 1
            idx_down = idx + n_phi
            idx_down_right = idx_down + 1
            # Triangle 1 of the cell:
            triangles.append([idx, idx_right, idx_down])
            # Triangle 2 of the cell:
            triangles.append([idx_right, idx_down_right, idx_down])
    triangles = np.array(triangles)

    # --- Create the Mesh3d trace ---
    mesh = go.Mesh3d(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        vertexcolor=vertex_colors,  # assign our per-vertex colors
        flatshading=True,
        showscale=False  # no color scale bar
    )

    # --- Plot the figure ---
    fig = go.Figure(data=[mesh])
    fig.update_layout(scene=dict(aspectmode='data'))

    return fig

def center_feature_matrix(arr):
    return (arr - arr.mean(axis=0)) / (arr.std(axis=0))


def convert_polarization_to_feature_matrix(polarization, components, vars=None, key='morans_i', var_suffix='pol', fill_na_with_mean=True, rescale=False):
    '''
    Receives polarization data in longform format (as in pixelgen datasets) and converts to a data format for usage in an anndata (obs x features)
    '''
    longform = polarization[['component', 'marker', key]]
    longform = longform[longform['component'].isin(components)]
    result = longform.pivot(index='component', columns='marker', values=key)
    if vars:
        result = result[vars]
    if var_suffix:
        result.columns = [f'{col}_{var_suffix}' for col in result.columns]
    if fill_na_with_mean:
        result = fill_with_mean(result)
    if rescale:
        result = (result + 1) / 2
    result = result.reindex(components)
    return result

def convert_colocalization_to_feature_matrix(coloc, components, vars=None, pairs=None, key='pearson', var_suffix=None, fill_na_with_mean=True, rescale=False):
    '''
    Receives colocalization data in longform format (as in pixelgen datasets) and converts to a data format for usage in an anndata (obs x features)
    If vars is not None, only vars from the list will be put into pairs and considered.
    If pairs is not None, only pairs from the list generated (from all vars or specified vars) will be considered.
    Note, pairs must be sorted.
    '''
    longform_coloc = coloc[['component', 'marker_1', 'marker_2', key]].copy()
    if vars:
        longform_coloc['is_var'] = longform_coloc['marker_1'].isin(vars) & longform_coloc['marker_2'].isin(vars)
        longform_coloc = longform_coloc[longform_coloc['is_var']]
    longform_coloc['pair'] = [pair2str(m1, m2, sort=True) for m1, m2 in zip(longform_coloc['marker_1'], longform_coloc['marker_2'])]
    if pairs:
        longform_coloc['is_pair'] = longform_coloc(['pair']).isin(pairs)
        longform_coloc = longform_coloc[longform_coloc['is_pair']]
    result = longform_coloc.pivot(index='component', columns='pair', values=key)
    if var_suffix:
        result.columns = [f'{col}_{var_suffix}' for col in result.columns]
    if fill_na_with_mean:
        result = fill_with_mean(result)
    if rescale:
        if key == 'pearson':
            result = (result + 1) / 2
    result = result.reindex(components)
    return result

def get_pxl_dataset(base_url, base_dataset_dir, filenames, aggregation_sample_names):
    for filename in filenames:
        os.system(f'curl -L -O -C - --create-dirs --output-dir {base_dataset_dir} {base_url}/{filename}')

    datasets = [pixelator.read(base_dataset_dir / filename) for filename in filenames]
    pg_data = pixelator.simple_aggregate(
        aggregation_sample_names, datasets
    )
    return pg_data


def plot_histograms(adata=None, keys=None, dfs=None, names=None, vars=None, hue=None, n_cols=8, kind='kde'):
    '''
    Pass either adata with obsm keys, or list of dfs with same columns
    If passing hue, hue must be an entry in adata.obs, and keys must be of length 1
    Returns flattened axes
    keys: obsm keys
    kind: kde/hist

    '''
    assert dfs or all((adata, keys))
    assert not hue or len(keys) == 1
    if adata:
        dfs = [adata.obsm[key] for key in keys]
    if vars is None:
        vars = dfs[0].columns
    if names is None:
        if keys:
            names = keys
        else:
            names = [f'{i}' for i in range(len(dfs))]
    n_plots = len(vars)
    n_rows = (n_plots + n_cols - 1) // n_cols 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = axes.flatten()
    if hue:
        dfs[0] = dfs[0].copy()
        dfs[0][hue] = adata.obs[hue]
    for ax, var in zip(axes, vars):
        plot_f = {'kde': sns.kdeplot, 'hist': sns.histplot}[kind]
        kwargs = {'kde': dict(fill=True, alpha=0.5), 'hist': dict(alpha=0.5)}[kind]
        for i, df in enumerate(dfs):
            if hue:
                plot_f(df, x=var, ax=ax, hue=hue, legend=(ax==axes[0]), **kwargs)
            else:
                plot_f(df, x=var, ax=ax, color=sns.color_palette()[i], label=names[i], **kwargs)
        if ax == axes[0] and not hue:
            ax.legend(loc='lower left', bbox_to_anchor=(0, 1.05))
        # ax.set_title(var)
    
    fig.tight_layout()
    return axes

def arr_to_df(arr, df):
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def plot_mean_std(df, axis=0, ax=None, **kwargs):
    ax = sns.scatterplot(x=df.mean(axis=axis), y=df.std(axis=axis), ax=ax, **kwargs)
    ax.set_xlabel('Mean')
    ax.set_ylabel('Std')
    return ax

def pca_neighbors_umap(adata, latent_name, neighbors_kwargs={}, umap_tl_kwargs={}, umap_pl_kwargs={}, umap_title=None,):
    calc_PCA(adata, rep=latent_name, key_added=latent_name)
    sc.pp.neighbors(adata, use_rep=f'{latent_name}_pca', key_added=latent_name, **neighbors_kwargs)
    sc.tl.umap(adata, neighbors_key=latent_name, **umap_tl_kwargs)
    fig = sc.pl.umap(adata, **umap_pl_kwargs, return_fig=True)
    if umap_title:
        fig.suptitle(umap_title)
    return fig


def train_model(adata, model_cls, setup_kwargs, model_kwargs, train_kwargs, modalities_latent_names=[], generate_loss_plots=True):
    '''
    modalities_latent_names: list of tuples with name of modality (combined is None, main is 'X') and name of latent to put in adata
    '''
    model_cls.setup_anndata(adata, **setup_kwargs)
    model = model_cls(adata, **model_kwargs)
    model.train(**train_kwargs)
    if generate_loss_plots:
        ax = plot_losses(model)

    for modality, latent_name in modalities_latent_names:
        adata.obsm[latent_name] = model.get_latent_representation(modality=modality)
    
    return model


def _log(s, logger):
    if logger:
        logger.info(s)
    else:
        print(s)


def compute_hotspot_pol_and_coloc(edgelist, adata, components, vars=None, marker_count_threshold=10, knn_neighbors=30, njobs=8, logger=None):
    '''
    Computes polarization and coloc based on Hotspot.
    edgelist: pixelgen edgelist
    adata: must have 'counts' layer
    components: list of components to compute for
    vars: list of vars to compute pol and coloc for (typically, all except isotype controls)
    marker_count_threshold: only compute pol and coloc for markers with counts above this threshold
    knn_neighbors: number of neighbors for kNN graph based on A-pixel graph
    '''
    
    results = []

    _log('Beginning conversion...', logger)

    with ProcessPoolExecutor(max_workers=njobs) as executor:
        futures = [
            executor.submit(
                _process_component, component, i, logger,
                edgelist=edgelist,
                adata=adata,
                vars=vars,
                marker_count_threshold=marker_count_threshold, 
                knn_neighbors=knn_neighbors,
            ) for i, component in enumerate(components)
        ]
        results = [future.result() for future in futures]
        _log('Finished conversion!', logger)

    all_pol, all_coloc = zip(*results)
        
    pol = pd.concat(all_pol, axis=0, ignore_index=True)
    coloc = pd.concat(all_coloc, axis=0, ignore_index=True)

    return pol, coloc

    # all_pol = []
    # all_coloc = []
    # for component in tqdm(components):
    #     try:
    #         with io.capture_output() as captured:
    #             comp_pol, comp_coloc = compute_hotspot_pol_and_coloc_single_component(
    #                 edgelist=edgelist,
    #                 adata=adata,
    #                 component=component, 
    #                 vars=vars,
    #                 marker_count_threshold=marker_count_threshold, 
    #                 knn_neighbors=knn_neighbors,
    #             )
    #     except Exception as e:
    #         print(f'ERROR IN COMPONENT {component}:\n{e}')
    #         break
    #     all_pol.append(comp_pol)
    #     all_coloc.append(comp_coloc)
    # pol = pd.concat(all_pol, axis=0, ignore_index=True)
    # coloc = pd.concat(all_coloc, axis=0, ignore_index=True)
    # return pol, coloc

def _process_component(component, idx, logger, **kwargs):
    _log(f'Calc {idx}', logger)
    try:
        with io.capture_output() as captured:
            pol, coloc = compute_hotspot_pol_and_coloc_single_component(
                component=component, 
                **kwargs,
            )
            return pol, coloc
    except Exception as e:
        _log(f'ERROR IN COMPONENT {component}:\n{e}', logger)
        return pd.DataFrame(), pd.DataFrame()

def compute_hotspot_pol_and_coloc_single_component(edgelist, adata, component, vars=None, marker_count_threshold=10, knn_neighbors=30):
    
    # For graph computation pass all vars, to not disconnect the graph
    results = get_pixel_nbhd_counts(edgelist, adata, pixel_type='a', nbhd_radius=2, 
                                    components=[component], compute_pixel_graph=True, verbose=False, vars=None)
    pixel_counts = results['pixel_counts']
    pixel_counts_pivot = pd.pivot_table(pixel_counts, index='upi_int', columns='marker', values='Count', observed=True, fill_value=0).astype(int)
    g = results['graphs'][0]
    distances = dijkstra(g.A, directed=False, unweighted=True)

    # counts_abundant = pixel_counts_pivot[abundant_markers]
    # sizes_abundant = counts_abundant.sum(axis=1)
    # nonempty = sizes_abundant[sizes_abundant != 0].index
    # counts_abundant = counts_abundant.loc[nonempty]
    # distances_abundant = distances[np.ix_(nonempty, nonempty)]

    # pixel_adata = anndata.AnnData(counts_abundant)
    # pixel_adata.obsp['dijkstra'] = distances_abundant
    # pixel_adata.obs['total_counts'] = pixel_adata.to_df().sum(axis=1)

    pixel_adata = anndata.AnnData(pixel_counts_pivot)
    pixel_adata.obsp['dijkstra'] = distances
    pixel_adata.obs['total_counts'] = pixel_adata.to_df().sum(axis=1)

    hs = hotspot.Hotspot(
        pixel_adata,
        layer_key=None,
        model='bernoulli',
        distances_obsp_key='dijkstra',
        umi_counts_obs_key='total_counts'
    )
    hs.create_knn_graph(weighted_graph=False, n_neighbors=knn_neighbors)
    
    hs_pol = hs.compute_autocorrelations()
    hs_pol = hs_pol.reset_index().rename(columns={'Gene': 'marker', 'C': 'pol_hs'})

    abundant_markers = [v for v in vars if adata.to_df('counts').loc[component, v] > marker_count_threshold]
    hs_pol = hs_pol[hs_pol['marker'].isin(abundant_markers)]

    local_correlations = hs.compute_local_correlations(abundant_markers)

    long_coloc = hs.local_correlation_c.reset_index().melt(id_vars='marker', var_name='marker_2', value_name='coloc_hs').rename(columns={'marker': 'marker_1'})
    long_coloc = long_coloc[long_coloc['marker_1'] != long_coloc['marker_2']]
    long_coloc['pair_name'] = [pair2str(m1, m2, sort=True) for m1, m2 in zip(long_coloc['marker_1'], long_coloc['marker_2'])]
    long_coloc = long_coloc.drop_duplicates(subset='pair_name')

    hs_pol['component'] = component
    long_coloc['component'] = component

    return hs_pol, long_coloc

# coloc_key = 'jaccard'
# longform_coloc = pg_data.colocalization[['component', 'marker_1', 'marker_2', coloc_key]].join(adata.obs['sample_class'], on='component')
# longform_coloc['pair'] = [pair2str(m1, m2, sort=True) for m1, m2 in zip(longform_coloc['marker_1'], longform_coloc['marker_2'])]
# for (sample_class, cur_adata) in zip(('resting', 'stimulated'), (resting_adata, stim_adata)):
#     coloc = longform_coloc[longform_coloc['sample_class'] == sample_class]
#     coloc = coloc.pivot(index='component', columns='pair', values=coloc_key)
#     coloc = coloc.loc[cur_adata.obs.index]
#     cur_adata.obsm['pixelgen_coloc'] = coloc
# pol_key = 'morans_i'
# longform_polarization = pg_data.polarization[['component', 'marker', pol_key]].join(adata.obs['sample_class'], on='component')
# for (sample_class, cur_adata) in zip(('resting', 'stimulated'), (resting_adata, stim_adata)):
#     polarization = longform_polarization[longform_polarization['sample_class'] == sample_class]
#     polarization = polarization.pivot(index='component', columns='marker', values=pol_key)
#     polarization = polarization.loc[cur_adata.obs.index]
#     cur_adata.obsm['pixelgen_polarization'] = polarization
# resting_annotations, stim_annotations = [anndata.read_h5ad(p) for p in ('/home/labs/nyosef/eitangr/pixelgen/PixelGen/datasets/resting_annotated_with_asymm_coloc.h5ad', '/home/labs/nyosef/eitangr/pixelgen/PixelGen/datasets/stim_annotated_with_asymm_coloc.h5ad')]
# for cur_adata, cur_annotations in zip((resting_adata, stim_adata), (resting_annotations, stim_annotations)):
#     cur_adata.obs['cell_type'] = cur_annotations.obs['cell_type']
#     cur_adata.obs['low_res_cell_type'] = cur_annotations.obs['low_res_cell_type']
#     cur_adata.layers['clr_by_abs'] = clr_transformation(cur_adata.to_df(layer='counts'), axis=0)
#     cur_adata.X = cur_adata.layers['dsb']
#     sc.pp.pca(cur_adata, n_comps=50)
#     sc.pp.neighbors(cur_adata, n_neighbors=15)
#     sc.tl.umap(cur_adata)
#     print(cur_adata.obsm.keys())
#     print(cur_adata.layers.keys())
#     sc.pl.umap(cur_adata, color=['sample', 'cell_type', 'low_res_cell_type', 'CD8'], wspace=0.4, return_fig=True)
# resting_adata.write_h5ad('/home/labs/nyosef/eitangr/pixelgen/PixelGen/datasets/resting_data_with_scores_annotated.h5ad')
# stim_adata.write_h5ad('/home/labs/nyosef/eitangr/pixelgen/PixelGen/datasets/stim_data_with_scores_annotated.h5ad')
