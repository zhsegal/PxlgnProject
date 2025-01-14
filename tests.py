import pxl_utils
from pxl_utils import dotdict

import pixelator
import anndata

import os
from pathlib import Path
import itertools
from tqdm import tqdm


from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


import scanpy as sc


def test_convert_edgelist_to_protein_pair_colocalization(verbose=True):

    # One observation with 5 variables with increasing counts
    count_df = pd.DataFrame({f'{i}': [i] for i in range(1, 6)}, index={'obs': ['0']})
    adata = anndata.AnnData(count_df, count_df.index.to_frame(), count_df.columns.to_frame())

    adata = anndata.AnnData(np.array([[5,5,5]]))
    adata.obs_names = ['Cell1']
    adata.var_names = marker_names = [f'Marker{i}' for i in range(1,4)]

    umis_to_markers = {
        **{f'Mol{i}': 'Marker1' for i in range(1,6)},
        **{f'Mol{i}': 'Marker2' for i in range(6, 11)},
        **{f'Mol{i}': 'Marker3' for i in range(11, 16)}
    }

    a_pixels_to_umis = {
        'UPIA1': {'Mol1', 'Mol4', 'Mol5', 'Mol6', 'Mol7',},
        'UPIA2': {'Mol2', 'Mol3', 'Mol8'},
        'UPIA3': {'Mol9', 'Mol10', 'Mol11', 'Mol12'},
        'UPIA4': {'Mol13', 'Mol14', 'Mol15'}
    }
    umis_to_a_pixels = {}
    umis_to_b_pixels = {}

    b_pixels_to_umis = {
        'UPIB1': {'Mol2'}.union(a_pixels_to_umis['UPIA1']),
        'UPIB2': {'Mol9', 'Mol3', 'Mol8'},
        'UPIB3': {'Mol13','Mol10', 'Mol11', 'Mol12'},
        'UPIB4': {'Mol14', 'Mol15'},
    }

    upia_connectivity = {('UPIA1', 'UPIA2'), ('UPIA2', 'UPIA3'), ('UPIA3','UPIA4')}.union({(upia, upia) for upia in a_pixels_to_umis.keys()})

    upia_pxl_to_nbhd = {
        'UPIA1': ('UPIA1', 'UPIA2'),
        'UPIA2': ('UPIA1', 'UPIA2', 'UPIA3'),
        'UPIA3': ('UPIA2', 'UPIA3', 'UPIA4'),
        'UPIA4': ('UPIA3', 'UPIA4'),
    }

    for pxl, mol_set in a_pixels_to_umis.items():
        for mol in mol_set:
            umis_to_a_pixels[mol] = pxl
    
    for pxl, mol_set in b_pixels_to_umis.items():
        for mol in mol_set:
            umis_to_b_pixels[mol] = pxl

    assert set.union(*a_pixels_to_umis.values()) == set.union(*b_pixels_to_umis.values())
    edges = []
    for umi, marker in umis_to_markers.items():
        edges.append(
            {'component': 'Cell1', 'umi': umi, 'upia': umis_to_a_pixels[umi], 'upib': umis_to_b_pixels[umi], 'marker': umis_to_markers[umi]}
        )
    pg_data = dotdict(
        {
            'edgelist': pd.DataFrame(edges),
            'adata': adata,
        }
    )

    apxls_u = a_pixels_to_ungrouped_marker_counts = {
        pixel_name: {
            marker_name: sum([1 for umi in pixel_content if umis_to_markers[umi] == marker_name]) for marker_name in marker_names
        } for (pixel_name, pixel_content) in a_pixels_to_umis.items()
    }

    apxls_g = a_pixels_to_grouped_marker_counts = {
        pixel_name: {
            marker_name: 1 if pixel_ungrouped_counts[marker_name] else 0 for marker_name in marker_names
        } for (pixel_name, pixel_ungrouped_counts) in a_pixels_to_ungrouped_marker_counts.items()
    }

    nbhds_u = nbhds_to_ungrouped_marker_counts = {
         pixel_name: {
            marker_name: sum([apxls_u[nbhd_pxl][marker_name] for nbhd_pxl in nbhd_pixels]) for marker_name in marker_names
        } for (pixel_name, nbhd_pixels) in upia_pxl_to_nbhd.items()       
    }

    nbhds_g = nbhds_to_grouped_marker_counts = {
        pixel_name: {
            marker_name: 1 if any([apxls_g[nbhd_pxl][marker_name] for nbhd_pxl in nbhd_pixels]) else 0 for marker_name in marker_names
        } for (pixel_name, nbhd_pixels) in upia_pxl_to_nbhd.items()
    }




    # ungrouped_protein_pair_coloc = pxl_utils.convert_edgelist_to_protein_pair_colocalization(pg_data, nbhd_radius=1, group_markers=False)
    # print(ungrouped_protein_pair_coloc)
    results = pxl_utils.convert_edgelist_to_protein_pair_colocalization(pg_data, nbhd_radius=1, group_markers=True, detailed_info=True)

    if verbose:

        print(a_pixels_to_umis)
        print(a_pixels_to_ungrouped_marker_counts)
        print(nbhds_u)
        print(list(results.layers['marker_pair_coloc'].iloc[0]))
        print(results.info['nbhds_marker_counts'].join(results.info['upia_to_upia_int'].reset_index().set_index('upia_int')['upia'], on='nbhd_center_upia_int').sort_values(by='upia'))

        print(pd.Series([
            sum([apxls_u[pxl][m1] * int(bool(nbhds_u[pxl][m2])) for pxl in nbhds_u.keys()]) for (m1, m2) in results.info['marker_diff_pair_tuples']
        ], index=results.info['marker_diff_pair_tuples']))

    assert list(results.layers['marker_pair_coloc'].iloc[0]) == [
        sum([apxls_u[pxl][m1] * int(bool(nbhds_u[pxl][m2])) for pxl in nbhds_u.keys()]) for (m1, m2) in results.info['marker_diff_pair_tuples']
    ]


    # assert list(grouped_protein_pair_coloc['marker_pair_intersection'].iloc[0]) == [
    #         sum([nbhds_g[pxl][m1] & nbhds_g[pxl][m2] for pxl in nbhds_g.keys()]) for (m1, m2) in grouped_protein_pair_coloc['marker_pair_names_tuples']
    #     ]

    # assert list(grouped_protein_pair_coloc['marker_pair_union'].iloc[0]) == [
    #         sum([nbhds_g[pxl][m1] | nbhds_g[pxl][m2] for pxl in nbhds_g.keys()]) for (m1, m2) in grouped_protein_pair_coloc['marker_pair_names_tuples']
    #     ]

    # for type in ('grouped', 'ungrouped'):
    #     coloc_df = grouped_protein_pair_coloc if type == 'grouped' else ungrouped_protein_pair_coloc
    #     apxls_c = apxls_g if type == 'grouped' else apxls_u

    #     assert (coloc_df.iloc[0] == pd.Series([
    #         sum([apxls_c[pxl1][marker_pair[0]]*apxls_c[pxl2][marker_pair[1]] + apxls_c[pxl1][marker_pair[1]]*apxls_c[pxl2][marker_pair[0]] \
    #              for pxl1, pxl2 in upia_connectivity]) for marker_pair in coloc_df.columns
    #     ], index=coloc_df.columns,
    #     )).all()


if __name__ == '__main__':
    test_convert_edgelist_to_protein_pair_colocalization()