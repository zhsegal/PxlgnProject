import datetime
import itertools
import seaborn as sns
import pandas as pd
import numpy as np


def log(s, logger):
    if logger:
        logger.info(s)
    else:
        print(s)

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

def get_marker_pairs(marker_names, only_diff=True, only_sorted_pairs=True, as_strings=False):
    '''
    Turn list of marker names into list of marker pair names:
    only_diff: return only pairs of different markers
    only_sorted_pairs: return only the pairs which are sorted in alphabetical order (i.e. removes duplicate pairs)
    as_strings: If true, return the pairs as a list of joined strings. Else, return a list of 2-tuples of strings.
    '''
    diff = (lambda m1, m2 : m1 != m2) if only_diff else (lambda m1, m2: True)
    ordered = (lambda m1, m2: m1 < m2) if only_sorted_pairs else (lambda m1, m2: True)
    marker_pair_tuples = sorted([(m1, m2) for (m1, m2) in itertools.product(marker_names, marker_names) if diff(m1, m2) and ordered(m1, m2)])
    if not as_strings:
        return marker_pair_tuples
    else:
        return [pair2str(*p) for p in marker_pair_tuples] 


def arr_to_df(arr, df):
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def plot_mean_std(df, axis=0, ax=None, **kwargs):
    ax = sns.scatterplot(x=df.mean(axis=axis), y=df.std(axis=axis), ax=ax, **kwargs)
    ax.set_xlabel('Mean')
    ax.set_ylabel('Std')
    return ax

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

def center_feature_matrix(arr):
    return (arr - arr.mean(axis=0)) / (arr.std(axis=0))