import seaborn as sns
import pandas as pd
import anndata
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.distributions import MixtureSameFamily, Categorical
import scanpy as sc
from tqdm import tqdm
from scipy.sparse import csr_matrix

class DistributionConcatenator:
    """Utility class to concatenate Pytorch distributions and move them to cpu.

    All distributions must be of the same type.
    """

    def __init__(self):
        self._params = None
        self._mixtures = []
        self.distribution_cls = None
        self._is_mixture = None

    def store_distribution(self, dist: torch.distributions.Distribution):
        """Add a dictionary of distributions to the concatenator.

        Parameters
        ----------
        dist:
            A Pytorch distribution.
        """
        self._is_mixture = type(dist) == MixtureSameFamily
        dist_ = dist if not self._is_mixture else dist.component_distribution
        if self._params is None:
            self._params = {name: [] for name in dist_.arg_constraints.keys()}
            self.distribution_cls = dist_.__class__
        
        if self._is_mixture:
            self._mixtures.append(dist.mixture_distribution.probs)

        new_params = {name: getattr(dist_, name).cpu() for name in dist_.arg_constraints.keys()}
        for param_name, param in new_params.items():
            self._params[param_name].append(param)

    def get_concatenated_distributions(self, axis=0):
        """Returns a concatenated `Distribution` object along the specified axis."""
        concat_params = {key: torch.cat(value, dim=axis) for key, value in self._params.items()}
        if self._is_mixture:
            concat_mixture = torch.cat(self._mixtures, dim=axis)
            return MixtureSameFamily(
                mixture_distribution=Categorical(probs=concat_mixture),
                component_distribution=self.distribution_cls(**concat_params)
            )
        return self.distribution_cls(**concat_params)



def plot_losses(model):
    fig, ax = plt.subplots(2, 5, figsize=(18, 6), sharex=True, sharey='col')
    ax[0][0].set_title('Reconstruction')
    ax[0][1].set_title('KL*Weight')
    ax[0][2].set_title('KL')
    ax[0][3].set_title('Loss')
    ax[0][4].set_title('ELBO')

    n_epochs = len(model.history['validation_loss'])


    for i, eval_set in enumerate(('train', 'validation')):
        full_loss_key = {'validation': 'validation_loss', 'train': 'train_loss_epoch'}[eval_set]
        for j, loss_type in enumerate(('reconstruction_loss', 'kl*weight', 'kl_local', 'full_loss', 'elbo')):
            if loss_type == 'kl*weight':
                if 'kl_weight' in model.history:
                    kl_weight = model.history['kl_weight']['kl_weight']
                    key = f'kl_local_{eval_set}'
                    loss = model.history[key][key]*kl_weight
                else:
                    recon_key = f'reconstruction_loss_{eval_set}'
                    loss = model.history[full_loss_key][full_loss_key] - model.history[recon_key][recon_key]
            elif loss_type == 'full_loss':
                key = full_loss_key
                loss = model.history[key][key]
            else:
                key = f'{loss_type}_{eval_set}'
                loss = model.history[key][key]
            sns.lineplot(loss, ax=ax[i][j], color=sns.color_palette()[j])
            ax[i][j].set_ylabel(None)
    
        ax[i][0].set_ylabel(eval_set)

    return ax


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


def pca_neighbors_umap(adata, latent_name, neighbors_kwargs={}, umap_tl_kwargs={}, umap_pl_kwargs={}, umap_title=None,):
    calc_PCA(adata, rep=latent_name, key_added=latent_name)
    sc.pp.neighbors(adata, use_rep=f'{latent_name}_pca', key_added=latent_name, **neighbors_kwargs)
    sc.tl.umap(adata, neighbors_key=latent_name, **umap_tl_kwargs)
    fig = sc.pl.umap(adata, **umap_pl_kwargs, return_fig=True)
    if umap_title:
        fig.suptitle(umap_title)
    return fig


def add_one_hot_encoding_obsm(adata, obs_column, key_added=None):
    if not key_added:
        key_added = obs_column
    adata.obsm[obs_column] = pd.get_dummies(adata.obs[[obs_column]], dtype=int)


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


def plot_histograms(adata=None, keys=None, dfs=None, names=None, vars=None, hue=None, n_cols=8, kind='kde'):
    '''
    Plots feature histograms of an adata.
    Pass either adata with obsm keys, or list of dfs with same columns (and matching list of names)
    vars: features to plot.
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
