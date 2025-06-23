import seaborn as sns
import pandas as pd
import anndata
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.distributions import MixtureSameFamily, Categorical
import scanpy as sc

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

    # n_epochs = len(model.history['validation_loss'])
    # validation_losses = ['reconstruction_loss_validation', 'kl_local_validation']
    # training_losses = ['reconstruction_loss_train', 'kl_local_train']

    # validation_df = pd.melt(pd.DataFrame({l: model.history[l][l] for l in validation_losses}).reset_index(), id_vars=['epoch'], value_name='Loss', var_name='Loss Type')
    # train_df = pd.melt(pd.DataFrame({l: model.history[l][l] for l in training_losses}).reset_index(), id_vars=['epoch'], value_name='Loss', var_name='Loss Type')
    # pd.set_option("future.no_silent_downcasting", True)
    # validation_df.replace({'reconstruction_loss_validation': 'Reconstruction', 'kl_local_validation': 'KL'}, inplace=True)
    # train_df.replace({'reconstruction_loss_train': 'Reconstruction', 'kl_local_train': 'KL'}, inplace=True)
    fig, ax = plt.subplots(2, 5, figsize=(18, 6), sharex=True, sharey='col')
    ax[0][0].set_title('Reconstruction')
    ax[0][1].set_title('KL*Weight')
    ax[0][2].set_title('KL')
    ax[0][3].set_title('Loss')
    ax[0][4].set_title('ELBO')

    n_epochs = len(model.history['validation_loss'])

    # if 'kl_weight' in model.history:
    #     kl_weight = model.history['kl_weight']
    # else:
    #     if kl_warmup == 0:
    #         kl_weight = torch.ones(shape=(n_epochs,))
    #     else:
    #         kl_weight = np.arange(start=0)

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

    # sns.lineplot(validation_df, x='epoch', y='Loss', hue='Loss Type', ax=ax[0])
    # sns.lineplot(train_df, x='epoch', y='Loss', hue='Loss Type', ax=ax[1])
    # ax[0].set_title('Validation')
    # ax[1].set_title('Train')
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