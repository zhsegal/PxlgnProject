from collections.abc import Iterable, Sequence, Iterator
from typing import Callable, Literal, Optional, Union

from matplotlib import pyplot as plt
import seaborn as sns

import scanpy as sc
import scipy
from scipy.sparse import csr_matrix, coo_array
from scipy.interpolate import griddata

import pandas as pd
import numpy as np

from PixelGen.scvi_utils import get_rep, calc_PCA


def distr_autocorrelation_in_latent(adata, latent_keys, names, rep_key, vars=None, pca_kwargs={}, neighbors_kwargs={}):
    '''
    latent_keys: list of keys in adata.obsm
    names: name for each latent (for dataframe result)
    rep_key: representation (layer or obsm key) for which to compute autocorrelation in the latent
    vars: list of vars to include in computation in rep_key
    '''
    ret = []

    for latent_key in latent_keys:
        calc_PCA(adata, rep=latent_key, key_added=latent_key, **pca_kwargs)
        sc.pp.neighbors(adata, use_rep=f'{latent_key}_pca', key_added=latent_key, **neighbors_kwargs)

    for (latent_key, name) in zip(latent_keys, names):

        # Need to work around the fact that use_graph is not supported in anndata
        s_conn = 'connectivities'
        neighbors_conn_key = f'{latent_key}_{s_conn}'
        requested_conn = adata.obsp[neighbors_conn_key]
        existing_conn = adata.obsp.get(s_conn)
        adata.obsp[s_conn] = requested_conn

        rep = get_rep(adata, rep_key)
        if vars is None:
            vars = rep.columns.values
        rep = rep.loc[:, vars]
        rep = rep.to_numpy().T

        autocorr = pd.DataFrame(
            {
                'morans': sc.metrics.morans_i(adata, vals=rep),
                'gearys': sc.metrics.gearys_c(adata, vals=rep),
                'latent': name,
            },
            index=vars,
        )

        if existing_conn is not None:
            adata.obsp[s_conn] = existing_conn
        
        ret.append(autocorr)

    return pd.concat(ret, axis=0)



class MultiModalVIMetrics:
    MEAN_BASELINE = 'mean_baseline'
    GROUND_TRUTH = 'ground_truth'

    def __init__(
        self,
        adata,
        models : dict,
        mean_baseline=True,
        pca_key='X_pca',
        compute_autocorrelation=True,
        additional_autocorr_keys=[]
    ):
        self.adata = adata.copy()
        self.models = models
        # self.separated_features = separated_features
        self.mean_baseline = mean_baseline
        self.adata_modalities = self._adata_modalities()
        self.modalities = list(self.adata_modalities.keys())
        print(f'Modalities indentified: {self.modalities}')
        self.pca_key = pca_key
        self.compute_autocorrelation = compute_autocorrelation
        self.all_model_keys = sorted(
            list(self.models.keys()) + [pca_key] + [self.MEAN_BASELINE]
        )
        self.palette = dict(zip(self.all_model_keys, sns.color_palette()[:len(self.all_model_keys)]))
        self.additional_autocorr_keys = additional_autocorr_keys
        self.adata_distrs = {
            **self.adata_modalities,
            **{k: self.adata.obsm[k] for k in additional_autocorr_keys},
        }
        self.autocorr_keys = sorted(list(self.adata_distrs.keys()))
    
    def run(self):
        self._compute_reconstructions()
        self._compute_squared_errors()
        self._compute_latents()
        if self.compute_autocorrelation:
            self._compute_autocorrelation()
    
    def _compute_reconstructions(self):
        self.reconstructions = {}
        self.reconstruction_means = {}
        for model_name, model in self.models.items():
            recon = model.get_normalized_expression(return_l2_error=True, return_px_distrs=True, 
                                                    use_mean_latent=True, return_mean_expression=False)
            recon_means = model.get_normalized_expression(return_l2_error=True, return_px_distrs=True, 
                                                    use_mean_latent=True, return_mean_expression=True)
            self.reconstructions[model_name] = recon
            self.reconstruction_means[model_name] = recon_means
        if self.mean_baseline:
            self.reconstructions[self.MEAN_BASELINE] = {'exprs': {}, 'errors': {}}
            for mod_name, mod in self.adata_modalities.items():
                mod_mean = mod.copy()
                mod_mean[:] = mod.mean(axis=0)
                self.reconstructions[self.MEAN_BASELINE]['exprs'][mod_name] = mod_mean
                self.reconstructions[self.MEAN_BASELINE]['errors'][mod_name] = np.square(mod_mean - mod)
            self.reconstruction_means[self.MEAN_BASELINE] = self.reconstructions[self.MEAN_BASELINE].copy()
        
        
    def _compute_squared_errors(self):
        empty_dict = lambda : {model_name: {} for model_name in self.reconstructions.keys()}
        self.errors_per_feature = empty_dict()
        self.errors_per_feature_means = empty_dict()
        self.mean_modality_errors = empty_dict()
        self.mean_modality_errors_means = empty_dict()

        for model_name, recon in self.reconstructions.items():
            recon_means = self.reconstruction_means[model_name]
            recon_errors = recon['errors']
            recon_means_errors = recon_means['errors']
            for errors, feature_errors, mean_modality_errors in zip(
                (recon_errors, recon_means_errors), 
                (self.errors_per_feature, self.errors_per_feature_means), 
                (self.mean_modality_errors, self.mean_modality_errors_means),
            ):
                for mod, mod_errors in errors.items():
                    feature_errors[model_name][mod] = mod_errors.mean(axis=0)
                    mean_modality_errors[model_name][mod] = mod_errors.mean(axis=0).mean()

    def _compute_latents(self,):
        for model_name, model in self.models.items():
            self.adata.obsm[model_name] = model.get_latent_representation()
        if self.pca_key not in self.adata.obsm:
            raise RuntimeError('The provided pca_key is not in adata.obsm')
    
    def _compute_autocorrelation(self,):
        latents = list(self.models.keys()) + [self.pca_key]
        self.autocorr = {}
        for mod in self.autocorr_keys:
            self.autocorr[mod] = distr_autocorrelation_in_latent(
                adata=self.adata, 
                latent_keys=latents,
                names=latents,
                rep_key=mod,
            )
        

    def mean_modality_errors_barplot(self, model_names=None, reconstruction_mean=False, skip_mean_baseline=False,):
        model_names = self._model_names_pp(model_names, include_mean_baseline= not skip_mean_baseline)
        errs = self.mean_modality_errors if not reconstruction_mean else self.mean_modality_errors_means
        errs = pd.DataFrame(errs)[model_names].melt(
            var_name='model_name', value_name='error', ignore_index=False).rename_axis('modality').reset_index()
        fig, axes = self._modalities_barplot_figure()
        for i, modality in enumerate(self.modalities):
            errs_mod = errs[errs['modality'] == modality]
            sns.barplot(errs_mod, y='error', hue='model_name', ax=axes[i], palette=self.palette,
                        legend=i == 0)
            if i == 0:
                self._move_legend(axes[i])
            axes[i].set_title(modality)
        fig.suptitle(f'Squared Error - Modality mean{"" if not reconstruction_mean else " (recon means)"}')
        fig.tight_layout()
        return fig
                
    
    def errors_barplot(self, modality, features=None, 
                       model_names=None, reconstruction_mean=False, auto_filter_features=None,
                       skip_mean_baseline=False,
                       ):
        '''
        None = all
        auto_filter_features: integer, show top k features with highest variance in errors between models. If None show all
        '''
        model_names = self._model_names_pp(model_names, include_mean_baseline=not skip_mean_baseline)
        features = self._features_pp(key=modality, features=features)
        fig, ax = self._feature_barplot_figure(key=modality, 
            n_features=len(features) if auto_filter_features is None else auto_filter_features, 
            n_models=len(model_names)
        )
    
        all_errs = {}
        errors_per_feature = self.errors_per_feature if not reconstruction_mean else self.errors_per_feature_means
        for model_name, errors in errors_per_feature.items():
            if model_name not in model_names:
                continue
            mod_errors = errors.get(modality)
            if mod_errors is not None:
                mod_errors = mod_errors[features]
                all_errs[model_name] = mod_errors
    
        # index: features, columns: models
        all_errs = pd.DataFrame(all_errs)

        if skip_mean_baseline:
            errs_no_baseline = all_errs
        else:
            errs_no_baseline = all_errs.drop(columns=self.MEAN_BASELINE)
    
        if auto_filter_features:
            top_features = errs_no_baseline.var(axis=1).nlargest(auto_filter_features).index
            all_errs = all_errs.loc[top_features]

        all_errs = pd.DataFrame(all_errs).rename_axis('feature').reset_index().melt(id_vars='feature', var_name='model_name', value_name='error')

        sns.barplot(all_errs, ax=ax, x='feature', y='error', hue='model_name', palette=self.palette)
        ax.tick_params(axis='x', labelrotation=45)
        self._move_legend(ax)
        ax.set_title(f'Squared reconstruction error - {modality}{"" if not reconstruction_mean else " (recon means)"}')
        return fig
    
    def autocorr_barplot(self, autocorr_key, features=None, model_names=None, auto_filter_features=None, figsize=None):
        '''
        None = all
        auto_filter_features: integer, show top k features with highest variance in autocorr between models. If None show all
        '''
        model_names = self._model_names_pp(model_names=model_names, include_pca=True)
        features = self._features_pp(key=autocorr_key, features=features)

        fig, ax = self._feature_barplot_figure(
            key=autocorr_key, n_features=len(features) if auto_filter_features is None else auto_filter_features,
              n_models=len(model_names), figsize=figsize
        )
        autocorr = self.autocorr[autocorr_key].rename_axis('features').reset_index()
        autocorr = autocorr[autocorr['latent'].isin(model_names)]
        autocorr = autocorr[autocorr['features'].isin(features)]
        if auto_filter_features is not None:
            autocorr_by_model = pd.pivot_table(autocorr, values='morans', index='latent', columns='features')
            top_features = autocorr_by_model.var(axis=0).nlargest(auto_filter_features).index
            autocorr = autocorr[autocorr['features'].isin(top_features)]
    
        sns.barplot(autocorr, ax=ax, x='features', y='morans', hue='latent', palette=self.palette)
        ax.tick_params(axis='x', labelrotation=45)
        self._move_legend(ax)
        ax.set_title(f'Autocorrelation of {autocorr_key} in latents')

        fig.tight_layout()
        
        return fig
    
    def mean_autocorr_barplot(self, model_names=None):
        '''None = all'''
        model_names = self._model_names_pp(model_names=model_names, include_pca=True)
        fig, axes = self._autocorr_barplot_figure()
        for i, (mod, autocorr) in enumerate(self.autocorr.items()):
            autocorr = autocorr[autocorr['latent'].isin(model_names)]
            autocorr_by_model = autocorr.groupby('latent')['morans'].mean().reset_index()    
            sns.barplot(autocorr_by_model, y='morans', hue='latent', ax=axes[i], legend = i == 0, palette=self.palette)
            if i == 0:
                self._move_legend(axes[i])
            axes[i].set_title(mod)
        fig.suptitle('Autocorrelation of features in latents')
        fig.tight_layout()
        return fig
    
    def feature_histplot(self, modality, features, model_names=None, hue=None, residuals=False, reconstruction_mean=False):
        '''
        model_names: None = all
        hue: pass a feature in adata.obs for hue in the histplot
        residuals: if True, plot residuals in relation to ground truth (instead of features themselves)
        '''
        model_names = self._model_names_pp(model_names=model_names, include_ground_truth=True)
        features = self._features_pp(key=modality, features=features)
    
        fig, axes = plt.subplots(len(features), len(model_names), figsize=(len(model_names)*3, len(features)*3), 
                                 sharex='row', squeeze=False)
        for i, feature in enumerate(features):
            for j, model_name in enumerate(model_names):
                ax = axes[i,j]
                ax.set_title(model_name)
                if model_name == self.GROUND_TRUTH:
                    recon = pd.DataFrame(self.adata_modalities[modality][feature])
                else:
                    try:
                        if not reconstruction_mean:
                            recon = pd.DataFrame(self.reconstructions[model_name]['exprs'][modality][feature])
                        else:
                            recon = pd.DataFrame(self.reconstruction_means[model_name]['exprs'][modality][feature])
                        if residuals:
                            recon = recon - pd.DataFrame(self.adata_modalities[modality][feature])
                    except KeyError:
                        continue
                if hue is not None:
                    recon[hue] = self.adata.obs[hue]
                sns.histplot(recon, x=feature, stat='density', hue=hue, ax=ax)
        fig.suptitle(f'{modality} Feature Histograms{"" if not residuals else " - Residuals"}{"" if not reconstruction_mean else " (recon means)"}')
        fig.tight_layout()
        return fig
    
    def top_autocorr_features_barplot(self, key, top=10, model_names=None):
        model_names = self._model_names_pp(model_names=model_names, include_pca=True)
        fig, axes = plt.subplots(len(model_names), 1, figsize=(6, len(model_names)*3))
        for i, model_name in enumerate(model_names):
            autocorr = self.autocorr[key]
            autocorr = autocorr[autocorr['latent'] == model_name].rename_axis('features')
            top_features = autocorr['morans'].nlargest(top).index
            autocorr = autocorr.loc[top_features].reset_index()
            sns.barplot(autocorr, x='features', y='morans', ax=axes[i])
            axes[i].annotate(text=model_name, xy=(0, 0.5), xytext=(-50, 0), va='center', ha='right', 
                            xycoords='axes fraction',
                            textcoords='offset points',
                        )
            axes[i].tick_params(axis='x', labelrotation=45)

        fig.suptitle(f'Top autocorrelating features of {key} in latents')
        fig.tight_layout()
        return fig



    def _adata_modalities(self):
        modalities = {}
        for model in self.models.values():
            modalities.update(model.get_modality_df_dict())
        return modalities
    
    def _model_names_pp(self, model_names, include_pca=False, include_mean_baseline=False, include_ground_truth=False):
        if model_names is None:
            model_names = sorted(list(self.models.keys()))
        if include_pca:
            model_names += [self.pca_key]
        if include_mean_baseline:
            model_names += [self.MEAN_BASELINE]
        if include_ground_truth:
            model_names += [self.GROUND_TRUTH]
        return model_names
    
    def _features_pp(self, key, features):
        if features is None:
            features = list(self.adata_distrs[key].columns)
        features = sorted(features)
        return features
    
    def _feature_barplot_figure(self, key, n_features, n_models, figsize=None):
        if n_features is None:
            n_features = len(self.adata_distrs[key].columns)
        if n_features > 100:
            raise RuntimeError('Cannot plot bars for more than 100 features')
        if figsize is None:
            figsize = (3 + n_features * 8 / 30 * n_models / 4,4)
            if figsize[0] < 5:
                figsize = (5,4)
        fig, ax = plt.subplots(1, figsize=figsize)
        return fig, ax
    
    def _modalities_barplot_figure(self,):
        fig, axes = plt.subplots(1, len(self.modalities), figsize=(4*len(self.modalities), 4))
        return fig, axes
    
    def _autocorr_barplot_figure(self,):
        fig, axes = plt.subplots(1, len(self.autocorr_keys), figsize=(4*len(self.autocorr_keys), 4))
        return fig, axes
    
    def _move_legend(self, ax):
        ax.legend(loc='center right', bbox_to_anchor=(0, 0.5), borderaxespad=5)




        


# Old function, computes correlation based on random pairs (correlation between Euclidean distance in latent and the target feature)

# def latent_to_distr_correlation(adata, latent_key_lst, names, distr_key, vars=None, n_pcs_lst=None, num_pairs=1000, seed=0):
#     '''
#     Pass n_pcs to select a number of PCs if latent_key is a PCA representation
#     '''
#     rng = np.random.default_rng(seed=seed)
#     obs_pairs = [rng.choice(len(adata.obs), size=(2,)) for _ in range(num_pairs)]

#     if n_pcs_lst is None:
#         n_pcs_lst = [None]*len(names)

#     ret = []

#     for (latent_key, n_pcs, name) in zip(latent_key_lst, n_pcs_lst, names):

#         distr = adata.obsm[distr_key]
#         latent_dists = []
#         distr_dists = []
#         latent = np.array(adata.obsm[latent_key])
#         if n_pcs is not None:
#             latent = latent[:, :n_pcs]
#         if vars is None:
#             vars = distr.columns.values
#         distr = distr.loc[:, vars]

#         distr = np.array(distr)

#         for obs1, obs2 in obs_pairs:
#             latent_dists.append(np.linalg.norm(latent[obs1, :] - latent[obs2, :]))
#             distr_dists.append(np.linalg.norm(distr[obs1, :] - distr[obs2, :]))
        
#         ret.append(
#             {
#                 'Latent': name,
#                 'Corr': scipy.stats.pearsonr(latent_dists, distr_dists)[0],
#             }
#         )
    
#     return pd.DataFrame(ret)