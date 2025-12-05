from collections.abc import Iterable, Sequence, Iterator
from typing import Callable, Literal, Optional, Union
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

from matplotlib import pyplot as plt
import seaborn as sns
import torch
import scanpy as sc
import scipy
from scipy.stats import norm

from scipy.sparse import csr_matrix, coo_array
from scipy.interpolate import griddata
from torch import tensor,float32
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score as NMI,
    adjusted_rand_score as ARI,
    calinski_harabasz_score as CH,
    davies_bouldin_score as DB,
    silhouette_score as SIL,
)
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from scvi_utils import get_rep, calc_PCA


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
        biological_key='cell_type',
        batch_key=None,
        compute_autocorrelation=False,
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
        self.biological_key=biological_key
        self.batch_key=batch_key
    
    def run(self):
        self._compute_negative_log_liklihood()
        self._compute_reconstructions()
        self._compute_squared_errors()
        self._compute_latents()
        if self.compute_autocorrelation:
            self._compute_autocorrelation()
        
        self._compute_scib_metrics()
        self._compute_latent_quality_metrics()
        self._compute_latent_comparison_metrics()
        
    
    def _match_dimensions(self, Xa, Xb):
        """Reduce higher-dim embedding with PCA to match the lower dimension."""
        da, db = Xa.shape[1], Xb.shape[1]
        if da == db:
            return Xa, Xb
        dmin = min(da, db)
        if da > dmin:
            Xa = PCA(n_components=dmin, random_state=0).fit_transform(Xa)
        if db > dmin:
            Xb = PCA(n_components=dmin, random_state=0).fit_transform(Xb)
        return Xa, Xb
    
    
    def _neighbor_rank_distance_pair(self,Xa,Xb,metric="euclidean"):
        """Mean symmetric NRD between Xa and Xb (dims already matched)."""
        n = Xa.shape[0]

        # A→B
        nn_b = NearestNeighbors(n_neighbors=n, metric=metric).fit(Xb)
        neigh_idx_ab = nn_b.kneighbors(Xa, return_distance=False)
        ranks_ab = [np.where(neigh_idx_ab[i] == i)[0][0] + 1 for i in range(n)]

        # B→A
        nn_a = NearestNeighbors(n_neighbors=n, metric=metric).fit(Xa)
        neigh_idx_ba = nn_a.kneighbors(Xb, return_distance=False)
        ranks_ba = [np.where(neigh_idx_ba[i] == i)[0][0] + 1 for i in range(n)]

        return ((np.array(ranks_ab) + np.array(ranks_ba)) / 2).mean()
    
    def _lisi_enrichment_score_pair(self, Xa, Xb, k=30, metric="euclidean"):
        """Mean 2-modality LISI enrichment score between Xa and Xb (dims already matched)."""
        X = np.vstack([Xa, Xb])
        labels = np.array(["A"] * Xa.shape[0] + ["B"] * Xb.shape[0])
        n = X.shape[0]

        nn = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(X)
        neigh_idx = nn.kneighbors(X, return_distance=False)[:, 1:]

        scores = [np.mean(labels[neigh_idx[i]] == labels[i]) for i in range(n)]
        mean_score = np.mean(scores)

        # expected baseline
        _, counts = np.unique(labels, return_counts=True)
        props = counts / counts.sum()
        expected = np.sum(props**2)

        return mean_score / expected

        
    def _compute_latent_comparison_metrics(self):
        latents = list(self.models.keys()) + [self.pca_key]
        index_dict = {f"latent{i+1}": name for i, name in enumerate(latents)}

        results = []
        for i, la in enumerate(latents):
            for j, lb in enumerate(latents):
                if j <= i:
                    continue
                Xa, Xb = self.adata.obsm[la], self.adata.obsm[lb]
                Xa, Xb = self._match_dimensions(Xa, Xb)

                nrd = self._neighbor_rank_distance_pair(Xa, Xb)
                lisi = self._lisi_enrichment_score_pair(Xa, Xb)

                results.append({
                    "latent_a": f"latent{i+1}",
                    "latent_b": f"latent{j+1}",
                    "NRD": nrd,
                    "LISI": lisi,
                })

        self.latent_comparison_results = pd.DataFrame(results)
        self.latent_index=index_dict
        
    def plot_latent_comparison_metrics(self):
        """
    Plot NRD and LISI results for latent pairs with short latent indices.
    """
        sns.set(style="whitegrid", context="talk")

        df_melt = self.latent_comparison_results.melt(
            id_vars=["latent_a", "latent_b"],
            value_vars=["NRD", "LISI"],
            var_name="Metric",
            value_name="Score"
        )
        df_melt["pair"] = df_melt["latent_a"] + " vs " + df_melt["latent_b"]

        metrics = df_melt["Metric"].unique()
        n_metrics = len(metrics)

        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5), sharey=False)

        if n_metrics == 1:
            axes = [axes]

        palette = sns.color_palette("Set2", len(df_melt["pair"].unique()))

        for i, metric in enumerate(metrics):
            ax = axes[i]
            sub = df_melt[df_melt["Metric"] == metric]
            sns.barplot(
            data=sub,
            x="pair", y="Score",
            hue="pair", dodge=False, legend=False,
            palette=palette,
            ax=ax
                )
            ax.set_title(metric)
            ax.set_xlabel("Latent Pair")
            ax.set_ylabel("Score")
            ax.tick_params(axis="x", rotation=30)

            # add numeric labels
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f}",
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha="center", va="bottom", fontsize=10, color="black", xytext=(0, 3),
                            textcoords="offset points")
        fig.tight_layout()
        return self.latent_index
        
    
    
    def _compute_latent_quality_metrics(self):
        latents = list(self.models.keys()) + [self.pca_key]
        label_key=self.biological_key
        n_clusters="auto"
        
        results = []

        y_true = self.adata.obs[label_key].astype("category")
        mask = ~y_true.isna()
        y_true_labeled = y_true[mask]

        if isinstance(n_clusters, str) and n_clusters == "auto":
            k = y_true_labeled.cat.categories.size
        else:
            k = int(n_clusters)

        for obsm_key in latents:
            X = self.adata.obsm[obsm_key][mask]

            km = KMeans(n_clusters=k, n_init="auto")
            y_pred = km.fit_predict(X)

            # scores
            nmi = NMI(y_true_labeled, y_pred)
            ari = ARI(y_true_labeled, y_pred)
            ch  = CH(X, y_pred)
            db  = DB(X, y_pred)
            sil = SIL(X, y_pred, metric="euclidean") if np.unique(y_pred).size > 1 else np.nan

            results.append({
                "latent": obsm_key,
                "NMI": nmi,
                "ARI": ari,
                "Calinski_Harabasz": ch,
                "Davies_Bouldin": db,
                "Silhouette": sil,
            })

        self.latent_quality_results = pd.DataFrame(results).set_index("latent")
        
    def plot_latent_quality_metrics(self):
        sns.set(style="whitegrid", context="talk")

        df_melt = self.latent_quality_results.reset_index().melt(id_vars="latent", 
                                                var_name="Metric", 
                                                value_name="Score")

        metrics = self.latent_quality_results.columns
        n_metrics = len(metrics)

        fig, axes = plt.subplots(1, n_metrics, figsize=(10*n_metrics, 10), sharey=False)

        if n_metrics == 1:
            axes = [axes]

        palette = sns.color_palette("Set2", len(self.latent_quality_results))

        for i, metric in enumerate(metrics):
            ax = axes[i]
            sns.barplot(
            data=df_melt[df_melt["Metric"] == metric],
            x="latent", y="Score",
            hue="latent",       # explicitly tie colors to latent
            dodge=False,        # don’t split bars
            legend=False,       # avoid duplicate legend
            palette=palette,
            ax=ax
        )
            ax.set_title(metric)
            ax.set_xlabel("Latent Embedding")
            ax.set_ylabel("Score")
            ax.tick_params(axis="x", rotation=30)
        
        fig.tight_layout()
        
        return self.latent_quality_results
        
    
    def _compute_scib_metrics(self):
        latents = list(self.models.keys()) + [self.pca_key]
        
        bm = Benchmarker(
        self.adata,
        label_key=self.biological_key,
        batch_key=self.batch_key,
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        embedding_obsm_keys=latents,
        
    )
                
        bm.benchmark()
        self.bm=bm
        
    def _compute_negative_log_liklihood(self):
        self.nll = {}

        # loop over trained models
        for model_name, model in self.models.items():
            recon = model.get_normalized_expression(
                return_l2_error=True, return_px_distrs=True,
                use_mean_latent=True, return_mean_expression=False
            )
            model_results = {}

            for mod_name, mod in self.adata_modalities.items():
                if mod_name in recon["distrs"].keys():
                    X = torch.tensor(np.asarray(mod), dtype=torch.float32)
                    distr = recon["distrs"][mod_name]
                    log_probs = distr.log_prob(X)
                    nll_matrix = -log_probs.detach().cpu().numpy()
                    model_results[mod_name] = nll_matrix.mean()
            self.nll[model_name] = model_results

        # ------------------------
        # compute baseline per modality
        # ------------------------
        baseline_results = {}
        for mod_name, mod in self.adata_modalities.items():
            X = np.asarray(mod)
            mu = X.mean(axis=0)
            var = X.var(axis=0) + 1e-6  # avoid zero variance
            std = np.sqrt(var)

            # logpdf under Gaussian baseline
            log_probs = norm(loc=mu, scale=std).logpdf(X)
            nll_matrix = -log_probs
            baseline_results[mod_name] = nll_matrix.mean()

        self.nll["baseline"] = baseline_results
                
            
    def plot_negative_likelihood(self):
    # Convert dict -> tidy DataFrame
        df = pd.DataFrame(self.nll).T  # rows = models (including "baseline"), cols = modalities
        df = df.reset_index().melt(id_vars="index", var_name="modality", value_name="NLL")
        df = df.rename(columns={"index": "model"})

        # Ensure baseline always comes last in plotting order
        model_order = [m for m in df["model"].unique() if m != "baseline"] + ["baseline"]

        modalities = df["modality"].unique()
        n_mods = len(modalities)

        sns.set(style="whitegrid", context="talk")
        fig, axes = plt.subplots(1, n_mods, figsize=(6 * n_mods, 5), sharey=False)
        if n_mods == 1:
            axes = [axes]

        for i, mod in enumerate(modalities):
            ax = axes[i]
            sub = df[df["modality"] == mod]

            # Differentiate baseline with a distinct color
            palette = {
                m: "gray" if m == "baseline" else c
                for m, c in zip(model_order, sns.color_palette("Set2", len(model_order)))
            }

            sns.barplot(
                data=sub,
                x="model", y="NLL",
                hue="model", dodge=False, legend=False,
                order=model_order,
                palette=palette, ax=ax
            )

            ax.set_title(f"Modality: {mod}")
            ax.set_xlabel("Model")
            ax.set_ylabel("Global NLL")
            ax.tick_params(axis="x", rotation=30)

            # annotate bar values
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.3f}",
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha="center", va="bottom", fontsize=10, color="black", xytext=(0, 3),
                            textcoords="offset points")

        fig.tight_layout()

        
        
        
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
        
    def plot_scib_metrics(self):
        return self.bm.plot_results_table()
        
    
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
        n_mods = len(self.modalities)
        fig, axes = plt.subplots(1, len(self.modalities), figsize=(8*len(self.modalities), 8))
        if n_mods == 1:
            axes = [axes]
        return fig, axes
    
    def _autocorr_barplot_figure(self,):
        fig, axes = plt.subplots(1, len(self.autocorr_keys), figsize=(8*len(self.autocorr_keys), 8))
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