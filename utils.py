from typing import Literal

import itertools
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from sklearn.preprocessing import QuantileTransformer
import scanpy as sc
import anndata as ad
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from scipy.stats import median_abs_deviation


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import gridspec
import warnings

# Suppress warnings that might arise during statistical calculations on sparse data
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from community import community_louvain
from IPython.display import display

import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from community import community_louvain

import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from community import community_louvain

def run_marker_community_analysis(adata, coloc_key=None, threshold=None, resolution=1.0):
    # 1. Select Colocalization Data
    if not coloc_key:
        priority = ['spatial_asinh5_top500_var', 'HOTSPOT_top500_var', 'HOTSPOT']
        coloc_key = next((k for k in priority if k in adata.obsm), None)
    
    if not coloc_key:
        print("No colocalization data found in adata.obsm."); return
        
    coloc_df = pd.DataFrame(adata.obsm[coloc_key], index=adata.obs_names)
    pair_cols = [c for c in coloc_df.columns if '/' in str(c)]
    
    # 2. Build Graph 
    G = nx.Graph()
    edge_weights = coloc_df[pair_cols].abs().mean()
    
    if threshold is None:
        p75 = np.percentile(edge_weights, 75)
        threshold = max(p75, 0.01)
    
    for pair, weight in edge_weights.items():
        if weight >= threshold:
            m1, m2 = pair.split('/')
            if m1 != m2:
                if G.has_edge(m1, m2):
                    G[m1][m2]['weight'] = (G[m1][m2]['weight'] + weight) / 2
                else:
                    G.add_edge(m1, m2, weight=weight)
    
    if G.number_of_nodes() == 0:
        print(f"No edges found at threshold {threshold:.4f}."); return

    # 3. Community Detection (Louvain)
    partition = community_louvain.best_partition(G, resolution=resolution, random_state=42)
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    # 4. Print Communities (No interpretation)
    print(f"Using: {coloc_key} | Threshold: {threshold:.4f}")
    print(f"\n{'='*40}\nMARKER COMMUNITIES FOUND\n{'='*40}")
    for comm_id, markers in sorted(communities.items(), key=lambda x: -len(x[1])):
        if len(markers) >= 2:
            print(f"Community {comm_id} ({len(markers)} markers): {', '.join(sorted(markers))}")

    # 5. Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 20))
    
    # Network Plot
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=weights, edge_color='gray', ax=ax1)
    nx.draw_networkx_nodes(G, pos, node_color=[partition[n] for n in G.nodes()], 
                           cmap=plt.cm.Set3, node_size=500, alpha=0.8, ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax1)
    ax1.set_title(f"Colocalization Network (Threshold: {threshold:.4f})"); ax1.axis('off')

    # Community Sorted Heatmap
    order = sorted(G.nodes(), key=lambda x: (partition[x], x))
    corr_matrix = pd.DataFrame(0.0, index=order, columns=order)
    for pair, weight in edge_weights.items():
        m1, m2 = pair.split('/')
        if m1 in order and m2 in order:
            corr_matrix.loc[m1, m2] = corr_matrix.loc[m2, m1] = weight
            
    sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, ax=ax2, square=True)
    ax2.set_title("Marker Correlation (Sorted by Community)")
    plt.tight_layout(); plt.show()

    return G, partition

# Run the technical analysis
# Run the technical analysis

# Execute



def run_pseudobulk_pipeline(adata, test_cond='PHA', ref_cond='unstim', groupby='sample'):
    # 1. Identify Data Modalities
    modalities = {'Abundance': {'layer': 'arcsinh' if 'arcsinh' in adata.layers else None}}
    if 'HOTSPOT_top500_var' in adata.obsm: modalities['Hotspot'] = {'obsm': 'HOTSPOT_top500_var'}
    if 'spatial_asinh5_top500_var' in adata.obsm: modalities['JoinCount'] = {'obsm': 'spatial_asinh5_top500_var'}

    for name, params in modalities.items():
        print(f"\n{'='*40}\nAnalyzing: {name}\n{'='*40}")
        
        # 2. Aggregate to Pseudobulk
        if 'obsm' in params:
            data = pd.DataFrame(adata.obsm[params['obsm']], index=adata.obs_names)
        else:
            X = adata.layers[params['layer']] if params['layer'] else adata.X
            data = pd.DataFrame(X.toarray() if hasattr(X, 'toarray') else X, index=adata.obs_names, columns=adata.var_names)
        
        pb = data.assign(group=adata.obs[groupby].astype(str)).groupby('group').mean()
        meta = adata.obs.groupby(groupby.astype(str) if hasattr(groupby, 'astype') else groupby)[['condition']].first()
        
        # 3. Differential Expression (t-test)
        t_idx, r_idx = meta[meta.condition == test_cond].index, meta[meta.condition == ref_cond].index
        if len(t_idx) < 2 or len(r_idx) < 2: continue

        results = []
        for feat in pb.columns:
            t_val, r_val = pb.loc[t_idx, feat], pb.loc[r_idx, feat]
            pval = stats.ttest_ind(t_val, r_val, equal_var=False, nan_policy='omit').pvalue
            
            # Percent expressing in test group for dot sizing
            mask = adata.obs[groupby].isin(t_idx)
            pct = (data.loc[mask, feat] > 0).mean() * 100
            
            results.append({'feature': feat, 'logFC': np.mean(t_val) - np.mean(r_val), 'pval': pval or 1.0, 
                            f'mean_{test_cond}': np.mean(t_val), f'mean_{ref_cond}': np.mean(r_val), 'pct': pct})
        
        res = pd.DataFrame(results).assign(pval_adj=lambda x: multipletests(x.pval, method='fdr_bh')[1])
        res['nlog10'] = -np.log10(res.pval_adj + 1e-300)
        res['dir'] = np.where(res.pval_adj > 0.05, 'NS', np.where(res.logFC > 0, f'Up in {test_cond}', f'Up in {ref_cond}'))

        # 4. Volcano Plot with Sizing and 20 Labels (10 Up, 10 Down)
        plt.figure(figsize=(10, 7))
        pal = {f'Up in {test_cond}': '#d62728', f'Up in {ref_cond}': '#1f77b4', 'NS': 'lightgrey'}
        sns.scatterplot(data=res, x='logFC', y='nlog10', hue='dir', palette=pal, size='pct', sizes=(20, 200), alpha=0.6)
        
        # Labeling logic
        top_up = res[res.dir == f'Up in {test_cond}'].nlargest(10, 'logFC')
        top_down = res[res.dir == f'Up in {ref_cond}'].nsmallest(10, 'logFC')
        for _, r in pd.concat([top_up, top_down]).iterrows():
            plt.text(r['logFC'], r['nlog10'], r['feature'], fontsize=8, weight='bold', alpha=0.8)
        
        plt.title(f'{name}: {test_cond} vs {ref_cond}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Direction / % Expressing')
        sns.despine(); plt.show()

        # 5. Heatmap (Z-scored Significant Features)
        sig_hits = res[res.pval_adj < 0.05].nsmallest(25, 'pval_adj')
        if not sig_hits.empty:
            scaled_pb = pd.DataFrame(StandardScaler().fit_transform(pb[sig_hits.feature]), index=pb.index, columns=sig_hits.feature)
            sns.clustermap(scaled_pb.T, col_colors=meta.condition.map({test_cond: '#d62728', ref_cond: '#1f77b4'}), 
                          cmap='vlag', center=0, figsize=(8, 6), method='ward')
            plt.show()
        
        # 6. Top 10 Significant Table per Condition (20 Total)
        print(f"\nTop 10 Significant Features per Condition ({name}):")
        top_test_table = res[res.dir == f'Up in {test_cond}'].nsmallest(10, 'pval_adj')
        top_ref_table = res[res.dir == f'Up in {ref_cond}'].nsmallest(10, 'pval_adj')
        display(pd.concat([top_test_table, top_ref_table])[['feature', 'logFC', 'pval_adj', 'dir']])






pbmc_surface_markers = {
    "T cells": ["CD3e", "CD2", "CD5", "CD7", "TCRab"],
    # "CD4 T": ["CD4", "CD45RA", "CD45RO", "CD27", "CD28"],
    "CD8 T": ["CD8", "CD57", "KLRG1", "CX3CR1"],
    # "Gamma-delta T": ["TCRgd", "TCRVg9", "TCRVd2"],
    # "NK cells": ["CD56", "CD94", "CD335", "NKp80", "CD314"],
    # "B cells": ["CD19", "CD20", "CD22", "CD79a", "CD37"],
    # "Plasma cells": ["CD138", "CD38", "CD319"],
    # "Monocytes": ["CD14", "CD16", "CD11b", "CD33", "CD64"],
    # "DCs": ["CD11c", "HLA-DR", "CD86", "CD80"],
    # "Platelets": ["CD41", "CD62P", "CD9", "CD36"],
}
# all_known_markers = set(m for markers in pbmc_surface_markers.values() for m in markers)



ALL_MARKERS = {m for markers in pbmc_surface_markers.values() for m in markers}
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

def run_de_analysis(adata, condition='condition', test_group='PHA', ref_group='unstim',
                    cell_filter=None, layer='arcsinh', n_label=10, display_name_fn=None):
    
    # 1. Filter and Run DE Stats
    if cell_filter: 
        adata = adata[adata.obs.eval(cell_filter)].copy()
    
    print(f"Running Wilcoxon for {test_group} vs {ref_group}...")
    sc.tl.rank_genes_groups(adata, groupby=condition, groups=[test_group], reference=ref_group, method='wilcoxon', layer=layer)
    df = sc.get.rank_genes_groups_df(adata, group=test_group).copy()
    
    # 2. Metrics
    df['nlog10'] = -np.log10(df['pvals_adj'] + 1e-300)
    df['color'] = np.where((df.pvals_adj < 0.05) & (df.logfoldchanges > 0.5), f'Up in {test_group}',
                  np.where((df.pvals_adj < 0.05) & (df.logfoldchanges < -0.5), f'Up in {ref_group}', 'NS'))
    
    # Prevalence calculation
    mask = adata.obs[condition] == test_group
    data = adata[mask].layers[layer] if layer in adata.layers else adata[mask].X
    if hasattr(data, "toarray"): data = data.toarray()
    
    pct_map = dict(zip(adata.var_names, (data > 0).mean(axis=0) * 100))
    df['pct'] = df['names'].map(pct_map)

    # 3. Visualization: Volcano Plot — condition-consistent colors, larger fonts
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='logfoldchanges', y='nlog10', hue='color', size='pct',
                    palette={f'Up in {test_group}':'#55a868', f'Up in {ref_group}':'#4c72b0', 'NS':'lightgrey'},
                    sizes=(20, 200), alpha=0.6)

    # Label top hits on plot with display names
    top_hits = pd.concat([df.nlargest(n_label, 'logfoldchanges'), df.nsmallest(n_label, 'logfoldchanges')])
    _dn = display_name_fn if display_name_fn is not None else (lambda x: x)
    for _, r in top_hits.iterrows():
        plt.text(r['logfoldchanges'], r['nlog10'], _dn(r['names']), fontsize=11, weight='bold')

    plt.xlabel('log fold change', fontsize=14)
    plt.ylabel('-log10 adjusted p-value', fontsize=14)
    plt.title(f'DE: {test_group} vs {ref_group}', fontsize=15)
    plt.tick_params(labelsize=13)
    sns.despine(); plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    # 4. Visualization: Heatmap
    marker_list = [m for markers in pbmc_surface_markers.values() for m in markers if m in adata.var_names]
    if len(marker_list) < 5: marker_list = top_hits['names'].tolist()

    sc.pl.heatmap(adata, var_names=marker_list, groupby=condition, 
                  layer=layer, standard_scale='var', cmap='RdBu_r', 
                  swap_axes=True, show=True)

    # 5. PRINTED TABLES (Modified to show Top 10 Up and Top 10 Down)
    top_up = df[df['logfoldchanges'] > 0].nlargest(10, 'logfoldchanges')
    top_down = df[df['logfoldchanges'] < 0].nsmallest(10, 'logfoldchanges')
    
    print(f"\n{'='*40}\nTOP 10 UPREGULATED IN {test_group} (PHA)\n{'='*40}")
    display(top_up[['names', 'logfoldchanges', 'pvals_adj', 'pct']].style.background_gradient(cmap='Reds', subset=['logfoldchanges']))
    
    print(f"\n{'='*40}\nTOP 10 UPREGULATED IN {ref_group} (Unstim)\n{'='*40}")
    display(top_down[['names', 'logfoldchanges', 'pvals_adj', 'pct']].style.background_gradient(cmap='Blues_r', subset=['logfoldchanges']))
    
    return df

# Run it

def robust_cv(x, axis, epsilon=0.1):
    """
    Median–IQR coefficient of variation with a robust stabilizer.
    
    The stabilizer (epsilon) prevents division by zero when the median is 0,
    which is common for sparse, zero-centered biological data (Features CV).

    Parameters
    ----------
    x : np.ndarray
        The input expression data (dense, after nan_to_num).
    axis : int
        Axis over which to calculate the CV (0 for Features, 1 for Cells).
    epsilon : float
        The stabilizing constant added to the absolute median (default 0.1).
        
    Returns
    -------
    np.ndarray
        The calculated robust CV values.
    """
    
    # Ensure NaNs are handled (as in your original code)
    x = np.nan_to_num(x, nan=0.0)

    # 1. Calculate Median and Quartiles (IQR)
    # Use np.median and np.percentile since NaNs were already converted to 0.0
    med = np.median(x, axis=axis)
    q25 = np.percentile(x, 25, axis=axis)
    q75 = np.percentile(x, 75, axis=axis)
    iqr = q75 - q25

    # 2. Calculate CV with Robust Stabilizer
    # The absolute value of the median is used, and the larger epsilon (0.1) 
    # ensures stability even when the median is exactly 0.
    
    # Formula: IQR / (|Median| + epsilon)
    cv = iqr / (np.abs(med) + epsilon)
    
    # Clean up any potential extreme values after division (e.g., if IQR was also 0)
    cv = np.nan_to_num(cv, nan=0.0, posinf=0.0, neginf=0.0)

    return cv
def get_dense(data):
    """Converts sparse matrix to dense numpy array if necessary."""
    return data.toarray() if hasattr(data, "toarray") else np.array(data)

def calculate_metrics(obs, gen, model_name, modality_name):
    """Calculates CV metrics (Observed, Generated, Delta) and stats (MAE, R, Rho) for Features and Cells."""
    results = []
    
    for axis, level in [(0, "Features"), (1, "Cells")]:
        c_obs, c_gen = robust_cv(obs, axis), robust_cv(gen, axis)
        delta = c_gen - c_obs
        
        p, _ = stats.pearsonr(c_obs, c_gen)
        s, _ = stats.spearmanr(c_obs, c_gen)
        mae = np.mean(np.abs(delta))

        results.append({
            'Model': model_name,
            'Modality': modality_name,
            'Level': level, 
            'MAE': mae, 'Pearson': p, 'Spearman': s,
            'Observed CV': c_obs, 'Generated CV': c_gen, 'Delta CV': delta
        })
    return results

# --- 1. Load Data and Aggregate Metrics ---


# --- 2. Plotting Function (Generates the Composite Figure) ---

def plot_composite_ppc(modality_name, df, scatter_color, model_names):
    
    df_modality = df[df['Modality'] == modality_name]
    if df_modality.empty and modality_name == 'Spatial':
        print(f"\nSkipping {modality_name} plot: No models with this modality or data found.")
        return

    N_models = len(model_names)
    # N_cols is now N_models (scatters) + 1 (violin) + 1 (table)
    N_cols = N_models + 2 

    # Figure width increased to accommodate the table column
    fig = plt.figure(figsize=(4.5 * N_cols, 12)) 
    
    # Added extra column width ratio (0.8) for the table
    gs = gridspec.GridSpec(2, N_cols, 
                           height_ratios=[1, 1], width_ratios=[1] * N_models + [0.75, 0.8], left=0.05,
                           wspace=0.3, hspace=0.4, top=0.9) 
    
    table_data = [] 
    
    # Loop over Rows: Features (i=0) and Cells (i=1)
    for i, level in enumerate(["Features", "Cells"]):
        
        df_level = df_modality[df_modality['Level'] == level]
        violin_data = []
        
        # --- Scatter Plots (Columns 0 to N-1) ---
        for j, model_name in enumerate(model_names):
            ax_s = fig.add_subplot(gs[i, j])
            row = df_level[df_level['Model'] == model_name]

            if row.empty:
                # N/A placeholder for missing modality
                ax_s.text(0.5, 0.5, "N/A", fontsize=16, ha='center', va='center', color='gray')
                ax_s.set_xticks([]); ax_s.set_yticks([]); ax_s.set_frame_on(False)
                stats_row = ["N/A"] * 3
            else:
                c_obs = np.concatenate(row['Observed CV'].values)
                c_gen = np.concatenate(row['Generated CV'].values)
                delta = np.concatenate(row['Delta CV'].values)
                
                # Plot Scatter
                ax_s.scatter(c_obs, c_gen, c=scatter_color, s=15, alpha=0.5, ec='white', lw=0.5)
                
                # Robust Limits and Diagonal (99th percentile zoom)
                lim_max = np.percentile(np.concatenate([c_obs, c_gen]), 99) * 1.1
                ax_s.plot([0, lim_max], [0, lim_max], 'k--', lw=1.5)
                ax_s.set_xlim(0, lim_max); ax_s.set_ylim(0, lim_max)
                ax_s.set_aspect('equal')
                
                # Labels - Use full model_name in title
                ax_s.set_title(model_name, fontsize=10) 
                ax_s.set_xlabel("Observed CV" if i == 1 else "")
                ax_s.set_ylabel("Generated CV" if j == 0 else "")

                # Prepare for Violin plot
                df_temp = pd.DataFrame({'Delta CV': delta, 'Model': model_name})
                violin_data.append(df_temp)

                # Store Stats for Table
                stats_row = [f"{row['MAE'].iloc[0]:.2f}", f"{row['Pearson'].iloc[0]:.2f}", f"{row['Spearman'].iloc[0]:.2f}"]
            
            table_data.append(stats_row)

        # --- Violin Plot (Column N) ---
        ax_v = fig.add_subplot(gs[i, N_models])
        
        if violin_data:
            violin_df = pd.concat(violin_data)
            q_abs = np.percentile(np.abs(violin_df['Delta CV'].values), 98)
            violin_df['Delta CV'] = violin_df['Delta CV'].clip(-q_abs, q_abs)

            # Fixed FutureWarning/UserWarning
            sns.violinplot(data=violin_df, x='Model', y='Delta CV', hue='Model', ax=ax_v, 
                           palette='tab10', inner="quartile", cut=0, legend=False)
            
            ax_v.axhline(0, color='gray', ls=':')
            ax_v.set_title(f"{level} Error", fontsize=10)
            ax_v.set_xlabel("")
            ax_v.set_ylabel("Delta CV (Gen - Obs)" if N_models > 0 else "")
            
            ax_v.set_xticks(np.arange(len(model_names))) 
            ax_v.set_xticklabels(model_names, rotation=45, ha='right') 

        sns.despine(ax=ax_v)
        
        # --- Table (Column N+1) ---
        ax_table_col = fig.add_subplot(gs[i, N_models + 1])
        ax_table_col.axis('off')

        # Create a simple table for this row's metrics
        table_metrics_row = []
        for k in range(N_models):
            # If Features row (i=0), take data[k]. If Cells row (i=1), take data[N_models + k].
            metrics = table_data[k] if i == 0 else table_data[N_models + k]
            table_metrics_row.append(metrics)

        table_df = pd.DataFrame(table_metrics_row, index=model_names, 
                                columns=["MAE", "Pearson", "Spearman"])
        
        # Manually create header for the table column
        ax_table_col.text(0.5, 1.0, f'{level} Metrics', ha='center', va='bottom', fontsize=12, fontweight='bold')


        mpl_table = ax_table_col.table(cellText=table_df.values, rowLabels=table_df.index.tolist(), 
                                       colLabels=table_df.columns.tolist(), loc='center', 
                                       cellLoc='center', edges='open')
        
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(7) # Compressed font size
        mpl_table.scale(0.8, 1.2) # Compressed scale
        
    # --- Figure Title ---
    fig.text(0.05, 0.96, f"{modality_name} Posterior Predictive Checks", fontsize=16, fontweight='bold')
    plt.show()



def plot_latent(adata, ax, z, title, color='condition'):
    """Plot one latent onto one axes."""
    if z is None or z.shape[0] == 0:
        ax.text(0.5, 0.5, "no latent", ha="center")
        ax.set_title(title)
        ax.axis("off")
        return
    
    ad = sc.AnnData(X=z, obs=adata.obs.copy())
    sc.pp.neighbors(ad, n_neighbors=15, use_rep="X")
    sc.tl.umap(ad)
    sc.pl.umap(ad, color=[color], ax=ax,size=70, show=False)
    ax.set_title(title)

def plot_model_latents(models_dict, adata,color='condition'):
    model_items = list(models_dict.items())
    n = len(model_items)

    fig, axes = plt.subplots(n, 3, figsize=(30, 8*n))
    axes = np.atleast_2d(axes)
    
    for i, (model_name, model) in enumerate(model_items):

        try:
            mods = list(model.get_modality_df_dict().keys())
        except:
            mods = []

        # Panel 0: joint
        try:
            z_joint = model.get_latent_representation(adata)
        except:
            z_joint = None
        plot_latent(adata,axes[i, 0], z_joint, f"{model_name} – joint",color)

        # Panel 1: modality 0 (if exists)
        if len(mods) > 0:
            try:
                z0 = model.get_latent_representation(adata, modality=mods[0])
            except:
                z0 = None
            plot_latent(adata, axes[i, 1], z0, f"{model_name} – {mods[0]}",color)
        else:
            plot_latent(adata, axes[i, 1], None, f"{model_name} – no modality 0",color)

        # Panel 2: modality 1 (if exists)
        if len(mods) > 1:
            try:
                z1 = model.get_latent_representation(adata, modality=mods[1])
            except:
                z1 = None
            plot_latent(adata, axes[i, 2], z1, f"{model_name} – {mods[1]}",color)
        else:
            plot_latent(adata, axes[i, 2], None, f"{model_name} – no modality 1",color)

    
    plt.tight_layout()
    plt.show()



def mask_adata(adata_masked, spatial_key, mask_frac=0.1, asigned_values=[0.0,0.0]):

    n_cells = adata_masked.n_obs

    # --- reproducible RNG ---
    rng = np.random.default_rng(seed=0)
    mask_indices = rng.choice(n_cells, int(mask_frac * n_cells), replace=False)


    adata_masked.obs["spatial_masked"] = False
    adata_masked.obs.iloc[mask_indices, adata_masked.obs.columns.get_loc("spatial_masked")] = True

    X_spatial = adata_masked.obsm[spatial_key].to_numpy().copy()

    half = len(mask_indices) // 2
    mask_indices_low = mask_indices[:half]
    mask_indices_high = mask_indices[half:]

    # --- assign corruption values ---
    X_spatial[mask_indices_low, :] = asigned_values[0]       # assign value to first group
    X_spatial[mask_indices_high, :] = asigned_values[1]  # assign value to second group

    # --- write back as DataFrame (preserving structure) ---
    adata_masked.obsm[f'{spatial_key}_masked'] = pd.DataFrame(
        X_spatial,
        index=adata_masked.obs_names,
        columns=adata_masked.obsm[spatial_key].columns
    )

    # --- keep mask column as boolean for clarity ---
    adata_masked.obs["spatial_masked"] = adata_masked.obs["spatial_masked"].astype(bool)

    print(f"Masked {len(mask_indices)} cells ({mask_frac*100:.1f}%) — "
        f"{len(mask_indices_low)} set to 0.0, {len(mask_indices_high)} set to 1000.0")

def plot_gene_heatmap(adata, layer, grouping_obs):
    adata_tmp = ad.AnnData(
        X=adata.layers[layer],
        obs=adata.obs.copy(),
        var=adata.var.copy()
    )
    sc.tl.rank_genes_groups(adata_tmp,grouping_obs, method="wilcoxon",)


    diff_exp_df = sc.get.rank_genes_groups_df(adata_tmp, group=None)
    diff_exp_df["-log10(adjusted p-value)"] = -np.log10(diff_exp_df["pvals_adj"])
    diff_exp_df["Significant"] = diff_exp_df["pvals_adj"] < 0.01
    df = diff_exp_df.pivot(index=["names"], columns=["group"], values=["logfoldchanges"])

    markers_for_heatmap = set(
        diff_exp_df[
            (np.abs(diff_exp_df["logfoldchanges"]) > 3) & diff_exp_df["Significant"]
        ]["names"]
    )
    markers_to_add=[]

    markers_for_heatmap.update(markers_to_add)

    df = df[df.index.isin(markers_for_heatmap)]

    df.columns = [cluster for _, cluster in df.columns]
    fig = sns.clustermap(df, yticklabels=True, linewidths=0.1, cmap="vlag", vmin=-5, vmax=5);
    fig.fig.set_size_inches(10, 15)



def build_spatial_obsms_with_scaler(adata, spatial_df, value_col="join_count_z", tanh_scale=4.0):
    """
    Create spatial DataFrames in adata.obsm, including Raw, Tanh, Arcsinh, 
    Quantile, and the new Robust Scaled Arcsinh.
    
    The Robust Scaled Arcsinh transformation (spatial_asinh_robust) is calculated as:
    (arcsinh(data) - median(arcsinh(data))) / IQR(arcsinh(data))
    
    Args:
        adata (AnnData): The AnnData object to store results in .obsm.
        spatial_df (pd.DataFrame): Long-format DataFrame with colocalization scores.
        value_col (str): The column containing the Z-scores (e.g., 'join_count_z').
        tanh_scale (float): Scaling factor for the tanh transformation.
    """

    # --- 1. Pivot Long → Wide ---
    df = spatial_df.reset_index().copy()
    df["pair"] = df["marker_1"] + "/" + df["marker_2"]
    wide = (
        df.pivot_table(index="component", columns="pair", values=value_col, aggfunc="first")
          .reindex(adata.obs_names)
          .fillna(0.0)
    )

    base = wide.to_numpy(float)

    # --- 2. Standard Transforms ---
    spatial_raw = pd.DataFrame(base, index=wide.index, columns=wide.columns)
    
    spatial_tanh = pd.DataFrame(
        tanh_scale * np.tanh(base / tanh_scale),
        index=wide.index,
        columns=wide.columns
    )
    
    # Calculate Arcsinh base data
    asinh_base = np.arcsinh(base)
    spatial_asinh = pd.DataFrame(
        asinh_base,
        index=wide.index,
        columns=wide.columns
    )

    # Quantile Transformer
    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(1000, base.shape[1]),
        random_state=0
    )
    # Apply QuantileTransformer across features (columns)
    quant = qt.fit_transform(base)
    spatial_quantile = pd.DataFrame(
        quant,
        index=wide.index,
        columns=wide.columns
    )

    # --- 3. Robust Scaled Arcsinh Transform (The New, Stabilized Feature) ---
    
    # We apply RobustScaler to the existing asinh_base data.
    # RobustScaler removes the median and scales the data to the Interquartile Range (IQR).
    # This addresses the extreme clustering around zero observed in the original arcsinh plots.
    
    rs = RobustScaler(with_centering=True, with_scaling=True)
    
    # RobustScaler expects features (proteins/pairs) as columns.
    robust_scaled_data = rs.fit_transform(asinh_base)
    
    spatial_asinh_robust = pd.DataFrame(
        robust_scaled_data,
        index=wide.index,
        columns=wide.columns
    )

    # ---- 4. Assign to obsm ----
    adata.obsm["spatial_raw"]          = spatial_raw
    adata.obsm[f"spatial_tanh{int(tanh_scale)}"] = spatial_tanh
    adata.obsm["spatial_asinh"]        = spatial_asinh
    adata.obsm["spatial_quantile"]     = spatial_quantile
    adata.obsm["spatial_asinh_robust"] = spatial_asinh_robust # NEW KEY

    print("Created obsm keys:",
          "spatial_raw",
          f"spatial_tanh{int(tanh_scale)}",
          "spatial_asinh",
          "spatial_quantile",
          "spatial_asinh_robust")
    print("Shape:", base.shape)



def build_spatial_obsms(adata, spatial_df, value_col="join_count_z", tanh_scale=4.0):
    """Create spatial_{raw,tanh,asinh,quantile} DataFrames in adata.obsm."""

    # ---- pivot long → wide ----
    df = spatial_df.reset_index().copy()
    df["pair"] = df["marker_1"] + "/" + df["marker_2"]
    wide = (
        df.pivot_table(index="component", columns="pair", values=value_col, aggfunc="first")
          .reindex(adata.obs_names)
          .fillna(0.0)
    )

    base = wide.to_numpy(float)

    # ---- transforms ----
    spatial_raw = pd.DataFrame(base, index=wide.index, columns=wide.columns)
    spatial_tanh = pd.DataFrame(
        tanh_scale * np.tanh(base / tanh_scale),
        index=wide.index,
        columns=wide.columns
    )
    spatial_asinh = pd.DataFrame(
        np.arcsinh(base),
        index=wide.index,
        columns=wide.columns
    )

    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(1000, base.shape[1]),
        random_state=0
    )
    quant = np.vstack([qt.fit_transform(row[:, None]).ravel() for row in base])
    spatial_quantile = pd.DataFrame(
        quant,
        index=wide.index,
        columns=wide.columns
    )

    # ---- assign to obsm ----
    adata.obsm["spatial_raw"]      = spatial_raw
    adata.obsm[f"spatial_tanh{int(tanh_scale)}"] = spatial_tanh
    adata.obsm["spatial_asinh"]    = spatial_asinh
    adata.obsm["spatial_quantile"] = spatial_quantile

    print("Created obsm keys:",
          "spatial_raw",
          f"spatial_tanh{int(tanh_scale)}",
          "spatial_asinh",
          "spatial_quantile")
    print("Shape:", base.shape)


def build_spatial_pca_obsm(
    adata,
    source_key: str,
    n_components: int,
    target_key: str | None = None,
    standardize: Literal["none", "center", "zscore"] = "center",
    whiten: bool = False,
    random_state: int = 0,
):
    """PCA-reduce a spatial obsm modality and write the reduced version back to obsm.

    Use the returned ``target_key`` directly in
    ``MultiModalSCVI.setup_anndata(extra_modality_keys=[target_key])``.

    Parameters
    ----------
    adata
        AnnData object whose ``obsm[source_key]`` holds a (n_obs, n_features) DataFrame
        of spatial features (typically join-count statistics for protein pairs).
    source_key
        Key into ``adata.obsm`` for the spatial DataFrame to reduce.
    n_components
        Number of principal components to keep.
    target_key
        Output key in ``adata.obsm``. Defaults to ``f"{source_key}_pca{n_components}"``.
    standardize
        Pre-PCA standardization applied to the input matrix:
        - "center": no explicit step (sklearn's PCA centers internally).
        - "zscore": StandardScaler with mean and std before PCA.
        - "none": pass the matrix straight through.
    whiten
        Forwarded to sklearn ``PCA(whiten=...)``. When True, components are scaled to
        unit variance — useful so the encoder sees PCs on a comparable scale.
    random_state
        Forwarded to sklearn ``PCA``.

    Returns
    -------
    str
        The ``target_key`` written to ``adata.obsm`` (and used as the key for the
        accompanying entry in ``adata.uns``).
    """
    src = adata.obsm[source_key]
    if not isinstance(src, pd.DataFrame):
        raise TypeError(f"adata.obsm['{source_key}'] must be a DataFrame.")
    X = src.to_numpy(dtype=float, copy=True)

    scaler = None
    if standardize == "center":
        pass
    elif standardize == "zscore":
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(X)
    elif standardize != "none":
        raise ValueError("standardize must be one of {'none', 'center', 'zscore'}")

    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    Z = pca.fit_transform(X)

    if target_key is None:
        target_key = f"{source_key}_pca{n_components}"

    cols = [f"PC{i+1}" for i in range(n_components)]
    adata.obsm[target_key] = pd.DataFrame(Z, index=src.index, columns=cols)

    adata.uns[f"{target_key}_info"] = {
        "source_key": source_key,
        "standardize": standardize,
        "whiten": whiten,
        "n_components": n_components,
        "components": pca.components_,
        "mean": pca.mean_,
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "scaler_mean": None if scaler is None else scaler.mean_,
        "scaler_scale": None if scaler is None else scaler.scale_,
    }

    print(
        f"Created obsm key: {target_key}  shape={Z.shape}  "
        f"explained_variance_ratio_sum={pca.explained_variance_ratio_.sum():.4f}"
    )
    return target_key


def build_spatial_celltype_obsm(
    adata,
    celltype_key="cell_type_annot",
    score_key="spatial_raw",     # raw z used for thresholding/scoring
    value_key="spatial_asinh5",  # transformed values written into the modality
    z_thresh=2.04,
    top_x=100,
    use_abs=True,                # |z| > thresh; count pos + neg colocalization
    target_key=None,
):
    """Select spatial protein pairs per cell type by summed significant |z|, union across
    cell types, and write the ``value_key`` subset to ``obsm[target_key]``.

    For each pair, in each cell type, sum the pair's significant z-scores
    (``|z| > z_thresh``) over that type's cells; keep the top ``top_x`` pairs per type;
    the modality is the union of those lists. Returns ``target_key``.
    """
    Z = adata.obsm[score_key]            # cells x pairs (DataFrame)
    sig = Z.abs() if use_abs else Z
    sig = sig.where(sig > z_thresh, 0.0)  # zero out non-significant entries (keep magnitude)

    labels = adata.obs[celltype_key].astype(str)
    per_ct = {}                          # celltype -> ordered top-X pair list
    for ct, idx in labels.groupby(labels).groups.items():
        score = sig.loc[idx].sum(axis=0)            # per-pair score within this cell type
        per_ct[ct] = score.sort_values(ascending=False).head(top_x).index.tolist()

    # union, preserving first-seen order
    final_pairs, seen = [], set()
    for ct in per_ct:
        for p in per_ct[ct]:
            if p not in seen:
                seen.add(p); final_pairs.append(p)

    if target_key is None:
        target_key = f"{value_key}_ct_top{top_x}"
    adata.obsm[target_key] = adata.obsm[value_key][final_pairs].copy()

    adata.uns[f"{target_key}_info"] = {
        "celltype_key": celltype_key, "score_key": score_key, "value_key": value_key,
        "z_thresh": z_thresh, "top_x": top_x, "use_abs": use_abs,
        "per_celltype_pairs": per_ct, "n_pairs": len(final_pairs),
    }
    print(f"Created obsm key: {target_key}  shape={adata.obsm[target_key].shape}  "
          f"({len(per_ct)} cell types x top {top_x} -> {len(final_pairs)} unique pairs)")
    return target_key


def _resolve_pair_col(m1, m2, valid_pairs, sep="/"):
    """Return the existing pair-column name for (m1, m2), order-agnostic, or None."""
    a, b = f"{m1}{sep}{m2}", f"{m2}{sep}{m1}"
    if a in valid_pairs:
        return a
    if b in valid_pairs:
        return b
    return None


def get_marker_triplets(markers, valid_pairs=None, sep="/"):
    """Enumerate sorted marker triplets and their three constituent pair-columns.

    Parameters
    ----------
    markers
        Marker names to enumerate ``C(n, 3)`` triplets over.
    valid_pairs
        If given (e.g. ``adata.obsm['spatial_raw'].columns``), keep only triplets
        whose three pairs all exist there; pair order ("A/B" vs "B/A") is resolved
        automatically.
    sep
        Pair/triplet separator, matching ``build_spatial_obsms`` (default "/").

    Returns
    -------
    list[tuple[str, tuple[str, str, str]]]
        For each kept triplet: ``("A/B/C", (col_AB, col_BC, col_AC))``.
    """
    valid = set(valid_pairs) if valid_pairs is not None else None
    out = []
    for a, b, c in itertools.combinations(list(dict.fromkeys(markers)), 3):
        a, b, c = sorted((a, b, c))
        if valid is None:
            cols = (f"{a}{sep}{b}", f"{b}{sep}{c}", f"{a}{sep}{c}")
        else:
            cab = _resolve_pair_col(a, b, valid, sep)
            cbc = _resolve_pair_col(b, c, valid, sep)
            cac = _resolve_pair_col(a, c, valid, sep)
            if cab is None or cbc is None or cac is None:
                continue
            cols = (cab, cbc, cac)
        out.append((f"{a}{sep}{b}{sep}{c}", cols))
    return out


def _load_panel_markers(panel_name, panel_json=None):
    """Flat, deduplicated marker list from a panel in ``marker_panels.json``."""
    panel_json = panel_json or "New_Data/marker_panels.json"
    with open(panel_json) as f:
        panels = json.load(f)
    panel = panels[panel_name]
    if isinstance(panel, dict):
        markers = []
        for group in panel.values():
            markers.extend(group)
        return list(dict.fromkeys(markers))
    return list(panel)


def build_triplet_obsm(
    adata,
    source_key="spatial_raw",
    marker_panel=None,
    markers=None,
    panel_json=None,
    metric="determinant",
    tanh_scale=4.0,
    value_transform="asinh",
    top_k=None,
    chunk=2000,
    target_key=None,
):
    """Build a per-cell 3-way colocalization modality from pairwise z-scores.

    For each marker triplet (A, B, C) the three pairwise z-scores (A/B, B/C, A/C)
    in ``obsm[source_key]`` are mapped to bounded pseudo-correlations
    ``rho = tanh(z / tanh_scale)`` and combined into one number per cell via
    ``metric``:

    - ``"determinant"``    : ``1 - det(R)`` with
      ``det(R) = 1 + 2*rAB*rBC*rAC - (rAB^2 + rBC^2 + rAC^2)`` — total triadic
      association (``-0.5*log det`` is the Gaussian total correlation). det < 0
      marks a "frustrated" triad (not realizable as a joint Gaussian).
    - ``"logdet"``         : ``-0.5*log(clip(det, eps, 1))`` (Gaussian total corr.).
    - ``"triple_product"`` : ``rAB*rBC*rAC`` — signed irreducibly-3-way / balance term.
    - ``"min"`` / ``"mean"``: bottleneck / average of the three rho (interpretable).

    Use the returned ``target_key`` directly in
    ``MultiModalSCVI.setup_anndata(extra_modality_keys=[target_key])``.

    Parameters
    ----------
    source_key
        Key into ``adata.obsm`` for the wide pairwise z-score DataFrame
        (cells × pairs, columns like "A/B"). Default "spatial_raw".
    marker_panel / markers / panel_json
        Provide a panel name (looked up in ``panel_json``, default
        ``New_Data/marker_panels.json``) or an explicit ``markers`` list.
    value_transform
        Post-combine transform matching the pair modality: "asinh", "tanh", "raw".
    top_k
        If set, keep only the ``top_k`` highest-variance triplets.
    chunk
        Triplets processed per block (caps peak memory at ~3 × n_cells × chunk).

    Returns
    -------
    str
        The ``target_key`` written to ``adata.obsm`` (provenance in
        ``adata.uns[f"{target_key}_info"]``).
    """
    W = adata.obsm[source_key]
    if not isinstance(W, pd.DataFrame):
        raise TypeError(f"adata.obsm['{source_key}'] must be a DataFrame.")

    if markers is None:
        if marker_panel is None:
            raise ValueError("Provide either `markers` or `marker_panel`.")
        markers = _load_panel_markers(marker_panel, panel_json)

    triplets = get_marker_triplets(markers, valid_pairs=W.columns)
    if not triplets:
        raise ValueError(
            f"No triplets with all three pairs present in obsm['{source_key}']."
        )

    names = [t[0] for t in triplets]
    col_index = {c: i for i, c in enumerate(W.columns)}
    iAB = np.fromiter((col_index[t[1][0]] for t in triplets), int, len(triplets))
    iBC = np.fromiter((col_index[t[1][1]] for t in triplets), int, len(triplets))
    iAC = np.fromiter((col_index[t[1][2]] for t in triplets), int, len(triplets))

    base = W.to_numpy(float)
    s = float(tanh_scale)
    T = len(triplets)
    out = np.empty((base.shape[0], T), dtype=float)

    for start in range(0, T, chunk):
        sl = slice(start, min(start + chunk, T))
        a = np.tanh(base[:, iAB[sl]] / s)
        b = np.tanh(base[:, iBC[sl]] / s)
        c = np.tanh(base[:, iAC[sl]] / s)
        if metric in ("determinant", "logdet"):
            det = 1.0 + 2.0 * a * b * c - (a * a + b * b + c * c)
            out[:, sl] = (1.0 - det) if metric == "determinant" \
                else -0.5 * np.log(np.clip(det, 1e-6, 1.0))
        elif metric == "triple_product":
            out[:, sl] = a * b * c
        elif metric == "min":
            out[:, sl] = np.minimum(np.minimum(a, b), c)
        elif metric == "mean":
            out[:, sl] = (a + b + c) / 3.0
        else:
            raise ValueError(f"Unknown metric: {metric!r}")

    if value_transform == "asinh":
        out = np.arcsinh(out)
    elif value_transform == "tanh":
        out = s * np.tanh(out / s)
    elif value_transform != "raw":
        raise ValueError("value_transform must be 'asinh', 'tanh', or 'raw'.")

    df = pd.DataFrame(out, index=W.index, columns=names)

    if top_k is not None and top_k < df.shape[1]:
        keep = set(df.var(axis=0).sort_values(ascending=False).head(top_k).index)
        df = df[[c for c in names if c in keep]]  # preserve enumeration order

    if target_key is None:
        suffix = f"_top{top_k}" if top_k is not None else ""
        target_key = f"triplet_{metric}_{value_transform}{suffix}"
    adata.obsm[target_key] = df
    adata.uns[f"{target_key}_info"] = {
        "source_key": source_key,
        "marker_panel": marker_panel,
        "metric": metric,
        "tanh_scale": s,
        "value_transform": value_transform,
        "top_k": top_k,
        "n_markers": len(markers),
        "n_triplets": df.shape[1],
        "triplets": list(df.columns),
    }
    print(
        f"Created obsm key: {target_key}  shape={df.shape}  "
        f"({len(markers)} markers -> {len(triplets)} valid triplets"
        f"{'' if top_k is None else f', top {top_k} by var'})"
    )
    return target_key


def ridgeplot_spatial(
    df,
    marker1_prefix,
    marker2_prefix,
    condition_col="condition",
    value_col="join_count_z",
    ncols=2,
    tanh_scale=4.0,
):
    # ---- Filter marker pairs ----
    d = df[
        df.marker_1.str.lower().str.startswith(marker1_prefix.lower()) &
        df.marker_2.str.lower().str.startswith(marker2_prefix.lower())
    ].copy()

    raw = d[value_col].values
    arcsinh = np.arcsinh(raw)
    tanh = tanh_scale * np.tanh(raw / tanh_scale)

    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(1000, len(raw)),
        random_state=0,
    )
    quantile = qt.fit_transform(raw.reshape(-1, 1)).ravel()

    layers = {
        "raw": raw,
        f"tanh{int(tanh_scale)}": tanh,
        "asinh": arcsinh,
        "quantile": quantile,
    }

    # ---- Plot grid ----
    n = len(layers)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
    axes = axes.flatten()

    uniq = d[condition_col].unique()
    palette = sns.color_palette("Set2", len(uniq))

    for ax, (name, vals) in zip(axes, layers.items()):
        df_plot = pd.DataFrame({"value": vals, condition_col: d[condition_col].values})

        for cond, color in zip(uniq, palette):
            sns.kdeplot(
                data=df_plot[df_plot[condition_col] == cond],
                x="value",
                ax=ax,
                fill=True,
                alpha=0.7,
                bw_adjust=0.7,
                linewidth=1.2,
                color=color,
                label=cond,
            )

        ax.set_title(name)
        ax.legend(frameon=False)
        ax.set_yticks([])

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Spatial data transformations - {marker1_prefix.upper()}–{marker2_prefix.upper()}",

        fontsize=16
    )
    fig.tight_layout()
    plt.show()




def extract_column(adata, marker, layer):
    """Extract a 1D vector for a given marker from a dense or sparse layer."""
    X = adata.X if layer is None else adata.layers[layer]
    col_idx = adata.var_names.get_loc(marker)
    col = X[:, col_idx]

    if issparse(col):
        return col.toarray().ravel()
    else:
        return np.asarray(col).ravel()


def ridgeplot_multilayer(
    adata,
    marker,
    layers=["counts", "log1p", "clr", "arcsinh"],
    condition_col="condition",
    ncols=2
):
    n_layers = len(layers)
    nrows = int(np.ceil(n_layers / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), sharex=False)
    axes = axes.flatten()

    for i, layer in enumerate(layers):
        ax = axes[i]

        # extract data
        x = extract_column(adata, marker, layer)
        conditions = adata.obs[condition_col].values

        df = pd.DataFrame({marker: x, condition_col: conditions})

        sns.set_theme(style="white")
        pal = sns.color_palette("Set2", df[condition_col].nunique())

        # ridge-style plot (KDE per condition)
        for cond, color in zip(df[condition_col].unique(), pal):
            sub = df[df[condition_col] == cond]
            sns.kdeplot(
                data=sub,
                x=marker,
                ax=ax,
                fill=True,
                alpha=0.8,
                linewidth=1.3,
                bw_adjust=0.6,
                color=color,
                label=cond
            )

        ax.set_title(f"{layer} layer", fontsize=12)
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.legend(frameon=False)

    # turn off any unused empty axes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Abundance data transformations -  {marker}", fontsize=16, y=1.02)
    fig.tight_layout()
    plt.show()


def plot_tau_yanai(adata, sample_col="sample_name", tau_threshold=0.995):
    """Compute Yanai tau for each component and plot UMI count vs tau per sample."""

    # -------- Tau (Yanai) function --------
    def tau_yanai(x):
        x = np.asarray(x, dtype=float)
        p = x / np.max(x) if np.max(x) > 0 else x
        return (np.sum(1 - p) / (len(p) - 1)) if len(p) > 1 else 0.0

    # -------- Extract dense matrix safely --------
    if hasattr(adata.X, "A"):              # CSR/CSC .A attribute
        X = adata.X.A
    elif hasattr(adata.X, "toarray"):      # generic sparse
        X = adata.X.toarray()
    else:                                  # already dense
        X = adata.X

    X = X.astype(np.float32, copy=False)

    # -------- Compute tau + UMI count dataframe --------
    rows = []
    for i, idx in enumerate(adata.obs.index):
        sample = adata.obs.loc[idx, sample_col]
        umi = float(X[i].sum())
        tau = tau_yanai(X[i])
        rows.append((idx, sample, umi, tau))

    tau_df = pd.DataFrame(rows, columns=["component", "sample", "umi_count", "tau"]).set_index("component")

    # -------- Plot --------
    samples = tau_df["sample"].unique()
    fig, axes = plt.subplots(1, len(samples), figsize=(5 * len(samples), 4), sharex=True, sharey=True)

    if len(samples) == 1:
        axes = [axes]

    for ax, s in zip(axes, samples):
        df_s = tau_df[tau_df["sample"] == s]
        high_tau = df_s["tau"] > tau_threshold

        ax.scatter(df_s.loc[~high_tau, "tau"], df_s.loc[~high_tau, "umi_count"],
                   alpha=0.6, color="blue", label=f"τ ≤ {tau_threshold}")
        ax.scatter(df_s.loc[high_tau, "tau"], df_s.loc[high_tau, "umi_count"],
                   alpha=0.8, color="red", label=f"τ > {tau_threshold}")

        ax.set_title(f"Sample: {s}")
        ax.set_xlabel("τ (skewness)")
        ax.set_ylabel("UMI count")
        ax.legend()

    plt.tight_layout()
    plt.show()

    return tau_df      # return table in case you want to use it
