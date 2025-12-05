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

# --- Configuration & Utility Functions ---

def get_dense(data):
    """Converts sparse matrix to dense numpy array if necessary."""
    return data.toarray() if hasattr(data, "toarray") else np.array(data)

def robust_cv(x, axis, epsilon=0.1):
    """Median–IQR coefficient of variation with a robust stabilizer (epsilon=0.1)."""
    x = np.nan_to_num(x, nan=0.0)

    med = np.median(x, axis=axis)
    q25 = np.percentile(x, 25, axis=axis)
    q75 = np.percentile(x, 75, axis=axis)
    iqr = q75 - q25

    cv = iqr / (np.abs(med) + epsilon)
    
    cv = np.nan_to_num(cv, nan=0.0, posinf=0.0, neginf=0.0)
    return cv

def get_mean(x, axis):
    """Calculates the arithmetic mean across the specified axis."""
    x = np.nan_to_num(x, nan=0.0)
    return np.mean(x, axis=axis)

def get_std(x, axis):
    """Calculates the standard deviation across the specified axis."""
    x = np.nan_to_num(x, nan=0.0)
    return np.std(x, axis=axis)

def get_fano(x, axis):
    """Calculates the Fano Factor (Variance / Mean) across the specified axis."""
    x = np.nan_to_num(x, nan=0.0)
    
    var = np.var(x, axis=axis)
    mean = np.mean(x, axis=axis)
    
    fano = var / (mean + 1e-6)
    
    fano = np.nan_to_num(fano, nan=0.0, posinf=0.0, neginf=0.0)
    return fano

def calculate_metrics(obs, gen, model_name, modality_name, metric_type='CV'):
    """Calculates CV, Mean, Fano, or Std metrics and associated statistics."""
    results = []
    
    metric_map = {
        'CV': (robust_cv, 'Observed CV', 'Generated CV', 'Delta CV'),
        'Mean': (get_mean, 'Observed Mean', 'Generated Mean', 'Delta Mean'),
        'Fano': (get_fano, 'Observed Fano', 'Generated Fano', 'Delta Fano'),
        'Std': (get_std, 'Observed Std', 'Generated Std', 'Delta Std')
    }
    
    if metric_type not in metric_map:
        raise ValueError("metric_type must be one of 'CV', 'Mean', 'Fano', or 'Std'")

    metric_func, obs_key, gen_key, delta_key = metric_map[metric_type]

    for axis, level in [(0, "Features"), (1, "Cells")]:
        m_obs = metric_func(obs, axis)
        m_gen = metric_func(gen, axis)
        
        delta = m_gen - m_obs
        
        # Calculate correlation metrics
        finite_mask = np.isfinite(m_obs) & np.isfinite(m_gen)
        m_obs_f, m_gen_f = m_obs[finite_mask], m_gen[finite_mask]
        
        if len(m_obs_f) > 1:
            p, _ = stats.pearsonr(m_obs_f, m_gen_f)
            s, _ = stats.spearmanr(m_obs_f, m_gen_f)
        else:
            p, s = np.nan, np.nan
            
        mae = np.mean(np.abs(delta[np.isfinite(delta)]))

        results.append({
            'Model': model_name,
            'Modality': modality_name,
            'Level': level, 
            'Metric Type': metric_type,
            'MAE': mae, 'Pearson': p, 'Spearman': s,
            obs_key: m_obs, gen_key: m_gen, delta_key: delta
        })
    return results

# --- 3. Plotting Function (Unified with Matplotlib Log Scale Fix) ---

def plot_composite_ppc(modality_name, df, metric_type, scatter_color):
    
    df_modality = df[(df['Modality'] == modality_name) & (df['Metric Type'] == metric_type)]
    
    if df_modality.empty:
        print(f"\nSkipping {modality_name} {metric_type} plot: No data found.")
        return

    # Dynamic keys and labels based on metric_type
    M_OBS, M_GEN, D_M = f'Observed {metric_type}', f'Generated {metric_type}', f'Delta {metric_type}'
    plot_title = f"{modality_name} Posterior Predictive Checks ({metric_type})"
    
    # Identify the single model name for plotting columns
    all_model_names = df_modality['Model'].unique().tolist()
    N_models = len(all_model_names)
    N_cols = N_models + 2 

    # Figure setup
    fig = plt.figure(figsize=(4.5 * N_cols, 12)) 
    gs = gridspec.GridSpec(2, N_cols, 
                           height_ratios=[1, 1], width_ratios=[1] * N_models + [0.75, 0.8], 
                           left=0.05, wspace=0.3, hspace=0.4, top=0.9) 
    
    
    # Loop over Rows: Features (i=0) and Cells (i=1)
    for i, level in enumerate(["Features", "Cells"]):
        
        df_level = df_modality[df_modality['Level'] == level]
        violin_data = []
        metrics_for_table = {} 
        
        # --- Scatter Plots (Columns 0 to N-1) ---
        for j, model_name in enumerate(all_model_names):
            ax_s = fig.add_subplot(gs[i, j])
            row = df_level[df_level['Model'] == model_name]

            if row.empty:
                ax_s.text(0.5, 0.5, "N/A", fontsize=16, ha='center', va='center', color='gray')
                ax_s.set_xticks([]); ax_s.set_yticks([]); ax_s.set_frame_on(False)
                metrics_for_table[model_name] = ["N/A"] * 3
            else:
                c_obs = np.concatenate(row[M_OBS].values)
                c_gen = np.concatenate(row[M_GEN].values)
                delta = np.concatenate(row[D_M].values)
                
                # --- MODALITY AND LEVEL-SPECIFIC SCALING FIX ---
                # Force log scale only for the problematic 'Spatial Cells' row
                use_log_scale = (modality_name == 'Spatial' and level == 'Cells') 
                
                # Plot Scatter using the ORIGINAL (linear) data
                ax_s.scatter(c_obs, c_gen, c=scatter_color, s=15, alpha=0.5, ec='white', lw=0.5)
                
                if use_log_scale:
                    # Use Matplotlib's built-in log scale for better visualization of small numbers
                    ax_s.set_xscale('log')
                    ax_s.set_yscale('log')
                    scale_label = " (log-scale)"
                    
                    # Calculate robust limits on the linear data for the diagonal
                    data_combined = np.concatenate([c_obs, c_gen])
                    # Filter for only positive values to avoid log(0) issues
                    data_combined_pos = data_combined[data_combined > 1e-6]
                    
                    if data_combined_pos.size > 0:
                        lim_min = np.percentile(data_combined_pos, 1)
                        lim_max = np.percentile(data_combined_pos, 99.5) * 1.1
                    else:
                        # Fallback for extremely sparse data
                        lim_min, lim_max = 1e-3, 1 

                    ax_s.set_xlim(lim_min, lim_max)
                    ax_s.set_ylim(lim_min, lim_max)
                    
                    # Diagonal line uses the new log limits
                    ax_s.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1.5)

                else:
                    # Linear scale for everything else (Abundance and Spatial Features)
                    lim_max_robust = np.percentile(np.concatenate([c_obs, c_gen]), 99.5) * 1.05
                    lim_max_final = max(lim_max_robust, 0.5) 
                    
                    ax_s.set_xlim(0, lim_max_final)
                    ax_s.set_ylim(0, lim_max_final)
                    
                    # Diagonal line uses linear limits
                    ax_s.plot([0, lim_max_final], [0, lim_max_final], 'k--', lw=1.5)
                    scale_label = ""
                
                ax_s.set_aspect('equal')
                
                # Labels reflect the conditional scale
                ax_s.set_title(model_name, fontsize=10) 
                ax_s.set_xlabel(f"Observed {metric_type}{scale_label}" if i == 1 else "")
                ax_s.set_ylabel(f"Generated {metric_type}{scale_label}" if j == 0 else "")

                # Prepare for Violin plot (uses UNTRANSFORMED delta)
                df_temp = pd.DataFrame({'Delta Metric': delta, 'Model': model_name})
                violin_data.append(df_temp)

                # Store Stats for Table
                metrics_for_table[model_name] = [f"{row['MAE'].iloc[0]:.2f}", 
                                                 f"{row['Pearson'].iloc[0]:.2f}", 
                                                 f"{row['Spearman'].iloc[0]:.2f}"]
            
        # --- Violin Plot (Column N_models) ---
        ax_v = fig.add_subplot(gs[i, N_models])
        
        if violin_data:
            violin_df = pd.concat(violin_data)
            q_abs = np.percentile(np.abs(violin_df['Delta Metric'].values), 98)
            violin_df['Delta Metric'] = violin_df['Delta Metric'].clip(-q_abs, q_abs)

            sns.violinplot(data=violin_df, x='Model', y='Delta Metric', hue='Model', ax=ax_v, 
                           palette='tab10', inner="quartile", cut=0, legend=False)
            
            ax_v.axhline(0, color='gray', ls=':')
            ax_v.set_title(f"{level} Error", fontsize=10)
            ax_v.set_xlabel("")
            ax_v.set_ylabel(f"Delta {metric_type} (Gen - Obs)")
            
            available_model_names = [m for m in all_model_names if m in df_level['Model'].values]
            ax_v.set_xticks(np.arange(len(available_model_names))) 
            ax_v.set_xticklabels(available_model_names, rotation=45, ha='right') 

        sns.despine(ax=ax_v)
        
        # --- Table (Column N_models + 1) ---
        ax_table_col = fig.add_subplot(gs[i, N_models + 1])
        ax_table_col.axis('off')

        # Use the stored metrics dictionary
        table_metrics_row = [metrics_for_table[m] for m in all_model_names]

        table_df = pd.DataFrame(table_metrics_row, index=all_model_names, 
                                columns=["MAE", "Pearson", "Spearman"])
        
        # Table Header
        ax_table_col.text(0.5, 1.0, f'{level} Metrics', ha='center', va='bottom', fontsize=12, fontweight='bold')

        mpl_table = ax_table_col.table(cellText=table_df.values, rowLabels=table_df.index.tolist(), 
                                       colLabels=table_df.columns.tolist(), loc='center', 
                                       cellLoc='center', edges='open')
        
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(7) 
        mpl_table.scale(0.8, 1.2) 
        
    # --- Figure Title ---
    fig.text(0.05, 0.96, plot_title, fontsize=16, fontweight='bold')
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
