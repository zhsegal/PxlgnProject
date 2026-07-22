"""
B cell colocalization & abundance analysis utilities.
Independent from nalm_utils.py to avoid conflicts.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import networkx as nx
import matplotlib.patches as mpatches


# ═══════════════════════════════════════════════════════════════════════════════
# Synapse scores (rounds 1–5 design — see New_Data/synapse_scores_design.md)
# ═══════════════════════════════════════════════════════════════════════════════

# Marker names as they appear in adata.var_names. CEACAM8 = CD66b in this
# dataset. Markers / pairs missing at runtime are skipped with a one-line
# warning.
#
# Panel spec keys:
#   abundance      — markers averaged for the abundance arm of the score.
#   auto_within    — marker set whose all-pairs colocs are auto-discovered
#                     from obsm[coloc_key] (broad signal-averaging base).
#   curated_pairs  — explicit cross-cluster pairs to add on top (both
#                     orderings checked). These are the design-doc pairs.
DEFAULT_SYNAPSE_PANELS = {
    'cd8_synapse': {
        'cell_types': {'CD8'},
        'abundance': [
            'CD3e', 'CD8', 'CD2', 'CD28', 'CD134', 'CD137',
            'CD226', 'TIGIT', 'CD279', 'VISTA',
            'CD11a', 'CD50', 'KLRG1', 'CD94', 'CD48', 'CD352', 'CD53',
            'CD25', 'CD71', 'CD69',
        ],
        'auto_within': [
            'CD3e', 'CD8', 'CD2', 'CD28', 'CD134', 'CD137', 'CD226',
            'TIGIT', 'CD279', 'VISTA',
            'CD11a', 'CD50', 'KLRG1', 'CD94', 'CD48', 'CD352', 'CD53',
        ],
        'curated_pairs': [
            ('CD3e', 'CD8'),
            ('CD11a', 'CD45'),
            ('CD19', 'CD45'),
        ],
    },
    'cd4_synapse': {
        'cell_types': {'CD4'},
        'abundance': [
            'CD3e', 'CD4', 'CD2', 'CD28', 'CD134', 'CD137',
            'CD226', 'TIGIT', 'CD279', 'VISTA',
            'CD11a', 'CD50', 'CD48', 'CD352', 'CD53',
            'CD25', 'CD69', 'CD154',
        ],
        'auto_within': [
            'CD3e', 'CD4', 'CD2', 'CD28', 'CD134', 'CD137', 'CD226',
            'TIGIT', 'CD279', 'VISTA',
            'CD11a', 'CD50', 'CD48', 'CD352', 'CD53',
        ],
        'curated_pairs': [
            ('CD3e', 'HLA-DR-DP-DQ'),
            ('CD4',  'HLA-DR-DP-DQ'),
            ('CD44', 'HLA-DR-DP-DQ'),
            ('CD45', 'HLA-DR-DP-DQ'),
            ('CD11a', 'CD45'),
            ('CD2',  'SLAMF6'),
        ],
    },
    'apc_activation': {
        'cell_types': {'B'},
        'abundance': [
            'HLA-DR-DP-DQ', 'HLA-DR', 'HLA-DQ', 'HLA-ABC',
            'CD80', 'CD86', 'CD40',
            'CD19', 'CD20', 'CD79a',
            'CD54', 'CD58', 'CD50', 'CD102',
        ],
        'auto_within': [
            'HLA-DR-DP-DQ', 'HLA-DR', 'HLA-DQ', 'HLA-ABC',
            'CD80', 'CD86', 'CD40',
            'CD19', 'CD20', 'CD79a',
            'CD54', 'CD58', 'CD50', 'CD102',
        ],
        'curated_pairs': [
            ('CD80', 'HLA-DR-DP-DQ'),
            ('CD86', 'HLA-DR-DP-DQ'),
            ('CD40', 'HLA-DR-DP-DQ'),
            ('CD54', 'HLA-DR-DP-DQ'),
            ('CD58', 'HLA-DR-DP-DQ'),
            ('CD80', 'CD86'),
        ],
    },
    'apc_inhibitory': {
        'cell_types': {'B'},
        'abundance': [
            'CD274', 'CD273', 'CD32', 'CD72', 'CD305',
            'CD22', 'CD66b', 'CD162', 'CD39', 'CD73',
        ],
        'auto_within': [
            'CD274', 'CD273', 'CD32', 'CD72', 'CD305',
            'CD22', 'CD66b', 'CD162', 'CD39', 'CD73',
        ],
        'curated_pairs': [
            ('CD274', 'CD305'),
            ('CD273', 'CD274'),
            ('CD162', 'CD22'),
            ('CD162', 'CD305'),
            ('CD162', 'CD19'),
            ('CD274', 'CD22'),
            ('CD39',  'CD73'),
        ],
    },
}


def _safe_z(arr):
    """Z-score ignoring NaN; returns all-NaN if std==0 or <2 finite values."""
    arr = np.asarray(arr, dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() < 2:
        return np.full_like(arr, np.nan, dtype=float)
    mu, sd = arr[mask].mean(), arr[mask].std(ddof=0)
    if sd == 0:
        return np.full_like(arr, np.nan, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    out[mask] = (arr[mask] - mu) / sd
    return out


def _pick_pair_col(coloc_cols, a, b):
    """Return the actual column name for an unordered marker pair, or None."""
    for candidate in (f'{a}/{b}', f'{b}/{a}'):
        if candidate in coloc_cols:
            return candidate
    return None


def compute_synapse_scores(
    adata,
    panels=None,
    layer='arcsinh',
    coloc_key='spatial_asinh5',
    abund_weight=1/3,
    coloc_weight=2/3,
    cell_type_col='cell_type_annot',
    score_mode='mean',
    verbose=True,
):
    """Compute 1:2 abundance:spatial-weighted per-cell synapse scores.

    For each panel, restrict to the panel's cell types and compute
        score = abund_w · z(abund_arm) + coloc_w · z(coloc_arm)
    where z() is across cells of that cell type.

    score_mode controls how the panel is aggregated:
      - 'mean'        : abund_arm = mean across markers (current default).
                        High-variance markers dominate before z-scoring.
      - 'sum_zscore'  : abund_arm = sum of per-marker z-scores.
                        Every marker contributes equally — more democratic.
    Same logic applies to coloc pairs.

    Returns
    -------
    pd.DataFrame indexed by adata.obs_names with three columns per panel:
        {name}, {name}_abundance, {name}_coloc
    Cells outside the panel's cell_types are NaN.
    """
    if score_mode not in ('mean', 'sum_zscore'):
        raise ValueError(f'score_mode must be mean or sum_zscore, got {score_mode!r}')
    if panels is None:
        panels = DEFAULT_SYNAPSE_PANELS

    sp_full = adata.obsm[coloc_key]
    if not isinstance(sp_full, pd.DataFrame):
        sp_full = pd.DataFrame(sp_full, index=adata.obs_names)
    coloc_cols = set(sp_full.columns)
    available_vars = set(adata.var_names)

    out = pd.DataFrame(index=adata.obs_names, dtype=float)

    for name, spec in panels.items():
        cell_types = spec['cell_types']
        markers = [m for m in spec['abundance'] if m in available_vars]
        missing_m = [m for m in spec['abundance'] if m not in available_vars]

        # Positive pairs = (a) all coloc cols within auto_within
        # union (b) curated_pairs (both orderings).
        auto_within = set(spec.get('auto_within', []))
        auto_pairs = set()
        for col in coloc_cols:
            parts = col.split('/')
            if len(parts) == 2 and parts[0] in auto_within and parts[1] in auto_within:
                auto_pairs.add(col)

        curated_selected = []
        missing_curated = []
        for a, b in spec.get('curated_pairs', []):
            col = _pick_pair_col(coloc_cols, a, b)
            if col is None:
                missing_curated.append(f'{a}/{b}')
            else:
                curated_selected.append(col)

        # Negative pairs — coloc that should be LOW in a productive synapse
        # (e.g. CD45/CD3e exclusion from cSMAC). Each contributes with sign −1.
        neg_selected = []
        missing_neg = []
        for a, b in spec.get('negative_pairs', []):
            col = _pick_pair_col(coloc_cols, a, b)
            if col is None:
                missing_neg.append(f'{a}/{b}')
            else:
                neg_selected.append(col)

        selected_pos_pairs = sorted(auto_pairs.union(curated_selected))
        selected_neg_pairs = sorted(set(neg_selected))

        if verbose:
            print(f'[{name}] cell_types={sorted(cell_types)}  '
                  f'abund {len(markers)}/{len(spec["abundance"])}  '
                  f'coloc+ {len(selected_pos_pairs)} '
                  f'(auto {len(auto_pairs)} + curated {len(curated_selected)}'
                  f'/{len(spec.get("curated_pairs", []))})  '
                  f'coloc- {len(selected_neg_pairs)}'
                  f'/{len(spec.get("negative_pairs", []))}')
            if missing_m:
                print(f'    missing markers: {missing_m}')
            if missing_curated:
                print(f'    missing curated pairs: {missing_curated}')
            if missing_neg:
                print(f'    missing negative pairs: {missing_neg}')

        mask = adata.obs[cell_type_col].isin(cell_types).values
        idx = np.where(mask)[0]

        # Abundance arm
        if markers and len(idx) > 0:
            X = adata[idx, markers].layers[layer]
            if hasattr(X, 'toarray'):
                X = X.toarray()
            X = np.asarray(X, dtype=float)
            if score_mode == 'mean':
                abund_arm = X.mean(axis=1)
            else:  # sum_zscore
                Xz = np.apply_along_axis(_safe_z, 0, X)
                abund_arm = np.nansum(Xz, axis=1)
        else:
            abund_arm = np.full(len(idx), np.nan)

        # Coloc arm — positive and negative aggregated separately, then
        # each z-scored across cells before signed combination. Z-per-arm
        # prevents the larger set from drowning the smaller.
        def _agg_pairs(pair_list):
            if not pair_list or len(idx) == 0:
                return None
            v = sp_full.iloc[idx][pair_list].values.astype(float)
            if score_mode == 'mean':
                return np.nanmean(v, axis=1)
            else:  # sum_zscore
                return np.nansum(np.apply_along_axis(_safe_z, 0, v), axis=1)

        pos_arm = _agg_pairs(selected_pos_pairs)
        neg_arm = _agg_pairs(selected_neg_pairs)

        z_pos = _safe_z(pos_arm) if pos_arm is not None else None
        z_neg = _safe_z(neg_arm) if neg_arm is not None else None

        if z_pos is not None and z_neg is not None:
            coloc_arm = z_pos - z_neg
        elif z_pos is not None:
            coloc_arm = pos_arm
        elif z_neg is not None:
            coloc_arm = -neg_arm
        else:
            coloc_arm = np.full(len(idx), np.nan)

        z_abund = _safe_z(abund_arm)
        z_coloc = _safe_z(coloc_arm)

        # Combine — if one arm is all-NaN, fall back to the other at full weight
        if np.all(np.isnan(z_abund)):
            score = z_coloc
        elif np.all(np.isnan(z_coloc)):
            score = z_abund
        else:
            score = abund_weight * z_abund + coloc_weight * z_coloc

        # Per-arm diagnostics (NaN when that arm absent)
        z_coloc_pos = z_pos if z_pos is not None else np.full(len(idx), np.nan)
        z_coloc_neg = z_neg if z_neg is not None else np.full(len(idx), np.nan)

        # Place into full-length columns (NaN outside cell_types)
        for col, vals in [
            (name,                   score),
            (f'{name}_abundance',    z_abund),
            (f'{name}_coloc',        z_coloc),
            (f'{name}_coloc_pos',    z_coloc_pos),
            (f'{name}_coloc_neg',    z_coloc_neg),
        ]:
            full = np.full(adata.n_obs, np.nan, dtype=float)
            full[idx] = vals
            out[col] = full

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting helpers for synapse scores (extracted from synapse_analysis.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_p(p):
    """Compact p-value formatter."""
    if not np.isfinite(p):
        return 'p=NaN'
    if p < 1e-4:
        return f'p={p:.1e}'
    return f'p={p:.3f}'


def build_synapse_obs_long(adata, score_cols, sys_order,
                           cell_type_col='cell_type_annot',
                           system_col='cell_system',
                           time_col='time', cond_col='condition'):
    """Long-form obs slice with `time_cond`, restricted to `sys_order`."""
    cols = [cell_type_col, system_col, time_col, cond_col] + list(score_cols)
    df = adata.obs[cols].copy()
    df['time_cond'] = (df[time_col].astype(str) + ' '
                       + df[cond_col].astype(str))
    return df[df[system_col].isin(sys_order)]


def plot_synapse_score_grid(
    obs_long, score_panels, variants,
    sys_order, sys_palette,
    x_levels, mode='split',
    cell_type_col='cell_type_annot',
    system_col='cell_system',
    suptitle=None, col_width=4.0, row_height=3.0,
    annotate=True,
):
    """Violin grid of synapse scores — rows=variants, cols=score panels.

    mode='split':
        x-axis is `time_cond` (`x_levels`), violins split by `system_col`.
    mode='by_system':
        Restrict to ONE time_cond (`x_levels` must be length-1 string),
        x-axis is `system_col`. Title gets Mann-Whitney NALM-vs-HB p.

    Parameters
    ----------
    score_panels : list of (col_base, {cell_type_filter})
    variants     : list of (label, col_suffix)
    """
    if mode == 'by_system':
        if not isinstance(x_levels, str):
            raise ValueError("mode='by_system' expects x_levels=<time_cond>")
        tc = x_levels
        sub_obs = obs_long[obs_long['time_cond'] == tc]
    else:
        sub_obs = obs_long

    n_rows, n_cols = len(variants), len(score_panels)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(col_width * n_cols, row_height * n_rows),
                             sharey='row', squeeze=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.01)

    for r_idx, (v_name, v_suf) in enumerate(variants):
        for c_idx, (base, ctypes) in enumerate(score_panels):
            ax = axes[r_idx, c_idx]
            col = base + v_suf
            if col not in sub_obs.columns:
                ax.set_visible(False); continue
            s = sub_obs[sub_obs[cell_type_col].isin(ctypes)][
                [system_col, 'time_cond', col]].dropna()
            if s.empty:
                ax.set_visible(False); continue

            if mode == 'split':
                sns.violinplot(
                    data=s, x='time_cond', y=col, hue=system_col,
                    order=x_levels, hue_order=sys_order,
                    split=True, inner='quartile', cut=0, linewidth=0.55,
                    palette=sys_palette, ax=ax,
                )
                ax.tick_params(axis='x', rotation=15, labelsize=8)
                ttl = f'{base} — {v_name}'
                if not (r_idx == 0 and c_idx == 0):
                    leg = ax.get_legend()
                    if leg is not None: leg.remove()
                else:
                    ax.legend(fontsize=7, title='', loc='upper left')
            else:  # by_system
                sns.violinplot(
                    data=s, x=system_col, y=col,
                    order=sys_order, inner='quartile', cut=0,
                    linewidth=0.6, palette=sys_palette, ax=ax,
                )
                ax.set_xticklabels(['HT/NALM', 'HT/HB'], fontsize=8)
                ttl = f'{base} — {v_name}'
                if annotate:
                    a = s.loc[s[system_col] == sys_order[0], col].values
                    b = s.loc[s[system_col] == sys_order[1], col].values
                    if len(a) >= 5 and len(b) >= 5:
                        _, p = mannwhitneyu(a, b, alternative='two-sided')
                        d = np.median(a) - np.median(b)
                        ttl += (f'\nΔmed={d:+.2f}  {_fmt_p(p)}  '
                                f'(n={len(a)} vs {len(b)})')

            ax.axhline(0, color='grey', lw=0.5, ls='--')
            ax.set_title(ttl, fontsize=9)
            ax.set_xlabel('')
            ax.set_ylabel(f'{v_name}\nz-score' if c_idx == 0 else '')

    plt.tight_layout()
    return fig, axes


def plot_synapse_ablation_violins(
    adata, score_panels, focus_conds, sys_order, sys_palette,
    arms=(('abundance', '_abundance'),
          ('coloc',     '_coloc'),
          ('combined',  '')),
    variant_suffix='', variant_label='mean',
    cell_type_col='cell_type_annot',
    system_col='cell_system',
    time_col='time', cond_col='condition',
):
    """Ablation violins: rows = arms, cols = score panels. One figure per cond.

    Returns list of (cond, fig, axes).
    """
    figs = []
    for tc in focus_conds:
        n_rows, n_cols = len(arms), len(score_panels)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(3.2 * n_cols, 3.0 * n_rows),
                                 sharey='row', squeeze=False)
        fig.suptitle(f'Ablation — HT/NALM vs HT/HB — {tc} ({variant_label})',
                     fontsize=12, y=1.01)

        for r_idx, (arm_name, arm_suf) in enumerate(arms):
            for c_idx, (base, ctypes) in enumerate(score_panels):
                ax = axes[r_idx, c_idx]
                col = base + variant_suffix + arm_suf
                if col not in adata.obs.columns:
                    ax.set_visible(False); continue
                s = adata.obs[
                    adata.obs[cell_type_col].isin(ctypes)
                    & adata.obs[system_col].isin(sys_order)
                ][[system_col, time_col, cond_col, col]].dropna()
                s['time_cond'] = (s[time_col].astype(str) + ' '
                                  + s[cond_col].astype(str))
                s = s[s['time_cond'] == tc]
                if s.empty:
                    ax.set_visible(False); continue

                sns.violinplot(
                    data=s, x=system_col, y=col,
                    order=sys_order, inner='quartile', cut=0,
                    linewidth=0.6, palette=sys_palette, ax=ax,
                )
                ax.axhline(0, color='grey', lw=0.5, ls='--')

                a = s.loc[s[system_col] == sys_order[0], col].values
                b = s.loc[s[system_col] == sys_order[1], col].values
                if len(a) >= 5 and len(b) >= 5:
                    _, p = mannwhitneyu(a, b, alternative='two-sided')
                    d = np.median(a) - np.median(b)
                    ttl = (f'{base} — {arm_name}\n'
                           f'Δmed={d:+.2f}  {_fmt_p(p)}')
                else:
                    ttl = f'{base} — {arm_name}'
                ax.set_title(ttl, fontsize=9)
                ax.set_xlabel('')
                ax.set_ylabel(arm_name if c_idx == 0 else '')
                ax.set_xticklabels(['HT/NALM', 'HT/HB'], fontsize=8)

        plt.tight_layout()
        figs.append((tc, fig, axes))
    return figs


def build_ablation_table(
    adata, score_panels, variants, focus_conds, sys_order,
    arms=(('abundance', '_abundance'),
          ('coloc',     '_coloc'),
          ('combined',  '')),
    cell_type_col='cell_type_annot',
    system_col='cell_system',
    time_col='time', cond_col='condition',
):
    """Long-form (variant × score × arm × cond) table of Δmedian + MW p."""
    rows = []
    for v_name, v_suf in variants:
        for base, ctypes in score_panels:
            for arm_name, arm_suf in arms:
                col = base + v_suf + arm_suf
                if col not in adata.obs.columns:
                    continue
                s = adata.obs[
                    adata.obs[cell_type_col].isin(ctypes)
                    & adata.obs[system_col].isin(sys_order)
                ][[system_col, time_col, cond_col, col]].dropna()
                s['time_cond'] = (s[time_col].astype(str) + ' '
                                  + s[cond_col].astype(str))
                for tc in focus_conds:
                    ss = s[s['time_cond'] == tc]
                    a = ss.loc[ss[system_col] == sys_order[0], col].values
                    b = ss.loc[ss[system_col] == sys_order[1], col].values
                    if len(a) < 5 or len(b) < 5:
                        continue
                    _, p = mannwhitneyu(a, b, alternative='two-sided')
                    rows.append({
                        'variant':  v_name,
                        'score':    base,
                        'arm':      arm_name,
                        'cond':     tc,
                        'n_NALM':   len(a),
                        'n_HB':     len(b),
                        'med_NALM': float(np.median(a)),
                        'med_HB':   float(np.median(b)),
                        'd_med':    float(np.median(a) - np.median(b)),
                        'p':        float(p),
                        'nlog10p':  float(-np.log10(max(p, 1e-300))),
                    })
    return pd.DataFrame(rows)


def compute_derived_metrics(
    adata,
    coloc_key='spatial_asinh5',
    cd4_baseline_subtract=True,
    sample_keys=('cell_system', 'time', 'condition'),
    cell_type_col='cell_type_annot',
    time_col='time',
    cond_col='condition',
    system_col='cell_system',
    trog_b_markers=('CD19', 'CD22', 'HLA-DR-DP-DQ', 'CD10'),
):
    """Write four derived metrics to adata.obs.

    Requires that compute_synapse_scores has already populated cd8_synapse,
    cd4_synapse, apc_activation, apc_inhibitory in adata.obs.

      kill_permission       = cd8_synapse − apc_inhibitory     (CD8 cells; apc_inhibitory broadcast as sample mean from B cells)
      apc_functional_state  = apc_activation − apc_inhibitory  (B cells only)
      helper_licensing      = cd4_mean_per_sample × apc_act_mean_per_sample, broadcast to all cells in that sample
      trogocytosis_score    = mean coloc of pairs with ≥1 endpoint in trog_b_markers on T cells (CD4/CD8)

    Baseline subtraction applies a per-cell_system offset to cd4_synapse
    (6h Mock mean) inside helper_licensing only — the stored cd4_synapse stays raw.
    """
    obs = adata.obs

    # --- per-sample sample-level means (B-side and T-side) ----------------
    sample_idx = list(sample_keys)
    sample_group = obs.groupby(sample_idx, observed=True)

    b_inhib_per_sample = sample_group.apply(
        lambda d: d.loc[d[cell_type_col] == 'B', 'apc_inhibitory'].mean()
    ).rename('apc_inhibitory_sample_mean')

    b_act_per_sample = sample_group.apply(
        lambda d: d.loc[d[cell_type_col] == 'B', 'apc_activation'].mean()
    ).rename('apc_activation_sample_mean')

    cd4_per_sample = sample_group.apply(
        lambda d: d.loc[d[cell_type_col] == 'CD4', 'cd4_synapse'].mean()
    ).rename('cd4_synapse_sample_mean')

    sample_means = pd.concat([b_inhib_per_sample, b_act_per_sample, cd4_per_sample], axis=1)
    sample_df = obs[sample_idx].merge(sample_means, left_on=sample_idx, right_index=True, how='left')
    sample_df.index = obs.index

    # --- kill_permission (CD8 only) ---------------------------------------
    cd8_mask = (obs[cell_type_col] == 'CD8').values
    kill = np.full(adata.n_obs, np.nan, dtype=float)
    kill[cd8_mask] = (
        obs.loc[cd8_mask, 'cd8_synapse'].values
        - sample_df.loc[cd8_mask, 'apc_inhibitory_sample_mean'].values
    )
    adata.obs['kill_permission'] = kill

    # --- apc_functional_state (B only) ------------------------------------
    b_mask = (obs[cell_type_col] == 'B').values
    afs = np.full(adata.n_obs, np.nan, dtype=float)
    afs[b_mask] = (
        obs.loc[b_mask, 'apc_activation'].values
        - obs.loc[b_mask, 'apc_inhibitory'].values
    )
    adata.obs['apc_functional_state'] = afs

    # --- helper_licensing (per-sample, broadcast to all cells) ------------
    cd4_means = sample_df['cd4_synapse_sample_mean'].copy()
    if cd4_baseline_subtract:
        # Per-cell_system baseline: mean of cd4_synapse_sample_mean across
        # the 6h Mock samples for that system.
        baseline = (
            sample_df[(obs[time_col] == '6h') & (obs[cond_col] == 'Mock')]
            .groupby(system_col, observed=True)['cd4_synapse_sample_mean'].mean()
        )
        for sys_val, offset in baseline.items():
            if not np.isnan(offset):
                cd4_means.loc[obs[system_col] == sys_val] -= offset

    adata.obs['helper_licensing'] = (
        cd4_means.values * sample_df['apc_activation_sample_mean'].values
    )

    # --- trogocytosis_score (T cells only) --------------------------------
    sp_full = adata.obsm[coloc_key]
    if not isinstance(sp_full, pd.DataFrame):
        sp_full = pd.DataFrame(sp_full, index=adata.obs_names)
    b_marker_set = set(trog_b_markers)
    trog_pairs = [c for c in sp_full.columns
                  if len(c.split('/')) == 2
                  and any(m in b_marker_set for m in c.split('/'))]

    t_mask = obs[cell_type_col].isin(['CD8', 'CD4']).values
    trog = np.full(adata.n_obs, np.nan, dtype=float)
    if trog_pairs and t_mask.any():
        idx = np.where(t_mask)[0]
        trog[t_mask] = sp_full.iloc[idx][trog_pairs].mean(axis=1, skipna=True).values
    adata.obs['trogocytosis_score'] = trog


# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Data utilities
# ═══════════════════════════════════════════════════════════════════════════════

def load_marker_panel(panel_name):
    """Load marker list from marker_panels.json by panel name (flat list)."""
    import json
    with open('marker_panels.json') as f:
        marker_panels = json.load(f)

    if panel_name in marker_panels:
        panel = marker_panels[panel_name]
        # If panel is a dict of categories, flatten to list
        if isinstance(panel, dict):
            markers = []
            for group in panel.values():
                markers.extend(group)
            return list(dict.fromkeys(markers))  # deduplicate, preserve order
        return list(panel)
    return []


def load_marker_panel_grouped(panel_name):
    """Load a marker panel keeping its category grouping."""
    import json
    with open('marker_panels.json') as f:
        marker_panels = json.load(f)
    return marker_panels.get(panel_name, {})


def get_marker_cols(col_list, marker):
    """Return all pair columns containing marker on either side."""
    return [c for c in col_list if marker in c.split('/')]


def get_pairwise_cols(col_list, protein_set):
    """Return pair columns where both proteins are in protein_set."""
    return [c for c in col_list
            if len(c.split('/')) == 2
            and c.split('/')[0] in protein_set
            and c.split('/')[1] in protein_set]


def select_top_partners(sp_sub, marker, all_cols, n=30):
    """Select top n colocalization partners by mean in sp_sub."""
    cols = get_marker_cols(all_cols, marker)
    if not cols:
        return []
    means = sp_sub[cols].mean()
    top = means.nlargest(n)
    return [c.replace(marker + '/', '').replace('/' + marker, '')
            for c in top.index.tolist()]


def build_mean_matrix(sp_sub, pair_cols, markers):
    """Build symmetric marker×marker colocalization matrix from pair columns."""
    mat = pd.DataFrame(0.0, index=markers, columns=markers)
    means = sp_sub[pair_cols].mean()
    for pair_name, val in means.items():
        parts = pair_name.split('/')
        if len(parts) == 2:
            a, b = parts
            if a in mat.index and b in mat.columns:
                mat.loc[a, b] = val
                mat.loc[b, a] = val
    return mat


def compute_ward_linkage(mat_diff):
    """Compute Ward linkage from difference matrix."""
    dist = mat_diff.abs().max().max() - mat_diff.abs() + 1e-10
    np.fill_diagonal(dist.values, 0)
    condensed = squareform(dist.values, checks=False)
    return linkage(condensed, method='ward')


def compute_marker_diff_coloc(sp_a, sp_b, pair_cols, marker_name):
    """Mann-Whitney U test for marker colocalization between two systems."""
    results = []
    for c in pair_cols:
        partner_raw = c.replace(marker_name + '/', '').replace('/' + marker_name, '')
        vals_a = sp_a[c].values
        vals_b = sp_b[c].values
        mean_a = vals_a.mean()
        mean_b = vals_b.mean()
        mean_diff = mean_b - mean_a
        _, pval = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
        results.append({
            'pair': c, 'partner': display_name(partner_raw),
            'mean_a': mean_a, 'mean_b': mean_b,
            'mean_diff': mean_diff, 'pval': pval,
        })

    df = pd.DataFrame(results)
    _, df['padj'], _, _ = multipletests(df['pval'], method='fdr_bh')
    return df.sort_values('padj')


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting utilities
# ═══════════════════════════════════════════════════════════════════════════════

def sig_label(padj):
    """Convert p-value to significance stars."""
    if padj < 0.001: return '***'
    if padj < 0.01:  return '**'
    if padj < 0.05:  return '*'
    return 'ns'


def compute_da(adata_sub, sys_a, sys_b):
    """Mann-Whitney U + BH-FDR differential abundance between two systems."""
    X = pd.DataFrame(
        np.array(adata_sub.layers['arcsinh'], dtype=np.float32),
        index=adata_sub.obs_names,
        columns=adata_sub.var_names,
    )
    grp = adata_sub.obs['cell_system']
    a_mask = (grp == sys_a).values
    b_mask = (grp == sys_b).values

    results = []
    for marker in X.columns:
        a_vals = X.values[a_mask, X.columns.get_loc(marker)]
        b_vals = X.values[b_mask, X.columns.get_loc(marker)]
        mean_diff = a_vals.mean() - b_vals.mean()
        _, pval = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
        results.append({'marker': marker, 'mean_diff': mean_diff, 'pval': pval})

    df = pd.DataFrame(results)
    _, df['padj'], _, _ = multipletests(df['pval'], method='fdr_bh')
    return df


def compute_da_multigroup(adata_sub, group_key, layer='arcsinh'):
    """Kruskal-Wallis + pairwise Mann-Whitney for multi-group DA.

    Returns DataFrame with marker, kw_stat, kw_pval, kw_padj, mean_*, and per-pair columns.
    """
    from scipy.stats import kruskal
    from itertools import combinations

    X = np.array(adata_sub.layers[layer], dtype=np.float32)
    markers = list(adata_sub.var_names)
    groups = adata_sub.obs[group_key].values
    group_names = sorted(set(groups))

    # Build masks
    masks = {g: groups == g for g in group_names}

    results = []
    pairs = list(combinations(group_names, 2))

    for idx, marker in enumerate(markers):
        vals = {g: X[masks[g], idx] for g in group_names}
        row = {'marker': marker}

        # Kruskal-Wallis across all groups
        arrays = [vals[g] for g in group_names]
        if all(len(a) > 0 for a in arrays):
            stat, pval = kruskal(*arrays)
        else:
            stat, pval = np.nan, 1.0
        row['kw_stat'] = stat
        row['kw_pval'] = pval

        # Group means
        for g in group_names:
            row[f'mean_{g}'] = vals[g].mean() if len(vals[g]) > 0 else np.nan

        # Pairwise Mann-Whitney
        for ga, gb in pairs:
            if len(vals[ga]) > 0 and len(vals[gb]) > 0:
                _, pw = mannwhitneyu(vals[ga], vals[gb], alternative='two-sided')
                row[f'pval_{ga}_vs_{gb}'] = pw
                row[f'diff_{ga}_vs_{gb}'] = vals[ga].mean() - vals[gb].mean()
            else:
                row[f'pval_{ga}_vs_{gb}'] = 1.0
                row[f'diff_{ga}_vs_{gb}'] = 0.0

        results.append(row)

    df = pd.DataFrame(results)
    _, df['kw_padj'], _, _ = multipletests(df['kw_pval'], method='fdr_bh')

    # Also correct pairwise p-values
    for ga, gb in pairs:
        col = f'pval_{ga}_vs_{gb}'
        _, df[f'padj_{ga}_vs_{gb}'], _, _ = multipletests(df[col], method='fdr_bh')

    return df.sort_values('kw_padj')


def plot_umap_markers(adata, markers, title_prefix='', layer='arcsinh', ncols=3, aliases_dict=None):
    """Plot UMAP for each marker individually with proper titles and common names."""
    import json

    # Load aliases if not provided
    if aliases_dict is None:
        with open('marker_panels.json') as f:
            marker_panels = json.load(f)
        aliases_dict = marker_panels.get('marker_aliases', {})

    n_markers = len(markers)
    n_rows = (n_markers + ncols - 1) // ncols

    fig, axes = plt.subplots(n_rows, ncols, figsize=(5*ncols, 4*n_rows))
    if n_rows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, marker in enumerate(markers):
        ax = axes[i]
        sc.pl.umap(adata, color=marker, layer=layer, ax=ax, show=False)
        # Get common name from aliases, or use marker name if not found
        common_name = aliases_dict.get(marker, marker)
        title = f'{marker}\n({common_name})' if common_name != marker else marker
        ax.set_title(title, fontsize=9, fontweight='bold')

    # Hide unused subplots
    for i in range(n_markers, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title_prefix, fontsize=12, y=1.00)
    plt.tight_layout()
    plt.show()


def plot_da_bars(adata, cell_types, sys_a, sys_b, aliases_dict=None):
    """Plot differential abundance for cell types between two systems."""
    import json

    # Load aliases if not provided
    if aliases_dict is None:
        with open('marker_panels.json') as f:
            marker_panels = json.load(f)
        aliases_dict = marker_panels.get('marker_aliases', {})

    def format_marker_label(marker):
        """Format marker name with common name alias."""
        common_name = aliases_dict.get(marker, marker)
        return f'{marker}\n({common_name})' if common_name != marker else marker

    n_types = len(cell_types)
    fig, axes = plt.subplots(2, n_types, figsize=(10*n_types, 14))
    if n_types == 1:
        axes = axes.reshape(2, 1)

    for col, cell_type in enumerate(cell_types):
        sub = adata[adata.obs['cell_type_annot'] == cell_type].copy()
        da = compute_da(sub, sys_a, sys_b)

        top_a = da.nlargest(10, 'mean_diff').sort_values('mean_diff')
        top_b = da.nsmallest(10, 'mean_diff').sort_values('mean_diff')

        for row, (top, color, title_sys) in enumerate([
            (top_b, '#1f77b4', sys_b),
            (top_a, '#d62728', sys_a),
        ]):
            ax = axes[row, col]
            # Format marker labels with common names
            formatted_labels = [format_marker_label(m) for m in top['marker']]
            ax.barh(formatted_labels, top['mean_diff'].values, color=color, alpha=0.8)
            ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_xlabel('Mean diff in arcsinh  (A − B)')
            ax.set_title(f'{cell_type}  –  top 10 higher in\n{title_sys.split(" + ")[0]}')

            for i, (_, row_) in enumerate(top.iterrows()):
                sig = sig_label(row_['padj'])
                x_pos = row_['mean_diff'] + (0.02 if row_['mean_diff'] >= 0 else -0.02)
                ha = 'left' if row_['mean_diff'] >= 0 else 'right'
                ax.text(x_pos, i, sig, va='center', ha=ha, fontsize=8)

    plt.tight_layout()
    plt.show()

    # Print ranked lists
    for cell_type in cell_types:
        sub = adata[adata.obs['cell_type_annot'] == cell_type].copy()
        da = compute_da(sub, sys_a, sys_b)

        top_a = da.nlargest(10, 'mean_diff')[['marker', 'mean_diff', 'padj']]
        top_b = da.nsmallest(10, 'mean_diff')[['marker', 'mean_diff', 'padj']]

        print(f'{"="*55}')
        print(f'  {cell_type}  —  higher in {sys_a.split(" + ")[0]}')
        print(f'{"="*55}')
        print(top_a.to_string(index=False))
        print()
        print(f'{"="*55}')
        print(f'  {cell_type}  —  higher in {sys_b.split(" + ")[0]}')
        print(f'{"="*55}')
        print(top_b.to_string(index=False))
        print()


def display_name(marker):
    """Return 'CD279 (PD-1)' style label; identity if alias == marker."""
    import json
    with open('marker_panels.json') as f:
        marker_panels = json.load(f)
    aliases = marker_panels.get('marker_aliases', {})
    alias = aliases.get(marker)
    if alias and alias != marker:
        return f"{marker} ({alias})"
    return marker


def plot_marker_panel_violins(
    adata,
    panel_name,
    group_key,
    layer="arcsinh",
    adata_compare=None,
    compare_label="Comparison",
    primary_label="Primary",
):
    """Violin plots for a marker panel — one figure per subcategory.

    When *adata_compare* is provided, both datasets are shown side-by-side on
    the same axes using ``hue`` (split violins), with different colour palettes.
    """
    from scipy.stats import kruskal

    panel = load_marker_panel_grouped(panel_name)
    available = set(adata.var_names)
    groups = sorted(adata.obs[group_key].unique())
    display_order = groups[::-1]

    X_pri = pd.DataFrame(
        np.array(adata.layers[layer], dtype=np.float32),
        index=adata.obs_names, columns=adata.var_names,
    )
    has_cmp = adata_compare is not None
    if has_cmp:
        X_cmp = pd.DataFrame(
            np.array(adata_compare.layers[layer], dtype=np.float32),
            index=adata_compare.obs_names, columns=adata_compare.var_names,
        )

    hue_order = [primary_label, compare_label] if has_cmp else None
    hue_palette = {primary_label: "#4C72B0", compare_label: "#DD8452"} if has_cmp else None

    # Filter panel to available markers, rank by KW significance, keep top 5
    top_per_cat = 5
    panel_filtered = {}
    for cat, markers in panel.items():
        present = [m for m in markers if m in available]
        if not present:
            continue
        kw_pvals = {}
        for m in present:
            arrays = [X_pri.loc[adata.obs[group_key] == g, m].values for g in groups]
            if all(len(a) > 0 for a in arrays):
                _, pval = kruskal(*arrays)
            else:
                pval = 1.0
            kw_pvals[m] = pval
        ranked = sorted(present, key=lambda m: kw_pvals[m])
        panel_filtered[cat] = ranked[:top_per_cat]

    # ── One figure per subcategory ──────────────────────────────────────────
    for cat, markers in panel_filtered.items():
        n_cols = len(markers)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6), squeeze=False)

        for col_i, marker in enumerate(markers):
            ax = axes[0][col_i]

            # Build combined DataFrame
            df_pri = pd.DataFrame({
                "expression": X_pri[marker].values,
                "group": adata.obs[group_key].values,
                "system": primary_label,
            })
            if has_cmp:
                df_cmp = pd.DataFrame({
                    "expression": X_cmp[marker].values,
                    "group": adata_compare.obs[group_key].values,
                    "system": compare_label,
                })
                df_v = pd.concat([df_pri, df_cmp], ignore_index=True)
            else:
                df_v = df_pri

            sns.violinplot(
                data=df_v, x="group", y="expression", ax=ax,
                hue="system" if has_cmp else None,
                hue_order=hue_order,
                palette=hue_palette,
                cut=0, inner="quartile", density_norm="width",
                order=display_order, split=has_cmp,
            )
            ax.set_xlabel("")
            ax.set_ylabel(layer if col_i == 0 else "")
            ax.tick_params(axis="x", labelsize=9, rotation=20)

            # KW p-values for both datasets
            kw_parts = []
            for lbl, X_df, ad in [(primary_label, X_pri, adata)] + (
                [(compare_label, X_cmp, adata_compare)] if has_cmp else []
            ):
                arrays = [X_df.loc[ad.obs[group_key] == g, marker].values for g in groups]
                if all(len(a) > 0 for a in arrays):
                    _, kw_pval = kruskal(*arrays)
                else:
                    kw_pval = 1.0
                kw_parts.append(f"{lbl}: {sig_label(kw_pval)}")
            kw_text = "  |  ".join(kw_parts)

            ax.set_title(
                f"{display_name(marker)}\nKW  {kw_text}",
                fontsize=11, fontweight="bold",
            )

            # Only show legend on first subplot
            if col_i == 0 and has_cmp:
                ax.legend(fontsize=8, loc="upper right")
            elif ax.get_legend() is not None:
                ax.get_legend().remove()

        fig.suptitle(cat, fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.show()

    # ── Print DA summary per subcategory ───────────────────────────────────
    from itertools import combinations

    groups = sorted(adata.obs[group_key].unique())
    display_order = groups[::-1]
    pairs = list(combinations(groups, 2))

    datasets = [(primary_label, adata)]
    if has_cmp:
        datasets.append((compare_label, adata_compare))

    for cat, markers in panel_filtered.items():
        print(f"\n{'#' * 70}")
        print(f"  {cat}")
        print(f"{'#' * 70}")
        cat_markers = set(markers)

        # 1 & 2: Intra-system pairwise comparisons
        for lbl, ad in datasets:
            da = compute_da_multigroup(ad, group_key, layer=layer)
            da_cat = da[da["marker"].isin(cat_markers)].copy()

            print(f"\n{'=' * 60}")
            print(f"  [{lbl}] Intra-system comparisons")
            print(f"{'=' * 60}")

            for ga, gb in pairs:
                diff_col = f"diff_{ga}_vs_{gb}"
                padj_col = f"padj_{ga}_vs_{gb}"
                if diff_col not in da_cat.columns:
                    continue
                sub = da_cat[["marker", diff_col, padj_col]].copy()
                sub.columns = ["marker", "mean_diff", "padj"]
                sig = sub[sub["padj"] < 0.05].sort_values("padj")
                if len(sig) == 0:
                    print(f"\n  {ga} vs {gb}: no significant markers")
                    continue
                print(f"\n  {ga} vs {gb}  —  {len(sig)} significant (FDR < 0.05):")
                sig_out = sig[["marker", "mean_diff", "padj"]].copy()
                sig_out["marker"] = sig_out["marker"].map(display_name)
                print(sig_out.to_string(index=False))

        # 3: Between-system comparisons (per condition)
        if has_cmp:
            print(f"\n{'=' * 60}")
            print(f"  Between systems: {primary_label} vs {compare_label}")
            print(f"{'=' * 60}")

            X_p = X_pri[list(cat_markers)]
            X_c = X_cmp[list(cat_markers)]
            grp_p = adata.obs[group_key].values
            grp_c = adata_compare.obs[group_key].values

            for g in display_order:
                vals_p = X_p.loc[grp_p == g]
                vals_c = X_c.loc[grp_c == g]
                if len(vals_p) == 0 or len(vals_c) == 0:
                    print(f"\n  {g}: insufficient data")
                    continue

                results = []
                for m in markers:
                    vp = vals_p[m].values
                    vc = vals_c[m].values
                    if len(vp) > 0 and len(vc) > 0:
                        _, pval = mannwhitneyu(vp, vc, alternative="two-sided")
                        results.append({
                            "marker": m,
                            "mean_diff": vp.mean() - vc.mean(),
                            "pval": pval,
                        })
                res_df = pd.DataFrame(results)
                if len(res_df) == 0:
                    continue
                _, res_df["padj"], _, _ = multipletests(res_df["pval"], method="fdr_bh")
                sig = res_df[res_df["padj"] < 0.05].sort_values("padj")
                if len(sig) == 0:
                    print(f"\n  {g}: no significant markers")
                    continue
                print(f"\n  {g}  —  {len(sig)} significant (FDR < 0.05):")
                sig_out = sig[["marker", "mean_diff", "padj"]].copy()
                sig_out["marker"] = sig_out["marker"].map(display_name)
                print(f"    (positive = higher in {primary_label})")
                print(sig_out.to_string(index=False))


def plot_top_coloc_partners(sp_sub, markers, title, color, n_top=10):
    """Plot top N colocalization partners for each marker (bar charts)."""
    fig, axes = plt.subplots(1, len(markers), figsize=(6*len(markers), 5), sharey=False)
    if len(markers) == 1:
        axes = [axes]

    for ax, marker in zip(axes, markers):
        cols = get_marker_cols(sp_sub.columns, marker)
        if not cols:
            ax.set_title(f'{marker}\n(not found)')
            continue

        means = sp_sub[cols].mean()
        top = means.nlargest(n_top)
        partners_raw = top.index.map(lambda c: c.replace(marker + '/', '').replace('/' + marker, ''))
        partners = [display_name(p) for p in partners_raw]

        print(f'{"=" * 45}')
        print(f"  {display_name(marker)}  —  top {n_top} neighbors ({title})")
        print(f'{"=" * 45}')
        tbl = pd.DataFrame({"partner": partners, "mean_coloc": top.values})
        print(tbl.to_string(index=False))
        print()

        ax.barh(partners[::-1], top.values[::-1], color=color, alpha=0.8)
        ax.set_xlabel('Mean colocalization (arcsinh)')
        ax.set_title(f'{display_name(marker)}\n({title})')
        ax.tick_params(axis='y', labelsize=8)

    plt.suptitle(f'Top {n_top} colocalization partners — {title}', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_diff_coloc(sp_a, sp_b, markers, all_cols, sys_a_label='System A', sys_b_label='System B'):
    """Plot differential colocalization partners per marker (2×n grid)."""
    fig, axes = plt.subplots(2, len(markers), figsize=(6*len(markers), 10), sharey=False)
    if len(markers) == 1:
        axes = axes.reshape(2, 1)

    for col_i, marker in enumerate(markers):
        cols = get_marker_cols(all_cols, marker)
        if not cols:
            for row_i in range(2):
                axes[row_i, col_i].set_title(f'{marker}\n(not found)')
            continue

        vals_a = sp_a[cols]
        vals_b = sp_b[cols]

        results = []
        for c in cols:
            a = vals_a[c].values
            b = vals_b[c].values
            md = b.mean() - a.mean()
            _, pval = mannwhitneyu(a, b, alternative='two-sided')
            partner_raw = c.replace(marker + '/', '').replace('/' + marker, '')
            results.append({'partner': display_name(partner_raw), 'mean_diff': md, 'pval': pval})

        df = pd.DataFrame(results)
        _, df['padj'], _, _ = multipletests(df['pval'], method='fdr_bh')

        top_a = df.nlargest(10, 'mean_diff').sort_values('mean_diff')
        top_b = df.nsmallest(10, 'mean_diff').sort_values('mean_diff')

        print(f'{"=" * 50}')
        print(f"  {display_name(marker)}  —  higher in {sys_b_label}")
        print(f'{"=" * 50}')
        print(top_a[["partner", "mean_diff", "padj"]].to_string(index=False))
        print()
        print(f'{"=" * 50}')
        print(f"  {display_name(marker)}  —  higher in {sys_a_label}")
        print(f'{"=" * 50}')
        print(top_b[["partner", "mean_diff", "padj"]].to_string(index=False))
        print()

        for row_i, (top, color, sys_label) in enumerate([
            (top_b, '#1f77b4', sys_a_label),
            (top_a, '#d62728', sys_b_label),
        ]):
            ax = axes[row_i, col_i]
            ax.barh(top['partner'], top['mean_diff'], color=color, alpha=0.8)
            ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_xlabel('Mean diff (arcsinh)', fontsize=8)
            ax.set_title(f'{display_name(marker)}\nhigher in {sys_label}', fontsize=9)
            ax.tick_params(axis='y', labelsize=7)

            for i, (_, row_) in enumerate(top.iterrows()):
                sig = sig_label(row_['padj'])
                x_pos = row_['mean_diff'] + (0.01 if row_['mean_diff'] >= 0 else -0.01)
                ha = 'left' if row_['mean_diff'] >= 0 else 'right'
                ax.text(x_pos, i, sig, va='center', ha=ha, fontsize=7)

    plt.suptitle(f'Top 10 differential partners — {sys_a_label} vs {sys_b_label}', y=1.01)
    plt.tight_layout()
    plt.show()


def plot_clustermaps(matrices, row_linkage, title_suffix='', figsize_factor=0.18):
    """Plot multiple clustermaps with shared dendrogram."""
    n = len(matrices[0][0])  # from first matrix
    fig_s = max(10, n * figsize_factor)
    tk_fs = max(4, min(7, 180 // n))

    for mat, label, cmap in matrices:
        vmin = -mat.abs().max().max()
        vmax = mat.abs().max().max()

        g = sns.clustermap(
            mat, cmap=cmap, center=0, vmin=vmin, vmax=vmax,
            row_linkage=row_linkage, col_linkage=row_linkage,
            figsize=(fig_s, fig_s), linewidths=0,
            xticklabels=True, yticklabels=True,
            cbar_kws={'shrink': 0.4, 'label': label},
            dendrogram_ratio=0.08,
            cbar_pos=(0.02, 0.82, 0.03, 0.15),
        )
        g.ax_heatmap.tick_params(axis='both', labelsize=tk_fs)
        g.fig.suptitle(f'{label}\n{title_suffix}', fontsize=12, y=1.01)
        plt.show()


def plot_clusters(linkage_obj, markers, mat_dict, n_clust_list=[3, 5, 7]):
    """Print cluster membership at multiple k values with within-cluster colocalization."""
    for n_clust in n_clust_list:
        labels = fcluster(linkage_obj, t=n_clust, criterion='maxclust')
        cluster_map = pd.DataFrame({
            'protein': markers,
            'cluster': labels,
        }).sort_values(['cluster', 'protein'])

        print(f'\n{"="*60}')
        print(f'  {n_clust} clusters (Ward linkage)')
        print(f'{"="*60}')

        for c in sorted(cluster_map['cluster'].unique()):
            members = cluster_map[cluster_map['cluster'] == c]['protein'].tolist()
            print(f'\n  Cluster {c}  (n={len(members)}):')

            for name, mat in mat_dict.items():
                if len(members) > 1:
                    sub = mat.loc[members, members]
                    mask = np.triu(np.ones(sub.shape, dtype=bool), k=1)
                    mean_coloc = sub.values[mask].mean()
                    print(f'    {name} within-cluster coloc: {mean_coloc:.3f}', end='  ')
                else:
                    print(f'    {name} within-cluster coloc: 0.0', end='  ')
            print()
            print(f'    {", ".join(members)}')


def plot_networks(mat_a, mat_b, pos, title_a='System A', title_b='System B',
                  color_a='#1f77b4', color_b='#d62728', top_edges=60):
    """Side-by-side network graphs with shared layout."""
    def build_graph(mat, top_n):
        G = nx.Graph()
        G.add_nodes_from(mat.index)
        edges = [(a, b, mat.loc[a, b])
                 for i, a in enumerate(mat.index)
                 for j, b in enumerate(mat.columns)
                 if j > i and mat.loc[a, b] > 0]
        edges.sort(key=lambda x: x[2], reverse=True)
        for a, b, w in edges[:top_n]:
            G.add_edge(a, b, weight=w)
        return G

    G_a = build_graph(mat_a, top_edges)
    G_b = build_graph(mat_b, top_edges)

    def draw_net(ax, G, title, base_color):
        if G.number_of_edges() == 0:
            ax.set_title(title); ax.axis('off'); return
        weights  = np.array([G[u][v]['weight'] for u, v in G.edges()])
        wmax     = weights.max()
        widths   = 0.5 + 4.5 * (weights / wmax)
        alphas   = 0.2 + 0.7 * (weights / wmax)
        node_deg = dict(G.degree(weight='weight'))
        deg_max  = max(node_deg.values()) if node_deg else 1
        node_size = [200 + 2000 * node_deg.get(nd, 0) / deg_max for nd in G.nodes()]
        node_col  = ['#e84545' if nd in ['CD20', 'CD19', 'IgM'] else base_color for nd in G.nodes()]

        for (u, v), w, a in zip(G.edges(), widths, alphas):
            ax.plot(*zip(pos[u], pos[v]), color=base_color, linewidth=w, alpha=a)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=node_col, alpha=0.85)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    draw_net(axes[0], G_a, f'{title_a}  (top {top_edges} pairs)', color_a)
    draw_net(axes[1], G_b, f'{title_b}  (top {top_edges} pairs)', color_b)
    fig.suptitle('Colocalization networks\n(node size = weighted degree, hub markers in red)',
                 fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_diff_network(mat_diff, pos, quantile=0.80, title=''):
    """Plot differential network (only top % changed edges)."""
    thresh = mat_diff.abs().stack().quantile(quantile)

    G = nx.Graph()
    G.add_nodes_from(mat_diff.index)
    for i, a in enumerate(mat_diff.index):
        for j, b in enumerate(mat_diff.columns):
            if j <= i: continue
            d = mat_diff.loc[a, b]
            if abs(d) >= thresh:
                G.add_edge(a, b, diff=d)

    isolated = [nd for nd in G.nodes() if G.degree(nd) == 0]
    G.remove_nodes_from(isolated)
    active_pos = {nd: pos[nd] for nd in G.nodes()}

    diffs = np.array([G[u][v]['diff'] for u, v in G.edges()])
    dmax = np.abs(diffs).max() if len(diffs) else 1
    widths = 1 + 5 * np.abs(diffs) / dmax
    colors = ['#d62728' if d > 0 else '#1f77b4' for d in diffs]

    fig, ax = plt.subplots(figsize=(12, 9))
    for (u, v), w, c in zip(G.edges(), widths, colors):
        ax.plot(*zip(active_pos[u], active_pos[v]), color=c, linewidth=w, alpha=0.7)

    node_col = ['#e84545' if nd in ['CD20', 'CD19', 'IgM'] else '#aaaaaa' for nd in G.nodes()]
    nx.draw_networkx_nodes(G, active_pos, ax=ax, node_color=node_col, node_size=300, alpha=0.9)
    nx.draw_networkx_labels(G, active_pos, ax=ax, font_size=7)

    patch_a = mpatches.Patch(color='#d62728', label='System A')
    patch_b = mpatches.Patch(color='#1f77b4', label='System B')
    ax.legend(handles=[patch_a, patch_b], loc='lower right', fontsize=10)
    ax.set_title(f'Differential network (|diff| ≥ {thresh:.3f}, top {int((1-quantile)*100)}%)\n{title}',
                 fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def run_spatial_comparison(
    marker: str,
    sp_a: pd.DataFrame,
    sp_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    color_a: str = "#d62728",
    color_b: str = "#1f77b4",
) -> None:
    """Run a full spatial colocalization comparison for a single marker.

    Shows top colocalization partners in each group, then differential
    (top 10 both directions).
    """
    print(f'\n{"=" * 70}')
    print(f"  {display_name(marker)}  :  {label_a}  vs  {label_b}")
    print(f'{"=" * 70}\n')

    all_cols = sp_a.columns

    plot_top_coloc_partners(sp_a, [marker], label_a, color_a)
    plot_top_coloc_partners(sp_b, [marker], label_b, color_b)
    plot_diff_coloc(sp_a, sp_b, [marker], all_cols,
                    sys_a_label=label_a, sys_b_label=label_b)
