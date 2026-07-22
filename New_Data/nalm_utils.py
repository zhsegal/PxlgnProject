"""Reusable utilities for NALM-6 vs healthy B colocalization analysis."""
from __future__ import annotations

import json as _json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
from scipy.stats import mannwhitneyu
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests

__all__ = [
    # helpers
    "sig_label", "get_marker_cols", "split_pair", "extract_partner", "split_triplet",
    "load_marker_panel", "load_marker_panel_grouped", "display_name",
    # statistics
    "compute_da", "compute_da_multigroup", "compute_marker_diff_coloc",
    "compute_triplet_diff",
    # scoring
    "compute_b_cell_score",
    # matrix & clustering
    "build_mean_matrix", "get_pairwise_cols", "compute_ward_linkage",
    "select_top_partners",
    # plotting
    "plot_raw_vs_arcsinh",
    "plot_da_bars", "plot_top_coloc_partners",
    "plot_diff_coloc", "plot_neighborhood_suite", "draw_force_net",
    "run_spatial_comparison",
    "plot_marker_panel_violins",
    "plot_lfc_scatter", "plot_synapse_suite",
    "plot_condition_comparison",
    "plot_ma_blina_vs_mock", "plot_delta_lfc_ma",
    # printing
    "print_da_results", "print_cluster_analysis",
]

# Default path for marker panel JSON
_MARKER_PANEL_PATH = Path(__file__).parent / "marker_panels.json"

# Lazy-loaded alias cache
_ALIASES: dict[str, str] | None = None


def _load_aliases() -> dict[str, str]:
    global _ALIASES
    if _ALIASES is None:
        with open(_MARKER_PANEL_PATH) as f:
            data = _json.load(f)
        _ALIASES = data.get("marker_aliases", {})
    return _ALIASES


def display_name(marker: str) -> str:
    """Return 'CD279 (PD-1)' style label; identity if alias == marker."""
    aliases = _load_aliases()
    alias = aliases.get(marker)
    if alias and alias != marker:
        return f"{marker} ({alias})"
    return marker


# ── 1. Helpers ───────────────────────────────────────────────────────────────

def load_marker_panel(panel_name: str, path: str | Path | None = None) -> list[str]:
    """Load a marker panel from the JSON file and return a flat list of markers.

    Parameters
    ----------
    panel_name : str
        Top-level key in the JSON file (e.g. 'b_cell_markers').
    path : str or Path, optional
        Path to JSON file. Defaults to ``marker_panels.json`` next to this module.

    Returns
    -------
    list[str]
        Flat, deduplicated list of marker names.
    """
    p = Path(path) if path else _MARKER_PANEL_PATH
    with open(p) as f:
        panels = _json.load(f)
    panel = panels[panel_name]
    # Flatten if dict of categories
    if isinstance(panel, dict):
        markers = []
        for group in panel.values():
            markers.extend(group)
        return list(dict.fromkeys(markers))  # deduplicate, preserve order
    return list(panel)


def sig_label(padj: float) -> str:
    """Convert adjusted p-value to significance stars."""
    if padj < 0.001:
        return "***"
    if padj < 0.01:
        return "**"
    if padj < 0.05:
        return "*"
    return "ns"


def get_marker_cols(columns, marker: str) -> list[str]:
    """Return pair columns that include *marker* on either side of '/'."""
    return [c for c in columns if marker in c.split("/")]


def split_pair(pair_name: str, sep: str = "/") -> tuple[str, str]:
    """Split a 'A/B' pair column name into ('A', 'B')."""
    return tuple(pair_name.split(sep))


def extract_partner(col: str, marker: str) -> str:
    """Extract the partner protein name from a pair column like 'CD20/CD3e'."""
    return col.replace(marker + "/", "").replace("/" + marker, "")


def split_triplet(triplet_name: str, sep: str = "/") -> tuple[str, str, str]:
    """Split an 'A/B/C' triplet column name into ('A', 'B', 'C')."""
    return tuple(triplet_name.split(sep))


# ── 2. Statistical functions ─────────────────────────────────────────────────

def compute_da(adata_sub, sys_a: str, sys_b: str, layer: str = "arcsinh") -> pd.DataFrame:
    """Mann-Whitney U + BH-FDR differential abundance between two cell systems.

    Parameters
    ----------
    adata_sub : AnnData
        Subset with ``obs['cell_system']`` containing *sys_a* and *sys_b*.
    sys_a, sys_b : str
        Cell system labels to compare.
    layer : str
        Layer to use for expression values.

    Returns
    -------
    pd.DataFrame
        Columns: marker, mean_diff (sys_a − sys_b), pval, padj.
    """
    X = np.array(adata_sub.layers[layer], dtype=np.float32)
    markers = list(adata_sub.var_names)
    grp = adata_sub.obs["cell_system"]
    a_mask = (grp == sys_a).values
    b_mask = (grp == sys_b).values

    results = []
    for idx, marker in enumerate(markers):
        a_vals = X[a_mask, idx]
        b_vals = X[b_mask, idx]
        mean_diff = a_vals.mean() - b_vals.mean()
        _, pval = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
        results.append({"marker": marker, "mean_diff": mean_diff, "pval": pval})

    df = pd.DataFrame(results)
    _, df["padj"], _, _ = multipletests(df["pval"], method="fdr_bh")
    return df


def compute_da_multigroup(
    adata_sub,
    group_key: str,
    layer: str = "arcsinh",
) -> pd.DataFrame:
    """Kruskal-Wallis + pairwise Mann-Whitney for multi-group DA.

    Parameters
    ----------
    adata_sub : AnnData
        Subset containing the groups to compare.
    group_key : str
        obs column defining the groups.
    layer : str
        Layer for expression values.

    Returns
    -------
    pd.DataFrame
        Columns: marker, kw_stat, kw_pval, kw_padj, plus per-pair columns.
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
        row = {"marker": marker}

        # Kruskal-Wallis across all groups
        arrays = [vals[g] for g in group_names]
        if all(len(a) > 0 for a in arrays):
            stat, pval = kruskal(*arrays)
        else:
            stat, pval = np.nan, 1.0
        row["kw_stat"] = stat
        row["kw_pval"] = pval

        # Group means
        for g in group_names:
            row[f"mean_{g}"] = vals[g].mean() if len(vals[g]) > 0 else np.nan

        # Pairwise Mann-Whitney
        for ga, gb in pairs:
            if len(vals[ga]) > 0 and len(vals[gb]) > 0:
                _, pw = mannwhitneyu(vals[ga], vals[gb], alternative="two-sided")
                row[f"pval_{ga}_vs_{gb}"] = pw
                row[f"diff_{ga}_vs_{gb}"] = vals[ga].mean() - vals[gb].mean()
            else:
                row[f"pval_{ga}_vs_{gb}"] = 1.0
                row[f"diff_{ga}_vs_{gb}"] = 0.0

        results.append(row)

    df = pd.DataFrame(results)
    _, df["kw_padj"], _, _ = multipletests(df["kw_pval"], method="fdr_bh")

    # Also correct pairwise p-values
    for ga, gb in pairs:
        col = f"pval_{ga}_vs_{gb}"
        _, df[f"padj_{ga}_vs_{gb}"], _, _ = multipletests(df[col], method="fdr_bh")

    return df.sort_values("kw_padj")


def compute_marker_diff_coloc(
    sp_a: pd.DataFrame,
    sp_b: pd.DataFrame,
    columns: list[str],
    marker: str,
) -> pd.DataFrame:
    """Mann-Whitney U test on colocalization pairs for *marker* between two groups.

    Parameters
    ----------
    sp_a, sp_b : pd.DataFrame
        Colocalization data for the two groups (cells × pairs).
    columns : list[str]
        Pair columns to test (should all involve *marker*).
    marker : str
        Reference protein (used to extract partner names).

    Returns
    -------
    pd.DataFrame
        Columns: pair, partner, mean_a, mean_b, mean_diff, pval, padj.
        Sorted by padj ascending.
    """
    results = []
    for c in columns:
        partner = extract_partner(c, marker)
        vals_a = sp_a[c].values
        vals_b = sp_b[c].values
        mean_diff = vals_a.mean() - vals_b.mean()
        _, pval = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        results.append({
            "pair": c,
            "partner": partner,
            "mean_a": vals_a.mean(),
            "mean_b": vals_b.mean(),
            "mean_diff": mean_diff,
            "pval": pval,
        })

    df = pd.DataFrame(results)
    _, df["padj"], _, _ = multipletests(df["pval"], method="fdr_bh")
    return df.sort_values("padj")


def compute_triplet_diff(
    sp_a: pd.DataFrame,
    sp_b: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Mann-Whitney U + BH-FDR on 3-way colocalization triplets between two groups.

    Parameters
    ----------
    sp_a, sp_b : pd.DataFrame
        Triplet colocalization data per group (cells × triplets), e.g. a per-group
        subset of ``adata.obsm['triplet_determinant_asinh']``.
    columns : list[str], optional
        Triplet columns to test. Defaults to the columns shared by both frames.

    Returns
    -------
    pd.DataFrame
        Columns: triplet, mean_a, mean_b, mean_diff, pval, padj. Sorted by padj.
    """
    if columns is None:
        cols_b = set(sp_b.columns)
        columns = [c for c in sp_a.columns if c in cols_b]

    results = []
    for c in columns:
        vals_a = sp_a[c].values
        vals_b = sp_b[c].values
        _, pval = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        results.append({
            "triplet": c,
            "mean_a": vals_a.mean(),
            "mean_b": vals_b.mean(),
            "mean_diff": vals_a.mean() - vals_b.mean(),
            "pval": pval,
        })

    df = pd.DataFrame(results)
    _, df["padj"], _, _ = multipletests(df["pval"], method="fdr_bh")
    return df.sort_values("padj")


def compute_b_cell_score(
    adata,
    b_markers: list[str],
    layer: str = "arcsinh",
    obs_key: str = "b_cell_score",
) -> None:
    """Compute B cell score = sum(B markers) / sum(all markers) per cell.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    b_markers : list[str]
        B cell marker names (must be in ``adata.var_names``).
    layer : str
        Layer to use for expression values.
    obs_key : str
        Key to store the score in ``adata.obs``.
    """
    X = np.array(adata.layers[layer], dtype=np.float32)
    var_names = list(adata.var_names)
    b_idx = [var_names.index(m) for m in b_markers if m in var_names]
    missing = [m for m in b_markers if m not in var_names]
    if missing:
        print(f"Warning: markers not found in var_names: {missing}")

    b_sum = X[:, b_idx].sum(axis=1)
    total = X.sum(axis=1)
    total = np.where(total == 0, 1.0, total)  # avoid division by zero
    adata.obs[obs_key] = b_sum / total


# ── 3. Matrix & clustering operations ────────────────────────────────────────

def build_mean_matrix(
    sp_sub: pd.DataFrame,
    pair_cols: list[str],
    markers: list[str],
    sep: str = "/",
) -> pd.DataFrame:
    """Build a symmetric marker × marker mean colocalization matrix."""
    mat = pd.DataFrame(0.0, index=markers, columns=markers)
    means = sp_sub[pair_cols].mean()
    for pair_name, val in means.items():
        a, b = split_pair(pair_name, sep)
        mat.loc[a, b] = val
        mat.loc[b, a] = val
    return mat


def get_pairwise_cols(
    columns, proteins: set, sep: str = "/"
) -> list[str]:
    """Filter pair columns to those where both proteins are in *proteins*."""
    return [
        c for c in columns
        if len(c.split(sep)) == 2
        and c.split(sep)[0] in proteins
        and c.split(sep)[1] in proteins
    ]


def compute_ward_linkage(mat_diff: pd.DataFrame) -> np.ndarray:
    """Compute Ward hierarchical linkage from a difference matrix.

    Uses distance = max(|values|) − |values| + eps so that large absolute
    differences cluster together.
    """
    dist = mat_diff.abs().max().max() - mat_diff.abs() + 1e-10
    np.fill_diagonal(dist.values, 0)
    condensed = squareform(dist.values, checks=False)
    return linkage(condensed, method="ward")


def select_top_partners(
    sp_data: pd.DataFrame,
    marker: str,
    columns,
    n: int = 30,
) -> list[str]:
    """Select top-N colocalization partners for *marker* by mean value.

    Parameters
    ----------
    sp_data : pd.DataFrame
        Colocalization data (cells × pairs).
    marker : str
        Reference protein.
    columns : Index or list
        All available pair columns.
    n : int
        Number of partners to return.

    Returns
    -------
    list[str]
        Partner protein names (not column names).
    """
    marker_cols = get_marker_cols(columns, marker)
    means = sp_data[marker_cols].mean()
    top_cols = means.nlargest(n).index.tolist()
    return [extract_partner(c, marker) for c in top_cols]


# ── 4. Plotting functions ────────────────────────────────────────────────────


def plot_raw_vs_arcsinh(
    adata,
    markers: list[str],
    group_key: str = "cell_system",
    groups: list[tuple[str, str]] | None = None,
) -> None:
    """Plot side-by-side histograms of raw counts vs arcsinh for each marker.

    Parameters
    ----------
    adata : AnnData
        Annotated data with 'raw' and 'arcsinh' layers.
    markers : list[str]
        Marker names to plot.
    group_key : str
        obs column to split populations.
    groups : list of (group_value, color)
        Groups to overlay. If None, uses all unique values with default colors.
    """
    if groups is None:
        unique_groups = adata.obs[group_key].unique().tolist()
        default_colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
        groups = [(g, default_colors[i % len(default_colors)])
                  for i, g in enumerate(unique_groups)]

    fig, axes = plt.subplots(len(markers), 2, figsize=(14, 4 * len(markers)))
    if len(markers) == 1:
        axes = axes[np.newaxis, :]

    for i, marker in enumerate(markers):
        midx = list(adata.var_names).index(marker)

        for ax, layer, xlabel in [
            (axes[i, 0], "raw", "Raw count"),
            (axes[i, 1], "arcsinh", "arcsinh(count)"),
        ]:
            for group_val, color in groups:
                mask = adata.obs[group_key] == group_val
                vals = adata[mask].layers[layer][:, midx]
                ax.hist(vals, bins=50, alpha=0.5, color=color,
                        label=group_val, density=True)
            ax.set_title(f"{display_name(marker)} — {layer}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_da_bars(
    adata,
    cell_types: list[str],
    sys_a: str,
    sys_b: str,
    layer: str = "arcsinh",
    top_n: int = 10,
) -> plt.Figure:
    """Plot differential abundance bar charts (2 × len(cell_types) grid).

    Row 0: markers higher in *sys_b* (blue).
    Row 1: markers higher in *sys_a* (red).
    """
    fig, axes = plt.subplots(2, len(cell_types), figsize=(7 * len(cell_types), 12))
    if len(cell_types) == 1:
        axes = axes[:, np.newaxis]

    for col, cell_type in enumerate(cell_types):
        sub = adata[adata.obs["cell_type_annot"] == cell_type].copy()
        da = compute_da(sub, sys_a, sys_b, layer=layer)
        print_da_results(da, cell_type, sys_a, sys_b, top_n=top_n)

        top_a = da.nlargest(top_n, "mean_diff").sort_values("mean_diff")
        top_b = da.nsmallest(top_n, "mean_diff").sort_values("mean_diff")

        for row, (top, color, title_sys) in enumerate([
            (top_b, "#1f77b4", sys_b),
            (top_a, "#d62728", sys_a),
        ]):
            ax = axes[row][col]
            labels = [display_name(m) for m in top["marker"]]
            ax.barh(labels, top["mean_diff"], color=color, alpha=0.8)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel(f"Mean diff in {layer}  ({sys_a}  −  {sys_b})")
            ax.set_title(f"{cell_type}  –  top {top_n} higher in\n{title_sys}")

            for i, (_, r) in enumerate(top.iterrows()):
                s = sig_label(r["padj"])
                x_pos = r["mean_diff"] + (0.02 if r["mean_diff"] >= 0 else -0.02)
                ha = "left" if r["mean_diff"] >= 0 else "right"
                ax.text(x_pos, i, s, va="center", ha=ha, fontsize=8)

    plt.tight_layout()
    plt.show()



def plot_top_coloc_partners(
    sp_data: pd.DataFrame,
    markers: list[str],
    system_label: str,
    color: str,
    top_n: int = 10,
) -> plt.Figure:
    """Bar plots of top-N colocalization partners per marker.

    Parameters
    ----------
    sp_data : pd.DataFrame
        Colocalization data for one group (cells × pairs).
    markers : list[str]
        Reference proteins to analyse.
    system_label : str
        Label for the cell system (used in titles and suptitle).
    color : str
        Bar color.
    top_n : int
        Number of top partners to show.
    """
    fig, axes = plt.subplots(1, len(markers), figsize=(18, 5), sharey=False)
    if len(markers) == 1:
        axes = [axes]

    for ax, bm in zip(axes, markers):
        cols = get_marker_cols(sp_data.columns, bm)
        if not cols:
            ax.set_title(f"{bm}\n(not found)")
            continue

        means = sp_data[cols].mean()
        top = means.nlargest(top_n)
        partners = top.index.map(lambda c: extract_partner(c, bm))
        partner_labels = [display_name(p) for p in partners]

        print(f'{"=" * 45}')
        print(f"  {display_name(bm)}  —  top {top_n} neighbors ({system_label})")
        print(f'{"=" * 45}')
        tbl = pd.DataFrame({"partner": partner_labels, "mean_coloc": top.values})
        print(tbl.to_string(index=False))
        print()

        ax.barh(partner_labels[::-1], top.values[::-1], color=color, alpha=0.8)
        ax.set_xlabel("Mean colocalization\n(arcsinh)")
        ax.set_title(f"{display_name(bm)} neighbors\n({system_label})")
        ax.tick_params(axis="y", labelsize=8)

    plt.suptitle(
        f"Top {top_n} colocalization partners for B-cell markers — {system_label}",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


def plot_diff_coloc(
    sp_a: pd.DataFrame,
    sp_b: pd.DataFrame,
    markers: list[str],
    columns,
    sys_a_label: str,
    sys_b_label: str,
    top_n: int = 10,
) -> plt.Figure:
    """Differential colocalization bar plots (2 × len(markers) grid).

    Row 0 (blue): partners more colocalized in *sp_b* (sys_b).
    Row 1 (red):  partners more colocalized in *sp_a* (sys_a).
    """
    fig, axes = plt.subplots(2, len(markers), figsize=(18, 10), sharey=False)
    if len(markers) == 1:
        axes = axes[:, np.newaxis]

    for col_i, bm in enumerate(markers):
        cols = get_marker_cols(columns, bm)
        if not cols:
            for row_i in range(2):
                axes[row_i][col_i].set_title(f"{bm}\n(not found)")
            continue

        df = compute_marker_diff_coloc(sp_a, sp_b, cols, bm)

        top_a = df.nlargest(top_n, "mean_diff").sort_values("mean_diff")
        top_b = df.nsmallest(top_n, "mean_diff").sort_values("mean_diff")

        print(f'{"=" * 50}')
        print(f"  {display_name(bm)}  —  higher in {sys_a_label}")
        print(f'{"=" * 50}')
        print(top_a[["partner", "mean_diff", "padj"]].to_string(index=False))
        print()
        print(f'{"=" * 50}')
        print(f"  {display_name(bm)}  —  higher in {sys_b_label}")
        print(f'{"=" * 50}')
        print(top_b[["partner", "mean_diff", "padj"]].to_string(index=False))
        print()

        for row_i, (top, color, sys_label) in enumerate([
            (top_b, "#1f77b4", sys_b_label),
            (top_a, "#d62728", sys_a_label),
        ]):
            ax = axes[row_i][col_i]
            partner_labels = [display_name(p) for p in top["partner"]]
            ax.barh(partner_labels, top["mean_diff"], color=color, alpha=0.8)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Mean diff\n(arcsinh)", fontsize=8)
            ax.set_title(f"{display_name(bm)}\nhigher in {sys_label}", fontsize=9)
            ax.tick_params(axis="y", labelsize=7)

            for i, (_, r) in enumerate(top.iterrows()):
                s = sig_label(r["padj"])
                x_pos = r["mean_diff"] + (0.01 if r["mean_diff"] >= 0 else -0.01)
                ha = "left" if r["mean_diff"] >= 0 else "right"
                ax.text(x_pos, i, s, va="center", ha=ha, fontsize=7)

    plt.suptitle(
        f"Top {top_n} differential colocalization partners — "
        f"{sys_a_label} vs {sys_b_label}  |  CD8 T cells",
        y=1.01,
    )
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


def draw_force_net(
    mat: pd.DataFrame,
    title: str,
    ax: plt.Axes | None = None,
    seed: int = 42,
    top_n: int = 60,
    highlight_node: str = "CD20",
    node_color_map: dict[str, str] | None = None,
    layout: str = "spring",
) -> None:
    """Force-directed network from a symmetric matrix.

    Keeps only the *top_n* strongest edges by |weight|.
    Blue = positive, red = negative, node size = weighted degree.

    Parameters
    ----------
    node_color_map : dict[str, str], optional
        Mapping of marker name → hex color. Overrides the default
        grey/highlight coloring. Nodes not in the map use '#555555'.
        When provided, edge-type legend is suppressed in favour of
        category legend (from mat._node_category_labels).
    layout : str
        Layout algorithm: 'spring' (Fruchterman-Reingold), 'kamada_kawai',
        or 'spectral'.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    # Collect and keep top edges
    all_edges = []
    for i, a in enumerate(mat.index):
        for j, b in enumerate(mat.columns):
            if j <= i:
                continue
            w = mat.loc[a, b]
            if abs(w) > 1e-6:
                all_edges.append((a, b, w))
    all_edges.sort(key=lambda x: abs(x[2]), reverse=True)
    keep = all_edges[:top_n]

    G = nx.Graph()
    for a, b, w in keep:
        G.add_edge(a, b, weight=w)

    if G.number_of_edges() == 0:
        ax.set_title(title)
        ax.axis("off")
        return

    # Layout
    G_layout = G.copy()
    for u, v, d in G_layout.edges(data=True):
        d["weight"] = abs(d["weight"])
    n_nodes = len(G.nodes())
    if layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G_layout)
    elif layout == "spectral":
        pos = nx.spectral_layout(G_layout)
    else:  # spring / fruchterman-reingold
        pos = nx.fruchterman_reingold_layout(
            G_layout,
            seed=seed,
            k=1.5 / (n_nodes ** 0.35),
            iterations=500,
        )

    # Edges
    edges = list(G.edges())
    weights = np.array([G[u][v]["weight"] for u, v in edges])
    abs_w = np.abs(weights)
    wmax = abs_w.max()
    widths = 1.0 + 5.0 * (abs_w / wmax)
    alphas = 0.4 + 0.55 * (abs_w / wmax)
    edge_colors = ["#1f77b4" if w > 0 else "#d62728" for w in weights]

    for (u, v), width, alpha, color in zip(edges, widths, alphas, edge_colors):
        ax.plot(*zip(pos[u], pos[v]), color=color, linewidth=width, alpha=alpha)

    # Nodes
    node_deg = {
        nd: sum(abs(G[nd][nb]["weight"]) for nb in G.neighbors(nd))
        for nd in G.nodes()
    }
    deg_max = max(node_deg.values()) if node_deg else 1
    node_sizes = [200 + 2000 * node_deg.get(nd, 0) / deg_max for nd in G.nodes()]
    if node_color_map is not None:
        node_colors = [node_color_map.get(nd, "#555555") for nd in G.nodes()]
    else:
        node_colors = [
            "#e84545" if nd == highlight_node else "#555555" for nd in G.nodes()
        ]

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
        alpha=0.85, edgecolors="white", linewidths=0.5,
    )
    label_map = {nd: display_name(nd) for nd in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=label_map, ax=ax, font_size=6, font_weight="bold")

    n_edges = G.number_of_edges()
    handles = []
    if node_color_map is not None:
        cat_labels = getattr(mat, "_node_category_labels", None)
        if cat_labels:
            for cat_name, cat_color in cat_labels.items():
                if any(node_color_map.get(nd) == cat_color for nd in G.nodes()):
                    handles.append(mpatches.Patch(color=cat_color, label=cat_name))
    else:
        handles = [
            mpatches.Patch(color="#1f77b4", label="Positive edge"),
            mpatches.Patch(color="#d62728", label="Negative edge"),
        ]
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.8)
    ax.set_title(f"{title}\n(top {n_edges} edges by |weight|)", fontsize=11)
    ax.axis("off")


def plot_neighborhood_suite(
    matrices: list[tuple[pd.DataFrame, str, str]],
    row_linkage: np.ndarray,
    suptitle_suffix: str,
    cluster_matrices: dict[str, pd.DataFrame] | None = None,
    cluster_title: str | None = None,
    highlight_node: str = "CD20",
    top_edges: int = 60,
    k_values: list[int] | None = None,
) -> None:
    """Plot heatmap + network for each matrix, then print cluster analysis.

    Parameters
    ----------
    matrices : list of (mat, label, cmap)
        Each entry is a symmetric matrix, its label, and colormap name.
        The last entry is assumed to be the difference matrix (gets its own vmax).
    row_linkage : np.ndarray
        Shared Ward linkage for all clustermaps.
    suptitle_suffix : str
        Appended to each figure's suptitle.
    cluster_matrices : dict[str, pd.DataFrame], optional
        Named matrices for within-cluster stats (e.g. {'Healthy': mat_h}).
        If provided, cluster analysis is printed after the plots.
    cluster_title : str, optional
        Header prefix for cluster analysis output.
    highlight_node : str
        Node to highlight in the network graph.
    top_edges : int
        Number of edges to keep in the network.
    k_values : list[int], optional
        Cluster counts to try (default [3, 5, 7]).
    """
    # Compute shared vmax for condition matrices, separate for diff
    cond_mats = [m for m, _, _ in matrices[:-1]]
    vmax_cond = max(m.abs().max().max() for m in cond_mats) if cond_mats else 1.0
    vmax_diff = matrices[-1][0].abs().max().max()

    n_markers = len(matrices[0][0])
    fig_size = max(10, n_markers * 0.18)
    tick_fs = max(4, min(7, 180 // n_markers))

    for i, (mat, label, cmap) in enumerate(matrices):
        vm = vmax_diff if i == len(matrices) - 1 else vmax_cond

        # Rename axes to display names for heatmap labels
        dn_map = {m: display_name(m) for m in mat.index}
        mat_display = mat.rename(index=dn_map, columns=dn_map)

        # Heatmap
        g = sns.clustermap(
            mat_display, cmap=cmap, center=0, vmin=-vm, vmax=vm,
            row_linkage=row_linkage, col_linkage=row_linkage,
            figsize=(fig_size, fig_size), linewidths=0,
            xticklabels=True, yticklabels=True,
            cbar_kws={"shrink": 0.4, "label": label},
            dendrogram_ratio=0.08,
            cbar_pos=(0.02, 0.82, 0.03, 0.15),
        )
        g.ax_heatmap.tick_params(axis="both", labelsize=tick_fs)
        g.fig.suptitle(
            f"CD20 neighborhood — {label}\n({suptitle_suffix})",
            fontsize=12, y=1.01,
        )
        plt.show()

        # Matching network
        fig_net, ax_net = plt.subplots(figsize=(10, 10))
        draw_force_net(
            mat,
            f"CD20 neighborhood network — {label}\n({suptitle_suffix})",
            ax=ax_net, top_n=top_edges, highlight_node=highlight_node,
        )
        plt.tight_layout()
        plt.show()

    # Cluster analysis
    if cluster_matrices is not None:
        markers = list(matrices[0][0].index)
        title = cluster_title or suptitle_suffix
        print_cluster_analysis(row_linkage, markers, cluster_matrices, title, k_values)


# ── 5. Printing / reporting functions ────────────────────────────────────────

def print_da_results(
    da_df: pd.DataFrame,
    cell_type: str,
    sys_a: str,
    sys_b: str,
    top_n: int = 10,
) -> None:
    """Print top-N DA markers higher in each system."""
    top_a = da_df.nlargest(top_n, "mean_diff")[["marker", "mean_diff", "padj"]].copy()
    top_b = da_df.nsmallest(top_n, "mean_diff")[["marker", "mean_diff", "padj"]].copy()
    top_a["marker"] = top_a["marker"].map(display_name)
    top_b["marker"] = top_b["marker"].map(display_name)

    print(f'{"=" * 55}')
    print(f"  {cell_type}  —  higher in {sys_a}")
    print(f'{"=" * 55}')
    print(top_a.to_string(index=False))
    print()
    print(f'{"=" * 55}')
    print(f"  {cell_type}  —  higher in {sys_b}")
    print(f'{"=" * 55}')
    print(top_b.to_string(index=False))
    print()


def print_cluster_analysis(
    row_linkage: np.ndarray,
    markers: list[str],
    matrices: dict[str, pd.DataFrame],
    title: str,
    k_values: list[int] | None = None,
) -> None:
    """Print cluster membership and within-cluster colocalization stats.

    Parameters
    ----------
    row_linkage : np.ndarray
        Ward linkage array.
    markers : list[str]
        Protein names (same order as linkage was computed).
    matrices : dict[str, pd.DataFrame]
        Named matrices to compute within-cluster stats from.
        E.g. ``{'Healthy': mat_h}`` or ``{'NALM-6': mat_n, 'Healthy': mat_h}``.
    title : str
        Header prefix.
    k_values : list[int], optional
        Cluster counts to try (default [3, 5, 7]).
    """
    if k_values is None:
        k_values = [3, 5, 7]

    mat_names = list(matrices.keys())

    for n_clust in k_values:
        labels = fcluster(row_linkage, t=n_clust, criterion="maxclust")
        cluster_map = pd.DataFrame({
            "protein": markers,
            "cluster": labels,
        }).sort_values(["cluster", "protein"])

        print(f'\n{"=" * 60}')
        print(f"  {title} — {n_clust} clusters (Ward linkage)")
        print(f'{"=" * 60}')

        for c in sorted(cluster_map["cluster"].unique()):
            members = cluster_map[cluster_map["cluster"] == c]["protein"].tolist()

            stats = []
            for name in mat_names:
                mat = matrices[name]
                if len(members) > 1:
                    sub = mat.loc[members, members]
                    mask = np.triu(np.ones(sub.shape, dtype=bool), k=1)
                    mc = sub.values[mask].mean()
                else:
                    mc = 0.0
                stats.append(f"{name}: {mc:.3f}")

            stat_str = "  |  ".join(stats)
            print(f"\n  Cluster {c}  (n={len(members)})")
            print(f"    Within-cluster coloc: {stat_str}")
            print(f"    {', '.join(display_name(m) for m in members)}")


def load_marker_panel_grouped(
    panel_name: str, path: str | Path | None = None,
) -> dict[str, list[str]]:
    """Load a marker panel keeping its category grouping.

    Returns
    -------
    dict[str, list[str]]
        Category name → list of markers.
    """
    p = Path(path) if path else _MARKER_PANEL_PATH
    with open(p) as f:
        panels = _json.load(f)
    return panels[panel_name]


def plot_marker_panel_violins(
    adata,
    panel_name: str,
    group_key: str,
    layer: str = "arcsinh",
    adata_compare=None,
    compare_label: str = "Comparison",
    primary_label: str = "Primary",
) -> None:
    """Violin plots for a marker panel — one figure per subcategory.

    When *adata_compare* is provided, both datasets are shown side-by-side on
    the same axes using ``hue`` (split violins), with different colour palettes.

    Parameters
    ----------
    adata : AnnData
        Primary dataset.
    panel_name : str
        Key in marker_panels.json (e.g. 'cd8_t_cell_markers').
    group_key : str
        obs column defining the groups.
    layer : str
        Expression layer.
    adata_compare : AnnData, optional
        Second dataset for side-by-side comparison.
    compare_label : str
        Label for the comparison dataset.
    primary_label : str
        Label for the primary dataset.
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


def plot_lfc_scatter(
    adata,
    time_val: str,
    sys_a: str,
    sys_b: str,
    cell_type: str,
    cond_num: str = "Blinatumomab",
    cond_den: str = "Mock",
    layer: str = "raw",
    pseudo: float = 1.0,
    top_k: int = 10,
    label_a: str | None = None,
    label_b: str | None = None,
    color_above: str = "#1f77b4",
    color_below: str = "#d62728",
    ax: plt.Axes | None = None,
    highlight_markers: list[str] | None = None,
) -> pd.DataFrame:
    """Per-marker LFC (cond_num / cond_den) scatter — system_a (x) vs system_b (y).

    If *highlight_markers* is given, exactly those markers are highlighted
    (silently skipping any not present in adata.var_names). Otherwise the top
    *top_k* markers farthest from the y=x diagonal are highlighted.
    Highlighted markers are annotated and printed in a table.

    Returns the LFC DataFrame.
    """
    label_a = label_a or sys_a
    label_b = label_b or sys_b

    def _lfc(system_val):
        mask = (
            (adata.obs["time"] == time_val) &
            (adata.obs["cell_type_annot"] == cell_type) &
            (adata.obs["cell_system"] == system_val)
        )
        sub = adata[mask]
        X = np.array(sub.layers[layer], dtype=np.float32)
        num = X[(sub.obs["condition"] == cond_num).values].mean(axis=0)
        den = X[(sub.obs["condition"] == cond_den).values].mean(axis=0)
        return pd.Series(np.log2((num + pseudo) / (den + pseudo)), index=sub.var_names)

    lfc_a = _lfc(sys_a)
    lfc_b = _lfc(sys_b)
    df = pd.DataFrame({"a": lfc_a, "b": lfc_b}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    df["off_diag"] = (df["b"] - df["a"]) / np.sqrt(2)

    if highlight_markers is not None:
        top_idx = pd.Index([m for m in highlight_markers if m in df.index])
        highlight_label = f"selected ({len(top_idx)}/{len(highlight_markers)} present)"
    else:
        top_idx = df["off_diag"].abs().nlargest(top_k).index
        highlight_label = f"top {top_k} farthest from y=x"
    colors = np.where(df.loc[top_idx, "off_diag"] > 0, color_above, color_below)
    color_map = dict(zip(top_idx, colors))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    lim = max(df[["a", "b"]].abs().max().max() * 1.15, 0.1)
    ax.axhline(0, color="grey", lw=0.7, ls="--")
    ax.axvline(0, color="grey", lw=0.7, ls="--")
    ax.plot([-lim, lim], [-lim, lim], color="grey", lw=0.7, ls=":")

    other_idx = df.index.difference(top_idx)
    ax.scatter(
        df.loc[other_idx, "a"], df.loc[other_idx, "b"],
        s=30, alpha=0.55, color="#bbbbbb", edgecolor="white", linewidth=0.5,
    )
    ax.scatter(
        df.loc[top_idx, "a"], df.loc[top_idx, "b"],
        s=70, alpha=0.9, c=[color_map[m] for m in top_idx],
        edgecolor="black", linewidth=0.6, zorder=3,
    )
    for m in top_idx:
        ax.annotate(
            display_name(m),
            (df.loc[m, "a"], df.loc[m, "b"]),
            fontsize=9, fontweight="bold", color=color_map[m],
            xytext=(4, 4), textcoords="offset points",
        )

    r = df["a"].corr(df["b"])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"LFC {cond_num} vs {cond_den}\n{label_a}")
    ax.set_ylabel(f"LFC {cond_num} vs {cond_den}\n{label_b}")
    ax.set_title(f"{cell_type}  —  {time_val}   (n={len(df)}, Pearson r={r:.2f})")

    print(f'\n{"=" * 65}')
    print(f"  {cell_type} {time_val}  —  {highlight_label}")
    print(f"  {color_above} = higher LFC in {label_b} | {color_below} = higher LFC in {label_a}")
    print(f'{"=" * 65}')
    top_tbl = df.loc[top_idx, ["a", "b", "off_diag"]].copy()
    top_tbl.index = [display_name(m) for m in top_tbl.index]
    top_tbl.columns = [f"LFC_{label_a}", f"LFC_{label_b}", "off_diag"]
    top_tbl = top_tbl.reindex(top_tbl["off_diag"].abs().sort_values(ascending=False).index)
    print(top_tbl.round(3).to_string())

    return df


def plot_synapse_suite(
    sp_a: pd.DataFrame,
    sp_b: pd.DataFrame,
    categories: dict,
    label_a: str,
    label_b: str,
    diff_label: str | None = None,
    suite_name: str = "Synapse",
    highlight_node: str | None = None,
    top_edges: int = 80,
    layout: str = "kamada_kawai",
    cluster_k: list | None = None,
    var_filter=None,
) -> tuple:
    """Heatmaps + force-directed networks for a categorized marker suite,
    contrasting two systems and their difference.

    Generic for any synapse-like view (immune synapse, APC synapse, etc.).

    Parameters
    ----------
    sp_a, sp_b : pd.DataFrame
        Spatial colocalization DataFrames (cells × pairs) for the two systems.
    categories : dict
        ``{category_name: (marker_list, hex_color)}``. Markers absent from
        *var_filter* (if given) are dropped.
    label_a, label_b : str
        Display labels for the two systems.
    diff_label : str, optional
        Label for the difference matrix.
    suite_name : str
        Prefix for figure suptitles (e.g. ``"Immune synapse"``).
    highlight_node : str, optional
        Node to highlight in network plots.
    var_filter : iterable, optional
        Restrict to markers present in this iterable (e.g. ``adata.var_names``).
    cluster_k : list[int], optional
        If provided, runs ``print_cluster_analysis`` with these cluster counts.

    Returns
    -------
    (mat_a, mat_b, mat_diff)
    """
    if diff_label is None:
        diff_label = f"Diff ({label_a} − {label_b})"

    var_set = set(var_filter) if var_filter is not None else None
    cat_present = {}
    node_cmap = {}
    markers_set = set()
    for cat, (ms, color) in categories.items():
        if var_set is not None:
            ms = [m for m in ms if m in var_set]
        if not ms:
            continue
        cat_present[cat] = (ms, color)
        markers_set.update(ms)
        for m in ms:
            node_cmap[m] = color
    markers = sorted(markers_set)
    n = len(markers)
    if n == 0:
        raise ValueError("No markers from the provided categories were found.")

    print(f"{suite_name} marker set ({n}):")
    for cat, (ms, _) in cat_present.items():
        print(f"  {cat}: {[display_name(m) for m in ms]}")

    pair_cols = get_pairwise_cols(sp_a.columns, set(markers))
    mat_a = build_mean_matrix(sp_a, pair_cols, markers)
    mat_b = build_mean_matrix(sp_b, pair_cols, markers)
    mat_diff = mat_a - mat_b

    row_linkage = compute_ward_linkage(mat_diff)

    cl = {cat: color for cat, (_, color) in cat_present.items()}
    for mm in (mat_a, mat_b, mat_diff):
        mm._node_category_labels = cl

    cat_handles = [
        mpatches.Patch(color=c, label=cat)
        for cat, (_, c) in cat_present.items()
    ]

    vmax = max(mat_a.abs().max().max(), mat_b.abs().max().max())
    vmax_diff = mat_diff.abs().max().max()

    fig_sz = max(10, n * 0.28)
    tk_fs = max(5, min(8, 200 // n))
    sub = f"{suite_name} markers (n={n})"

    configs = [
        (mat_a,    label_a,    "RdBu_r",   vmax),
        (mat_b,    label_b,    "RdBu_r",   vmax),
        (mat_diff, diff_label, "coolwarm", vmax_diff),
    ]

    for mat, lbl, cmp, vm in configs:
        dn = {m: display_name(m) for m in mat.index}
        md = mat.rename(index=dn, columns=dn)
        rc = pd.Series(
            {display_name(m): node_cmap.get(m, "#555555") for m in mat.index},
            name="Category",
        )

        g = sns.clustermap(
            md, cmap=cmp, center=0, vmin=-vm, vmax=vm,
            row_linkage=row_linkage, col_linkage=row_linkage,
            row_colors=rc, col_colors=rc,
            figsize=(fig_sz, fig_sz), linewidths=0,
            xticklabels=True, yticklabels=True,
            cbar_kws={"shrink": 0.4, "label": lbl},
            dendrogram_ratio=0.08, cbar_pos=(0.02, 0.82, 0.03, 0.15),
        )
        g.ax_heatmap.tick_params(axis="both", labelsize=tk_fs)
        g.ax_heatmap.legend(
            handles=cat_handles, loc="upper left", fontsize=7,
            framealpha=0.9, title="Category", title_fontsize=8,
            bbox_to_anchor=(-0.3, 1.0),
        )
        g.fig.suptitle(f"{suite_name} — {lbl}\n({sub})", fontsize=12, y=1.01)
        plt.show()

        fig_net, ax_net = plt.subplots(figsize=(10, 10))
        draw_force_net(
            mat, f"{suite_name} network — {lbl}\n({sub})",
            ax=ax_net, top_n=top_edges, highlight_node=highlight_node,
            node_color_map=node_cmap, layout=layout,
        )
        plt.tight_layout()
        plt.show()

    if cluster_k is not None:
        print_cluster_analysis(
            row_linkage, markers,
            {label_a: mat_a, label_b: mat_b},
            suite_name,
            k_values=cluster_k,
        )

    return mat_a, mat_b, mat_diff


def plot_condition_comparison(
    adata,
    time_val: str,
    cell_system: str,
    cell_type: str,
    cond_a: str = "Blinatumomab",
    cond_b: str = "Mock",
    layer: str = "arcsinh",
    obsm_key: str = "spatial_asinh5",
    top_n: int = 15,
    system_label: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2x2 panel comparing two conditions within (time × system × cell_type).

    Row 0: abundance (top-N markers higher in cond_b | cond_a).
    Row 1: spatial colocalization (top-N pairs higher in cond_b | cond_a).

    Returns (abundance_da, spatial_da) DataFrames.
    """
    label = system_label or cell_system
    mask = (
        (adata.obs["time"] == time_val) &
        (adata.obs["cell_type_annot"] == cell_type) &
        (adata.obs["cell_system"] == cell_system)
    )
    sub = adata[mask]
    cond = sub.obs["condition"].values
    n_a = (cond == cond_a).sum()
    n_b = (cond == cond_b).sum()
    print(f"\n[{time_val}] {cell_type} in {label}: {cond_a} n={n_a}, {cond_b} n={n_b}")

    def _mw_da(values_a, values_b, labels):
        rows = []
        for j, name in enumerate(labels):
            a, b = values_a[:, j], values_b[:, j]
            mean_diff = a.mean() - b.mean()
            _, p = mannwhitneyu(a, b, alternative="two-sided")
            rows.append({"feature": name, "mean_diff": mean_diff, "pval": p})
        df = pd.DataFrame(rows)
        _, df["padj"], _, _ = multipletests(df["pval"], method="fdr_bh")
        return df

    X = np.array(sub.layers[layer], dtype=np.float32)
    da_ab = _mw_da(X[cond == cond_a], X[cond == cond_b], list(sub.var_names))

    sp = sub.obsm[obsm_key]
    if not isinstance(sp, pd.DataFrame):
        sp = pd.DataFrame(sp, index=sub.obs_names)
    sp_vals = sp.values
    da_sp = _mw_da(sp_vals[cond == cond_a], sp_vals[cond == cond_b], list(sp.columns))

    def _pair_label(pair_name):
        a, b = split_pair(pair_name)
        return f"{display_name(a)} / {display_name(b)}"

    def _plot_bars(ax, top, color, xlabel, title, label_fmt, tick_fs=8):
        labels_ = [label_fmt(f) for f in top["feature"]]
        ax.barh(labels_, top["mean_diff"], color=color, alpha=0.85)
        ax.axvline(0, color="black", lw=0.7, ls="--")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=tick_fs)
        xrng = max(top["mean_diff"].abs().max(), 1e-6)
        pad = xrng * 0.02
        for i, (_, r) in enumerate(top.iterrows()):
            s = sig_label(r["padj"])
            xp = r["mean_diff"] + (pad if r["mean_diff"] >= 0 else -pad)
            ha = "left" if r["mean_diff"] >= 0 else "right"
            ax.text(xp, i, s, va="center", ha=ha, fontsize=7)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    top_up_ab = da_ab.nlargest(top_n, "mean_diff").sort_values("mean_diff")
    top_dn_ab = da_ab.nsmallest(top_n, "mean_diff").sort_values("mean_diff")
    _plot_bars(axes[0, 0], top_dn_ab, "#1f77b4",
               f"Mean diff ({layer}, {cond_a} − {cond_b})",
               f"Abundance — top {top_n} higher in {cond_b}",
               label_fmt=display_name)
    _plot_bars(axes[0, 1], top_up_ab, "#d62728",
               f"Mean diff ({layer}, {cond_a} − {cond_b})",
               f"Abundance — top {top_n} higher in {cond_a}",
               label_fmt=display_name)

    top_up_sp = da_sp.nlargest(top_n, "mean_diff").sort_values("mean_diff")
    top_dn_sp = da_sp.nsmallest(top_n, "mean_diff").sort_values("mean_diff")
    _plot_bars(axes[1, 0], top_dn_sp, "#1f77b4",
               f"Mean diff ({cond_a} − {cond_b})",
               f"Spatial coloc — top {top_n} pairs higher in {cond_b}",
               label_fmt=_pair_label, tick_fs=7)
    _plot_bars(axes[1, 1], top_up_sp, "#d62728",
               f"Mean diff ({cond_a} − {cond_b})",
               f"Spatial coloc — top {top_n} pairs higher in {cond_a}",
               label_fmt=_pair_label, tick_fs=7)

    fig.suptitle(f"{cell_type} — {label} — {time_val}  ({cond_a} vs {cond_b})",
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.show()

    print(f"\n--- {time_val} {label} abundance (top higher in {cond_a}) ---")
    out = top_up_ab.iloc[::-1][["feature", "mean_diff", "padj"]].copy()
    out["feature"] = out["feature"].map(display_name)
    print(out.to_string(index=False))
    print(f"\n--- {time_val} {label} spatial coloc (top higher in {cond_a}) ---")
    out_sp = top_up_sp.iloc[::-1][["feature", "mean_diff", "padj"]].copy()
    out_sp["feature"] = out_sp["feature"].map(_pair_label)
    print(out_sp.to_string(index=False))

    return da_ab, da_sp


def plot_ma_blina_vs_mock(
    adata,
    systems: list[tuple[str, str]],
    time_val: str,
    cell_type: str,
    layer: str = "arcsinh",
    obsm_key: str = "spatial_asinh5",
    fdr_thresh: float = 0.05,
    top_label_abundance: int = 10,
    top_label_spatial: int = 8,
    suptitle: str | None = None,
) -> dict:
    """Per-system MA plots (LFC vs mean signal) for Blinatumomab vs Mock.

    Plots a 2 × len(*systems*) grid: rows = (abundance, spatial coloc),
    cols = systems. Per panel, x = mean signal across Blina ∪ Mock,
    y = Blina − Mock LFC. Points are coloured by FDR direction; the top
    significant features by |LFC| are annotated.

    Parameters
    ----------
    systems : list of (label, cell_system_value)
        Each entry produces one column of the figure.

    Returns
    -------
    dict
        ``{label: {'abundance': df, 'spatial': df,
                   'n_blina': int, 'n_mock': int}}``
        Each DataFrame has columns: feature, mean_signal, mean_a, mean_b,
        lfc, pval, padj, n_a, n_b. Suitable to feed into
        :func:`plot_delta_lfc_ma`.
    """
    def _compute_ma_table(values_a, values_b, names):
        rows = []
        for j, name in enumerate(names):
            a = np.asarray(values_a[:, j], dtype=np.float64)
            b = np.asarray(values_b[:, j], dtype=np.float64)
            mean_combined = np.concatenate([a, b]).mean()
            lfc = a.mean() - b.mean()
            if a.std() == 0 and b.std() == 0:
                pval = 1.0
            else:
                try:
                    _, pval = mannwhitneyu(a, b, alternative="two-sided")
                except ValueError:
                    pval = 1.0
            rows.append({
                "feature": name, "mean_signal": mean_combined,
                "mean_a": a.mean(), "mean_b": b.mean(),
                "lfc": lfc, "pval": pval,
                "n_a": len(a), "n_b": len(b),
            })
        df = pd.DataFrame(rows)
        _, df["padj"], _, _ = multipletests(df["pval"].fillna(1.0), method="fdr_bh")
        return df

    def _subset(system_val):
        mask = (
            (adata.obs["time"] == time_val) &
            (adata.obs["cell_type_annot"] == cell_type) &
            (adata.obs["cell_system"] == system_val)
        )
        sub = adata[mask]
        cond = sub.obs["condition"].values
        is_blina = cond == "Blinatumomab"
        is_mock = cond == "Mock"
        X = np.asarray(sub.layers[layer], dtype=np.float32)
        da_ab = _compute_ma_table(X[is_blina], X[is_mock], list(sub.var_names))
        sp = sub.obsm[obsm_key]
        if not isinstance(sp, pd.DataFrame):
            sp = pd.DataFrame(sp, index=sub.obs_names)
        sp_vals = sp.values
        da_sp = _compute_ma_table(
            sp_vals[is_blina], sp_vals[is_mock], list(sp.columns)
        )
        return da_ab, da_sp, int(is_blina.sum()), int(is_mock.sum())

    def _pair_label(p):
        a, b = split_pair(p)
        return f"{display_name(a)} / {display_name(b)}"

    def _plot_ma(ax, df, title, label_fn, top_label):
        df = df.copy()
        df["sig"] = df["padj"] < fdr_thresh
        up = df[df["sig"] & (df["lfc"] > 0)]
        down = df[df["sig"] & (df["lfc"] < 0)]
        ns = df[~df["sig"]]
        ax.scatter(ns["mean_signal"], ns["lfc"], s=12, color="lightgrey",
                   alpha=0.6, edgecolors="none", label=f"n.s. ({len(ns)})")
        ax.scatter(down["mean_signal"], down["lfc"], s=18, color="#1f77b4",
                   alpha=0.85, edgecolors="none",
                   label=f"\u2191 Mock ({len(down)})")
        ax.scatter(up["mean_signal"], up["lfc"], s=18, color="#d62728",
                   alpha=0.85, edgecolors="none",
                   label=f"\u2191 Blina ({len(up)})")
        ax.axhline(0, color="black", lw=0.7, ls="--")
        sig_only = df[df["sig"]].copy()
        sig_only["absLFC"] = sig_only["lfc"].abs()
        top = sig_only.nlargest(top_label, "absLFC")
        for _, r in top.iterrows():
            ax.annotate(label_fn(r["feature"]),
                        (r["mean_signal"], r["lfc"]),
                        fontsize=7, alpha=0.9,
                        xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel(f"Mean signal ({layer}, Blina \u222a Mock)")
        ax.set_ylabel("LFC (Blina \u2212 Mock)")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8, frameon=False)

    ma_results = {}
    n_sys = len(systems)
    fig, axes = plt.subplots(2, n_sys, figsize=(7.5 * n_sys, 11), squeeze=False)
    for col, (sys_label, sys_val) in enumerate(systems):
        da_ab, da_sp, n_blina, n_mock = _subset(sys_val)
        ma_results[sys_label] = {
            "abundance": da_ab, "spatial": da_sp,
            "n_blina": n_blina, "n_mock": n_mock,
        }
        _plot_ma(axes[0, col], da_ab,
                 title=(f"{sys_label} \u00b7 Abundance \u00b7 {time_val}  "
                        f"(Blina n={n_blina} / Mock n={n_mock})"),
                 label_fn=display_name, top_label=top_label_abundance)
        _plot_ma(axes[1, col], da_sp,
                 title=f"{sys_label} \u00b7 Spatial coloc \u00b7 {time_val}",
                 label_fn=_pair_label, top_label=top_label_spatial)
    if suptitle is None:
        suptitle = (f"{cell_type} cells \u2014 Blina vs Mock MA plots "
                    "(LFC vs mean signal)")
    fig.suptitle(suptitle, fontsize=14, y=1.00)
    plt.tight_layout()
    plt.show()
    return ma_results


def plot_delta_lfc_ma(
    ma_results: dict,
    label_a: str,
    label_b: str,
    cell_type: str = "B",
    time_val: str = "6h",
    color_a: str = "#9467bd",
    color_b: str = "#2ca02c",
    fdr_thresh: float = 0.05,
    top_label_abundance: int = 12,
    top_label_spatial: int = 10,
    print_top: int = 15,
    suptitle: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ΔLFC vs mean signal (LFC of LFC) — *label_a* vs *label_b*.

    Pairs the per-system MA tables produced by :func:`plot_ma_blina_vs_mock`
    and plots ΔLFC = LFC(*label_a*) − LFC(*label_b*) against the mean signal
    averaged across the two systems. Points are coloured by ΔLFC sign
    (*color_a* = higher in *label_a*, *color_b* = higher in *label_b*) — same
    convention as :func:`plot_lfc_scatter` (purple above y=x, green below).
    Marker size / opacity encode FDR-significance buckets (none / one side /
    both sides at *fdr_thresh*).

    Returns
    -------
    (delta_abundance, delta_spatial) : tuple[pd.DataFrame, pd.DataFrame]
    """
    def _delta_table(df_a, df_b):
        merged = df_a[["feature", "mean_signal", "mean_a", "mean_b", "lfc", "padj"]].merge(
            df_b[["feature", "mean_signal", "mean_a", "mean_b", "lfc", "padj"]],
            on="feature", suffixes=(f"_{label_a}", f"_{label_b}"),
        )
        merged["mean_signal_combined"] = (
            merged[f"mean_signal_{label_a}"] + merged[f"mean_signal_{label_b}"]
        ) / 2.0
        merged["delta_lfc"] = (
            merged[f"lfc_{label_a}"] - merged[f"lfc_{label_b}"]
        )
        merged["either_sig"] = (
            (merged[f"padj_{label_a}"] < fdr_thresh)
            | (merged[f"padj_{label_b}"] < fdr_thresh)
        )
        merged["both_sig"] = (
            (merged[f"padj_{label_a}"] < fdr_thresh)
            & (merged[f"padj_{label_b}"] < fdr_thresh)
        )
        return merged

    def _pair_label(p):
        a, b = split_pair(p)
        return f"{display_name(a)} / {display_name(b)}"

    def _plot(ax, df, label_fn, top_label):
        df = df.copy()
        df["color"] = np.where(df["delta_lfc"] >= 0, color_a, color_b)
        ns = df[~df["either_sig"]]
        one = df[df["either_sig"] & ~df["both_sig"]]
        both = df[df["both_sig"]]
        ax.scatter(ns["mean_signal_combined"], ns["delta_lfc"],
                   s=10, color="lightgrey", alpha=0.5, edgecolors="none",
                   label=f"n.s. either side ({len(ns)})")
        ax.scatter(one["mean_signal_combined"], one["delta_lfc"],
                   s=18, c=one["color"], alpha=0.55, edgecolors="none",
                   label=f"sig. one side ({len(one)})")
        ax.scatter(both["mean_signal_combined"], both["delta_lfc"],
                   s=26, c=both["color"], alpha=0.95,
                   edgecolors="black", linewidths=0.4,
                   label=f"sig. both sides ({len(both)})")
        ax.axhline(0, color="black", lw=0.7, ls="--")
        df["absDelta"] = df["delta_lfc"].abs()
        candidates = df[df["either_sig"]].copy()
        if len(candidates) == 0:
            candidates = df.copy()
        top = candidates.nlargest(top_label, "absDelta")
        for _, r in top.iterrows():
            ax.annotate(label_fn(r["feature"]),
                        (r["mean_signal_combined"], r["delta_lfc"]),
                        fontsize=7, alpha=0.95,
                        xytext=(3, 2), textcoords="offset points")
        direction_handles = [
            mpatches.Patch(color=color_a,
                           label=f"\u0394LFC > 0 \u2192 \u2191 in {label_a}"),
            mpatches.Patch(color=color_b,
                           label=f"\u0394LFC < 0 \u2192 \u2191 in {label_b}"),
        ]
        sig_legend = ax.legend(loc="upper left", fontsize=7, frameon=False)
        ax.add_artist(sig_legend)
        ax.legend(handles=direction_handles, loc="lower right",
                  fontsize=7, frameon=False)
        ax.set_xlabel(
            "Mean signal (Blina \u222a Mock, averaged across systems)"
        )
        ax.set_ylabel(f"\u0394LFC = LFC({label_a}) \u2212 LFC({label_b})")

    delta_ab = _delta_table(
        ma_results[label_a]["abundance"],
        ma_results[label_b]["abundance"],
    )
    delta_sp = _delta_table(
        ma_results[label_a]["spatial"],
        ma_results[label_b]["spatial"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    _plot(axes[0], delta_ab, label_fn=display_name,
          top_label=top_label_abundance)
    axes[0].set_title(
        f"{cell_type} \u00b7 Abundance \u00b7 \u0394LFC "
        f"{label_a} \u2212 {label_b} \u00b7 {time_val}"
    )
    _plot(axes[1], delta_sp, label_fn=_pair_label,
          top_label=top_label_spatial)
    axes[1].set_title(
        f"{cell_type} \u00b7 Spatial coloc \u00b7 \u0394LFC "
        f"{label_a} \u2212 {label_b} \u00b7 {time_val}"
    )
    if suptitle is None:
        suptitle = (
            f"{cell_type} cells \u2014 Blina-vs-Mock \u0394LFC "
            f"({label_a} \u2212 {label_b}) standardised by mean signal"
        )
    fig.suptitle(suptitle, fontsize=14, y=1.00)
    plt.tight_layout()
    plt.show()

    def _print_top(df, label_fn, name):
        cand = df[df["either_sig"]].copy() if df["either_sig"].any() else df.copy()
        cand["absDelta"] = cand["delta_lfc"].abs()
        out = cand.nlargest(print_top, "absDelta")[
            ["feature", "mean_signal_combined",
             f"lfc_{label_a}", f"lfc_{label_b}", "delta_lfc",
             f"padj_{label_a}", f"padj_{label_b}"]
        ].copy()
        out["feature"] = out["feature"].map(label_fn)
        print(f"\n--- {name} \u00b7 top {print_top} by |\u0394LFC| "
              "(sig in either system) ---")
        print(out.round(3).to_string(index=False))

    _print_top(delta_ab, display_name, "Abundance")
    _print_top(delta_sp, _pair_label, "Spatial coloc")

    return delta_ab, delta_sp
