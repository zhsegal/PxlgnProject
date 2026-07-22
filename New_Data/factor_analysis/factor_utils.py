"""Utilities for factor interpretation following Gaublomme et al. framework."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, kruskal, ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Import helpers from parent nalm_utils
# ---------------------------------------------------------------------------
_PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PARENT_DIR))
from nalm_utils import display_name, sig_label, load_marker_panel  # noqa: E402

__all__ = [
    # constants
    "MARKER_PROGRAMS",
    "MARKER_TO_GENE",
    "CACHE_DIR",
    "ADATA_PATH",
    "CD8_MODEL_DIR",
    "CD8_ADATA_PATH",
    # notebook helpers
    "configure_mpl",
    "prepare_adata_for_model",
    "extract_model_outputs",
    # legacy analysis (backward compat)
    "score_programs",
    "correlate_factors_with_programs",
    "compute_marker_factor_loadings",
    "test_factor_metadata_categorical",
    "select_root_cell",
    "correlate_factors_with_pseudotime",
    "build_semantic_map",
    # legacy heatmaps (backward compat)
    "build_factor_metadata_heatmap",
    "build_factor_program_heatmap",
    # legacy plots (backward compat)
    "plot_loadings_heatmap",
    "plot_factor_metadata_violins",
    "plot_factor_pseudotime_grid",
    "plot_pseudotime_distributions",
    "plot_program_correlation_matrix",
    "plot_metadata_scatter_grid",
    "plot_program_scatter_grid",
    "plot_umap_factor_scores",
    "plot_factor_scatter",
    # Gaublomme-style pipeline (new)
    "plot_embedding_distances",
    "compute_pseudotime",
    "plot_umap_overview",
    "discover_marker_programs",
    "characterize_programs_from_W",
    "run_gep_discovery",
    "plot_marker_umap",
    "plot_program_loadings_w",
    "plot_factor_diagnostics",
    "compute_program_scores",
    "compute_metadata_associations",
    "factor_program_correlation",
    "build_interpretation_table",
    "plot_correlation_heatmap",
    "plot_factor_scatter_grid",
    "plot_training_health",
    # synapse axis helpers
    "build_synapse_features",
    "select_pca_columns",
    "run_synapse_pca",
    "build_synapse_score_df",
    "orient_pc1_by_metadata",
    "plot_synapse_scree",
    "plot_synapse_pc_loadings",
    "plot_synapse_metadata_heatmap",
    "plot_synapse_pc_scatter_grid",
    "plot_synapse_pc_violins_by_system",
    "plot_factor_synapse_deepdive",
    "plot_synapse_umap",
    # quadrant analysis
    "assign_quadrants",
    "plot_quadrant_scatter",
    "plot_quadrant_composition",
    "plot_quadrant_gep_scores",
    "run_quadrant_de",
    "plot_quadrant_de_summary",
    "run_quadrant_analysis",
    # re-exports
    "display_name",
    "sig_label",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_DIR = _PARENT_DIR / "cache"
ADATA_PATH = CACHE_DIR / "adata_cytovi_annotated_compat.h5ad"
CD8_MODEL_DIR = CACHE_DIR / "cytovi_cd8_model"
CD8_ADATA_PATH = CACHE_DIR / "adata_cd8_cytovi.h5ad"

# ---------------------------------------------------------------------------
# Curated marker programs (biology-driven, spanning multiple panels)
# ---------------------------------------------------------------------------
MARKER_PROGRAMS: dict[str, list[str]] = {
    "T_activation":    ["CD25", "CD38", "CD69", "CD71", "HLA-DR", "CD95"],
    "T_exhaustion":    ["CD279", "CD366", "TIGIT", "VISTA", "CD152", "CD244",
                        "CD159a", "CD305", "CD39", "CD73"],
    "T_costimulation": ["CD28", "CD27", "CD134", "CD137", "CD226", "CD278", "CD357"],
    "T_cytotoxicity":  ["CD56", "CD94", "CD158a", "CD158b", "CD161", "CD314", "GPR56"],
    "T_memory_diff":   ["CD45RA", "CD45RO", "CD44", "CD127", "CD57", "KLRG1"],
    "Adhesion_homing": ["CD11a", "CD18", "CD29", "CD49D", "CD54", "CD103", "CX3CR1"],
}

# ---------------------------------------------------------------------------
# Marker → gene symbol mapping for DS569k ProteinCLIP lookup
# ---------------------------------------------------------------------------
MARKER_TO_GENE: dict[str, list[str]] = {
    "HLA-ABC": ["HLA-A", "HLA-B", "HLA-C"],
    "B2M": ["B2M"],
    "CD11b": ["ITGAM"],
    "CD11c": ["ITGAX"],
    "CD18": ["ITGB2"],
    "CD82": ["CD82"],
    "CD8": ["CD8A"],
    "TCRab": ["TRAC", "TRBC1"],
    "HLA-DR": ["HLA-DRA"],
    "CD45": ["PTPRC"],
    "CD14": ["CD14"],
    "CD16": ["FCGR3A"],
    "CD19": ["CD19"],
    "CD45RB": ["PTPRC"],
    "CD44": ["CD44"],
    "CD52": ["CD52"],
    "CD59": ["CD59"],
    "CD45RA": ["PTPRC"],
    "CD36": ["CD36"],
    "CD2": ["CD2"],
    "CD29": ["ITGB1"],
    "CD5": ["CD5"],
    "CD162": ["SELPLG"],
    "CD27": ["CD27"],
    "CD35": ["CR1"],
    "CD26": ["DPP4"],
    "CD49D": ["ITGA4"],
    "CD37": ["CD37"],
    "CD41": ["ITGA2B"],
    "CD20": ["MS4A1"],
    "CD64": ["FCGR1A"],
    "CD22": ["CD22"],
    "CD274": ["CD274"],
    "CD328": ["SIGLEC7"],
    "CD25": ["IL2RA"],
    "CD279": ["PDCD1"],
    "CD335": ["NCR1"],
    "CD152": ["CTLA4"],
    "CD86": ["CD86"],
    "CD13": ["ANPEP"],
    "CD156c": ["ADAM10"],
    "CD158a": ["KIR2DL1"],
    "CD158b": ["KIR2DL3"],
    "CD159a": ["KLRC1"],
    "CD159c": ["KLRC2"],
    "CD163": ["CD163"],
    "CD169": ["SIGLEC1"],
    "CD1a": ["CD1A"],
    "CD1c": ["CD1C"],
    "CD206": ["MRC1"],
    "CD226": ["CD226"],
    "CD273": ["PDCD1LG2"],
    "CD28": ["CD28"],
    "CD352": ["SLAMF6"],
    "CD45RO": ["PTPRC"],
    "CD49e": ["ITGA5"],
    "CD56": ["NCAM1"],
    "CD79a": ["CD79A"],
    "CD80": ["CD80"],
    "CD81": ["CD81"],
    "CD85j": ["LILRB1"],
    "CD93": ["CD93"],
    "IgD": ["IGHD"],
    "IgE": ["IGHE"],
    "IgM": ["IGHM"],
    "KLRG1": ["KLRG1"],
    "NKp80": ["KLRF1"],
    "TCRgd": ["TRGC1", "TRDC"],
    "CD10": ["MME"],
    "CD103": ["ITGAE"],
    "CD199": ["CCR9"],
    "CD1b": ["CD1B"],
    "CD21": ["CR2"],
    "CD231": ["TSPAN7"],
    "CD277": ["BTN3A1"],
    "CD305": ["LAIR1"],
    "CD371": ["CLEC12A"],
    "CD89": ["FCAR"],
    "CD90": ["THY1"],
    "CD70": ["CD70"],
    "GPR56": ["ADGRG1"],
    "HLA-DQ": ["HLA-DQA1", "HLA-DQB1"],
    "TCRVd2": ["TRDV2"],
    "HLA-DR-DP-DQ": ["HLA-DRA", "HLA-DPA1", "HLA-DQA1"],
    "Siglec-9": ["SIGLEC9"],
    "TCRva7.2": ["TRAV1-2"],
    "CD95": ["FAS"],
    "TCRVg9": ["TRGV9"],
    "CD127": ["IL7R"],
    "CD141": ["THBD"],
    "CD31": ["PECAM1"],
    "CD6": ["CD6"],
    "CD57": ["B3GAT1"],
    "CD73": ["NT5E"],
    "CD366": ["HAVCR2"],
    "CD357": ["TNFRSF18"],
    "CD193": ["CCR3"],
    "CD319": ["SLAMF7"],
    "TIGIT": ["TIGIT"],
    "CD134": ["TNFRSF4"],
    "CD102": ["ICAM2"],
    "CD123": ["IL3RA"],
    "CD150": ["SLAMF1"],
    "CD154": ["CD40LG"],
    "CD158": ["KIR2DL1"],
    "CD161": ["KLRB1"],
    "CD180": ["CD180"],
    "CD191": ["CCR1"],
    "CD192": ["CCR2"],
    "CD1d": ["CD1D"],
    "CD200": ["CD200"],
    "CD229": ["LY9"],
    "CD24": ["CD24"],
    "CD244": ["CD244"],
    "CD268": ["TNFRSF13C"],
    "CD278": ["ICOS"],
    "CD32": ["FCGR2A"],
    "CD33": ["CD33"],
    "CD337": ["NCR3"],
    "CD38": ["CD38"],
    "CD39": ["ENTPD1"],
    "CD40": ["CD40"],
    "CD48": ["CD48"],
    "CD50": ["ICAM3"],
    "CD55": ["CD55"],
    "CD58": ["CD58"],
    "CD62P": ["SELP"],
    "CD69": ["CD69"],
    "CD72": ["CD72"],
    "CD84": ["CD84"],
    "CD9": ["CD9"],
    "CD94": ["KLRD1"],
    "TCRVB5": ["TRBV5-1"],
    "CD3e": ["CD3E"],
    "CD4": ["CD4"],
    "CD11a": ["ITGAL"],
    "CD43": ["SPN"],
    "CD7": ["CD7"],
    "CD53": ["CD53"],
    "CD302": ["CD302"],
    "VISTA": ["VSIR"],
    "CD269": ["TNFRSF17"],
    "CD138": ["SDC1"],
    "CD137": ["TNFRSF9"],
    "CD66b": ["CEACAM8"],
    "CX3CR1": ["CX3CR1"],
    "CD326": ["EPCAM"],
    "CD209": ["CD209"],
    "CD34": ["CD34"],
    "CD369": ["CLEC7A"],
    "CD54": ["ICAM1"],
    "CD71": ["TFRC"],
    "CD47": ["CD47"],
    "CD117": ["KIT"],
    "CD314": ["KLRK1"],
    "FMC63": [],
    "mIgG1": [],
    "mIgG2a": [],
    "mIgG2b": [],
}

# CD45 isoform markers that all map to PTPRC — need symmetry-breaking noise
_CD45_ISOFORMS = {"CD45", "CD45RA", "CD45RB", "CD45RO"}

# Colour palette shared across all category-level plots
_PROGRAM_PALETTE: dict[str, tuple] = {}


def _get_program_palette(programs: dict | None = None) -> dict[str, tuple]:
    """Return a stable colour map for program names."""
    if programs is None:
        programs = MARKER_PROGRAMS
    colours = sns.color_palette("tab10", n_colors=len(programs))
    return dict(zip(programs.keys(), colours))


# ===================================================================
# Notebook setup helpers
# ===================================================================

def configure_mpl(dpi: int = 120) -> None:
    """Set standard matplotlib defaults for notebook figures."""
    plt.rcParams.update({
        "figure.dpi": dpi, "savefig.dpi": 150,
        "font.size": 9, "axes.titlesize": 10,
        "figure.facecolor": "white",
    })


def filter_non_t_markers(adata, non_t_markers: dict):
    """Drop canonical non-T-cell lineage markers from adata, protecting any
    marker that appears in a T-cell canonical program (MARKER_PROGRAMS).

    Prints a per-category report of what was removed and what was protected.

    Parameters
    ----------
    adata : AnnData
    non_t_markers : dict[str, list[str]]
        Mapping of lineage category -> marker names.

    Returns
    -------
    AnnData (a copy, unchanged if nothing matched).
    """
    protected = {m for prog in MARKER_PROGRAMS.values() for m in prog}
    present = set(adata.var_names)
    removed_by_cat, to_remove = {}, set()
    for cat, markers in non_t_markers.items():
        hits = [m for m in markers if m in present and m not in protected]
        if hits:
            removed_by_cat[cat] = hits
            to_remove.update(hits)

    if not to_remove:
        print("\nNo non-T-cell markers found in adata.var_names.")
        return adata

    keep = [v for v in adata.var_names if v not in to_remove]
    n_before = adata.n_vars
    out = adata[:, keep].copy()
    print(f"\nRemoved {len(to_remove)} non-T-cell markers "
          f"({n_before} -> {out.n_vars} vars):")
    for cat, hits in removed_by_cat.items():
        print(f"  {cat}: {', '.join(hits)}")

    protected_hits = sorted({
        m for markers in non_t_markers.values() for m in markers
        if m in present and m in protected
    })
    if protected_hits:
        print(f"  (protected by T-cell programs, kept: {', '.join(protected_hits)})")
    return out


def prepare_adata_for_model(
    adata,
    emb_data: dict,
    n_top: int = 150,
) -> tuple:
    """Select top-variable markers, build semantic_map, and filter adata.

    Adds a 'counts' layer (rounded raw values) and drops markers without
    protein embeddings.

    Returns
    -------
    adata_filtered : AnnData restricted to matched markers.
    semantic_map : torch.Tensor [n_matched, embed_dim].
    """
    adata = adata.copy()
    adata.layers["counts"] = np.round(adata.layers["raw"]).astype(np.float32)
    var_per_marker = np.var(adata.layers["counts"], axis=0)
    n_top = min(n_top, adata.n_vars)
    top_markers = adata.var_names[np.argsort(var_per_marker)[::-1][:n_top]].tolist()
    print(f"Top {n_top} most variable markers: {top_markers}")
    adata = adata[:, top_markers].copy()

    semantic_map, matched_markers, missing = build_semantic_map(adata, emb_data)
    if missing:
        print(f"Dropping {len(missing)} markers without embeddings")
        adata = adata[:, matched_markers].copy()
    print(f"adata shape: {adata.shape}, semantic_map shape: {semantic_map.shape}")
    return adata, semantic_map


def extract_model_outputs(model, adata, n_latent: int) -> tuple:
    """Extract Z, W, and denoised expression from a trained SemanticSCVI model.

    Stores X_SemanticSCVI in adata.obsm and denoised_semantic in adata.layers
    in-place.

    Returns
    -------
    Z_arr : ndarray [n_cells, n_latent].
    W_arr : ndarray [n_markers, n_latent].
    W_df  : DataFrame [n_markers, n_latent] with F0..Fn columns.
    marker_names : list[str].
    z_var : ndarray [n_latent] — per-factor variance.
    """
    Z_arr = model.get_latent_representation()
    W_df = model.get_loadings()
    W_df.columns = [f"F{i}" for i in range(n_latent)]
    denoised = model.get_normalized_expression()

    adata.obsm["X_SemanticSCVI"] = Z_arr
    adata.layers["denoised_semantic"] = (
        denoised.values if isinstance(denoised, pd.DataFrame) else denoised
    )

    W_arr = W_df.values
    marker_names = list(W_df.index)
    z_var = Z_arr.var(axis=0)

    print(f"Z shape: {Z_arr.shape}")
    print(f"W shape: {W_arr.shape}, min={W_arr.min():.4f}, max={W_arr.max():.4f}")
    print("\nTop 5 markers per factor (by W weight):")
    for col in W_df.columns:
        top5 = W_df[col].nlargest(5)
        print(f"  {col}: " + ", ".join(f"{m} ({v:.3f})" for m, v in top5.items()))

    return Z_arr, W_arr, W_df, marker_names, z_var


# ===================================================================
# Semantic map alignment
# ===================================================================

def build_semantic_map(adata, embeddings: dict, embed_dim: int = 128):
    """Align protein embeddings to adata.var order for SemanticSCVI.

    Parameters
    ----------
    adata : AnnData
        Must have var_names matching marker names in *embeddings*.
    embeddings : dict
        Must contain 'marker_names' (list[str]) and 'embeddings' (ndarray, n_markers × embed_dim).
    embed_dim : int
        Embedding dimensionality (default 128 for ProteinCLIP).

    Returns
    -------
    semantic_map : torch.Tensor, shape (n_matched, embed_dim)
    matched_markers : list[str]
    missing_markers : list[str]
    """
    import torch

    emb_names = list(embeddings["marker_names"])
    emb_matrix = np.array(embeddings["embeddings"])
    name_to_idx = {n: i for i, n in enumerate(emb_names)}

    matched, missing = [], []
    rows = []
    for marker in adata.var_names:
        if marker in name_to_idx:
            rows.append(emb_matrix[name_to_idx[marker]])
            matched.append(marker)
        else:
            missing.append(marker)

    if missing:
        warnings.warn(f"build_semantic_map: {len(missing)} markers missing embeddings: {missing}")
    if not matched:
        raise ValueError("No markers matched between adata.var_names and embeddings.")

    semantic_map = torch.tensor(np.stack(rows), dtype=torch.float32)
    return semantic_map, matched, missing


# ===================================================================
# Module 1: Program Scoring
# ===================================================================

def score_programs(
    adata,
    programs: dict[str, list[str]] | None = None,
    layer: str = "arcsinh",
) -> pd.DataFrame:
    """Score each cell for each marker program (mean expression).

    Returns DataFrame (n_cells x n_programs).
    """
    if programs is None:
        programs = MARKER_PROGRAMS

    expr = pd.DataFrame(
        adata.layers[layer],
        index=adata.obs_names,
        columns=adata.var_names,
    )

    scores: dict[str, pd.Series] = {}
    for name, markers in programs.items():
        present = [m for m in markers if m in expr.columns]
        missing = [m for m in markers if m not in expr.columns]
        if len(present) < 2:
            print(f"  {name}: skipped — only {len(present)}/{len(markers)} markers found")
            continue
        print(f"  {name}: using {len(present)}/{len(markers)} markers — {', '.join(present)}")
        if missing:
            print(f"    (missing: {', '.join(missing)})")
        scores[name] = expr[present].mean(axis=1)

    return pd.DataFrame(scores, index=adata.obs_names)


# ===================================================================
# Module 2: Factor–Program Correlation
# ===================================================================

def correlate_factors_with_programs(
    factor_scores: np.ndarray,
    program_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson r between each factor and each program score.

    Returns (r_df, p_df) both shaped (n_factors x n_programs).
    """
    n_factors = factor_scores.shape[1]
    programs = program_scores.columns.tolist()
    factor_labels = [f"F{i}" for i in range(n_factors)]

    r_mat = np.zeros((n_factors, len(programs)))
    p_mat = np.zeros_like(r_mat)

    for i in range(n_factors):
        for j, prog in enumerate(programs):
            mask = np.isfinite(program_scores[prog].values)
            r, p = pearsonr(factor_scores[mask, i], program_scores[prog].values[mask])
            r_mat[i, j] = r
            p_mat[i, j] = p

    r_df = pd.DataFrame(r_mat, index=factor_labels, columns=programs)
    p_df = pd.DataFrame(p_mat, index=factor_labels, columns=programs)
    return r_df, p_df


def compute_marker_factor_loadings(
    adata,
    factor_scores: np.ndarray,
    layer: str = "arcsinh",
) -> pd.DataFrame:
    """Proxy loadings: Pearson r between each marker and each factor.

    Returns DataFrame (n_markers x n_factors).
    """
    expr = adata.layers[layer]
    n_factors = factor_scores.shape[1]
    factor_labels = [f"F{i}" for i in range(n_factors)]
    r_mat = np.zeros((adata.n_vars, n_factors))

    for j in range(n_factors):
        for i in range(adata.n_vars):
            r, _ = pearsonr(expr[:, i], factor_scores[:, j])
            r_mat[i, j] = r

    return pd.DataFrame(r_mat, index=adata.var_names, columns=factor_labels)


# ===================================================================
# Module 3: Factor–Metadata Association
# ===================================================================

def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    return 1.0 - 2.0 * u_stat / (n1 * n2)


def test_factor_metadata_categorical(
    factor_scores: np.ndarray,
    obs: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Test association between each factor and each categorical metadata column.

    Mann-Whitney U (2 groups) or Kruskal-Wallis (>2 groups), BH-FDR per factor.
    """
    n_factors = factor_scores.shape[1]
    rows: list[dict] = []

    for fi in range(n_factors):
        fvals = factor_scores[:, fi]
        for col in columns:
            if col not in obs.columns:
                continue
            groups = obs[col].dropna().unique()
            if len(groups) < 2:
                continue

            if len(groups) == 2:
                g0, g1 = groups
                a = fvals[obs[col] == g0]
                b = fvals[obs[col] == g1]
                u_stat, p = mannwhitneyu(a, b, alternative="two-sided")
                effect = _rank_biserial(u_stat, len(a), len(b))
                rows.append(dict(factor=f"F{fi}", metadata=col, test="mann_whitney",
                                 stat=u_stat, pval=p, effect_size=effect,
                                 groups=f"{g0} vs {g1}", n_groups=2))
            else:
                group_vals = [fvals[obs[col] == g] for g in groups]
                h_stat, p = kruskal(*group_vals)
                n_total = sum(len(gv) for gv in group_vals)
                k = len(groups)
                eta2 = (h_stat - k + 1) / (n_total - k)
                rows.append(dict(factor=f"F{fi}", metadata=col, test="kruskal_wallis",
                                 stat=h_stat, pval=p, effect_size=eta2,
                                 groups=", ".join(str(g) for g in groups), n_groups=k))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    padj_all = np.ones(len(df))
    for fi_label in df["factor"].unique():
        mask = df["factor"] == fi_label
        _, padj, _, _ = multipletests(df.loc[mask, "pval"], method="fdr_bh")
        padj_all[mask.values] = padj
    df["padj"] = padj_all
    return df


# ===================================================================
# Module 4: Pseudotime helpers
# ===================================================================

def select_root_cell(
    adata,
    layer: str = "arcsinh",
    naive_markers: list[str] | None = None,
    activation_markers: list[str] | None = None,
) -> str:
    """Pick the most naive T cell as pseudotime root (max naive − activation score)."""
    if naive_markers is None:
        naive_markers = ["CD45RA", "CD27", "CD28", "CD127", "CD7"]
    if activation_markers is None:
        activation_markers = ["CD38", "HLA-DR", "TIGIT", "CD95", "CD25", "CD69", "KLRG1"]

    expr = pd.DataFrame(adata.layers[layer], index=adata.obs_names, columns=adata.var_names)
    naive_present = [m for m in naive_markers if m in expr.columns]
    act_present   = [m for m in activation_markers if m in expr.columns]
    score = expr[naive_present].mean(axis=1) - expr[act_present].mean(axis=1)
    return score.idxmax()


def correlate_factors_with_pseudotime(
    factor_scores: np.ndarray,
    pseudotime: pd.Series,
) -> pd.DataFrame:
    """Pearson + Spearman correlation between each factor and pseudotime."""
    mask = np.isfinite(pseudotime.values)
    pt = pseudotime.values[mask]
    fs = factor_scores[mask]

    rows = []
    for i in range(fs.shape[1]):
        pr, pp = pearsonr(fs[:, i], pt)
        sr, sp = spearmanr(fs[:, i], pt)
        rows.append(dict(factor=f"F{i}", pearson_r=pr, pearson_p=pp,
                         spearman_r=sr, spearman_p=sp))
    return pd.DataFrame(rows)


# ===================================================================
# Module 5: Interpretation Summary
# ===================================================================

def build_interpretation_table(
    r_program: pd.DataFrame,
    p_program: pd.DataFrame,
    meta_results: pd.DataFrame,
    bio_threshold: float = 0.05,
    meta_threshold: float = 0.05,
) -> pd.DataFrame:
    """Classify each factor: biology_aligned / batch_effect / novel_axis / noise."""
    rows = []
    for factor in r_program.index:
        abs_r = r_program.loc[factor].abs()
        best_prog = abs_r.idxmax()
        best_r    = r_program.loc[factor, best_prog]
        best_p_prog = p_program.loc[factor, best_prog]

        fmeta = meta_results[meta_results["factor"] == factor]
        if len(fmeta) > 0:
            best_meta_row  = fmeta.loc[fmeta["effect_size"].abs().idxmax()]
            best_meta      = best_meta_row["metadata"]
            best_meta_eff  = best_meta_row["effect_size"]
            best_meta_padj = best_meta_row["padj"]
        else:
            best_meta = "N/A"; best_meta_eff = 0.0; best_meta_padj = 1.0

        sig_prog = best_p_prog < bio_threshold
        sig_meta = best_meta_padj < meta_threshold

        if sig_meta and sig_prog:
            cls = "biology_aligned"
        elif sig_meta:
            cls = "batch_effect"
        elif sig_prog:
            cls = "novel_axis"
        else:
            cls = "noise"

        rows.append(dict(factor=factor, top_program=best_prog,
                         program_r=round(best_r, 3), program_p=best_p_prog,
                         top_metadata=best_meta, metadata_effect=round(best_meta_eff, 3),
                         metadata_padj=best_meta_padj, classification=cls))
    return pd.DataFrame(rows)


# ===================================================================
# Module 6: Heatmaps
# ===================================================================

def build_factor_metadata_heatmap(
    test_results: pd.DataFrame,
    metric: str = "effect_size",
    figsize: tuple[float, float] = (8, 6),
):
    """Heatmap of factor–metadata associations (effect size + sig stars)."""
    pivot      = test_results.pivot(index="factor", columns="metadata", values=metric)
    pval_pivot = test_results.pivot(index="factor", columns="metadata", values="padj")
    annot = pval_pivot.map(sig_label)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, annot=pivot, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": metric})
    ax.set_title("Factor–Metadata Associations")
    ax.set_ylabel(""); ax.set_xlabel("")
    plt.tight_layout()
    return fig, ax


def build_factor_program_heatmap(
    r_values: pd.DataFrame,
    p_values: pd.DataFrame,
    figsize: tuple[float, float] = (9, 6),
):
    """Heatmap of factor–program Pearson correlations with sig stars."""
    annot = p_values.map(sig_label)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(r_values, annot=r_values, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=ax, vmin=-0.5, vmax=0.5,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Factor–Program Correlations")
    ax.set_ylabel(""); ax.set_xlabel("")
    plt.tight_layout()
    return fig, ax


# ===================================================================
# Module 7: Plotting
# ===================================================================

def plot_loadings_heatmap(
    loadings: pd.DataFrame,
    programs: dict[str, list[str]] | None = None,
    nonneg: bool = False,
):
    """Loadings heatmap restricted to program markers, grouped by category.

    Only markers defined in *programs* are shown — in category order
    (activation → exhaustion → …). Factors (columns) are clustered by
    dendrogram. A colour bar on the left identifies each category.

    Parameters
    ----------
    nonneg : bool
        If True, use a sequential colourmap (YlOrRd) with vmin=0 —
        appropriate for non-negative decoder weights (e.g. SemanticSCVI W).
        If False (default), use RdBu_r centred at 0 for correlations.

    Returns (fig, clustermap_object).
    """
    if programs is None:
        programs = MARKER_PROGRAMS

    palette = _get_program_palette(programs)

    # Build ordered marker list from programs only (skip any missing from loadings)
    marker_order: list[str] = []
    marker_cat:   dict[str, str] = {}
    for prog_name, markers in programs.items():
        for m in markers:
            if m in loadings.index and m not in marker_cat:
                marker_order.append(m)
                marker_cat[m] = prog_name

    if not marker_order:
        raise ValueError("None of the program markers are present in the loadings index.")

    plot_data  = loadings.loc[marker_order]
    row_colors = pd.Series(
        [palette[marker_cat[m]] for m in marker_order],
        index=marker_order,
        name="program",
    )

    # Colourmap and scale depend on whether loadings are non-negative weights
    if nonneg:
        cmap_kwargs = dict(cmap="YlOrRd", vmin=0, vmax=plot_data.values.max() * 1.05)
        cbar_label = "Loading weight"
        title = "Decoder Loadings W (SemanticSCVI)\n(grouped by program)"
    else:
        cmap_kwargs = dict(cmap="RdBu_r", center=0, vmin=-0.7, vmax=0.7)
        cbar_label = "Pearson r"
        title = "Proxy Loadings: Marker–Factor Correlations\n(grouped by program)"

    height = max(7, len(marker_order) * 0.32)
    g = sns.clustermap(
        plot_data,
        **cmap_kwargs,
        linewidths=0.3,
        row_cluster=False,   # preserve category order
        col_cluster=True,    # cluster factors by similarity
        row_colors=row_colors,
        yticklabels=[display_name(m) for m in marker_order],
        figsize=(11, height),
        cbar_pos=(1.02, 0.4, 0.02, 0.2),
        cbar_kws={"label": cbar_label},
        dendrogram_ratio=(0.0, 0.12),
    )
    g.ax_heatmap.set_title(title, pad=10)
    g.ax_heatmap.set_xlabel("Factor")
    g.ax_heatmap.set_ylabel("")

    # Legend for category colours
    used_cats = dict.fromkeys(marker_cat[m] for m in marker_order)  # preserve order
    patches = [mpatches.Patch(color=palette[c], label=c)
               for c in used_cats]
    g.ax_heatmap.legend(handles=patches, loc="upper right",
                        bbox_to_anchor=(1.25, 1.05), fontsize=8,
                        title="Program", title_fontsize=8, frameon=True)
    return g.fig, g


def plot_factor_metadata_violins(
    factor_scores: np.ndarray,
    meta_results: pd.DataFrame,
    obs: pd.DataFrame,
    top_n: int = 6,
    figsize: tuple[float, float] | None = None,
):
    """Violin plots for the top-N factor–metadata associations."""
    top_pairs = meta_results.sort_values("padj").head(top_n)
    ncols = 3
    nrows = (top_n + ncols - 1) // ncols
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for idx, (_, row) in enumerate(top_pairs.iterrows()):
        ax = axes.flat[idx]
        fi  = int(row["factor"][1:])
        col = row["metadata"]
        df_plot = pd.DataFrame({"factor_score": factor_scores[:, fi],
                                col: obs[col].values})
        sns.violinplot(data=df_plot, x=col, y="factor_score",
                       ax=ax, cut=0, inner="box")
        ax.set_title(f"{row['factor']} vs {col}\n(padj={row['padj']:.2e})", fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.set_xlabel(""); ax.set_ylabel(f"Factor {fi} score")

    for idx in range(top_n, nrows * ncols):
        axes.flat[idx].set_visible(False)

    plt.suptitle("Top Factor–Metadata Associations", y=1.02, fontsize=13)
    plt.tight_layout()
    return fig, axes


def plot_factor_pseudotime_grid(
    factor_scores: np.ndarray,
    pseudotime: np.ndarray,
    pt_corr: pd.DataFrame,
    condition: np.ndarray,
    n_latent: int = 10,
    condition_colors: dict | None = None,
    figsize: tuple[float, float] = (20, 7),
):
    """2×(n_latent/2) grid of factor scores vs pseudotime, coloured by condition."""
    if condition_colors is None:
        unique_conds = list(dict.fromkeys(condition))
        pal = sns.color_palette("tab10", n_colors=len(unique_conds))
        condition_colors = dict(zip(unique_conds, pal))

    ncols = 5
    nrows = (n_latent + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    for i in range(n_latent):
        ax = axes.flat[i]
        for cond_label, color in condition_colors.items():
            mask = condition == cond_label
            ax.scatter(pseudotime[mask], factor_scores[mask, i],
                       c=color, s=2, alpha=0.3, label=cond_label, rasterized=True)
        rho = pt_corr.loc[pt_corr["factor"] == f"F{i}", "spearman_r"].values[0]
        ax.set_title(f"F{i} (ρ={rho:.2f})", fontsize=9)
        ax.set_xlabel("Pseudotime" if i >= (nrows - 1) * ncols else "")
        ax.set_ylabel("Factor score" if i % ncols == 0 else "")

    axes.flat[0].legend(fontsize=7, markerscale=4)
    for idx in range(n_latent, nrows * ncols):
        axes.flat[idx].set_visible(False)

    plt.suptitle("Factor Scores along Pseudotime (CD8 T cells)", y=1.01, fontsize=13)
    plt.tight_layout()
    return fig, axes


def plot_pseudotime_distributions(
    pseudotime: np.ndarray,
    obs: pd.DataFrame,
    figsize: tuple[float, float] = (12, 4),
):
    """Pseudotime density histograms by condition and condition×time."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for cond in obs["condition"].unique():
        mask = obs["condition"] == cond
        axes[0].hist(pseudotime[mask], bins=50, alpha=0.5, label=cond, density=True)
    axes[0].set(xlabel="Pseudotime", ylabel="Density", title="Pseudotime by Condition")
    axes[0].legend()

    for cond in obs["condition"].unique():
        for time in obs["time"].unique():
            mask = (obs["condition"] == cond) & (obs["time"] == time)
            if mask.sum() > 0:
                axes[1].hist(pseudotime[mask], bins=40, alpha=0.4, density=True,
                             label=f"{cond} {time}")
    axes[1].set(xlabel="Pseudotime", ylabel="Density",
                title="Pseudotime by Condition × Time")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    return fig, axes


def plot_program_correlation_matrix(
    program_scores: pd.DataFrame,
    figsize: tuple[float, float] = (8, 7),
):
    """Heatmap of program–program Pearson correlations."""
    prog_corr = program_scores.corr(method="pearson")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(prog_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Program–Program Correlations\n(independent programs = independent biology)")
    plt.tight_layout()
    return fig, ax


def plot_metadata_scatter_grid(
    factor_scores: np.ndarray,
    fx: int,
    fy: int,
    obs: pd.DataFrame,
    meta_cols: list[str],
    figsize: tuple[float, float] | None = None,
):
    """Row of 2D factor scatters coloured by different metadata columns."""
    if figsize is None:
        figsize = (5 * len(meta_cols), 4)
    fig, axes = plt.subplots(1, len(meta_cols), figsize=figsize)
    if len(meta_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, meta_cols):
        plot_factor_scatter(factor_scores, fx, fy, obs[col].values,
                            color_label=col, categorical=True, ax=ax)
    plt.suptitle(f"Factor F{fx} vs F{fy} — Metadata", y=1.02, fontsize=13)
    plt.tight_layout()
    return fig, axes


def plot_program_scatter_grid(
    factor_scores: np.ndarray,
    fx: int,
    fy: int,
    program_scores: pd.DataFrame,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
):
    """Grid of 2D factor scatters coloured by program scores."""
    n_progs = len(program_scores.columns)
    nrows = (n_progs + ncols - 1) // ncols
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for idx, prog in enumerate(program_scores.columns):
        plot_factor_scatter(factor_scores, fx, fy, program_scores[prog].values,
                            color_label=prog, categorical=False, ax=axes.flat[idx])
    for idx in range(n_progs, nrows * ncols):
        axes.flat[idx].set_visible(False)
    plt.suptitle(f"Factor F{fx} vs F{fy} — Program Scores", y=1.01, fontsize=13)
    plt.tight_layout()
    return fig, axes


def plot_umap_factor_scores(
    umap_coords: np.ndarray,
    factor_scores: np.ndarray,
    n_latent: int = 10,
    umap_key: str = "UMAP",
    figsize: tuple[float, float] = (20, 7),
):
    """2×(n_latent/2) UMAP grid coloured by each factor score."""
    ncols = 5
    nrows = (n_latent + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    for i in range(n_latent):
        ax = axes.flat[i]
        sc_plot = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                             c=factor_scores[:, i], s=2, alpha=0.4,
                             cmap="RdBu_r", rasterized=True)
        ax.set_title(f"F{i}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc_plot, ax=ax, shrink=0.6)

    for idx in range(n_latent, nrows * ncols):
        axes.flat[idx].set_visible(False)

    plt.suptitle(f"UMAP ({umap_key}) coloured by Factor Scores", y=1.01, fontsize=13)
    plt.tight_layout()
    return fig, axes


def plot_factor_scatter(
    factor_scores: np.ndarray,
    fx: int,
    fy: int,
    color_values,
    color_label: str = "",
    categorical: bool = False,
    ax=None,
    **kwargs,
):
    """2D scatter of cells in factor space, coloured by a metadata or continuous value."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    if categorical:
        cats = pd.Categorical(color_values)
        palette = sns.color_palette("tab10", n_colors=len(cats.categories))
        for ci, cat in enumerate(cats.categories):
            mask = cats == cat
            ax.scatter(factor_scores[mask, fx], factor_scores[mask, fy],
                       c=[palette[ci]], label=cat, s=4, alpha=0.4, rasterized=True)
        ax.legend(fontsize=7, markerscale=3, loc="best")
    else:
        sc = ax.scatter(factor_scores[:, fx], factor_scores[:, fy],
                        c=color_values, s=4, alpha=0.4, cmap="viridis",
                        rasterized=True, **kwargs)
        plt.colorbar(sc, ax=ax, label=color_label, shrink=0.8)

    ax.set_xlabel(f"Factor {fx}")
    ax.set_ylabel(f"Factor {fy}")
    ax.set_title(color_label)
    return ax


# ===================================================================
# Gaublomme-style pipeline functions (ported from gaublomme_utils.py)
# Adapted for PixelGen: protein markers, no mouse2human mapping,
# arcsinh layer, condition/time/cell_system metadata.
# ===================================================================

# ---------------------------------------------------------------------------
# Embedding distance validation
# ---------------------------------------------------------------------------

# Default similar / dissimilar marker pairs for embedding QC
_SIMILAR_PAIRS = [
    ('CD158a', 'CD158b'), ('IgD', 'IgM'), ('CD11a', 'CD11b'),
    ('CD1a', 'CD1c'), ('CD150', 'CD84'), ('CD279', 'CD152'),
    ('CD80', 'CD86'), ('CD16', 'CD32'), ('CD191', 'CD192'),
    ('CD134', 'CD137'),
]
_DISSIMILAR_PAIRS = [
    ('CD3e', 'CD19'), ('CD4', 'CD14'), ('CD56', 'CD19'),
    ('CD8', 'CD20'), ('CD94', 'CD79a'), ('HLA-DR', 'CD16'),
    ('CD3e', 'CD14'), ('CD45RA', 'CD163'), ('CD28', 'CD36'),
    ('TIGIT', 'CD41'),
]


def plot_embedding_distances(
    emb_data: dict,
    similar_pairs: list[tuple[str, str]] | None = None,
    dissimilar_pairs: list[tuple[str, str]] | None = None,
    results_dir=None,
) -> pd.DataFrame:
    """Bar chart of cosine distances for similar vs dissimilar marker pairs.

    Returns a DataFrame with columns [type, marker_A, marker_B, cosine_distance].
    """
    from scipy.spatial.distance import cosine as cosine_dist

    if similar_pairs is None:
        similar_pairs = _SIMILAR_PAIRS
    if dissimilar_pairs is None:
        dissimilar_pairs = _DISSIMILAR_PAIRS

    emb_lookup = dict(zip(emb_data['marker_names'], emb_data['embeddings']))
    rows = []
    for label, pairs in [('similar', similar_pairs), ('dissimilar', dissimilar_pairs)]:
        for a, b in pairs:
            if a in emb_lookup and b in emb_lookup:
                d = cosine_dist(emb_lookup[a], emb_lookup[b])
                rows.append({'type': label, 'marker_A': a, 'marker_B': b,
                             'cosine_distance': round(d, 4)})
    df = pd.DataFrame(rows)
    sim_mean = df.loc[df.type == 'similar', 'cosine_distance'].mean()
    dis_mean = df.loc[df.type == 'dissimilar', 'cosine_distance'].mean()
    print(f'Similar mean={sim_mean:.4f}  Dissimilar mean={dis_mean:.4f}  '
          f'Separation ratio: {dis_mean / sim_mean:.2f}x')

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {'similar': '#2196F3', 'dissimilar': '#F44336'}
    all_labels = []
    n_sim = df[df.type == 'similar'].shape[0]
    for typ in ['similar', 'dissimilar']:
        sub = df[df.type == typ]
        offset = 0 if typ == 'similar' else n_sim
        x = range(offset, offset + len(sub))
        ax.bar(x, sub.cosine_distance.values, color=colors[typ],
               label=typ, edgecolor='white')
        all_labels += [f'{r.marker_A} vs\n{r.marker_B}' for _, r in sub.iterrows()]
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Cosine distance')
    ax.set_title('Functional Embedding Distances')
    ax.legend()
    ax.axvline(n_sim - 0.5, color='gray', ls='--', lw=0.8)
    plt.tight_layout()
    if results_dir:
        fig.savefig(Path(results_dir) / 'embedding_distances.png',
                    dpi=150, bbox_inches='tight')
    plt.show()
    return df


# ---------------------------------------------------------------------------
# Training health check
# ---------------------------------------------------------------------------

def plot_training_health(model, results_dir=None):
    """4-panel loss curves for SemanticSCVI training."""
    history = model.history
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ELBO
    ax = axes[0, 0]
    for key, label in [("elbo_train", "train"), ("elbo_validation", "val")]:
        if key in history:
            ax.plot(history[key].index, history[key].values, label=label)
    ax.set_title("ELBO"); ax.set_xlabel("Epoch"); ax.legend()

    # Reconstruction loss
    ax = axes[0, 1]
    for key, label in [("reconstruction_loss_train", "train"),
                       ("reconstruction_loss_validation", "val")]:
        if key in history:
            ax.plot(history[key].index, history[key].values, label=label)
    ax.set_title("Reconstruction Loss"); ax.set_xlabel("Epoch"); ax.legend()

    # KL divergence
    ax = axes[1, 0]
    for key, label, ls in [("kl_local_train", "kl_local train", "-"),
                           ("kl_local_validation", "kl_local val", "-"),
                           ("kl_global_train", "kl_global train", "--")]:
        if key in history:
            ax.plot(history[key].index, history[key].values, label=label, ls=ls)
    ax.set_title("KL Divergence"); ax.set_xlabel("Epoch"); ax.legend()

    # Semantic coherence loss
    ax = axes[1, 1]
    if "coherence_loss_train" in history:
        ax.plot(history["coherence_loss_train"].index,
                history["coherence_loss_train"].values, label="coherence")
    if "semantic_scale_train" in history:
        ax2 = ax.twinx()
        ax2.plot(history["semantic_scale_train"].index,
                 history["semantic_scale_train"].values,
                 label="semantic_scale", color="tab:orange", ls="--")
        ax2.set_ylabel("Semantic Scale"); ax2.legend(loc="upper left")
    ax.set_title("Semantic Coherence Loss"); ax.set_xlabel("Epoch")
    ax.legend(loc="upper right")

    fig.suptitle("SemanticSCVI Training Health Check", fontsize=14, y=1.01)
    fig.tight_layout()
    if results_dir:
        fig.savefig(Path(results_dir) / "training_health_check.pdf", bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Pseudotime & UMAP
# ---------------------------------------------------------------------------

def compute_pseudotime(adata, Z_arr, seed=42):
    """Compute diffusion pseudotime on SemanticSCVI latent space.

    Returns pseudotime array and stores in adata.obs["dpt_pseudotime"].
    """
    if "X_SemanticSCVI" not in adata.obsm:
        adata.obsm["X_SemanticSCVI"] = Z_arr
    sc.pp.neighbors(adata, use_rep="X_SemanticSCVI", n_neighbors=20, random_state=seed)
    sc.tl.diffmap(adata)
    root_cell = select_root_cell(adata, layer="arcsinh")
    adata.uns["iroot"] = np.where(adata.obs_names == root_cell)[0][0]
    sc.tl.dpt(adata)
    pseudotime = adata.obs["dpt_pseudotime"].values
    print(f"Pseudotime range: {np.nanmin(pseudotime):.3f} - {np.nanmax(pseudotime):.3f}")
    print(f"Root cell: {root_cell}")
    return pseudotime


def plot_umap_overview(adata, meta_cols=("condition", "time", "cell_system"),
                       seed=42, results_dir=None):
    """Compute UMAP on SemanticSCVI latent and plot overview panels."""
    sc.pp.neighbors(adata, use_rep="X_SemanticSCVI", n_neighbors=20, random_state=seed)
    sc.tl.umap(adata, min_dist=0.3, random_state=seed)
    adata.obsm["X_umap_semantic"] = adata.obsm["X_umap"].copy()

    n = len(meta_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, meta_cols):
        if col in adata.obs.columns:
            sc.pl.umap(adata, color=col, ax=ax, show=False, frameon=False)
    fig.suptitle("SemanticSCVI UMAP Overview", fontsize=13, y=1.02)
    plt.tight_layout()
    if results_dir:
        fig.savefig(Path(results_dir) / "semantic_umap_overview.pdf", bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# GEP Discovery (Leiden on markers in W-space)
# ---------------------------------------------------------------------------

def discover_marker_programs(
    W_arr: np.ndarray,
    marker_names: list,
    n_top: int = 50,
    n_neighbors: int = 10,
    resolution: float = 1.0,
    metric: str = "cosine",
    seed: int = 42,
) -> dict:
    """Cluster top markers in W-space using Leiden.

    Parameters
    ----------
    W_arr : (n_markers, n_factors) decoder weight matrix.
    marker_names : marker names corresponding to rows of W_arr.
    n_top : number of top markers to cluster (by L2 norm of W row).
    n_neighbors, resolution, metric : Leiden parameters.
    seed : random seed.

    Returns
    -------
    dict with keys: labels, n_programs, program_genes, program_sizes,
        marker_adata, top_marker_idx, top_marker_names.
    """
    w_norms = np.linalg.norm(W_arr, axis=1)
    n_top = min(n_top, len(marker_names))
    top_idx = np.argsort(w_norms)[::-1][:n_top]
    top_idx = np.sort(top_idx)

    W_top = W_arr[top_idx]
    top_names = [marker_names[i] for i in top_idx]

    marker_ad = anndata.AnnData(X=W_top.copy())
    marker_ad.obs_names = pd.Index(top_names)
    marker_ad.obs["w_norm"] = w_norms[top_idx]

    sc.pp.neighbors(marker_ad, n_neighbors=n_neighbors, metric=metric,
                    use_rep="X", random_state=seed)
    sc.tl.leiden(marker_ad, resolution=resolution, random_state=seed)
    sc.tl.umap(marker_ad, random_state=seed)

    labels = marker_ad.obs["leiden"].values.astype(int)
    program_genes = {}
    program_sizes = {}
    for pid in sorted(np.unique(labels)):
        mask = labels == pid
        markers = [top_names[i] for i in np.where(mask)[0]]
        program_genes[pid] = markers
        program_sizes[pid] = len(markers)

    return {
        "labels": labels,
        "n_programs": len(program_sizes),
        "program_genes": program_genes,
        "program_sizes": program_sizes,
        "marker_adata": marker_ad,
        "top_marker_idx": top_idx,
        "top_marker_names": top_names,
    }


def characterize_programs_from_W(
    W_arr: np.ndarray,
    marker_names: list,
    programs: dict,
) -> pd.DataFrame:
    """Characterize each marker program: markers, mean loading profile, dominant factors."""
    n_factors = W_arr.shape[1]
    name_to_idx = {g: i for i, g in enumerate(marker_names)}
    rows = []

    for pid in sorted(programs["program_genes"]):
        markers = programs["program_genes"][pid]
        idx = [name_to_idx[m] for m in markers]
        W_sub = W_arr[idx]
        mean_load = W_sub.mean(axis=0)
        max_load = mean_load.max()
        dominant = [f for f in range(n_factors) if mean_load[f] > 0.5 * max_load]

        w_norms = np.linalg.norm(W_sub, axis=1)
        sort_order = np.argsort(w_norms)[::-1]
        sorted_markers = [markers[i] for i in sort_order]

        rows.append({
            "program": pid,
            "size": len(markers),
            "marker_list": ", ".join(sorted_markers),
            "dominant_factors": dominant,
            "mean_loading": mean_load.tolist(),
        })

    return pd.DataFrame(rows)


def plot_marker_umap(marker_ad, results_dir=None):
    """UMAP of markers in W-space colored by Leiden program."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(marker_ad, color="leiden", ax=ax, show=False, legend_loc="on data",
               title="Marker Programs in W-space (Leiden)")
    plt.tight_layout()
    if results_dir:
        plt.savefig(Path(results_dir) / "marker_programs_umap.png",
                    dpi=150, bbox_inches="tight")
    plt.show()


def plot_program_loadings_w(W_arr, marker_names, programs, results_dir=None):
    """Heatmap of mean W-loading per program x factor."""
    name_to_idx = {g: i for i, g in enumerate(marker_names)}
    n_factors = W_arr.shape[1]
    pids = sorted(programs["program_genes"].keys())

    mat = np.zeros((len(pids), n_factors))
    for row, pid in enumerate(pids):
        markers = programs["program_genes"][pid]
        idx = [name_to_idx[m] for m in markers]
        mat[row] = W_arr[idx].mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, max(3, len(pids) * 0.4)))
    sns.heatmap(mat, ax=ax, cmap="YlOrRd",
                xticklabels=[f"Z[{f}]" for f in range(n_factors)],
                yticklabels=[f"GEP_{pid} (n={programs['program_sizes'][pid]})"
                             for pid in pids],
                annot=True, fmt=".2f", annot_kws={"fontsize": 7})
    ax.set_title("Mean W-loading per Marker Program x Factor")
    plt.tight_layout()
    if results_dir:
        plt.savefig(Path(results_dir) / "program_loading_profiles.png",
                    dpi=150, bbox_inches="tight")
    plt.show()


def run_gep_discovery(
    W_arr, marker_names,
    n_top=50, n_neighbors=10, resolution=1.0,
    seed=42, results_dir=None,
):
    """Discover marker programs, characterize, and plot.

    Returns (programs, program_char_df).
    """
    programs = discover_marker_programs(
        W_arr, marker_names, n_top=n_top,
        n_neighbors=n_neighbors, resolution=resolution,
        metric="cosine", seed=seed,
    )
    print(f"Discovered {programs['n_programs']} GEPs from {n_top} markers")
    for pid in sorted(programs["program_sizes"]):
        print(f"  GEP_{pid}: {programs['program_sizes'][pid]} markers")

    program_char_df = characterize_programs_from_W(W_arr, marker_names, programs)
    for _, row in program_char_df.iterrows():
        pid = row["program"]
        print(f"\n--- GEP_{pid} ({row['size']} markers, dominant: {row['dominant_factors']}) ---")
        print(f"Markers: {row['marker_list']}")

    plot_marker_umap(programs["marker_adata"], results_dir=results_dir)
    plot_program_loadings_w(W_arr, marker_names, programs, results_dir=results_dir)

    return programs, program_char_df


# ---------------------------------------------------------------------------
# Factor Diagnostics (orthogonality, marker heatmap, hierarchical modules)
# ---------------------------------------------------------------------------

def _plot_factor_marker_heatmap(W_arr, marker_names, n_top=50, results_dir=None):
    """Clustermap of top-loading markers x factors."""
    loadings = pd.DataFrame(W_arr, index=marker_names)
    max_loadings = loadings.abs().max(axis=1)
    n_top = min(n_top, len(marker_names))
    top_markers = max_loadings.sort_values(ascending=False).head(n_top).index
    subset = loadings.loc[top_markers]
    subset.columns = [f"Z[{f}]" for f in range(W_arr.shape[1])]

    g = sns.clustermap(
        subset, method="ward", metric="euclidean", z_score=None,
        cmap="viridis", col_cluster=False, row_cluster=True,
        figsize=(max(8, W_arr.shape[1] * 0.8), min(18, n_top * 0.35)),
        yticklabels=True,
    )
    g.fig.suptitle(f"Top {n_top} Marker Loadings x Factors", y=1.02, fontsize=12)
    if results_dir:
        g.savefig(Path(results_dir) / "factor_marker_clustermap.png",
                  dpi=150, bbox_inches="tight")
    plt.show()


def _plot_marker_module_detection(W_arr, marker_names, n_top=50, max_k=15,
                                  results_dir=None):
    """Detect marker modules via hierarchical clustering with silhouette-optimized k.

    Returns dict with n_modules, silhouette, module_genes, labels, top_marker_names.
    """
    loadings = pd.DataFrame(W_arr, index=marker_names)
    max_vals = loadings.abs().max(axis=1)
    n_top = min(n_top, len(marker_names))
    top_idx = max_vals.sort_values(ascending=False).head(n_top).index
    subset = loadings.loc[top_idx]

    dists = pdist(subset.values, metric="correlation")
    dist_matrix = squareform(dists)
    Z_link = linkage(dists, method="average")

    best_k, best_score = 2, -1
    for k in range(2, min(max_k + 1, n_top)):
        labels = fcluster(Z_link, t=k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(dist_matrix, labels, metric="precomputed")
        if score > best_score:
            best_score = score
            best_k = k

    final_labels = fcluster(Z_link, t=best_k, criterion="maxclust")
    top_names = list(top_idx)

    module_genes = {}
    print(f"Detected {best_k} marker modules (silhouette: {best_score:.3f})")
    for cid in range(1, best_k + 1):
        mask = final_labels == cid
        markers = [top_names[i] for i in np.where(mask)[0]]
        module_genes[cid] = markers
        print(f"  Module {cid} ({len(markers)} markers): {', '.join(markers[:30])}"
              + ("..." if len(markers) > 30 else ""))

    # Plot gene-gene correlation heatmap with module colors
    cluster_colors = pd.Series(final_labels, index=top_idx).map(
        dict(zip(range(1, best_k + 1), sns.color_palette("tab20", best_k)))
    )
    g = sns.clustermap(
        subset.T.corr(),
        row_linkage=Z_link, col_linkage=Z_link,
        row_colors=cluster_colors, col_colors=cluster_colors,
        cmap="vlag", center=0,
        xticklabels=True, yticklabels=True,
        figsize=(8, 8),
    )
    g.fig.suptitle(f"{best_k} Marker Modules (silhouette={best_score:.3f})",
                   y=1.02, fontsize=12)
    if results_dir:
        g.savefig(Path(results_dir) / "marker_module_heatmap.png",
                  dpi=150, bbox_inches="tight")
    plt.show()

    return {
        "n_modules": best_k,
        "silhouette": best_score,
        "module_genes": module_genes,
        "labels": final_labels,
        "top_marker_names": top_names,
    }


def plot_factor_diagnostics(
    Z_arr, W_arr, marker_names,
    n_factors=None, n_top_marker=50, n_top_module=50, max_k=15,
    results_dir=None,
):
    """Combined diagnostics: orthogonality + marker heatmap + modules.

    Returns module_info dict.
    """
    if n_factors is None:
        n_factors = Z_arr.shape[1]

    # 1. Factor orthogonality
    z_corr = np.corrcoef(Z_arr.T)
    fig, ax = plt.subplots(figsize=(7, 6))
    mask_tri = np.triu(np.ones_like(z_corr, dtype=bool), k=1)
    sns.heatmap(
        z_corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"fontsize": 8},
        xticklabels=[f"Z[{f}]" for f in range(n_factors)],
        yticklabels=[f"Z[{f}]" for f in range(n_factors)],
        mask=mask_tri,
    )
    ax.set_title("Z Factor Correlation Matrix (Pearson)")
    plt.tight_layout()
    if results_dir:
        plt.savefig(Path(results_dir) / "factor_orthogonality.png",
                    dpi=150, bbox_inches="tight")
    plt.show()

    print("\nCorrelated factor pairs (|r| > 0.3):")
    found = False
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            r = z_corr[i, j]
            if abs(r) > 0.3:
                print(f"  Z[{i}] - Z[{j}]: r={r:.3f}")
                found = True
    if not found:
        print("  None — factors are approximately orthogonal.")

    # 2. Marker loading clustermap
    print(f"\n--- Top {n_top_marker} Marker Loading Clustermap ---")
    _plot_factor_marker_heatmap(W_arr, marker_names, n_top=n_top_marker,
                                results_dir=results_dir)

    # 3. Marker module detection
    print(f"\n--- Marker Module Detection (top {n_top_module}, max_k={max_k}) ---")
    module_info = _plot_marker_module_detection(
        W_arr, marker_names, n_top=n_top_module, max_k=max_k,
        results_dir=results_dir,
    )

    return module_info


# ---------------------------------------------------------------------------
# Program Scoring (data-derived modules)
# ---------------------------------------------------------------------------

def compute_program_scores(
    adata, program_markers, selected_programs=None, layer="arcsinh",
) -> pd.DataFrame:
    """Compute per-cell z-scored program scores from discovered modules.

    Parameters
    ----------
    adata : AnnData with the specified layer.
    program_markers : dict[int, list[str]], marker lists per module.
    selected_programs : list of int, programs to score. None = all.
    layer : str, layer to use.

    Returns DataFrame (n_cells, n_selected_programs), z-scored.
    """
    if selected_programs is None:
        selected_programs = sorted(program_markers.keys())

    expr = adata.layers[layer]
    var_names = list(adata.var_names)
    var_idx = {g: i for i, g in enumerate(var_names)}

    scores = {}
    for pid in selected_programs:
        markers = program_markers[pid]
        col_idx = [var_idx[m] for m in markers if m in var_idx]
        if len(col_idx) == 0:
            scores[f"GEP_{pid}"] = np.zeros(adata.n_obs)
            continue
        vals = np.asarray(expr[:, col_idx])
        mean_expr = vals.mean(axis=1)
        mu, sigma = mean_expr.mean(), mean_expr.std()
        if sigma > 0:
            mean_expr = (mean_expr - mu) / sigma
        scores[f"GEP_{pid}"] = mean_expr

    return pd.DataFrame(scores, index=adata.obs_names)


# ---------------------------------------------------------------------------
# Metadata Associations (wide format, one row per factor)
# ---------------------------------------------------------------------------

def compute_metadata_associations(Z_arr, adata, z_var):
    """Compute factor x metadata associations for PixelGen data.

    Tests: time (Spearman), pseudotime (Spearman), condition (rank-biserial
    or eta-squared), cell_system (rank-biserial or eta-squared).

    Returns assoc_df with one row per factor.
    """
    n_factors = Z_arr.shape[1]

    def _rb(U, n1, n2):
        return 2.0 * U / (n1 * n2) - 1.0

    def _categorical_effect(z, col_vals):
        """Compute effect size for a categorical variable."""
        groups = [g for g in pd.unique(col_vals) if pd.notna(g)]
        if len(groups) < 2:
            return 0.0, 1.0
        if len(groups) == 2:
            g0, g1 = groups[0], groups[1]
            a = z[col_vals == g0]
            b = z[col_vals == g1]
            if len(a) < 2 or len(b) < 2:
                return 0.0, 1.0
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            return _rb(u, len(a), len(b)), p
        else:
            group_vals = [z[col_vals == g] for g in groups]
            group_vals = [gv for gv in group_vals if len(gv) > 0]
            if len(group_vals) < 2:
                return 0.0, 1.0
            h_stat, p = kruskal(*group_vals)
            n_total = sum(len(gv) for gv in group_vals)
            k = len(group_vals)
            eta2 = (h_stat - k + 1) / (n_total - k) if n_total > k else 0.0
            return eta2, p

    rows = []
    for f in range(n_factors):
        z = Z_arr[:, f]
        row = {"factor": f, "z_var": z_var[f]}

        # Time (Spearman if present and numeric)
        if "time" in adata.obs.columns:
            try:
                time_vals = adata.obs["time"].values.astype(float)
                rho, p = spearmanr(z, time_vals, nan_policy="omit")
                row["spearman_time"] = rho
                row["time_p"] = p
            except (ValueError, TypeError):
                rb, p = _categorical_effect(z, adata.obs["time"].values)
                row["spearman_time"] = rb
                row["time_p"] = p

        # Pseudotime
        if "dpt_pseudotime" in adata.obs.columns:
            pt = adata.obs["dpt_pseudotime"].values
            rho, p = spearmanr(z, pt, nan_policy="omit")
            row["pseudotime_rho"] = rho
            row["pseudotime_p"] = p

        # Condition
        if "condition" in adata.obs.columns:
            rb, p = _categorical_effect(z, adata.obs["condition"].values)
            row["rb_condition"] = rb
            row["mw_condition_p"] = p

        # Cell system
        if "cell_system" in adata.obs.columns:
            rb, p = _categorical_effect(z, adata.obs["cell_system"].values)
            row["rb_cell_system"] = rb
            row["mw_cell_system_p"] = p

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Factor-Program Correlation (Spearman, BH corrected)
# ---------------------------------------------------------------------------

def factor_program_correlation(
    Z_arr: np.ndarray,
    prog_scores_df: pd.DataFrame,
    method: str = "spearman",
) -> tuple:
    """Correlate each factor with each program score.

    Returns (corr_df, pval_df, padj_df), all shaped (n_factors, n_programs).
    """
    n_factors = Z_arr.shape[1]
    programs = list(prog_scores_df.columns)
    factor_names = [f"Z[{f}]" for f in range(n_factors)]

    corr_fn = spearmanr if method == "spearman" else pearsonr
    corr_mat = np.zeros((n_factors, len(programs)))
    pval_mat = np.zeros((n_factors, len(programs)))

    for fi in range(n_factors):
        for pi, prog in enumerate(programs):
            rho, p = corr_fn(Z_arr[:, fi], prog_scores_df[prog].values)
            corr_mat[fi, pi] = rho
            pval_mat[fi, pi] = p

    flat_p = pval_mat.flatten()
    _, padj_flat, _, _ = multipletests(flat_p, method="fdr_bh")
    padj_mat = padj_flat.reshape(pval_mat.shape)

    corr_df = pd.DataFrame(corr_mat, index=factor_names, columns=programs)
    pval_df = pd.DataFrame(pval_mat, index=factor_names, columns=programs)
    padj_df = pd.DataFrame(padj_mat, index=factor_names, columns=programs)
    return corr_df, pval_df, padj_df


# ---------------------------------------------------------------------------
# Interpretation Table (Gaublomme-style, wide assoc_df input)
# ---------------------------------------------------------------------------

def build_interpretation_table(
    assoc_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    padj_df: pd.DataFrame,
    program_char_df: pd.DataFrame = None,
    alpha: float = 0.05,
    metadata_threshold: float = 0.15,
) -> pd.DataFrame:
    """Build Gaublomme-style interpretation table.

    Accepts either the new wide assoc_df format (one row per factor with named
    effect columns) or the legacy format from the old pipeline.
    """
    # Detect format: new format has "z_var" column
    if "z_var" not in assoc_df.columns:
        # Legacy format: delegate to the old implementation
        return _build_interpretation_table_legacy(
            assoc_df, corr_df, padj_df, alpha)

    effect_cols = [c for c in ["spearman_time", "pseudotime_rho",
                               "rb_condition", "rb_cell_system"]
                   if c in assoc_df.columns]
    effect_labels = {
        "spearman_time": "Time", "pseudotime_rho": "Pseudotime",
        "rb_condition": "Condition", "rb_cell_system": "Cell System",
    }

    rows = []
    for f in range(len(assoc_df)):
        effects = assoc_df[effect_cols].iloc[f].values.astype(float)
        abs_effects = np.abs(effects)
        meta_sig = np.any(abs_effects > metadata_threshold)
        best_meta_idx = np.argmax(abs_effects)
        best_meta = effect_labels.get(effect_cols[best_meta_idx],
                                      effect_cols[best_meta_idx])
        best_meta_val = effects[best_meta_idx]

        factor_label = f"Z[{f}]"
        if factor_label in padj_df.index:
            padj_row = padj_df.loc[factor_label]
            corr_row = corr_df.loc[factor_label]
            bio_sig = np.any(padj_row.values < alpha)
            best_prog_idx = np.argmax(np.abs(corr_row.values))
            best_prog = corr_row.index[best_prog_idx]
            best_prog_rho = corr_row.iloc[best_prog_idx]
        else:
            bio_sig = False
            best_prog = "N/A"
            best_prog_rho = 0.0

        if meta_sig and bio_sig:
            interp = "signal"
        elif meta_sig and not bio_sig:
            interp = "possible batch effect"
        elif not meta_sig and bio_sig:
            interp = "novel axis"
        else:
            interp = "uninterpretable"

        rows.append({
            "factor": f,
            "z_var": assoc_df["z_var"].iloc[f],
            "interpretation": interp,
            "top_metadata": f"{best_meta} ({best_meta_val:+.3f})",
            "top_program": f"{best_prog} (rho={best_prog_rho:.3f})",
            "metadata_significant": meta_sig,
            "biology_significant": bio_sig,
        })

    return pd.DataFrame(rows)


def _build_interpretation_table_legacy(r_program, p_program, meta_results,
                                       bio_threshold=0.05):
    """Legacy interpretation table (old format with separate r_program, meta_results)."""
    rows = []
    for factor in r_program.index:
        abs_r = r_program.loc[factor].abs()
        best_prog = abs_r.idxmax()
        best_r = r_program.loc[factor, best_prog]
        best_p_prog = p_program.loc[factor, best_prog]

        fmeta = meta_results[meta_results["factor"] == factor]
        if len(fmeta) > 0:
            best_meta_row = fmeta.loc[fmeta["effect_size"].abs().idxmax()]
            best_meta = best_meta_row["metadata"]
            best_meta_eff = best_meta_row["effect_size"]
            best_meta_padj = best_meta_row["padj"]
        else:
            best_meta = "N/A"; best_meta_eff = 0.0; best_meta_padj = 1.0

        sig_prog = best_p_prog < bio_threshold
        sig_meta = best_meta_padj < 0.05

        if sig_meta and sig_prog:
            cls = "biology_aligned"
        elif sig_meta:
            cls = "batch_effect"
        elif sig_prog:
            cls = "novel_axis"
        else:
            cls = "noise"

        rows.append(dict(factor=factor, top_program=best_prog,
                         program_r=round(best_r, 3), program_p=best_p_prog,
                         top_metadata=best_meta,
                         metadata_effect=round(best_meta_eff, 3),
                         metadata_padj=best_meta_padj, classification=cls))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Combined Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    assoc_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    padj_df: pd.DataFrame,
    canon_corr_df: pd.DataFrame = None,
    canon_padj_df: pd.DataFrame = None,
    results_dir=None,
    row_labels=None,
    title="Factor × Feature Correlation",
):
    """Combined heatmap: rows × [metadata + data GEPs + canonical GEPs].

    Rows default to ``Z[0]..Z[n-1]``. Pass ``row_labels`` to override
    (e.g. synapse PC names).
    """
    all_effect_cols = ["spearman_time", "pseudotime_rho", "rb_condition", "rb_cell_system"]
    all_meta_labels = ["Time", "Pseudotime", "Condition", "Cell System"]
    effect_cols = [c for c in all_effect_cols if c in assoc_df.columns]
    meta_labels = [all_meta_labels[all_effect_cols.index(c)] for c in effect_cols]
    n_factors = len(assoc_df)

    meta_vals = assoc_df[effect_cols].values.astype(float)
    prog_vals = corr_df.values

    blocks = [meta_vals, prog_vals]
    col_labels = meta_labels + list(corr_df.columns)
    padj_blocks = [np.full_like(meta_vals, np.nan), padj_df.values]

    if canon_corr_df is not None:
        blocks.append(canon_corr_df.values)
        col_labels += [f"c:{c}" for c in canon_corr_df.columns]
        padj_blocks.append(canon_padj_df.values if canon_padj_df is not None
                           else np.full_like(canon_corr_df.values, np.nan))

    combined = np.hstack(blocks)
    padj_combined = np.hstack(padj_blocks)
    if row_labels is None:
        row_labels = [f"Z[{f}]" for f in range(n_factors)]

    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 0.7),
                                    max(4, n_factors * 0.5)))
    vmax = max(np.abs(combined).max(), 0.01)
    im = ax.imshow(combined, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(n_factors))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title)

    for i in range(n_factors):
        for j in range(len(col_labels)):
            val = combined[i, j]
            padj_val = padj_combined[i, j]
            star = ""
            if not np.isnan(padj_val):
                star = ("***" if padj_val < 0.001 else
                        ("**" if padj_val < 0.01 else
                         ("*" if padj_val < 0.05 else "")))
            txt_color = "white" if abs(val) > 0.6 * vmax else "black"
            ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center",
                    fontsize=5.5, color=txt_color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Effect size / Spearman rho")
    plt.tight_layout()
    if results_dir:
        plt.savefig(Path(results_dir) / "correlation_heatmap.png",
                    dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Unified Factor Scatter Grid
# ---------------------------------------------------------------------------

def _transform_color_values(vals, method="winsorize", percentile_lo=2,
                            percentile_hi=98):
    """Transform continuous color values for better visual contrast."""
    if method == "none":
        return vals
    vals = np.asarray(vals, dtype=float)
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return vals
    if method == "winsorize":
        lo = np.percentile(finite, percentile_lo)
        hi = np.percentile(finite, percentile_hi)
        out = np.clip(vals, lo, hi)
        if hi > lo:
            out = (out - lo) / (hi - lo)
        return out
    if method == "quantile":
        from scipy.stats import rankdata
        out = np.full_like(vals, np.nan)
        mask = np.isfinite(vals)
        ranks = rankdata(vals[mask], method="average")
        out[mask] = (ranks - 1) / max(ranks.max() - 1, 1)
        return out
    return vals


def plot_factor_scatter_grid(
    Z_arr, adata,
    color_items, prog_scores_df=None, canonical_scores_df=None,
    f1=0, f2=1, max_pts=2000, seed=42,
    color_transform="winsorize",
    results_dir=None, save_name="factor_scatter_grid.png",
):
    """Unified multi-panel factor scatter plot.

    color_items: list of "meta:<col>", "gep:<name>", "canonical:<name>",
        or "marker:<name>" strings.
    """
    rng = np.random.default_rng(seed)
    n = Z_arr.shape[0]
    sub_idx = (rng.choice(n, min(max_pts, n), replace=False)
               if n > max_pts else np.arange(n))
    x = Z_arr[sub_idx, f1]
    y = Z_arr[sub_idx, f2]

    n_panels = len(color_items)
    ncols = min(n_panels, 4)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4 * nrows), squeeze=False)

    for idx, item in enumerate(color_items):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        prefix, name = item.split(":", 1) if ":" in item else ("meta", item)

        is_cat = False
        vals_raw = None
        if prefix == "meta":
            vals_raw = adata.obs[name].values[sub_idx]
            try:
                vals = vals_raw.astype(float)
            except (ValueError, TypeError):
                is_cat = True
        elif prefix == "gep" and prog_scores_df is not None:
            vals = prog_scores_df[name].values[sub_idx]
        elif prefix == "canonical" and canonical_scores_df is not None:
            vals = canonical_scores_df[name].values[sub_idx]
        elif prefix == "marker":
            gi = list(adata.var_names).index(name)
            layer_key = "arcsinh" if "arcsinh" in adata.layers else "denoised_semantic"
            vals = np.asarray(adata.layers[layer_key][sub_idx, gi]).flatten()
        else:
            continue

        title = name.replace("_", " ")
        if is_cat:
            cats = [c for c in pd.unique(vals_raw) if pd.notna(c)]
            cmap_cat = plt.cm.get_cmap("tab10", len(cats))
            for ci, cat in enumerate(sorted(cats)):
                mask_c = vals_raw == cat
                ax.scatter(x[mask_c], y[mask_c], c=[cmap_cat(ci)], s=4,
                           alpha=0.5, label=str(cat), rasterized=True,
                           edgecolors="none")
            ax.legend(fontsize=6, markerscale=3, loc="best", framealpha=0.7)

            # Rank-biserial stats for 2-group categorical
            real_cats = sorted(cats)
            if len(real_cats) == 2:
                g1 = vals_raw == real_cats[0]
                g2 = vals_raw == real_cats[1]
                n1, n2 = g1.sum(), g2.sum()
                if n1 > 0 and n2 > 0:
                    ux, px = mannwhitneyu(x[g1], x[g2], alternative="two-sided")
                    uy, py = mannwhitneyu(y[g1], y[g2], alternative="two-sided")
                    rbx = 2.0 * ux / (n1 * n2) - 1.0
                    rby = 2.0 * uy / (n1 * n2) - 1.0
                    _star = lambda p: ("***" if p < 0.001 else
                                       ("**" if p < 0.01 else
                                        ("*" if p < 0.05 else "")))
                    ax.text(0.02, 0.02,
                            f"rb$_x$={rbx:+.2f}{_star(px)}  "
                            f"rb$_y$={rby:+.2f}{_star(py)}",
                            transform=ax.transAxes, fontsize=6,
                            va="bottom", ha="left",
                            bbox=dict(fc="white", alpha=0.7, ec="none", pad=1))
        else:
            finite_mask = np.isfinite(vals)
            rho_x = spearmanr(x[finite_mask], vals[finite_mask]).statistic
            rho_y = spearmanr(y[finite_mask], vals[finite_mask]).statistic
            p_x = spearmanr(x[finite_mask], vals[finite_mask]).pvalue
            p_y = spearmanr(y[finite_mask], vals[finite_mask]).pvalue

            vals_vis = _transform_color_values(vals, method=color_transform)
            sc_plot = ax.scatter(x, y, c=vals_vis, s=4, alpha=0.5,
                                cmap="RdYlBu_r", vmin=0, vmax=1,
                                rasterized=True, edgecolors="none")
            plt.colorbar(sc_plot, ax=ax, shrink=0.7)

            _star = lambda p: ("***" if p < 0.001 else
                               ("**" if p < 0.01 else
                                ("*" if p < 0.05 else "")))
            xf, vf = x[finite_mask], vals[finite_mask]
            yf = y[finite_mask]

            # Rescale predicted variable into the orthogonal display axis
            # so lines render inside the scatter frame. Slope labels are in
            # raw (variable per Z[·]) units.
            vmin, vmax = float(np.min(vf)), float(np.max(vf))
            v_span = (vmax - vmin) if vmax > vmin else 1.0

            if len(vf) > 10:
                # Red line: variable ~ Z[f1]  (displayed rescaled to Z[f2] range)
                sl_r, ic_r = np.polyfit(xf, vf, 1)
                xsp = np.linspace(xf.min(), xf.max(), 50)
                pred_r = sl_r * xsp + ic_r
                ymin, ymax = float(np.min(yf)), float(np.max(yf))
                pred_r_disp = ymin + (pred_r - vmin) / v_span * (ymax - ymin)
                ax.plot(xsp, pred_r_disp,
                        color="#E74C3C", lw=2.5, alpha=0.9, zorder=5)
                ax.annotate(f"$\\beta_x$={sl_r:+.2g} ($\\rho_x$={rho_x:+.2f}{_star(p_x)})",
                            xy=(xsp[-1], pred_r_disp[-1]), xytext=(-4, 6),
                            textcoords="offset points", fontsize=6,
                            color="#E74C3C", fontweight="bold",
                            ha="right", va="bottom",
                            bbox=dict(fc="white", alpha=0.85, ec="none",
                                      pad=0.5))

                # Blue line: variable ~ Z[f2]  (displayed rescaled to Z[f1] range)
                sl_b, ic_b = np.polyfit(yf, vf, 1)
                ysp = np.linspace(yf.min(), yf.max(), 50)
                pred_b = sl_b * ysp + ic_b
                xmin, xmax = float(np.min(xf)), float(np.max(xf))
                pred_b_disp = xmin + (pred_b - vmin) / v_span * (xmax - xmin)
                ax.plot(pred_b_disp, ysp,
                        color="#3498DB", lw=2.5, alpha=0.9, zorder=5)
                ax.annotate(f"$\\beta_y$={sl_b:+.2g} ($\\rho_y$={rho_y:+.2f}{_star(p_y)})",
                            xy=(pred_b_disp[-1], ysp[-1]), xytext=(6, -4),
                            textcoords="offset points", fontsize=6,
                            color="#3498DB", fontweight="bold",
                            ha="left", va="top",
                            bbox=dict(fc="white", alpha=0.85, ec="none",
                                      pad=0.5))

        ax.set_xlabel(f"Z[{f1}]", fontsize=8)
        ax.set_ylabel(f"Z[{f2}]", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=6)

    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"Z[{f1}] vs Z[{f2}]", fontsize=11, y=1.01)
    plt.tight_layout()
    if results_dir:
        fig.savefig(Path(results_dir) / save_name, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Quadrant Analysis Suite
# ---------------------------------------------------------------------------

_QUAD_COLORS = {"Q1": "#E74C3C", "Q2": "#3498DB", "Q3": "#2ECC71", "Q4": "#F39C12"}


def assign_quadrants(Z_arr, f1, f2, color_vals=None, method="regression",
                     percentile=50):
    """Assign cells to quadrants in 2-factor space.

    method='regression': regression lines from top-tercile color cells.
    method='median': split at median of each factor.
    """
    x = Z_arr[:, f1]
    y = Z_arr[:, f2]
    labels = np.empty(len(x), dtype=object)

    if method == "regression":
        if color_vals is None:
            raise ValueError("color_vals required for method='regression'")
        c = np.asarray(color_vals, dtype=float)
        finite = np.isfinite(c)
        hi_thresh = np.percentile(c[finite], 67)
        hi = finite & (c >= hi_thresh)

        if hi.sum() < 20:
            print("  Warning: <20 high-color cells, falling back to median")
            return assign_quadrants(Z_arr, f1, f2, method="median")

        sl_r, ic_r = np.polyfit(x[hi], y[hi], 1)
        sl_b, ic_b = np.polyfit(y[hi], x[hi], 1)

        above_red = y - (sl_r * x + ic_r) >= 0
        right_blue = x - (sl_b * y + ic_b) >= 0

        labels[above_red & right_blue] = "Q1"
        labels[above_red & ~right_blue] = "Q2"
        labels[~above_red & ~right_blue] = "Q3"
        labels[~above_red & right_blue] = "Q4"

        line_info = {
            "method": "regression",
            "red_slope": sl_r, "red_intercept": ic_r,
            "blue_slope": sl_b, "blue_intercept": ic_b,
        }
        rho_x = spearmanr(x[finite], c[finite]).statistic
        rho_y = spearmanr(y[finite], c[finite]).statistic
        print(f"  Regression lines from top-tercile cells (n={hi.sum()}):")
        print(f"    Red (y~x):  y = {sl_r:.3f}*x + {ic_r:.3f}  (rho_x={rho_x:+.2f})")
        print(f"    Blue (x~y): x = {sl_b:.3f}*y + {ic_b:.3f}  (rho_y={rho_y:+.2f})")
    else:
        if method == "median":
            thresh_x = np.median(x)
            thresh_y = np.median(y)
        else:
            thresh_x = np.percentile(x, percentile)
            thresh_y = np.percentile(y, percentile)

        labels[(x >= thresh_x) & (y >= thresh_y)] = "Q1"
        labels[(x < thresh_x) & (y >= thresh_y)] = "Q2"
        labels[(x < thresh_x) & (y < thresh_y)] = "Q3"
        labels[(x >= thresh_x) & (y < thresh_y)] = "Q4"

        line_info = {"method": method, "thresh_x": thresh_x, "thresh_y": thresh_y}

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        n_q = (labels == q).sum()
        print(f"  {q}: {n_q} cells ({100*n_q/len(x):.1f}%)")

    return labels, line_info


def plot_quadrant_scatter(Z_arr, labels, f1, f2, line_info, results_dir=None):
    """Scatter of Z[f1] vs Z[f2] colored by quadrant with boundary lines."""
    fig, ax = plt.subplots(figsize=(7, 6))
    x = Z_arr[:, f1]
    y = Z_arr[:, f2]

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        mask = labels == q
        n_q = mask.sum()
        ax.scatter(x[mask], y[mask], c=_QUAD_COLORS[q], s=3, alpha=0.4,
                   label=f"{q} (n={n_q})", rasterized=True, edgecolors="none")

    xlim = (x.min() - 0.1, x.max() + 0.1)
    ylim = (y.min() - 0.1, y.max() + 0.1)

    if line_info["method"] == "regression":
        xsp = np.linspace(xlim[0], xlim[1], 100)
        ysp = np.linspace(ylim[0], ylim[1], 100)
        sl_r, ic_r = line_info["red_slope"], line_info["red_intercept"]
        ax.plot(xsp, sl_r * xsp + ic_r, color="#E74C3C", ls="--", lw=1.5,
                alpha=0.8, label=f"y~x (slope={sl_r:.2f})")
        sl_b, ic_b = line_info["blue_slope"], line_info["blue_intercept"]
        ax.plot(sl_b * ysp + ic_b, ysp, color="#3498DB", ls="--", lw=1.5,
                alpha=0.8, label=f"x~y (slope={sl_b:.2f})")
    else:
        ax.axvline(line_info["thresh_x"], color="gray", ls="--", lw=1, alpha=0.7)
        ax.axhline(line_info["thresh_y"], color="gray", ls="--", lw=1, alpha=0.7)

    ax.set_xlim(xlim); ax.set_ylim(ylim)

    offset_x = (xlim[1] - xlim[0]) * 0.03
    offset_y = (ylim[1] - ylim[0]) * 0.03
    for q, ha, va, cx, cy in [
        ("Q1", "right", "top", xlim[1] - offset_x, ylim[1] - offset_y),
        ("Q2", "left", "top", xlim[0] + offset_x, ylim[1] - offset_y),
        ("Q3", "left", "bottom", xlim[0] + offset_x, ylim[0] + offset_y),
        ("Q4", "right", "bottom", xlim[1] - offset_x, ylim[0] + offset_y),
    ]:
        ax.text(cx, cy, q, fontsize=14, fontweight="bold", color=_QUAD_COLORS[q],
                ha=ha, va=va, alpha=0.7)

    ax.set_xlabel(f"Z[{f1}]", fontsize=10)
    ax.set_ylabel(f"Z[{f2}]", fontsize=10)
    ax.set_title(f"Quadrant Analysis: Z[{f1}] vs Z[{f2}]", fontsize=12)
    ax.legend(fontsize=7, markerscale=3, loc="best", framealpha=0.8)
    plt.tight_layout()
    if results_dir:
        plt.savefig(Path(results_dir) / "quadrant_scatter.png",
                    dpi=150, bbox_inches="tight")
    plt.show()


def plot_quadrant_composition(labels, adata,
                              meta_cols=("condition", "cell_system"),
                              results_dir=None):
    """Stacked bar charts of metadata composition per quadrant."""
    from scipy.stats import chi2_contingency

    quads = ["Q1", "Q2", "Q3", "Q4"]
    n_meta = len(meta_cols)
    fig, axes = plt.subplots(1, n_meta, figsize=(5 * n_meta, 5), squeeze=False)

    for mi, col_name in enumerate(meta_cols):
        ax = axes[0, mi]
        vals = adata.obs[col_name].values
        categories = sorted([c for c in pd.unique(vals) if pd.notna(c)])

        counts = np.zeros((len(quads), len(categories)), dtype=int)
        for qi, q in enumerate(quads):
            qmask = labels == q
            for ci, cat in enumerate(categories):
                counts[qi, ci] = (qmask & (vals == cat)).sum()

        if counts.sum() > 0 and counts.shape[0] > 1 and counts.shape[1] > 1:
            chi2, pval, _, _ = chi2_contingency(counts)
        else:
            chi2, pval = 0, 1.0

        totals = counts.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1
        props = counts / totals

        bottom = np.zeros(len(quads))
        cmap = plt.cm.get_cmap("Set2", len(categories))
        for ci, cat in enumerate(categories):
            ax.bar(quads, props[:, ci], bottom=bottom, label=str(cat),
                   color=cmap(ci), edgecolor="white", lw=0.5)
            bottom += props[:, ci]

        ax.set_title(f"{col_name}\n($\\chi^2$={chi2:.1f}, p={pval:.2e})", fontsize=9)
        ax.set_ylabel("Proportion")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
        ax.set_ylim(0, 1)

        for qi, q in enumerate(quads):
            ax.text(qi, -0.06, f"n={int(totals[qi, 0])}", ha="center", fontsize=7,
                    transform=ax.get_xaxis_transform())

    fig.suptitle("Metadata Composition per Quadrant", fontsize=12, y=1.02)
    plt.tight_layout()
    if results_dir:
        plt.savefig(Path(results_dir) / "quadrant_composition.png",
                    dpi=150, bbox_inches="tight")
    plt.show()


def plot_quadrant_gep_scores(labels, prog_scores_df, canonical_scores_df,
                             results_dir=None):
    """Violin plots of GEP scores per quadrant."""
    all_scores = pd.concat([prog_scores_df, canonical_scores_df], axis=1)
    programs = list(all_scores.columns)
    n_progs = len(programs)
    ncols = min(4, n_progs)
    nrows = int(np.ceil(n_progs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                             squeeze=False)
    quads = ["Q1", "Q2", "Q3", "Q4"]

    for pi, prog in enumerate(programs):
        row, col = divmod(pi, ncols)
        ax = axes[row, col]
        data = []
        for q in quads:
            qmask = labels == q
            data.append(all_scores[prog].values[qmask])

        parts = ax.violinplot(data, positions=range(len(quads)),
                              showmedians=True, widths=0.7)
        for qi, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(_QUAD_COLORS[quads[qi]])
            pc.set_alpha(0.6)
        for key in ("cmins", "cmaxes", "cmedians", "cbars"):
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(0.8)

        valid = [d for d in data if len(d) > 0]
        if len(valid) >= 2:
            _, kw_p = kruskal(*valid)
            star = ("***" if kw_p < 0.001 else ("**" if kw_p < 0.01 else
                    ("*" if kw_p < 0.05 else "ns")))
        else:
            star = "?"

        ax.set_xticks(range(len(quads)))
        ax.set_xticklabels(quads, fontsize=8)
        ax.set_title(f"{prog} ({star})", fontsize=8)
        ax.tick_params(labelsize=6)

    for pi in range(n_progs, nrows * ncols):
        row, col = divmod(pi, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("GEP Scores per Quadrant", fontsize=12, y=1.01)
    plt.tight_layout()
    if results_dir:
        plt.savefig(Path(results_dir) / "quadrant_gep_scores.png",
                    dpi=150, bbox_inches="tight")
    plt.show()


def run_quadrant_de(adata, labels, layer="arcsinh", n_top=20, alpha=0.05):
    """Mann-Whitney U DE: each quadrant vs all other cells.

    Returns dict[quadrant -> DataFrame] with columns:
        marker, log2FC, U_stat, pval, padj.
    """
    expr = np.asarray(adata.layers[layer])
    marker_names = list(adata.var_names)
    quads = ["Q1", "Q2", "Q3", "Q4"]
    de_results = {}

    for q in quads:
        qmask = labels == q
        rest = ~qmask
        if qmask.sum() < 5 or rest.sum() < 5:
            print(f"  {q}: too few cells ({qmask.sum()}), skipping")
            continue

        mean_q = expr[qmask].mean(axis=0)
        mean_r = expr[rest].mean(axis=0)
        log2fc = np.log2(mean_q + 1e-6) - np.log2(mean_r + 1e-6)

        pvals = np.ones(len(marker_names))
        ustats = np.zeros(len(marker_names))
        for gi in range(len(marker_names)):
            u, p = mannwhitneyu(expr[qmask, gi], expr[rest, gi],
                                alternative="two-sided")
            pvals[gi] = p
            ustats[gi] = u

        _, padj, _, _ = multipletests(pvals, method="fdr_bh")

        df = pd.DataFrame({
            "marker": marker_names,
            "log2FC": log2fc,
            "U_stat": ustats,
            "pval": pvals,
            "padj": padj,
        }).sort_values("padj")

        n_sig = (df["padj"] < alpha).sum()
        de_results[q] = df

        top = df.head(n_top)
        up = (top["log2FC"] > 0).sum()
        print(f"\n  {q}: {n_sig} DE markers (padj<{alpha}), showing top {n_top}:")
        print(f"    {up} up, {n_top - up} down")
        for _, r in top.iterrows():
            direction = "+" if r["log2FC"] > 0 else "-"
            sig = ("***" if r["padj"] < 0.001 else
                   ("**" if r["padj"] < 0.01 else "*"))
            print(f"    {direction} {r['marker']:>12s}  log2FC={r['log2FC']:+.2f}"
                  f"  padj={r['padj']:.1e} {sig}")

    return de_results


def plot_quadrant_de_summary(de_results, n_top=10, results_dir=None):
    """Dot plot of top DE markers per quadrant."""
    quads = [q for q in ["Q1", "Q2", "Q3", "Q4"] if q in de_results]
    if not quads:
        print("No DE results to plot.")
        return

    rows = []
    for q in quads:
        df = de_results[q]
        for _, r in df.head(n_top).iterrows():
            rows.append({
                "quadrant": q,
                "marker": r["marker"],
                "log2FC": r["log2FC"],
                "neg_log10_padj": min(-np.log10(r["padj"] + 1e-300), 50),
            })
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        print("No DE markers to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.18)))

    y_labels = []
    y_pos = []
    y_colors = []
    pos = 0
    for q in quads:
        sub = plot_df[plot_df["quadrant"] == q]
        for _, r in sub.iterrows():
            y_labels.append(r["marker"])
            y_pos.append(pos)
            y_colors.append(_QUAD_COLORS[q])
            pos += 1
        pos += 0.5

    x_vals = plot_df["log2FC"].values
    sizes = plot_df["neg_log10_padj"].values
    size_scale = 200 / max(sizes.max(), 1)

    ax.scatter(x_vals, y_pos, s=sizes * size_scale, c=y_colors, alpha=0.7,
               edgecolors="black", lw=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("log2 Fold Change (quadrant vs rest)", fontsize=9)
    ax.set_title(f"Top {n_top} DE Markers per Quadrant", fontsize=11)
    ax.invert_yaxis()

    for q in quads:
        ax.scatter([], [], c=_QUAD_COLORS[q], s=40, label=q)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.8)

    plt.tight_layout()
    if results_dir:
        plt.savefig(Path(results_dir) / "quadrant_de_dotplot.png",
                    dpi=150, bbox_inches="tight")
    plt.show()


def run_quadrant_analysis(
    Z_arr, adata, f1, f2,
    prog_scores_df, canonical_scores_df,
    color_vals=None, color_name=None,
    method="regression", n_top_de=20, results_dir=None,
):
    """Full quadrant pipeline: assign -> scatter -> composition -> GEPs -> DE."""
    title = color_name or "variable"
    print(f"=== Quadrant Analysis: Z[{f1}] vs Z[{f2}] ({title}) ===\n")

    print("Assigning quadrants...")
    labels, line_info = assign_quadrants(
        Z_arr, f1, f2, color_vals=color_vals, method=method,
    )
    adata.obs["quadrant"] = pd.Categorical(
        labels, categories=["Q1", "Q2", "Q3", "Q4"])

    plot_quadrant_scatter(Z_arr, labels, f1, f2, line_info, results_dir=results_dir)

    print("\nMetadata composition per quadrant:")
    meta_cols = [c for c in ["condition", "cell_system"]
                 if c in adata.obs.columns]
    if meta_cols:
        plot_quadrant_composition(labels, adata, meta_cols=meta_cols,
                                 results_dir=results_dir)

    print("\nGEP scores per quadrant:")
    plot_quadrant_gep_scores(labels, prog_scores_df, canonical_scores_df,
                            results_dir=results_dir)

    print("\nDifferential expression (each quadrant vs rest):")
    de_results = run_quadrant_de(adata, labels, n_top=n_top_de)

    plot_quadrant_de_summary(de_results, n_top=min(n_top_de, 10),
                            results_dir=results_dir)

    return {"labels": labels, "line_info": line_info, "de_results": de_results}


# ---------------------------------------------------------------------------
# Synapse-axis feature builder (for 03_synapse_axis_analysis.ipynb)
# ---------------------------------------------------------------------------

def build_synapse_features(
    adata,
    t_markers: list,
    b_markers: list,
    abundance_obsm: str = "arcsinh",
    coloc_obsm: str = "spatial_asinh5",
    blocks: tuple = ("abundance", "within_coloc", "cross_coloc"),
) -> tuple:
    """Per-cell synapse feature matrix for T- and B-side markers.

    Block structure (columns prefixed for downstream slicing):
      - ``abund:<marker>``              abundance of each marker (T ∪ B)
      - ``coloc:<a>/<b>``               within-set pairwise coloc (all C(n,2) pairs)
      - ``cross:<t>/<b>``               T×B cross-coloc pairs only
    Lookup tries ``a/b`` then ``b/a`` in ``obsm[coloc_obsm]``; pairs absent in
    both orders are dropped and reported in ``missing``.

    Returns
    -------
    features : pd.DataFrame  shape (n_cells, n_features), index = adata.obs_names
    missing  : dict          {"abundance": [...], "coloc": [...], "cross": [...]}
    """
    abund_df = adata.obsm[abundance_obsm]
    coloc_df = adata.obsm[coloc_obsm]
    coloc_cols = set(coloc_df.columns)
    all_markers = list(t_markers) + list(b_markers)

    missing = {"abundance": [], "coloc": [], "cross": []}
    parts = []

    def _lookup_pair(a, b):
        if f"{a}/{b}" in coloc_cols:
            return f"{a}/{b}"
        if f"{b}/{a}" in coloc_cols:
            return f"{b}/{a}"
        return None

    if "abundance" in blocks:
        present = [m for m in all_markers if m in abund_df.columns]
        missing["abundance"] = [m for m in all_markers if m not in abund_df.columns]
        block = abund_df[present].copy()
        block.columns = [f"abund:{m}" for m in present]
        parts.append(block)

    if "within_coloc" in blocks:
        keep_cols, out_names = [], []
        for i, a in enumerate(all_markers):
            for b in all_markers[i + 1:]:
                col = _lookup_pair(a, b)
                if col is None:
                    missing["coloc"].append(f"{a}/{b}")
                else:
                    keep_cols.append(col)
                    out_names.append(f"coloc:{a}/{b}")
        block = coloc_df[keep_cols].copy()
        block.columns = out_names
        parts.append(block)

    if "cross_coloc" in blocks:
        keep_cols, out_names = [], []
        for a in t_markers:
            for b in b_markers:
                col = _lookup_pair(a, b)
                if col is None:
                    missing["cross"].append(f"{a}/{b}")
                else:
                    keep_cols.append(col)
                    out_names.append(f"cross:{a}/{b}")
        block = coloc_df[keep_cols].copy()
        block.columns = out_names
        parts.append(block)

    features = pd.concat(parts, axis=1)
    features.index = adata.obs_names
    return features, missing


# =============================================================================
# Synapse-axis helpers (notebook 03)
# =============================================================================

def select_pca_columns(features, feature_blocks):
    """Return columns from ``features`` whose prefix matches the given blocks."""
    prefixes = {'abundance': 'abund:', 'within_coloc': 'coloc:', 'cross_coloc': 'cross:'}
    allow = {prefixes[b] for b in feature_blocks if b in prefixes}
    return [c for c in features.columns if any(c.startswith(p) for p in allow)]


def run_synapse_pca(features, pca_cols, pc_max=50, pc_variance_threshold=0.80, seed=42):
    """Standardize then fit PCA on ``features[pca_cols]``.

    Returns ``(pca, Z_pcs, cumvar, k_auto, n_fit)``. ``k_auto`` is the smallest
    k with cumulative explained variance >= threshold (or ``n_fit`` if never
    reached).
    """
    X_raw = features[pca_cols].values
    X_std = StandardScaler().fit_transform(X_raw)
    n_fit = int(min(pc_max, X_std.shape[0], X_std.shape[1]))
    pca = PCA(n_components=n_fit, random_state=seed).fit(X_std)
    Z_pcs = pca.transform(X_std)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    if (cumvar >= pc_variance_threshold).any():
        k_auto = int(np.argmax(cumvar >= pc_variance_threshold)) + 1
    else:
        k_auto = n_fit
    return pca, Z_pcs, cumvar, k_auto, n_fit


def build_synapse_score_df(features, Z_pcs, n_top=5):
    """Return a DataFrame of the top-``n_top`` PCs as per-cell synapse scores."""
    n_top = int(min(n_top, Z_pcs.shape[1]))
    return pd.DataFrame(
        {f'synapse_PC{i + 1}': Z_pcs[:, i] for i in range(n_top)},
        index=features.index,
    )


def orient_pc1_by_metadata(synapse_df, Z_pcs, pca, adata,
                           hi_cond='Blinatumomab', hi_time='48h',
                           lo_cond='Mock', lo_time='6h', verbose=True):
    """Flip PC1 sign in-place so the ``hi_*`` group mean exceeds the ``lo_*`` group mean."""
    if not {'condition', 'time'} <= set(adata.obs.columns):
        return False
    hi_mask = (adata.obs['condition'] == hi_cond) & (adata.obs['time'].astype(str) == hi_time)
    lo_mask = (adata.obs['condition'] == lo_cond) & (adata.obs['time'].astype(str) == lo_time)
    if hi_mask.sum() == 0 or lo_mask.sum() == 0:
        return False
    hi_mean = synapse_df.loc[hi_mask.values, 'synapse_PC1'].mean()
    lo_mean = synapse_df.loc[lo_mask.values, 'synapse_PC1'].mean()
    if hi_mean < lo_mean:
        synapse_df['synapse_PC1'] *= -1
        Z_pcs[:, 0]               *= -1
        pca.components_[0]        *= -1
        if verbose:
            print(f"Flipped PC1 sign so {hi_cond}/{hi_time} > {lo_cond}/{lo_time}")
        return True
    return False


def plot_synapse_scree(pca, cumvar, k_auto, pc_variance_threshold, n_features,
                       save_path=None, truncate=True):
    """Bar of per-PC variance + cumulative line.

    By default truncates to the selected ``k_auto`` PCs (set ``truncate=False``
    to plot all fitted PCs and see the full tail).
    """
    n_fit = len(cumvar)
    n_show = int(k_auto) if truncate else n_fit
    xs = np.arange(1, n_show + 1)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(xs, pca.explained_variance_ratio_[:n_show], color='steelblue',
           alpha=0.8, label='per-PC')
    ax.set_xlabel('PC'); ax.set_ylabel('Explained variance ratio')
    suffix = f' — showing PC1..PC{n_show} of {n_fit} fitted' if truncate else ''
    ax.set_title(f'Scree — synapse features ({n_features} cols){suffix}')
    ax2 = ax.twinx()
    ax2.plot(xs, cumvar[:n_show], color='darkorange', marker='o', lw=1.5,
             label='cumulative')
    ax2.axhline(pc_variance_threshold, color='grey', ls='--', lw=0.8)
    ax2.axvline(k_auto, color='red', ls='--', lw=0.8)
    ax2.set_ylabel('Cumulative variance', color='darkorange')
    ax2.text(k_auto + 0.1, pc_variance_threshold - 0.04,
             f'k_auto={k_auto}', color='red', fontsize=9)
    ax2.set_ylim(0, 1.02)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_synapse_pc_loadings(pca, pca_cols,
                             pcs=('PC1', 'PC2', 'PC3', 'PC4', 'PC5'),
                             n_top=20, save_path=None, ncols=3):
    """Horizontal bar charts of top-|loading| features per PC.

    Returns the full loadings DataFrame (one column per fitted PC) for downstream reuse.
    """
    all_cols = [f'PC{i + 1}' for i in range(pca.components_.shape[0])]
    loadings = pd.DataFrame(pca.components_.T, index=pca_cols, columns=all_cols)

    pcs_to_plot = [pc for pc in pcs if pc in loadings.columns]
    ncols = int(min(ncols, len(pcs_to_plot)))
    nrows = int(np.ceil(len(pcs_to_plot) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows), squeeze=False)
    for ax, pc in zip(axes.flat, pcs_to_plot):
        top = loadings[pc].abs().sort_values(ascending=False).head(n_top).index
        sub = loadings.loc[top, pc].sort_values()
        colors = ['#d62728' if v < 0 else '#1f77b4' for v in sub.values]
        ax.barh(range(len(sub)), sub.values, color=colors)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub.index, fontsize=8)
        ax.axvline(0, color='k', lw=0.5)
        ax.set_title(f'{pc} — top {n_top} |loadings|')
        ax.set_xlabel('loading')
    for k in range(len(pcs_to_plot), nrows * ncols):
        axes.flat[k].axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return loadings


def plot_synapse_metadata_heatmap(syn_assoc, save_path=None):
    """Effect-size heatmap with significance-star annotations (Cell 6)."""
    effect_cols = [('spearman_time',   'time (ρ)'),
                   ('pseudotime_rho',  'pseudotime (ρ)'),
                   ('rb_condition',    'condition (rb)'),
                   ('rb_cell_system',  'cell_system (η²)')]
    pval_cols   = {'spearman_time':  'time_p',
                   'pseudotime_rho': 'pseudotime_p',
                   'rb_condition':   'mw_condition_p',
                   'rb_cell_system': 'mw_cell_system_p'}
    eff_df = pd.DataFrame({lab: syn_assoc[c] for c, lab in effect_cols if c in syn_assoc.columns})
    p_df   = pd.DataFrame({lab: syn_assoc[pval_cols[c]]
                           for c, lab in effect_cols
                           if c in syn_assoc.columns and pval_cols[c] in syn_assoc.columns})

    def _stars(p):
        if pd.isna(p): return ''
        if p < 1e-3:   return '***'
        if p < 1e-2:   return '**'
        if p < 5e-2:   return '*'
        return ''

    annot = eff_df.copy().astype(object)
    for r in eff_df.index:
        for c in eff_df.columns:
            s = _stars(p_df.loc[r, c]) if c in p_df.columns else ''
            annot.loc[r, c] = f"{eff_df.loc[r, c]:.2f}{s}"

    fig, ax = plt.subplots(figsize=(1.4 * len(eff_df.columns) + 2,
                                    0.55 * len(eff_df.index) + 1.5))
    vmax = max(0.1, np.nanmax(np.abs(eff_df.values)))
    sns.heatmap(eff_df.astype(float), annot=annot.values, fmt='',
                cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                linewidths=0.4, linecolor='white',
                cbar_kws={'label': 'effect size'}, ax=ax)
    ax.set_title('Synapse scores × metadata  (*: p<.05, **: p<.01, ***: p<.001)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel(''); ax.set_ylabel('')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_synapse_pc_scatter_grid(synapse_df, adata, pc_x='synapse_PC1', pc_y='synapse_PC2',
                                 meta_cols=('condition', 'time', 'cell_system', 'dpt_pseudotime'),
                                 save_path=None):
    """2D scatter of (PC_x, PC_y) with one panel per metadata column."""
    meta_cols = [c for c in meta_cols if c in adata.obs.columns]
    n_panels = len(meta_cols)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows), squeeze=False)
    x_vals = synapse_df[pc_x].values
    y_vals = synapse_df[pc_y].values

    for i, col in enumerate(meta_cols):
        ax = axes[i // n_cols, i % n_cols]
        vals = adata.obs[col].values
        if pd.api.types.is_numeric_dtype(adata.obs[col]):
            sc_h = ax.scatter(x_vals, y_vals, c=vals, cmap='viridis', s=6, alpha=0.7)
            plt.colorbar(sc_h, ax=ax, shrink=0.75, label=col)
        else:
            cats = pd.Categorical(vals)
            palette = plt.get_cmap('tab10')(np.linspace(0, 1, len(cats.categories)))
            for j, cat in enumerate(cats.categories):
                mask = cats.codes == j
                ax.scatter(x_vals[mask], y_vals[mask], s=6, alpha=0.7,
                           color=palette[j], label=str(cat))
            ax.legend(fontsize=7, markerscale=1.5, loc='best')
        ax.set_xlabel(pc_x); ax.set_ylabel(pc_y)
        ax.set_title(col)
    for k in range(n_panels, n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_synapse_pc_violins_by_system(synapse_df, adata,
                                      pcs=('synapse_PC1', 'synapse_PC2', 'synapse_PC3',
                                           'synapse_PC4', 'synapse_PC5'),
                                      save_dir=None):
    """Per-PC violin: x = time×condition, hue = cell_system (cd8.ipynb style).

    Kruskal–Wallis across the 4 time×condition groups, per system, shown in the title.
    """
    SYS_HH = 'healthy B + healthy T'
    SYS_NH = 'NALM-6 + healthy T'
    SYS_PP = 'patient B + patient T'
    SYS_NP = 'NALM-6 + patient T'
    ALL_SYSTEMS    = [SYS_HH, SYS_NH, SYS_PP, SYS_NP]
    SYSTEM_PALETTE = {SYS_HH: '#4878d0', SYS_NH: '#ee854a',
                      SYS_PP: '#6acc65', SYS_NP: '#d65f5f'}
    SYS_SHORT      = {SYS_HH: 'HB+HT', SYS_NH: 'N6+HT',
                      SYS_PP: 'PB+PT', SYS_NP: 'N6+PT'}

    time_cond = (
        adata.obs['time'].astype(str) + ' ' + adata.obs['condition'].astype(str)
    )
    TC_LABELS = {'6h Mock': '6h\nMock', '6h Blinatumomab': '6h\nBlina',
                 '48h Mock': '48h\nMock', '48h Blinatumomab': '48h\nBlina'}
    TC_ORDER  = [t for t in ['6h Mock', '6h Blinatumomab', '48h Mock', '48h Blinatumomab']
                 if t in time_cond.unique()]
    tc_order_short  = [TC_LABELS[t] for t in TC_ORDER]
    sys_order_short = [SYS_SHORT[s] for s in ALL_SYSTEMS
                       if s in adata.obs['cell_system'].unique()]
    palette_short   = {SYS_SHORT[s]: c for s, c in SYSTEM_PALETTE.items()}

    figs = []
    for pc_name in pcs:
        y = synapse_df[pc_name].values
        df_v = pd.DataFrame({
            pc_name:     y,
            'Condition': time_cond.map(TC_LABELS).values,
            'System':    adata.obs['cell_system'].map(SYS_SHORT).values,
        }).dropna(subset=['Condition', 'System'])

        kw_parts = []
        for sys_full, sys_short in SYS_SHORT.items():
            if sys_full not in adata.obs['cell_system'].unique():
                continue
            sys_mask = (adata.obs['cell_system'] == sys_full).values
            arrays = [y[sys_mask & (time_cond == tc).values] for tc in TC_ORDER]
            arrays = [a for a in arrays if len(a) > 1]
            if len(arrays) > 1:
                _, pval = kruskal(*arrays)
                kw_parts.append(f'{sys_short}: {sig_label(pval)}')

        fig, ax = plt.subplots(figsize=(14, 5))
        sns.violinplot(
            data=df_v, x='Condition', y=pc_name, hue='System',
            order=tc_order_short, hue_order=sys_order_short,
            palette=palette_short, cut=0, inner='quartile',
            density_norm='width', ax=ax,
        )
        ax.set_title(
            f'{pc_name}    KW across time×condition — ' + '  |  '.join(kw_parts),
            fontsize=12, fontweight='bold',
        )
        ax.set_xlabel('')
        ax.axhline(0, color='grey', lw=0.5, ls='--')
        ax.legend(title='System', fontsize=9, loc='best', ncol=2)
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(Path(save_dir) / f'synapse_{pc_name}_by_system_condition.png',
                        dpi=150, bbox_inches='tight')
        plt.show()
        figs.append(fig)


def plot_factor_synapse_deepdive(best_f, Z_arr, synapse_df_aligned, W_df, pc_loadings,
                                 adata, pc_name='synapse_PC1', n_top=15, save_path=None):
    """3-panel deep-dive for the factor with strongest correlation to ``pc_name``:
    (a) scatter Z_f vs PC, colored by condition
    (b) top |W| loadings for that factor
    (c) top PC loadings for that PC (reference column)
    """
    z_vals = Z_arr[:, best_f]
    pc_vals = synapse_df_aligned[pc_name].values

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    if 'condition' in adata.obs.columns:
        cats = pd.Categorical(adata.obs['condition'].values)
        palette = plt.get_cmap('tab10')(np.linspace(0, 1, len(cats.categories)))
        for j, cat in enumerate(cats.categories):
            mask = cats.codes == j
            ax.scatter(z_vals[mask], pc_vals[mask], s=7, alpha=0.6,
                       color=palette[j], label=str(cat))
        ax.legend(fontsize=8, title='condition')
    else:
        ax.scatter(z_vals, pc_vals, s=7, alpha=0.6)
    ax.set_xlabel(f'Z[{best_f}]'); ax.set_ylabel(pc_name)
    ax.set_title(f'Z[{best_f}] vs {pc_name}')

    ax = axes[1]
    if hasattr(W_df, 'iloc') and best_f < W_df.shape[1]:
        w_vec = W_df.iloc[:, best_f]
    else:
        w_vec = pd.Series(W_df[:, best_f])
    top_w = w_vec.abs().sort_values(ascending=False).head(n_top).index
    sub = w_vec.loc[top_w].sort_values()
    colors = ['#d62728' if v < 0 else '#1f77b4' for v in sub.values]
    ax.barh(range(len(sub)), sub.values, color=colors)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub.index, fontsize=8)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_title(f'Top markers in W[:, {best_f}]')
    ax.set_xlabel('W loading')

    ax = axes[2]
    pc_col = pc_name.replace('synapse_', '')   # 'PC1' etc
    top_pc = pc_loadings[pc_col].abs().sort_values(ascending=False).head(n_top).index
    sub = pc_loadings.loc[top_pc, pc_col].sort_values()
    colors = ['#d62728' if v < 0 else '#1f77b4' for v in sub.values]
    ax.barh(range(len(sub)), sub.values, color=colors)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub.index, fontsize=7)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_title(f'Top features in {pc_name}')
    ax.set_xlabel(f'{pc_col} loading')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_synapse_umap(Z_pcs, adata, n_pcs=10,
                      colors=('condition', 'time', 'cell_system'),
                      n_pc_panels=5,
                      n_neighbors=15, min_dist=0.3, seed=42,
                      save_path=None):
    """2D UMAP of the first ``n_pcs`` synapse-feature PCs.

    Row 1: one panel per metadata column in ``colors``.
    Row 2: one panel per PC (``n_pc_panels`` of them), coloring the UMAP
    by each cell's PC score (continuous viridis).
    """
    import umap
    n_use = int(min(n_pcs, Z_pcs.shape[1]))
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=seed)
    coords = reducer.fit_transform(Z_pcs[:, :n_use])

    colors = [c for c in colors if c in adata.obs.columns]
    n_meta = len(colors)
    n_pc   = int(min(n_pc_panels, Z_pcs.shape[1]))
    ncols  = max(n_meta, n_pc)
    nrows  = (1 if n_meta else 0) + (1 if n_pc else 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows),
                             squeeze=False)

    # Row 1 — metadata panels
    if n_meta:
        for j, col in enumerate(colors):
            ax = axes[0, j]
            vals = adata.obs[col].values
            if pd.api.types.is_numeric_dtype(adata.obs[col]):
                sc_h = ax.scatter(coords[:, 0], coords[:, 1], c=vals,
                                  cmap='viridis', s=6, alpha=0.7)
                plt.colorbar(sc_h, ax=ax, shrink=0.8, label=col)
            else:
                cats = pd.Categorical(vals)
                palette = plt.get_cmap('tab10')(np.linspace(0, 1, len(cats.categories)))
                for k, cat in enumerate(cats.categories):
                    mask = cats.codes == k
                    ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.7,
                               color=palette[k], label=str(cat))
                ax.legend(fontsize=8, markerscale=1.5, loc='best')
            ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
            ax.set_title(f'UMAP (PC1..PC{n_use}) — {col}')
        for k in range(n_meta, ncols):
            axes[0, k].axis('off')

    # Row 2 — per-PC continuous score panels
    if n_pc:
        pc_row = 1 if n_meta else 0
        for j in range(n_pc):
            ax = axes[pc_row, j]
            pc_vals = Z_pcs[:, j]
            vmax = np.nanpercentile(np.abs(pc_vals), 99)
            sc_h = ax.scatter(coords[:, 0], coords[:, 1], c=pc_vals,
                              cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                              s=6, alpha=0.8)
            plt.colorbar(sc_h, ax=ax, shrink=0.8, label=f'PC{j + 1}')
            ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
            ax.set_title(f'UMAP — synapse PC{j + 1}')
        for k in range(n_pc, ncols):
            axes[pc_row, k].axis('off')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return coords

