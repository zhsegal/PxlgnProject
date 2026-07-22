from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import networkx as nx
import torch
from tqdm.auto import tqdm
import maxflow
import anndata

# -----------------------------------------------------------------------------
# 1. Per-cell T/B ratio on graph nodes
# -----------------------------------------------------------------------------

def compute_weighted_cell_ratio(
    layout_cell: str,
    *,
    pg_data,
    adata,
    logfc_cd8: Dict[str, float],
    logfc_cd4: Dict[str, float],
    k: int = 2,
    cell_type_column: str = 'doublets_status',
    epsilon: float = 1e-6,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, nx.Graph]:
    """
    For a single cell graph:
      - Build UMI graph
      - Compute T/B neighbor-weighted log2 ratio per UMI
      - Merge with layout coordinates

    Parameters
    ----------
    layout_cell : str
        Cell ID (component) to extract from `pg_data`.
    pg_data : PixelGen-like object
        Must support .filter(components=...).edgelist().to_df()
        and .precomputed_layouts().to_df().
    adata : AnnData
        Must have adata.obs['cell_group'] giving 'CD8' or 'CD4' per cell.
    logfc_cd8, logfc_cd4 : dict
        Mapping protein -> logFC weight, for CD8 and CD4 B/T signatures.
    k : int
        Neighborhood order: 1 = immediate neighbors, 2 = up to distance 2.
    epsilon : float
        Small constant to avoid division by zero in log2 ratio.

    Returns
    -------
    merged_layout : DataFrame
        Layout with 't_b_ratio' per UMI.
    node_table : DataFrame
        Per-UMI table with t/b counts and ratio.
    edge_list : DataFrame
        UMI-UMI edges with markers.
    layout : DataFrame
        Original layout table.
    G : networkx.Graph
        UMI graph with node attribute 'protein'.
    """
    if k not in (1, 2):
        raise ValueError("k must be 1 or 2")

    # Extract subgraph and layout for this cell
    cell_pg = pg_data.filter(components=layout_cell)
    edge_list = cell_pg.edgelist().to_df()
    layout = cell_pg.precomputed_layouts().to_df()

    # Cell type (CD8/CD4)
    cell_type = adata.obs.loc[layout_cell, cell_type_column]
    if cell_type == "CD8_doublets":
        weights = logfc_cd8
    elif cell_type == "CD4_doublets":
        weights = logfc_cd4
    else:
        raise ValueError(f"Unsupported cell_type '{cell_type}' for {layout_cell}")

    # Build UMI graph
    G = nx.Graph()
    for _, row in edge_list.iterrows():
        u, v = row["umi1"], row["umi2"]
        G.add_edge(u, v)
        G.nodes[u]["protein"] = row["marker_1"]
        G.nodes[v]["protein"] = row["marker_2"]

    # Compute neighbor-weighted T/B counts and ratio
    node_records = []

    for node in G.nodes():
        protein = G.nodes[node].get("protein")

        if k == 1:
            neighbors = list(G.neighbors(node))
        else:  # k == 2
            neighbors = set(
                nx.single_source_shortest_path_length(G, node, cutoff=2)
            )
            neighbors.discard(node)

        t_count = 0.0
        b_count = 0.0

        for nbr in neighbors:
            n_protein = G.nodes[nbr].get("protein")
            w = float(weights.get(n_protein, 0.0))

            if w > 0:
                # interpret positive weights as B-like
                b_count += w
            elif w < 0:
                # negative weights as T-like (sign flipped)
                t_count += abs(w)

        ratio = np.log2((t_count + epsilon) / (b_count + epsilon))

        node_records.append(
            {
                "umi": node,
                "protein": protein,
                "t_neighbors": t_count,
                "b_neighbors": b_count,
                "t_b_ratio": ratio,
            }
        )

    node_table = pd.DataFrame(node_records)

    merged_layout = layout.merge(
        node_table[["umi", "t_b_ratio"]],
        left_on="index",
        right_on="umi",
        how="left",
    )

    return merged_layout, node_table, edge_list, layout, G


# -----------------------------------------------------------------------------
# 2. Exact T/B partition via Boykov–Kolmogorov min-cut
# -----------------------------------------------------------------------------

def partition_graph_bk(
    G: nx.Graph,
    df: pd.DataFrame,
    *,
    umi_col: str = "umi",
    ratio_col: str = "t_b_ratio",
    gamma: float | str = "auto",   # unary contrast
    delta: float = 1e-9,           # numerical floor for logs
    lambda_pair: float = 0.5,      # pairwise strength
    fill_value: float = 0.0,       # default ratio if missing
    use_edge_weight: bool = True,  # use G edge 'weight' if present
    contrast_sigma: Optional[float] = None,
    seed_abs: Optional[float] = None,
    prior_shift: float = 0.0,
) -> Tuple[Dict[str, str], List[Tuple[str, str]], List[Tuple[str, str]], int, pd.DataFrame]:
    """
    Partition a UMI graph into T/B regions via BK min-cut on logisticized t_b_ratio.

    Parameters
    ----------
    G : networkx.Graph
        Nodes are UMIs; edges define adjacency.
    df : DataFrame
        Must contain columns [umi_col, ratio_col] with one row per UMI.
    umi_col : str
        Column in df with UMI IDs matching G nodes.
    ratio_col : str
        Column in df with t_b_ratio values.
    gamma : float or 'auto'
        Unary contrast strength: larger => steeper logistic. 'auto' scales to your data.
    delta : float
        Numerical floor for log costs.
    lambda_pair : float
        Pairwise strength; smaller => sharper boundaries.
    fill_value : float
        Used for missing ratios.
    use_edge_weight : bool
        If True, use G[u][v]['weight'] (default 1.0) in pairwise term.
    contrast_sigma : float or None
        If set, contrast-sensitive Potts: w_ij *= exp(- (diff^2) / (2*sigma^2)).
    seed_abs : float or None
        If set, nodes with |ratio| >= seed_abs are hard-clamped to their sign.
    prior_shift : float
        Additive shift to ratios before logistic (nudges global T/B bias).

    Returns
    -------
    labels_dict : dict
        {umi: 'T' or 'B'}.
    edges_T, edges_B : list of (umi1, umi2)
        Edges fully inside T or B region.
    cut_size : int
        Number of original edges crossing the T/B cut.
    node_table : DataFrame
        [umi_col, ratio_col, 'label'].
    """
      # --- node indexing
    node_ids = list(G.nodes())
    idx_of = {u: i for i, u in enumerate(node_ids)}
    n = len(node_ids)

    # --- align ratios
    s = (
        df[[umi_col, ratio_col]]
        .groupby(umi_col, as_index=True)[ratio_col]
        .mean()
    )
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    t_b_ratio = s.reindex(node_ids).fillna(fill_value).to_numpy(float)

    # optional bias
    t_b_ratio_biased = t_b_ratio + float(prior_shift)

    # auto-gamma: 75th percentile -> ~0.9 prob
    if gamma == "auto":
        q = np.percentile(np.abs(t_b_ratio_biased), 75)
        q = float(q) if np.isfinite(q) and q > 0 else 1.0
        gamma_val = np.log(9.0) / q
    else:
        gamma_val = float(gamma)

    # logistic unary
    x = np.clip(gamma_val * t_b_ratio_biased, -50.0, 50.0)
    p_T = 1.0 / (1.0 + np.exp(-x))
    D_T = -np.log(p_T + delta)
    D_B = -np.log(1.0 - p_T + delta)

    # hard seeds
    if seed_abs is not None and seed_abs > 0:
        BIG = 1e12
        mask_pos = t_b_ratio_biased >= seed_abs
        mask_neg = t_b_ratio_biased <= -seed_abs

        D_T[mask_pos] = 0.0
        D_B[mask_pos] = BIG
        D_T[mask_neg] = BIG
        D_B[mask_neg] = 0.0

    # --- deduped undirected edge list
    edge_weight_accum: Dict[Tuple[int, int], float] = {}
    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        iu, iv = idx_of[u], idx_of[v]
        if iu == iv:
            continue
        if iu > iv:
            iu, iv = iv, iu

        w = float(data.get("weight", 1.0)) if use_edge_weight else 1.0
        edge_weight_accum[(iu, iv)] = edge_weight_accum.get((iu, iv), 0.0) + w

    if edge_weight_accum:
        edges_idx = np.array(list(edge_weight_accum.keys()), dtype=int)
        base_w = np.array(list(edge_weight_accum.values()), dtype=float)
    else:
        edges_idx = np.empty((0, 2), dtype=int)
        base_w = np.empty((0,), dtype=float)

    # contrast-sensitive Potts
    if edges_idx.size and contrast_sigma is not None and contrast_sigma > 0:
        diff = t_b_ratio_biased[edges_idx[:, 0]] - t_b_ratio_biased[edges_idx[:, 1]]
        cs = np.exp(-(diff * diff) / (2.0 * contrast_sigma**2))
        w_ij = (lambda_pair * base_w * cs).astype(float)
    else:
        w_ij = (lambda_pair * base_w).astype(float)

    # --- build BK graph
    g = maxflow.Graph[float]()
    nodes = g.add_nodes(n)

    # t-links
    for i in range(n):
        g.add_tedge(nodes[i], float(D_T[i]), float(D_B[i]))

    # n-links
    for (u_i, v_i), w in zip(edges_idx, w_ij):
        w = float(w)
        if w > 0:
            g.add_edge(nodes[int(u_i)], nodes[int(v_i)], w, w)

    # solve
    _ = g.maxflow()
    segments = np.fromiter(
        (g.get_segment(nodes[i]) for i in range(n)),
        dtype=int,
        count=n,
    )

    labels_dict = {
        node_ids[i]: ("T" if segments[i] == 1 else "B")
        for i in range(n)
    }

    # cut size in deduped graph
    cut_size = int(
        (segments[edges_idx[:, 0]] != segments[edges_idx[:, 1]]).sum()
        if edges_idx.size
        else 0
    )

    # split original edges (no dedup here)
    edges_T, edges_B = [], []
    for u, v in G.edges():
        lu, lv = labels_dict[u], labels_dict[v]
        if lu == lv == "T":
            edges_T.append((u, v))
        elif lu == lv == "B":
            edges_B.append((u, v))

    node_table = pd.DataFrame(
        {
            umi_col: node_ids,
            ratio_col: t_b_ratio,
            "label": np.where(segments == 1, "T", "B"),
        }
    )

    return labels_dict, edges_T, edges_B, cut_size, node_table


# -----------------------------------------------------------------------------
# 3. Colocalization Z-scores by label shuffling
# -----------------------------------------------------------------------------

def compute_coloc_zscores_label_shuffle_torch(
    edges_df: pd.DataFrame,
    n_sims: int = 100,
    seed: int = 42,
    device: str | torch.device | None = None,
):
    """
    Compute co-localization Z-scores by shuffling marker labels across UMIs.

    Keeps network structure (UMI-UMI edges) fixed.

    Parameters
    ----------
    edges_df : DataFrame
        Must contain ['umi1','umi2','marker_1','marker_2'].
    n_sims : int
        Number of Monte Carlo shuffles.
    seed : int
        Random seed.
    device : 'cuda', 'cpu', or None
        If None, auto-selects 'cuda' when available.

    Returns
    -------
    dict with:
        coloc_obs   : DataFrame (marker×marker)
        coloc_mean  : DataFrame
        coloc_std   : DataFrame
        coloc_z     : DataFrame
        abundance   : Series (unique-UMI count per marker)
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- encode markers
    markers = pd.Index(sorted(set(edges_df["marker_1"]) | set(edges_df["marker_2"])))
    m2i = {m: i for i, m in enumerate(markers)}
    nM = len(markers)

    # --- UMI -> marker mapping (one marker per UMI)
    umi_marker = pd.concat(
        [
            edges_df[["umi1", "marker_1"]].rename(
                columns={"umi1": "umi", "marker_1": "marker"}
            ),
            edges_df[["umi2", "marker_2"]].rename(
                columns={"umi2": "umi", "marker_2": "marker"}
            ),
        ]
    ).drop_duplicates("umi")

    umis = umi_marker["umi"].values
    markers_per_umi = umi_marker["marker"].map(m2i).values
    nU = len(umis)

    umi_to_idx = {u: i for i, u in enumerate(umis)}
    src_umi = torch.tensor(
        edges_df["umi1"].map(umi_to_idx).to_numpy(),
        dtype=torch.long,
        device=device,
    )
    tgt_umi = torch.tensor(
        edges_df["umi2"].map(umi_to_idx).to_numpy(),
        dtype=torch.long,
        device=device,
    )

    # --- observed counts
    src_marker = torch.tensor(
        edges_df["marker_1"].map(m2i).to_numpy(),
        dtype=torch.long,
        device=device,
    )
    tgt_marker = torch.tensor(
        edges_df["marker_2"].map(m2i).to_numpy(),
        dtype=torch.long,
        device=device,
    )

    obs = torch.zeros((nM, nM), dtype=torch.float32, device=device)
    obs.index_put_(
        (src_marker, tgt_marker),
        torch.ones_like(src_marker, dtype=torch.float32),
        accumulate=True,
    )

    # --- abundance (unique UMIs per marker)
    abundance = (
        umi_marker.groupby("marker")["umi"]
        .nunique()
        .reindex(markers, fill_value=0)
        .astype(float)
    )

    # --- Monte Carlo
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    base_labels = torch.tensor(markers_per_umi, dtype=torch.long, device=device)

    mean = torch.zeros((nM, nM), dtype=torch.float64, device=device)
    m2 = torch.zeros_like(mean)
    tmp = torch.zeros((nM, nM), dtype=torch.float32, device=device)

    for s in range(n_sims):
        # permutation of labels across UMIs
        shuffled_labels = base_labels[torch.randperm(nU, generator=rng, device=device)]
        src_labels = shuffled_labels[src_umi]
        tgt_labels = shuffled_labels[tgt_umi]

        tmp.zero_()
        tmp.index_put_(
            (src_labels, tgt_labels),
            torch.ones_like(src_labels, dtype=torch.float32),
            accumulate=True,
        )

        # Welford online mean/var
        delta = tmp - mean
        mean += delta / (s + 1)
        m2 += delta * (tmp - mean)

    std = torch.sqrt(m2 / max(1, n_sims - 1))
    z = (obs - mean) / (std + 1e-9)

    # --- to pandas
    coloc_obs = pd.DataFrame(obs.cpu().numpy(), index=markers, columns=markers)
    coloc_mean = pd.DataFrame(mean.cpu().numpy(), index=markers, columns=markers)
    coloc_std = pd.DataFrame(std.cpu().numpy(), index=markers, columns=markers)
    coloc_z = pd.DataFrame(z.cpu().numpy(), index=markers, columns=markers)
    abundance_s = pd.Series(abundance.values, index=markers, name="unique_umi_count")

    return dict(
        coloc_obs=coloc_obs,
        coloc_mean=coloc_mean,
        coloc_std=coloc_std,
        coloc_z=coloc_z,
        abundance=abundance_s,
    )


# -----------------------------------------------------------------------------
# 4. End-to-end: cell-wise colocalization to disk
# -----------------------------------------------------------------------------

def run_cellwise_coloc_analysis_to_disk(
    doublets,
    *,
    pg_data,
    adata,
    logfc_cd8: Dict[str, float],
    logfc_cd4: Dict[str, float],
    out_dir: str | Path = "coloc_results",
    k: int = 2,
    n_sims: int = 100,
    seed: int = 42,
    overwrite: bool = False,
    checkpoint_every: int = 50,
):
    """
    Cell-wise Monte Carlo colocalization pipeline:

      1) For each cell:
         - build UMI graph
         - compute T/B ratio per node
         - partition into T/B regions with BK min-cut
      2) For each region (T/B):
         - subset edges
         - run label-shuffle Monte Carlo colocalization
         - compute raw + scaled abundances
         - save per-cell, per-region results (parquet)
      3) Aggregate all results into big tables.

    Parameters
    ----------
    doublets : AnnData
        Cells to iterate over (doublets.obs.index used as cell IDs).
    pg_data : PixelGen-like object
        Same object passed to `compute_weighted_cell_ratio`.
    adata : AnnData
        Full cell metadata (used for cell_group).
    logfc_cd8, logfc_cd4 : dict
        Protein->logFC weights for CD8/CD4.
    out_dir : str or Path
        Output directory for parquet files.
    k : int
        Neighborhood order for ratios (1 or 2).
    n_sims : int
        Number of Monte Carlo shuffles.
    seed : int
        Base random seed for MC.
    overwrite : bool
        If False, will try to reload existing parquet files.
    checkpoint_every : int
        Save aggregated checkpoints every N cells.

    Returns
    -------
    abundance_all : DataFrame
        marker × sample_id, raw abundances.
    abundance_scaled_all : DataFrame
        marker × sample_id, abundances scaled by fraction of UMIs in region.
    zscores_all : DataFrame
        Long-form Z-scores: [marker_1, marker_2, sample_id, z_score].
    coloc_mean_all : DataFrame
        Long-form mean scores: [marker_1, marker_2, sample_id, mean_score].
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    abundance_results = []
    abundance_scaled_results = []
    zscore_results = []
    meanscore_results = []

    for i, cell in enumerate(
        tqdm(doublets.obs.index, desc="Processing cells"),
        start=1,
    ):
        for region_label in ("T", "B"):
            sample_id = f"{cell}_{region_label}"

            z_path = out_dir / f"zscore_{sample_id}.parquet"
            m_path = out_dir / f"mscore_{sample_id}.parquet"
            a_path = out_dir / f"abundance_{sample_id}.parquet"
            as_path = out_dir / f"abundance_scaled_{sample_id}.parquet"

            # --- reuse existing results if present ---
            if (
                not overwrite
                and z_path.exists()
                and a_path.exists()
            ):
                try:
                    z = pd.read_parquet(z_path)
                    a = pd.read_parquet(a_path)
                    if "marker" in a.columns:
                        a = a.set_index("marker")["abundance"]

                    if as_path.exists():
                        a_scaled = pd.read_parquet(as_path)
                        if "marker" in a_scaled.columns:
                            a_scaled = a_scaled.set_index("marker")["abundance_scaled"]
                    else:
                        a_scaled = a

                    abundance_results.append(a.rename(sample_id))
                    abundance_scaled_results.append(a_scaled.rename(sample_id))
                    zscore_results.append(z)
                    continue
                except Exception:
                    print(f"⚠️ Reload failed for {sample_id}, recomputing.")

            # --- full compute path ---
            try:
                # Build graph + node ratios
                (
                    _merged_layout,
                    node_table,
                    edge_list,
                    _layout,
                    G,
                ) = compute_weighted_cell_ratio(
                    layout_cell=cell,
                    pg_data=pg_data,
                    adata=adata,
                    logfc_cd8=logfc_cd8,
                    logfc_cd4=logfc_cd4,
                    k=k,
                )

                labels_dict, edges_T, edges_B, cut_size, _node_table = partition_graph_bk(
                    G,
                    node_table,
                    umi_col="umi",
                    ratio_col="t_b_ratio",
                    gamma="auto",
                    delta=1e-6,
                    lambda_pair=0.3,
                    contrast_sigma=0.8,
                    seed_abs=2.0,
                )

                # Choose region
                keep_edges = (
                    set(map(tuple, edges_T))
                    if region_label == "T"
                    else set(map(tuple, edges_B))
                )

                # Subset edges_df
                edges_df = edge_list.loc[
                    edge_list.apply(
                        lambda r: (r["umi1"], r["umi2"]) in keep_edges,
                        axis=1,
                    )
                ]

                if edges_df.empty:
                    continue

                # Monte Carlo colocalization
                results = compute_coloc_zscores_label_shuffle_torch(
                    edges_df=edges_df,
                    n_sims=n_sims,
                    seed=seed,
                )

                # scaling factor by fraction of UMIs in region
                n_cut = pd.unique(edges_df[["umi1", "umi2"]].values.ravel()).size
                n_total = pd.unique(edge_list[["umi1", "umi2"]].values.ravel()).size
                scale_factor = n_total / max(n_cut, 1)

                # raw + scaled abundance
                abundance_df = results["abundance"].rename(sample_id)
                abundance_scaled_df = abundance_df * scale_factor

                abundance_results.append(abundance_df)
                abundance_scaled_results.append(abundance_scaled_df)

                # save abundances
                (
                    abundance_df.to_frame(name="abundance")
                    .reset_index(names="marker")
                    .to_parquet(a_path, index=False)
                )
                (
                    abundance_scaled_df.to_frame(name="abundance_scaled")
                    .reset_index(names="marker")
                    .to_parquet(as_path, index=False)
                )

                # Z-scores (long)
                z = results["coloc_z"].copy()
                z.index.name = "marker_1"
                z.columns.name = "marker_2"
                z_long = (
                    z.stack()
                    .rename("z_score")
                    .reset_index()
                    .assign(sample_id=sample_id)
                )
                zscore_results.append(z_long)
                z_long.to_parquet(z_path, index=False)

                # mean-scores (long)
                m = results["coloc_mean"].copy()
                m.index.name = "marker_1"
                m.columns.name = "marker_2"
                m_long = (
                    m.stack()
                    .rename("mean_score")
                    .reset_index()
                    .assign(sample_id=sample_id)
                )
                meanscore_results.append(m_long)
                m_long.to_parquet(m_path, index=False)

            except Exception as e:
                print(f"⚠️ Skipped {sample_id}: {e}")
                continue

        # --- periodic checkpointing ---
        if checkpoint_every and (i % checkpoint_every == 0):
            try:
                ckpt_a = out_dir / f"abundance_checkpoint_{i}.parquet"
                ckpt_as = out_dir / f"abundance_scaled_checkpoint_{i}.parquet"
                ckpt_z = out_dir / f"zscores_checkpoint_{i}.parquet"

                if abundance_results:
                    pd.concat(abundance_results, axis=1).to_parquet(ckpt_a)
                    pd.concat(abundance_scaled_results, axis=1).to_parquet(ckpt_as)
                if zscore_results:
                    pd.concat(zscore_results, ignore_index=True).to_parquet(ckpt_z)

                print(f"💾 Checkpoint saved at iteration {i}")
            except Exception as e:
                print(f"⚠️ Checkpoint save failed at iteration {i}: {e}")

    # --- final combine ---
    abundance_all = (
        pd.concat(abundance_results, axis=1) if abundance_results else pd.DataFrame()
    )
    abundance_scaled_all = (
        pd.concat(abundance_scaled_results, axis=1)
        if abundance_scaled_results
        else pd.DataFrame()
    )
    zscores_all = (
        pd.concat(zscore_results, ignore_index=True)
        if zscore_results
        else pd.DataFrame()
    )
    coloc_mean_all = (
        pd.concat(meanscore_results, ignore_index=True)
        if meanscore_results
        else pd.DataFrame()
    )

    return abundance_all, abundance_scaled_all, zscores_all, coloc_mean_all

def concat_abundance_to_adata(adata, abundance_df, layer_name="doublets"):
    """
    Concatenate an abundance dataframe (marker × sample) as new cells to an existing AnnData.

    Parameters
    ----------
    adata : AnnData
        Original protein-level AnnData.
    abundance_df : pd.DataFrame
        marker × sample (e.g. abundance_all)
    layer_name : str
        Name for obs annotation for new abundance cells.

    Returns
    -------
    AnnData
        Combined AnnData with new abundance samples appended as observations.
    """

    # --- Align markers ---
    shared_markers = adata.var_names.intersection(abundance_df.index)
    if len(shared_markers) == 0:
        raise ValueError("No overlapping protein markers between adata.var_names and abundance_df.index")

    # Subset and align columns
    abundance_df = abundance_df.loc[shared_markers]

    # --- Transpose to samples × markers ---
    new_X = abundance_df.T
    new_obs = pd.DataFrame(index=new_X.index)
    new_obs["source"] = layer_name
    new_obs["region"] = new_obs.index.str.extract(r"_(T|B)$")[0].fillna("NA")

    # Reuse adata.var subset
    new_var = adata.var.loc[shared_markers].copy()

    # --- Create a new AnnData for the abundance samples ---
    new_adata = anndata.AnnData(
        X=new_X.to_numpy(dtype=np.float32),
        obs=new_obs,
        var=new_var,
    )

    # --- Concatenate with original ---
    combined = anndata.concat(
        [adata[:, shared_markers], new_adata],
        join="outer",
        axis=0,  # concatenate along cells
        merge="same",
        label="dataset",
        keys=["original", layer_name],
        index_unique=None,
    )

    return combined

def add_doublets_metadata(t_cells, adata_combined):
    t_map = (
        t_cells.obs
        .assign(prefix=lambda d: d.index.str.split("_").str[0])
        .set_index("prefix")[["condition", "sample_name", "cell_group"]]
    )

    # 2) Work on combined obs
    ac = adata_combined.obs.copy()
    mask = ac["dataset"] == "doublets"

    # 3) Extract prefix only for doublets
    prefix = ac.index[mask].str.split("_").str[0]

    # 4) Look up metadata for those prefixes
    lookup = t_map.reindex(prefix).reset_index(drop=True)

    # 5) Assign into existing columns ONLY for doublets
    ac.loc[mask, ["condition", "sample_name", "cell_group"]] = lookup.values

    # 6) Write back
    adata_combined.obs = ac
    return adata_combined

B_CD8_logfc_dict={'CD40': np.float32(6.2898097),
 'CD21': np.float32(5.5763106),
 'CD22': np.float32(5.27185),
 'CD19': np.float32(4.7072916),
 'CD32': np.float32(4.637353),
 'IgM': np.float32(4.217416),
 'CD54': np.float32(3.8962739),
 'CD72': np.float32(3.728628),
 'CD79a': np.float32(3.4088624),
 'CD20': np.float32(3.0901158),
 'CD80': np.float32(3.0267005),
 'HLA-DQ': np.float32(2.9593556),
 'CD180': np.float32(2.869054),
 'CD268': np.float32(2.687452),
 'CD10': np.float32(2.6281714),
 'CD69': np.float32(2.596234),
 'CD84': np.float32(2.4542124),
 'CD70': np.float32(2.3794875),
 'HLA-DR': np.float32(2.2383494),
 'HLA-DR-DP-DQ': np.float32(2.1274824),
 'CD31': np.float32(-2.711627),
 'CD52': np.float32(-2.7201114),
 'CD366': np.float32(-2.940809),
 'CD25': np.float32(-2.9976668),
 'FMC63': np.float32(-3.1916533),
 'CD43': np.float32(-3.3009002),
 'CD11a': np.float32(-3.3042922),
 'CD314': np.float32(-3.3219802),
 'CD18': np.float32(-3.5199463),
 'CD49e': np.float32(-3.5882366),
 'CD2': np.float32(-4.070872),
 'CD5': np.float32(-4.108194),
 'CD6': np.float32(-4.3686314),
 'CD3e': np.float32(-4.5021653),
 'CD162': np.float32(-4.5559945),
 'CD26': np.float32(-4.5650887),
 'CD226': np.float32(-4.5855117),
 'CD8': np.float32(-4.896089),
 'CD50': np.float32(-4.9749846),
 'CD44': np.float32(-5.7353725)}


B_CD4_logfc_dict={'CD40': np.float32(6.461159),
 'CD21': np.float32(5.685767),
 'CD22': np.float32(5.412799),
 'CD32': np.float32(4.833199),
 'CD19': np.float32(4.6325073),
 'CD54': np.float32(4.251058),
 'IgM': np.float32(4.2483077),
 'CD72': np.float32(3.9162552),
 'HLA-DQ': np.float32(3.6397052),
 'CD79a': np.float32(3.4097598),
 'CD20': np.float32(3.2463245),
 'HLA-DR': np.float32(3.1058195),
 'HLA-DR-DP-DQ': np.float32(3.0756428),
 'CD80': np.float32(2.9782872),
 'CD45RA': np.float32(2.9234211),
 'CD180': np.float32(2.8666377),
 'CD268': np.float32(2.702681),
 'CD84': np.float32(2.69288),
 'CD10': np.float32(2.6664326),
 'CD70': np.float32(2.645956),
 'CD28': np.float32(-2.9662206),
 'FMC63': np.float32(-3.0113204),
 'TCRab': np.float32(-3.022153),
 'CD25': np.float32(-3.1291914),
 'CD127': np.float32(-3.20463),
 'CD11a': np.float32(-3.2485867),
 'CD278': np.float32(-3.4784415),
 'CD18': np.float32(-3.4876003),
 'CD45RO': np.float32(-3.5076637),
 'CD2': np.float32(-3.7042341),
 'CD49e': np.float32(-3.8749592),
 'CD26': np.float32(-4.0317907),
 'CD226': np.float32(-4.2594004),
 'CD162': np.float32(-4.4962306),
 'CD3e': np.float32(-4.5454793),
 'CD50': np.float32(-4.879214),
 'CD6': np.float32(-5.0243044),
 'CD4': np.float32(-5.3830047),
 'CD5': np.float32(-5.512065),
 'CD44': np.float32(-6.244488)}