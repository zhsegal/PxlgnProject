"""Per-cell graph phenotype panel for PNA cells (topology + local proximity).

Companion to chunglu_triplets.py. Where that module scores *marker* motifs (doublets /
triplets) against a degree-corrected null, this one scores the *graph itself*: degree
structure, clustering, connectivity, and B-cell membrane patches.

Why a null is mandatory here
----------------------------
The PNA graph is bipartite (umi1 / umi2 sides), so its A-pixel one-mode projection is a
union of cliques by construction: every umi2 node of degree k contributes a k-clique among
its umi1 neighbours. Raw transitivity therefore measures the umi2 degree distribution, not
biology. Every non-degree feature is reported three ways:

    <feat>        observed value
    <feat>_ratio  obs / mean(null)      -- effect size vs degree-preserving wiring
    <feat>_z      (obs - mean) / sd     -- within-cell significance; INFLATES with graph size

The null is the bipartite degree-preserving edge swap (chunglu_triplets.bipartite_edgeswap_sample),
which fixes both sides' degree sequences exactly. Degree features are therefore invariant
under it and get no ratio/z. Calibration: on a synthetic Chung-Lu graph every ratio is 1.000
and |z| < 2; on a real CD8 cell transitivity_ratio is 1.79 and degree_assortativity_ratio 2.85.

Depth is the whole problem -- and DO NOT fix it by subsampling
-------------------------------------------------------------
Neither ratio nor z is depth-stable. Measured on one CD8 cell by molecule downsampling:

    budget                 20k     40k     60k    full(80k)
    transitivity_ratio     1.06    1.20    1.45    1.79
    degree_assortativity   1.10    1.36    1.85    2.77

Structure only becomes *detectable* as molecules accumulate, so a deeper cell scores as more
structured at identical biology. That matters here because depth tracks the contrasts:
Blinatumomab cells carry ~40-46% more edges than Mock (6h NALM 139,652 vs 100,072, p=4e-23;
6h HB 119,577 vs 81,950, p=6e-41) and the two systems differ too (p=3e-07).

The obvious fix -- downsample every cell to a common edge count -- MAKES IT WORSE, and this
was measured, not assumed. On 48 depth-stratified S001 CD8 cells (n_edges 31.9k-287.8k),
mean |Spearman rho| to native depth:

    unmatched 0.403  ->  matched@45k 0.881  ->  matched@64k 0.865
    features passing |rho|<0.3:   2/10  ->  0/10  ->  0/10

Because depth inflates edges AND pixels together. Fixing edges at 45k while leaving the pixel
count free makes deep cells artificially sparse and shallow cells dense -- an inverted, larger
confound (rho ~ -0.9 for transitivity/clustering/degree). Fixing n1, n2 and E simultaneously
is not available either: the ~9x depth spread collapses induced mean degree below 1, which
shatters the projection (at a 30k budget a real cell goes from 1 connected component to 4,324,
and average_k_core drops to 1.73 -- pixelator's own "this is debris" threshold is 1.5).

So depth is handled by COHORT SELECTION in the notebook, not by subsampling here: compare
cells inside a native-depth band (a 90-150k band leaves 726 CD8 cells with system depth
MWU p=0.76), and nearest-neighbour match on n_edges for Blina vs Mock. Graphs stay at native
depth, where structure is actually measurable.

`subsample_edges` and --edge-budgets are retained only to REPRODUCE the depth curve above as
a diagnostic. Default is no matching. Do not turn them on for a result.

Literature note: no MPX/PNA paper uses pure topology as a phenotype (it is QC / cell-calling
only); every published discriminative metric is node-attributed. This panel is exploratory,
which is exactly why the size-confound controls are not optional. Local proximity / LNE
(local_proximity) is the one block with published (vendor) precedent.

CLI (one sample per process; see run_graph_phenotype.lsf):
    python graph_phenotype.py --sample S001 \
        --adata cache/adata_cytovi_annotated_compat.h5ad \
        --pxl results/S001/layout/layout/S001.layout.pxl \
        --output-dir results/graph_phenotype --n-workers 16
"""

import argparse
import json
import time
import zlib
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from chunglu_triplets import _batched, bipartite_edgeswap_sample, build_cell, load_cell

# B-cell markers used for membrane-patch (trogocytosis) detection. From the PNA preprint's
# patch method; all three are in the 159-marker panel.
PATCH_MARKERS = ("CD20", "CD22", "CD40")

# B-cell immunological-synapse panels (from synapse_analysis.ipynb / synapse_scores_design.md).
# Scored as LNE too: on a CD8 cell they flag B-side synapse markers trogocytosed onto the T
# membrane, split into the activating Signal-2 hub vs the inhibitory brake-raft. Full panels.
ACTIVATING_MARKERS = ("HLA-DR-DP-DQ", "HLA-DR", "HLA-DQ", "HLA-ABC", "CD80", "CD86", "CD40",
                      "CD19", "CD20", "CD79a", "CD54", "CD58", "CD50", "CD102")
INHIBITORY_MARKERS = ("CD274", "CD273", "CD32", "CD72", "CD305", "CD22", "CD66b", "CD162")

# Features that get a null ratio / z. Degree features are excluded: the edge-swap null
# preserves degree exactly, so their null distribution is a point mass at the observed value.
NULL_FEATURES = (
    "transitivity", "avg_clustering", "average_k_core", "n_components",
    "largest_cc_frac", "proj_density", "proj_degree_mean", "degree_assortativity",
    # whole-graph (raw, sides ignored) connectivity. g_density is edge-swap invariant -> no null.
    "g_n_components", "g_largest_cc_frac", "g_average_k_core", "g_degree_assortativity",
)

# Path/diameter on the raw graph: exact all-pairs shortest paths are infeasible on ~60k-node
# cells, so they are computed on a random node-subsample of the largest component, this size.
PATH_SUBSAMPLE = 600


def _gini(arr):
    """Gini coefficient via the sorted-array formula. O(n log n)."""
    arr = np.sort(np.asarray(arr, dtype=float))
    n = len(arr)
    s = arr.sum()
    if n == 0 or s == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * arr) - (n + 1) * s) / (n * s))


# ---------------------------------------------------------------------------
# Edge-budget matching
# ---------------------------------------------------------------------------

def subsample_edges(edges, budget, seed=0):
    """Uniformly subsample `budget` edges without replacement (molecule downsampling).

    Each PNA edge is one proximity-ligation molecule, so dropping edges at random is what a
    shallower run of the same cell would have produced. Nodes left with no edges fall out of
    the induced subgraph -- that shrinkage is the point: it is what equalises mean degree
    across cells of different depth.

    Returns None when the cell has fewer edges than `budget`. Such a cell CANNOT be depth
    matched, and returning it unsubsampled would quietly put an unmatched cell back into a
    matched comparison -- reintroducing exactly the confound the budget exists to remove.
    The caller must drop it (see `budget_met`).
    """
    # Falsy budget (None or 0) means "no matching". Testing `is None` only is a trap: a
    # caller passing 0 would fall through to rng.choice(size=0) and get an EMPTY graph,
    # which then dies downstream on idx.max() of an empty array.
    if not budget:
        return edges
    if edges.shape[0] < budget:
        return None
    if edges.shape[0] == budget:
        return edges
    rng = np.random.default_rng(int(seed))
    keep = rng.choice(edges.shape[0], size=int(budget), replace=False)
    return edges[keep]


# ---------------------------------------------------------------------------
# Cohort selection -- how depth IS controlled (see module docstring)
# ---------------------------------------------------------------------------

def depth_band(obs, lo, hi, depth_col="n_edges"):
    """Cells whose NATIVE depth falls in [lo, hi]. No graph is modified.

    Balancing depth by choosing which cells to compare, rather than by mutating graphs, is
    the only control here that actually works. For the CD8 system contrast a 90k-150k band
    leaves 726 cells (NALM 338 / HB 388) with depth MWU p=0.76 -- balanced by construction.
    """
    return obs[(obs[depth_col] >= lo) & (obs[depth_col] <= hi)]


def match_on_depth(obs, group_col, group_a, group_b, depth_col="n_edges", caliper=0.05,
                   seed=0):
    """Greedy 1:1 nearest-neighbour match of group_a to group_b on depth.

    A band alone does not balance Blina vs Mock -- inside a 90-150k band the two still differ
    (p<0.001, fold 1.10-1.18) because Blinatumomab cells are systematically deeper. This pairs
    each group_a cell with the nearest unused group_b cell in log-depth, rejecting pairs whose
    relative depth gap exceeds `caliper`. Returns the index of the matched cells (both arms).
    """
    rng = np.random.default_rng(int(seed))
    a = obs[obs[group_col] == group_a]
    b = obs[obs[group_col] == group_b]
    if a.empty or b.empty:
        return obs.index[:0]
    la = np.log(a[depth_col].to_numpy(float))
    lb = np.log(b[depth_col].to_numpy(float))
    order = rng.permutation(len(la))          # de-bias greedy order
    used = np.zeros(len(lb), dtype=bool)
    keep_a, keep_b = [], []
    for i in order:
        gaps = np.abs(lb - la[i])
        gaps[used] = np.inf
        j = int(np.argmin(gaps))
        if not np.isfinite(gaps[j]) or gaps[j] > caliper:
            continue
        used[j] = True
        keep_a.append(a.index[i])
        keep_b.append(b.index[j])
    return pd.Index(list(keep_a) + list(keep_b))


# ---------------------------------------------------------------------------
# Topology (blocks 2-4)
# ---------------------------------------------------------------------------

def _build_igraph(edges, side):
    """Bipartite igraph over the nodes actually touched by `edges`.

    Returns (G, idx, n1, n2) where G.vs['type'] is False for side-1 (umi1 / A-pixel) nodes,
    so bipartite_projection(which=0) yields the A-pixel one-mode graph the MPX/PNA papers use.
    `idx` maps compact igraph vertex ids back to the caller's node ids.
    """
    idx = np.unique(edges)
    if idx.size == 0:
        return ig.Graph(n=0, directed=False), idx, 0, 0
    remap = np.full(int(idx.max()) + 1, -1, dtype=np.int64)
    remap[idx] = np.arange(idx.size)
    e = remap[edges]
    G = ig.Graph(n=idx.size, edges=[tuple(r) for r in e], directed=False)
    types = (side[idx] == 2)
    G.vs["type"] = types.tolist()
    return G, idx, int((~types).sum()), int(types.sum())


def compute_topology(edges, side, path_cc_cap=1000, want_path=True):
    """Blocks 2-4 for one graph. Returns a flat dict of scalars.

    Bipartite degree stats are exact; projected-graph stats (clustering, coreness,
    components, assortativity) are the ones that need a null to be interpretable.
    """
    G, idx, n1, n2 = _build_igraph(edges, side)
    E = G.ecount()
    # NB: u1_u2_frac, not *_ratio -- the `_ratio` suffix is reserved for obs/null statistics.
    out = {"n_nodes": int(idx.size), "n1": n1, "n2": n2, "n_edges": E,
           "bip_density": E / (n1 * n2) if n1 and n2 else np.nan,
           "u1_u2_frac": n1 / n2 if n2 else np.nan}

    is1 = np.asarray(G.vs["type"]) == False  # noqa: E712 - numpy elementwise
    deg = np.asarray(G.degree())
    # combined-side degree (all nodes, both sides) -- the "regardless of side" view
    if deg.size:
        out["deg_mean"] = float(deg.mean())
        out["deg_std"] = float(deg.std())
        out["deg_skew"] = float(stats.skew(deg)) if deg.size > 2 else np.nan
        out["deg_gini"] = _gini(deg)
        out["deg_kappa"] = (float((deg ** 2).mean() / deg.mean() ** 2)
                            if deg.mean() > 0 else np.nan)
    for tag, mask in (("u1", is1), ("u2", ~is1)):
        d = deg[mask]
        if d.size == 0:
            continue
        out[f"{tag}_degree_mean"] = float(d.mean())
        out[f"{tag}_degree_median"] = float(np.median(d))
        out[f"{tag}_degree_max"] = int(d.max())
        out[f"{tag}_degree_std"] = float(d.std())
        out[f"{tag}_degree_skew"] = float(stats.skew(d)) if d.size > 2 else np.nan
        out[f"{tag}_degree_gini"] = _gini(d)
        # heterogeneity kappa = <d^2>/<d>^2 ; 1.0 = regular graph, grows with hubbiness
        m1 = d.mean()
        out[f"{tag}_kappa"] = float((d ** 2).mean() / m1 ** 2) if m1 > 0 else np.nan

    # Whole-graph topology: treat the cell as a plain graph, every node a node (sides ignored).
    # A strictly bipartite graph has no triangles, so transitivity/clustering are identically 0
    # and are not emitted -- only connectivity metrics carry information here.
    gcc = G.connected_components()
    gsz = gcc.sizes()
    glargest = max(gsz) if gsz else 0
    out.update({
        "g_density": G.density(),
        "g_n_components": len(gsz),
        "g_largest_cc_frac": glargest / idx.size if idx.size else np.nan,
        "g_average_k_core": float(np.mean(G.coreness())) if idx.size else np.nan,
        "g_degree_assortativity": G.assortativity_degree(directed=False) if E else np.nan,
    })
    if want_path and glargest > 1:
        lcc = [v for v, m in enumerate(gcc.membership) if m == gsz.index(glargest)]
        if len(lcc) > PATH_SUBSAMPLE:
            lcc = np.random.default_rng(0).choice(lcc, PATH_SUBSAMPLE, replace=False).tolist()
        giant = G.subgraph(lcc).connected_components().giant()
        if giant.vcount() > 1:
            out["g_avg_path_length"] = giant.average_path_length()
            out["g_diameter"] = giant.diameter()

    if n1 < 2:
        return out

    P = G.bipartite_projection(which=0)
    pdeg = np.asarray(P.degree())
    cc = P.connected_components()
    sizes = cc.sizes()
    largest = max(sizes) if sizes else 0

    out.update({
        "proj_n_edges": P.ecount(),
        "proj_density": P.density(),
        "proj_degree_mean": float(pdeg.mean()),
        "proj_degree_max": int(pdeg.max()) if pdeg.size else 0,
        "proj_degree_gini": _gini(pdeg),
        "n_components": len(sizes),
        "largest_cc_frac": largest / n1 if n1 else np.nan,
        "transitivity": P.transitivity_undirected(mode="nan"),
        "avg_clustering": P.transitivity_avglocal_undirected(mode="zero"),
        # average_k_core: the pixelator QC connectivity metric (intact surface at >1.5)
        "average_k_core": float(np.mean(P.coreness())),
        "degree_assortativity": P.assortativity_degree(directed=False),
    })

    if want_path and 1 < largest <= path_cc_cap:
        sub = P.subgraph([v for v, m in enumerate(cc.membership)
                          if m == sizes.index(largest)])
        out["avg_path_length"] = sub.average_path_length()
        out["diameter"] = sub.diameter()
    return out


def topology_with_null(edges, side, n_perm=20, seed=0, path_cc_cap=1000):
    """Observed topology plus `_ratio` and `_z` against the degree-preserving null.

    Null = bipartite edge swap (both side degree sequences fixed), so this isolates wiring
    structure from the degree sequence. Path/spectral features are skipped in the null loop
    (want_path=False) -- they are the expensive ones and are reported observed-only.
    """
    obs = compute_topology(edges, side, path_cc_cap=path_cc_cap, want_path=True)
    if n_perm <= 0:
        return obs

    draws = {k: [] for k in NULL_FEATURES}
    for i in range(n_perm):
        ne = bipartite_edgeswap_sample(edges, side, seed=seed + i)
        nt = compute_topology(ne, side, path_cc_cap=path_cc_cap, want_path=False)
        for k in NULL_FEATURES:
            v = nt.get(k)
            if v is not None and np.isfinite(v):
                draws[k].append(v)

    for k in NULL_FEATURES:
        o = obs.get(k)
        vals = np.asarray(draws[k], dtype=float)
        if o is None or not np.isfinite(o) or vals.size < 2:
            continue
        mu, sd = float(vals.mean()), float(vals.std(ddof=1))
        obs[f"{k}_null_mean"] = mu
        obs[f"{k}_ratio"] = o / mu if mu != 0 else np.nan
        obs[f"{k}_z"] = (o - mu) / sd if sd > 1e-12 else np.nan
    return obs


# ---------------------------------------------------------------------------
# Local proximity / Local Neighbourhood Enrichment (block 6) -- PixelGen method
# (pixelatorR::local_proximity, analytical, mode="any"; see cache/local_proximity_scores.R)
# ---------------------------------------------------------------------------

def expand_adjacency_matrix(A, k):
    """Binary k-hop reachability matrix: (i, j) = 1 iff a path of length <= k joins i, j.

    Port of pixelatorR::expand_adjacency_matrix -- binarise((A + I)^k). The diagonal is 1, so
    a node's k-hop neighbourhood includes itself. int32 avoids overflow of the intermediate
    common-neighbour counts before each binarisation step.
    """
    n = A.shape[0]
    R = (A + sp.eye(n, format="csr")).astype(np.int32)
    R.data[:] = 1
    Ak = R.copy()
    for _ in range(k - 1):
        Ak = Ak @ R
        Ak.data[:] = 1              # reachability, not walk count
    return Ak


def local_proximity(edges, labels, side, marker_idx, K, k=3):
    """Per-node analytical LNE (log2 observed/expected), mode="any", for a marker set.

    Port of pixelatorR::local_proximity (analytical, mode="any"). Each PNA node carries exactly
    one marker, so the observed statistic is the count of nodes carrying ANY marker in
    `marker_idx` within the k-hop neighbourhood (self included). Expected is the cell's own
    marker frequency x neighbourhood size, computed PER SIDE (umi1/umi2 have different marker
    frequencies), so it isolates *local* enrichment from the cell-global abundance.
    """
    A, L, _, _, _ = build_cell(edges, labels, K)
    Ak = expand_adjacency_matrix(A, k)

    xvec = np.asarray(L[:, sorted(marker_idx)].sum(1)).ravel()   # 1 if node carries a patch marker
    obs = Ak @ xvec                                              # any-mode: sum over the neighbourhood

    is1 = (side == 1).astype(np.float64)
    is2 = (side == 2).astype(np.float64)
    deg_A, deg_B = Ak @ is1, Ak @ is2                            # per-side neighbourhood sizes (incl self)
    nA, nB = int(is1.sum()), int(is2.sum())
    f1 = xvec[side == 1].sum() / nA if nA else 0.0               # any-mode: sum of per-marker side freqs
    f2 = xvec[side == 2].sum() / nB if nB else 0.0
    exp = f1 * deg_A + f2 * deg_B

    floor = k - 1
    return np.log2(np.maximum(obs, floor) / np.maximum(exp, floor))


def lne_features(edges, labels, side, patch_idx, K, k=3, prefix=""):
    """Per-cell summaries of the analytical LNE over a marker set (mode="any").

    Replaces the old detect_patches. LNE is a *local relative-enrichment* score, so the signal
    for focal trogocytosis patches lives in the tail (lne_max / lne_p90), not the mean: a
    uniformly B-marked B cell scores ~0 everywhere, a CD8 with a focal transferred patch spikes
    locally. lne_pos_frac (a fraction) is the depth-robust coverage analog; lne_marker_nodes is
    the raw #patch-marker nodes (depth-linked), kept for continuity with patch_marker_nodes.

    `prefix` namespaces the keys so several marker sets can live in one row (e.g. prefix="act_"
    yields act_lne_mean, ...); the default "" keeps the original trogocytosis-patch column names.
    """
    keys = ("lne_mean", "lne_p90", "lne_max", "lne_pos_frac", "lne_marker_nodes")
    empty = {f"{prefix}{k_}": 0.0 for k_ in keys[:-1]} | {f"{prefix}lne_marker_nodes": 0}
    if edges.shape[0] == 0 or not patch_idx:
        return empty
    lne = local_proximity(edges, labels, side, patch_idx, K, k=k)
    return {f"{prefix}lne_mean": float(lne.mean()),
            f"{prefix}lne_p90": float(np.percentile(lne, 90)),
            f"{prefix}lne_max": float(lne.max()),
            f"{prefix}lne_pos_frac": float((lne > 0).mean()),
            f"{prefix}lne_marker_nodes": int(np.isin(labels, sorted(patch_idx)).sum())}


# ---------------------------------------------------------------------------
# Polarization (block 5) -- free, read off the existing chunglu doublet self-pairs
# ---------------------------------------------------------------------------

def polarization_summary(doublets_path, min_abs_z=None):
    """Per-cell self-clustering (polarization) summary from a chunglu doublet parquet.

    The PNA "self-clustering proximity score" is the self-pair (marker_1 == marker_2) join
    count against the degree-corrected null -- already computed per cell by chunglu_triplets.
    Returns a frame indexed by component with mean/max/std/gini of the self-pair z and
    log2_ratio, plus the Gini of polarization that the CAR-T MPX paper used to show
    exhaustion destabilises the population.
    """
    d = pd.read_parquet(doublets_path,
                        columns=["component", "marker_1", "marker_2",
                                 "join_count_z", "log2_ratio"])
    d = d[d.marker_1 == d.marker_2]
    if min_abs_z is not None:
        d = d[d.join_count_z.abs() >= min_abs_z]
    g = d.groupby("component")
    out = pd.DataFrame({
        "pol_z_mean": g.join_count_z.mean(),
        "pol_z_max": g.join_count_z.max(),
        "pol_z_std": g.join_count_z.std(),
        "pol_lr_mean": g.log2_ratio.mean(),
        "pol_n_markers": g.size(),
    })
    # Gini needs strictly non-negative input; shift the z distribution per cell.
    out["pol_z_gini"] = g.join_count_z.apply(lambda s: _gini(s - min(s.min(), 0)))
    return out


# ---------------------------------------------------------------------------
# Per-cell worker
# ---------------------------------------------------------------------------

def _process_batch(args):
    """One row per (component, edge_budget). Long form -- the notebook pivots on budget."""
    cells, marker_names, budgets, n_perm, patch_markers, path_cc_cap = args
    marker_to_idx = {m: i for i, m in enumerate(marker_names)}
    patch_idx = {marker_to_idx[m] for m in patch_markers if m in marker_to_idx}
    act_idx = {marker_to_idx[m] for m in ACTIVATING_MARKERS if m in marker_to_idx}
    inh_idx = {marker_to_idx[m] for m in INHIBITORY_MARKERS if m in marker_to_idx}
    rows = []
    for component, comp_el in cells:
        try:
            edges, labels, side, n = load_cell(comp_el, marker_to_idx)
            if edges.shape[0] == 0:
                continue
            # Deterministic per-cell seed. Not hash(): it is salted per process, so a rerun
            # would draw different nulls and the parquet would not reproduce.
            cseed = zlib.crc32(component.encode()) % (2 ** 31)
            # LNE runs once on the FULL graph at native depth: it is a depth-dependent
            # enrichment score, so the native graph is the reference (not a subsampled one).
            # Trogocytosis patch set + the activating / inhibitory B-synapse panels (act_/inh_).
            K = len(marker_names)
            patch = lne_features(edges, labels, side, patch_idx, K)
            patch.update(lne_features(edges, labels, side, act_idx, K, prefix="act_"))
            patch.update(lne_features(edges, labels, side, inh_idx, K, prefix="inh_"))
            for budget in budgets:
                row = {"component": component, "n_edges_full": int(edges.shape[0]),
                       "edge_budget": budget or 0, **patch}
                sub = subsample_edges(edges, budget, seed=cseed)
                row["budget_met"] = sub is not None
                if sub is not None:
                    row.update(topology_with_null(sub, side, n_perm=n_perm, seed=cseed,
                                                  path_cc_cap=path_cc_cap))
                rows.append(row)
        except Exception as e:
            print(f"  WARN component {component}: {type(e).__name__}: {e}", flush=True)
    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------
# Per-sample driver
# ---------------------------------------------------------------------------

def run_sample(sample, adata_path, pxl_path, output_dir, n_workers=16, min_umi=25000,
               budgets=(None,), n_perm=20, cell_types=None, path_cc_cap=1000,
               marker_names_path=None):
    import anndata as ad
    from pixelator import read_pna

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(adata_path, backed="r")
    obs = adata.obs

    # Reuse the chunglu marker list when present so marker indices line up across modules.
    mn_path = Path(marker_names_path) if marker_names_path else output_dir / "marker_names.json"
    if mn_path.exists():
        marker_names = json.loads(mn_path.read_text())
    else:
        marker_names = adata.var_names.tolist()
        mn_path.write_text(json.dumps(marker_names))

    sel = obs[obs["sample"] == sample]
    if "tau_type" in sel:
        sel = sel[sel["tau_type"] == "normal"]
    if "n_umi" in sel:
        sel = sel[sel["n_umi"] >= min_umi]
    if cell_types:
        sel = sel[sel["cell_type_annot"].isin(cell_types)]
    components = sel.index.tolist()
    print(f"[{sample}] {len(components)} cells | budgets={list(budgets)} n_perm={n_perm}"
          f"{f' | types={cell_types}' if cell_types else ''}", flush=True)
    if not components:
        print(f"[{sample}] no cells, exiting", flush=True)
        return

    pg = read_pna([pxl_path])

    def load_chunk(comps):
        out = []
        for comp in comps:
            el = pg.filter(components=[comp]).edgelist().to_df()
            out.append((comp, el[["umi1", "umi2", "marker_1", "marker_2"]]))
        return out

    frames = []
    chunk_size = max(n_workers * 2, 16)
    pool = None
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=n_workers)
    t0, done = time.time(), 0
    for chunk in _batched(components, int(np.ceil(len(components) / chunk_size))):
        cells = load_chunk(chunk)
        work = [(b, marker_names, list(budgets), n_perm, PATCH_MARKERS, path_cc_cap)
                for b in _batched(cells, n_workers)]
        results = pool.map(_process_batch, work) if pool else [_process_batch(w) for w in work]
        for df in results:
            if df is not None:
                frames.append(df)
        done += len(cells)
        print(f"[{sample}]   {done}/{len(components)} cells ({time.time()-t0:.0f}s)", flush=True)
    if pool is not None:
        pool.shutdown()

    out = pd.concat(frames, ignore_index=True)
    out["sample"] = sample
    path = output_dir / f"{sample}_graph_phenotype.parquet"
    out.to_parquet(path)
    met = out.groupby("edge_budget")["budget_met"].mean()
    print(f"[{sample}] wrote {path} ({out.component.nunique():,} cells x {len(budgets)} "
          f"budgets = {len(out):,} rows, {out.shape[1]} cols) in {time.time()-t0:.0f}s\n"
          f"[{sample}] budget_met fraction: {met.round(3).to_dict()}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Per-cell graph phenotype panel for one PNA sample")
    p.add_argument("--sample", required=True)
    p.add_argument("--adata", required=True)
    p.add_argument("--pxl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--min-umi", type=int, default=25000)
    p.add_argument("--edge-budgets", default="0",
                   help="Comma-separated edge budgets (molecule downsampling); 0 = native "
                        "depth. DEFAULT 0 AND YOU ALMOST CERTAINLY WANT TO LEAVE IT: "
                        "downsampling measurably worsens the depth confound (mean |rho| "
                        "0.40 -> 0.88; see module docstring). Depth is handled by cohort "
                        "selection in the notebook. Non-zero budgets reproduce the depth "
                        "curve as a diagnostic only.")
    p.add_argument("--n-perm", type=int, default=20,
                   help="Degree-preserving null draws per cell. 0 disables ratio/z.")
    p.add_argument("--cell-types", default=None,
                   help="Comma-separated cell_type_annot values to keep (default: all)")
    p.add_argument("--marker-names", default=None,
                   help="marker_names.json to reuse (default: <output-dir>/marker_names.json)")
    a = p.parse_args()
    budgets = [int(b) or None for b in a.edge_budgets.split(",")]
    run_sample(a.sample, a.adata, a.pxl, a.output_dir,
               n_workers=a.n_workers, min_umi=a.min_umi,
               budgets=budgets, n_perm=a.n_perm,
               cell_types=a.cell_types.split(",") if a.cell_types else None,
               marker_names_path=a.marker_names)


if __name__ == "__main__":
    main()
