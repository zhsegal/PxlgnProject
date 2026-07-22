"""Chung-Lu degree-corrected colocalization (doublets + triplets) for PNA cells.

Per cell, under the soft Chung-Lu null (fix every node's marker label and degree,
randomize wiring; p_ij = d_i d_j / 2E), every motif count has a closed-form mean and
variance in the per-marker degree-moment strengths
    s_X = sum_{v in X} d_v,  r_X = sum d_v^2,  t_X = sum d_v^3,  u_X = sum d_v^4 .

We compute, per cell:
  * doublets  J_AB  (marker pair join count)            -> stored per cell (long form)
  * triplets  W_ABC (unordered open wedge / 2-path)     -> stream-aggregated over cells

PNA graph model: node = one UMI molecule carrying exactly one marker; edge = proximity
ligation. The edgelist is bipartite at the umi-id level (umi1 / umi2 are disjoint id
spaces) but the math below only needs the simple undirected degree sequence.

Formulas are from PROMPT_run_chunglu_on_real_pna.md (sections 3-6); see that file for
derivations. Pitfalls honored: P1 (2E = sum d, every denominator), P2 (undirected simple
graph), P3 (one label per node), P5 (wedge only, A<B<C), P7 (clip Var>0), P8 (Poisson for
rare), P9 (stream triplets, never store per cell x per triple).

CLI (one sample per process; see run_chunglu_triplets.lsf):
    python chunglu_triplets.py --sample S001 \
        --adata cache/adata_annotated.h5ad \
        --pxl results/S001/layout/layout/S001.layout.pxl \
        --output-dir results/chunglu --n-workers 16
"""

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# Canonical per-cell objects
# ---------------------------------------------------------------------------

def load_cell(comp_el, marker_to_idx):
    """Edgelist of ONE component -> (edges, labels, n).

    comp_el: DataFrame with columns umi1, umi2, marker_1, marker_2 (one row per edge).
    marker_to_idx: dict marker_name -> global index 0..K-1 (identical across all cells).

    Returns
        edges  : int array (E, 2), nodes relabeled 0..n-1, undirected, de-duplicated, no self-loops.
        labels : int array (n,), labels[i] = global marker index of node i.
        side   : int array (n,), side[i] in {1,2} -- the PNA graph is bipartite (umi1 / umi2 sides;
                 edges form only across sides), so the null is computed per side.
        n      : number of nodes.
    """
    u1 = comp_el["umi1"].to_numpy()
    u2 = comp_el["umi2"].to_numpy()
    m1 = comp_el["marker_1"].to_numpy()
    m2 = comp_el["marker_2"].to_numpy()

    # node id -> (marker, side) (P3: one marker per node). umi1 -> side 1, umi2 -> side 2.
    node_marker, node_side = {}, {}
    for arr_u, arr_m, sd in ((u1, m1, 1), (u2, m2, 2)):
        for nid, mk in zip(arr_u, arr_m):
            prev = node_marker.get(nid)
            if prev is None:
                node_marker[nid] = mk
                node_side[nid] = sd
            elif prev != mk:
                raise ValueError(f"node {nid} maps to >1 marker: {prev} vs {mk}")

    # contiguous relabel 0..n-1
    uid = {nid: i for i, nid in enumerate(node_marker)}
    n = len(uid)
    labels = np.empty(n, dtype=np.int64)
    side = np.empty(n, dtype=np.int64)
    for nid, i in uid.items():
        labels[i] = marker_to_idx[node_marker[nid]]
        side[i] = node_side[nid]

    a = np.fromiter((uid[x] for x in u1), dtype=np.int64, count=len(u1))
    b = np.fromiter((uid[x] for x in u2), dtype=np.int64, count=len(u2))

    # P2: drop self-loops, de-duplicate undirected {u,v}
    keep = a != b
    a, b = a[keep], b[keep]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    key = lo.astype(np.int64) * n + hi
    _, uniq = np.unique(key, return_index=True)
    edges = np.column_stack((lo[uniq], hi[uniq]))
    return edges, labels, side, n


def build_cell(edges, labels, K):
    """Shared per-cell matrices (PROMPT section 3). Returns A, L, d, twoE, N."""
    n = labels.shape[0]
    u, v = edges[:, 0], edges[:, 1]
    data = np.ones(2 * len(u))
    A = sp.csr_matrix((data, (np.r_[u, v], np.r_[v, u])), shape=(n, n))
    A.data[:] = 1.0                    # collapse any accidental multiedges to 1
    A.setdiag(0)
    A.eliminate_zeros()                # no self-loops
    d = np.asarray(A.sum(1)).ravel()   # degree vector (float)
    twoE = float(d.sum())              # = 2 * number_of_edges  (P1)
    L = sp.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, K))
    N = (A @ L).toarray()              # N[i, Y] = # of Y-labelled neighbours of node i
    return A, L, d, twoE, N


def side_strengths(d, labels, side, K):
    """Per-marker, per-side degree-moment strengths -> (s1,r1,t1, s2,r2,t2), each length K.

    The PNA graph is bipartite: edges form only across sides, so the Chung-Lu null is
    derived per side. Normaliser is m = number of edges = d.sum()/2 (NOT 2E = 2m).
    """
    def moms(mask):
        lab, dd = labels[mask], d[mask]
        return (np.bincount(lab, weights=dd,        minlength=K),
                np.bincount(lab, weights=dd ** 2,   minlength=K),
                np.bincount(lab, weights=dd ** 3,   minlength=K))
    s1, r1, t1 = moms(side == 1)
    s2, r2, t2 = moms(side == 2)
    return s1, r1, t1, s2, r2, t2


# ---------------------------------------------------------------------------
# Doublets (PROMPT section 4) -- exact for the pair (independent edges)
# ---------------------------------------------------------------------------

def doublet_table(component, A, L, d, labels, side, K, marker_names):
    """Per-cell doublet long-form table (bipartite null), observed pairs (J>0, A<=B).

    Includes self-pairs (A=A, polarization) as PNA proximity tables do.
    """
    J = (L.T @ (A @ L)).toarray()                  # K x K ; off-diag = observed J_XY
    s1, r1, _, s2, r2, _ = side_strengths(d, labels, side, K)
    m = d.sum() / 2.0

    # off-diagonal (A != B): two cross-side configurations
    mean_pair = (np.outer(s1, s2) + np.outer(s2, s1)) / m
    var_pair = mean_pair - (np.outer(r1, r2) + np.outer(r2, r1)) / m ** 2
    # self-pairs (A == B): the two terms collapse to ONE configuration
    np.fill_diagonal(mean_pair, (s1 * s2) / m)
    np.fill_diagonal(var_pair, (s1 * s2) / m - (r1 * r2) / m ** 2)

    # observed self-count is the halved diagonal (within-X cross edge counted at both ends)
    Jobs = J.copy()
    np.fill_diagonal(Jobs, np.diag(J) / 2.0)

    iu, ju = np.triu_indices(K, k=0)               # include diagonal (self-pairs)
    Jv = Jobs[iu, ju]
    keep = Jv > 0
    iu, ju, Jv = iu[keep], ju[keep], Jv[keep]
    mv = mean_pair[iu, ju]
    vv = np.clip(var_pair[iu, ju], 1e-12, None)    # P7
    z = (Jv - mv) / np.sqrt(vv)
    log2r = np.log2(np.clip(Jv, 1e-9, None) / np.clip(mv, 1e-9, None))
    pois = poisson.sf(Jv - 1, mv)                  # P8: prefer where mv < 10

    return pd.DataFrame({
        "component": component,
        "marker_1": [marker_names[i] for i in iu],
        "marker_2": [marker_names[j] for j in ju],
        "J": Jv,
        "EW": mv,
        "Var": vv,
        "join_count_z": z,
        "log2_ratio": log2r,
        "poisson_p": pois,
    })


# ---------------------------------------------------------------------------
# Triplets (PROMPT section 5) -- unordered open wedge (2-path)
# ---------------------------------------------------------------------------

def _o(x, y, z):
    return np.einsum("a,b,c->abc", x, y, z, optimize=True)


def center_tensor(N, labels, K):
    """T[g,X,Y] = sum_{h: label g} N[h,X] N[h,Y]  (per-center co-neighbour tensor)."""
    T = np.zeros((K, K, K))
    for g in range(K):
        idx = np.where(labels == g)[0]
        if idx.size:
            Ng = N[idx]
            T[g] = Ng.T @ Ng
    return T


def observed_wedges(T):
    """W_obs[A,B,C] = T[A,B,C] + T[B,A,C] + T[C,A,B] (center can be any of the three)."""
    return T + np.transpose(T, (1, 0, 2)) + np.transpose(T, (1, 2, 0))


def triplet_mean(s1, r1, s2, r2, m):
    """Bipartite wedge mean for the FIXED-DEGREE (configuration) null, exact to leading order.

    A node anchoring k motif edges contributes the falling factorial d^(k) instead of d^k
    (soft Chung-Lu uses d^k). The wedge center anchors 2 edges -> use rc = r - s (= sum d(d-1));
    the two spokes anchor 1 edge each -> s (unchanged). This matches the degree-preserving
    edge-swap null on the real graph (soft would overestimate by ~1 - 1/dbar)."""
    rc1, rc2 = r1 - s1, r2 - s2
    return (_o(rc1, s2, s2) + _o(s2, rc1, s2) + _o(s2, s2, rc1)        # center side 1, spokes side 2
            + _o(rc2, s1, s1) + _o(s1, rc2, s1) + _o(s1, s1, rc2)) / m ** 2   # center side 2


def triplet_var_lead(EW, s1, r1, t1, s2, r2, t2, m):
    """Bipartite leading-order wedge variance for the fixed-degree null.

    Same falling-factorial rule applied to single-node self-moments: r -> rc = r - s,
    t -> tc = t - 3r + 2s (= sum d(d-1)(d-2)). Products of distinct nodes (s, s^2, s1*s2)
    are unchanged. Validate the SD against edge-swap MC (variance is reported, not gated)."""
    rc1, rc2 = r1 - s1, r2 - s2
    tc1, tc2 = t1 - 3 * r1 + 2 * s1, t2 - 3 * r2 + 2 * s2
    s1sq, s2sq, s1s2 = s1 * s1, s2 * s2, s1 * s2
    Cc = (_o(tc1, s2, s2sq) + _o(s2, tc1, s2sq)
          + 2 * _o(rc1, rc2, s1s2) + 2 * _o(rc2, rc1, s1s2)
          + _o(s1, tc2, s1sq) + _o(tc2, s1, s1sq))
    bracket = Cc + np.transpose(Cc, (2, 0, 1)) + np.transpose(Cc, (1, 2, 0))
    return EW + bracket / m ** 3


def triplet_moments_for_list(s1, r1, t1, s2, r2, t2, m, triples, return_centers=False):
    """Scalar (per-triple) form of triplet_mean / triplet_var_lead, vectorized over a list.

    triples: int array (T, 3) of distinct marker indices. Avoids the dense K^3 tensors --
    just indexes the same falling-factorial terms. Returns (EW, Var), each shape (T,).
    The tensor transposes (2,0,1)/(1,2,0) reduce to the cyclic shifts Cc(a,b,c)+Cc(b,c,a)+Cc(c,a,b).

    return_centers=True additionally returns the PER-CENTER means (EW_cA, EW_cB, EW_cC):
    the wedge mean split by which of the three markers is the wedge center (anchors 2 edges).
    center A pairs spokes (B, C); center B pairs (A, C); center C pairs (A, B). They sum to EW.
    Used by the pairwise (connected) decomposition, which scales each center by the enrichments
    of its two incident pairs.
    """
    ia, ib, ic = triples[:, 0], triples[:, 1], triples[:, 2]
    rc1, rc2 = r1 - s1, r2 - s2
    tc1, tc2 = t1 - 3 * r1 + 2 * s1, t2 - 3 * r2 + 2 * s2
    s1sq, s2sq, s1s2 = s1 * s1, s2 * s2, s1 * s2

    # per-center means (center anchors 2 edges -> rc = r - s; the two spokes -> s), summed over sides
    EW_cA = (rc1[ia] * s2[ib] * s2[ic] + rc2[ia] * s1[ib] * s1[ic]) / m ** 2
    EW_cB = (s2[ia] * rc1[ib] * s2[ic] + s1[ia] * rc2[ib] * s1[ic]) / m ** 2
    EW_cC = (s2[ia] * s2[ib] * rc1[ic] + s1[ia] * s1[ib] * rc2[ic]) / m ** 2
    EW = EW_cA + EW_cB + EW_cC

    def Cc(a, b, c):
        return (tc1[a] * s2[b] * s2sq[c] + s2[a] * tc1[b] * s2sq[c]
                + 2 * rc1[a] * rc2[b] * s1s2[c] + 2 * rc2[a] * rc1[b] * s1s2[c]
                + s1[a] * tc2[b] * s1sq[c] + tc2[a] * s1[b] * s1sq[c])

    bracket = Cc(ia, ib, ic) + Cc(ib, ic, ia) + Cc(ic, ia, ib)
    Var = EW + bracket / m ** 3
    if return_centers:
        return EW, Var, EW_cA, EW_cB, EW_cC
    return EW, Var


def triplet_list_table(component, N, labels, side, d, K, triples, marker_names):
    """Per-cell wedge table for a CLOSED LIST of triples (mirrors doublet_table).

    triples: int array (T, 3), each row sorted distinct marker indices. Observed W via the
    wedge_count formula read off per-center co-neighbour matrices M_g = Ng^T Ng (only for
    centers present in this cell -- no dense K^3). Returns one row per listed triple.
    """
    m = d.sum() / 2.0
    s1, r1, t1, s2, r2, t2 = side_strengths(d, labels, side, K)
    EW, Var, EW_cA, EW_cB, EW_cC = triplet_moments_for_list(
        s1, r1, t1, s2, r2, t2, m, triples, return_centers=True)

    # per-center co-neighbour matrices, only for marker labels appearing in the list & present
    present = set(np.unique(labels).tolist())
    centers = [g for g in np.unique(triples).tolist() if g in present]
    Mg = {}
    for g in centers:
        idx = np.where(labels == g)[0]
        Ng = N[idx]
        Mg[g] = Ng.T @ Ng                      # K x K (sum_{h: label g} N[h,X] N[h,Y])

    def _center_term(g_arr, x_arr, y_arr):
        out = np.zeros(g_arr.shape[0])
        for i, (g, x, y) in enumerate(zip(g_arr, x_arr, y_arr)):
            Mgi = Mg.get(int(g))
            if Mgi is not None:
                out[i] = Mgi[x, y]
        return out

    ia, ib, ic = triples[:, 0], triples[:, 1], triples[:, 2]
    W_cA = _center_term(ia, ib, ic)            # center A (label marker_1), co-neighbours B,C
    W_cB = _center_term(ib, ia, ic)            # center B
    W_cC = _center_term(ic, ia, ib)            # center C
    W = W_cA + W_cB + W_cC

    vv = np.clip(Var, 1e-12, None)             # P7
    z = (W - EW) / np.sqrt(vv)
    log2r = np.log2(np.clip(W, 1e-9, None) / np.clip(EW, 1e-9, None))

    # per-center observed wedge + null mean (sum to W / EW). Enable the pairwise (connected)
    # decomposition: scale each center by the enrichments of its two incident pairs.
    return pd.DataFrame({
        "component": component,
        "marker_1": [marker_names[i] for i in ia],
        "marker_2": [marker_names[i] for i in ib],
        "marker_3": [marker_names[i] for i in ic],
        "W": W,
        "EW": EW,
        "Var": vv,
        "triplet_z": z,
        "log2_ratio": log2r,
        "W_cA": W_cA, "W_cB": W_cB, "W_cC": W_cC,
        "EW_cA": EW_cA, "EW_cB": EW_cB, "EW_cC": EW_cC,
    })


# ---------------------------------------------------------------------------
# Per-cell worker + streaming triplet accumulator
# ---------------------------------------------------------------------------

class TripletAccumulator:
    """Dense K^3 running sums over cells (P9: never store per cell x per triple)."""

    def __init__(self, K):
        self.K = K
        self.sum_W = np.zeros((K, K, K))
        self.sum_EW = np.zeros((K, K, K))
        self.sum_Z = np.zeros((K, K, K))
        self.sum_Z2 = np.zeros((K, K, K))
        self.n_cells = np.zeros((K, K, K), dtype=np.int64)

    def add(self, W, EW, Var):
        Var = np.clip(Var, 1e-12, None)            # P7
        Z = (W - EW) / np.sqrt(Var)
        self.sum_W += W
        self.sum_EW += EW
        self.sum_Z += Z
        self.sum_Z2 += Z * Z
        self.n_cells += (W > 0)                     # cell contributed this motif

    def merge(self, other):
        self.sum_W += other.sum_W
        self.sum_EW += other.sum_EW
        self.sum_Z += other.sum_Z
        self.sum_Z2 += other.sum_Z2
        self.n_cells += other.n_cells

    def to_frame(self, marker_names):
        """Upper triples A<B<C with sum_W>0 -> sparse long form."""
        K = self.K
        ia, ib, ic = np.array(list(combinations(range(K), 3))).T
        w = self.sum_W[ia, ib, ic]
        keep = w > 0
        ia, ib, ic, w = ia[keep], ib[keep], ic[keep], w[keep]
        return pd.DataFrame({
            "marker_1": [marker_names[i] for i in ia],
            "marker_2": [marker_names[i] for i in ib],
            "marker_3": [marker_names[i] for i in ic],
            "sum_W": w,
            "sum_EW": self.sum_EW[ia, ib, ic],
            "sum_Z": self.sum_Z[ia, ib, ic],
            "sum_Z2": self.sum_Z2[ia, ib, ic],
            "n_cells": self.n_cells[ia, ib, ic],
        })


def _process_batch(args):
    """Worker: process a list of (component, comp_el) -> (result_df, accumulator).

    Two modes by the 4th arg `triples`:
      * triples is None  -> doublet long-form + pooled K^3 triplet accumulator (default run).
      * triples is (T,3) -> per-cell wedge table for the closed list ONLY (acc is None).
    """
    cells, marker_names, var_mode, triples = args
    K = len(marker_names)
    marker_to_idx = {m: i for i, m in enumerate(marker_names)}
    percell_mode = triples is not None
    acc = None if percell_mode else TripletAccumulator(K)
    frames = []
    for component, comp_el in cells:
        try:
            edges, labels, side, n = load_cell(comp_el, marker_to_idx)
            if edges.shape[0] == 0:
                continue
            A, L, d, twoE, N = build_cell(edges, labels, K)
            if twoE == 0:
                continue
            m = d.sum() / 2.0
            if percell_mode:
                frames.append(triplet_list_table(component, N, labels, side, d, K, triples, marker_names))
            else:
                frames.append(doublet_table(component, A, L, d, labels, side, K, marker_names))
                s1, r1, t1, s2, r2, t2 = side_strengths(d, labels, side, K)
                T = center_tensor(N, labels, K)
                W = observed_wedges(T)
                EW = triplet_mean(s1, r1, s2, r2, m)
                Var = triplet_var_lead(EW, s1, r1, t1, s2, r2, t2, m)
                acc.add(W, EW, Var)
        except Exception as e:
            print(f"  WARN component {component}: {e}", flush=True)
    df = pd.concat(frames, ignore_index=True) if frames else None
    return df, acc


def _batched(seq, n_batches):
    seq = list(seq)
    k = max(1, int(np.ceil(len(seq) / n_batches)))
    for i in range(0, len(seq), k):
        yield seq[i:i + k]


# ---------------------------------------------------------------------------
# MCMC harness (PROMPT section 6) -- used by the validation notebook, not the full run
# ---------------------------------------------------------------------------

def bipartite_chunglu_sample_edges(d, side, seed):
    """Bipartite soft Chung-Lu, O(m) (Norros-Reittu Poisson sampler -- no dense matrix).

    Cross-side edge multiplicities ~ Poisson(d_i d_j / m); equivalently draw T~Poisson(m)
    edges with side-1 endpoint ~ d1/m and side-2 endpoint ~ d2/m, then collapse multi-edges.
    Same mean (and ~same variance for sparse p) as the Bernoulli model the analytic describes;
    scales to real cells (n ~ 1e5) where a dense n1 x n2 Bernoulli matrix is infeasible."""
    rng = np.random.default_rng(int(seed))
    idx1 = np.where(side == 1)[0]
    idx2 = np.where(side == 2)[0]
    d1, d2 = d[idx1], d[idx2]
    m = d.sum() / 2.0
    T = rng.poisson(m)
    if T == 0:
        return np.empty((0, 2), np.int64)
    src = idx1[rng.choice(idx1.size, size=T, p=d1 / d1.sum())]
    dst = idx2[rng.choice(idx2.size, size=T, p=d2 / d2.sum())]
    return np.unique(np.column_stack((src, dst)), axis=0).astype(np.int64)   # collapse multi-edges


def bipartite_chunglu_enriched_sample_edges(d, side, labels, e, seed):
    """Pairwise-ENRICHED bipartite Chung-Lu, O(present_labels^2 + T).

    Cross-side edge (side-1 node i labelled X, side-2 node j labelled Y) has weight
    proportional to d_i * d_j * e[X, Y], where e is the observed marker-pair enrichment
    (J_XY / EW_pair_XY; 1 = no enrichment). This is a generative null with ONLY pairwise
    marker coupling and NO 3-way structure -- the gold-standard against which the mean-field
    connected/pairwise wedge prediction (Wpair) is validated. e=1 everywhere reduces to
    `bipartite_chunglu_sample_edges` (the degree null).

    e: (K, K) enrichment matrix (symmetric use; the doublet null is aggregated over both
    cross-side configurations, so we apply the same e to X-on-side1 / Y-on-side2).

    Total edges ~ Poisson(sum_XY s1_X s2_Y e_XY / m), so the expected X-Y cross-edge count is
    s1_X s2_Y e_XY / m = e_XY * EW_pair(X,Y): each pair carries exactly its observed enrichment
    and e=1 everywhere reduces to `bipartite_chunglu_sample_edges` (T ~ Poisson(m)). This is the
    soft (Chung-Lu) generative null the mean-field W_pair describes -- validate the pairwise
    enrichment factorisation against it, on the SOFT per-center means (the fixed- vs soft-degree
    correction is validated separately upstream).
    """
    rng = np.random.default_rng(int(seed))
    idx1 = np.where(side == 1)[0]
    idx2 = np.where(side == 2)[0]
    m = d.sum() / 2.0
    lab1, lab2 = labels[idx1], labels[idx2]
    d1, d2 = d[idx1], d[idx2]
    X_present = np.unique(lab1)
    Y_present = np.unique(lab2)
    # per-(X,Y) total weight = s1_X * s2_Y * e[X,Y]
    s1p = np.array([d1[lab1 == X].sum() for X in X_present])
    s2p = np.array([d2[lab2 == Y].sum() for Y in Y_present])
    Wm = np.outer(s1p, s2p) * np.clip(e[np.ix_(X_present, Y_present)], 0.0, None)
    Wf = Wm.ravel()
    tot = Wf.sum()
    if tot <= 0:
        return np.empty((0, 2), np.int64)
    T = rng.poisson(tot / m)                    # expected edges = sum_ij d_i d_j e / m
    if T == 0:
        return np.empty((0, 2), np.int64)
    pick = rng.choice(Wf.size, size=T, p=Wf / tot)
    xi, yi = np.unravel_index(pick, Wm.shape)
    src = np.empty(T, np.int64)
    dst = np.empty(T, np.int64)
    for k, X in enumerate(X_present):          # within chosen X (side 1) pick a node ~ degree
        mask = xi == k
        cnt = int(mask.sum())
        if cnt:
            nodes = idx1[lab1 == X]
            pr = d1[lab1 == X]
            src[mask] = nodes[rng.choice(nodes.size, size=cnt, p=pr / pr.sum())]
    for k, Y in enumerate(Y_present):          # within chosen Y (side 2) pick a node ~ degree
        mask = yi == k
        cnt = int(mask.sum())
        if cnt:
            nodes = idx2[lab2 == Y]
            pr = d2[lab2 == Y]
            dst[mask] = nodes[rng.choice(nodes.size, size=cnt, p=pr / pr.sum())]
    return np.unique(np.column_stack((src, dst)), axis=0).astype(np.int64)   # collapse multi-edges


def bipartite_edgeswap_sample(edges, side, nswap_mult=20, seed=0):
    """Bipartite degree-preserving null, O(m) vectorized (configuration model = the target
    distribution a bipartite-preserving double-edge swap converges to).

    Permute side-2 endpoints against fixed side-1 endpoints: each side's degree sequence is
    preserved exactly (up to multi-edge collapse) and every edge stays cross-side.
    `nswap_mult` kept for signature compatibility (unused)."""
    rng = np.random.default_rng(int(seed))
    e1 = np.where(side[edges[:, 0]] == 1, edges[:, 0], edges[:, 1])   # side-1 endpoints
    e2 = np.where(side[edges[:, 0]] == 1, edges[:, 1], edges[:, 0])   # side-2 endpoints
    e2p = e2[rng.permutation(e2.shape[0])]
    return np.unique(np.column_stack((e1, e2p)), axis=0).astype(np.int64)


def join_count(edges, labels, K, pair):
    """Observed join count for pair=(X,Y), X!=Y (shared statistic, P6)."""
    _, _, _, _, N = build_cell(edges, labels, K)
    X, Y = pair
    return float(N[labels == X][:, Y].sum())


def wedge_count(edges, labels, K, trip):
    """Observed wedge W_{a,b,c} for distinct labels (shared statistic, P6)."""
    _, _, _, _, N = build_cell(edges, labels, K)
    a, b, c = trip
    return float(N[labels == a][:, b] @ N[labels == a][:, c]
                 + N[labels == b][:, a] @ N[labels == b][:, c]
                 + N[labels == c][:, a] @ N[labels == c][:, b])


# ---------------------------------------------------------------------------
# Per-sample driver
# ---------------------------------------------------------------------------

def run_sample(sample, adata_path, pxl_path, output_dir, n_workers=16,
               var_mode="lead", min_umi=25000, triples=None):
    import anndata as ad
    from pixelator import read_pna

    percell_mode = triples is not None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(adata_path, backed="r")
    obs = adata.obs

    # Global marker list (identical across all cells/samples) -> stable triplet indices.
    marker_names_path = output_dir / "marker_names.json"
    if marker_names_path.exists():
        marker_names = json.loads(marker_names_path.read_text())
    else:
        marker_names = adata.var_names.tolist()
        marker_names_path.write_text(json.dumps(marker_names))
    K = len(marker_names)

    # QC'd component list for THIS sample (P10 metadata carried via the adata obs).
    sel = obs[obs["sample"] == sample]
    if "tau_type" in sel:
        sel = sel[sel["tau_type"] == "normal"]
    if "n_umi" in sel:
        sel = sel[sel["n_umi"] >= min_umi]
    components = sel.index.tolist()
    print(f"[{sample}] {len(components)} QC'd cells, K={K} markers"
          f"{f' | per-cell list of {len(triples)} triplets' if percell_mode else ''}", flush=True)
    if not components:
        print(f"[{sample}] no cells, exiting", flush=True)
        return

    print(f"[{sample}] streaming edgelists + compute (chunked)...", flush=True)
    pg = read_pna([pxl_path])
    marker_to_idx = {m: j for j, m in enumerate(marker_names)}

    def load_chunk(comps):
        """Load each component's edgelist (memory-safe per-component filter)."""
        out = []
        for comp in comps:
            comp_el = pg.filter(components=[comp]).edgelist().to_df()
            out.append((comp, comp_el[["umi1", "umi2", "marker_1", "marker_2"]]))
        return out

    # sanity print on the first cell (PROMPT section 2/3)
    first = load_chunk(components[:1])[0]
    edges, labels, side, n = load_cell(first[1], marker_to_idx)
    _, _, d, twoE, _ = build_cell(edges, labels, K)
    print(f"[{sample}] sanity cell {first[0]}: n={n} (n1={int((side==1).sum())} "
          f"n2={int((side==2).sum())}) E={edges.shape[0]} m={twoE/2:.0f} 2E={twoE:.0f} "
          f"deg[mean/med/max]={d.mean():.1f}/{np.median(d):.0f}/{d.max():.0f} "
          f"markers_present={len(set(labels.tolist()))}", flush=True)

    # Stream in chunks: load chunk I/O in main, compute in pool, merge, discard (bounds memory).
    acc = None if percell_mode else TripletAccumulator(K)
    frames = []
    chunk_size = max(n_workers * 4, 32)
    use_pool = n_workers > 1
    pool = None
    if use_pool:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=n_workers)
    t0 = time.time()
    done = 0
    for chunk in _batched(components, int(np.ceil(len(components) / chunk_size))):
        cells = load_chunk(chunk)
        if use_pool:
            work = [(b, marker_names, var_mode, triples) for b in _batched(cells, n_workers)]
            results = pool.map(_process_batch, work)
        else:
            results = [_process_batch((cells, marker_names, var_mode, triples))]
        for df, a in results:
            if df is not None:
                frames.append(df)
            if a is not None:
                acc.merge(a)
        done += len(cells)
        print(f"[{sample}]   {done}/{len(components)} cells  ({time.time()-t0:.0f}s)", flush=True)
    if pool is not None:
        pool.shutdown()
    print(f"[{sample}] computed in {time.time()-t0:.0f}s", flush=True)

    if percell_mode:
        percell = pd.concat(frames, ignore_index=True)
        percell["sample"] = sample
        ppath = output_dir / f"{sample}_triplets_percell.parquet"
        percell.to_parquet(ppath)
        print(f"[{sample}] wrote {ppath} ({len(percell):,} rows)", flush=True)
        return

    doublets = pd.concat(frames, ignore_index=True)
    doublets["sample"] = sample
    dpath = output_dir / f"{sample}_doublets.parquet"
    doublets.to_parquet(dpath)
    print(f"[{sample}] wrote {dpath} ({len(doublets):,} rows)", flush=True)

    triplets = acc.to_frame(marker_names)
    triplets["sample"] = sample
    tpath = output_dir / f"{sample}_triplets_pooled.parquet"
    triplets.to_parquet(tpath)
    print(f"[{sample}] wrote {tpath} ({len(triplets):,} triples)", flush=True)


def main():
    p = argparse.ArgumentParser(description="Chung-Lu doublets+triplets for one PNA sample")
    p.add_argument("--sample", required=True)
    p.add_argument("--adata", required=True, help="cache/adata_annotated.h5ad (QC'd cell list + metadata)")
    p.add_argument("--pxl", required=True, help="<sample>.layout.pxl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--var-mode", choices=["lead"], default="lead",
                   help="only leading-order variance (no bipartite exact form)")
    p.add_argument("--min-umi", type=int, default=25000)
    p.add_argument("--triplet-list", default=None,
                   help="JSON with {'triples': [[a,b,c],...]} of marker indices. When given, "
                        "computes PER-CELL wedge Z for that closed list only (no pooled K^3 / doublets).")
    a = p.parse_args()
    triples = None
    if a.triplet_list:
        triples = np.asarray(json.loads(Path(a.triplet_list).read_text())["triples"], dtype=np.int64)
    run_sample(a.sample, a.adata, a.pxl, a.output_dir,
               n_workers=a.n_workers, var_mode=a.var_mode, min_umi=a.min_umi, triples=triples)


if __name__ == "__main__":
    main()
