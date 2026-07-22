"""Information-decomposition helpers: counts -> pairs -> triplets.

How much of each co-proximity rung is a restatement of the rung below it?
  * pairs -> triplets : predict triplet_z from its three constituent pair z's (R^2).
  * counts -> pairs   : predict pair z (and raw count J) from marker abundance.

Two views of the pair->triplet redundancy:
  1. EMPIRICAL variance-explained: leave-cells-out CV R^2 (linear + rank), within/between-cell.
  2. MECHANISTIC connected decomposition: a mean-field "no-3-way-interaction" prediction of the
     wedge, Wpair = sum over the 3 wedge centers of (product of the center's two incident pair
     enrichments) * (per-center degree-null mean EW_c). Because triplet_z and connected_z share
     the same sqrt(Var) scale, triplet_z - connected_z is exactly the pairwise-explainable part.

Reuses split_triplet / display_name from nalm_utils. Heavy frames handled with polars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import polars as pl
    _HAS_POLARS = True
except Exception:  # pragma: no cover
    _HAS_POLARS = False

from nalm_utils import split_triplet, display_name  # noqa: F401  (re-exported for the notebook)

__all__ = [
    "canon_pair", "triplet_pair_keys", "needed_pair_set",
    "build_joined_table", "add_pairwise_prediction",
    "linear_r2", "grouped_cv_r2", "within_between_r2",
    "split_triplet", "display_name",
]


# ── pair keys ─────────────────────────────────────────────────────────────────

def canon_pair(a: str, b: str, sep: str = "/") -> str:
    """Unordered pair key, lexicographically canonical: canon('B','A') == 'A/B'."""
    return f"{a}{sep}{b}" if a <= b else f"{b}{sep}{a}"


def triplet_pair_keys(m1: str, m2: str, m3: str):
    """The three canonical constituent-pair keys of triplet (m1,m2,m3):
    (AB, AC, BC) with A=m1, B=m2, C=m3."""
    return canon_pair(m1, m2), canon_pair(m1, m3), canon_pair(m2, m3)


def _vec_canon(a, b) -> np.ndarray:
    """Vectorised canon_pair over two string columns (avoids per-row Python)."""
    a = np.asarray(a, dtype=object).astype(str)
    b = np.asarray(b, dtype=object).astype(str)
    lo = np.where(a <= b, a, b)
    hi = np.where(a <= b, b, a)
    return np.char.add(np.char.add(lo.astype(str), "/"), hi.astype(str))


def needed_pair_set(trp: pd.DataFrame) -> set[str]:
    """All canonical constituent pairs across the triplet frame (to pre-filter the big doublets).

    Deduplicates triplets first (the frame is per cell × triplet — millions of rows, but only
    a few hundred distinct triplets)."""
    u = trp[["marker_1", "marker_2", "marker_3"]].drop_duplicates()
    keys = set()
    for m1, m2, m3 in u.itertuples(index=False):
        keys.update(triplet_pair_keys(m1, m2, m3))
    return keys


# ── join triplet frame with its three doublet z's ────────────────────────────

def build_joined_table(trp: pd.DataFrame, dbl_small: pd.DataFrame,
                       sets: dict | None = None) -> pd.DataFrame:
    """Per (cell, triplet) frame joined with its three constituent pair statistics.

    Parameters
    ----------
    trp : per-cell triplet frame (infodecomp schema): component, marker_1/2/3, W, EW, Var,
        triplet_z, W_cA/W_cB/W_cC, EW_cA/EW_cB/EW_cC, sample.
    dbl_small : doublet frame PRE-FILTERED to the needed pairs (needed_pair_set), with columns
        component, marker_1, marker_2, join_count_z, J, EW.  (EW here = doublet null mean.)
    sets : optional {('A','B','C') sorted-name-tuple -> 'selected'|'random'} tag map.

    Returns a pandas frame with z_AB/z_AC/z_BC, pair enrichments e_AB/e_AC/e_BC (J/EW, clipped),
    and passthrough triplet columns.  Uses polars for the (component,pair) joins when available.
    """
    d = dbl_small.copy()
    d["pair"] = _vec_canon(d["marker_1"], d["marker_2"])
    d["e"] = d["J"].to_numpy() / np.clip(d["EW"].to_numpy(), 1e-9, None)
    d = d[["component", "pair", "join_count_z", "e"]]

    t = trp.copy()
    t["pAB"] = _vec_canon(t["marker_1"], t["marker_2"])
    t["pAC"] = _vec_canon(t["marker_1"], t["marker_3"])
    t["pBC"] = _vec_canon(t["marker_2"], t["marker_3"])

    if _HAS_POLARS:
        dl = pl.from_pandas(d)
        tl = pl.from_pandas(t)
        for tag, pcol in [("AB", "pAB"), ("AC", "pAC"), ("BC", "pBC")]:
            j = dl.rename({"pair": pcol, "join_count_z": f"z_{tag}", "e": f"e_{tag}"})
            tl = tl.join(j, on=["component", pcol], how="left")
        t = tl.to_pandas()
    else:  # pandas fallback
        for tag, pcol in [("AB", "pAB"), ("AC", "pAC"), ("BC", "pBC")]:
            j = d.rename(columns={"pair": pcol, "join_count_z": f"z_{tag}", "e": f"e_{tag}"})
            t = t.merge(j, on=["component", pcol], how="left")

    # missing pair (J==0 in that cell -> absent from doublets): z=0, enrichment=0
    for tag in ("AB", "AC", "BC"):
        t[f"z_{tag}"] = t[f"z_{tag}"].fillna(0.0)
        t[f"e_{tag}"] = t[f"e_{tag}"].fillna(0.0)

    if sets is not None:
        key = list(zip(t["marker_1"], t["marker_2"], t["marker_3"]))
        t["set"] = [sets.get(tuple(sorted(k)), "unknown") for k in key]
    return t


def add_pairwise_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Add the mean-field pairwise wedge Wpair, connected_z, and doublet_explained_fraction.

    Wpair = e_AB e_AC EW_cA + e_AB e_BC EW_cB + e_AC e_BC EW_cC
        (each wedge center scaled by the enrichments of its two incident pairs; e=1 -> EW).
    connected_z = (W - Wpair) / sqrt(Var)           (irreducible 3-way part)
    doublet_explained_fraction = (Wpair - EW) / (W - EW)   (share of the excess the pairs explain)
    """
    df = df.copy()
    sd = np.sqrt(np.clip(df["Var"].to_numpy(), 1e-12, None))
    Wpair = (df["e_AB"] * df["e_AC"] * df["EW_cA"]
             + df["e_AB"] * df["e_BC"] * df["EW_cB"]
             + df["e_AC"] * df["e_BC"] * df["EW_cC"]).to_numpy()
    df["W_pair"] = Wpair
    df["connected_z"] = (df["W"].to_numpy() - Wpair) / sd
    excess = df["W"].to_numpy() - df["EW"].to_numpy()
    frac = (Wpair - df["EW"].to_numpy()) / np.where(np.abs(excess) < 1e-9, np.nan, excess)
    df["doublet_explained_fraction"] = frac
    return df


# ── variance explained ───────────────────────────────────────────────────────

def linear_r2(y: np.ndarray, X: np.ndarray) -> float:
    """In-sample OLS R^2 (with intercept). X shape (n, p)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan


def _spearman_r2(pred: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(pred, y)[0]   # [0] works across scipy versions (namedtuple / .statistic)
    return float(r ** 2) if np.isfinite(r) else np.nan


def grouped_cv_r2(df: pd.DataFrame, y_col: str, x_cols: list[str],
                  group_col: str = "component", n_splits: int = 5,
                  max_n: int | None = 400_000, seed: int = 0) -> dict:
    """Leave-cells-out (GroupKFold) cross-validated R^2 of y ~ X.

    Blocks on `group_col` (whole cells held out) so within-cell correlation cannot leak.
    Returns pooled out-of-fold Pearson-R^2 and Spearman-R^2, plus the in-sample linear R^2.
    Optionally grouped-subsamples to max_n rows first (keeps whole cells) for speed.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LinearRegression

    d = df[[y_col, group_col] + x_cols].dropna()
    if max_n is not None and len(d) > max_n:
        rng = np.random.default_rng(seed)
        groups = d[group_col].unique()
        rng.shuffle(groups)
        keep, cum = [], 0
        sizes = d.groupby(group_col).size()
        for g in groups:
            keep.append(g)
            cum += int(sizes[g])
            if cum >= max_n:
                break
        d = d[d[group_col].isin(keep)]

    y = d[y_col].to_numpy(float)
    X = d[x_cols].to_numpy(float)
    g = d[group_col].to_numpy()

    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=min(n_splits, d[group_col].nunique()))
    for tr, te in gkf.split(X, y, groups=g):
        m = LinearRegression().fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    ss_tot = ((y - y.mean()) ** 2).sum()
    cv_r2 = float(1 - ((y - oof) ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan
    return {
        "n": int(len(y)), "n_cells": int(d[group_col].nunique()),
        "cv_r2": cv_r2,
        "cv_spearman_r2": _spearman_r2(oof, y),
        "insample_r2": linear_r2(y, X),
    }


def within_between_r2(df: pd.DataFrame, y_col: str, x_cols: list[str],
                      group_col: str = "component") -> dict:
    """Within-cell (per-cell demeaned) vs between-cell (cell means) linear R^2.

    Separates 'busy cell' redundancy (a globally high-coloc cell lifts every motif) from genuine
    motif-level redundancy. between_r2 uses one row per cell (group means); within_r2 regresses
    the demeaned y on demeaned X (the fixed-effects / within estimator).
    """
    d = df[[y_col, group_col] + x_cols].dropna()
    gm = d.groupby(group_col)
    y_bar = gm[y_col].transform("mean")
    within = pd.DataFrame({c: d[c] - gm[c].transform("mean") for c in x_cols})
    within_y = d[y_col] - y_bar
    within_r2 = linear_r2(within_y.to_numpy(), within.to_numpy())

    cell_means = gm[[y_col] + x_cols].mean()
    between_r2 = linear_r2(cell_means[y_col].to_numpy(), cell_means[x_cols].to_numpy())
    return {"within_r2": within_r2, "between_r2": between_r2,
            "pooled_r2": linear_r2(d[y_col].to_numpy(), d[x_cols].to_numpy()),
            "n_cells": int(d[group_col].nunique())}
