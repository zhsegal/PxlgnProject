"""Combined per-cell triplet list for the information-decomposition analysis.

Writes results/chunglu/triplet_infodecomp_list.json =
    the 171 SELECTED candidate triples (from triplet_candidate_list.json)
    UNION ~N_RANDOM distinct RANDOM triples (unbiased baseline).

The selected list is enriched for co-proximity signal, so redundancy (doublet -> triplet)
estimated on it is biased upward. The random set anchors the "added value" claim. Random
triples are drawn with marker probability proportional to mean abundance so they are
populated enough to carry a meaningful wedge z (degenerate all-zero ones are dropped later
in the notebook by nonzero rate).

Output schema (superset of build_triplet_list.py; run_sample only reads "triples"):
    {"triples": [[a,b,c], ...],          # sorted distinct global marker indices
     "sets":    ["selected"|"random", ...],   # aligned to triples
     "names":   [[A,B,C], ...],          # aligned to triples
     "marker_names": [...]}
"""
import json
from pathlib import Path

import numpy as np
import scanpy as sc

BASE = Path(__file__).resolve().parent
OUTDIR = BASE / "results" / "chunglu"
MARKERS = OUTDIR / "marker_names.json"
SELECTED = OUTDIR / "triplet_candidate_list.json"
ADATA = BASE / "cache" / "adata_annotated.h5ad"
OUT = OUTDIR / "triplet_infodecomp_list.json"

N_RANDOM = 500
SEED = 0


def main():
    marker_names = json.loads(MARKERS.read_text())
    K = len(marker_names)

    selected = json.loads(SELECTED.read_text())["triples"]
    selected = [tuple(sorted(t)) for t in selected]
    seen = set(selected)

    # marker sampling weight = mean abundance (broadly-present markers -> populated triples)
    adata = sc.read_h5ad(ADATA, backed="r")
    assert adata.var_names.tolist() == marker_names, "marker order mismatch vs marker_names.json"
    X = adata.X[:]  # (cells, K) — small (159 markers)
    w = np.asarray(X.mean(0)).ravel().astype(float)
    w = np.clip(w, 1e-9, None)
    p = w / w.sum()

    rng = np.random.default_rng(SEED)
    random_triples = []
    tries = 0
    while len(random_triples) < N_RANDOM and tries < 200 * N_RANDOM:
        tries += 1
        t = tuple(sorted(rng.choice(K, size=3, replace=False, p=p).tolist()))
        if t in seen:
            continue
        seen.add(t)
        random_triples.append(t)

    triples = list(selected) + random_triples
    sets = ["selected"] * len(selected) + ["random"] * len(random_triples)
    names = [[marker_names[i] for i in t] for t in triples]

    OUT.write_text(json.dumps({
        "triples": [list(t) for t in triples],
        "sets": sets,
        "names": names,
        "marker_names": marker_names,
    }))
    print(f"selected: {len(selected)}  random: {len(random_triples)}  total: {len(triples)}")
    print(f"wrote -> {OUT}")
    for t in random_triples[:3]:
        print("   random", list(t), [marker_names[i] for i in t])


if __name__ == "__main__":
    main()
