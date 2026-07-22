"""
Per-cell community detection on marker colocalization graphs.

Runs 3 algorithms on each cell's proximity-derived marker x marker colocalization matrix:
  1. Leiden CPM (Constant Potts Model)
  2. Signed Spectral Clustering
  3. Leiden Modularity (RBConfigurationVertexPartition)

Usage:
    python run_percell_communities.py --proximity-parquet <path> --output-dir <path> \
        --resolution 0.1 --k-spectral 5 --n-workers 16

Or submit via LSF:
    bsub < run_percell_communities.lsf
"""

import argparse
import time
import numpy as np
import pandas as pd
import igraph as ig
import leidenalg
from scipy import linalg as la
from sklearn.cluster import KMeans
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ISOTYPE_CONTROLS = {"mIgG1", "mIgG2a", "mIgG2b"}


def detect_communities_single_cell(args):
    """Run 3 community detection algorithms on a single cell's colocalization data."""
    component, cell_df, gamma, marker_list, k_spectral = args
    try:
        n = len(marker_list)
        marker_idx = {m: i for i, m in enumerate(marker_list)}
        mat = np.zeros((n, n))

        for _, row in cell_df.iterrows():
            m1, m2 = row["marker_1"], row["marker_2"]
            if m1 in marker_idx and m2 in marker_idx:
                i, j = marker_idx[m1], marker_idx[m2]
                mat[i, j] = row["join_count_z"]
                mat[j, i] = row["join_count_z"]

        # Build graph (shared by Leiden algorithms)
        G = ig.Graph.Full(n, loops=False)
        G.vs["name"] = marker_list
        weights = [mat[e.source, e.target] for e in G.es]
        G.es["weight"] = weights

        # --- Algorithm 1: Leiden CPM ---
        partition_cpm = leidenalg.find_partition(
            G, leidenalg.CPMVertexPartition,
            weights="weight", resolution_parameter=gamma, seed=42
        )
        leiden_cpm_labels = {marker_list[i]: partition_cpm.membership[i] for i in range(n)}

        # --- Algorithm 2: Signed Spectral Clustering ---
        A = mat.copy()
        np.fill_diagonal(A, 0)
        A_pos = np.maximum(A, 0)
        A_neg = np.abs(np.minimum(A, 0))
        D_pos = np.diag(A_pos.sum(axis=1))
        D_neg = np.diag(A_neg.sum(axis=1))
        L_signed = D_pos + D_neg - (A_pos - A_neg)

        eigenvalues, eigenvectors = la.eigh(L_signed)
        X = eigenvectors[:, :k_spectral]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X = X / norms

        kmeans = KMeans(n_clusters=k_spectral, n_init=10, random_state=42)
        spec_labels_arr = kmeans.fit_predict(X)
        spectral_labels = {marker_list[i]: int(spec_labels_arr[i]) for i in range(n)}

        # --- Algorithm 3: Leiden Modularity (RBConfiguration) ---
        # RBConfiguration doesn't accept negative weights, so build a positive-only graph
        G_pos = ig.Graph.Full(n, loops=False)
        G_pos.vs["name"] = marker_list
        pos_weights = [max(mat[e.source, e.target], 0) for e in G_pos.es]
        G_pos.es["weight"] = pos_weights

        partition_mod = leidenalg.find_partition(
            G_pos, leidenalg.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=gamma, seed=42
        )
        leiden_mod_labels = {marker_list[i]: partition_mod.membership[i] for i in range(n)}

        return component, leiden_cpm_labels, spectral_labels, leiden_mod_labels
    except Exception:
        return component, None, None, None


def main():
    parser = argparse.ArgumentParser(description="Per-cell community detection (3 algorithms)")
    parser.add_argument("--proximity-parquet", type=Path, required=True,
                        help="Path to proximity parquet file")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for parquet files")
    parser.add_argument("--resolution", type=float, default=0.1,
                        help="Resolution parameter for Leiden algorithms")
    parser.add_argument("--k-spectral", type=int, default=5,
                        help="Number of clusters for spectral clustering")
    parser.add_argument("--n-workers", type=int, default=16,
                        help="Number of parallel workers")
    args = parser.parse_args()

    print(f"Loading proximity data from {args.proximity_parquet}...")
    proximity = pd.read_parquet(args.proximity_parquet)

    # Filter
    prox = proximity[
        ~proximity["marker_1"].isin(ISOTYPE_CONTROLS) &
        ~proximity["marker_2"].isin(ISOTYPE_CONTROLS) &
        (proximity["marker_1"] != proximity["marker_2"])
    ]

    all_markers = sorted(set(prox["marker_1"].unique()) | set(prox["marker_2"].unique()))
    print(f"Markers: {len(all_markers)}, Components: {prox.component.nunique()}")

    # Group by component
    print("Grouping by component...")
    grouped = dict(list(prox.groupby("component")[["marker_1", "marker_2", "join_count_z"]]))

    components = list(grouped.keys())
    args_list = [(comp, grouped[comp], args.resolution, all_markers, args.k_spectral)
                 for comp in components]

    print(f"Running 3 algorithms on {len(components):,} cells "
          f"with {args.n_workers} workers...")
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        results = list(executor.map(detect_communities_single_cell, args_list, chunksize=100))

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed/len(components)*1000:.1f}ms per cell)")

    # Split results
    leiden_cpm_results = {comp: lab for comp, lab, _, _ in results if lab is not None}
    spectral_results = {comp: lab for comp, _, lab, _ in results if lab is not None}
    leiden_mod_results = {comp: lab for comp, _, _, lab in results if lab is not None}
    print(f"Successful: {len(leiden_cpm_results):,}/{len(components):,}")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("percell_leiden_cpm", leiden_cpm_results),
                       ("percell_spectral", spectral_results),
                       ("percell_leiden_modularity", leiden_mod_results)]:
        df = pd.DataFrame.from_dict(data, orient="index")
        df.index.name = "component"
        path = args.output_dir / f"{name}.parquet"
        df.to_parquet(path)
        print(f"{name} saved to {path}")


if __name__ == "__main__":
    main()
