# Prompt — Run the Chung–Lu degree-corrected colocalization (doublets + triplets) on the real PNA dataset

*Hand this whole file to Claude. It is a self-contained runbook. The math is exact and must be
followed literally; the data-loading is "inspect first, then conform to a canonical form" so it
adapts to the actual PXL/AnnData API without guessing.*

---

## 0. Goal & scope

For the real PixelGen **PNA** dataset, compute, **per cell**, the degree-corrected
colocalization scores under the **Chung–Lu** null — **analytically, no MCMC** — for

* **doublets** (marker pairs): the join count `J_AB`;
* **triplets** (unordered, hub-agnostic 3-marker wedges): `W_{A,B,C}` with
  `A–B–C ≡ A–C–B ≡ C–B–A ≡ …`.

Then **validate quality** by running an explicit **MCMC** null on a *few* selected (cell, motif)
pairs and comparing its empirical mean/variance to the analytic formulas.

The method, derivations, and prior validations are in the companion files
`PNA_chunglu_pairs_triplets_summary.pdf`, `PNA_degree_corrected_null_models.pdf`,
`PNA_triplet_analytic_moments.pdf`, and the reference notebook `triplet_wedge_validation.ipynb`.
**Use those exact formulas (reproduced in the Appendix here). Do not improvise alternative
estimators.**

Work in the project folder. Produce a script or notebook plus a short results report.
Use a `TaskCreate` task list and a final verification step.

---

## 1. The model in one paragraph (so you implement the right thing)

A cell is a graph: **nodes = protein molecules**, each carrying exactly **one marker label**;
**edges = spatial proximity**. Let `d_i` be the node degree and `2E = Σ_i d_i`. The Chung–Lu null
fixes every node's label and degree and randomizes the wiring; in its soft form each pair `{i,j}`
is an independent edge with probability `p_ij = d_i d_j / 2E`. Under this null every motif count's
mean and variance are closed-form in four **per-marker degree-moment strengths**
`s_X=Σ_{v∈X} d_v`, `r_X=Σ d_v²`, `t_X=Σ d_v³`, `u_X=Σ d_v⁴`. The governing rule: *a node that
anchors k of a motif's edges contributes a factor `d^k`* (so a join-count endpoint → `s`; a wedge
center → `r`, a wedge spoke → `s`). Triangles are **out of scope** (Chung–Lu has ~no triangles);
the triplet object is the **open wedge (2-path)** only.

---

## 2. Step 0 — Inspect the data, then conform to a canonical form (do this first)

Do **not** assume the PXL/AnnData API. Inspect, then produce the canonical per-cell objects below.

1. Locate the dataset (PXL file(s) and the AnnData used elsewhere in the project). The **graph
   (edge list) lives in the PXL**, not in AnnData. Identify the loader (e.g. `pixelator`/pixelgen
   `read(...)` → a dataset exposing per-component edge lists; print the object, its attributes, and
   one component's edge list to learn the schema).
2. For **one** cell, determine and **print**: how to get (a) the **edge list** (two node-id
   columns) and (b) each **node's marker label**. In PNA each node is one molecule with one marker;
   confirm this (a node must map to exactly one marker — verify there is no many-markers-per-node
   ambiguity). If edges carry **weights/multiplicity**, note it (see Pitfall P4).
3. Apply the project's standard **cell QC filter before analysis**:
   `tau_type == "normal"` **AND** `n_umi ≥ 25000`. Carry `cell_system`, `condition`, `time`,
   `target` for later aggregation. (These conventions are from the project notes; confirm the
   column names exist.)

**Canonical per-cell object** (the rest of the runbook depends only on this — produce a function
`load_cell(cell_id) -> (edges, labels, marker_names)`):

* `edges`: integer array shape `(E, 2)`, nodes relabeled `0..n-1` **contiguously per cell**,
  **undirected, de-duplicated, no self-loops** (drop `u==v`; drop duplicate `{u,v}`).
* `labels`: integer array shape `(n,)`, `labels[i] ∈ {0..K-1}` = marker index of node `i`.
* `marker_names`: list length `K` mapping index → marker string (global, identical across cells).

Sanity prints for the first cell: `n`, `E`, `2E=Σd`, degree distribution summary, number of
markers present, and that `labels` has no `-1`/NaN.

---

## 3. Step 1 — Build the per-cell matrices (shared by everything)

Implement once and reuse for observed counts, analytic moments, and MCMC. **Exact code:**

```python
import numpy as np
import scipy.sparse as sp

def build_cell(edges, labels, K):
    n = labels.shape[0]
    u, v = edges[:, 0], edges[:, 1]
    # symmetric, simple adjacency (CSR), entries = 1.0
    data = np.ones(2 * len(u))
    A = sp.csr_matrix((data, (np.r_[u, v], np.r_[v, u])), shape=(n, n))
    A.data[:] = 1.0                      # collapse any accidental multiedges to 1
    A.setdiag(0); A.eliminate_zeros()    # no self-loops
    d = np.asarray(A.sum(1)).ravel()     # degree vector (float)
    twoE = d.sum()                       # = 2 * number_of_edges
    # one-hot label matrix L (n x K), exactly one 1 per row
    L = sp.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, K))
    N = (A @ L).toarray()                # N[i, Y] = # of Y-labelled neighbours of node i
    return A, L, d, twoE, N
```

Notes that prevent mistakes:
* `2E = Σ d_i`, **not** the edge count. Everywhere "`2E`" below means `twoE`.
* `N = A @ L` is the one reusable object: it gives observed doublets *and* triplets.
* Use `float64`. Guard against `twoE == 0` (skip degenerate cells).

---

## 4. Step 2 — Doublets (marker pairs): observed, analytic null, score

### 4.1 Observed join counts (all pairs at once)
```python
J = (L.T @ (A @ L)).toarray()          # K x K ; J = L^T A L
# Off-diagonal J[X,Y] (X != Y) IS the observed join count J_XY (each X-Y edge once).
# Diagonal J[X,X] counts within-X edges twice -> observed J_XX = J[X,X] / 2.
```

### 4.2 Analytic moment-strength vectors
```python
s = np.bincount(labels, weights=d,       minlength=K)   # Σ d
r = np.bincount(labels, weights=d**2,    minlength=K)   # Σ d^2
t = np.bincount(labels, weights=d**3,    minlength=K)   # Σ d^3
u = np.bincount(labels, weights=d**4,    minlength=K)   # Σ d^4
```

### 4.3 Analytic null mean & variance (EXACT for the pair — independent edges)
For markers `A ≠ B`:
```
E[J_AB]   = s_A s_B / (2E)
Var[J_AB] = s_A s_B / (2E)  -  r_A r_B / (2E)^2
```
Vectorized over all pairs:
```python
mean_pair = np.outer(s, s) / twoE
var_pair  = np.outer(s, s) / twoE - np.outer(r, r) / twoE**2
# self-pairs (A==B), if you report them:
np.fill_diagonal(mean_pair, (s*s - r) / (4*twoE))
# (variance of self-pairs needs its own derivation; exclude diagonal unless required)
```

### 4.4 Score
```python
Z_pair = (J - mean_pair) / np.sqrt(np.clip(var_pair, 1e-12, None))
log2_ratio_pair = np.log2(np.clip(J, 1e-9, None) / np.clip(mean_pair, 1e-9, None))
```
For pairs with small `mean_pair` (rule of thumb `< 10`), the normal `Z` over-states
significance — also compute a **Poisson upper-tail** p-value
`scipy.stats.poisson.sf(J-1, mean_pair)` and prefer it there. Sign: `Z>0` co-proximity,
`Z<0` exclusion.

---

## 5. Step 3 — Triplets (unordered hub-agnostic wedge): observed, analytic null, score

### 5.1 Observed wedge counts
A wedge = a 2-path (center + two neighbours). `W_{A,B,C}` counts 2-paths whose three node-labels
are `{A,B,C}` (any order), each counted once at its center. Build the **per-center co-neighbour
tensor** then symmetrize:
```python
groups = [np.where(labels == g)[0] for g in range(K)]
def center_tensor(N, groups, K):
    T = np.zeros((K, K, K))
    for g in range(K):
        idx = groups[g]
        if idx.size:
            Ng = N[idx]                 # (n_g, K)
            T[g] = Ng.T @ Ng            # T[g,X,Y] = Σ_{h: label g} N[h,X] N[h,Y]
    return T
T = center_tensor(N, groups, K)
# observed wedge for distinct A,B,C (center can be any of the three):
#   W_obs[A,B,C] = T[A,B,C] + T[B,A,C] + T[C,A,B]
```
Correctness: for distinct `A,B,C`, `N[h,B]*N[h,C]` counts ordered (B-neighbour, C-neighbour) pairs
which are automatically distinct nodes and distinct from the center — each wedge counted **once**.
**Only use distinct triples `A<B<C`**; the diagonal entries of `T[g]` (equal labels) belong to
duplicate-label motifs (`{A,A,C}`) — handle separately or skip.

### 5.2 Analytic null mean (EXACT) and variance (leading + exact)
Use the SAME `s,r,t,u`. Helper `_o(x,y,z)=np.einsum('a,b,c->abc',x,y,z,optimize=True)`.

**Mean (exact):**
```
E[W_{ABC}] = ( r_A s_B s_C + r_B s_A s_C + r_C s_A s_B ) / (2E)^2
```
```python
def _o(x,y,z): return np.einsum('a,b,c->abc', x, y, z, optimize=True)
EW = (_o(r,s,s) + _o(s,r,s) + _o(s,s,r)) / twoE**2
```

**Variance — leading order (scalable default; conservative):**
```
Var ≈ E[W] + (1/(2E)^3) [ s_A^2 (t_B s_C + 2 r_B r_C + s_B t_C)
                        + s_B^2 (t_A s_C + 2 r_A r_C + s_A t_C)
                        + s_C^2 (t_A s_B + 2 r_A r_B + s_A t_B) ]
```
```python
s2 = s*s
B = ( _o(s2,t,s) + 2*_o(s2,r,r) + _o(s2,s,t)
    + _o(t,s2,s) + 2*_o(r,s2,r) + _o(s,s2,t)
    + _o(t,s,s2) + 2*_o(r,r,s2) + _o(s,t,s2) )
Var_lead = EW + B / twoE**3
```

**Variance — exact (independent-edge; use for heavy-tailed cells / hub-heavy triples):**
adds the `(1−p)` corrections via the 4th moment `u`:
```python
D  = EW - (_o(u,r,r) + _o(r,u,r) + _o(r,r,u)) / twoE**4
Cc = ( (_o(t,s,s2) + 2*_o(r,r,s2) + _o(s,t,s2)) / twoE**3
     - (_o(u,r,s2) + 2*_o(t,t,s2) + _o(r,u,s2)) / twoE**4
     - (_o(t,s,r)  + _o(s,t,r))                 / twoE**3
     + (_o(u,r,r)  + _o(r,u,r))                 / twoE**4 )
C = Cc + np.transpose(Cc,(2,0,1)) + np.transpose(Cc,(1,2,0))
Var_exact = D + C
```

### 5.3 Score
`Z = (W_obs − EW)/sqrt(Var)`, `log2_ratio = log2(W_obs/EW)`; Poisson upper tail when `EW` small.
Use `Var_exact` for the quality check and for heavy-tailed cells; `Var_lead` is fine elsewhere
(it only over-states the spread, i.e. conservative).

### 5.4 Memory/scale for the FULL run (important)
`K≈159` ⇒ a per-cell `K×K×K` tensor is `159³·8B ≈ 32 MB` (fine per cell) but `C(159,3)=657,359`
triplets × thousands of cells is **too large to store per-cell-per-triplet**. Therefore:
* **Doublets:** store the per-cell `K×K` `Z`/`log2_ratio`/`J`/`EW` (cheap).
* **Triplets:** choose ONE of —
  (a) restrict to a **pre-registered marker panel / hypothesis set** of triples (recommended for
  biology); or
  (b) compute per-cell tensors and **stream-aggregate immediately** (never store per cell): e.g.
  accumulate, per triple, `Σ_cells W_obs` and `Σ_cells EW` for a pooled test, and/or per-condition
  running stats of `Z` — then discard the cell's tensor. Provide the aggregation as a reduction
  inside the per-cell loop.
* Iterate cells one at a time; reuse buffers; consider `float32` for `N` if memory-bound.

---

## 6. Step 4 — MCMC quality check on a few (cell, motif) pairs

Goal: confirm the analytic mean/variance match an explicit null simulation. Pick **3 cells**
(small, medium, large by `E`) and per cell **~6 doublets and ~6 triplets** spanning the range
(include some with **high-degree/hub** markers and some without; include some with large and some
with small analytic `EW`). Use **one shared statistic function** for observed, analytic-target, and
MCMC so they are guaranteed identical.

Run **two** nulls (they check different things):

### 6.1 Chung–Lu independent-edge MC — *checks the formulas*
Resamples from the exact model the analytic formulas describe ⇒ should match **mean & variance to
Monte-Carlo error**.
```python
import networkx as nx
def chunglu_sample_edges(d, seed):           # d = real degree vector used as weights
    G = nx.expected_degree_graph(d, selfloops=False, seed=int(seed))
    e = np.fromiter((x for ed in G.edges() for x in ed), np.int64)
    return e.reshape(-1,2)
```
Check `max(d)**2 < 2E` (else `p_ij` would exceed 1 and the sampler clips, mildly biasing the
comparison — note it if it happens).

### 6.2 Degree-preserving edge-swap MC — *checks modelling adequacy on the REAL graph*
The principled hard-configuration null: keep the **exact** degrees of the real graph, randomize
wiring by double-edge swaps. Expect mean ≈ analytic and variance **close** (soft Chung–Lu is
slightly higher; for wedges, edge-swap also destroys clustering so it matches the wedge null).
Use **igraph** for speed (C-level swaps); networkx `connected_double_edge_swap` is correct but slow.
```python
import igraph as ig
def edgeswap_sample(edges, n, nswap_mult=10, seed=0):
    g = ig.Graph(n=n, edges=[tuple(e) for e in edges], directed=False)
    g.rewire(n=nswap_mult * g.ecount(), mode="simple")   # degree-preserving, no self/multi-edges
    return np.array(g.get_edgelist(), dtype=np.int64)
```
Use `nswap_mult` in `10..50`; restart from the real graph for each sample (fresh `g`).
For very large cells, pick a smaller cell for this check or reduce `nswap_mult`/samples.

### 6.3 The harness
```python
def wedge_count(edges, labels, K, trip):     # trip=(A,B,C), distinct labels
    # build N for THIS sampled graph, then the SAME observed formula (center can be A, B, or C):
    A,L,d,twoE,N = build_cell(edges, labels, K)
    a, b, c = trip
    return (N[labels==a][:, b] @ N[labels==a][:, c]    # center a, spokes b,c
          + N[labels==b][:, a] @ N[labels==b][:, c]    # center b, spokes a,c
          + N[labels==c][:, a] @ N[labels==c][:, b])   # center c, spokes a,b  =  W_{a,b,c}
def join_count(edges, labels, K, pair):
    A,L,d,twoE,N = build_cell(edges, labels, K)
    X,Y = pair
    return float((N[labels==X][:,Y]).sum())          # = # X-Y edges (X!=Y)
```
For each selected (cell, motif, null): run `NS≈500–1000` samples, collect the statistic, compute
`mc_mean, mc_sd`. Tabulate against analytic `EW`/`sqrt(Var)`:

| cell | motif | analytic E | MC mean | analytic SD (exact) | MC SD | E ratio | SD ratio |

Also overlay: the **Poisson/diagonal** SD `sqrt(EW)` to show it underestimates (covariance matters),
and a small histogram of the MC null with the observed value marked.

### 6.4 Acceptance criteria
* **Mean:** analytic/MC within **±3%** for both nulls (mean is exact).
* **Variance (Chung–Lu MC):** analytic-exact/MC within **±5–10%** (the rest is MC noise at
  `NS≈500`); leading-order may run a few % high, and noticeably high (up to ~1.5–1.7×) **only** on
  hub-dominated triples — that is expected and conservative.
* **Variance (edge-swap MC):** analytic close; document any systematic gap (soft vs hard).
* **Z calibration (optional, strong):** for a real cell, compute `Z` for many well-populated
  motifs; `Z` should be ≈ `N(0,1)` in spread (`σ≈1`), confirming the moments.
Report ratios; if mean is off by >3%, **stop and debug** (almost always a `2E` vs edge-count bug,
a directed/duplicated-edge bug, or a label-indexing bug).

---

## 7. Pitfalls checklist (read before coding)

* **P1 `2E`** is `Σ d_i` (= 2× edges), used in *every* denominator. The single most common bug.
* **P2 Undirected, simple graph.** De-duplicate `{u,v}`, drop self-loops, symmetric `A` with unit
  entries. A directed or double-counted edge list silently doubles counts.
* **P3 One label per node.** `L` has exactly one `1` per row. Verify no node maps to multiple
  markers (PNA: a node is one molecule = one marker).
* **P4 Edge weights.** If the PXL edges carry multiplicity/weights, the method as written uses the
  **unweighted simple graph** (degree). Default to that. If you instead want to preserve *strength*
  (Σ weights), that is a different null — do **not** mix; decide explicitly and state it.
* **P5 Triplets are the WEDGE (open 2-path), unordered.** Use only `A<B<C`; never feed triangle
  counts to these formulas (Chung–Lu has ~no triangles — triangles stay on the label-shuffle).
* **P6 Same statistic everywhere.** Observed, analytic target, and MCMC must compute the identical
  count; share one function.
* **P7 Variance choice.** `Var_lead` is conservative; use `Var_exact` for the validation and for
  heavy-tailed cells. Clip variance to `>0` before `sqrt`.
* **P8 Rare motifs.** When `EW` is small, use the Poisson tail, not the normal `Z`.
* **P9 Don't store per-cell × per-triplet** for all triplets (memory). Restrict or stream-aggregate
  (§5.4).
* **P10 Aggregation.** Colocalization is per-cell. Aggregate to conditions with the project's
  nonparametrics (Mann–Whitney / Kruskal–Wallis), **blocking on `cell_system`**; control FDR with
  Benjamini–Hochberg (or Benjamini–Yekutieli given overlapping triplets share markers). Respect the
  48 h small-`n` collapse — don't over-read where cells are few.
* **P11 expected_degree_graph clipping.** If `max(d)² ≥ 2E`, `p_ij` would exceed 1; note it (rare in
  large cells) as it slightly biases the Chung–Lu MC comparison, not the analytic.

---

## 8. Deliverables

1. A module/notebook with: `load_cell`, `build_cell`, the doublet block (§4), the triplet block
   (§5), the MCMC harness (§6), and the full-dataset driver with streaming aggregation (§5.4).
2. A **validation report** (table + a few plots) for the selected (cell, motif) pairs vs both MCMC
   nulls, against the §6.4 criteria.
3. The **results**: per-cell doublet `Z`/`log2_ratio` tables, and the chosen triplet output
   (panel-restricted or aggregated), plus per-condition comparisons (blocked on `cell_system`,
   FDR-controlled).
4. A short methods note stating exactly which graph (unweighted/degree), which variance
   (lead/exact), filters applied, and the validation outcome.

Begin with Step 0 (inspect + canonical form) and **print intermediate sanity checks** at each step.
Do not proceed past the MCMC quality check until the mean criterion (§6.4) passes.

---

## Appendix — all formulas in one place

Degree moments per marker `X`: `s_X=Σ_{v∈X}d_v`, `r_X=Σ d_v²`, `t_X=Σ d_v³`, `u_X=Σ d_v⁴`;
`2E=Σ_i d_i`.

**Doublet (pair), exact:**
```
E[J_AB]   = s_A s_B / 2E
Var[J_AB] = s_A s_B / 2E − r_A r_B / (2E)^2          (A≠B)
E[J_AA]   = (s_A^2 − r_A) / (4E)                      (self-pair)
```

**Triplet (unordered wedge):**
```
E[W_{ABC}]      = (r_A s_B s_C + r_B s_A s_C + r_C s_A s_B) / (2E)^2          (exact)

Var_lead        = E[W] + (1/(2E)^3) [ s_A^2(t_B s_C + 2 r_B r_C + s_B t_C)
                                     + s_B^2(t_A s_C + 2 r_A r_C + s_A t_C)
                                     + s_C^2(t_A s_B + 2 r_A r_B + s_A t_B) ]

Var_exact       = Var_lead  −  (O(p) corrections via u):
                  diagonal:  − (u_A r_B r_C + u_B r_A r_C + u_C r_A r_B)/(2E)^4
                  + matching (1−p) covariance terms  (code in §5.2)
```

**Scores:** `Z=(obs−E)/sqrt(Var)`, `log2_ratio=log2(obs/E)`; Poisson upper tail
`P(Pois(E) ≥ obs)` when `E` is small. `Z>0` co-proximity, `Z<0` exclusion.

**Why pair is exact but triplet needs care:** `J_AB` is a sum of *independent* edge indicators
(closed-form variance, no approximation). `W` is a sum of *products* of edge indicators (wedges),
which are correlated whenever they **share an edge**; that covariance dominates the variance — a
Poisson/independent baseline `Var≈E[W]` underestimates the SD ~2× — and `Var_lead`/`Var_exact`
capture it. Validated against MCMC in `triplet_wedge_validation.ipynb`.
