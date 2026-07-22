# Claude Code task — make the PNA colocalization null **bipartite**

## Why
The PNA cell graph is **bipartite**: every node is on the **umi1** or **umi2** side, and edges
form **only across sides**. Our current Chung–Lu null treats it as a general graph
(`p_ij = d_i d_j / 2E` for any pair), which over-predicts wedge counts for side-asymmetric markers.
Fix: re-derive the null per side. Math is in `PNA_bipartite_null_implementation.pdf`; the reference
validation is `bipartite_check.py`. **Observed-count code does not change — only the null formulas
and the strengths do.**

## The 5 changes

**1. Tag each node's side.** From the edgelist's two endpoint columns (umi1, umi2): all umi1 nodes →
`side=1`, all umi2 nodes → `side=2`. Keep node ids distinct between sides. Add a per-cell
`side` array (`shape (n,)`, values in `{1,2}`). Let `m = number of edges` (so `2E = 2m`, and each
side's degrees sum to `m`).

**2. Per-side strength vectors** (replace the single `s,r,t`):
```python
def side_strengths(d, labels, side, K):
    m1 = (side == 1); m2 = (side == 2)
    def moms(mask):
        return (np.bincount(labels[mask], weights=d[mask],     minlength=K),
                np.bincount(labels[mask], weights=d[mask]**2,  minlength=K),
                np.bincount(labels[mask], weights=d[mask]**3,  minlength=K))
    s1, r1, t1 = moms(m1)
    s2, r2, t2 = moms(m2)
    return s1, r1, t1, s2, r2, t2     # each length K
m = d.sum() / 2.0                     # edge count
```

**3. Replace the DOUBLET null** (exact):
```python
# E[J_AB]   = (s1_A s2_B + s2_A s1_B) / m
# Var[J_AB] =  E[J_AB] - (r1_A r2_B + r2_A r1_B) / m^2
mean_pair = (np.outer(s1, s2) + np.outer(s2, s1)) / m
var_pair  = mean_pair - (np.outer(r1, r2) + np.outer(r2, r1)) / m**2
```
(Self-pair `A=A`: `E = s1_A * s2_A / m`.)

**4. Replace the TRIPLET (wedge) null.** Hub-agnostic, unordered; center carries `r` on its side,
spokes carry `s` from the **opposite** side. With `_o(x,y,z)=np.einsum('a,b,c->abc',x,y,z,optimize=True)`:
```python
# --- mean (exact): E[W] = (1/m^2) * sum over center X of ( r1_X s2_Y s2_Z + r2_X s1_Y s1_Z ) ---
EW = ( _o(r1, s2, s2) + _o(s2, r1, s2) + _o(s2, s2, r1)        # center on side 1, spokes side 2
     + _o(r2, s1, s1) + _o(s1, r2, s1) + _o(s1, s1, r2) ) / m**2   # center on side 2, spokes side 1

# --- variance (leading order) ---
# per far-label f (others g,h):  (s2_f)^2 (t1_g s2_h + t1_h s2_g)
#                              + 2 s1_f s2_f (r1_g r2_h + r1_h r2_g)
#                              +  (s1_f)^2 (s1_g t2_h + s1_h t2_g)
s1sq, s2sq, s1s2 = s1*s1, s2*s2, s1*s2
Cc = ( _o(t1, s2, s2sq) + _o(s2, t1, s2sq)        # far on last axis (=c); g,h on a,b
     + 2*_o(r1, r2, s1s2) + 2*_o(r2, r1, s1s2)
     + _o(s1, t2, s1sq) + _o(t2, s1, s1sq) )
bracket = Cc + np.transpose(Cc, (2,0,1)) + np.transpose(Cc, (1,2,0))   # far on each of the 3 axes
Var_W = EW + bracket / m**3
```
Take only the `A<B<C` entries. This is the verbatim analogue of the general-case code; it matches
`Var_W_bip` in `bipartite_check.py`, which is the validation reference. A 4th-moment exact
correction exists but the leading form is conservative and recommended.

**5. Drop triangles.** A bipartite graph has no triangles — remove any triangle/Q1 path; the wedge
is the only three-node motif.

## Keep unchanged
`N = A @ L`, observed `J = L.T @ N`, observed wedge `W = Σ_g N_g.T N_g`, the `Z`/`log2_ratio`/Poisson
scoring, per-cell→condition aggregation. The real graph already encodes the bipartite structure, so
observed counts are correct as-is.

## Validate before trusting (a few cells × a few motifs)
* **Bipartite Chung–Lu MC** (checks the formulas): sample cross-side edges with `p_ij=d_i d_j/m`
  (`nx.bipartite.configuration`-style, or `bipartite_check.py`'s `rng.random((n1,n2))<P`), recompute
  the **same** observed statistic, compare analytic mean/variance. Accept: **mean within ±3%**,
  **variance within ±10%** (leading order is a few % high — conservative).
* **Degree-preserving edge-swap MC** (checks modelling on the real graph): swap only
  `a–b,c–d → a–d,c–b` so edges stay cross-side (bipartite-preserving); compare. The analytic should
  now agree (it didn't before the fix).
* Sanity: for side-balanced markers the new and old means coincide; for side-asymmetric markers they
  must differ and only the new one matches MC.

## Pitfalls
- Normaliser is **`m` (edge count)**, not `2E`. (`2E = 2m`.)
- Strengths must be split by side **before** `bincount`.
- A generic (non-bipartite) edge-swap/rewire would create same-side edges — use the
  bipartite-preserving swap for the MC target.
- Don't touch the observed-count code.
