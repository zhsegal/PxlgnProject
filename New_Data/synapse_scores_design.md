# Four synapse scores for the blinatumomab MPX dataset

**Date:** 2026-05-16 · **Author:** Claude (design doc) · **Status:** proposal for review before implementation.

Distillation of the rounds 1–5 findings into four per-cell composite scores you can attach to `adata.obs`. Each score fuses **abundance** (CLR or arcsinh5 layer) with **spatial colocalization** (`spatial_raw` z-scores) on a small, hand-picked marker panel.

## 0. Design principle

Round 4 settled the central architectural question: **abundance of synapse machinery is symmetric across targets; spatial deployment is the gate.** The same checkpoints rise on engaging CD8 regardless of target; only on NALM-6 are they assembled into a productive cluster (PSGL-1 hub on T side, PD-L1/PD-L2/LAIR-1/CD22 brake-raft on B side).

Two operational consequences:

1. Each score is the weighted sum of an **abundance arm** and a **spatial-assembly arm**, with the spatial arm weighted higher (2:1) — abundance alone is not diagnostic of an engaged synapse.
2. Standardize each component first (z-score within the relevant cell type and within sample-baseline, where appropriate), then combine. This prevents one high-variance marker from dominating.

**General formula (per cell *i*, score *S*):**

```
S_i  =  (1/3) · mean_j  z(abund_ij)   +   (2/3) · mean_k  z(coloc_ik)
```

where *j* indexes the abundance marker panel and *k* indexes the colocalization pair panel. `z()` is z-scoring across all cells of the appropriate type (CD8, CD4, or B). For the spatial term, the `spatial_raw` values are already z-scored at the cell-pair level, so a light arcsinh(z/5) before averaging is enough; further z-scoring across cells is optional and only needed to put the two arms on the same scale.

For directional scores (everything points the same way: activation → up; inhibition → up), invert any markers known to drop with engagement before averaging (none in the panels below — they are all picked to be coherent).

The four scores below are independent — you can compute them all and use ratios (e.g., `apc_activation / apc_inhibitory`, or `cd8_synapse - apc_inhibitory`) to capture the kinetic race that decides selective killing.

---

## 1. CD8 immune synapse score — the productive killing synapse

**Captures:** assembly of a competent cSMAC + exclusion zone with the PSGL-1-organized inhibitory cluster nucleating beside it. Expected pattern from round 3–4: high in NALM-6 + Blina CD8 at 6 h, much lower in healthy-B + Blina CD8.

### Abundance panel (early-engagement activation markers)

| Marker | Rationale |
|---|---|
| **CD137 (4-1BB)** | Coordinated 6 h costim upregulation; strongest NALM-specific in round 3 |
| **CD25 (IL-2Rα)** | Productive engagement; ramps and stays high |
| **CD71 (TfR1)** | Proliferation commitment — only on productively engaged T |
| **CD134 (OX40)** | Costim leg of the frustrated module |
| **CD69** | Earliest activation; symmetric in CD4 but more diagnostic for CD8 here |

(Optional extension: CD27, HLA-DR, CD38.)

### Colocalization pair panel (cSMAC + exclusion + frustrated-inhibitory hub)

| Pair | Captures |
|---|---|
| **CD3e × CD8** | Signal-1 cSMAC core |
| **CD11a × CD45** | Exclusion-zone tightening (HT/NALM-specific in round 3) |
| **CD162 (PSGL-1) × CD3e** | PSGL-1 recruited to the cSMAC — the spatial gate (round 4) |
| **CD162 × TIGIT** | The inhibitory hub assembling next to costim — frustrated module |
| **CD162 × VISTA** | Same; redundant inhibitory layer |
| **CD19 × CD45** | Trogocytosis at the synapse — direct engagement footprint |

(Optional: CD11a × HLA-ABC for the "tightening of structural identity" finding.)

### Use

- High score on NALM-6 + Blina CD8 at 6 h, lower at 48 h (synapse relaxation, round 3).
- On healthy-B + Blina CD8 the abundance arm rises (round 4 finding 1) but the spatial arm stays flat — the score is therefore **systematically lower** on healthy-B because the spatial arm is the gate. This is exactly the desired property: it isolates productive synapse formation.

---

## 2. CD4 immune synapse score — the productive helper synapse

**Captures:** MHC-II-anchored helper synapse assembly. From round 5: cluster 1 (CD11a, CD2, SLAMF6, CD3e, CD44, CD45, CD50) is 2.1× tighter on NALM-6 than on healthy B for CD4; top up-coloc pairs are MHC-II-anchored.

### Abundance panel

| Marker | Rationale |
|---|---|
| **CD154 (CD40L)** | Cardinal CD4 helper effector — already at ceiling on HT/NALM CD4 at baseline (round 5 finding 4) |
| **CD69** | Largest off-diagonal HT/NALM vs HT/HB CD4 marker at 6 h Blina (+0.97 LFC, round 5) |
| **CD134 (OX40)** | Coordinated costim with CD40L |
| **CD137 (4-1BB)** | Same |
| **CD25** | Helper licensing; peaks at 48 h in HT/HB, 6 h in HT/NALM |

(Optional: CD278/ICOS for canonical CD4 costim — wasn't a top hit in round 5 but is biologically correct.)

### Colocalization pair panel (MHC-II-anchored cluster 1)

| Pair | Captures |
|---|---|
| **CD3e × HLA-DR-DP-DQ** | MHC-II–TCR cSMAC organizer — the CD4-defining axis |
| **CD4 × HLA-DR-DP-DQ** | Co-receptor docking to MHC-II |
| **CD44 × HLA-DR-DP-DQ** | Top HT/NALM CD4 coloc pair in round 5 (+0.279) |
| **CD45 × HLA-DR-DP-DQ** | Helper cSMAC core (+0.217 in round 5) |
| **CD11a × CD45** | Structural cluster-1 axis (same as CD8) |
| **CD2 × SLAMF6** | Helper-specific costim adhesion |

### Use

- **Kinetics are reversed vs CD8** (round 5 finding 3): HT/NALM CD4 score peaks at 6 h, drops at 48 h (target gone). HT/HB CD4 score is low at 6 h, peaks at 48 h (frustrated late helper without an effector partner).
- Subtract the **6 h Mock between-system delta** from each cell's score to remove the baseline HT/NALM pre-activation confound (round 5 finding 1, priority 1).

---

## 3. APC-activation score — the productive Signal-2 hub on B cells

**Captures:** the canonical APC competency a B cell brings to the synapse: MHC-II display, costim ligands (CD80/CD86), CD40, ICAM-1/LFA-3 adhesion docking. Expected pattern: high on healthy B baseline (constitutively APC-competent); paradoxically *induced* on healthy B at 48 h Blina (round 3 — CD86 +2.27, CD40 up); persistently low on NALM-6 (CD80⁻, CD86± deficit is universal across all conditions).

### Abundance panel

| Marker | Rationale |
|---|---|
| **CD80** | Universal NALM-6 deficit (LFC −1.23 to −1.67, *all* conditions) — strongest discriminator |
| **CD86** | Healthy-B paradoxical 48 h Blina surge (+2.27, FDR 1.6e-36) |
| **CD40** | Healthy-B induction under Blina; pairs with CD40L on CD4 |
| **HLA-DR-DP-DQ** | MHC-II display — the platform |
| **CD20** | Mature-B identity (NALM-6 is CD20⁻) — anchors "APC-competent stage" |
| **CD54 (ICAM-1)** | T-cell adhesion docking — Signal-3 architectural |

### Colocalization pair panel (productive Signal-2 cluster)

| Pair | Captures |
|---|---|
| **CD80 × HLA-DR-DP-DQ** | Canonical Signal-2 + MHC-II co-display |
| **CD86 × HLA-DR-DP-DQ** | Same |
| **CD40 × HLA-DR-DP-DQ** | Helper-licensed APC maturation |
| **CD54 × HLA-DR-DP-DQ** | ICAM-1 docking at the MHC-II contact |
| **CD58 × HLA-DR-DP-DQ** | LFA-3 docking |
| **CD80 × CD86** | Coordinated costim ligand expression |

### Use

- **Healthy B always high** (baseline APC-competent); **NALM-6 always low** (CD80 deficit anchors this). The difference is constitutive, not Blina-driven.
- Watch for the **paradoxical 48 h Blina rise on healthy B** (round 3) — the score should jump there. This is the CRS-bystander hypothesis substrate.
- Pairs nicely with the CD4 helper score: where CD4 helper score is high *and* APC-activation is high, you should see productive licensing (and consequent CD8 killing); where CD4 helper score is high but APC-activation is low (NALM-6 case), licensing is unproductive and CD40L "leaks" to bystanders.

---

## 4. APC-inhibitory score — the brake-raft on B cells

**Captures:** the late-assembled inhibitory raft on NALM-6 (round 4) — PD-L1 + PD-L2 + LAIR-1 + CD22 + PSGL-1 + CD80, organized around PSGL-1 on the B-cell side — plus the constitutive purinergic brake on healthy B (CD39/CD73). High score = brake actively deployed; very high score AND high APC-activation on the same cell = the "tolerogenic synapse" healthy B wins the kinetic race with.

### Abundance panel

| Marker | Rationale |
|---|---|
| **CD274 (PD-L1)** | Induced on both targets under Blina, but reorganized only on NALM |
| **CD273 (PD-L2)** | NALM-6-specific induction at 6 h Blina (+0.44); flat on healthy B |
| **CEACAM8** | NALM-6-specific induction (+0.52, round 4 slide 5) |
| **CD305 (LAIR-1)** | ITIM-bearing brake; clusters with PD-L1 |
| **CD22 (Siglec-2)** | BCR ITIM brake; organizer of the healthy-B tolerogenic synapse |
| **CD39 / CD73** | Constitutive Breg adenosine brake on healthy B (low on NALM-6) |

(Optional: CD72 — inhibitory FcR-like; appears as a PSGL-1 partner on B in round 4.)

### Colocalization pair panel (the brake-raft)

| Pair | Captures |
|---|---|
| **CD274 (PD-L1) × CD305 (LAIR-1)** | Tandem-inhibition signature, +0.033 in HT/NALM Blina |
| **CD273 (PD-L2) × CD274 (PD-L1)** | Both PD-1 ligands clustering — redundant checkpoint layer |
| **CD162 (PSGL-1) × CD22** | B-side PSGL-1 organizer of the brake-raft |
| **CD162 × CD305 (LAIR-1)** | PSGL-1 ↔ LAIR-1 ITIM coupling |
| **CD162 × CD19** | PSGL-1 recruited to the BiTE-bound CD19 |
| **CD274 × CD22** | PD-L1 clustering with the BCR brake |
| **CD39 × CD73** | Constitutive purinergic axis (healthy-B baseline brake) |

### Use

- **High and constitutive on healthy B** (CD39/CD73 axis + CD22 tolerogenic core, regardless of condition).
- **Low at baseline on NALM-6, rising sharply at 6 h Blina, falling by 48 h** as NALM dies (PD-L2 collapses, round 4 finding 5). This kinetic — a delayed brake-raft assembly that loses the kinetic race — is the unique signature of the score.
- **Together with `cd8_synapse` and a synapse-tightness latency, predicts which target survives**: when `apc_inhibitory` rises faster than `cd8_synapse` tightens (the healthy-B pattern), the kill fails; when `cd8_synapse` is already saturated before `apc_inhibitory` assembles (the NALM-6 pattern), the kill succeeds.

---

## 5. Suggested derived metrics (per cell or per sample)

Beyond the four raw scores, the most useful combinations:

| Metric | Definition | Captures |
|---|---|---|
| **Kill-permission ratio** | `cd8_synapse − apc_inhibitory` (matched B–T pair, otherwise sample mean) | Net pressure to kill at a given moment |
| **APC functional state** | `apc_activation − apc_inhibitory` | "Engaged-tolerogenic" vs "engaged-licensable" — distinguishes healthy-B (high both) from NALM (low activation, late inhibitory) from quiescent B (low both) |
| **Helper licensing index** | `cd4_synapse × apc_activation` (multiplicative — both must be present) | Where productive helper licensing actually happens; bystander leakage when CD4 high but matched APC-activation low |
| **Trogocytosis sub-score** (CD8/CD4) | mean coloc of `CD19, CD22, HLA-DR-DP-DQ, CD10` *pairs on the T cell* | Direct membrane-transfer fingerprint independent of activation |

---

## 6. Implementation notes

1. **Layer choice.** For the abundance arm use `adata.layers["arcsinh5"]` (or CLR — they give similar score behavior; arcsinh5 is more stable across the heavy-tail markers like CD25/CD71). Standardize per marker within each cell type (CD4 / CD8 / B) before averaging.
2. **Coloc pair availability.** `spatial_raw` keys depend on pair-ordering convention — check both orderings (e.g., `CD3e_CD8` and `CD8_CD3e`) when extracting; some pairs may be on `spatial_asinh5` only. The `build_spatial_obsms` utility already handles symmetry.
3. **Missing values.** Cells with `n_umi < 25k` or `tau_type != "normal"` are already filtered, but coloc pairs can be `NaN` for cells where one marker is below detection. Use nan-aware means (`np.nanmean`) when averaging across the pair panel.
4. **Baseline subtraction (CD4 score in particular).** To isolate true drug-induced helper-synapse competence from the HT/NALM pre-activated baseline (round 5 finding 1), compute the 6 h Mock between-system mean of `cd4_synapse` for each `cell_system` and subtract it from each cell's score.
5. **Validation reference points.** A correctly computed score set should reproduce, qualitatively:
   - `cd8_synapse` high in NALM-6 + Blina 6 h CD8, near-zero in healthy-B + Blina 6 h CD8.
   - `cd4_synapse` peaks at 6 h Blina in HT/NALM CD4, 48 h Blina in HT/HB CD4.
   - `apc_activation` high on healthy B at all conditions, ~zero on NALM-6.
   - `apc_inhibitory` constitutive-high on healthy B, ramps then collapses on NALM-6 across 6→48 h Blina.
6. **What this won't catch.** A cell-pair-level (matched B↔T) brake-vs-kill kinetic race needs the 3D layout + per-cell-pair matching (priority 2 in round 4). These scores live at the single-cell level and answer "is this cell in synapse mode?" — not "is this specific contact going to result in lysis?". The next step after the scores work is to link them across matched pairs using the 3D coordinates.

---

## 7. Markers I deliberately left out (and why)

- **CD11a abundance on T cells.** Stable across all conditions (rounds 3–4); spatial-only signal. It's in the coloc panel, not the abundance panel.
- **IgM/IgD/IgE on B.** Their drop is the death signature itself, not a synapse signal. Use as a separate "death-progress" metric, not in the APC scores.
- **CD10 (NEP) on B.** Dynamic and paradoxical (NALM-6 should be CD10⁺ but is depleted in co-culture, round 3); it confounds the developmental-stage axis and would muddy the activation score.
- **CD45RA / CD45RO on T.** Memory-state markers, not synapse-engagement markers; they belong in a differentiation score, not a synapse score.
- **CD158a/b/CD94 (KIRs).** Late (48 h) inhibitory markers on CD8 — a separate "late-exhaustion" score, not the same kinetic phase as the brake-raft.

---

*Companion notebook implementation (proposed): one function per score, returning an (n_cells,) array; one combined function returning a DataFrame with all four scores + four derived metrics; one validator that produces the qualitative-reproduction table in §6.5.*
