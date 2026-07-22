# PixelGen Blinatumomab Co-culture: Project Summary

A self-contained briefing for analyzing single-cell PixelGen MPX data from a blinatumomab B-ALL co-culture experiment. Covers method, experimental design, data structure, transformations, and preliminary biological findings.

---

## 1. PixelGen Molecular Pixelation (MPX) — How the Method Works

**Technology**: PixelGen Technologies' Molecular Pixelation, run with the **PNA (Proximity Network Assay)** chemistry (panel design `pna-2`). Single-cell, single-molecule resolution surface proteomics by sequencing — *no microscope, no imaging*. Each antibody is conjugated to a DNA tag. On fixed cells, neighboring antibodies are connected through a rolling proximity reaction whose products are sequenced. This produces, per cell:

- A **graph** where nodes = individual antibody molecules and edges = proximity events between molecules.
- The graph is decomposed into a **3D layout** (here: pmds_3d algorithm) reconstructing the surface coordinates of every detected antibody molecule.

**Per-cell measurements derived from the graph**:

| Modality | What it captures | Type |
|---|---|---|
| **Abundance** | Counts per marker per cell (analogous to CITE-seq / flow) | cells × markers count matrix |
| **Spatial colocalization** | Per-pair z-score: are two markers spatially closer than expected by chance? Positive = clustered; negative = mutually excluded; ≈0 = random | cells × marker-pairs |
| **Polarization** | Per-cell, per-marker score (Hotspot-style): is the marker locally clustered or uniformly distributed across the surface? | cells × markers |
| **3D coordinates** | Coordinates of every antibody molecule in 3D space, per cell | molecule-level |

This makes MPX the only method that gives genuine per-cell, nanometer-scale **membrane organization** (synapse architecture, lipid raft proximity, trogocytosis footprints) at scale.

**Panel used**: `proxiome-immuno-156-FMC63` — **159 markers** after collapse, including:
- Pan-immune lineage markers (CD3e, TCRab, CD4, CD8, CD19, CD20, CD56, CD14, HLA-DR…)
- B-cell developmental + functional markers (CD10, CD19, CD20, CD21, CD22, CD24, CD32, CD35, CD37, CD40, CD72, CD80, CD86, CD138, CD180, CD268, CD269, CD79a, IgM, IgD, IgE)
- T-cell costimulation/coinhibition (CD27, CD28, CD134/OX40, CD137/4-1BB, CD152/CTLA-4, CD154/CD40L, CD226/DNAM-1, CD244/2B4, CD278/ICOS, CD279/PD-1, CD357/GITR, TIGIT, VISTA/B7-H5, CD366/TIM-3)
- NK-like (CD56, CD94, CD158a, CD158b, CD159a, CD161, CD314, KLRG1)
- Adhesion / synapse (CD11a/LFA-1α, CD18/ITGB2, CD29, CD49D, CD50/ICAM-3, CD54/ICAM-1, CD58/LFA-3, CD103, CX3CR1)
- Memory / differentiation (CD44, CD45RA, CD45RO, CD45RB, CD127, CD57, KLRG1)
- Tetraspanins (CD9, CD37, CD53, CD63, CD81, CD82, CD231)
- MHC / structural (HLA-ABC, HLA-DR-DP-DQ, B2M, CD43, CD44, CD52)
- Breg / purinergic (CD39, CD73)
- Plasma cell (CD138)
- And FMC63 (the anti-CD19 scFv used by tisagenlecleucel and blinatumomab — useful as a synapse landmark).

A complete marker → alias dictionary (e.g. CD279 → PD-1, CD134 → OX40, CD11a → ITGAL) is curated alongside grouped functional panels for B-cell, CD4, and CD8 analysis.

---

## 2. Experimental Design — Blinatumomab Co-culture

### Biological context
Blinatumomab is a clinically used **CD19×CD3 bispecific T-cell engager (BiTE)** for relapsed/refractory B-ALL. It forces a synthetic immunological synapse between any T cell and any CD19⁺ target — bypassing TCR specificity and (as we will see in the data) bypassing classical LFA-1/ICAM-1 adhesion as well.

### Design
**Three "B cell + T cell" systems × two timepoints × two conditions** plus a fourth cross system. Sequencing by Novogene (≈630 GB total, Q30 > 96.6%, contract X201SC26026419-Z01-F001), delivered March 2026.

| Sample | Time | Condition | B-cell target | T cells |
|--------|------|-----------|---------------|---------|
| S001 | 6h | Mock | healthy B | healthy T |
| S002 | 6h | 1 ng/ml Blina | healthy B | healthy T |
| S003 | 48h | Mock | healthy B | healthy T |
| S004 | 48h | Blina | healthy B | healthy T |
| S005 | 6h | Mock | NALM-6 | healthy T |
| S006 | 6h | Blina | NALM-6 | healthy T |
| S007 | 48h | Mock | NALM-6 | healthy T |
| S008 | 48h | Blina | NALM-6 | healthy T |
| S009 | 6h | Mock | patient B | patient T |
| S010 | 6h | Blina | patient B | patient T |
| S011 | 48h | Mock | patient B | patient T |
| S012 | 48h | Blina | patient B | patient T |
| S013 | 6h | Mock | NALM-6 | patient T |
| S014 | 6h | Blina | NALM-6 | patient T |
| S016 | 48h | Blina | NALM-6 | patient T |

**S015 (48h Mock NALM-6 + patient T) was missing from the delivery.**

NALM-6 = a human B-cell acute lymphoblastic leukemia cell line, the reference target for blina in vitro.

---

## 3. Pipeline & QC

**Pipeline**: pixelator v0.21.3 PNA. Stages: amplicon → demux (`--strategy paired`, panel `proxiome-immuno-156-FMC63`) → collapse → graph (`--multiplet-recovery`) → denoise → analysis (`--compute-proximity` ⇒ colocalization + polarization) → layout (pmds_3d).

Critical pitfalls discovered during processing:
- `--strategy paired` is **mandatory**; the default independent strategy collapses everything into 1 cell per sample.
- The graph step requires ≥145 GB RAM for ~100M molecules.
- `--compute-proximity` and pmds_3d layout spawn many threads; LSF thread-affinity caps must be removed.
- Paired collapse output uses different naming (`SAMPLE.collapsed.parquet`) than the independent strategy.

**Cell yield** (after pipeline, before QC): **16,958 cells** across 15 samples × 159 markers, mean 1,130 cells/sample (range 859–1,505).

**QC filtering**: `tau_type == "normal"` (pixelator's outlier flag) + `n_umi ≥ 25,000` ⇒ **15,782 cells**. After doublet/CD4-CD8 ambiguous removal: **14,217 cells**.

**Cell type composition** (after annotation): B = 6,399; CD4 = 5,632; CD8 = 2,186.

---

## 4. Data Structure & Transformations

The data going into downstream analysis is an AnnData object (`cells × markers = 15,782 × 159`) with the following:

### Per-cell QC metadata
`n_umi`, `n_umi1/2`, `n_edges`, `n_antibodies`, `tau`, `tau_type`, `isotype_fraction`, `intracellular_fraction`, `reads_in_component`, plus the experimental metadata `sample`, `time`, `condition`, `target`, `tcells`, `cell_system`.

### Abundance transformations applied

| Transform | Purpose | Notes |
|---|---|---|
| **Raw counts** | Untransformed input | Heavily skewed, library-size confounded |
| **CLR** (centered log-ratio) | Removes compositional/library-size effects | Standard for surface protein counts; primary input for scVI |
| **log1p** | Quick visualization | `log1p(raw)` |
| **arcsinh (cofactor 5)** | CyTOF/MPX-style stabilization | Used for downstream DE on T cells |
| **arcsinh + per-batch min-max scale** | Input layer for CytoVI | Per-`cell_system` batch normalization, then merged |

### Spatial-feature transformations applied
The colocalization output is a per-cell matrix of marker-pair z-scores (≈12,000 pair columns for 159 markers, after symmetry removal). Stored variants:

| obsm key | Transform | Purpose |
|---|---|---|
| `spatial_raw` | Raw colocalization z-scores | Direct statistical testing on selected pairs |
| `spatial_asinh5` | `arcsinh(z / 5)` | Tame extreme z-scores while preserving sign |
| `spatial_asinh5_top500var` | Top 500 most variable pair columns from above | Compact input for scVI / multimodal latent |

(Additional transformations available in the codebase: tanh-based saturating, quantile normalization. These can be selected via the `build_spatial_obsms` utility.)

### Latent representations computed

| Latent | Method | Batch key | Use |
|---|---|---|---|
| **z_scvi** (20-d) | scVI on CLR layer | `sample` | First-pass UMAP/Leiden |
| **X_CytoVI** (20-d) | CytoVI on per-batch-scaled arcsinh | `cell_system` | Annotation; used in production |
| **Multimodal scVI** (optional) | `MultiModalSCVI` joint encoder over CLR abundance + spatial obsm | `sample` | Joint abundance × spatial latent — implemented in repo, not the headline analysis yet |

The multimodal model supports four aggregation strategies for combining modalities: shared encoder, fixed-weight product-of-experts, learned global modality weights, or per-cell modality weights.

### Annotation
Leiden clustering on the CytoVI latent → 18 clusters → hand-mapped to **B / CD4 / CD8 / Doublets / CD4-CD8** based on a dotplot of CD3e, TCRab, CD5, CD7, CD2, CD4, CD8, CD19, CD20, CD22, IgM, IgD on the denoised expression layer. Doublets and CD4/CD8 ambiguous clusters are removed before composition analysis.

### Marker panels (curated, used throughout downstream analysis)

**B cell** (6 functional groups):
1. Core identity & co-receptors: CD19, CD20, CD22, CD79a
2. Immunoglobulins: IgM, IgD, IgE
3. APC hub: CD40, CD80, CD86
4. Development/survival/memory: CD10, CD21, CD24, CD138, CD268, CD269
5. Inhibitory/innate interface: CD32, CD35, CD37, CD72, CD180
6. Breg markers: CD39, CD73

**CD8 T cell** (7 functional groups):
- Pan-T identity: CD3e, TCRab, CD2, CD5, CD6, CD7, CD8
- Costimulation/survival: CD28, CD27, CD134, CD137, CD226, CD278, CD357
- Activation/proliferation: CD25, CD38, CD69, CD71, HLA-DR, CD95
- NK-like cytotoxicity: CD56, CD94, CD158a, CD158b, CD161, CD314, GPR56
- Exhaustion: CD279, CD366, TIGIT, VISTA, CD152, CD244, CD159a, CD305, CD39, CD73
- Memory: CD45RA, CD45RO, CD44, CD127, CD57, KLRG1
- Adhesion/homing: CD11a, CD18, CD29, CD49D, CD54, CD103, CX3CR1
- pSMAC adhesion (sub-panel): CD18, CD11a, CD50, CD29, CD49D

---

## 5. Headline Result — Selective Killing of NALM-6 by Blinatumomab

The defining phenotypic readout of the experiment, computed as **log₂(B / T)** ratio per sample across 4 systems × 2 conditions × 2 timepoints (with B-cell, CD4, CD8 fractions per sample as the building blocks):

- **NALM-6 + healthy T** and **NALM-6 + patient T**: B-cell fraction collapses sharply between 6 h and 48 h Blinatumomab. log₂(B/T) drops dramatically. Mock condition is preserved.
- **Healthy B + Healthy T** and **Patient B + Patient T**: B-cell fraction is essentially preserved at 48 h Blinatumomab — no measurable depletion in our window.

**Interpretation**: At 1 ng/ml, blinatumomab in vitro produces **selective killing of NALM-6 B-ALL blasts** within 48 h. Primary healthy B cells and primary patient B cells are not measurably depleted. This is the central biological observation of the project, because blinatumomab is mechanistically supposed to be agnostic to B-cell identity (it engages any CD19⁺ cell). The differential killing therefore demands explanation, and the rest of the analysis investigates the molecular and spatial differences that could account for it.

Two leading hypotheses, both supported by phenotypic and spatial findings below:

1. **NALM-6 lack costimulatory machinery** (CD80/CD86 deficit, no APC function) and therefore cannot anergize the engaging T cell once a synthetic synapse is forced; primary B cells deliver normal "Signal 2" and may dampen the engagement.
2. **Healthy B cells produce immunosuppressive adenosine** via constitutively high CD39/CD73 (Breg-like), partly braking T-cell activation in the healthy co-culture but not in NALM-6.

Neither alone is a complete explanation, but together they form a testable mechanistic frame.

---

## 6. Preliminary Phenotypic Findings

Statistics throughout: per-marker Mann-Whitney U + Kruskal-Wallis on arcsinh-transformed abundance, Benjamini-Hochberg FDR correction.

### 6a. B cells — NALM-6 are arrested pre-B blasts; healthy B are mature

The data converge on a single biological narrative: **NALM-6 leukemic blasts are profoundly arrested at an immature B-cell precursor stage and functionally disconnected from every normal immune network.**

| Functional axis | Healthy B | NALM-6 | Effect size |
|---|---|---|---|
| **Identity** (CD19⁺CD20⁻ vs CD19⁺CD20⁺) | CD20⁺ mature | CD20⁻ pre-B | CD20 mean diff 2.95–3.58, FDR < 1e-40 |
| **BCR repertoire** | IgM/IgD/IgE balanced | Isolated **IgM surge** at 48h Mock | IgM diff +1.47, FDR < 1e-178 — implies arrest before class-switch |
| **APC hub (Signal 2)** | CD40⁺ CD80⁺ CD86⁺ | CD40⁺ **CD80⁻** CD86± | CD80 diff −1.23 to −1.67 across **all** conditions (universal, never closes) |
| **Development markers** | CD10⁺ CD24⁺ | CD10/CD24 dramatically depleted | CD10 diff −4.89 to −6.53 (atypical for a CD10⁺ B-ALL line — suggests dynamic loss in co-culture) |
| **Survival** | BAFF-R/BCMA-dependent | BCMA low, oncogene-driven | NALM-6 BAFF-independent, cannot form memory |
| **Innate sensing** | CD32, CD35, CD37, CD180 expressed | All depressed | "Deaf to environment" — no LPS sensing, no inhibitory Fc brake |
| **Breg markers** | High baseline CD39/CD73 | Low baseline, **time-dependent rise 6h→48h Mock** | Latent adaptive immunosuppression |

**Mechanism of blina's surface-Ig collapse**: in both systems blinatumomab suppresses IgM/IgD/IgE — this is the cell-stress / death signature itself, not isotype-targeted suppression. It is direct evidence of drug efficacy. Suppression of CD39/CD73 in NALM-6 under blina shows the drug kills these cells **before** they complete their adaptive evasion upgrade (the time-dependent rise seen in the Mock arm).

**Paradoxical activation under attack**: even as blinatumomab-engaged T cells kill them, healthy B cells **upregulate CD86** (mean diff +2.27 at 48h Blina, FDR 1.6e-36) and CD40 — they sense the IFN-γ/CD40L surge and try to participate. NALM-6 cannot mount this response, confirming developmental arrest.

**Spontaneous leukemic plasticity**: NALM-6 are **not phenotypically static**. Across 6h → 48h Mock, CD20, CD22, CD39, CD73 all rise. This suggests an intrinsic maturation/evasion program that activates without external stimulation — and frames a real risk of adaptive resistance to immunotherapy if drug exposure is incomplete or delayed.

### 6b. CD8 T cells — frustrated activation under BiTE engagement

In NALM-6 + Blinatumomab co-culture (vs healthy or Mock):
- **Coordinated costimulatory upregulation at 6 h Blina**: OX40 (CD134), 4-1BB (CD137), CD27.
- **Multi-pathway exhaustion signature** rising in parallel: VISTA, TIM-3, TIGIT, PD-1.
- **KIR acquisition** (CD158, CD158a, CD94/KLRD1) at 6h Blina.
- **CD11a abundance is stable** across conditions — the changes are spatial, not abundance-level.
- **CD39 purinergic scarring persists** in NALM-6-exposed CD8 even after blina-induced spatial normalization → "the physical wound heals, the metabolic wound persists."

The CD8 picture: classical activation markers and exhaustion checkpoints rise together, on the same cells, at the same time — the spatial analysis below explains how this dual signature is organized on the membrane.

---

## 7. Preliminary Spatial Findings — The Synthetic Synapse

The colocalization analysis (focused on CD8 T cells, CD134/OX40, TIGIT, and CD11a/LFA-1α as probes for three biological axes: costimulation, exhaustion, adhesion) decomposes the BiTE-induced T-cell membrane into spatially distinct compartments at 6 h Blinatumomab:

```
T CELL MEMBRANE (NALM-6 + Blina, 6 h)

┌─── EXCLUSION ZONE ──────────┐   ┌── SYNAPSE CORE ──────────┐
│ CD11a/CD18 (LFA-1, idle)    │   │ Trogocytosed B-cell:     │
│ CD45 (phosphatase)          │   │   CD19, CD10, CD24       │
│ CD43 (leukosialin)          │   │   HLA-DR-DP-DQ, CD22     │
│ CD44 (HCAM)                 │   │   CD37, CD80, IgM        │
│ B2M / HLA-ABC (MHC-I)       │   │ Adhesion anchors:        │
│ CD45RA                      │   │   CD54 (ICAM-1)          │
│ CD50 (ICAM-3)               │   │   CD58 (LFA-3)           │
└─────────────────────────────┘   │   CD9/CD81/CD53 (TEMs)   │
                                  │ ADAM10 (membrane remodel)│
┌── FRUSTRATED COSTIM ────────┐   └──────────────────────────┘
│ OX40 + VISTA (early brake)  │   ┌── CHECKPOINT ZONE ───────┐
│ OX40 + CX3CR1 (effector)    │   │ TIGIT + PD-L2 (CD273)    │
│ OX40 + CD90, CD73           │   │ TIGIT + CX3CR1, CD94     │
│ uncoupled from TCR/CD45     │   │ TIGIT + PD-1 + 4-1BB     │
└─────────────────────────────┘   └──────────────────────────┘
                  ↕ Blinatumomab CD3-CD19 bridge ↕
┌─────────────────────────────────────────────────────────────┐
│                  NALM-6 TARGET CELL                          │
│  CD19, CD10, CD24, HLA-DR, CD22, CD37, CD80                 │
└─────────────────────────────────────────────────────────────┘
```

### Key spatial findings

1. **The synthetic synapse bypasses LFA-1/ICAM-1 adhesion entirely.** CD11a (LFA-1α), the canonical pSMAC adhesion integrin of natural T-cell synapses, is **excluded** from the blinatumomab-enforced contact zone. It anchors the exclusion zone instead, alongside bulky phosphatases (CD45) and structural glycoproteins (CD43, CD44). The CD11a/CD18 heterodimer remains structurally assembled (z ≈ 0.62) but is functionally idle. This has not been described before for BiTE synapses and is a candidate explanation for why BiTE T cells exhaust faster than physiologically engaged T cells — they never get the integrin signal that natural adhesion provides.

2. **Massive trogocytosis, spatially organized.** B-cell markers (CD19, CD10, CD24, CD22, HLA-DR-DP-DQ, CD37, CD80, IgM, CD138, CD21, CD273) are detected on purified CD8 T cells, **clustered together at the synapse core** and **anti-correlated with the exclusion zone**. This is direct spatial evidence of extensive membrane transfer concentrated at the contact zone. Trogocytosis-painted T cells could become fratricide targets — relevant for dosing strategy.

3. **Costimulation and exhaustion physically share a domain.** OX40 colocalizes simultaneously with CX3CR1 (terminal effector), VISTA (early checkpoint), CD90 (lipid raft), CD73 (purinergic), and trogocytosed B-cell markers. TIGIT colocalizes simultaneously with PD-L2, PD-1, 4-1BB, CD94. Activation and inhibition machinery occupy the same membrane microdomain → "frustrated activation" — the cell wires its own brakes physically next to its accelerators.

4. **Temporal switching of inhibitory partners.** OX40's spatial brake switches from **VISTA at 6 h** to **KIR receptors (CD158, CD94) at 48 h**. This is a programmed temporal sequence of checkpoint recruitment, not a single static mechanism. It suggests checkpoint blockade strategies that work at one timepoint may be irrelevant at another.

5. **Spatial normalization at 48 h, but residual scars.** By 48 h, the synapse architecture fully resets — CD11a's exclusion zone relaxes (HLA-ABC colocalization drops 0.564 → 0.143, p=7e-5), B-marker exclusion blurs, OX40/TIGIT lose their synapse partners. But the trogocytosis B-marker signature persists at reduced intensity, and the **CD39 abundance scar persists**. The physical synapse heals; the metabolic imprint does not.

6. **Tightening of the structural identity network under leukemic stress.** In NALM-6 vs healthy at the same condition, every core T-cell marker shows tighter colocalization with CD11a (e.g. CD45: 0.88 vs 0.50, CD3e: 0.47 vs 0.24). As synapse-recruited molecules migrate toward the contact, the remaining structural components consolidate — the exclusion zone becomes denser.

---

## 8. Therapeutic / Mechanistic Implications

Drawn from the phenotypic and spatial layers together:

| Combination | Rationale | Predicted benefit |
|---|---|---|
| Blina + **anti-VISTA / anti-KIR** | VISTA at 6 h, KIRs at 48 h are physically clustered with OX40 — direct spatial brakes on costimulation | Amplify early cytotoxicity before exhaustion programs lock in |
| Blina + **CD28 or 4-1BB agonist** | NALM-6 lacks CD80/CD86 (no Signal 2); BiTE engagement runs on artificial CD3 alone, predisposing to T-cell exhaustion | Restore costimulation, slow exhaustion |
| Blina + **BTK inhibitor (ibrutinib)** | NALM-6 IgM surge at 48 h Mock indicates active BCR survival signaling | Synergistic killing, prevents IgM-driven adaptive survival |
| Blina + **CD39/CD73 inhibitor** | NALM-6 shows time-dependent CD39/CD73 rise; healthy B express constitutively high levels (Breg-like brake) | Prevent adaptive adenosine evasion; remove healthy-B brake on T-cell activation |
| **Anti-CD20 pre-treatment (rituximab) → Blina** | Healthy B cells are spared by blina but produce immunosuppressive adenosine via CD39/CD73 | Remove healthy-B inhibitory compartment to unmask T-cell killing; potentially also reduces CRS |
| Blina + **LFA-1 / ICAM-1 agonist** | LFA-1 is excluded from the BiTE synapse; recruiting it could create more physiological adhesion | Stronger, more durable contact; reduced premature T-cell disengagement |

The **CRS hypothesis**: healthy B cells upregulate CD86/CD40 in response to blina-driven inflammation. They may inadvertently hyper-activate bystander T cells, contributing to cytokine release syndrome. This predicts that rituximab pre-treatment (clearing the healthy-B compartment) should reduce CRS severity — an existing clinical question this dataset speaks to.

---

## 9. Open Questions for Continued Analysis

1. **Quantify the differential killing rigorously**: per-sample log₂(B/T) statistics across 4 systems × 2 timepoints. Distinguish "NALM-6 selective killing" from "primary B sparing" — they are not the same claim.
2. **CD4 T cells** are under-explored compared to CD8. Do they show the same frustrated activation / synapse architecture, or a distinct profile (e.g. helper-skewed)?
3. **Patient B cells**: phenotypically intermediate between NALM-6 and healthy? If they look like NALM-6 they should be killed; if healthy, spared. Empirically they appear spared — but the biology is not yet characterized at marker resolution.
4. **Patient T cells vs healthy T cells on NALM-6**: does patient T-cell dysfunction reduce killing efficiency? Two systems (NALM-6 + healthy T vs NALM-6 + patient T) allow this comparison directly.
5. **Joint multimodal latent**: combine abundance + colocalization + polarization in a single embedding to find cell states defined by spatial features that abundance alone misses (e.g. trogocytosis-positive CD8 subsets).
6. **3D synapse rendering**: representative T cell : NALM-6 contact pairs, using the 3D layout to visually confirm the trogocytosis core / exclusion zone story.
7. **Why is CD10 so dramatically lost in NALM-6** (−4.89 to −6.53) when NALM-6 is canonically CD10⁺ in the literature? Is this co-culture-induced dedifferentiation? Reversible?
8. **Mechanism of latent leukemic plasticity**: what drives the 6h→48h Mock rise in CD20, CD22, CD39, CD73 in NALM-6 without external stimulation? Is it the in vitro stress response, an intrinsic maturation program, or evasion mimicry?
9. **Is the Breg-brake hypothesis testable here?** Compare healthy-B-cell spatial CD39/CD73 organization with T-cell activation signatures across systems.

---

## 10. Practical Tips for Analysis

- **Always use the CLR or arcsinh-transformed layer**, never raw counts, for differential expression on PixelGen abundance.
- **Filter on `tau_type == "normal"`** (pixelator's outlier flag) and a sensible UMI cutoff (≥25k worked well here) before any analysis.
- **Use `cell_system` as the batch covariate** for VAE integration — it's the dominant batch effect (donor + target type confound).
- **Spatial features have ≈12k columns** (all marker pairs) — for VAE inputs, restrict to top variable pairs (e.g. top 500). For hypothesis-driven analysis, restrict by panel membership before testing.
- **B-cell markers detected on T cells are real** (trogocytosis), not contamination — exclude them from T-cell phenotypic DE but **retain** them for colocalization analysis where they map the synapse.
- **Sample sizes drop sharply at 48 h** (NALM-6 CD8 = 25, healthy CD8 = 55) — many spatial differences fail to reach significance there; do not overinterpret null results at 48 h.
- **The colocalization z-score is signed and pre-normalized** — direct mean comparisons across cells are valid; use Mann-Whitney for robustness against the heavy tails.

---

*Generated 2026-05-02. Project: PixelGen Blinatumomab Co-culture (PNA, panel proxiome-immuno-156-FMC63). Data: 15 samples, 14,217 QC-passed annotated cells, 159 markers, ~12k colocalization pair features.*
