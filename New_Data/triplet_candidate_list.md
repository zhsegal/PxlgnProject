# Candidate three-way colocalization triplets — blinatumomab PNA synapse

**Purpose.** A pre-registered, biologically-motivated set of A–B–C triplets to screen with the wedge / triangle permutation machinery from `threeway_colocalization_methods.md`. Organized by synapse module so you can run module-by-module rather than a blind 6.6×10⁵-triplet scan (the multiplicity pitfall, §6.6 of the methods doc). **218 triplets across 19 modules.**

---

## How to read the table

Each triplet is a **hub-centered wedge** `A–B–C` where **B is the hub** — the molecule you condition on (the shared "spot"). The native test for each is the B-centered wedge enrichment `W_{ABC}` (Q2) and/or the labeled triangle `T_{ABC}` (Q1), scored against the label-shuffle null exactly as the pairwise join counts are (Framework III, §4).

**Columns**

- **Triplet (A–B–C)** — the three markers; B (middle) is the proposed hub/organizer.
- **Hub** — repeated for sorting.
- **Cell / condition** — where the effect is expected to live. `CD8`, `CD4`, `B-NALM` (NALM-6 target), `B-healthy` (primary healthy/patient B). "painted-T" = trogocytosis-painted T cell (see Module 3 note).
- **Sign** — expected direction: **+** co-cluster (wedge upper tail / closure), **−** exclusion (lower tail / segregation), **±** frustrated/segregated hub (one arm +, one −; diagnostic is the wedge *lower* tail + negative λ_ABC).
- **Q** — the sharpest question for this triplet: **Q1** triangle (all three at one spot), **Q2** wedge (two partnerships share the same hub molecule — your "a+b ∧ b+c"), **Q3** cooperativity (more than the pairwise edges predict), **AND** both arms individually real (min-statistic).

**Marker caveat.** Markers are drawn from the recovered panel inventory (`marker_panel_reference.md`). A few used here — **B2M**, **CD150 (SLAMF1)** as the SLAM stand-in for round-5's SLAMF6 — should be confirmed against `adata.var_names` before running. PSGL-1 = CD162, LAIR-1 = CD305, ICAM-2 = CD102, ICAM-3 = CD50, LFA-3 = CD58, CEACAM8 = CD66b, OX40 = CD134, 4-1BB = CD137, PD-L1 = CD274, PD-L2 = CD273, PD-1 = CD279, TIM-3 = CD366.

---

## A. T-CELL SYNAPSE (CD8 killing synapse + CD4 helper synapse)

### Module 1 — CD8 cSMAC / Signal-1 core (productive killing synapse)

Hub is the TCR/CD3 core. Expect tight triangles on NALM-6+Blina 6h CD8, near-random on healthy-B+Blina (the spatial gate is target-specific, round 4).

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 1 | TCRab–CD3e–CD8 | CD3e | CD8, NALM 6h Blina | + | Q1 | cSMAC core; the literal Signal-1 spot |
| 2 | CD3e–TCRab–CD8 | TCRab | CD8, NALM 6h Blina | + | Q1 | same triad, TCRab as hub — robustness check |
| 3 | CD8–CD3e–CD2 | CD3e | CD8, NALM 6h Blina | + | Q2 | co-receptor + CD2 adhesion docking onto TCR |
| 4 | CD2–CD3e–CD5 | CD3e | CD8, NALM 6h Blina | + | Q2 | CD2/CD5 pan-T tuning at the cSMAC |
| 5 | CD5–CD3e–CD6 | CD3e | CD8, NALM 6h Blina | + | Q2 | CD5/CD6 inhibitory-tuning scaffold on TCR |
| 6 | CD7–CD3e–CD2 | CD3e | CD8, NALM 6h Blina | + | Q2 | pan-T identity convergence on the hub |
| 7 | CD8–CD3e–CD45 | CD3e | CD8, NALM 6h Blina | ± | Q2 | CD45 phosphatase access to TCR — kinetic-segregation test |
| 8 | CD58–CD2–CD3e | CD2 | CD8/painted-T, NALM 6h Blina | + | Q2 | CD2–LFA-3 adhesion feeding the TCR core |
| 9 | CD3e–CD2–CD48 | CD2 | CD8, NALM 6h Blina | + | Q2 | CD2–CD48 (SLAMF2) adhesion alternative ligand |
| 10 | CD8–TCRab–CD5 | TCRab | CD8, NALM 6h Blina | + | Q2 | co-receptor + tuning on the αβ receptor |
| 11 | CD162–CD3e–TCRab | CD3e | CD8, NALM 6h Blina | + | Q2 | PSGL-1 recruited to cSMAC — the spatial gate (round 4) |
| 12 | CD162–CD3e–CD8 | CD3e | CD8, NALM 6h Blina | + | Q2 | PSGL-1 ↔ Signal-1 core coupling |

### Module 2 — CD8 exclusion zone (idle LFA-1 + bulky glycocalyx)

The blinatumomab synapse **bypasses LFA-1/ICAM-1**: CD11a/CD18 stay heterodimerized but idle, anchoring an exclusion zone with CD45/CD43/CD44/B2M/HLA-ABC/CD45RA/CD50. Within-zone triplets co-cluster (+); zone-vs-core triplets exclude (−).

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 13 | CD18–CD11a–CD45 | CD11a | CD8, NALM 6h Blina | + | Q1 | idle LFA-1 heterodimer beside the phosphatase |
| 14 | CD43–CD11a–CD45 | CD11a | CD8, NALM 6h Blina | + | Q2 | leukosialin glycocalyx in the exclusion zone |
| 15 | CD44–CD11a–CD45 | CD11a | CD8, NALM 6h Blina | + | Q2 | HCAM bulk excluded with LFA-1 |
| 16 | HLA-ABC–CD11a–CD45 | CD11a | CD8, NALM 6h Blina | + | Q2 | MHC-I consolidates into exclusion zone (round 3) |
| 17 | HLA-ABC–CD11a–B2M | CD11a | CD8, NALM 6h Blina | + | Q1 | MHC-I/β2m structural unit, zone-anchored |
| 18 | CD45RA–CD11a–CD45 | CD11a | CD8, NALM 6h Blina | + | Q2 | naïve isoform with the phosphatase bulk |
| 19 | CD50–CD11a–CD18 | CD11a | CD8, NALM 6h Blina | + | Q2 | ICAM-3 with idle LFA-1 (pSMAC sub-panel) |
| 20 | CD11a–CD18–CD29 | CD18 | CD8, NALM 6h Blina | + | Q2 | integrin β-chain clustering (pSMAC sub-panel) |
| 21 | CD49D–CD29–CD18 | CD29 | CD8, NALM 6h Blina | + | Q2 | VLA-4 with LFA-1 β2 — adhesion module |
| 22 | CD43–CD44–CD45 | CD44 | CD8, NALM 6h Blina | + | Q1 | glycocalyx exclusion-core triad |
| 23 | CD19–CD11a–CD45 | CD11a | painted-T, NALM 6h Blina | − | Q2 | trogocytosed CD19 anti-correlated with exclusion zone |
| 24 | CD3e–CD11a–CD45 | CD11a | CD8, NALM 6h Blina | − | Q2 | Signal-1 core excluded from idle-LFA-1 zone |
| 25 | CD54–CD11a–CD18 | CD11a | CD8, NALM 6h Blina | − | Q2 | ICAM-1 ligand *not* engaging idle LFA-1 — bypass signature |

### Module 8 — CD8 productive activation / proliferation (NALM-specific)

These rise **only** on HT/NALM (CD25/CD71 spike absent on healthy-B engagement). Co-cluster on engaged effectors.

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 26 | CD71–CD25–CD69 | CD25 | CD8, NALM 48h Blina | + | Q1 | productive-engagement proliferation triad |
| 27 | CD71–CD25–CD137 | CD25 | CD8, NALM Blina | + | Q2 | IL-2Rα + TfR1 anchored to 4-1BB costim |
| 28 | CD69–CD25–HLA-DR | CD25 | CD8, NALM 48h Blina | + | Q2 | early + late activation convergence |
| 29 | CD38–CD25–CD71 | CD25 | CD8, NALM 48h Blina | + | Q2 | metabolic activation cluster |
| 30 | CD95–CD69–HLA-DR | CD69 | CD8, NALM 48h Blina | + | Q2 | Fas with activation markers (AICD priming) |
| 31 | CD134–CD25–CD71 | CD25 | CD8, NALM Blina | + | Q2 | costim leg anchored to proliferation module |
| 32 | CD278–CD38–CD25 | CD38 | CD8, NALM Blina | + | Q2 | ICOS with metabolic activation |

### Module 9 — CD4 helper synapse (MHC-II–anchored)

Hub is MHC-II (HLA-DR-DP-DQ) or CD3e. Cluster-1 is 2.1× tighter on NALM than healthy-B (round 5); kinetics reversed vs CD8 (peaks 6h on HT/NALM).

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 33 | CD3e–HLA-DR-DP-DQ–CD4 | HLA-DR-DP-DQ | CD4, NALM 6h Blina | + | Q1 | helper cSMAC organizer — defining axis |
| 34 | CD4–HLA-DR-DP-DQ–CD44 | HLA-DR-DP-DQ | CD4, NALM 6h Blina | + | Q2 | top HT/NALM CD4 coloc pair (round 5) on MHC-II |
| 35 | CD45–HLA-DR-DP-DQ–CD3e | HLA-DR-DP-DQ | CD4, NALM 6h Blina | + | Q2 | helper cSMAC core (+0.217 round 5) |
| 36 | CD44–HLA-DR-DP-DQ–CD45 | HLA-DR-DP-DQ | CD4, NALM 6h Blina | + | Q2 | structural cluster-1 on MHC-II |
| 37 | CD2–HLA-DR-DP-DQ–CD3e | HLA-DR-DP-DQ | CD4, NALM 6h Blina | + | Q2 | CD2 adhesion into helper cSMAC |
| 38 | CD278–CD3e–HLA-DR-DP-DQ | CD3e | CD4, NALM 6h Blina | + | Q2 | ICOS canonical CD4 costim on TCR |
| 39 | CD154–CD3e–HLA-DR-DP-DQ | CD3e | CD4, NALM 6h Blina | + | Q2 | CD40L at the helper synapse (ceiling on HT/NALM) |
| 40 | CD11a–CD45–CD3e | CD45 | CD4, NALM 6h Blina | + | Q2 | cluster-1 structural axis (shared with CD8) |
| 41 | CD2–CD150–CD3e | CD150 | CD4, NALM 6h Blina | + | Q2 | SLAM-family helper costim adhesion |
| 42 | CD4–CD3e–CD5 | CD3e | CD4, NALM 6h Blina | + | Q2 | co-receptor + tuning on helper TCR |
| 43 | CD154–CD134–CD3e | CD134 | CD4, NALM 6h Blina | + | Q2 | CD40L + OX40 costim convergence (CD4) |
| 44 | CD278–CD134–CD137 | CD134 | CD4, NALM 6h Blina | + | Q1 | ICOS/OX40/4-1BB coordinated costim (CD4) |

---

## B. CD8 FRUSTRATED ACTIVATION — costim ⊥ exhaustion on one membrane

### Module 4 — OX40 frustrated-costimulation hub

OX40 (CD134) coclusters simultaneously with effector (CX3CR1), brake (VISTA), raft (CD90), purinergic (CD73). The signature claim is **±**: accelerators and brakes on the *same* hub. The wedge **lower tail** distinguishes a segregated (frustrated) hub.

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 45 | VISTA–CD134–CX3CR1 | CD134 | CD8, NALM 6h Blina | ± | Q3 | brake + effector on costim hub — the frustration motif |
| 46 | VISTA–CD134–CD90 | CD134 | CD8, NALM 6h Blina | + | Q2 | early checkpoint + lipid raft on OX40 |
| 47 | VISTA–CD134–CD73 | CD134 | CD8, NALM 6h Blina | + | Q2 | checkpoint + purinergic brake co-hub |
| 48 | CD73–CD134–CD90 | CD134 | CD8, NALM 6h Blina | + | Q2 | purinergic raft platform on OX40 |
| 49 | CX3CR1–CD134–CD90 | CD134 | CD8, NALM 6h Blina | + | Q2 | terminal-effector + raft on costim |
| 50 | CD137–CD134–CD27 | CD134 | CD8, NALM 6h Blina | + | Q1 | coordinated 6h costim triad (round 3) |
| 51 | CD137–CD134–CD226 | CD134 | CD8, NALM 6h Blina | + | Q2 | 4-1BB + DNAM-1 activating costim |
| 52 | CD27–CD134–CD28 | CD134 | CD8, NALM 6h Blina | + | Q2 | TNFR + CD28 costim convergence |
| 53 | VISTA–CD134–TIGIT | CD134 | CD8, NALM 6h Blina | ± | Q3 | costim hub bridging two distinct brakes |
| 54 | CD73–CD134–CD39 | CD134 | CD8, NALM 6h Blina | + | Q2 | purinergic scar (CD39/CD73) on costim hub |
| 55 | CX3CR1–CD134–GPR56 | CD134 | CD8, NALM 6h Blina | + | Q2 | terminal-effector identity on OX40 |
| 56 | CD137–CD134–CD357 | CD134 | CD8, NALM 6h Blina | + | Q2 | GITR with the costim cluster |

### Module 5 — TIGIT exhaustion / checkpoint zone

TIGIT coclusters with PD-1, PD-L2, 4-1BB, CX3CR1, CD94. Tandem/redundant checkpoints — motivates combination blockade.

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 57 | CD279–TIGIT–CD273 | TIGIT | CD8, NALM 6h Blina | + | Q1 | PD-1 + PD-L2 on the TIGIT hub |
| 58 | CD279–TIGIT–CD137 | TIGIT | CD8, NALM 6h Blina | ± | Q3 | checkpoint hub next to 4-1BB accelerator |
| 59 | CD273–TIGIT–CD94 | TIGIT | CD8, NALM 6h Blina | + | Q2 | PD-L2 + NK-inhibitory on TIGIT |
| 60 | CX3CR1–TIGIT–CD94 | TIGIT | CD8, NALM 6h Blina | + | Q2 | effector + NK-brake convergence |
| 61 | CD279–TIGIT–CD366 | TIGIT | CD8, NALM 6h Blina | + | Q1 | three checkpoints (PD-1/TIM-3/TIGIT) at one spot |
| 62 | CD366–TIGIT–VISTA | TIGIT | CD8, NALM 6h Blina | + | Q2 | TIM-3 + VISTA redundant brake layer |
| 63 | CD244–TIGIT–CD279 | TIGIT | CD8, NALM 6h Blina | + | Q2 | 2B4 inhibitory SLAM with PD-1 |
| 64 | CD152–TIGIT–CD279 | TIGIT | CD8, NALM 6h Blina | + | Q2 | CTLA-4 + PD-1 on TIGIT hub |
| 65 | CD305–TIGIT–CD279 | TIGIT | CD8, NALM 6h Blina | + | Q2 | LAIR-1 ITIM with PD-1 |
| 66 | CD273–CD279–CD274 | CD279 | CD8, NALM 6h Blina | + | Q1 | PD-1 engaging both ligands (cis/trogo) |
| 67 | VISTA–CD279–CD366 | CD279 | CD8, NALM 6h Blina | + | Q2 | PD-1 hub with VISTA + TIM-3 |
| 68 | CD159a–TIGIT–CD94 | TIGIT | CD8, NALM 6h Blina | + | Q2 | NKG2A/CD94 inhibitory with TIGIT |

### Module 6 — Temporal switch of OX40's brake (6h VISTA → 48h KIR)

The diagnostic is a **condition contrast**: triplet 45/53 (VISTA arm) strong at 6h; the KIR triplets below strong at 48h.

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 69 | CD158a–CD134–CD94 | CD134 | CD8, NALM 48h Blina | + | Q2 | KIR brake replaces VISTA on OX40 at 48h |
| 70 | CD158b–CD134–CD94 | CD134 | CD8, NALM 48h Blina | + | Q2 | KIR2DL2/3 + CD94 late brake |
| 71 | CD159a–CD134–CD94 | CD134 | CD8, NALM 48h Blina | + | Q2 | NKG2A late inhibitory on costim |
| 72 | CD158a–CD134–CD158b | CD134 | CD8, NALM 48h Blina | + | Q2 | dual-KIR acquisition on OX40 |
| 73 | VISTA–CD134–CD158a | CD134 | CD8, NALM 6h→48h Blina | ± | Q3 | the switch itself: VISTA and KIR competing for the hub |
| 74 | KLRG1–CD158a–CD94 | CD94 | CD8, NALM 48h Blina | + | Q1 | terminal NK-like senescence module |

### Module 7 — NK-like cytotoxic program (CD8, NALM-specific)

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 75 | CD56–CD94–CD314 | CD94 | CD8, NALM 48h Blina | + | Q1 | NK-like cytotoxic acquisition |
| 76 | CD158a–CD94–CD161 | CD94 | CD8, NALM 48h Blina | + | Q2 | KIR + NKR-P1 inhibitory NK panel |
| 77 | GPR56–CX3CR1–CD56 | CX3CR1 | CD8, NALM 48h Blina | + | Q2 | terminal-effector NK-like identity |
| 78 | CD314–CD244–CD226 | CD244 | CD8, NALM 48h Blina | + | Q2 | NKG2D + DNAM-1 activating receptors |
| 79 | CD159a–CD94–CD159c | CD94 | CD8, NALM 48h Blina | + | Q1 | NKG2A/NKG2C heterodimer partners on CD94 |
| 80 | CD56–CD314–GPR56 | CD314 | CD8, NALM 48h Blina | + | Q2 | effector NK-like cytotoxic triad |

---

## C. B-CELL SYNAPSE — TWO ARCHITECTURES

### Module 10 — Type 1a: NALM-6 "killing-target" cluster

NALM-6 forms an HLA-I + ICAM-2 + LFA-3 + CD19 cluster (round 3 slide 26) — the BiTE-bound, MHC-I-anchored target architecture. Hub CD19 or HLA-ABC.

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 81 | HLA-ABC–CD19–CD58 | CD19 | B-NALM, 6h Blina | + | Q1 | killing-target core (MHC-I + LFA-3 + CD19) |
| 82 | HLA-ABC–CD19–CD102 | CD19 | B-NALM, 6h Blina | + | Q2 | MHC-I + ICAM-2 on the BiTE site |
| 83 | CD58–CD19–CD102 | CD19 | B-NALM, 6h Blina | + | Q2 | LFA-3 + ICAM-2 adhesion to engaging T |
| 84 | CD54–CD19–CD58 | CD19 | B-NALM, 6h Blina | + | Q2 | ICAM-1 + LFA-3 adhesion docking |
| 85 | FMC63–CD19–HLA-ABC | CD19 | B-NALM, 6h Blina | + | Q2 | BiTE scFv landmark on MHC-I-anchored CD19 |
| 86 | FMC63–CD19–CD58 | CD19 | B-NALM, 6h Blina | + | Q2 | BiTE-occupied CD19 at the adhesion contact |
| 87 | HLA-ABC–CD19–B2M | CD19 | B-NALM, 6h Blina | + | Q1 | MHC-I structural unit at target contact |
| 88 | HLA-DR-DP-DQ–CD19–HLA-ABC | CD19 | B-NALM, 6h Blina | + | Q2 | MHC-I/II co-display on the target |
| 89 | CD102–HLA-ABC–CD58 | HLA-ABC | B-NALM, 6h Blina | + | Q1 | adhesion-on-MHC-I killing-target triad |
| 90 | CD22–CD19–CD58 | CD19 | B-NALM, 6h Blina | ± | Q2 | does the BCR brake reach the killing-target cluster? |

### Module 11 — Type 1b: NALM-6 late inhibitory brake-raft

A second, *adjacent* NALM cluster (round 4): PD-L1 + PD-L2 + LAIR-1 + PSGL-1 + CD22 + CD80, organized around PSGL-1, assembled too late to win the kinetic race. Hub PSGL-1 (CD162) or PD-L1.

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 91 | CD274–CD162–CD305 | CD162 | B-NALM, 6h Blina | + | Q1 | PSGL-1 organizing PD-L1 + LAIR-1 brake |
| 92 | CD22–CD162–CD305 | CD162 | B-NALM, 6h Blina | + | Q2 | PSGL-1 ↔ BCR-brake ↔ LAIR-1 ITIM coupling |
| 93 | CD19–CD162–CD22 | CD162 | B-NALM, 6h Blina | + | Q2 | PSGL-1 recruited to BiTE-bound CD19 + CD22 |
| 94 | CD72–CD162–CD22 | CD162 | B-NALM, 6h Blina | + | Q2 | inhibitory FcR-like + Siglec on PSGL-1 |
| 95 | CD273–CD274–CD305 | CD274 | B-NALM, 6h Blina | + | Q1 | tandem PD-L1/PD-L2 + LAIR-1 (slide 5) |
| 96 | CD273–CD274–CD80 | CD274 | B-NALM, 6h Blina | + | Q2 | CD80 in raft (can bind PD-L1 in trans) |
| 97 | CD305–CD274–CD22 | CD274 | B-NALM, 6h Blina | + | Q2 | PD-L1 hub with two ITIM brakes |
| 98 | CD102–CD274–CD305 | CD274 | B-NALM, 6h Blina | + | Q2 | PD-L1 + ICAM-2 + LAIR-1 raft (slide 5) |
| 99 | CD162–CD22–CD72 | CD22 | B-NALM, 6h Blina | + | Q1 | ITIM brake triad on the Siglec hub |
| 100 | CD66b–CD274–CD273 | CD274 | B-NALM, 6h Blina | + | Q2 | CEACAM8 NALM-specific induction in the raft |
| 101 | CD274–CD162–CD19 | CD162 | B-NALM, 6h Blina | + | Q3 | does the brake-raft cooperate with the BiTE site beyond pairwise? |
| 102 | CD273–CD162–CD80 | CD162 | B-NALM, 6h Blina | + | Q2 | PD-L2 + costim ligand on PSGL-1 hub |

### Module 12 — Type 2: healthy-B CD22-anchored tolerogenic synapse

Healthy primary B recruit CD22 + CD50 + CD54 + CD72, plus MHC-I/II, CD45, Fc receptors — a *constitutive, pre-formed* tolerogenic synapse (round 3). Hub CD22. This is the contrast partner to Modules 10–11.

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 103 | CD50–CD22–CD54 | CD22 | B-healthy, all | + | Q1 | tolerogenic-synapse core (ICAM-3 + CD22 + ICAM-1) |
| 104 | CD50–CD22–CD72 | CD22 | B-healthy, all | + | Q2 | adhesion + inhibitory FcR on the Siglec hub |
| 105 | HLA-ABC–CD22–HLA-DR-DP-DQ | CD22 | B-healthy, Blina | + | Q2 | MHC-I/II recruited to CD22 (round 3 top coloc) |
| 106 | HLA-ABC–CD22–CD45RA | CD22 | B-healthy, Blina | + | Q2 | top HT/HB coloc CD22–HLA-ABC, CD22–CD45RA |
| 107 | CD45–CD22–HLA-DR | CD22 | B-healthy, Blina | + | Q2 | phosphatase + MHC-II on tolerogenic hub |
| 108 | CD32–CD22–CD35 | CD22 | B-healthy, all | + | Q1 | Fc-receptor brake cluster on CD22 |
| 109 | B2M–CD22–HLA-ABC | CD22 | B-healthy, Blina | + | Q2 | MHC-I structural unit recruited to CD22 |
| 110 | CD44–CD22–CD45 | CD22 | B-healthy, Blina | + | Q2 | glycocalyx + phosphatase on CD22 |
| 111 | CD35–CD22–CD45 | CD22 | B-healthy, Blina | + | Q2 | complement-receptor brake with phosphatase |
| 112 | CD54–CD22–CD58 | CD22 | B-healthy, all | + | Q2 | adhesion docking on the tolerogenic hub |
| 113 | CD180–CD22–CD32 | CD22 | B-healthy, all | + | Q2 | RP105 + FcγRIIb innate-brake on CD22 |
| 114 | CD72–CD22–CD32 | CD22 | B-healthy, all | + | Q1 | triple ITIM/inhibitory brake core |

### Module 13 — Breg purinergic brake (healthy B, constitutive)

CD39/CD73 adenosine axis — the chemical brake that complements the CD22 structural brake; low on NALM-6.

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 115 | CD39–CD73–CD22 | CD73 | B-healthy, all | + | Q1 | purinergic + tolerogenic-structural brake unite |
| 116 | CD39–CD73–CD45 | CD73 | B-healthy, all | + | Q2 | adenosine axis on the phosphatase platform |
| 117 | CD200–CD73–CD39 | CD73 | B-healthy, all | + | Q2 | CD200 inhibitory ligand with adenosine axis |
| 118 | CD24–CD73–CD39 | CD73 | B-healthy, all | + | Q2 | CD24 (Breg/Siglec-G context) with purinergic brake |
| 119 | CD39–CD22–CD72 | CD22 | B-healthy, all | + | Q2 | chemical + structural brake convergence |
| 120 | CD73–CD22–CD305 | CD22 | B-healthy/NALM | + | Q2 | purinergic + LAIR-1 ITIM on Siglec hub |

### Module 14 — B-cell APC Signal-2 hub (healthy B, productive)

CD80/CD86/CD40 on the MHC-II platform — the costim a competent APC brings. NALM-6 is CD80⁻ (universal deficit); healthy B paradoxically *induce* CD86/CD40 under Blina (CRS-bystander substrate). Hub HLA-DR-DP-DQ.

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 121 | CD80–HLA-DR-DP-DQ–CD86 | HLA-DR-DP-DQ | B-healthy, all | + | Q1 | canonical Signal-2 + MHC-II co-display |
| 122 | CD40–HLA-DR-DP-DQ–CD86 | HLA-DR-DP-DQ | B-healthy, 48h Blina | + | Q2 | helper-licensed APC maturation (paradoxical surge) |
| 123 | CD80–HLA-DR-DP-DQ–CD54 | HLA-DR-DP-DQ | B-healthy, all | + | Q2 | costim + ICAM-1 docking on MHC-II |
| 124 | CD58–HLA-DR-DP-DQ–CD54 | HLA-DR-DP-DQ | B-healthy, all | + | Q2 | LFA-3 + ICAM-1 adhesion at MHC-II contact |
| 125 | CD40–HLA-DR-DP-DQ–CD20 | HLA-DR-DP-DQ | B-healthy, all | + | Q2 | mature-B identity anchoring APC platform |
| 126 | CD86–HLA-DR-DP-DQ–CD21 | HLA-DR-DP-DQ | B-healthy, all | + | Q2 | costim with complement-receptor maturity |
| 127 | CD40–CD86–CD80 | CD86 | B-healthy, 48h Blina | + | Q1 | coordinated costim-ligand triad |
| 128 | CD40–HLA-DR-DP-DQ–CD274 | HLA-DR-DP-DQ | B-healthy, 48h Blina | ± | Q3 | licensing vs PD-L1 brake on the same APC platform |

---

## D. T–B INTERACTION — trogocytosis & the synthetic bridge

### Module 3 — Trogocytosis tetraspanin microdomain (the marquee finding)

Trogocytosed B-cell membrane is reassembled on the CD8 surface as a **tetraspanin-enriched microdomain (TEM)**. Top observed coloc: CD82–CD82, CD37–CD82, CD20–CD82, CD53–CD82, CD20–CD37, CD81–CD82. Hub = a tetraspanin (CD82/CD81/CD37/CD53). These are the strongest **Q3 (cooperativity)** candidates — the claim is a genuine three-body TEM, not two strong edges.

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 129 | CD20–CD82–CD37 | CD82 | painted-T (CD8), Blina | + | Q3 | trogocytosed B markers on the tetraspanin hub |
| 130 | CD20–CD82–CD53 | CD82 | painted-T (CD8), Blina | + | Q2 | CD53–CD82 observed; B identity in TEM |
| 131 | CD20–CD82–CD81 | CD82 | painted-T (CD8), Blina | + | Q2 | CD81–CD82 web carrying CD20 |
| 132 | HLA-DR–CD82–CD37 | CD82 | painted-T (CD8), Blina | + | Q3 | MHC-II trogocytosed into the TEM |
| 133 | CD22–CD82–CD37 | CD82 | painted-T (CD8), Blina | + | Q2 | trogocytosed BCR brake in TEM |
| 134 | CD19–CD81–CD9 | CD81 | painted-T (CD8), Blina | + | Q2 | core BiTE-target marker in the tetraspanin web |
| 135 | CD19–CD82–CD37 | CD82 | painted-T (CD8), Blina | + | Q3 | CD19 trogocytosed onto CD8 TEM |
| 136 | CD37–CD82–CD53 | CD82 | painted-T (CD8), Blina | + | Q1 | the tetraspanin web itself (TEM scaffold) |
| 137 | CD81–CD82–CD53 | CD82 | painted-T (CD8), Blina | + | Q1 | tetraspanin-only scaffold triad |
| 138 | CD9–CD81–CD63 | CD81 | painted-T (CD8), Blina | + | Q2 | TEM scaffold (CD9/CD81/CD63) |
| 139 | CD37–CD81–CD9 | CD81 | painted-T (CD8), Blina | + | Q2 | tetraspanin web cross-links |
| 140 | CD19–CD37–HLA-DR | CD37 | painted-T (CD8), Blina | + | Q1 | trogocytosed B-patch core on CD8 |
| 141 | CD19–CD20–CD22 | CD20 | painted-T (CD8), Blina | + | Q1 | trogocytosed B-identity triad on the T cell |
| 142 | CD80–CD82–CD37 | CD82 | painted-T (CD8), Blina | + | Q2 | trogocytosed costim ligand in TEM |
| 143 | IgM–CD82–CD37 | CD82 | painted-T (CD8), Blina | + | Q2 | trogocytosed surface Ig in TEM |
| 144 | CD10–CD81–CD19 | CD81 | painted-T (CD8), Blina | + | Q2 | trogocytosed pre-B marker (NALM signature) |
| 145 | CD24–CD81–CD19 | CD81 | painted-T (CD8), Blina | + | Q2 | trogocytosed CD24 with CD19 in TEM |
| 146 | CD273–CD82–CD37 | CD82 | painted-T (CD8), Blina | + | Q2 | trogocytosed PD-L2 — checkpoint painted onto T |
| 147 | CD138–CD81–CD19 | CD81 | painted-T (CD8), Blina | + | Q2 | trogocytosed CD138 in TEM |
| 148 | CD21–CD81–CD19 | CD81 | painted-T (CD8), Blina | + | Q2 | trogocytosed CD21 with CD19 |
| 149 | HLA-DR-DP-DQ–CD82–CD81 | CD82 | painted-T (CD8), Blina | + | Q2 | MHC-II in the tetraspanin web |
| 150 | CD43–CD162–CD44 | CD162 | painted-T (CD8), Blina | + | Q2 | glycocalyx PSGL-1 module (CD43/CD162 self-coloc seen) |
| 151 | CD20–CD37–CD81 | CD37 | painted-T (CD8), Blina | + | Q3 | B-tetraspanin CD37 hub carrying CD20 + CD81 |
| 152 | CD22–CD81–CD19 | CD81 | painted-T (CD8), Blina | + | Q2 | trogocytosed CD22 with CD19 on tetraspanin |
| 153 | HLA-DR–CD81–CD53 | CD81 | painted-T (CD8), Blina | + | Q2 | MHC-II on alternate tetraspanin hub |

### Module 15 — The synthetic BiTE bridge & receptor–ligand pairs (on painted-T)

In PNA each **cell is its own graph**, so a true cross-membrane contact is not a single-graph triplet. The measurable proxy is co-localization on the **trogocytosis-painted T cell**, where B ligands and their T receptors land together. Read these as "did the receptor and its trogocytosed ligand co-organize on the T surface." (Caveat noted per row.)

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 154 | CD19–CD3e–FMC63 | CD3e | painted-T, Blina | + | Q1 | BiTE bridge landmark: CD19 target + CD3 + scFv |
| 155 | CD19–CD3e–CD8 | CD3e | painted-T, Blina | + | Q2 | trogocytosed target at the Signal-1 core |
| 156 | FMC63–CD19–CD3e | CD19 | painted-T, Blina | + | Q2 | scFv-occupied CD19 hub bridging to CD3 |
| 157 | CD58–CD2–CD3e | CD2 | painted-T, Blina | + | Q2 | LFA-3 (B) ↔ CD2 (T) ↔ TCR coupling |
| 158 | CD80–CD28–CD3e | CD28 | painted-T, Blina | + | Q2 | Signal-2 ligand meeting CD28 at the TCR (if delivered) |
| 159 | CD274–CD279–CD3e | CD279 | painted-T, Blina | + | Q2 | trogocytosed PD-L1 engaging PD-1 near TCR |
| 160 | CD273–CD279–TIGIT | CD279 | painted-T, Blina | + | Q2 | PD-L2 + PD-1 + TIGIT inhibitory convergence |
| 161 | CD40–CD154–CD3e | CD154 | CD4/painted-T, Blina | + | Q2 | CD40–CD40L bridge near the helper TCR |
| 162 | CD54–CD11a–CD18 | CD11a | painted-T, Blina | − | Q2 | ICAM-1 (B) vs idle LFA-1 — bypass (re-tested cross-side) |
| 163 | HLA-DR-DP-DQ–CD4–CD3e | CD4 | CD4/painted-T, Blina | + | Q2 | B MHC-II engaging CD4 co-receptor at TCR |
| 164 | CD19–CD162–CD3e | CD162 | painted-T, Blina | + | Q2 | PSGL-1 organizing trogocytosed CD19 toward TCR |
| 165 | CD22–CD3e–CD8 | CD3e | painted-T, Blina | ± | Q2 | trogocytosed inhibitory CD22 reaching Signal-1 |

---

## E. CANONICAL COMPLEXES & CONTROLS

### Module 16 — Native B-cell tetraspanin / BCR coreceptor complexes (on B, not trogocytosed)

Positive controls and intrinsic B organization — the CD19/CD21/CD81 coreceptor complex is a textbook triangle. Use to validate the wedge/triangle pipeline against known biology.

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 166 | CD19–CD81–CD21 | CD81 | B-healthy, all | + | Q1 | canonical BCR coreceptor complex — POSITIVE CONTROL |
| 167 | CD19–CD21–CD81 | CD21 | B-healthy, all | + | Q1 | same triad, alternate hub — robustness |
| 168 | CD20–CD81–CD37 | CD81 | B (intrinsic) | + | Q2 | native B tetraspanin web (vs trogocytosed Mod 3) |
| 169 | CD19–CD81–CD22 | CD81 | B-healthy, all | + | Q2 | BCR coreceptor + Siglec brake |
| 170 | CD37–CD53–CD81 | CD53 | B (intrinsic) | + | Q1 | leukocyte tetraspanin web on B |
| 171 | CD79a–CD19–CD81 | CD19 | B, all | + | Q2 | BCR signaling subunit with coreceptor |
| 172 | CD20–CD19–CD21 | CD19 | B-healthy, all | + | Q2 | mature-B surface organization |

### Module 17 — B death / BCR-survival signature (NALM-6 escape program)

NALM-6 IgM surge + BCR survival signaling; the Ig collapse under Blina is the death signature. Tracks escape vs death kinetics.

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 173 | IgM–CD79a–CD19 | CD79a | B-NALM, 48h Mock | + | Q1 | active BCR survival unit (IgM surge) |
| 174 | IgM–IgD–CD19 | CD19 | B-NALM/healthy | + | Q2 | surface-Ig isotype co-organization |
| 175 | IgM–CD20–CD22 | CD20 | B-healthy | + | Q2 | mature BCR-brake organization |
| 176 | IgM–CD79a–CD22 | CD79a | B, all | + | Q2 | BCR + ITIM brake coupling |
| 177 | IgM–CD19–CD268 | CD19 | B, all | + | Q2 | BCR with BAFF-R survival receptor |
| 178 | CD269–CD138–IgM | CD138 | B-NALM | + | Q2 | BCMA + plasma-cell + Ig survival axis |

### Module 18 — NALM-6 "deaf to environment" innate/inhibitory interface

CD32/CD35/CD37/CD72/CD180 are depressed on NALM-6 (deaf to environment) but organized on healthy B — a *contrast* module (expect co-cluster on healthy, weak/random on NALM).

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 179 | CD32–CD35–CD180 | CD35 | B-healthy vs NALM | + | Q1 | innate-sensing cluster present on healthy, absent on NALM |
| 180 | CD37–CD32–CD72 | CD32 | B-healthy vs NALM | + | Q2 | FcR + tetraspanin + inhibitory FcR |
| 181 | CD180–CD32–CD35 | CD32 | B-healthy vs NALM | + | Q2 | RP105/TLR-accessory innate interface |
| 182 | CD72–CD180–CD32 | CD180 | B-healthy vs NALM | + | Q2 | inhibitory innate-brake triad |

### Module 19 — Adhesion / structural-identity consolidation (CD8, NALM stress)

Under leukemic stress every structural marker tightens onto CD11a (exclusion-zone densification). Condition contrast NALM vs healthy.

| # | Triplet (A–B–C) | Hub | Cell / condition | Sign | Q | Rationale |
|---|---|---|---|---|---|---|
| 183 | CD45–CD11a–HLA-ABC | CD11a | CD8, NALM vs healthy | + | Q2 | structural tightening (CD45 0.88 vs 0.50 NALM) |
| 184 | CD3e–CD11a–CD45 | CD11a | CD8, NALM vs healthy | + | Q2 | core-marker consolidation onto exclusion zone |
| 185 | CD43–CD11a–CD44 | CD11a | CD8, NALM Blina | + | Q2 | glycocalyx consolidation |
| 186 | CD49D–CD29–CD49D | CD29 | CD8, NALM Blina | + | Q2 | VLA-4 polarization (self-anchored) |
| 187 | CD103–CD29–CD11a | CD29 | CD8, NALM Blina | + | Q2 | CD103 integrin with the adhesion module |
| 188 | CD50–CD54–CD58 | CD54 | CD8/B contact | + | Q1 | ICAM-3/ICAM-1/LFA-3 adhesion triad |

---

## F. CROSS-CUTTING CONTRASTS (same triplet, different system — the high-value comparisons)

These re-use triplets above but the *finding* is the **between-condition delta**, scored per cell then compared with Mann–Whitney / Kruskal–Wallis blocking on `cell_system` (methods §4.4). Run each as paired contrasts.

| # | Triplet | Contrast | Expectation |
|---|---|---|---|
| 189 | TCRab–CD3e–CD8 (#1) | NALM 6h Blina vs healthy-B 6h Blina | cSMAC tight only on NALM (spatial gate) |
| 190 | VISTA–CD134–CX3CR1 (#45) | 6h Blina vs 48h Blina (NALM CD8) | frustrated module strong at 6h |
| 191 | CD158a–CD134–CD94 (#69) | 48h vs 6h (NALM CD8) | KIR brake replaces VISTA at 48h |
| 192 | CD50–CD22–CD54 (#103) | healthy-B vs NALM (Mock) | tolerogenic synapse constitutive on healthy only |
| 193 | CD274–CD162–CD305 (#91) | NALM 6h vs 48h Blina | brake-raft assembles 6h, collapses 48h (PD-L2 loss) |
| 194 | CD39–CD73–CD22 (#115) | healthy-B vs NALM (all) | Breg purinergic brake healthy-only |
| 195 | CD20–CD82–CD37 (#129) | NALM 6h vs 48h; healthy-B 6h | trogocytosis painted-T frequency tracks killing |
| 196 | CD80–HLA-DR-DP-DQ–CD86 (#121) | healthy-B 48h Blina vs Mock | paradoxical costim induction (CRS substrate) |
| 197 | CD3e–HLA-DR-DP-DQ–CD4 (#33) | CD4 6h vs 48h (NALM) | helper synapse peaks 6h (reversed vs CD8) |
| 198 | HLA-ABC–CD19–CD58 (#81) | NALM vs healthy-B (Blina) | killing-target architecture NALM-specific |
| 199 | CD19–CD81–CD21 (#166) | any B (positive control) | should fire everywhere — pipeline validation |
| 200 | CD11a–CD18–CD45 (#13) | NALM vs healthy CD8 (Blina) | exclusion-zone densification under leukemic stress |
| 201 | CD162–CD3e–TCRab (#11) | NALM vs healthy CD8 (Blina) | PSGL-1 gating of cSMAC NALM-specific |
| 202 | CD22–CD3e–CD8 (#165) | patient-B vs healthy-B | is patient B tolerogenic like healthy or target-like? |
| 203 | CD39–CD73–CD45 (#116) | CD8 NALM 48h vs healthy | persistent CD39 metabolic scar |
| 204 | CD273–CD274–CD305 (#95) | NALM patient-T vs healthy-T | does T-cell source change brake-raft assembly |
| 205 | CD80–HLA-DR-DP-DQ–CD86 (#121) | patient-B vs healthy-B | patient-B APC competence |
| 206 | CD20–CD82–CD37 (#129) | patient-T vs healthy-T on NALM | trogocytosis efficiency by T-cell source |
| 207 | CD56–CD94–CD314 (#75) | NALM 48h patient-T vs healthy-T | NK-like reprogramming by T source |
| 208 | CD3e–HLA-DR-DP-DQ–CD4 (#33) | patient-B vs healthy-B (CD4) | helper synapse competence by target |

### Optional polarization (self-pair) anchors

Self-pairs (A=A) measure single-marker polarization; as triplet hubs they ask "does this polarized cap also gather X and Y." A few worth scoring:

| # | Triplet | Hub | Cell / condition | Rationale |
|---|---|---|---|---|
| 209 | CD82–CD82–CD20 | CD82 | painted-T | tetraspanin self-polarization gathering trogocytosed CD20 |
| 210 | CD162–CD162–CD22 | CD162 | B-NALM | PSGL-1 cap organizing the brake-raft |
| 211 | CD22–CD22–CD50 | CD22 | B-healthy | CD22 cap nucleating the tolerogenic synapse |
| 212 | CD11a–CD11a–CD45 | CD11a | CD8 | LFA-1 cap defining the exclusion zone |
| 213 | CD134–CD134–VISTA | CD134 | CD8 | OX40 cap with its brake |
| 214 | CD19–CD19–FMC63 | CD19 | B-NALM | BiTE-occupancy polarization landmark |
| 215 | TIGIT–TIGIT–CD279 | TIGIT | CD8 | checkpoint cap |
| 216 | CD39–CD39–CD73 | CD39 | B-healthy | purinergic cap |
| 217 | CD3e–CD3e–CD8 | CD3e | CD8 | TCR microcluster polarization |
| 218 | HLA-DR-DP-DQ–HLA-DR-DP-DQ–CD80 | HLA-DR-DP-DQ | B-healthy | MHC-II platform polarization with costim |

---

## Recommended run order (high → low leverage)

1. **Positive controls first** (#166–167, #199): CD19–CD81–CD21 must fire — validates the wedge/triangle code on known biology before you trust novel hits.
2. **The marquee novel claim** (Module 3, #129–153): tetraspanin-TEM trogocytosis as **Q3 cooperativity** — this is the publishable single-molecule trogocytosis signature; run with the pairwise-preserving null to separate genuine three-body TEM from "two strong edges."
3. **The two B-synapse architectures head-to-head** (Modules 10–12 + contrasts #192, #193, #198): the killing-target vs tolerogenic vs brake-raft distinction is the project's central selective-killing mechanism.
4. **Frustrated activation** (Modules 4–5, contrast #190): OX40/TIGIT ± hubs — use the wedge **lower tail** to nail segregation vs coincidence.
5. **Temporal switch** (Module 6, #191): condition contrast, not a single-cell claim.
6. **Everything else** as confirmatory.

**Stats reminders** (from `threeway_colocalization_methods.md`): score wedge `W_{ABC}` and triangle `T_{ABC}` per cell on the **same** label-shuffles that give the pairwise z's; use **Q = zᵀΣ⁻¹z** with Σ from the shuffles for the omnibus, **T_min** where you claim "both arms real" (AND), and the **pairwise-preserving** null only for Q3. Feed **raw** join-count z (not arcsinh/tanh variants). Aggregate per-cell, block on `cell_system`, BH/BY-FDR, and don't over-read 48h NALM CD8 (n≈25).

*Generated 2026-06-29. Markers per `marker_panel_reference.md`; verify B2M and CD150 against `adata.var_names` before running.*
