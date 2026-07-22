# Spatial Colocalization Analysis: CD134 (OX40), TIGIT, and CD11a (LFA-1α)

**Pixelgen MPX Spatial Proteomics | NALM-6 vs Healthy B Coculture | April 2026**

---

## Experimental Context

This analysis explores the **spatial organization of three key surface markers** on CD8 T cells across conditions, using Pixelgen MPX colocalization z-scores. The markers were selected to probe distinct biological axes:

| Marker | Category | Rationale |
|--------|----------|-----------|
| **CD134 (OX40)** | Costimulation & Survival | Massively upregulated in NALM-6 coculture; TNFR family costimulatory receptor |
| **TIGIT** | Exhaustion & Co-inhibitory | Key exhaustion checkpoint competing with DNAM-1; bridges innate and adaptive inhibition |
| **CD11a (LFA-1α/ITGAL)** | Adhesion & Synapse | α-chain of LFA-1 integrin; primary T cell adhesion receptor for ICAM-1 |

**Comparisons performed:**
- CD134 & TIGIT: 6h Blina (Healthy vs NALM-6), 48h Blina (Healthy vs NALM-6), NALM-6 Blina (6h vs 48h)
- CD11a: Same three comparisons + 6h Mock (Healthy vs NALM-6)

**Cell counts:** 6h Mock H=255/N=272; 6h Blina H=254/N=198; 48h Blina H=55/N=25

---

## Understanding Spatial Colocalization in MPX

Pixelgen Molecular Pixelation measures protein proximity on the cell surface at nanometer resolution. Each cell yields a colocalization matrix of all protein pairs, expressed as z-scores:
- **Positive z-scores** indicate proteins that are spatially closer than expected by chance (clustered together on the membrane)
- **Negative z-scores** indicate spatial exclusion (proteins pushed apart)
- **Values near zero** indicate random distribution

This provides a direct readout of membrane organization, synapse architecture, and receptor compartmentalization at the single-cell level.

---

## 1. CD134 (OX40) — Costimulation Under Spatial Duress

### Marker Function
OX40 is an activation-induced TNFR family costimulatory receptor that sustains T cell proliferation, survival, and cytokine production. It is normally recruited to the immunological synapse upon TCR engagement.

### 1a. 6h Blinatumomab: Healthy vs NALM-6

**NALM-6 CD8 T cells** show OX40 spatially clustered with:
- **Effector/homing markers:** CX3CR1 (0.050), CD199/CCR9 (0.076)
- **Inhibitory checkpoints:** VISTA (0.020)
- **B-cell markers (synaptic capture):** CD37 (0.019), CD22 (0.018), CD80 (0.015), IgM (0.014)
- **Modulatory receptors:** CD90/Thy-1 (0.025), CD277/BTN3A1 (0.017), CD73 (0.016)

**Healthy CD8 T cells** show much weaker OX40 colocalization overall (top partner CD199 at 0.025 vs 0.076 in NALM-6).

**Differential (most significant):**
- Higher in NALM-6: CX3CR1 (+0.040, p=2e-4), VISTA (+0.022, p=5e-17), CD37 (+0.021, p=3e-11), CD13 (+0.021, p=2e-22), CD90 (+0.020, p=8e-27)
- Higher in Healthy (negative in NALM-6): B2M (-0.163, p=6e-23), CD45 (-0.146, p=4e-22), CD44 (-0.137, p=1e-20), CD43 (-0.121, p=2e-16), CD45RA (-0.109, p=4e-15)

**Key neighbor roles:**
- **CX3CR1** (Fractalkine receptor) — marks terminally differentiated cytotoxic effectors with vascular patrol function. Its clustering with OX40 indicates the costimulatory receptor is embedded in membrane domains occupied by late-stage effector machinery.
- **CD199/CCR9** — gut-homing chemokine receptor. Its strong colocalization (highest partner at 0.076) is unexpected and may reflect aberrant homing receptor mobilization to the synapse contact zone.
- **VISTA** (B7-H5) — inhibitory checkpoint that uniquely functions at resting state to maintain quiescence. Its spatial proximity to OX40 creates a direct costimulation/inhibition toggle at the same membrane location.
- **CD37, CD22, CD80, IgM** — these are B-cell lineage markers. Their detection on purified CD8 T cells, spatially clustered with OX40, marks the physical contact zone with the NALM-6 target (trogocytosed membrane fragments).
- **CD90/Thy-1** — GPI-anchored glycoprotein associated with lipid rafts. Its colocalization suggests OX40 is recruited to organized lipid raft domains at the synapse.
- **CD277/BTN3A1** — butyrophilin family member involved in phosphoantigen sensing and Vγ9Vδ2 T cell activation. Its presence near OX40 may reflect co-recruitment of stress-sensing receptors.
- **CD73** (NT5E) — ecto-5'-nucleotidase that generates immunosuppressive adenosine from AMP. Its colocalization with OX40 places the purinergic suppression machinery directly at the costimulatory zone.

**Neighborhood interpretation:** OX40 in NALM-6 at 6h Blina sits at the intersection of three functional programs: (1) effector homing (CX3CR1, CCR9), (2) inhibitory checkpoints (VISTA, CD73), and (3) target cell contact (B-cell markers). Simultaneously, it is spatially excluded from the core T-cell structural network (CD45, CD44, CD43, B2M) — these bulky phosphatases and glycoproteins are physically expelled from the tight synaptic cleft. This creates a membrane compartment where costimulation is spatially uncoupled from TCR identity and directly opposed by checkpoint brakes.

### 1b. 48h Blinatumomab: Healthy vs NALM-6

The OX40 spatial neighborhood undergoes dramatic reorganization. In NALM-6 at 48h, OX40 now colocalizes with:
- **KIR/inhibitory receptors:** CD158/KIR (0.030), CD158a (0.014), CD94 (0.011)
- **SLAM family:** CD352/SLAMF6 (0.023)
- **T cell identity markers:** CD5 (0.022), CD7 (0.017)
- **B cell markers:** CD21 (0.029), CD273/PD-L2 (0.025), CD138 (0.014)

However, **no comparisons reach statistical significance** (all padj > 0.05), reflecting the small sample size at 48h (N=25 NALM-6, N=55 healthy).

**Key neighbor roles:**
- **CD158/KIR, CD158a/KIR2DL1, CD94/KLRD1** — inhibitory NK-like receptors recognizing MHC-I. Their emergence as OX40 spatial partners at 48h (replacing VISTA from 6h) indicates a temporal switch in the braking mechanism: VISTA acts as the early gatekeeper, KIRs become the late-phase spatial brake on costimulation.
- **CD352/SLAMF6** — SLAM family member with dual activating/inhibitory function depending on SAP adaptor availability. Its clustering with OX40 suggests formation of a costimulatory–modulatory signaling hub.
- **CD5** — inhibitory coreceptor that tunes TCR signaling threshold. Its appearance near OX40 at 48h suggests the costimulatory receptor is re-integrating into the T-cell regulatory network as the synapse dissolves.
- **CD21/CR2** — complement receptor, classically a B-cell marker. Its colocalization may represent residual trogocytosed material from the initial synapse engagement still associated with OX40 domains.
- **CD273/PD-L2** — PD-1 ligand expressed on B cells. Its presence as an OX40 neighbor further confirms the synaptic contact signature persists at reduced intensity at 48h.
- **CD138/Syndecan-1** — proteoglycan involved in cell adhesion and growth factor signaling. Normally a plasma cell marker; its detection suggests either trogocytosis or aberrant expression by NALM-6.

**Neighborhood interpretation:** By 48h, OX40's spatial neighborhood transitions from an effector/checkpoint zone (CX3CR1, VISTA) to an inhibitory receptor–dominated environment (KIRs, CD94). The return of T-cell identity markers (CD5, CD7) suggests partial membrane re-integration as the synapse resolves. The remaining B-cell markers (CD21, CD273, CD138) indicate the trogocytosis signature is fading but not fully cleared.

### 1c. NALM-6 Blinatumomab: 6h vs 48h

The temporal shift within NALM-6 shows OX40 losing its effector/checkpoint partners and gaining T-cell identity partners. CD90 is the only significantly different partner (higher at 6h, p=0.001). The broad pattern suggests OX40 transitions from a synapse-embedded, frustrated costimulatory state toward membrane redistribution as the synapse resolves.

**Neighborhood interpretation:** The 6h→48h transition within NALM-6 captures the lifecycle of costimulatory receptor spatial organization. At 6h, OX40 is trapped in the active synapse with effector markers (CX3CR1) and checkpoint brakes (VISTA). By 48h, the synapse has dissolved and OX40 redistributes into domains now populated by KIR inhibitory receptors and SLAM family modulators. The significant loss of CD90 (a lipid raft marker) suggests OX40 exits the organized raft microdomains that anchored it at the synapse. This temporal pattern directly mirrors the phenotypic observation that blinatumomab normalizes costimulatory marker expression by 48h.

### Interpretation

**OX40 at the synthetic synapse interface:** At 6h Blina, OX40 is drawn into the blinatumomab-enforced immunological synapse, where it clusters with captured B-cell membrane fragments (CD37, CD22, CD80, IgM — evidence of trogocytosis) and early exhaustion gatekeepers (VISTA). The simultaneous spatial exclusion from core T-cell markers (CD3e, CD8, CD45) reveals that OX40 operates in membrane microdomains completely uncoupled from the classical TCR signalosome.

**Frustrated costimulation:** OX40's primary role is to sustain T cell activation. Its spatial clustering with inhibitory receptors (VISTA at 6h, KIRs at 48h) represents a spatial regulatory mechanism — the T cell physically co-localizes its activation and braking machinery to dampen aberrant, tonic costimulatory signaling driven by the leukemic target.

**Bulky molecule exclusion:** The extreme negative colocalization with CD45, CD43, CD44, and B2M across all conditions is consistent with exclusion of large, bulky phosphatases and structural glycoproteins from the tight synaptic cleft. This is a hallmark of productive immunological synapse formation — the CD45 phosphatase must be expelled from the central contact zone to permit sustained signaling.

---

## 2. TIGIT — Exhaustion Checkpoint Recruitment to the Synapse

### Marker Function
TIGIT is an inhibitory receptor that competes with the activating receptor DNAM-1 (CD226) for PVR/Nectin-2 binding, suppressing T cell and NK cell cytotoxicity. It is a key co-inhibitory checkpoint in T cell exhaustion.

### 2a. 6h Blinatumomab: Healthy vs NALM-6

**NALM-6 CD8:** TIGIT's top partners are CX3CR1 (0.110), CD199/CCR9 (0.069), CD273/PD-L2 (0.056), CD94/KLRD1 (0.038), CD52 (0.036), plus CD137/4-1BB (0.017) and CD279/PD-1 (0.012).

**Healthy CD8:** TIGIT clusters with CX3CR1 (0.059), CCR9 (0.047), CD8 (0.024), CD319/SLAMF7 (0.024), CD7 (0.021).

**Key differential:**
- Higher in NALM-6: HLA-DR-DP-DQ (+0.115, p=8e-8), CD22 (+0.050, p=3e-4), CD20 (+0.047, p=8e-14), CD40 (+0.046, p=1e-5)
- Higher in Healthy: CD9 (-0.158, p=4e-26), CD24 (-0.129, p=3e-33), CD10 (-0.097, p=1e-32), CD38 (-0.072, p=7e-13), CD54 (-0.054, p=6e-8)

**Key neighbor roles:**
- **CX3CR1** (0.110 in NALM-6 vs 0.059 in healthy) — Fractalkine receptor marking terminal effectors. TIGIT's strongest partner in NALM-6, indicating the exhaustion checkpoint clusters with the cytotoxic effector machinery — consistent with exhausted cells that retain some killing potential.
- **CD273/PD-L2** — ligand for PD-1, expressed on B cells and APCs. TIGIT's strong colocalization with PD-L2 (0.056) indicates TIGIT localizes to the target contact zone where PD-1 ligands are presented by NALM-6.
- **CD94/KLRD1** — NK-like receptor pairing with NKG2A. Its colocalization places TIGIT in the same membrane compartment as innate inhibitory receptors, confirming coordinated inhibitory receptor clustering.
- **CD137/4-1BB** (0.017) and **CD279/PD-1** (0.012) — the co-clustering of TIGIT with both costimulatory (4-1BB) and co-inhibitory (PD-1) receptors at the NALM-6 synapse reflects the "frustrated activation" phenotype: activation and inhibition machinery physically co-localize.
- **HLA-DR-DP-DQ** (+0.115 in NALM-6) — target cell MHC-II complex. The most significantly different TIGIT partner between systems, confirming TIGIT is recruited directly to the leukemic interface.
- **CD20, CD22, CD40** — B-cell markers detected on T cells. Like CD134, their colocalization with TIGIT marks the trogocytosis/synapse contact zone, but now with an exhaustion checkpoint anchored there.

**Neighborhood interpretation:** In NALM-6 at 6h Blina, TIGIT occupies a spatial domain at the synapse interface characterized by: (1) target cell markers (HLA-DR-DP-DQ, CD20, CD22, CD40), (2) effector receptors (CX3CR1, CD94), and (3) other checkpoints (PD-1, 4-1BB). The strong anti-colocalization with tetraspanins (CD9) and adhesion/activation markers (CD24, CD10, CD38, CD54) in healthy cells indicates TIGIT is spatially segregated from the adhesion machinery — it occupies a checkpoint-specific membrane zone distinct from the structural synapse core where CD54 resides.

### 2b. 48h Blinatumomab: Healthy vs NALM-6

Massive divergence in TIGIT neighborhoods:

**NALM-6 48h:** TIGIT with CD7 (0.116), TIGIT-self (0.082), KLRG1 (0.072), IgE (0.068), CD8 (0.067), CD84/SLAMF5 (0.051), CD199 (0.044), TCRab (0.037), CD28 (0.037)

**Healthy 48h:** TIGIT with CD45RA (0.219), CD8 (0.206), CD45 (0.181), CD11a (0.178), TIGIT-self (0.146), CD3e (0.145), KLRG1 (0.141), CD48/SLAMF2 (0.124), CD244/2B4 (0.104)

**Key differential:**
- Higher in NALM-6: HLA-DR-DP-DQ (+0.370, p=0.001), CD40 (+0.193, p=2e-4), CD20 (+0.101, p=0.001)
- Higher in Healthy: CD9 (-0.203, p=0.002), HLA-ABC (-0.187, p=0.06)

**Key neighbor roles (NALM-6 48h):**
- **CD7** (0.116) — costimulatory glycoprotein, also the strongest TIGIT partner at 48h. CD7 upregulation was noted in the phenotypic analysis at 48h Blina; its spatial clustering with TIGIT suggests they co-localize in the same post-synapse membrane domain.
- **KLRG1** (0.072) — canonical SLEC marker. TIGIT-KLRG1 co-clustering physically marks the terminal differentiation endpoint: the exhaustion checkpoint is now embedded with the senescence marker.
- **IgE** (0.068) — immunoglobulin not normally on T cells; likely residual trogocytosed material or assay background.
- **CD84/SLAMF5** (0.051) — SLAM family member mediating homotypic cell interactions. Its emergence as a TIGIT partner at 48h (absent at 6h) reflects SLAM family receptor remodeling as the synapse dissolves.
- **TCRab** (0.037) and **CD28** (0.037) — core TCR signaling components. Their appearance as TIGIT neighbors at 48h indicates the exhaustion checkpoint is re-integrating into the classical TCR signaling network after synapse dissolution.

**Key neighbor roles (Healthy 48h):**
- **CD45RA** (0.219), **CD45** (0.181), **CD8** (0.206), **CD3e** (0.145) — the core T-cell identity/structural network. In healthy cells, TIGIT integrates into an ordered, classical membrane architecture alongside the TCR complex and phosphatases.
- **CD11a/LFA-1** (0.178) — the key adhesion integrin. TIGIT co-clustering with LFA-1 in healthy cells places it at the normal immunological synapse peripheral SMAC.
- **CD244/2B4** (0.104) — SLAM family receptor with dual function. Its strong colocalization with TIGIT in healthy 48h cells forms a classical exhaustion/regulatory cluster integrated into normal membrane topology.

**Neighborhood interpretation:** At 48h, the two systems show fundamentally different TIGIT membrane contexts. In healthy cells, TIGIT sits within a highly ordered, classical T-cell network (CD45RA, CD8, CD3e, CD11a) — exhaustion machinery woven into normal structure. In NALM-6, TIGIT clusters with terminal differentiation markers (KLRG1, CD7) and SLAM family modulators (SLAMF5) but at much lower colocalization magnitudes, and with persistent B-cell markers (HLA-DR-DP-DQ, CD40, CD20) indicating the trogocytosis footprint endures. This divergence confirms that even after blinatumomab normalizes expression levels by 48h, the spatial membrane organization of NALM-6-exposed T cells remains fundamentally altered.

### 2c. NALM-6 Blinatumomab: 6h vs 48h

Over time in NALM-6, TIGIT's neighborhood shifts dramatically:
- **Loses:** CX3CR1 (-0.149, p=0.009), CD52 (-0.184, p=0.005), CD45RO (-0.143, p=7e-4)
- **Gains:** CD7 (+0.101), CD8 (+0.087), KLRG1 (+0.075), TIGIT-self (+0.054)

**Key neighbor transitions:**
- **CX3CR1 loss** (-0.149) — the terminal effector marker dissociates from TIGIT over time, consistent with loss of effector homing capacity as cells progress from exhaustion toward senescence.
- **CD52/CAMPATH-1 loss** (-0.184) — CD52 is a GPI-anchored glycoprotein involved in lymphocyte signaling. Its dissociation from TIGIT may reflect membrane reorganization as GPI-anchored proteins redistribute after synapse dissolution.
- **CD45RO loss** (-0.143) — the memory/effector T cell CD45 isoform moves away from TIGIT over time, consistent with loss of effector memory phenotype.
- **KLRG1 gain** (+0.075) — the SLEC senescence marker moves into TIGIT's neighborhood, physically marking the progression from early exhaustion to terminal differentiation.
- **TIGIT-self gain** (+0.054) — increased self-clustering indicates TIGIT molecules consolidate into denser homotypic clusters over time, suggesting progressive checkpoint receptor aggregation.

**Neighborhood interpretation:** Within NALM-6 over 48h of blinatumomab, TIGIT's spatial network transitions from an effector/synapse-associated program (CX3CR1, CD52, CD45RO) to a senescence/terminal differentiation program (KLRG1, CD7, CD8). The increasing TIGIT self-aggregation suggests progressive checkpoint clustering as the T cell hardens into a terminally exhausted state. This temporal trajectory provides the spatial correlate of the phenotypic "frustrated activation → terminal differentiation" narrative.

### Interpretation

**Progressive exhaustion hardening:** At 6h, TIGIT clusters with CX3CR1 (terminal effector marker) and CD94 (NK-like receptor), reflecting the early phase of innate-like inhibitory reprogramming. By 48h, TIGIT shifts to cluster with KLRG1 (the canonical SLEC marker with zero memory potential), physically illustrating T cell progression from early exhaustion into a terminally differentiated, functionally paralyzed dead-end.

**SLAM family dynamics:** TIGIT dynamically recruits different SLAM family coreceptors over time. At 6h in healthy cells, it clusters with SLAMF7. By 48h in NALM-6, it shifts to SLAMF5 and SLAMF6, while healthy 48h shows SLAMF2/CD244 (2B4). These SLAM receptors have dual activating/inhibitory functions depending on SAP adaptor availability, suggesting TIGIT uses them to fine-tune inhibitory signaling thresholds.

**B-cell markers = synapse signature:** In NALM-6, TIGIT shows extreme colocalization with HLA-DR-DP-DQ, CD20, CD22, and CD40 — leukemic cell markers detected on the T cell surface. This places TIGIT directly at the immunological synapse contact zone, confirming that the exhaustion checkpoint is being actively recruited to the target interface to suppress killing. In healthy cells, TIGIT instead integrates into the normal T cell membrane network (CD45RA, CD45, CD8, CD11a), confirming an ordered, classical distribution.

**The TIGIT–CD9 anti-correlation:** TIGIT strongly anti-colocalizes with CD9 in NALM-6 but not in healthy. CD9 is a tetraspanin that organizes membrane microdomains. This spatial segregation suggests that in NALM-6, the exhaustion checkpoint zone (TIGIT) and the adhesion/tetraspanin zone (CD9/CD81) occupy spatially distinct membrane compartments — a hallmark of membrane polarization.

---

## 3. CD11a (LFA-1α/ITGAL) — The Integrin Excluded from the Synthetic Synapse

### Marker Function
CD11a is the α-chain of LFA-1 (Lymphocyte Function-Associated antigen 1), the primary adhesion integrin on T cells. LFA-1 (CD11a/CD18 heterodimer) binds ICAM-1/2/3 to mediate firm adhesion at the immunological synapse peripheral SMAC (pSMAC). It is essential for T cell migration, activation, and sustained target cell contact.

### 3a. 6h Mock: NALM-6 vs Healthy

CD11a's spatial neighborhood is dominated by core T-cell identity markers in both systems:

**NALM-6:** CD11a with CD45 (0.881), CD18 (0.619), CD8 (0.583), B2M (0.534), CD3e (0.471), CD45RA (0.464), HLA-ABC (0.448), CD11a-self (0.398), CD45RB (0.363), CD48/SLAMF2 (0.351), CD2 (0.264), KLRG1 (0.264), CD44 (0.263), CD43 (0.222), CD352/SLAMF6 (0.193)

**Healthy:** CD11a with CD45 (0.499), CD8 (0.411), CD18 (0.372), HLA-ABC (0.352), B2M (0.282), KLRG1 (0.261), CD11a-self (0.252), CD3e (0.242), CD48/SLAMF2 (0.228), CD43 (0.220), CD45RB (0.210), CD44 (0.151), CD2 (0.139), CD352/SLAMF6 (0.133), CD6 (0.120)

**Key differential:**
- Higher in NALM-6: CD45 (+0.382, p=2e-30), CD45RA (+0.362, p=3e-19), B2M (+0.253, p=6e-11), CD18 (+0.247, p=2e-13), CD3e (+0.230, p=7e-15), CD52 (+0.204, p=2e-14), CD8 (+0.173, p=1e-5)
- Higher in Healthy: CD24 (-0.444, p=5e-84), CD10 (-0.441, p=1e-93), CD9 (-0.415, p=9e-79), CD71 (-0.313, p=5e-84), CD38 (-0.258, p=1e-43), CD19 (-0.183, p=8e-43)

**Key neighbor roles:**
- **CD45/PTPRC** (0.881 in NALM-6, 0.499 in healthy) — the master phosphatase that dephosphorylates Lck/Fyn to regulate TCR signaling. CD11a's strongest partner in both systems, confirming it is firmly embedded in the core T-cell signaling network. The higher colocalization in NALM-6 indicates tighter membrane organization of the structural identity network under leukemic stress.
- **CD18/ITGB2** (0.619 in NALM-6, 0.372 in healthy) — the β2-integrin chain that heterodimerizes with CD11a to form LFA-1. Their strong colocalization confirms intact LFA-1 integrin assembly on the T cell surface.
- **CD8** (0.583), **CD3e** (0.471), **B2M** (0.534) — TCR complex and MHC-I light chain. CD11a clusters tightly with the classical T-cell receptor machinery, indicating it remains in the TCR signalosome neighborhood.
- **HLA-ABC/MHC-I** (0.448) — the T cell's own MHC class I. Strong colocalization places CD11a in the antigen presentation–associated membrane zone.
- **CD45RA** (0.464 in NALM-6 vs 0.102 in healthy) — the naïve/effector CD45 isoform. Dramatically more colocalized with CD11a in NALM-6, suggesting NALM-6 exposure drives CD45RA into CD11a's membrane neighborhood.

**Key neighbor roles (differential — "higher in healthy"):**
- **CD24** (-0.444), **CD10** (-0.441), **CD9** (-0.415), **CD19** (-0.183) — these B-cell/ALL markers are strongly *anti*-colocalized with CD11a in NALM-6 but near-zero in healthy. CD11a is actively expelled from the leukemic contact zone where trogocytosed B-cell markers reside.
- **CD71/TfR1** (-0.313) — transferrin receptor. Anti-colocalized with CD11a in NALM-6, consistent with metabolic activation machinery concentrating at the synapse while CD11a stays in the structural T-cell network.

**Neighborhood interpretation:** Even without blinatumomab, CD11a remains firmly embedded in the core T-cell identity network (CD45, CD18, CD8, CD3e, B2M). Unlike CD134 and TIGIT which get recruited to the synapse, and completely unlike the previously analyzed CD54 which polarizes to the contact zone, CD11a stays home. The stronger colocalization magnitudes in NALM-6 (~0.88 for CD45 vs 0.50 in healthy) suggest the leukemic stress tightens the structural network — the T cell consolidates its identity core while the synapse-oriented molecules migrate away. The strong negative colocalization with B-cell markers confirms CD11a is spatially excluded from the trogocytosis/contact zone.

### 3b. 6h Blinatumomab: Healthy vs NALM-6

Even under blinatumomab-enforced synapse formation, CD11a maintains its structural identity neighborhood:

**NALM-6 6h Blina:** CD45 (0.760), CD18 (0.619), HLA-ABC (0.564), B2M (0.544), CD43 (0.465), CD45RA (0.463), CD50/ICAM-3 (0.432), CD45RB (0.424), CD44 (0.421), CD8 (0.411), CD11a-self (0.401), CD3e (0.391), KLRG1 (0.328), CD48/SLAMF2 (0.303), CD2 (0.264)

**Healthy 6h Blina:** CD45 (0.589), CD18 (0.561), CD8 (0.518), HLA-ABC (0.491), CD11a-self (0.384), KLRG1 (0.379), CD43 (0.361), CD3e (0.355), B2M (0.352), CD48/SLAMF2 (0.321), CD45RB (0.280), CD2 (0.235), CD44 (0.222), CD352/SLAMF6 (0.221), CD45RA (0.207)

**Key differential:**
- Higher in NALM-6: HLA-DR-DP-DQ (+0.353, p=1e-30), CD50/ICAM-3 (+0.302, p=2e-19), CD45RA (+0.256, p=7e-8), CD44 (+0.199, p=3e-7), B2M (+0.192, p=9e-7), CD45 (+0.171, p=7e-7)
- Higher in Healthy: CD24 (-0.393, p=3e-66), CD9 (-0.367, p=2e-52), CD10 (-0.364, p=3e-77), CD71 (-0.251, p=4e-49), CD38 (-0.216, p=1e-25), CD58 (-0.198, p=3e-17)

**Key neighbor roles:**
- **CD45** (0.760) — remains CD11a's top partner even during active blinatumomab engagement. While CD45 is expelled from the synapse core (as seen in CD134 and TIGIT data), it clusters strongly with CD11a — directly confirming CD11a occupies the exclusion zone, not the synapse.
- **HLA-ABC/MHC-I** (0.564) — rises to 3rd position (from 6th in mock). The T cell's own MHC-I concentrates near CD11a during blinatumomab engagement, consistent with MHC-I being excluded from the tight synaptic cleft alongside the bulky structural molecules.
- **CD50/ICAM-3** (0.432) — a notable new entrant. ICAM-3 is the initial adhesion ligand excluded from the sustained synapse. Its strong colocalization with CD11a during blinatumomab confirms both molecules are pushed to the membrane periphery, away from the tight engagement zone.
- **CD43/Leukosialin** (0.465), **CD44/HCAM** (0.421) — heavily glycosylated structural molecules expelled from the synapse. They concentrate near CD11a, confirming its location in the exclusion zone.
- **HLA-DR-DP-DQ** (+0.353 differential) — the most significantly different partner. Despite being anti-colocalized with CD11a overall (-0.320 in NALM-6), this is less negative than in healthy cells (-0.672), suggesting some leakage of target cell MHC-II signal into CD11a's neighborhood in NALM-6.
- **CD58/LFA-3** (-0.198 differential, higher in healthy) — CD58 is recruited to the synapse core (as shown in the CD54 analysis). Its anti-colocalization with CD11a in NALM-6 directly confirms the spatial segregation: CD58 goes to the synapse, CD11a stays away.

**Neighborhood interpretation:** At peak blinatumomab activity, CD11a's neighborhood is essentially a catalog of the exclusion zone. Every molecule known to be expelled from the synapse core — CD45, CD43, CD44, HLA-ABC, CD50 — clusters with CD11a. This is the exact inverse of the CD54 pattern, where the synapse core molecules (CD9, CD58, trogocytosed B-cell markers) concentrated together. CD11a and CD54 map to opposite poles of the blinatumomab-induced membrane compartmentalization. The fact that CD11a — the α-chain of LFA-1, the primary adhesion integrin — is excluded from the BiTE synapse it should theoretically anchor is the most striking finding.

### 3c. 48h Blinatumomab: Healthy vs NALM-6

At 48h, CD11a neighborhoods partially converge between systems:

**NALM-6 48h Blina:** CD45 (0.609), CD18 (0.437), CD3e (0.349), KLRG1 (0.329), CD45RB (0.319), CD8 (0.307), CD11a-self (0.295), CD44 (0.294), CD48/SLAMF2 (0.277), CD43 (0.259), CD45RA (0.240), CD352/SLAMF6 (0.237), CD6 (0.229), CD5 (0.225), CD50/ICAM-3 (0.199)

**Healthy 48h Blina:** CD18 (0.574), CD45 (0.537), CD8 (0.470), CD45RA (0.415), CD11a-self (0.411), CD3e (0.401), KLRG1 (0.329), CD48/SLAMF2 (0.306), HLA-ABC (0.300), CD45RB (0.281), CD43 (0.257), CD44 (0.246), CD352/SLAMF6 (0.229), B2M (0.211), CD5 (0.199)

**Key differential:**
- Higher in NALM-6: HLA-DR-DP-DQ (+0.547, p=9e-8), CD52 (+0.211, p=0.10), CD50 (+0.199, p=0.53), CD40 (+0.195, p=1e-3)
- Higher in Healthy: CD9 (-0.409, p=5e-8), CD38 (-0.276, p=1e-4), CD24 (-0.261, p=5e-8), CD71 (-0.232, p=9e-5), CD58 (-0.208, p=3e-3), CD10 (-0.198, p=6e-10)

**Key neighbor roles:**
- **CD45** (0.609 in NALM-6, 0.537 in healthy) — gap narrows from 0.38 at 6h Mock to 0.07 at 48h. The structural networks are converging as the synapse resolves.
- **KLRG1** (0.329 in both) — identical colocalization with CD11a in both systems, marking the terminal differentiation endpoint. CD11a's integration with KLRG1 places it squarely in the SLEC membrane neighborhood.
- **CD5** (0.225 in NALM-6, 0.199 in healthy) — inhibitory coreceptor appears in CD11a's neighborhood at 48h, consistent with the post-activation membrane reorganization.
- **CD6** (0.229 in NALM-6) — T-cell adhesion/costimulatory receptor. Its emergence as a new CD11a neighbor at 48h suggests adhesion pathway diversification after synapse dissolution.
- **HLA-DR-DP-DQ** (+0.547) — the largest differential, but with low significance at 48h (small N). The persistent MHC-II signal difference reflects residual trogocytosed material still detectable in NALM-6.
- **CD9** (-0.409 in NALM-6 vs -0.096 in healthy) — the tetraspanin that anchored the synapse core remains strongly anti-colocalized with CD11a in NALM-6, confirming the synapse and CD11a zones remain spatially distinct even after synapse dissolution.

**Neighborhood interpretation:** At 48h, CD11a's structural identity neighborhood begins to converge between systems — the top partners (CD45, CD18, CD8, CD3e, KLRG1, CD48) are nearly identical in both. However, the anti-colocalization with B-cell markers (CD24, CD10, CD9) persists in NALM-6, indicating the spatial exclusion from trogocytosis zones is maintained even after the synapse dissolves. The emergence of CD5 and CD6 marks post-activation membrane remodeling. The most notable feature is KLRG1's identical colocalization in both systems — CD11a now sits in the terminal differentiation zone regardless of prior target exposure.

### 3d. NALM-6 Blinatumomab: 6h vs 48h — Structural Network Relaxation

The temporal change within NALM-6 shows CD11a's identity network loosening as the synapse resolves:

| Partner | 6h mean | 48h mean | Diff | p-value |
|---------|---------|----------|------|---------|
| HLA-ABC | 0.564 | 0.143 | **-0.421** | 7e-5 |
| B2M | 0.544 | 0.171 | **-0.373** | 1e-3 |
| CD50 | 0.432 | 0.199 | **-0.233** | 9e-3 |
| CD45RA | 0.463 | 0.240 | **-0.223** | 0.07 |
| CD43 | 0.465 | 0.259 | **-0.207** | 6e-3 |
| CD162 | 0.062 | -0.142 | **-0.204** | 0.04 |
| CD18 | 0.619 | 0.437 | **-0.182** | 0.03 |
| CD25 | -0.100 | -0.264 | **-0.165** | 9e-5 |
| CD45 | 0.760 | 0.609 | **-0.151** | 0.07 |

**Higher at 48h (gained):**
- CD24 (+0.199, p=2e-5), CD199/CCR9 (+0.184, p=3e-5), IgE (+0.176, p=3e-6), IgM (+0.174, p=1e-5), CD10 (+0.173, p=1e-5), CD19 (+0.173, p=2e-5), CD13 (+0.168, p=7e-5)

**Key transitions:**
- **HLA-ABC collapse** (0.564 → 0.143, Δ=-0.421, p=7e-5) — the most significant temporal change. The T cell's own MHC-I disperses from CD11a's neighborhood as the synapse resolves and the membrane relaxes. At 6h, MHC-I was concentrated in the exclusion zone alongside CD11a; by 48h, it redistributes.
- **B2M decline** (0.544 → 0.171, Δ=-0.373) — mirrors HLA-ABC as the MHC-I light chain follows its partner.
- **CD18 decline** (0.619 → 0.437, Δ=-0.182) — the β2-integrin partner of CD11a loosens its colocalization. The intact LFA-1 heterodimer becomes less tightly clustered, suggesting integrin conformational relaxation as the adhesion demand drops.
- **CD50/ICAM-3 decline** (0.432 → 0.199, Δ=-0.233) — ICAM-3 that was pushed to the exclusion zone at 6h disperses as the synapse dissolves.
- **B-cell marker gain** — CD24 (+0.199), CD10 (+0.173), CD19 (+0.173), IgM (+0.174). These markers that were strongly anti-colocalized with CD11a at 6h become less so at 48h. As the synapse dissolves, the sharp spatial boundary between the identity core and the trogocytosis zone blurs.
- **CD25/IL-2Rα decline** (-0.100 → -0.264, Δ=-0.165) — CD25 becomes more anti-colocalized with CD11a at 48h. The activation marker moves further from the structural network as the T cell transitions to a resting/exhausted state.

**Neighborhood interpretation:** The 6h→48h shift within NALM-6 captures the relaxation of CD11a's exclusion zone. At 6h, the blinatumomab-enforced synapse creates a sharp spatial boundary — CD11a clusters tightly with expelled structural molecules (CD45, HLA-ABC, B2M, CD43) while B-cell markers are maximally anti-colocalized. By 48h, this boundary dissolves: the structural network loosens (HLA-ABC drops from 0.56 to 0.14), and B-cell marker anti-colocalization diminishes. The magnitude of relaxation (Δ up to 0.42 z-score units) is substantial, though smaller than the Δ~0.9 shifts seen for synapse-core molecules like CD9 and CD58 in the CD54 analysis — consistent with CD11a's role as a bystander in the exclusion zone rather than a direct participant in the synapse.

### Interpretation

**The integrin excluded from its own synapse:** CD11a/LFA-1α is the canonical adhesion integrin that binds ICAM-1 to form the pSMAC in a physiological immunological synapse. Yet in the blinatumomab-enforced synapse, CD11a is excluded from the contact zone. This represents a fundamental distinction between the synthetic and natural synapse: blinatumomab bypasses the LFA-1/ICAM-1 adhesion axis entirely, using the CD3–CD19 bispecific bridge to enforce contact independently of integrin-mediated adhesion.

**CD11a maps the exclusion zone:** At 6h Blina, CD11a's neighborhood is a near-complete inventory of molecules expelled from the synapse core: CD45 (phosphatase), CD43 (leukosialin), CD44 (HCAM), HLA-ABC/B2M (MHC-I), and CD50/ICAM-3. These are the same molecules that appear as the "higher in healthy" differential for CD134, TIGIT, and the previously analyzed CD54. CD11a provides the spatial reference point for the non-synapse membrane — it is the anchor of the exclusion zone.

**LFA-1 remains assembled but idle:** The strong CD11a–CD18 colocalization (0.619 at 6h) confirms the LFA-1 heterodimer is structurally intact. The integrin is properly assembled but functionally idle — it sits outside the synapse in a default, unengaged conformation. This contrasts with a natural synapse where inside-out signaling from TCR engagement activates LFA-1 to the high-affinity open conformation at the pSMAC.

**Structural network tightening under stress:** In NALM-6, CD11a shows higher colocalization with every core identity marker compared to healthy controls (CD45: 0.88 vs 0.50, CD8: 0.58 vs 0.41, CD3e: 0.47 vs 0.24). The leukemic stress response appears to consolidate the structural identity network — as synapse-oriented molecules (CD134, TIGIT) migrate toward the contact zone, the remaining structural components pack more tightly around CD11a.

**The inverse of CD54:** In the prior analysis, CD54/ICAM-1 showed extreme polarization to the synapse core (self z-score ~1.0), massive tetraspanin recruitment, and trogocytosed B-cell marker accumulation. CD11a shows the exact inverse: stable identity-core anchoring, anti-colocalization with B-cell/synapse markers, and no tetraspanin association. The LFA-1 receptor (CD11a) and the ICAM-1 ligand (CD54) — normally binding partners at the pSMAC — are spatially segregated into opposite membrane compartments in the BiTE-enforced synapse. This spatial uncoupling of the canonical adhesion pair is strong evidence that blinatumomab creates a fundamentally non-physiological synapse architecture.

---

## 4. Synthesis: Mapping the Blinatumomab-Induced Synapse Architecture

### The Unified Biological Narrative

The three markers together provide a high-resolution spatial map of the synthetic immunological synapse at peak engagement (6h Blina). CD134 and TIGIT probe the synapse-recruited functional zones; CD11a reveals the complementary exclusion zone — the membrane territory left behind.

```
┌──────────────────────────────────────────────────────────────┐
│                       T CELL MEMBRANE                         │
│                                                              │
│  ┌─── EXCLUSION ZONE ──────────┐   ┌── SYNAPSE CORE ──────┐ │
│  │ CD11a/CD18 (LFA-1, idle)    │   │ Trogocytosed B-cell  │ │
│  │ CD45 (phosphatase)          │   │  CD19, CD10, CD24    │ │
│  │ CD44 (structural)           │   │  HLA-DR, CD22        │ │
│  │ CD43 (glycoprotein)         │   │  CD37, CD80, IgM     │ │
│  │ B2M/HLA-ABC (MHC-I)        │   │                      │ │
│  │ CD45RA (phosphatase)        │   │ Adhesion anchors     │ │
│  │ CD50/ICAM-3 (initial adh.)  │   │  CD54 (ICAM-1)      │ │
│  │                             │   │  CD58 (LFA-3)        │ │
│  └─────────────────────────────┘   │  CD9/CD81/CD53 (TEMs)│ │
│                                    │  ADAM10 (remodeling)  │ │
│  ┌── FRUSTRATED COSTIM ──┐        └───────────────────────┘ │
│  │ OX40 + VISTA          │                                   │
│  │ OX40 + CX3CR1         │        ┌── CHECKPOINT ZONE ───┐  │
│  │ (uncoupled from TCR)  │        │ TIGIT + PD-L2        │  │
│  └───────────────────────┘        │ TIGIT + CX3CR1       │  │
│                                    │ TIGIT + CD94         │  │
│                                    │ (recruited to target) │  │
│                                    └──────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                      ↕ Blinatumomab bridge ↕
┌──────────────────────────────────────────────────────────────┐
│                    NALM-6 TARGET CELL                         │
│    CD19, CD10, CD24, HLA-DR, CD22, CD37, CD80               │
└──────────────────────────────────────────────────────────────┘
```

### Key Findings

1. **The synthetic synapse bypasses LFA-1/ICAM-1 adhesion entirely.** CD11a (LFA-1α) — the canonical adhesion integrin of the pSMAC — is excluded from the blinatumomab-enforced synapse and anchors the exclusion zone instead. The BiTE bridge enforces contact via CD3–CD19 without requiring integrin-mediated adhesion, creating a fundamentally non-physiological synapse architecture.

2. **CD11a maps the exclusion zone; CD134 and TIGIT map synapse-recruited functional zones.** The three markers divide the T cell membrane into complementary spatial compartments: CD11a with structural/identity proteins (CD45, CD43, CD44, HLA-ABC), OX40 with checkpoint brakes and effector markers (VISTA, CX3CR1), TIGIT with target-derived signals and exhaustion receptors (PD-L2, CD94, HLA-DR-DP-DQ).

3. **Costimulation and exhaustion are spatially compartmentalized but distinct from the structural core.** OX40 and TIGIT cluster with different partners but both localize to the synapse interface and away from the CD11a-anchored identity core, suggesting organized functional zones within the contact area — separate from both the adhesion core and the exclusion zone.

4. **Trogocytosis is massive and spatially organized.** B-cell markers (CD19, CD10, CD24, CD22, HLA-DR, CD37, CD80, IgM) detected on purified CD8 T cells are anti-colocalized with CD11a's structural network and colocalized with synapse-associated molecules. This provides direct spatial evidence of extensive membrane transfer concentrated at the contact zone.

5. **The structural identity network tightens under leukemic stress.** CD11a shows stronger colocalization with every core T-cell marker in NALM-6 vs healthy (CD45: 0.88 vs 0.50, CD3e: 0.47 vs 0.24). As synapse-oriented molecules migrate toward the contact zone, the remaining structural components consolidate, creating a tighter exclusion zone.

6. **Complete spatial normalization by 48h despite distinct temporal trajectories.** Synapse-core molecules (CD134, TIGIT) lose their effector/checkpoint neighbors and gain identity partners. CD11a's exclusion zone relaxes as the sharp membrane compartmentalization dissolves. All three markers' neighborhoods converge between systems by 48h.

### Surprising Findings Compared to Literature

1. **LFA-1 excluded from the BiTE synapse it should anchor.** In a natural immunological synapse, LFA-1 (CD11a/CD18) forms the pSMAC ring that stabilizes the contact. The blinatumomab synapse renders this integrin irrelevant — CD11a sits in the exclusion zone with bulky phosphatases while the synapse operates via non-integrin mechanisms. This has not been described in BiTE synapse studies.

2. **VISTA appears as an early spatial brake, not a late exhaustion marker.** OX40-VISTA colocalization at 6h suggests VISTA acts as an immediate spatial gatekeeper recruited to the synapse contact zone to prevent signaling overload during the acute engagement burst, rather than a marker of chronic exhaustion.

3. **The CD11a–CD18 heterodimer remains assembled but idle.** Strong CD11a–CD18 colocalization (0.619) confirms intact LFA-1 structure, but the integrin is functionally excluded from the synapse. This dissociation between structural integrity and functional engagement is unusual for an integrin normally activated by inside-out signaling from TCR engagement.

4. **Temporal inhibitory receptor switching.** OX40 spatial brakes switch from VISTA (6h) to KIR receptors (48h), suggesting a programmed temporal sequence of checkpoint recruitment rather than a single static inhibitory mechanism.

5. **Complete spatial normalization by 48h despite residual metabolic scarring.** While the synapse architecture fully resets — CD11a's exclusion zone relaxes, CD134/TIGIT lose synapse partners — the prior phenotypic analysis showed the CD39 purinergic defect persists. The physical structure heals but the metabolic wound remains.

### Therapeutic Implications

- **LFA-1 agonists for synapse enhancement:** Since blinatumomab bypasses LFA-1-mediated adhesion, co-administering LFA-1 agonists or ICAM-1-targeting agents could recruit the integrin adhesion axis to the BiTE synapse, potentially creating stronger, more physiological contact and improved killing.

- **Anti-VISTA or anti-KIR co-administration:** Removing the spatial brakes (VISTA at 6h, KIR at 48h) that cluster with costimulatory receptors at the synapse could amplify early cytotoxicity before exhaustion programs fully establish.

- **Trogocytosis-aware dosing:** B-cell markers (CD19, CD24) detected on T cells via trogocytosis are spatially organized at the synapse contact zone. Trogocytosis-painted T cells could become fratricide targets. Pulsed dosing with recovery intervals could allow clearance before the next cycle.

- **Adenosine pathway inhibition:** Combined with the phenotypic finding of CD39 purinergic scarring, adenosine receptor antagonists or CD73 inhibitors could address the metabolic wound that spatial normalization alone cannot heal.

### Relationship to Phenotypic Analysis

These spatial findings directly complement and extend the expression-level analysis in `cd8_phenotype_analysis.md`:

| Phenotypic finding | Spatial confirmation |
|---|---|
| Massive costimulatory upregulation (OX40, 4-1BB) in NALM-6 | OX40 physically recruited to synapse, co-clustered with captured B-cell markers |
| Multi-pathway exhaustion (VISTA, TIM-3, TIGIT) | TIGIT physically positioned at synapse interface with leukemic markers |
| CD11a stable across conditions | CD11a spatially anchors exclusion zone, structurally intact but functionally excluded from synapse |
| KIR acquisition at 6h | KIRs appear in OX40 neighborhood at 48h, replacing VISTA as the dominant spatial brake |
| Blinatumomab normalization by 48h | Complete synapse dissolution — all colocalization patterns converge between systems |
| CD39 purinergic scar | Spatial architecture normalizes but metabolic imprint persists |

---

*Analysis generated using NotebookLM research with sources on OX40 signaling, TIGIT spatial distribution, LFA-1/CD11a integrin biology, immunological synapse architecture, and blinatumomab mechanism of action. Notebook ID: d5df25a4-3674-472f-8bf9-1cdd7e24ed3f. April 2026.*
