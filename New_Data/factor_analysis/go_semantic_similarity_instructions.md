# Claude Code Agent Prompt: GO Functional Embeddings for PixelGen Surface Markers

## Context

We have a panel of ~160 surface protein markers from a PixelGen spatial proteomics experiment. We want to create protein embeddings where functionally similar proteins have similar embeddings — based purely on Gene Ontology biological process annotations, not sequence similarity. These embeddings will initialize the decoder of a semantic VAE.

## Steps

### Step 1: Download GO data files

Download these two files:
- **GO ontology (OBO format)**: `http://purl.obolibrary.org/obo/go/go-basic.obo`
- **Human GO annotations (GAF format)**: `http://current.geneontology.org/annotations/goa_human.gaf.gz`

### Step 2: Install dependencies

```bash
pip install goatools numpy scipy scikit-learn pandas
```

### Step 3: Parse GO ontology and annotations

Use `goatools` to:
1. Load the OBO file into a GO DAG.
2. Parse the GAF file to get gene → GO term associations.
3. Filter to **biological process (BP)** namespace only — this is the ontology branch that captures functional programs (signaling, activation, adhesion, etc.) which is what we care about for the semantic VAE. Optionally also include **molecular function (MF)** as a second pass to see if it changes results.
4. Filter annotations to **experimental evidence codes only**: EXP, IDA, IPI, IMP, IGI, IEP, HTP, HDA, HMP, HGI, HEP. Exclude IEA (electronic annotation) — these are noisy and will dilute the signal.

### Step 4: Map markers to GO terms

Use this gene symbol list to look up GO annotations. These are the same HGNC symbols from the marker-to-gene mapping:

```python
MARKER_TO_GENE = {
    "HLA-ABC": ["HLA-A", "HLA-B", "HLA-C"],
    "B2M": ["B2M"],
    "CD11b": ["ITGAM"],
    "CD11c": ["ITGAX"],
    "CD18": ["ITGB2"],
    "CD82": ["CD82"],
    "CD8": ["CD8A"],
    "TCRab": ["TRAC", "TRBC1"],
    "HLA-DR": ["HLA-DRA"],
    "CD45": ["PTPRC"],
    "CD14": ["CD14"],
    "CD16": ["FCGR3A"],
    "CD19": ["CD19"],
    "CD45RB": ["PTPRC"],
    "CD44": ["CD44"],
    "CD52": ["CD52"],
    "CD59": ["CD59"],
    "CD45RA": ["PTPRC"],
    "CD36": ["CD36"],
    "CD2": ["CD2"],
    "CD29": ["ITGB1"],
    "CD5": ["CD5"],
    "CD162": ["SELPLG"],
    "CD27": ["CD27"],
    "CD35": ["CR1"],
    "CD26": ["DPP4"],
    "CD49D": ["ITGA4"],
    "CD37": ["CD37"],
    "CD41": ["ITGA2B"],
    "CD20": ["MS4A1"],
    "CD64": ["FCGR1A"],
    "CD22": ["CD22"],
    "CD274": ["CD274"],
    "CD328": ["SIGLEC7"],
    "CD25": ["IL2RA"],
    "CD279": ["PDCD1"],
    "CD335": ["NCR1"],
    "CD152": ["CTLA4"],
    "CD86": ["CD86"],
    "CD13": ["ANPEP"],
    "CD156c": ["ADAM10"],
    "CD158a": ["KIR2DL1"],
    "CD158b": ["KIR2DL3"],
    "CD159a": ["KLRC1"],
    "CD159c": ["KLRC2"],
    "CD163": ["CD163"],
    "CD169": ["SIGLEC1"],
    "CD1a": ["CD1A"],
    "CD1c": ["CD1C"],
    "CD206": ["MRC1"],
    "CD226": ["CD226"],
    "CD273": ["PDCD1LG2"],
    "CD28": ["CD28"],
    "CD352": ["SLAMF6"],
    "CD45RO": ["PTPRC"],
    "CD49e": ["ITGA5"],
    "CD56": ["NCAM1"],
    "CD79a": ["CD79A"],
    "CD80": ["CD80"],
    "CD81": ["CD81"],
    "CD85j": ["LILRB1"],
    "CD93": ["CD93"],
    "IgD": ["IGHD"],
    "IgE": ["IGHE"],
    "IgM": ["IGHM"],
    "KLRG1": ["KLRG1"],
    "NKp80": ["KLRF1"],
    "TCRgd": ["TRGC1", "TRDC"],
    "CD10": ["MME"],
    "CD103": ["ITGAE"],
    "CD199": ["CCR9"],
    "CD1b": ["CD1B"],
    "CD21": ["CR2"],
    "CD231": ["TSPAN7"],
    "CD277": ["BTN3A1"],
    "CD305": ["LAIR1"],
    "CD371": ["CLEC12A"],
    "CD89": ["FCAR"],
    "CD90": ["THY1"],
    "CD70": ["CD70"],
    "GPR56": ["ADGRG1"],
    "HLA-DQ": ["HLA-DQA1", "HLA-DQB1"],
    "TCRVd2": ["TRDV2"],
    "HLA-DR-DP-DQ": ["HLA-DRA", "HLA-DPA1", "HLA-DQA1"],
    "Siglec-9": ["SIGLEC9"],
    "TCRva7.2": ["TRAV1-2"],
    "CD95": ["FAS"],
    "TCRVg9": ["TRGV9"],
    "CD127": ["IL7R"],
    "CD141": ["THBD"],
    "CD31": ["PECAM1"],
    "CD6": ["CD6"],
    "CD57": ["B3GAT1"],
    "CD73": ["NT5E"],
    "CD366": ["HAVCR2"],
    "CD357": ["TNFRSF18"],
    "CD193": ["CCR3"],
    "CD319": ["SLAMF7"],
    "TIGIT": ["TIGIT"],
    "CD134": ["TNFRSF4"],
    "CD102": ["ICAM2"],
    "CD123": ["IL3RA"],
    "CD150": ["SLAMF1"],
    "CD154": ["CD40LG"],
    "CD158": ["KIR2DL1"],
    "CD161": ["KLRB1"],
    "CD180": ["CD180"],
    "CD191": ["CCR1"],
    "CD192": ["CCR2"],
    "CD1d": ["CD1D"],
    "CD200": ["CD200"],
    "CD229": ["LY9"],
    "CD24": ["CD24"],
    "CD244": ["CD244"],
    "CD268": ["TNFRSF13C"],
    "CD278": ["ICOS"],
    "CD32": ["FCGR2A"],
    "CD33": ["CD33"],
    "CD337": ["NCR3"],
    "CD38": ["CD38"],
    "CD39": ["ENTPD1"],
    "CD40": ["CD40"],
    "CD48": ["CD48"],
    "CD50": ["ICAM3"],
    "CD55": ["CD55"],
    "CD58": ["CD58"],
    "CD62P": ["SELP"],
    "CD69": ["CD69"],
    "CD72": ["CD72"],
    "CD84": ["CD84"],
    "CD9": ["CD9"],
    "CD94": ["KLRD1"],
    "TCRVB5": ["TRBV5-1"],
    "CD3e": ["CD3E"],
    "CD4": ["CD4"],
    "CD11a": ["ITGAL"],
    "CD43": ["SPN"],
    "CD7": ["CD7"],
    "CD53": ["CD53"],
    "CD302": ["CD302"],
    "VISTA": ["VSIR"],
    "CD269": ["TNFRSF17"],
    "CD138": ["SDC1"],
    "CD137": ["TNFRSF9"],
    "CD66b": ["CEACAM8"],
    "CX3CR1": ["CX3CR1"],
    "CD326": ["EPCAM"],
    "CD209": ["CD209"],
    "CD34": ["CD34"],
    "CD369": ["CLEC7A"],
    "CD54": ["ICAM1"],
    "CD71": ["TFRC"],
    "CD47": ["CD47"],
    "CD117": ["KIT"],
    "CD314": ["KLRK1"],
    "FMC63": [],
    "mIgG1": [],
    "mIgG2a": [],
    "mIgG2b": [],
}
```

For multi-gene markers (e.g. HLA-ABC → HLA-A, HLA-B, HLA-C), take the **union** of GO terms across all constituent genes. For CD45 isoforms (CD45, CD45RA, CD45RB, CD45RO) they all map to PTPRC, so they share the same GO terms — that's fine, semantic similarity will handle it.

Skip markers with empty gene lists (isotype controls, FMC63) and markers whose genes have no experimental GO-BP annotations.

### Step 5: Compute information content (IC) for each GO term

The information content of a GO term measures its specificity:

```
IC(term) = -log2( freq(term) / freq(root) )
```

Where `freq(term)` is the number of human genes annotated to that term **or any of its descendants** (propagate annotations up the DAG). More specific terms have higher IC. Use the full set of human gene annotations from the GAF file as the background corpus, not just our 160 markers.

### Step 6: Compute pairwise Resnik semantic similarity

For each pair of markers (i, j), compute the **Best-Match Average (BMA) Resnik similarity**:

1. Let `GO_i` and `GO_j` be the sets of BP GO terms for markers i and j.
2. For each term `t_a` in `GO_i`, find its best match in `GO_j`:
   ```
   best_match(t_a, GO_j) = max over t_b in GO_j of IC( MICA(t_a, t_b) )
   ```
   where `MICA(t_a, t_b)` is the Most Informative Common Ancestor — the common ancestor of `t_a` and `t_b` with the highest IC.
3. BMA similarity:
   ```
   sim(i,j) = 0.5 * [ mean over t_a in GO_i of best_match(t_a, GO_j) 
                     + mean over t_b in GO_j of best_match(t_b, GO_i) ]
   ```

This produces a symmetric similarity matrix S of shape (n_markers, n_markers).

**Implementation note**: Computing MICA for all GO term pairs is expensive. A practical approach:
- For each GO term, precompute all its ancestors (walk up the DAG).
- For a pair of terms (t_a, t_b), the common ancestors are the intersection of their ancestor sets.
- The MICA is the common ancestor with the highest IC.
- Cache ancestor sets and IC values.

### Step 7: Convert similarity to distance, then embed

1. Convert similarity to distance: `D = max(S) - S` (so most similar = distance 0).
2. Run **metric MDS** (multidimensional scaling) on D to get a coordinate matrix of shape `(n_markers, k)` where `k = 128` to match the ProteinCLIP dimensionality. Use `sklearn.manifold.MDS(n_components=128, dissimilarity='precomputed', random_state=42)`.
3. If MDS with 128 dims is unstable (it may be, with only ~160 points), try `k = 64` or `k = 32` and zero-pad to 128.
4. L2-normalize each row so embeddings are on the unit sphere, matching ProteinCLIP's normalization.

### Step 8: Validate with sanity checks

Compute pairwise cosine distances on the GO embeddings and print these specific pairs:

**Expected similar (should have LOW cosine distance):**
| Pair | Why similar |
|------|-------------|
| CD279, CD152 | Both immune checkpoints (PD-1, CTLA-4) |
| CD279, CD366 | Both immune checkpoints (PD-1, TIM-3) |
| TIGIT, CD279 | Both immune checkpoints |
| CD28, CD278 | Both co-stimulatory (CD28, ICOS) |
| CD80, CD86 | Both B7 family co-stimulatory ligands |
| CD134, CD137 | Both TNFRSF co-stimulatory (OX40, 4-1BB) |
| CD25, CD127 | Both cytokine receptors on T cells |
| CD191, CD192 | Both CC chemokine receptors (CCR1, CCR2) |
| CD158a, CD158b | Both KIR receptors |
| CD11a, CD11b | Both integrin alpha chains |

**Expected dissimilar (should have HIGH cosine distance):**
| Pair | Why dissimilar |
|------|----------------|
| CD3e, CD19 | T-cell vs B-cell lineage |
| CD4, CD14 | T-helper vs monocyte |
| CD8, CD20 | T-cell vs B-cell |
| CD56, CD19 | NK vs B-cell |
| CD94, CD79a | NK receptor vs B-cell signaling |
| CD3e, CD14 | T-cell vs monocyte |
| TIGIT, CD41 | NK/T inhibitor vs platelet integrin |

Print the mean similar-pair distance and mean dissimilar-pair distance, plus the ratio (dissimilar/similar — higher is better separation).

### Step 9: Save outputs

Save `go_protein_embeddings.pkl` containing:
- `embeddings`: numpy array (n_markers, 128)
- `marker_names`: list of marker names in row order
- `similarity_matrix`: the raw Resnik BMA similarity matrix
- `marker_to_gene`: the mapping dict
- `missing_markers`: markers not found or with no GO annotations

Also save the similarity matrix separately as `go_similarity_matrix.csv` with marker names as row/column labels — this is useful for inspection.

Print a summary: total markers, markers with GO annotations, markers missing, embedding dimensions, separation ratio.
