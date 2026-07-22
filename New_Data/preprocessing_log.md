# PNA Preprocessing Log

## Data Delivery

- **Sequencing provider**: Novogene
- **Contract ID**: X201SC26026419-Z01-F001
- **Panel**: proxiome-immuno-156-FMC63 (159 markers after collapse)
- **Assay**: PNA (Proximity Network Assay)
- **Delivery date**: March 2026
- **Total data**: ~630GB across 6 tar archives (tar 06 = Undetermined reads only)

## Experimental Design: Blinatumomab Co-culture

B cells + T cells from healthy donors and patients, with/without Blinatumomab treatment.

| Sample | Time (h) | Condition | Target cells | T cells |
|--------|----------|-----------|-------------|---------|
| S001 | 6 | Mock | healthy B cells | healthy T cells |
| S002 | 6 | 1ng/ml Blinatumomab | healthy B cells | healthy T cells |
| S003 | 48 | Mock | healthy B cells | healthy T cells |
| S004 | 48 | 1ng/ml Blinatumomab | healthy B cells | healthy T cells |
| S005 | 6 | Mock | NALM-6 cells | healthy T cells |
| S006 | 6 | 1ng/ml Blinatumomab | NALM-6 cells | healthy T cells |
| S007 | 48 | Mock | NALM-6 cells | healthy T cells |
| S008 | 48 | 1ng/ml Blinatumomab | NALM-6 cells | healthy T cells |
| S009 | 6 | Mock | patient B cells | patient T cells |
| S010 | 6 | 1ng/ml Blinatumomab | patient B cells | patient T cells |
| S011 | 48 | Mock | patient B cells | patient T cells |
| S012 | 48 | 1ng/ml Blinatumomab | patient B cells | patient T cells |
| S013 | 6 | Mock | NALM-6 cells | patient T cells |
| S014 | 6 | 1ng/ml Blinatumomab | NALM-6 cells | patient T cells |
| S016 | 48 | 1ng/ml Blinatumomab | NALM-6 cells | patient T cells |

**Note**: S015 is missing from the delivery.

## Data Structure

### Tar archive contents
| Archive | Size | Samples |
|---------|------|---------|
| _01.tar | 116G | S001, S002, S003 |
| _02.tar | 117G | S013, S014, S016 |
| _03.tar | 128G | S010, S011, S012 |
| _04.tar | 131G | S007, S008, S009 |
| _05.tar | 136G | S004, S005, S006 |
| _06.tar | ~5G (still downloading) | Undetermined reads only |

### FASTQ naming convention
Each sample has 2 lanes x 2 reads = 4 FASTQ files:
```
SXXX_MKDL260003545-1A_23HJTTLT4_L5_1.fq.gz  (lane 5, read 1)
SXXX_MKDL260003545-1A_23HJTTLT4_L5_2.fq.gz  (lane 5, read 2)
SXXX_MKDL260003545-1A_23HJTTLT4_L6_1.fq.gz  (lane 6, read 1)
SXXX_MKDL260003545-1A_23HJTTLT4_L6_2.fq.gz  (lane 6, read 2)
```

Multi-lane FASTQs are concatenated before processing:
```
cat S001_*_L5_1.fq.gz S001_*_L6_1.fq.gz > S001_R1.fq.gz
cat S001_*_L5_2.fq.gz S001_*_L6_2.fq.gz > S001_R2.fq.gz
```

## Novogene QC Summary

All 15 samples pass QC:
- **Q20**: >99.2%
- **Q30**: >96.6%
- **Error rate**: 0.01%
- **GC content**: ~54.5%

## Pipeline: pixelator v0.21.3 PNA

### Software
- **pixelator**: v0.21.3
- **Path**: `/home/projects/nyosef/zvise/.local/share/mamba/envs/pxlgn/bin/pixelator`
- **Conda env**: `pxlgn`

### Pipeline steps

| Step | Command | Input | Output | Key parameters |
|------|---------|-------|--------|---------------|
| 0 | cat | Multi-lane FASTQs | Concatenated R1/R2 | — |
| 1 | `amplicon` | R1.fq.gz, R2.fq.gz | processed.fq.gz | `--design pna-2` |
| 2 | `demux` | processed.fq.gz | *.parquet | `--design pna-2 --strategy paired --panel proxiome-immuno-156-FMC63` |
| 3 | `collapse` | *.parquet | *.parquet | `--design pna-2 --panel proxiome-immuno-156-FMC63` |
| 4 | `graph` | *.parquet | .pxl | `--multiplet-recovery --panel proxiome-immuno-156-FMC63` |
| 5 | `denoise` | .pxl | .pxl | default parameters |
| 6 | `analysis` | .pxl | .pxl | `--compute-proximity` |
| 7 | `layout` | .pxl | .pxl | `--layout-algorithm pmds_3d` |

### Resource allocation (LSF)
- **Queue**: long
- **Memory**: 256GB (graph step requires ~145GB for ~100M molecules)
- **Threads**: no affinity constraint (analysis/layout with `--compute-proximity` spawn many threads)
- **GPU**: none

## Processing Status

All 15 samples completed successfully (2026-03-23).

| Sample | Extract | MD5 | amplicon | demux | collapse | graph | denoise | analysis | layout | Status |
|--------|---------|-----|----------|-------|----------|-------|---------|----------|--------|--------|
| S001 | OK | OK | OK | OK | OK | OK | OK | OK | OK | DONE |
| S002 | OK | OK | OK | OK | OK | OK | OK | OK | OK | DONE |
| S003 | OK | OK | OK | OK | OK | OK | OK | OK | OK | DONE |
| S004 | OK | OK | OK | OK | OK | OK | OK | OK | OK | DONE |
| S005 | OK | OK | OK | OK | OK | OK | OK | OK | OK | DONE |
| S006 | OK | OK | OK | OK | OK | OK | OK | OK | OK | DONE |
| S007 | OK | OK | OK | OK | OK | OK | OK | OK | OK | DONE |
| S008 | OK | OK | OK | OK | OK | OK | OK | OK | OK | DONE |
| S009 | OK | OK | OK | OK | OK | OK | OK | OK | OK | DONE |
| S010 | OK | OK | OK | OK | OK | OK | OK | OK | OK | DONE |
| S011 | OK | OK | OK | OK | OK | OK | OK | OK | OK | DONE |
| S012 | OK | OK | OK | OK | OK | OK | OK | OK | OK | DONE |
| S013 | OK | OK | OK | OK | OK | OK | OK | OK | OK | DONE |
| S014 | OK | OK | OK | OK | OK | OK | OK | OK | OK | DONE |
| S016 | OK | OK | OK | OK | OK | OK | OK | OK | OK | DONE |

## Output Files

Final `.pxl` files are in per-sample directories: `New_Data/results/<SAMPLE>/layout/layout/<SAMPLE>.layout.pxl`

| Sample | .pxl file | Size | Cells | Markers |
|--------|-----------|------|-------|---------|
| S001 | S001.layout.pxl | 2.5G | 1,270 | 159 |
| S002 | S002.layout.pxl | 3.7G | 1,349 | 159 |
| S003 | S003.layout.pxl | 3.2G | 1,071 | 159 |
| S004 | S004.layout.pxl | 3.8G | 1,236 | 159 |
| S005 | S005.layout.pxl | 3.8G | 1,472 | 159 |
| S006 | S006.layout.pxl | 3.2G | 1,048 | 159 |
| S007 | S007.layout.pxl | 2.9G | 859 | 159 |
| S008 | S008.layout.pxl | 3.3G | 973 | 159 |
| S009 | S009.layout.pxl | 3.3G | 1,061 | 159 |
| S010 | S010.layout.pxl | 3.2G | 1,059 | 159 |
| S011 | S011.layout.pxl | 3.2G | 950 | 159 |
| S012 | S012.layout.pxl | 4.0G | 1,505 | 159 |
| S013 | S013.layout.pxl | 3.3G | 913 | 159 |
| S014 | S014.layout.pxl | 3.1G | 1,177 | 159 |
| S016 | S016.layout.pxl | 3.1G | 1,015 | 159 |

**Total: 16,958 cells across 15 samples (mean: 1,130 cells/sample)**

## Verification

All 15 samples verified — loaded via `read_pna()` and confirmed cell counts and marker dimensions.

```python
from pixelator import read_pna
pg = read_pna(pxl_files)
adata = pg.adata()  # .adata() is a method, not a property
print(adata)  # 16,958 × 159
```

## Issues Encountered & Fixes

### 1. `--strategy paired` (critical)
Initial runs used the default independent collapse strategy, which produced a positional join bug causing nearly all data loss (1 cell instead of ~1,000+). **Fix**: added `--strategy paired` to the demux step. This was the root cause of the graph failures.

### 2. FileExistsError on `demux/demux/tmp`
Previous killed/failed runs left behind temporary directories. When jobs were resubmitted, demux failed with `FileExistsError`.
**Fix**: Added `rm -rf "${DEMUX_DIR}/demux/tmp"` before the demux step in `run_pixelator_pna.sh`.

### 3. Collapse detection naming mismatch
Paired collapse produces `SAMPLE.report.json` + `SAMPLE.collapsed.parquet`, but the script checked for `SAMPLE.collapse.*.report.json` (independent naming). Jobs silently exited after collapse.
**Fix**: Added paired report check and broadened parquet glob pattern.

### 4. Combine-collapse not needed for paired strategy
Paired collapse produces a single `.collapsed.parquet`, not partitioned m1/m2 files. The script tried to run combine-collapse with wrong glob patterns.
**Fix**: Detect `${SAMPLE}.collapsed.parquet` and skip combine-collapse entirely.

### 5. TERM_MEMLIMIT on graph step
Graph step with ~100M molecules needs ~145GB. Initial 96GB allocation was insufficient.
**Fix**: Increased to 256GB (`-R "rusage[mem=256G]"`).

### 6. `bmod -R` replacing entire resource spec
Running `bmod -R "affinity[thread*32]"` dropped memory from 256GB to 1GB (LSF replaces the entire resource string). This killed almost all running jobs.
**Lesson**: Always specify complete resource string with `bmod -R`.

### 7. TERM_THREADLIMIT on analysis/layout
`affinity[thread*N]` imposes hard thread limits. Analysis with `--compute-proximity` and layout with 900+ cells spawn too many threads. Tried thread*16, *32, *64 — all insufficient.
**Fix**: Removed affinity constraint entirely, using only `rusage[mem=256G]`.

### 8. S007 collapse ValueError
`negative axis 1 index: -1` in sparse matrix during collapse of marker pair [CD38, CD29]. Cleaned partial output and resubmitted — succeeded on retry.

## Notes

- **S015 missing**: Not included in the Novogene delivery. This is the 48h Mock condition for NALM-6 + patient T cells.
- **Tar 06**: Contains only Undetermined (unmatched barcode) reads. Not needed for sample processing.
- All 15 samples are in tars 01-05 and can be processed without tar 06.
- Per-sample output directories (`results/${SAMPLE}/`) used to avoid race conditions between parallel jobs.
- QC analysis notebook: `New_Data/qc_analysis.ipynb`
