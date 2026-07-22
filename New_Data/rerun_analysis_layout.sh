#!/bin/bash
# ============================================================================
# rerun_analysis_layout.sh — Re-run analysis + layout steps for all 15 samples
#
# Background: The original runs were killed by TERM_THREADLIMIT before the
# proximity or layout tables were written. Because pixelator creates the output
# .pxl file (as a copy) BEFORE computing, the pipeline's resume-check falsely
# treated the empty files as completed and skipped the steps on all later runs.
#
# Fix applied in run_pixelator_pna.sh:
#   - OMP/MKL/OPENBLAS_NUM_THREADS=1 before the analysis step
#   - --cores 4 on both pixelator analysis and layout commands
#
# This script:
#   1. Deletes the empty analysis.pxl and layout.pxl for all 15 samples
#   2. Submits one LSF job per sample (--submit-only mode of the main script,
#      but targeted to only run from Step 6 onward)
#
# Usage: bash rerun_analysis_layout.sh [--dry-run]
# ============================================================================

set -euo pipefail

BASE_DIR="/home/projects/nyosef/zvise/PixelGen/PixelGen/New_Data"
RESULTS_DIR="${BASE_DIR}/results"
SAMPLES=(S001 S002 S003 S004 S005 S006 S007 S008 S009 S010 S011 S012 S013 S014 S016)
EXTRACTED_DIR="${BASE_DIR}/raw_data/extracted"

DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN — no files will be deleted, no jobs submitted ==="
fi

echo "=========================================="
echo "Re-run analysis + layout for ${#SAMPLES[@]} samples"
echo "Date: $(date)"
echo "=========================================="

# ---- Step 1: Delete bad (empty) analysis and layout .pxl files ----
echo ""
echo "=== Step 1: Removing empty analysis/layout .pxl files ==="
for sample in "${SAMPLES[@]}"; do
    analysis_pxl="${RESULTS_DIR}/${sample}/analysis/analysis/${sample}.analysis.pxl"
    layout_pxl="${RESULTS_DIR}/${sample}/layout/layout/${sample}.layout.pxl"
    layout_pxl_shared="${RESULTS_DIR}/layout/${sample}.pxl"

    for f in "${analysis_pxl}" "${layout_pxl}" "${layout_pxl_shared}"; do
        if [ -f "${f}" ]; then
            echo "  Removing: ${f}"
            if [ "${DRY_RUN}" = false ]; then
                rm "${f}"
            fi
        fi
    done
done
echo "Done removing files."

# ---- Step 2: Submit LSF jobs ----
echo ""
echo "=== Step 2: Submitting LSF jobs ==="
mkdir -p "${RESULTS_DIR}/logs"

for sample in "${SAMPLES[@]}"; do
    # Find FASTQ dir
    FASTQ_DIR=""
    for dir in "${EXTRACTED_DIR}"/X201SC26026419-Z01-F001_*/01.RawData/${sample}; do
        if [ -d "$dir" ]; then
            FASTQ_DIR="$dir"
            break
        fi
    done

    if [ -z "${FASTQ_DIR}" ]; then
        echo "  ${sample}: SKIPPED (FASTQ dir not found)"
        continue
    fi

    echo "  Submitting ${sample}..."
    if [ "${DRY_RUN}" = false ]; then
        bsub -J "pxl-rerun-${sample}" \
             -q long \
             -R "rusage[mem=256G]" \
             -o "${RESULTS_DIR}/logs/lsf_${sample}_rerun_%J.out" \
             -e "${RESULTS_DIR}/logs/lsf_${sample}_rerun_%J.err" \
             bash "${BASE_DIR}/run_pixelator_pna.sh" "${sample}" "${FASTQ_DIR}" 4
    else
        echo "    [dry-run] bsub -J pxl-rerun-${sample} -q long -R rusage[mem=256G] ..."
    fi
done

echo ""
echo "All jobs submitted. Monitor with:"
echo "  bjobs -w | grep pxl-rerun-"
echo "  tail -f ${RESULTS_DIR}/logs/lsf_S001_rerun_*.out"
