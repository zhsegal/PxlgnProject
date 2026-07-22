#!/bin/bash
# ============================================================================
# run_all_samples.sh — Master orchestration: extract, verify, submit pipeline
#
# Usage: ./run_all_samples.sh [--extract-only] [--submit-only] [--verify-only]
#
# Steps:
#   1. Extract tars 01-05 (tar 06 is Undetermined reads, skipped)
#   2. Verify MD5 checksums of extracted FASTQs
#   3. Submit one LSF job per sample (15 jobs)
# ============================================================================

set -euo pipefail

BASE_DIR="/home/projects/nyosef/zvise/PixelGen/PixelGen/New_Data"
RAW_DIR="${BASE_DIR}/raw_data"
EXTRACTED_DIR="${RAW_DIR}/extracted"
RESULTS_DIR="${BASE_DIR}/results"

# Parse flags
DO_EXTRACT=true
DO_VERIFY=true
DO_SUBMIT=true
case "${1:-}" in
    --extract-only) DO_VERIFY=false; DO_SUBMIT=false ;;
    --verify-only)  DO_EXTRACT=false; DO_SUBMIT=false ;;
    --submit-only)  DO_EXTRACT=false; DO_VERIFY=false ;;
esac

SAMPLES=(S001 S002 S003 S004 S005 S006 S007 S008 S009 S010 S011 S012 S013 S014 S016)

# Tar-to-sample mapping
declare -A TAR_SAMPLES
TAR_SAMPLES[01]="S001 S002 S003"
TAR_SAMPLES[02]="S013 S014 S016"
TAR_SAMPLES[03]="S010 S011 S012"
TAR_SAMPLES[04]="S007 S008 S009"
TAR_SAMPLES[05]="S004 S005 S006"

echo "=========================================="
echo "Pixelator PNA preprocessing orchestrator"
echo "Date: $(date)"
echo "=========================================="

# ---- Step 1: Extract tars ----
if [ "${DO_EXTRACT}" = true ]; then
    echo ""
    echo "=== Step 1: Extracting tar archives ==="
    mkdir -p "${EXTRACTED_DIR}"

    for i in 01 02 03 04 05; do
        TAR_FILE="${RAW_DIR}/X201SC26026419-Z01-F001_${i}.tar"
        TAR_DIR="${EXTRACTED_DIR}/X201SC26026419-Z01-F001_${i}"

        if [ -d "${TAR_DIR}" ]; then
            echo "  tar ${i}: already extracted, skipping"
            continue
        fi

        if [ ! -f "${TAR_FILE}" ]; then
            echo "  ERROR: ${TAR_FILE} not found!"
            exit 1
        fi

        echo "  Extracting tar ${i} ($(du -h "${TAR_FILE}" | cut -f1))..."
        tar xf "${TAR_FILE}" -C "${EXTRACTED_DIR}"
        echo "  tar ${i}: extracted (samples: ${TAR_SAMPLES[$i]})"
    done

    echo "  Skipping tar 06 (contains only Undetermined reads)"
    echo "Extraction complete."
fi

# ---- Step 2: Verify MD5 checksums ----
if [ "${DO_VERIFY}" = true ]; then
    echo ""
    echo "=== Step 2: Verifying MD5 checksums ==="
    VERIFY_FAIL=0

    for sample in "${SAMPLES[@]}"; do
        # Find the sample's FASTQ directory
        SAMPLE_DIR=""
        for dir in "${EXTRACTED_DIR}"/X201SC26026419-Z01-F001_*/01.RawData/${sample}; do
            if [ -d "$dir" ]; then
                SAMPLE_DIR="$dir"
                break
            fi
        done

        if [ -z "${SAMPLE_DIR}" ]; then
            echo "  ${sample}: FASTQ directory NOT FOUND"
            VERIFY_FAIL=1
            continue
        fi

        MD5_FILE="${SAMPLE_DIR}/MD5.txt"
        if [ ! -f "${MD5_FILE}" ]; then
            echo "  ${sample}: MD5.txt not found in ${SAMPLE_DIR}"
            VERIFY_FAIL=1
            continue
        fi

        # Verify checksums (MD5.txt contains lines like: md5hash  filename)
        pushd "${SAMPLE_DIR}" > /dev/null
        if md5sum -c MD5.txt --quiet 2>/dev/null; then
            echo "  ${sample}: MD5 OK ($(ls *.fq.gz | wc -l) files)"
        else
            echo "  ${sample}: MD5 VERIFICATION FAILED"
            VERIFY_FAIL=1
        fi
        popd > /dev/null
    done

    if [ "${VERIFY_FAIL}" -eq 1 ]; then
        echo "WARNING: Some MD5 verifications failed. Check above."
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "All MD5 checksums verified."
    fi
fi

# ---- Step 3: Submit LSF jobs ----
if [ "${DO_SUBMIT}" = true ]; then
    echo ""
    echo "=== Step 3: Submitting LSF jobs ==="
    mkdir -p "${RESULTS_DIR}/logs"

    for i in "${!SAMPLES[@]}"; do
        sample="${SAMPLES[$i]}"

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
        bsub -J "pxl-${sample}" \
             -q long \
             -R "rusage[mem=64G]" \
             -R "affinity[thread*16]" \
             -o "${RESULTS_DIR}/logs/lsf_${sample}_%J.out" \
             -e "${RESULTS_DIR}/logs/lsf_${sample}_%J.err" \
             bash "${BASE_DIR}/run_pixelator_pna.sh" "${sample}" "${FASTQ_DIR}" 16
    done

    echo ""
    echo "All jobs submitted. Monitor with: bjobs -w | grep pxl-"
fi

echo ""
echo "Done."
