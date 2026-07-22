#!/bin/bash
# ============================================================================
# run_pixelator_pna.sh — Per-sample pixelator PNA pipeline
#
# Usage: ./run_pixelator_pna.sh <SAMPLE_NAME> <FASTQ_DIR> [THREADS]
#   SAMPLE_NAME: e.g., S001
#   FASTQ_DIR:   directory containing extracted FASTQ files for this sample
#   THREADS:     number of threads (default: 4)
#
# Runs the full pixelator single-cell-pna pipeline:
#   amplicon → demux → collapse → graph → denoise → analysis → layout
#
# Each step is skipped if its output already exists (resume-safe).
#
# Pixelator output structure (v0.21.3):
#   --output DIR → creates DIR/<step_name>/<files>
#   e.g., --output results/amplicon → results/amplicon/amplicon/S001.amplicon.fq.zst
#   Files are flat (no per-sample subdirectories), prefixed with sample name.
# ============================================================================

set -euo pipefail

SAMPLE="$1"
FASTQ_DIR="$2"
THREADS="${3:-4}"

PIXELATOR="/home/projects/nyosef/zvise/.local/share/mamba/envs/pxlgn/bin/pixelator"
PYTHON="/home/projects/nyosef/zvise/.local/share/mamba/envs/pxlgn/bin/python3"
BASE_DIR="/home/projects/nyosef/zvise/PixelGen/PixelGen/New_Data"
RESULTS_DIR="${BASE_DIR}/results"
CONCAT_DIR="${BASE_DIR}/fastqs_concat"
DESIGN="pna-2"
PANEL="proxiome-immuno-156-FMC63"
LOG_FILE="${RESULTS_DIR}/logs/${SAMPLE}.log"

# Use per-sample output directories to avoid race conditions between parallel jobs
SAMPLE_OUT="${RESULTS_DIR}/${SAMPLE}"

mkdir -p "${RESULTS_DIR}/logs" "${CONCAT_DIR}" "${SAMPLE_OUT}"

# Logging helper
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

log "=========================================="
log "Starting pixelator PNA pipeline for ${SAMPLE}"
log "FASTQ dir: ${FASTQ_DIR}"
log "Threads: ${THREADS}"
log "Output dir: ${SAMPLE_OUT}"
log "=========================================="

# ---- Step 0: Concatenate multi-lane FASTQs ----
log "Step 0: Concatenating multi-lane FASTQs..."

R1="${CONCAT_DIR}/${SAMPLE}_R1.fq.gz"
R2="${CONCAT_DIR}/${SAMPLE}_R2.fq.gz"

if [ ! -f "${R1}" ]; then
    cat "${FASTQ_DIR}"/${SAMPLE}_*_L*_1.fq.gz > "${R1}"
    log "  Created ${R1} ($(du -h "${R1}" | cut -f1))"
else
    log "  ${R1} already exists, skipping"
fi

if [ ! -f "${R2}" ]; then
    cat "${FASTQ_DIR}"/${SAMPLE}_*_L*_2.fq.gz > "${R2}"
    log "  Created ${R2} ($(du -h "${R2}" | cut -f1))"
else
    log "  ${R2} already exists, skipping"
fi

# ---- Step 1: Amplicon ----
# Output: <out>/amplicon/<SAMPLE>.amplicon.fq.zst + .report.json
AMPLICON_DIR="${SAMPLE_OUT}/amplicon"
AMPLICON_OUT="${AMPLICON_DIR}/amplicon/${SAMPLE}.amplicon.fq.zst"
AMPLICON_REPORT="${AMPLICON_DIR}/amplicon/${SAMPLE}.report.json"

# Also check old shared location from previous runs
if [ ! -f "${AMPLICON_REPORT}" ] && [ -f "${RESULTS_DIR}/amplicon/amplicon/${SAMPLE}.report.json" ]; then
    log "Step 1: amplicon — found in shared directory, copying to per-sample dir"
    mkdir -p "${AMPLICON_DIR}/amplicon"
    cp "${RESULTS_DIR}/amplicon/amplicon/${SAMPLE}".* "${AMPLICON_DIR}/amplicon/"
fi

if [ -f "${AMPLICON_REPORT}" ]; then
    log "Step 1: amplicon — already completed, skipping"
    log "  Output: ${AMPLICON_OUT} ($(du -h "${AMPLICON_OUT}" | cut -f1))"
else
    rm -f "${AMPLICON_OUT}" "${AMPLICON_DIR}/amplicon/${SAMPLE}.meta.json" 2>/dev/null || true
    log "Step 1: amplicon — QC & amplicon construction..."
    STEP_START=$(date +%s)

    ${PIXELATOR} single-cell-pna amplicon \
        --design ${DESIGN} \
        --sample-name "${SAMPLE}" \
        --threads ${THREADS} \
        --output "${AMPLICON_DIR}" \
        "${R1}" "${R2}" \
        2>&1 | tee -a "${LOG_FILE}"

    STEP_END=$(date +%s)
    log "  amplicon completed in $((STEP_END - STEP_START))s (exit: $?)"

    if [ ! -f "${AMPLICON_OUT}" ]; then
        log "ERROR: No amplicon output found at ${AMPLICON_OUT}"
        ls -R "${AMPLICON_DIR}/" 2>&1 | head -20 | tee -a "${LOG_FILE}"
        exit 1
    fi
    log "  Output: ${AMPLICON_OUT} ($(du -h "${AMPLICON_OUT}" | cut -f1))"
fi

# ---- Step 2: Demux ----
# Output: <out>/demux/<SAMPLE>.demux.m1.part_NNN.parquet (flat, no subdirs)
DEMUX_DIR="${SAMPLE_OUT}/demux"
DEMUX_REPORT="${DEMUX_DIR}/demux/${SAMPLE}.report.json"

if [ -f "${DEMUX_REPORT}" ]; then
    DEMUX_FILES=$(ls "${DEMUX_DIR}/demux/${SAMPLE}"*.parquet 2>/dev/null | grep -v report) || true
    log "Step 2: demux — already completed, skipping"
    log "  Output: $(echo "${DEMUX_FILES}" | wc -l) parquet files"
else
    log "Step 2: demux — demultiplex by antibody barcode..."
    # Clean leftover tmp dir from previous failed/killed runs
    rm -rf "${DEMUX_DIR}/demux/tmp" 2>/dev/null || true
    STEP_START=$(date +%s)

    ${PIXELATOR} single-cell-pna demux \
        --design ${DESIGN} \
        --panel ${PANEL} \
        --strategy paired \
        --threads ${THREADS} \
        --output "${DEMUX_DIR}" \
        "${AMPLICON_OUT}" \
        2>&1 | tee -a "${LOG_FILE}"

    STEP_END=$(date +%s)
    log "  demux completed in $((STEP_END - STEP_START))s (exit: $?)"

    DEMUX_FILES=$(ls "${DEMUX_DIR}/demux/${SAMPLE}"*.parquet 2>/dev/null | grep -v report) || true
    if [ -z "${DEMUX_FILES}" ]; then
        log "ERROR: No demux parquet output found for ${SAMPLE}"
        ls -R "${DEMUX_DIR}/" 2>&1 | head -30 | tee -a "${LOG_FILE}"
        exit 1
    fi
    log "  Output: $(echo "${DEMUX_FILES}" | wc -l) parquet files"
fi

# ---- Step 3: Collapse ----
COLLAPSE_DIR="${SAMPLE_OUT}/collapse"
# Paired collapse produces: S001.report.json + S001.collapsed.parquet
# Independent collapse produces: S001.collapse.m1.part_NNN.report.json + .parquet
COLLAPSE_REPORT_PAIRED="${COLLAPSE_DIR}/collapse/${SAMPLE}.report.json"
COLLAPSE_REPORT_PATTERN="${COLLAPSE_DIR}/collapse/${SAMPLE}.collapse.*.report.json"
COLLAPSE_REPORT_COUNT=$(ls ${COLLAPSE_REPORT_PATTERN} 2>/dev/null | wc -l || true)

if [ -f "${COLLAPSE_REPORT_PAIRED}" ] || [ "${COLLAPSE_REPORT_COUNT}" -gt 0 ]; then
    COLLAPSE_FILES=$(ls "${COLLAPSE_DIR}/collapse/${SAMPLE}"*.parquet 2>/dev/null) || true
    log "Step 3: collapse — already completed, skipping"
    log "  Output: $(echo "${COLLAPSE_FILES}" | wc -l) parquet files"
else
    log "Step 3: collapse — error correction & deduplication..."
    STEP_START=$(date +%s)

    ${PIXELATOR} single-cell-pna collapse \
        --design ${DESIGN} \
        --panel ${PANEL} \
        --threads ${THREADS} \
        --output "${COLLAPSE_DIR}" \
        ${DEMUX_FILES} \
        2>&1 | tee -a "${LOG_FILE}"

    STEP_END=$(date +%s)
    log "  collapse completed in $((STEP_END - STEP_START))s (exit: $?)"

    COLLAPSE_FILES=$(ls "${COLLAPSE_DIR}/collapse/${SAMPLE}"*.parquet 2>/dev/null) || true
    if [ -z "${COLLAPSE_FILES}" ]; then
        log "ERROR: No collapse output found for ${SAMPLE}"
        ls -R "${COLLAPSE_DIR}/" 2>&1 | head -30 | tee -a "${LOG_FILE}"
        exit 1
    fi
    log "  Output: $(echo "${COLLAPSE_FILES}" | wc -l) parquet files"
fi

# ---- Step 3b: Combine-Collapse ----
# Paired strategy: collapse produces a single .collapsed.parquet — skip combine-collapse
# Independent strategy: collapse produces partitioned m1/m2 files — need combine-collapse
COMBINE_DIR="${SAMPLE_OUT}/combine-collapse"

# Check if paired collapse produced a single file (no combine needed)
PAIRED_COLLAPSE="${COLLAPSE_DIR}/collapse/${SAMPLE}.collapsed.parquet"

if [ -f "${PAIRED_COLLAPSE}" ]; then
    # Paired strategy: use the single collapsed parquet directly
    COMBINED_PARQUET="${PAIRED_COLLAPSE}"
    log "Step 3b: combine-collapse — not needed (paired strategy, single parquet)"
    log "  Using: ${COMBINED_PARQUET} ($(du -h "${COMBINED_PARQUET}" | cut -f1))"
else
    # Independent strategy: need to merge partitioned files
    # Check if already done
    COMBINED_PARQUET=""
    if [ -f "${COMBINE_DIR}/collapse/${SAMPLE}.parquet" ]; then
        COMBINED_PARQUET="${COMBINE_DIR}/collapse/${SAMPLE}.parquet"
    elif [ -f "${COMBINE_DIR}/collapse/${SAMPLE}.collapse.parquet" ]; then
        COMBINED_PARQUET="${COMBINE_DIR}/collapse/${SAMPLE}.collapse.parquet"
    fi

    if [ -n "${COMBINED_PARQUET}" ]; then
        log "Step 3b: combine-collapse — already completed, skipping"
        log "  Output: ${COMBINED_PARQUET} ($(du -h "${COMBINED_PARQUET}" | cut -f1))"
    else
        log "Step 3b: combine-collapse — merging partitioned collapse output..."
        STEP_START=$(date +%s)

        COLLAPSE_INNER="${COLLAPSE_DIR}/collapse"
        ${PIXELATOR} single-cell-pna combine-collapse \
            --parquet-pattern "${COLLAPSE_INNER}/${SAMPLE}.collapse.*.parquet" \
            --report-pattern "${COLLAPSE_INNER}/${SAMPLE}.collapse.*.report.json" \
            --output "${COMBINE_DIR}" \
            2>&1 | tee -a "${LOG_FILE}"

        STEP_END=$(date +%s)
        log "  combine-collapse completed in $((STEP_END - STEP_START))s (exit: $?)"

        # Find the output
        if [ -f "${COMBINE_DIR}/collapse/${SAMPLE}.parquet" ]; then
            COMBINED_PARQUET="${COMBINE_DIR}/collapse/${SAMPLE}.parquet"
        elif [ -f "${COMBINE_DIR}/collapse/${SAMPLE}.collapse.parquet" ]; then
            COMBINED_PARQUET="${COMBINE_DIR}/collapse/${SAMPLE}.collapse.parquet"
        else
            COMBINED_PARQUET=$(find "${COMBINE_DIR}" -name "${SAMPLE}*.parquet" 2>/dev/null | head -1) || true
        fi
        if [ -z "${COMBINED_PARQUET}" ] || [ ! -f "${COMBINED_PARQUET}" ]; then
            log "ERROR: No combine-collapse output found for ${SAMPLE}"
            find "${COMBINE_DIR}" -type f 2>&1 | tee -a "${LOG_FILE}"
            exit 1
        fi
        log "  Output: ${COMBINED_PARQUET} ($(du -h "${COMBINED_PARQUET}" | cut -f1))"
    fi
fi

# Helper: find a .pxl file for a sample in a directory
find_pxl() {
    local dir="$1" sample="$2"
    if [ -d "${dir}" ]; then
        find "${dir}" -name "${sample}*.pxl" 2>/dev/null | head -1
    fi
    true
}

# Helper: check if a .pxl DuckDB file contains a given table
# Returns "yes" if the table exists, "no" otherwise.
# Usage: pxl_has_table <pxl_file> <table_name>
pxl_has_table() {
    local pxl_file="$1" table="$2"
    "${PYTHON}" - <<EOF 2>/dev/null
import duckdb, sys
try:
    c = duckdb.connect("${pxl_file}", read_only=True)
    rows = c.execute("SELECT table_name FROM information_schema.tables").fetchall()
    c.close()
    tables = [r[0] for r in rows]
    print("yes" if "${table}" in tables else "no")
except Exception:
    print("no")
EOF
    true
}

# ---- Step 4: Graph ----
GRAPH_DIR="${SAMPLE_OUT}/graph"
GRAPH_PXL=$(find_pxl "${GRAPH_DIR}" "${SAMPLE}")

if [ -n "${GRAPH_PXL}" ] && [ -f "${GRAPH_PXL}" ]; then
    log "Step 4: graph — already completed, skipping"
    log "  Output: ${GRAPH_PXL} ($(du -h "${GRAPH_PXL}" | cut -f1))"
else
    log "Step 4: graph — build cell components (creates .pxl)..."
    STEP_START=$(date +%s)

    ${PIXELATOR} single-cell-pna graph \
        --multiplet-recovery \
        --panel ${PANEL} \
        --output "${GRAPH_DIR}" \
        "${COMBINED_PARQUET}" \
        2>&1 | tee -a "${LOG_FILE}"

    STEP_END=$(date +%s)
    log "  graph completed in $((STEP_END - STEP_START))s (exit: $?)"

    GRAPH_PXL=$(find_pxl "${GRAPH_DIR}" "${SAMPLE}")
    if [ -z "${GRAPH_PXL}" ] || [ ! -f "${GRAPH_PXL}" ]; then
        log "ERROR: No graph .pxl output found for ${SAMPLE}"
        find "${GRAPH_DIR}" -type f 2>&1 | tee -a "${LOG_FILE}"
        exit 1
    fi
    log "  Output: ${GRAPH_PXL} ($(du -h "${GRAPH_PXL}" | cut -f1))"
fi

# ---- Step 5: Denoise ----
DENOISE_DIR="${SAMPLE_OUT}/denoise"
DENOISE_PXL=$(find_pxl "${DENOISE_DIR}" "${SAMPLE}")

if [ -n "${DENOISE_PXL}" ] && [ -f "${DENOISE_PXL}" ]; then
    log "Step 5: denoise — already completed, skipping"
    log "  Output: ${DENOISE_PXL} ($(du -h "${DENOISE_PXL}" | cut -f1))"
else
    log "Step 5: denoise — clean cell graphs..."
    STEP_START=$(date +%s)

    ${PIXELATOR} single-cell-pna denoise \
        --output "${DENOISE_DIR}" \
        "${GRAPH_PXL}" \
        2>&1 | tee -a "${LOG_FILE}"

    STEP_END=$(date +%s)
    log "  denoise completed in $((STEP_END - STEP_START))s (exit: $?)"

    DENOISE_PXL=$(find_pxl "${DENOISE_DIR}" "${SAMPLE}")
    if [ -z "${DENOISE_PXL}" ] || [ ! -f "${DENOISE_PXL}" ]; then
        log "ERROR: No denoise .pxl output found for ${SAMPLE}"
        find "${DENOISE_DIR}" -type f 2>&1 | tee -a "${LOG_FILE}"
        exit 1
    fi
    log "  Output: ${DENOISE_PXL} ($(du -h "${DENOISE_PXL}" | cut -f1))"
fi

# ---- Step 6: Analysis ----
ANALYSIS_DIR="${SAMPLE_OUT}/analysis"
ANALYSIS_PXL=$(find_pxl "${ANALYSIS_DIR}" "${SAMPLE}")

# Validate: if the file exists but is missing the proximity table, it was created
# by pixelator's copy-first pattern before a previous job was killed.  Remove it
# so the step runs again from scratch.
if [ -n "${ANALYSIS_PXL}" ] && [ -f "${ANALYSIS_PXL}" ]; then
    PROX_OK=$(pxl_has_table "${ANALYSIS_PXL}" "proximity")
    if [ "${PROX_OK}" != "yes" ]; then
        log "Step 6: analysis.pxl exists but is missing proximity table — removing to force re-run"
        rm -f "${ANALYSIS_PXL}"
        ANALYSIS_PXL=""
    fi
fi

if [ -n "${ANALYSIS_PXL}" ] && [ -f "${ANALYSIS_PXL}" ]; then
    log "Step 6: analysis — already completed, skipping"
    log "  Output: ${ANALYSIS_PXL} ($(du -h "${ANALYSIS_PXL}" | cut -f1))"
else
    log "Step 6: analysis — compute proximity scores..."
    STEP_START=$(date +%s)

    # --cores 1 forces sequential (single-threaded) joblib execution, avoiding
    # TERM_THREADLIMIT on LSF.  Thread env vars are belt-and-suspenders.
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1

    ${PIXELATOR} --cores 1 single-cell-pna analysis \
        --compute-proximity \
        --output "${ANALYSIS_DIR}" \
        "${DENOISE_PXL}" \
        2>&1 | tee -a "${LOG_FILE}"

    STEP_END=$(date +%s)
    log "  analysis completed in $((STEP_END - STEP_START))s (exit: $?)"

    ANALYSIS_PXL=$(find_pxl "${ANALYSIS_DIR}" "${SAMPLE}")
    if [ -z "${ANALYSIS_PXL}" ] || [ ! -f "${ANALYSIS_PXL}" ]; then
        log "ERROR: No analysis .pxl output found for ${SAMPLE}"
        find "${ANALYSIS_DIR}" -type f 2>&1 | tee -a "${LOG_FILE}"
        exit 1
    fi
    log "  Output: ${ANALYSIS_PXL} ($(du -h "${ANALYSIS_PXL}" | cut -f1))"
fi

# ---- Step 7: Layout ----
LAYOUT_DIR="${SAMPLE_OUT}/layout"
FINAL_PXL=$(find_pxl "${LAYOUT_DIR}" "${SAMPLE}")

# Validate: if the file exists but is missing the layouts table, it was a partial
# copy from a previous killed job.  Remove it so the step runs again.
if [ -n "${FINAL_PXL}" ] && [ -f "${FINAL_PXL}" ]; then
    LAYOUT_OK=$(pxl_has_table "${FINAL_PXL}" "layouts")
    if [ "${LAYOUT_OK}" != "yes" ]; then
        log "Step 7: layout.pxl exists but is missing layouts table — removing to force re-run"
        rm -f "${FINAL_PXL}"
        FINAL_PXL=""
    fi
fi

if [ -n "${FINAL_PXL}" ] && [ -f "${FINAL_PXL}" ]; then
    log "Step 7: layout — already completed, skipping"
    log "  Output: ${FINAL_PXL} ($(du -h "${FINAL_PXL}" | cut -f1))"
else
    log "Step 7: layout — 3D visualization coordinates..."
    STEP_START=$(date +%s)

    ${PIXELATOR} --cores 1 single-cell-pna layout \
        --layout-algorithm pmds_3d \
        --output "${LAYOUT_DIR}" \
        "${ANALYSIS_PXL}" \
        2>&1 | tee -a "${LOG_FILE}"

    STEP_END=$(date +%s)
    log "  layout completed in $((STEP_END - STEP_START))s (exit: $?)"

    FINAL_PXL=$(find_pxl "${LAYOUT_DIR}" "${SAMPLE}")
    if [ -z "${FINAL_PXL}" ] || [ ! -f "${FINAL_PXL}" ]; then
        log "ERROR: No layout .pxl output found for ${SAMPLE}"
        find "${LAYOUT_DIR}" -type f 2>&1 | tee -a "${LOG_FILE}"
        exit 1
    fi
    log "  Output: ${FINAL_PXL} ($(du -h "${FINAL_PXL}" | cut -f1))"
fi

# ---- Copy final .pxl to shared output directory ----
FINAL_OUTPUT_DIR="${RESULTS_DIR}/layout"
mkdir -p "${FINAL_OUTPUT_DIR}"
cp "${FINAL_PXL}" "${FINAL_OUTPUT_DIR}/${SAMPLE}.pxl"

log "=========================================="
log "Pipeline COMPLETE for ${SAMPLE}"
log "Final .pxl: ${FINAL_OUTPUT_DIR}/${SAMPLE}.pxl"
log "=========================================="
