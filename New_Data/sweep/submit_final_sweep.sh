#!/bin/bash
# Submit the final MultiModalSCVI sweep (fusion architectures x ref-notebook params),
# one bsub GPU job per config. Already-finished runs (DONE flag) are skipped, so
# re-running only fills gaps. Writes to cache/final_sweep (prior sweep cache untouched).
set -euo pipefail

ROOT=/home/projects/nyosef/zvise/PixelGen/PixelGen
SWEEP=$ROOT/New_Data/sweep
CACHE=$ROOT/New_Data/cache/final_sweep
LOGS=$SWEEP/logs_final
mkdir -p "$LOGS" "$CACHE"

ARCH=(baseline S1_moe S2_poetemp S3_mvtcae S4_mmvaeplus)
SPATIAL=(top500 ct100)
N_LATENT=(20 30)          # n_latent=10 dropped: per-modality latent must be >= scVI(20)
N_LAYERS=(2 3)
BATCH_MASK=(ff ft)

n_sub=0; n_skip=0
for a in "${ARCH[@]}"; do
for sp in "${SPATIAL[@]}"; do
for nl in "${N_LATENT[@]}"; do
for L in "${N_LAYERS[@]}"; do
for bm in "${BATCH_MASK[@]}"; do
  rid="arch-${a}__sp-${sp}__nl${nl}__L${L}__bm-${bm}"
  if [ -f "$CACHE/$rid/DONE" ]; then
    echo "skip  $rid"; n_skip=$((n_skip+1)); continue
  fi
  bsub -J "fsw-$rid" -q long-gpu \
    -gpu "num=1:j_exclusive=yes:gmem=16G" \
    -R "rusage[mem=64G]" -R "affinity[thread*8]" \
    -o "$LOGS/$rid.out" -e "$LOGS/$rid.err" \
    -- bash "$SWEEP/run_final_sweep.lsf" \
       --arch "$a" --spatial "$sp" --n-latent "$nl" --n-layers "$L" --batch-mask "$bm"
  n_sub=$((n_sub+1))
done; done; done; done; done

echo "submitted=$n_sub skipped=$n_skip"
