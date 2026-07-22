#!/bin/bash
# Submit the full MultiModalSCVI sweep, one bsub GPU job per config.
# Already-finished runs (DONE flag) are skipped, so re-running only fills gaps.
set -euo pipefail

ROOT=/home/projects/nyosef/zvise/PixelGen/PixelGen
SWEEP=$ROOT/New_Data/sweep
CACHE=$ROOT/New_Data/cache/sweep
LOGS=$SWEEP/logs
mkdir -p "$LOGS" "$CACHE"

SPATIAL=(top500 ct100)
N_LATENT=(10 20 30)
N_LAYERS=(2 3)
BATCH_MASK=(ff ft)
EXPERTS=(poe moe)

n_sub=0; n_skip=0
for sp in "${SPATIAL[@]}"; do
for nl in "${N_LATENT[@]}"; do
for L in "${N_LAYERS[@]}"; do
for bm in "${BATCH_MASK[@]}"; do
for ex in "${EXPERTS[@]}"; do
  rid="sp-${sp}__nl${nl}__L${L}__bm-${bm}__${ex}"
  if [ -f "$CACHE/$rid/DONE" ]; then
    echo "skip  $rid"; n_skip=$((n_skip+1)); continue
  fi
  bsub -J "sw-$rid" -q long-gpu \
    -gpu "num=1:j_exclusive=yes:gmem=16G" \
    -R "rusage[mem=64G]" -R "affinity[thread*8]" \
    -o "$LOGS/$rid.out" -e "$LOGS/$rid.err" \
    -- bash "$SWEEP/run_sweep.lsf" \
       --spatial "$sp" --n-latent "$nl" --n-layers "$L" --batch-mask "$bm" --experts "$ex"
  n_sub=$((n_sub+1))
done; done; done; done; done

echo "submitted=$n_sub skipped=$n_skip"
