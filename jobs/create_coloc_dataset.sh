#!/bin/bash

# To run (example):
# conda activate pixelgen-scvi
# ./PixelGen/jobs/create_coloc_dataset.sh ./PixelGen/datasets/pbmc-pha-v2.0/combined_resting_PHA_data.pxl pbmcs_pha_with_hs 16 True


bsub \
  -J coloc-dataset \
  -q long-gpu \
  -o PixelGen/jobs/create_coloc_dataset.out \
  -gpu num=1:j_exclusive=yes:gmem=32G \
  -R rusage[mem=512G] \
  -R affinity[thread*16] \
  -- "bash PixelGen/jobs/create_coloc_dataset.lsf $@"