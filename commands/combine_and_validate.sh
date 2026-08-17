#!/usr/bin/env bash
set -euo pipefail

echo "[$(date --iso-8601=seconds)] Stage 1/2: validating noisy results"
python -u reviewer_tools/validate_noisy_results.py \
  --targets reviewer_targets/noisy_sanity_targets_all_seeds.csv \
  --raw-root reviewer_results/noisy_sanity/raw_all_seeds \
  --shots 128,512,1024 \
  --simulator-seeds 0,1,2,3,4,5,6,7,8,9 \
  --n-member 100 \
  --n-nonmember 100 \
  --out reviewer_results/noisy_sanity/noisy_validation_report.csv \
  --strict

echo "[$(date --iso-8601=seconds)] Stage 1/2 complete"
echo "[$(date --iso-8601=seconds)] Stage 2/2: combining results and bootstrapping"
python -u reviewer_tools/combine_noisy_results.py \
  --targets reviewer_targets/noisy_sanity_targets_all_seeds.csv \
  --raw-root reviewer_results/noisy_sanity/raw_all_seeds \
  --out-dir reviewer_results/noisy_sanity/combined \
  --attacks loss,confidence,correctness \
  --fpr-points 0.05,0.10 \
  --crossfit-folds 5 \
  --bootstrap 5000 \
  --bootstrap-seed 2026 \
  --progress-every 5

echo "[$(date --iso-8601=seconds)] Stage 2/2 complete"
echo "[$(date --iso-8601=seconds)] All combine-and-validate stages completed successfully"
