#!/usr/bin/env bash
set -euo pipefail

mkdir -p reviewer_logs reviewer_results/lira_reference_mia reviewer_results/label_only_boundary

python reviewer_tools/run_lira_reference_multigpu.py \
  --targets reviewer_targets/multiseed_factorial_targets.csv \
  --repo-root . \
  --run-root reviewer_runs \
  --out-dir reviewer_results/lira_reference_mia \
  --num-references 16 \
  --bootstrap 5000 \
  --seed 2026 \
  --gpus 0,1,2,3,4,5,6,7 \
  --jobs-per-gpu 1 \
  --cpu-threads 2 \
  --phase all \
  --resume \
  2>&1 | tee reviewer_logs/lira_reference_mia.log

python reviewer_tools/run_label_only_boundary_multigpu.py \
  --targets reviewer_targets/multiseed_factorial_targets.csv \
  --repo-root . \
  --run-root reviewer_runs \
  --out-dir reviewer_results/label_only_boundary \
  --anchors 16 \
  --binary-steps 10 \
  --norm l2 \
  --query-batch-size 64 \
  --bootstrap 5000 \
  --seed 2026 \
  --gpus 0,1,2,3,4,5,6,7 \
  --jobs-per-gpu 1 \
  --cpu-threads 2 \
  --resume \
  2>&1 | tee reviewer_logs/label_only_boundary.log
