#!/usr/bin/env bash
set -euo pipefail
BACKEND="${QURIFT_NOISE_BACKEND:?Set QURIFT_NOISE_BACKEND to the IBM backend name}"
ACCOUNT_ARGS=()
if [[ -n "${QURIFT_IBM_ACCOUNT_NAME:-}" ]]; then
  ACCOUNT_ARGS=(--ibm-account-name "$QURIFT_IBM_ACCOUNT_NAME")
fi

python reviewer_tools/run_noisy_sanity_subset.py \
  --targets reviewer_targets/noisy_sanity_targets_core.csv \
  --repo-root . \
  --run-root reviewer_runs \
  --out-dir reviewer_results/noisy_sanity/raw_core \
  --backend-name "$BACKEND" \
  "${ACCOUNT_ARGS[@]}" \
  --modes exact,ideal_shot,noisy_shot \
  --shots 128,512,1024 \
  --simulator-seeds 0,1,2,3,4,5,6,7,8,9 \
  --transpiler-seed 2026 \
  --optimization-level 1 \
  --n-member 100 \
  --n-nonmember 100 \
  --sample-seed 2026 \
  --gpus 0,1,2,3,4,5,6,7 \
  --jobs-per-gpu 1 \
  --cpu-threads 2 \
  --resume
