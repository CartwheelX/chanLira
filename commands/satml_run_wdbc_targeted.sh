#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
QURIFT_JOBS_PER_GPU="${QURIFT_JOBS_PER_GPU:-1}"
mkdir -p satml_logs satml_runs

"${PYTHON_BIN}" -u reviewer_tools/run_multiseed_factorial.py \
  --targets satml_targets/wdbc_targeted_targets.csv \
  --repo-root . --out satml_runs \
  --gpus "${QURIFT_GPUS}" --jobs-per-gpu "${QURIFT_JOBS_PER_GPU}" \
  --cpu-threads 2 --resume \
  2>&1 | tee satml_logs/wdbc_targeted.log
