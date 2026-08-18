#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"

mkdir -p satml_logs satml_results/credit_geometry

"${PYTHON_BIN}" -u reviewer_tools/run_multiseed_geometry.py \
  --targets satml_targets/credit_geometry_targets.csv \
  --repo-root . \
  --out-dir satml_results/credit_geometry \
  --seeds 20261,20262,20263,20264,20265,20266,20267,20268 \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu 1 \
  --n-train 200 \
  --n-test 200 \
  --batch-size 16 \
  --bootstrap 10000 \
  --bootstrap-seed 2026 \
  --resume \
  2>&1 | tee satml_logs/credit_geometry.log
