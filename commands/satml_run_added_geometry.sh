#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
mkdir -p satml_logs satml_results/fashion_geometry satml_results/wdbc_geometry

"${PYTHON_BIN}" -u reviewer_tools/run_multiseed_geometry.py \
  --targets satml_targets/fashion_geometry_targets.csv --repo-root . \
  --out-dir satml_results/fashion_geometry --seeds 60261,60262,60263,60264,60265 \
  --gpus "${QURIFT_GPUS}" --jobs-per-gpu 1 --n-train 200 --n-test 200 \
  --batch-size 16 --bootstrap 10000 --bootstrap-seed 2026 --resume \
  2>&1 | tee satml_logs/fashion_geometry.log

"${PYTHON_BIN}" -u reviewer_tools/run_multiseed_geometry.py \
  --targets satml_targets/wdbc_geometry_targets.csv --repo-root . \
  --out-dir satml_results/wdbc_geometry --seeds 80261,80262,80263,80264,80265 \
  --gpus "${QURIFT_GPUS}" --jobs-per-gpu 1 --n-train 160 --n-test 200 \
  --batch-size 16 --bootstrap 10000 --bootstrap-seed 2026 --resume \
  2>&1 | tee satml_logs/wdbc_geometry.log
