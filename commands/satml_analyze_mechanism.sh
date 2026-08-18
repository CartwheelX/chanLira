#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p satml_results/mechanistic_pathway satml_logs

"${PYTHON_BIN}" -u satml_tools/analyze_mechanistic_pathway.py \
  --targets satml_targets/credit_factorial_targets.csv \
  --metrics satml_results/credit_factorial/target_metrics/retrained_target_metrics_raw.csv \
  --attacks satml_results/credit_factorial/threshold_mia/threshold_mia_raw.csv \
  --geometry satml_results/credit_geometry/geometry_raw.csv \
  --out-dir satml_results/mechanistic_pathway \
  --bootstrap 10000 --bootstrap-seed 2026 \
  2>&1 | tee satml_logs/mechanistic_pathway.log
