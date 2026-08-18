#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p satml_logs satml_results/fashion_mechanism satml_results/wdbc_mechanism

"${PYTHON_BIN}" -u satml_tools/analyze_mechanistic_pathway.py \
  --targets satml_targets/fashion_factorial_targets.csv \
  --metrics satml_results/fashion_factorial/target_metrics/retrained_target_metrics_raw.csv \
  --attacks satml_results/fashion_factorial/threshold_mia/threshold_mia_raw.csv \
  --geometry satml_results/fashion_geometry/geometry_raw.csv \
  --out-dir satml_results/fashion_mechanism --bootstrap 10000 --bootstrap-seed 2026 \
  2>&1 | tee satml_logs/fashion_mechanism.log

"${PYTHON_BIN}" -u satml_tools/analyze_mechanistic_pathway.py \
  --targets satml_targets/wdbc_targeted_targets.csv \
  --metrics satml_results/wdbc_targeted/target_metrics/retrained_target_metrics_raw.csv \
  --attacks satml_results/wdbc_targeted/threshold_mia/threshold_mia_raw.csv \
  --geometry satml_results/wdbc_geometry/geometry_raw.csv \
  --out-dir satml_results/wdbc_mechanism --bootstrap 10000 --bootstrap-seed 2026 \
  2>&1 | tee satml_logs/wdbc_mechanism.log
