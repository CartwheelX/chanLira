#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" satml_tools/build_fresh_selector_targets.py \
  --development-targets satml_targets/credit_factorial_targets.csv \
  --development-metrics satml_results/credit_factorial/target_metrics/retrained_target_metrics_raw.csv \
  --development-attacks satml_results/credit_factorial/threshold_mia/threshold_mia_raw.csv \
  --out-dir satml_targets/selector \
  --accuracy-tolerance 0.02 \
  --fresh-blocks 5 \
  --regularized-weight-decay 0.001

echo "[OK] The three policies are frozen. Fresh targets have not been inspected or trained yet."
