#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TARGETS="satml_targets/credit_factorial_targets.csv"
RUNS="satml_runs/satml_credit_factorial"
RESULTS="satml_results/credit_factorial"

mkdir -p "${RESULTS}" satml_logs

"${PYTHON_BIN}" -u reviewer_tools/extract_retrained_target_metrics.py \
  --attack-data-dir "${RUNS}" \
  --targets "${TARGETS}" \
  --out-dir "${RESULTS}/target_metrics"

"${PYTHON_BIN}" -u satml_tools/analyze_capacity_controls.py \
  --targets "${TARGETS}" \
  --metrics "${RESULTS}/target_metrics/retrained_target_metrics_raw.csv" \
  --out-dir "${RESULTS}/capacity_controls"

"${PYTHON_BIN}" -u reviewer_tools/threshold_mia_bootstrap.py \
  --attack-data-dir "${RUNS}" \
  --targets "${TARGETS}" \
  --out-dir "${RESULTS}/threshold_mia" \
  --bootstrap 10000 \
  --bootstrap-seed 2026 \
  --threshold-folds 5 \
  --threshold-seed 2026 \
  --fprs 0.01,0.05,0.10 \
  2>&1 | tee satml_logs/credit_threshold_mia.log

"${PYTHON_BIN}" -u satml_tools/analyze_paired_factorial.py \
  --targets "${TARGETS}" \
  --metrics "${RESULTS}/target_metrics/retrained_target_metrics_raw.csv" \
  --attack-results "${RESULTS}/threshold_mia/threshold_mia_raw.csv" \
  --out-dir "${RESULTS}/paired_analysis" \
  --bootstrap 10000 \
  --bootstrap-seed 2026

"${PYTHON_BIN}" satml_tools/validate_satml_study.py \
  --targets "${TARGETS}" \
  --expected-blocks 8 \
  --credit-data data/credit_default/credit_default.csv.gz \
  --run-root satml_runs \
  --attacks "${RESULTS}/threshold_mia/threshold_mia_raw.csv" \
  --out "${RESULTS}/protocol_validation.json"
