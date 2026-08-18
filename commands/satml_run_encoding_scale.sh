#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
TARGETS="satml_targets/credit_scaling_targets.csv"
RUNS="satml_runs/satml_credit_scaling"
RESULTS="satml_results/encoding_scale"

mkdir -p "${RESULTS}" satml_logs
"${PYTHON_BIN}" -u reviewer_tools/run_multiseed_factorial.py \
  --targets "${TARGETS}" --repo-root . --out satml_runs \
  --gpus "${QURIFT_GPUS}" --jobs-per-gpu 1 --cpu-threads 2 --resume \
  2>&1 | tee satml_logs/encoding_scale_train.log

"${PYTHON_BIN}" reviewer_tools/extract_retrained_target_metrics.py \
  --attack-data-dir "${RUNS}" --targets "${TARGETS}" --out-dir "${RESULTS}/target_metrics"
"${PYTHON_BIN}" reviewer_tools/threshold_mia_bootstrap.py \
  --attack-data-dir "${RUNS}" --targets "${TARGETS}" --out-dir "${RESULTS}/threshold_mia" \
  --bootstrap 10000 --bootstrap-seed 2026 --fprs 0.01,0.05,0.10

"${PYTHON_BIN}" satml_tools/analyze_encoding_scale.py \
  --factorial-targets satml_targets/credit_factorial_targets.csv \
  --scaling-targets "${TARGETS}" \
  --factorial-metrics satml_results/credit_factorial/target_metrics/retrained_target_metrics_raw.csv \
  --scaling-metrics "${RESULTS}/target_metrics/retrained_target_metrics_raw.csv" \
  --factorial-attacks satml_results/credit_factorial/threshold_mia/threshold_mia_raw.csv \
  --scaling-attacks "${RESULTS}/threshold_mia/threshold_mia_raw.csv" \
  --out-dir "${RESULTS}/paired_analysis" --bootstrap 10000 --bootstrap-seed 2026
