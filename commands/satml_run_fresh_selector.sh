#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
TARGETS="satml_targets/selector/fresh_selector_targets.csv"
RUNS="satml_runs/satml_selector_fresh"
RESULTS="satml_results/selector_fresh"

mkdir -p "${RESULTS}" satml_logs
"${PYTHON_BIN}" -u reviewer_tools/run_multiseed_factorial.py \
  --targets "${TARGETS}" --repo-root . --out satml_runs \
  --gpus "${QURIFT_GPUS}" --jobs-per-gpu 1 --cpu-threads 2 --resume \
  2>&1 | tee satml_logs/selector_fresh_train.log

"${PYTHON_BIN}" reviewer_tools/extract_retrained_target_metrics.py \
  --attack-data-dir "${RUNS}" --targets "${TARGETS}" --out-dir "${RESULTS}/target_metrics"
"${PYTHON_BIN}" reviewer_tools/threshold_mia_bootstrap.py \
  --attack-data-dir "${RUNS}" --targets "${TARGETS}" --out-dir "${RESULTS}/threshold_mia" \
  --bootstrap 10000 --bootstrap-seed 2026 --fprs 0.01,0.05,0.10

"${PYTHON_BIN}" -u experiments/gen_results/run_train_mia_attack_cvholdout_multigpu.py \
  --launcher --attack-data-dir "${RUNS}" --out "${RESULTS}/learned_mia" \
  --test-ratio 0.2 --cv-folds 5 --tune --n-trials 20 --max-epochs 150 --patience 15 \
  --device cuda --seed 2026 --cpu-threads 2 --resume --jobs-per-gpu 1 --gpus "${QURIFT_GPUS}"
"${PYTHON_BIN}" -u reviewer_tools/run_lira_reference_multigpu.py \
  --targets "${TARGETS}" --repo-root . --run-root satml_runs --out-dir "${RESULTS}/lira" \
  --num-references 16 --bootstrap 10000 --seed 2026 --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu 1 --cpu-threads 2 --resume
"${PYTHON_BIN}" -u reviewer_tools/run_label_only_boundary_multigpu.py \
  --targets "${TARGETS}" --repo-root . --run-root satml_runs --out-dir "${RESULTS}/label_only" \
  --n-member 200 --n-nonmember 2000 --anchors 16 --binary-steps 10 --bootstrap 10000 \
  --seed 2026 --gpus "${QURIFT_GPUS}" --jobs-per-gpu 1 --cpu-threads 2 --resume

"${PYTHON_BIN}" satml_tools/analyze_fresh_selector.py \
  --targets "${TARGETS}" \
  --metrics "${RESULTS}/target_metrics/retrained_target_metrics_raw.csv" \
  --attacks "${RESULTS}/threshold_mia/threshold_mia_raw.csv" \
  --attacks "${RESULTS}/learned_mia/attack_summary.csv" \
  --attacks "${RESULTS}/lira/lira_reference_mia_raw.csv" \
  --attacks "${RESULTS}/label_only/label_only_boundary_raw.csv" \
  --out-dir "${RESULTS}/paired_analysis" --bootstrap 10000 --bootstrap-seed 2026
