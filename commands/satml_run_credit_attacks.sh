#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
QURIFT_GPUS="${QURIFT_GPUS:-auto}"
TARGETS="satml_targets/credit_factorial_targets.csv"

mkdir -p satml_logs satml_results/credit_factorial

"${PYTHON_BIN}" -u experiments/gen_results/run_train_mia_attack_cvholdout_multigpu.py \
  --launcher \
  --attack-data-dir satml_runs/satml_credit_factorial \
  --out satml_results/credit_factorial/learned_mia \
  --test-ratio 0.2 \
  --cv-folds 5 \
  --tune \
  --n-trials 20 \
  --max-epochs 150 \
  --patience 15 \
  --device cuda \
  --seed 2026 \
  --cpu-threads 2 \
  --resume \
  --jobs-per-gpu 1 \
  --gpus "${QURIFT_GPUS}" \
  2>&1 | tee satml_logs/credit_learned_mia.log

"${PYTHON_BIN}" -u reviewer_tools/run_lira_reference_multigpu.py \
  --targets "${TARGETS}" \
  --repo-root . \
  --run-root satml_runs \
  --out-dir satml_results/credit_factorial/lira \
  --num-references 16 \
  --bootstrap 10000 \
  --seed 2026 \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu 1 \
  --cpu-threads 2 \
  --resume \
  2>&1 | tee satml_logs/credit_lira.log

"${PYTHON_BIN}" -u reviewer_tools/run_label_only_boundary_multigpu.py \
  --targets "${TARGETS}" \
  --repo-root . \
  --run-root satml_runs \
  --out-dir satml_results/credit_factorial/label_only \
  --n-member 200 \
  --n-nonmember 2000 \
  --anchors 16 \
  --binary-steps 10 \
  --bootstrap 10000 \
  --seed 2026 \
  --gpus "${QURIFT_GPUS}" \
  --jobs-per-gpu 1 \
  --cpu-threads 2 \
  --resume \
  2>&1 | tee satml_logs/credit_label_only.log

"${PYTHON_BIN}" -u satml_tools/analyze_paired_factorial.py \
  --targets "${TARGETS}" \
  --metrics satml_results/credit_factorial/target_metrics/retrained_target_metrics_raw.csv \
  --attack-results satml_results/credit_factorial/threshold_mia/threshold_mia_raw.csv \
  --attack-results satml_results/credit_factorial/learned_mia/attack_summary.csv \
  --attack-results satml_results/credit_factorial/lira/lira_reference_mia_raw.csv \
  --attack-results satml_results/credit_factorial/label_only/label_only_boundary_raw.csv \
  --out-dir satml_results/credit_factorial/paired_all_attacks \
  --bootstrap 10000 \
  --bootstrap-seed 2026
