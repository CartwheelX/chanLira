#!/usr/bin/env bash
set -euo pipefail

: "${QURIFT_NOISE_BACKEND:?Set QURIFT_NOISE_BACKEND to the IBM backend name}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_TARGET_IDS="MNIST_QNN_eff_su2_r1_d2_s43,MNIST_QNN_eff_su2_r5_d2_s43,MNIST_QNN_z_r1_d6_s43,MNIST_QNN_zz_r1_d6_s43,MNIST_QNN_zz_r5_d6_s43"
TARGET_IDS="${QURIFT_NOISE_TARGET_IDS:-${DEFAULT_TARGET_IDS}}"

mkdir -p satml_results/backend_snapshots satml_results/noise_budget satml_logs
SNAPSHOT_TAG="$(date -u +%Y%m%dT%H%M%SZ)"

IFS=',' read -r -a TARGET_ARRAY <<< "${TARGET_IDS}"
for TARGET_ID in "${TARGET_ARRAY[@]}"; do
  TARGET_ID="${TARGET_ID//[[:space:]]/}"
  [[ -n "${TARGET_ID}" ]] || continue
  MODEL_PATH="reviewer_runs/multiseed_factorial/${TARGET_ID}/target_model.pt"
  if [[ ! -s "${MODEL_PATH}" ]]; then
    printf 'Missing checkpoint: %s\n' "${MODEL_PATH}" >&2
    printf 'Import or regenerate the retained MNIST checkpoints before the noise study.\n' >&2
    exit 1
  fi
done

"${PYTHON_BIN}" reviewer_tools/probe_ibm_backend_noise.py \
  --backend-name "${QURIFT_NOISE_BACKEND}" \
  --require-noise \
  --out "satml_results/backend_snapshots/${SNAPSHOT_TAG}.json" \
  --snapshot-dir "satml_results/backend_snapshots/${SNAPSHOT_TAG}"

for TARGET_ID in "${TARGET_ARRAY[@]}"; do
  TARGET_ID="${TARGET_ID//[[:space:]]/}"
  [[ -n "${TARGET_ID}" ]] || continue
  "${PYTHON_BIN}" -u reviewer_tools/qurift_noisy_eval.py \
    --repo-root . \
    --targets reviewer_targets/multiseed_factorial_targets.csv \
    --target-id "${TARGET_ID}" \
    --run-root reviewer_runs \
    --out-dir "satml_results/noise_budget/${SNAPSHOT_TAG}" \
    --modes exact,ideal_shot,noisy_shot \
    --query-shot-pairs 1x2560,5x512,20x128 \
    --simulator-seeds 0,1,2,3,4,5,6,7,8,9 \
    --backend-name "${QURIFT_NOISE_BACKEND}" \
    --require-noise \
    --n-member 100 --n-nonmember 100 \
    --sample-seed 2026 --transpiler-seed 2026 --optimization-level 1 \
    --qiskit-batch-size 10 --device cuda --resume \
    2>&1 | tee "satml_logs/noise_budget_${SNAPSHOT_TAG}_${TARGET_ID}.log"
done

"${PYTHON_BIN}" satml_tools/analyze_noise_budget.py \
  --root satml_results/noise_budget \
  --out-dir satml_results/noise_budget/combined
