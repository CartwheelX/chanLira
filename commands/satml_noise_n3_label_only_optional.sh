#!/usr/bin/env bash
set -euo pipefail

: "${QURIFT_NOISE_SNAPSHOT:?Set QURIFT_NOISE_SNAPSHOT to the same frozen snapshot used for N1/N2/N3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TARGETS="satml_targets/noise/mnist_noise_n3_label_targets.csv"
OUT="satml_results/noise/n3_attack_breadth/noisy_label_only_optional"

mkdir -p "${OUT}" satml_logs
"${PYTHON_BIN}" satml_tools/build_noise_study_targets.py

while IFS=, read -r TARGET_ID _; do
  [[ "${TARGET_ID}" == "target_id" ]] && continue
  [[ -n "${TARGET_ID}" ]] || continue
  "${PYTHON_BIN}" -u satml_tools/noisy_label_only.py \
    --repo-root . --targets "${TARGETS}" --run-root reviewer_runs \
    --out-dir "${OUT}" --snapshot "${QURIFT_NOISE_SNAPSHOT}" \
    --target-id "${TARGET_ID}" --shots 512 --simulator-seeds 0 \
    --n-member 50 --n-nonmember 50 --anchors 8 --binary-steps 8 \
    --bootstrap 1000 --seed 2026 --device cuda \
    2>&1 | tee "satml_logs/noise_n3_label_${TARGET_ID}.log"
done < "${TARGETS}"

"${PYTHON_BIN}" satml_tools/noisy_label_only.py \
  --targets "${TARGETS}" --out-dir "${OUT}" \
  --snapshot "${QURIFT_NOISE_SNAPSHOT}" --aggregate
