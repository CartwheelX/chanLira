#!/usr/bin/env bash
set -euo pipefail

: "${QURIFT_LEGACY_REPO:?Set QURIFT_LEGACY_REPO to the completed NeurIPS repository path}"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p satml_results

"${PYTHON_BIN}" satml_tools/build_noise_study_targets.py

"${PYTHON_BIN}" satml_tools/import_legacy_mnist.py \
  --source-repo "${QURIFT_LEGACY_REPO}" \
  --destination-repo . \
  --targets satml_targets/noise/mnist_noise_n1_structural_targets.csv \
  --out satml_results/imported_mnist_manifest.json
