#!/usr/bin/env bash
set -euo pipefail

: "${QURIFT_LEGACY_REPO:?Set QURIFT_LEGACY_REPO to the completed NeurIPS repository path}"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p satml_results

"${PYTHON_BIN}" satml_tools/import_legacy_mnist.py \
  --source-repo "${QURIFT_LEGACY_REPO}" \
  --destination-repo . \
  --targets reviewer_targets/noisy_sanity_targets_core.csv \
  --out satml_results/imported_mnist_manifest.json
