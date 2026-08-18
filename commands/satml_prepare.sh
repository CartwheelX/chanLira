#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p satml_logs satml_results satml_runs satml_targets

if [[ ! -s data/credit_default/credit_default.csv.gz ]]; then
  "${PYTHON_BIN}" satml_tools/fetch_credit_default.py \
    --source auto \
    --out data/credit_default/credit_default.csv.gz
else
  echo "[SKIP] Reusing pinned Credit snapshot. Its checksum will be validated."
fi

if [[ ! -s data/wdbc/wdbc.csv.gz || ! -s data/wdbc/wdbc.csv.gz.manifest.json ]]; then
  "${PYTHON_BIN}" satml_tools/fetch_wdbc.py \
    --source auto \
    --out data/wdbc/wdbc.csv.gz
else
  echo "[SKIP] Reusing pinned WDBC snapshot. Its checksum will be validated."
fi

"${PYTHON_BIN}" satml_tools/build_satml_targets.py \
  --out-dir satml_targets \
  --blocks 8 \
  --scaling-blocks 5

"${PYTHON_BIN}" satml_tools/build_added_dataset_targets.py \
  --out-dir satml_targets

"${PYTHON_BIN}" satml_tools/build_noise_study_targets.py

"${PYTHON_BIN}" satml_tools/validate_satml_study.py \
  --targets satml_targets/credit_factorial_targets.csv \
  --expected-blocks 8 \
  --credit-data data/credit_default/credit_default.csv.gz \
  --out satml_results/design_validation.json

"${PYTHON_BIN}" satml_tools/validate_added_datasets.py \
  --targets satml_targets/fashion_factorial_targets.csv \
  --dataset fashion_mnist \
  --out satml_results/fashion_design_validation.json

"${PYTHON_BIN}" satml_tools/validate_added_datasets.py \
  --targets satml_targets/wdbc_targeted_targets.csv \
  --dataset breast_cancer_wdbc \
  --wdbc-data data/wdbc/wdbc.csv.gz \
  --out satml_results/wdbc_design_validation.json

echo "[OK] SaTML snapshots, frozen manifests, and design validations are ready."
