#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p satml_logs satml_results/paper_artifacts

"${PYTHON_BIN}" -u satml_tools/generate_satml_artifacts.py \
  --out-dir satml_results/paper_artifacts \
  2>&1 | tee satml_logs/generate_satml_artifacts.log

printf 'Tables:  satml_results/paper_artifacts/tables/satml_tables.md\n'
printf 'LaTeX:   satml_results/paper_artifacts/tables/satml_tables.tex\n'
printf 'Figures: satml_results/paper_artifacts/figures/\n'
