#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
BOOTSTRAP="${QURIFT_REPORT_BOOTSTRAP:-5000}"

echo "[$(date --iso-8601=seconds)] Stage 1/5: extract architecture target metrics"
"$PYTHON_BIN" -u reviewer_tools/extract_retrained_target_metrics.py \
  --attack-data-dir reviewer_runs/architecture_control \
  --targets reviewer_targets/architecture_control_targets.csv \
  --out-dir reviewer_results/architecture_metrics

echo "[$(date --iso-8601=seconds)] Stage 2/5: audit exact model resources"
"$PYTHON_BIN" -u reviewer_tools/count_model_resources.py \
  --run-root reviewer_runs/multiseed_factorial \
  --targets reviewer_targets/multiseed_factorial_targets.csv \
  --out-dir reviewer_results/factorial_resources \
  --fail-on-missing-exact
"$PYTHON_BIN" -u reviewer_tools/count_model_resources.py \
  --run-root reviewer_runs/architecture_control \
  --targets reviewer_targets/architecture_control_targets.csv \
  --out-dir reviewer_results/architecture_resources \
  --fail-on-missing-exact

echo "[$(date --iso-8601=seconds)] Stage 3/5: architecture threshold attacks"
"$PYTHON_BIN" -u reviewer_tools/threshold_mia_bootstrap.py \
  --attack-data-dir reviewer_runs/architecture_control \
  --targets reviewer_targets/architecture_control_targets.csv \
  --out-dir reviewer_results/architecture_threshold_mia \
  --bootstrap 10000 \
  --bootstrap-seed 2026 \
  --threshold-folds 5 \
  --threshold-seed 2026 \
  --fprs 0.05,0.10 \
  --attacks loss \
  --skip-high-low-contrasts

echo "[$(date --iso-8601=seconds)] Stage 4/5: paired architecture analysis"
"$PYTHON_BIN" -u reviewer_tools/architecture_control_analysis.py \
  --metrics reviewer_results/architecture_metrics/retrained_target_metrics_raw.csv \
  --mia reviewer_results/architecture_threshold_mia/threshold_mia_raw.csv \
  --resources reviewer_results/architecture_resources/model_resources_raw.csv \
  --out-dir reviewer_results/architecture_control \
  --bootstrap "$BOOTSTRAP" \
  --bootstrap-seed 2026

echo "[$(date --iso-8601=seconds)] Stage 5/5: tables, figures, and reviewer index"
"$PYTHON_BIN" -u reviewer_tools/generate_reviewer_artifacts.py \
  --repo-root . \
  --results-root reviewer_results \
  --out-dir reviewer_results/reviewer_artifacts \
  --bootstrap "$BOOTSTRAP" \
  --bootstrap-seed 2026

echo "[$(date --iso-8601=seconds)] Reviewer artifacts completed"
