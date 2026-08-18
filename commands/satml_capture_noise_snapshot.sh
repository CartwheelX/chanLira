#!/usr/bin/env bash
set -euo pipefail

: "${QURIFT_NOISE_BACKEND:?Set QURIFT_NOISE_BACKEND to the IBM backend name}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SNAPSHOT_TAG="${QURIFT_NOISE_SNAPSHOT_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
SNAPSHOT_DIR="satml_results/backend_snapshots/${SNAPSHOT_TAG}"

if [[ -e "${SNAPSHOT_DIR}" ]]; then
  printf '%s\n' "[ERROR] Refusing to overwrite existing snapshot directory: ${SNAPSHOT_DIR}" >&2
  exit 1
fi
mkdir -p "${SNAPSHOT_DIR}" satml_logs
"${PYTHON_BIN}" -u reviewer_tools/probe_ibm_backend_noise.py \
  --backend-name "${QURIFT_NOISE_BACKEND}" \
  --require-noise \
  --out "${SNAPSHOT_DIR}/probe.json" \
  --snapshot-dir "${SNAPSHOT_DIR}" \
  2>&1 | tee "satml_logs/noise_snapshot_${SNAPSHOT_TAG}.log"

"${PYTHON_BIN}" - <<PY
from pathlib import Path
from reviewer_tools.qurift_qiskit_bridge import load_backend_noise_snapshot

path = Path("${SNAPSHOT_DIR}").resolve()
context = load_backend_noise_snapshot(path, require_noise=True)
print(f"[VERIFIED] backend={context.metadata.resolved_backend_name} calibration={context.metadata.calibration_timestamp}")
print(f"export QURIFT_NOISE_SNAPSHOT='{path}'")
PY
