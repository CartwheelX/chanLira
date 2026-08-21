#!/usr/bin/env python3
"""Audit prerequisites for true circuit-executed noisy-reference LiRA."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def reference_candidates(root: Path, cell: str, reference_id: int) -> list[Path]:
    """Find retained legacy and newly canonicalized reference-bank layouts."""
    filename = f"reference_{reference_id:03d}.npz"
    directories = [root / cell]
    directories.extend(sorted(
        path for path in root.glob(f"{cell}_wd*") if path.is_dir()
    ))
    return [directory / filename for directory in directories]


def select_reference_path(root: Path, cell: str, reference_id: int) -> Path:
    candidates = reference_candidates(root, cell, reference_id)
    paired = [
        path for path in candidates
        if path.is_file() and path.with_suffix(".pt").is_file()
    ]
    if paired:
        return paired[0]
    existing = [path for path in candidates if path.is_file()]
    return existing[0] if existing else candidates[0]


def inspect_reference(path: Path, expected_id: int, expected_count: int) -> dict[str, object]:
    checkpoint = path.with_suffix(".pt")
    record: dict[str, object] = {
        "reference_id": expected_id,
        "score_path": str(path.resolve()),
        "score_exists": path.is_file(),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_exists": checkpoint.is_file() and checkpoint.stat().st_size > 0,
        "metadata_valid": False,
        "error": "",
    }
    if not path.is_file():
        record["error"] = "missing exact reference score file"
        return record
    try:
        with np.load(path, allow_pickle=False) as saved:
            required = {
                "scores", "inclusion", "sample_ids", "candidate_fingerprint",
                "target_template", "reference_id", "num_references", "reference_seed",
            }
            missing = sorted(required - set(saved.files))
            if missing:
                raise ValueError(f"missing arrays {missing}")
            if int(saved["reference_id"]) != expected_id:
                raise ValueError("reference ID mismatch")
            if int(saved["num_references"]) != expected_count:
                raise ValueError("reference-count mismatch")
            if len(saved["scores"]) != len(saved["sample_ids"]):
                raise ValueError("score/sample-ID length mismatch")
            record.update({
                "metadata_valid": True,
                "candidate_count": int(len(saved["sample_ids"])),
                "candidate_fingerprint": str(saved["candidate_fingerprint"]),
                "target_template": str(saved["target_template"]),
                "reference_seed": int(saved["reference_seed"]),
            })
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transfer-config", type=Path,
        default=Path("channel_lira_results/transfer_phase3/experiment_config.json"),
    )
    parser.add_argument(
        "--reference-root", type=Path,
        default=Path("reviewer_results/lira_reference_mia/reference_models"),
    )
    parser.add_argument(
        "--backend-snapshot", type=Path,
        default=Path("reviewer_results/noisy_reference_phase5/backend_snapshot"),
    )
    parser.add_argument("--num-references", type=int, default=16)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("channel_lira_results/noisy_reference_phase5"),
    )
    parser.add_argument(
        "--require-ready", action="store_true",
        help="Exit nonzero unless every prerequisite is available.",
    )
    args = parser.parse_args()
    transfer_config = args.transfer_config.resolve()
    reference_root = args.reference_root.resolve()
    backend_snapshot = args.backend_snapshot.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(transfer_config.read_text(encoding="utf-8"))
    cells = [str(value) for value in config["cells"]]
    references = []
    for cell in cells:
        for reference_id in range(args.num_references):
            path = select_reference_path(reference_root, cell, reference_id)
            record = inspect_reference(path, reference_id, args.num_references)
            record["cell"] = cell
            record["candidate_score_paths"] = [
                str(candidate.resolve())
                for candidate in reference_candidates(reference_root, cell, reference_id)
            ]
            references.append(record)
    exact_ready = sum(
        bool(row["score_exists"] and row["metadata_valid"]) for row in references
    )
    checkpoint_ready = sum(bool(row["checkpoint_exists"]) for row in references)
    snapshot_manifest = backend_snapshot / "snapshot_manifest.json"
    snapshot_ready = snapshot_manifest.is_file()
    fingerprints = sorted({
        str(row.get("candidate_fingerprint", ""))
        for row in references if row.get("metadata_valid")
    })
    candidates = sorted({
        int(row.get("candidate_count", 0))
        for row in references if row.get("metadata_valid")
    })
    expected = len(cells) * args.num_references
    ready = (
        exact_ready == expected
        and checkpoint_ready == expected
        and snapshot_ready
        and len(fingerprints) == 1
        and len(candidates) == 1
    )
    cell_argument = ",".join(cells)
    retrain_command = (
        "python3 reviewer_tools/run_lira_reference_multigpu.py "
        "--targets reviewer_targets/multiseed_factorial_targets.csv "
        "--out-dir reviewer_results/lira_reference_mia "
        f"--cells {cell_argument} --num-references {args.num_references} "
        "--save-reference-checkpoints --phase train --resume"
    )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ready": ready,
        "cells": cells,
        "num_references_per_cell": args.num_references,
        "expected_reference_checkpoints": expected,
        "valid_exact_reference_files": exact_ready,
        "available_reference_checkpoints": checkpoint_ready,
        "missing_reference_checkpoints": expected - checkpoint_ready,
        "candidate_counts": candidates,
        "candidate_fingerprints": fingerprints,
        "backend_snapshot": str(backend_snapshot),
        "backend_snapshot_manifest": str(snapshot_manifest),
        "backend_snapshot_ready": snapshot_ready,
        "retrain_command": retrain_command,
        "references": references,
    }
    (out_dir / "READINESS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    blockers = []
    if checkpoint_ready != expected:
        blockers.append(
            f"{expected - checkpoint_ready}/{expected} trained reference checkpoints are missing"
        )
    if not snapshot_ready:
        blockers.append("a reconstructable frozen backend snapshot is missing")
    if len(fingerprints) != 1 or len(candidates) != 1:
        blockers.append("reference candidate banks are inconsistent")
    blocker_lines = "\n".join(f"- {blocker}" for blocker in blockers) or "- none"
    status = "READY" if ready else "BLOCKED"
    report = f"""# True noisy-reference LiRA readiness

## Status: {status}

The existing exact reference banks are validated independently from trained model
weights. True noisy-reference LiRA must execute retained reference QNN checkpoints
through a reconstructable backend noise snapshot; Gaussian channel draws do not
satisfy this requirement.

| Item | Available | Required |
|---|---:|---:|
| Valid exact reference files | {exact_ready} | {expected} |
| Trained reference checkpoints | {checkpoint_ready} | {expected} |
| Frozen backend snapshot manifest | {int(snapshot_ready)} | 1 |

## Blockers

{blocker_lines}

The Phase-3 target outputs were generated from an IBM calibration dated
2026-07-30, but their summary metadata is insufficient to reconstruct the full
noise model. For a controlled comparison, capture a new frozen snapshot and rerun
both targets and reference checkpoints under that same snapshot, or recover the
original complete snapshot.

## New reference-ensemble retraining command

```bash
{retrain_command}
```

This command is deliberately scoped to the five Phase-3 cells and requests saved
weights. The runner accepts the retained base cell names even though newly trained
banks use the canonical `*_wd0` layout. The readiness audit prefers a complete
score/checkpoint pair and will detect that layout on its next run. The command does
not start automatically from this readiness audit.
"""
    (out_dir / "READINESS.md").write_text(report, encoding="utf-8")
    print(
        f"{status}: exact={exact_ready}/{expected}, checkpoints={checkpoint_ready}/{expected}, "
        f"snapshot={snapshot_ready}; wrote {out_dir}",
        flush=True,
    )
    if args.require_ready and not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
