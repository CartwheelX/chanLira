#!/usr/bin/env python3
"""Validate noisy subset completeness, strict noise loading, and sample pairing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--shots", default="128,512,1024")
    parser.add_argument("--simulator-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--n-member", type=int, default=100)
    parser.add_argument("--n-nonmember", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("reviewer_results/noisy_sanity/noisy_validation_report.csv"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    shots = [int(value) for value in args.shots.split(",") if value.strip()]
    simulator_seeds = [int(value) for value in args.simulator_seeds.split(",") if value.strip()]
    expected_conditions = 1 + 2 * len(shots) * len(simulator_seeds)
    targets = pd.read_csv(args.targets)
    rows: List[Dict[str, object]] = []

    for _, target in targets.iterrows():
        target_id = str(target["target_id"])
        directory = args.raw_root / target_id
        status_path = directory / "condition_status.csv"
        predictions_path = directory / "per_sample_predictions.csv"
        manifest_path = directory / "sample_manifest.csv"
        backend = read_json(directory / "backend_noise_metadata.json")
        status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
        predictions = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()
        manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()

        status_bad = 0 if status.empty else int(status["status"].astype(str).isin(["error", "skipped_no_noise_model"]).sum())
        manifest_members = 0 if manifest.empty else int((manifest["membership"].astype(int) == 1).sum())
        manifest_nonmembers = 0 if manifest.empty else int((manifest["membership"].astype(int) == 0).sum())
        n_conditions = 0
        paired_samples = False
        if not predictions.empty:
            condition_cols = ["mode", "shots", "simulator_seed"]
            n_conditions = int(predictions.groupby(condition_cols, dropna=False).ngroups)
            expected_ids = set(manifest["sample_id"].astype(str)) if not manifest.empty else set()
            paired_samples = all(
                set(group["sample_id"].astype(str)) == expected_ids
                for _, group in predictions.groupby(condition_cols, dropna=False)
            ) if expected_ids else False

        passed = all(
            [
                directory.exists(),
                backend.get("noise_model_loaded") is True,
                status_bad == 0,
                n_conditions >= expected_conditions,
                manifest_members == args.n_member,
                manifest_nonmembers == args.n_nonmember,
                paired_samples,
            ]
        )
        rows.append({
            "target_id": target_id,
            "structural_cell_id": target.get("structural_cell_id", ""),
            "model_seed": target.get("model_seed", ""),
            "target_directory_exists": directory.exists(),
            "noise_model_loaded": backend.get("noise_model_loaded"),
            "noise_backend": backend.get("resolved_noise_backend_name", backend.get("resolved_backend_name")),
            "calibration_timestamp": backend.get("calibration_timestamp"),
            "n_status_errors_or_skips": status_bad,
            "n_conditions": n_conditions,
            "expected_conditions": expected_conditions,
            "n_member": manifest_members,
            "n_nonmember": manifest_nonmembers,
            "samples_identical_across_conditions": paired_samples,
            "passed": passed,
        })

    report = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    summary = {
        "n_targets": int(len(report)),
        "n_passed": int(report["passed"].sum()) if not report.empty else 0,
        "n_failed": int((~report["passed"]).sum()) if not report.empty else 0,
        "all_passed": bool(not report.empty and report["passed"].all()),
        "report": str(args.out),
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.strict and not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
