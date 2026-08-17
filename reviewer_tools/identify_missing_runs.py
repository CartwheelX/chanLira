#!/usr/bin/env python3
"""Identify failed, incomplete, or missing reviewer runs and seeds."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from reviewer_common import atomic_write_csv, write_analysis_metadata

ERROR_RE = re.compile(r"(traceback|cuda out of memory|runtimeerror|valueerror|error:)", re.I)


def log_tail(path: Path, lines: int = 25) -> tuple[str, bool]:
    if not path.exists() or path.stat().st_size == 0:
        return "", False
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(content[-lines:])
        return tail, bool(ERROR_RE.search(tail))
    except Exception:
        return "", False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--status", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reviewer_results/run_inventory"),
    )
    args = parser.parse_args()

    targets = pd.read_csv(args.targets)
    if "target_id" not in targets.columns:
        raise SystemExit("Targets CSV has no target_id column")

    rows = []
    for _, target in targets.iterrows():
        experiment = str(target.get("experiment", "reviewer"))
        target_id = str(target["target_id"])
        output_dir = args.run_root / experiment / target_id
        model_path = output_dir / "target_model.pt"
        attack_path = output_dir / "target_attack_data.pt"
        log_path = output_dir / "train.log"
        metrics_path = output_dir / "target_export_summary.json"
        tail, error_detected = log_tail(log_path)
        record = target.to_dict()
        model_ok = model_path.exists() and model_path.stat().st_size > 0
        attack_ok = attack_path.exists() and attack_path.stat().st_size > 0
        record.update(
            {
                "expected_output_dir": str(output_dir),
                "model_exists": model_ok,
                "attack_exists": attack_ok,
                "log_exists": log_path.exists() and log_path.stat().st_size > 0,
                "metrics_json_exists": metrics_path.exists() and metrics_path.stat().st_size > 0,
                "model_bytes": model_path.stat().st_size if model_path.exists() else 0,
                "attack_bytes": attack_path.stat().st_size if attack_path.exists() else 0,
                "artifact_complete": bool(model_ok and attack_ok),
                "log_error_detected": error_detected,
                "log_tail": tail,
            }
        )
        rows.append(record)
    inventory = pd.DataFrame(rows)

    if args.status and args.status.exists():
        status = pd.read_csv(args.status).drop_duplicates("target_id", keep="last")
        status = status.add_prefix("launcher_")
        inventory = inventory.merge(
            status,
            left_on="target_id",
            right_on="launcher_target_id",
            how="left",
        )

    missing = inventory[~inventory["artifact_complete"]].copy()
    grouping = [
        column
        for column in (
            "experiment",
            "dataset",
            "architecture",
            "fm_kind",
            "reps",
            "depth",
            "model_seed",
            "data_seed",
        )
        if column in inventory.columns
    ]
    if grouping:
        completion = (
            inventory.groupby(grouping, dropna=False)["artifact_complete"]
            .agg(["count", "sum", "mean"])
            .reset_index()
            .rename(
                columns={
                    "count": "expected",
                    "sum": "complete",
                    "mean": "completion_fraction",
                }
            )
        )
    else:
        completion = pd.DataFrame()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = args.out_dir / "expected_run_inventory.csv"
    missing_path = args.out_dir / "missing_runs.csv"
    completion_path = args.out_dir / "seed_completion.csv"
    atomic_write_csv(inventory, inventory_path)
    atomic_write_csv(missing, missing_path)
    atomic_write_csv(completion, completion_path)
    write_analysis_metadata(
        args.out_dir / "analysis_metadata.json",
        script=Path(__file__).name,
        inputs=[str(args.targets), str(args.run_root), str(args.status or "")],
        outputs=[str(inventory_path), str(missing_path), str(completion_path)],
        ci_method="none",
        bootstrap_unit="none",
        bootstrap_replicates=0,
        notes="A run is complete only when both model and attack payload exist and are nonempty.",
    )
    print(
        f"[OK] complete={int(inventory['artifact_complete'].sum())}/{len(inventory)}; "
        f"missing={len(missing)}"
    )


if __name__ == "__main__":
    main()
