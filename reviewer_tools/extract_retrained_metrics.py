#!/usr/bin/env python3
"""Extract retrained train/validation/test metrics from every attack payload."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from reviewer_common import (
    atomic_write_csv,
    find_attack_files,
    flatten_scalar_meta,
    membership_convention,
    normalize_membership,
    split_metric,
    torch_load,
    write_analysis_metadata,
)

PAIR_RE = re.compile(
    r"(?P<pair_id>.+?_z\d+_zz\d+)_(?P<fm_kind>z|zz)_s(?P<seed>\d+)$",
    re.I,
)


def coalesce_columns(frame: pd.DataFrame, field: str) -> None:
    candidates = [
        field,
        f"{field}_target",
        f"meta_{field}",
        f"{field}_inferred",
    ]
    present = [name for name in candidates if name in frame.columns]
    if not present:
        return
    result = frame[present[0]]
    for name in present[1:]:
        result = result.where(result.notna(), frame[name])
    frame[field] = result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-data-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reviewer_results/retrained_metrics"),
    )
    args = parser.parse_args()

    files = find_attack_files(args.attack_data_dir)
    if not files:
        raise SystemExit(f"No attack payloads found under {args.attack_data_dir}")

    rows: list[Dict[str, Any]] = []
    for path in files:
        row: Dict[str, Any] = {
            "source_file": str(path),
            "status": "ok",
            "error": "",
        }
        try:
            payload = torch_load(path)
            if not isinstance(payload, dict):
                raise TypeError(f"payload_type={type(payload)!r}")
            meta = payload.get("meta", payload.get("metadata", {})) or {}
            target_id = str(meta.get("target_id", path.parent.name))
            row["target_id"] = target_id
            for split in ("train", "valid", "test"):
                row[f"{split}_acc"] = split_metric(payload, split, "acc")
                row[f"{split}_loss"] = split_metric(payload, split, "loss")
            row["gap"] = row["train_acc"] - row["test_acc"]
            row["membership_convention"] = membership_convention(payload)
            membership = normalize_membership(payload)
            row["n_member"] = int((membership == 1).sum())
            row["n_nonmember"] = int((membership == 0).sum())
            row["fpr_resolution"] = (
                1.0 / row["n_nonmember"] if row["n_nonmember"] else np.nan
            )
            row.update(flatten_scalar_meta(meta, prefix="meta_"))
            resource = payload.get("resource_counts", {}) or {}
            if isinstance(resource, dict):
                for key, value in resource.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        row[f"resource_{key}"] = value

            match = PAIR_RE.match(target_id)
            if match:
                row["pair_id_inferred"] = match.group("pair_id")
                row["fm_kind_inferred"] = match.group("fm_kind").lower()
                row["model_seed_inferred"] = int(match.group("seed"))
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
            row.setdefault("target_id", path.parent.name)
        rows.append(row)

    raw = pd.DataFrame(rows)
    if args.targets and args.targets.exists():
        targets = pd.read_csv(args.targets)
        raw = raw.merge(targets, on="target_id", how="left", suffixes=("", "_target"))

    for field in (
        "experiment",
        "structural_cell_id",
        "pair_id",
        "role",
        "dataset",
        "architecture",
        "fm_kind",
        "reps",
        "depth",
        "n_wires",
        "model_seed",
        "data_seed",
        "seed",
        "source_run_id",
    ):
        coalesce_columns(raw, field)
    if "model_seed" not in raw.columns and "seed" in raw.columns:
        raw["model_seed"] = raw["seed"]
    if "seed" not in raw.columns and "model_seed" in raw.columns:
        raw["seed"] = raw["model_seed"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "retrained_target_metrics_raw.csv"
    atomic_write_csv(raw, raw_path)

    group_columns = [
        column
        for column in (
            "experiment",
            "dataset",
            "architecture",
            "structural_cell_id",
            "pair_id",
            "role",
            "fm_kind",
            "reps",
            "depth",
            "n_wires",
        )
        if column in raw.columns
    ]
    metric_columns = [
        column
        for column in (
            "train_acc",
            "valid_acc",
            "test_acc",
            "train_loss",
            "valid_loss",
            "test_loss",
            "gap",
        )
        if column in raw.columns
    ]
    valid = raw[raw["status"].eq("ok")].copy()
    if group_columns and metric_columns and not valid.empty:
        summary = (
            valid.groupby(group_columns, dropna=False)[metric_columns]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        summary.columns = [
            "_".join(str(value) for value in column if str(value))
            if isinstance(column, tuple)
            else str(column)
            for column in summary.columns
        ]
        summary["error_bar_type"] = (
            "mean ± sample SD across independent target-model seeds; fixed data seed"
        )
    else:
        summary = pd.DataFrame()
    summary_path = args.out_dir / "retrained_target_metrics_summary.csv"
    atomic_write_csv(summary, summary_path)

    write_analysis_metadata(
        args.out_dir / "analysis_metadata.json",
        script=Path(__file__).name,
        inputs=[str(args.attack_data_dir), str(args.targets or "")],
        outputs=[str(raw_path), str(summary_path)],
        ci_method="none; extraction plus mean and sample SD",
        bootstrap_unit="none",
        bootstrap_replicates=0,
        error_bar_type="sample SD across target-model seeds",
        notes=(
            "Validation metrics require the reviewer patch. Older payloads remain usable "
            "but their validation fields are NA."
        ),
    )
    print(f"[OK] payloads={len(raw)} valid={len(valid)} -> {raw_path.resolve()}")


if __name__ == "__main__":
    main()
