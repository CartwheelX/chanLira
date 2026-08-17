#!/usr/bin/env python3
"""Correctness-only label-output MIA baseline.

This attack uses only the predicted class label and the known candidate label.
It is therefore a feasible label-output baseline, but it is not a boundary-
distance/query-augmentation label-only attack. The distinction is recorded in
all output metadata.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from reviewer_common import (
    CI_RECORD,
    atomic_write_csv,
    cross_fitted_threshold_metrics,
    find_attack_files,
    flatten_scalar_meta,
    normalize_membership,
    scalar_attack_scores,
    stable_seed,
    stratified_bootstrap_metric,
    torch_load,
    write_analysis_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-data-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path, default=None)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("reviewer_results/label_only")
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(args.targets) if args.targets and args.targets.exists() else None
    rows = []
    for path in find_attack_files(args.attack_data_dir):
        payload = torch_load(path)
        membership = normalize_membership(payload)
        correctness = scalar_attack_scores(payload).get("correctness")
        if correctness is None:
            continue
        meta = payload.get("meta", {}) or {}
        target_id = str(meta.get("target_id", path.parent.name))
        auc = float(roc_auc_score(membership, correctness))
        low, high, valid = stratified_bootstrap_metric(
            membership,
            correctness,
            roc_auc_score,
            args.bootstrap,
            stable_seed(args.seed, target_id, "label_only_correctness"),
        )
        record = {
            "target_id": target_id,
            "attack": "label_only_correctness",
            "auc": auc,
            "auc_record_ci95_low": low,
            "auc_record_ci95_high": high,
            "valid_record_bootstrap_replicates": valid,
            "n_member": int((membership == 1).sum()),
            "n_nonmember": int((membership == 0).sum()),
            "source_file": str(path),
            "ci_method": CI_RECORD,
            "scope_note": (
                "correctness-only label-output baseline; not a boundary-distance or "
                "query-augmentation label-only attack"
            ),
        }
        record.update(flatten_scalar_meta(meta, prefix=""))
        if targets is not None and "target_id" in targets.columns:
            matched = targets[targets["target_id"].astype(str) == target_id]
            if len(matched):
                for key, value in matched.iloc[0].to_dict().items():
                    if key not in record or pd.isna(record.get(key)):
                        record[key] = value
        record.update(cross_fitted_threshold_metrics(membership, correctness, 5, args.seed))
        rows.append(record)

    raw = pd.DataFrame(rows)
    raw_path = args.out_dir / "label_only_correctness_raw.csv"
    atomic_write_csv(raw, raw_path)
    group_columns = [
        column
        for column in ("experiment", "dataset", "architecture", "role", "fm_kind", "reps", "depth")
        if column in raw.columns
    ]
    metrics = [
        column
        for column in ("auc", "balanced_accuracy_crossfit", "membership_advantage_crossfit")
        if column in raw.columns
    ]
    if group_columns and metrics and not raw.empty:
        summary = raw.groupby(group_columns, dropna=False)[metrics].agg(["count", "mean", "std"]).reset_index()
        summary.columns = [
            "_".join(str(value) for value in column if str(value))
            if isinstance(column, tuple)
            else str(column)
            for column in summary.columns
        ]
        summary["error_bar_type"] = "mean ± sample SD across target-model seeds"
    else:
        summary = pd.DataFrame()
    summary_path = args.out_dir / "label_only_correctness_summary.csv"
    atomic_write_csv(summary, summary_path)
    write_analysis_metadata(
        args.out_dir / "analysis_metadata.json",
        script=Path(__file__).name,
        inputs=[str(args.attack_data_dir), str(args.targets or "")],
        outputs=[str(raw_path), str(summary_path)],
        ci_method=CI_RECORD,
        bootstrap_unit="member/non-member records",
        bootstrap_replicates=args.bootstrap,
        error_bar_type="mean ± sample SD across target-model seeds",
        notes=(
            "Correctness-only label-output baseline. It does not claim the stronger "
            "decision-boundary label-only threat model."
        ),
    )
    print(f"[OK] rows={len(raw)} -> {raw_path.resolve()}")


if __name__ == "__main__":
    main()
