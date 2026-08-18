#!/usr/bin/env python3
"""Choose structural policies on development blocks and create fresh targets."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


FRESH_DATA_SEEDS = (40261, 40262, 40263, 40264, 40265)
FRESH_MODEL_SEEDS = (50261, 50262, 50263, 50264, 50265)
CONFIG_COLUMNS = ("structural_cell_id", "fm_kind", "reps", "depth", "fm_ent", "fm_op", "pad_mode")


def select_policies(
    targets: pd.DataFrame,
    metrics: pd.DataFrame,
    attacks: pd.DataFrame,
    *,
    accuracy_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    if accuracy_tolerance < 0:
        raise ValueError("accuracy_tolerance must be non-negative")
    loss = attacks[attacks["attack"].astype(str).str.lower().eq("loss")].copy()
    for name, frame in (("targets", targets), ("metrics", metrics), ("loss attacks", loss)):
        if frame["target_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate target IDs")
    joined = (
        targets.merge(metrics[["target_id", "valid_acc"]], on="target_id", how="inner")
        .merge(loss[["target_id", "auc"]], on="target_id", how="inner")
    )
    if joined.empty:
        raise ValueError("No common development targets across design, metrics, and loss MIA")
    expected_blocks = targets["block_id"].nunique()
    summary = (
        joined.groupby(list(CONFIG_COLUMNS), dropna=False)
        .agg(
            mean_valid_acc=("valid_acc", "mean"),
            sd_valid_acc=("valid_acc", "std"),
            mean_loss_auc=("auc", "mean"),
            sd_loss_auc=("auc", "std"),
            n_blocks=("block_id", "nunique"),
        )
        .reset_index()
    )
    complete = summary[summary["n_blocks"] == expected_blocks].copy()
    if complete.empty:
        raise ValueError("No structural configuration is complete across all development blocks")
    utility = complete.sort_values(
        ["mean_valid_acc", "mean_loss_auc", "structural_cell_id"],
        ascending=[False, True, True],
    ).iloc[0]
    floor = float(utility["mean_valid_acc"] - accuracy_tolerance)
    eligible = complete[complete["mean_valid_acc"] >= floor].copy()
    privacy = eligible.sort_values(
        ["mean_loss_auc", "mean_valid_acc", "structural_cell_id"],
        ascending=[True, False, True],
    ).iloc[0]

    def as_config(row: pd.Series) -> dict[str, object]:
        output = {}
        for column in CONFIG_COLUMNS:
            value = row[column]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                value = None
            elif hasattr(value, "item"):
                value = value.item()
            output[column] = value
        return output

    decisions = {
        "utility_only": as_config(utility),
        "privacy_aware": as_config(privacy),
        "utility_regularized": as_config(utility),
    }
    return summary, decisions


def build_fresh_targets(
    development_targets: pd.DataFrame,
    decisions: dict[str, dict[str, object]],
    *,
    blocks: int,
    regularized_weight_decay: float,
) -> pd.DataFrame:
    if blocks > len(FRESH_DATA_SEEDS):
        raise ValueError(f"At most {len(FRESH_DATA_SEEDS)} fresh blocks are predefined")
    template = development_targets.iloc[0].to_dict()
    for key in ("target_id", "role", "structural_cell_id", "fm_kind", "reps", "depth", "fm_ent", "fm_op", "pad_mode"):
        template.pop(key, None)
    rows = []
    for index in range(blocks):
        fresh_block = f"selector_b{index + 1:02d}"
        for policy, config in decisions.items():
            rows.append(
                {
                    **template,
                    **config,
                    "target_id": f"CREDIT_SELECTOR_{policy}_b{index + 1:02d}",
                    "experiment": "satml_selector_fresh",
                    "role": policy,
                    "selector_policy": policy,
                    "block_id": fresh_block,
                    "seed": FRESH_MODEL_SEEDS[index],
                    "model_seed": FRESH_MODEL_SEEDS[index],
                    "init_seed": FRESH_MODEL_SEEDS[index],
                    "data_seed": FRESH_DATA_SEEDS[index],
                    "split_seed": FRESH_DATA_SEEDS[index],
                    "weight_decay": regularized_weight_decay if policy == "utility_regularized" else 0.0,
                }
            )
    frame = pd.DataFrame(rows)
    if frame["target_id"].duplicated().any() or len(frame) != blocks * 3:
        raise AssertionError("Fresh selector target construction failed")
    if set(frame["data_seed"]) & set(development_targets["data_seed"]):
        raise AssertionError("Fresh selector data seeds overlap development seeds")
    if set(frame["model_seed"]) & set(development_targets["model_seed"]):
        raise AssertionError("Fresh selector model seeds overlap development seeds")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-targets", type=Path, required=True)
    parser.add_argument("--development-metrics", type=Path, required=True)
    parser.add_argument("--development-attacks", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("satml_targets/selector"))
    parser.add_argument("--accuracy-tolerance", type=float, default=0.02)
    parser.add_argument("--fresh-blocks", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--regularized-weight-decay", type=float, default=1e-3)
    args = parser.parse_args()
    summary, decisions = select_policies(
        pd.read_csv(args.development_targets),
        pd.read_csv(args.development_metrics),
        pd.read_csv(args.development_attacks),
        accuracy_tolerance=args.accuracy_tolerance,
    )
    development = pd.read_csv(args.development_targets)
    fresh = build_fresh_targets(
        development,
        decisions,
        blocks=args.fresh_blocks,
        regularized_weight_decay=args.regularized_weight_decay,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "development_policy_candidates.csv", index=False)
    fresh.to_csv(args.out_dir / "fresh_selector_targets.csv", index=False)
    decision_record = {
        "accuracy_tolerance": args.accuracy_tolerance,
        "selection_accuracy_split": "development validation",
        "selection_privacy_signal": "development loss-threshold MIA AUC",
        "fresh_blocks": args.fresh_blocks,
        "fresh_data_seeds": list(FRESH_DATA_SEEDS[: args.fresh_blocks]),
        "fresh_model_seeds": list(FRESH_MODEL_SEEDS[: args.fresh_blocks]),
        "regularized_weight_decay": args.regularized_weight_decay,
        "policies": decisions,
    }
    (args.out_dir / "selection_decision.json").write_text(
        json.dumps(decision_record, indent=2, sort_keys=True, default=str, allow_nan=False), encoding="utf-8"
    )
    print(f"[OK] fresh targets={len(fresh)} -> {args.out_dir / 'fresh_selector_targets.csv'}")


if __name__ == "__main__":
    main()
