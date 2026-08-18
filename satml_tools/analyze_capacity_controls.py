#!/usr/bin/env python3
"""Verify capacity/resource controls for the Credit structural factorial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PARAMETERS = "resource_trainable_parameters_total"
GATES = "resource_quantum_gate_count_total"


def analyze_capacity(targets: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required_targets = {"target_id", "block_id", "fm_kind", "reps", "depth"}
    required_metrics = {"target_id", PARAMETERS, GATES}
    if not required_targets.issubset(targets.columns):
        raise ValueError(f"targets lack {sorted(required_targets - set(targets.columns))}")
    if not required_metrics.issubset(metrics.columns):
        raise ValueError(f"metrics lack {sorted(required_metrics - set(metrics.columns))}")
    merged = targets[list(required_targets)].merge(metrics[list(required_metrics)], on="target_id", how="inner")
    if len(merged) != len(targets) or merged.target_id.duplicated().any():
        raise ValueError(f"Resource merge incomplete or duplicated: targets={len(targets)} merged={len(merged)}")
    summary = (
        merged.groupby(["fm_kind", "reps", "depth"], as_index=False)
        .agg(
            n_targets=("target_id", "count"),
            trainable_parameters_mean=(PARAMETERS, "mean"),
            trainable_parameters_unique=(PARAMETERS, "nunique"),
            quantum_gate_count_mean=(GATES, "mean"),
            quantum_gate_count_unique=(GATES, "nunique"),
        )
    )
    contrasts = []
    for (block, fm, depth), group in merged.groupby(["block_id", "fm_kind", "depth"]):
        pivot = group.set_index("reps")
        if 1 not in pivot.index or 5 not in pivot.index:
            continue
        contrasts.append(
            {
                "block_id": block, "fm_kind": fm, "depth": depth, "contrast": "reps 5 - reps 1",
                "trainable_parameter_difference": float(pivot.at[5, PARAMETERS] - pivot.at[1, PARAMETERS]),
                "quantum_gate_count_difference": float(pivot.at[5, GATES] - pivot.at[1, GATES]),
            }
        )
    repetition = pd.DataFrame(contrasts)
    same_capacity = bool(len(repetition) and (repetition.trainable_parameter_difference == 0).all())
    more_encoder_work = bool(len(repetition) and (repetition.quantum_gate_count_difference > 0).all())
    checks = {
        "passed": same_capacity and more_encoder_work,
        "checks": [
            {"check": "repetition_keeps_trainable_parameter_count_fixed", "passed": same_capacity,
             "detail": "all paired reps 5 - reps 1 trainable-parameter differences must equal zero"},
            {"check": "repetition_increases_main_stack_gate_count", "passed": more_encoder_work,
             "detail": "all paired reps 5 - reps 1 gate-count differences must be positive"},
        ],
        "interpretation": (
            "Repetition effects compare equal trainable capacity while deliberately changing fixed-encoder operations. "
            "Depth is a separate factor and is not claimed to be capacity matched."
        ),
    }
    return summary, repetition, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("satml_results/credit_factorial/capacity_controls"))
    args = parser.parse_args()
    summary, repetition, checks = analyze_capacity(pd.read_csv(args.targets), pd.read_csv(args.metrics))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "structural_resource_summary.csv", index=False)
    repetition.to_csv(args.out_dir / "repetition_resource_contrasts.csv", index=False)
    (args.out_dir / "capacity_validation.json").write_text(json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8")
    for check in checks["checks"]:
        print(f"[{'PASS' if check['passed'] else 'FAIL'}] {check['check']}: {check['detail']}")
    if not checks["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
