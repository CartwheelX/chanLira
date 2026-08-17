#!/usr/bin/env python3
"""Analyze controlled QNN/HQNN/QCNN/MLP-QNN wrapper experiments.

The comparison is paired within structural role and target-model seed. Results
must be described as a complete-wrapper comparison because preprocessing and
classical heads differ across architecture families.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reviewer_common import atomic_write_csv, stable_seed, write_analysis_metadata


def coalesce(frame: pd.DataFrame, name: str) -> None:
    candidates = [name, f"{name}_metric", f"{name}_mia", f"{name}_resource"]
    present = [column for column in candidates if column in frame.columns]
    if not present:
        return
    value = frame[present[0]]
    for column in present[1:]:
        value = value.where(value.notna(), frame[column])
    frame[name] = value


def paired_architecture_effect(
    frame: pd.DataFrame,
    metric: str,
    architecture: str,
    baseline: str,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    subset = frame[frame["architecture"].isin([baseline, architecture])].copy()
    pivot = subset.pivot_table(
        index=["role", "model_seed"],
        columns="architecture",
        values=metric,
        aggfunc="mean",
    )
    if baseline not in pivot.columns or architecture not in pivot.columns:
        return {
            "metric": metric,
            "architecture": architecture,
            "baseline": baseline,
            "contrast": f"{architecture} minus {baseline}",
            "mean_difference": np.nan,
            "sd_paired_differences": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "n_structural_roles": 0,
            "n_paired_role_seed_units": 0,
            "valid_bootstrap_replicates": 0,
        }
    pivot = pivot.dropna(subset=[baseline, architecture]).reset_index()
    pivot["difference"] = pivot[architecture] - pivot[baseline]
    observed = float(pivot["difference"].mean()) if len(pivot) else np.nan
    observed_sd = (
        float(pivot["difference"].std(ddof=1)) if len(pivot) > 1 else np.nan
    )
    roles = pivot["role"].dropna().unique()
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(bootstrap):
        if not len(roles):
            break
        selected_roles = rng.choice(roles, len(roles), replace=True)
        role_effects: list[float] = []
        for role in selected_roles:
            effects = pivot.loc[pivot["role"] == role, "difference"].dropna().to_numpy(float)
            if not len(effects):
                continue
            role_effects.append(float(rng.choice(effects, len(effects), replace=True).mean()))
        if role_effects:
            values.append(float(np.mean(role_effects)))
    low = float(np.quantile(values, 0.025)) if values else np.nan
    high = float(np.quantile(values, 0.975)) if values else np.nan
    return {
        "metric": metric,
        "architecture": architecture,
        "baseline": baseline,
        "contrast": f"{architecture} minus {baseline}",
        "mean_difference": observed,
        "sd_paired_differences": observed_sd,
        "ci95_low": low,
        "ci95_high": high,
        "n_structural_roles": int(len(roles)),
        "n_paired_role_seed_units": int(len(pivot)),
        "valid_bootstrap_replicates": int(len(values)),
        "ci_method": (
            "paired hierarchical percentile bootstrap over structural roles "
            "with target-model seeds nested"
        ),
        "bootstrap_unit": "structural role; target-model seed nested",
        "bootstrap_replicates": bootstrap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--mia", type=Path, required=True)
    parser.add_argument("--resources", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reviewer_results/architecture_control"),
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--baseline", default="qnn")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    target_metrics = pd.read_csv(args.metrics)
    mia = pd.read_csv(args.mia)
    if "attack" in mia.columns:
        mia = mia[mia["attack"].astype(str).str.lower().eq("loss")].copy()
    mia_columns = [
        column
        for column in (
            "target_id",
            "auc",
            "balanced_accuracy_crossfit",
            "membership_advantage_crossfit",
            "crossfit_tpr",
            "crossfit_fpr",
        )
        if column in mia.columns
    ]
    combined = target_metrics.merge(
        mia[mia_columns], on="target_id", how="left", suffixes=("_metric", "_mia")
    )
    if args.resources is not None:
        resources = pd.read_csv(args.resources)
        resource_columns = [
            column
            for column in (
                "target_id",
                "trainable_parameters_total",
                "trainable_parameters_quantum",
                "trainable_parameters_classical",
                "quantum_gate_count_total",
                "quantum_one_qubit_gates",
                "quantum_two_qubit_gates",
                "gate_count_scope",
            )
            if column in resources.columns
        ]
        combined = combined.merge(
            resources[resource_columns],
            on="target_id",
            how="left",
            suffixes=("", "_resource"),
        )

    for field in ("architecture", "role", "model_seed", "dataset", "fm_kind", "reps", "depth"):
        coalesce(combined, field)
    if "model_seed" not in combined.columns and "seed" in combined.columns:
        combined["model_seed"] = combined["seed"]
    if "gap_acc" not in combined.columns and "gap" in combined.columns:
        combined["gap_acc"] = combined["gap"]
    combined["architecture"] = combined["architecture"].astype(str).str.lower()

    raw_path = args.out_dir / "architecture_control_raw.csv"
    atomic_write_csv(combined, raw_path)

    metric_columns = [
        column
        for column in (
            "train_acc",
            "valid_acc",
            "test_acc",
            "train_loss",
            "valid_loss",
            "test_loss",
            "gap_acc",
            "auc",
            "balanced_accuracy_crossfit",
            "membership_advantage_crossfit",
            "trainable_parameters_total",
            "trainable_parameters_quantum",
            "trainable_parameters_classical",
            "quantum_gate_count_total",
        )
        if column in combined.columns
    ]
    summary = (
        combined.groupby(["role", "architecture"], dropna=False)[metric_columns]
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
    summary_path = args.out_dir / "architecture_control_summary.csv"
    atomic_write_csv(summary, summary_path)

    overall_summary = (
        combined.groupby(["architecture"], dropna=False)[metric_columns]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    overall_summary.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in overall_summary.columns
    ]
    overall_summary["aggregation_note"] = (
        "descriptive across structural roles and target seeds; roles are not interchangeable datasets"
    )
    overall_path = args.out_dir / "architecture_control_overall_descriptive.csv"
    atomic_write_csv(overall_summary, overall_path)

    architectures = sorted(
        architecture
        for architecture in combined["architecture"].dropna().unique()
        if architecture != args.baseline.lower()
    )
    effect_metrics = [
        column
        for column in (
            "gap_acc",
            "test_acc",
            "test_loss",
            "auc",
            "balanced_accuracy_crossfit",
            "membership_advantage_crossfit",
        )
        if column in combined.columns
    ]
    effects: list[dict[str, Any]] = []
    for architecture in architectures:
        for metric in effect_metrics:
            effects.append(
                paired_architecture_effect(
                    combined,
                    metric,
                    architecture,
                    args.baseline.lower(),
                    args.bootstrap,
                    stable_seed(
                        args.bootstrap_seed,
                        "architecture",
                        architecture,
                        metric,
                    ),
                )
            )
    effects_path = args.out_dir / "architecture_control_effects.csv"
    atomic_write_csv(pd.DataFrame(effects), effects_path)

    completeness = (
        combined.groupby(["role", "architecture"], dropna=False)["model_seed"]
        .agg(n_rows="size", n_model_seeds="nunique")
        .reset_index()
    )
    completeness["expected_model_seeds"] = 3
    completeness["complete"] = completeness["n_model_seeds"].eq(3)
    completeness_path = args.out_dir / "architecture_control_completeness.csv"
    atomic_write_csv(completeness, completeness_path)

    write_analysis_metadata(
        args.out_dir / "analysis_metadata.json",
        script=Path(__file__).name,
        inputs=[str(args.metrics), str(args.mia), str(args.resources or "")],
        outputs=[
            str(raw_path),
            str(summary_path),
            str(overall_path),
            str(effects_path),
            str(completeness_path),
        ],
        ci_method=(
            "paired hierarchical percentile bootstrap over structural roles with "
            "target-model seeds nested"
        ),
        bootstrap_unit="structural role; target-model seed nested",
        bootstrap_replicates=args.bootstrap,
        error_bar_type="mean ± sample SD across target-model seeds",
        notes=(
            "This is a complete-wrapper comparison. QNN, HQNN, QCNN, and MLP-QNN "
            "differ in preprocessing and/or heads; effects must not be presented as a "
            "pure causal quantum-architecture comparison. Aggregate CIs are exploratory "
            "because only three structural roles are included."
        ),
    )
    print(f"[OK] Architecture summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
