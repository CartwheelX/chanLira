#!/usr/bin/env python3
"""Generate reviewer-facing tables and figures from completed QuRiFT runs.

The script deliberately keeps the different replication units separate:

* target-model seeds for the factorial and architecture experiments;
* data seeds for encoder geometry;
* simulator seeds for finite-shot/noisy evaluation; and
* attacker seeds for the learned prediction-vector MIA.

Outputs are written under ``reviewer_results/reviewer_artifacts`` by default.
Every compact table is saved as CSV and LaTeX, and every figure as PNG and PDF.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


FM_LABELS = {"eff_su2": "EffSU2", "z": "Z", "zz": "ZZ"}
FM_COLORS = {"eff_su2": "#0072B2", "z": "#D55E00", "zz": "#009E73"}
ARCH_ORDER = ["qnn", "hqnn", "qcnn", "mlp_qnn"]
ARCH_LABELS = {
    "qnn": "QNN",
    "hqnn": "HQNN",
    "qcnn": "QCNN",
    "mlp_qnn": "MLP-QNN",
}


def require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    if path.stat().st_size <= 1:
        raise ValueError(f"Required input is empty: {path}")
    return pd.read_csv(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def clean_for_export(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_float_dtype(output[column]):
            output[column] = output[column].replace([np.inf, -np.inf], np.nan)
    return output


def save_table(
    frame: pd.DataFrame,
    tables_dir: Path,
    name: str,
    caption: str,
    label: str,
) -> Dict[str, str]:
    frame = clean_for_export(frame)
    csv_path = tables_dir / f"{name}.csv"
    tex_path = tables_dir / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    latex = frame.to_latex(
        index=False,
        escape=True,
        na_rep="--",
        float_format=lambda value: f"{value:.3f}",
        caption=caption,
        label=label,
        position="t",
    )
    write_text(tex_path, latex)
    return {"csv": str(csv_path), "tex": str(tex_path)}


def save_figure(fig: plt.Figure, figures_dir: Path, name: str) -> Dict[str, str]:
    png_path = figures_dir / f"{name}.png"
    pdf_path = figures_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def mean_sd_text(mean: Any, sd: Any, digits: int = 3) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(sd):
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f} ± {float(sd):.{digits}f}"


def ci_text(low: Any, high: Any, digits: int = 3) -> str:
    if pd.isna(low) or pd.isna(high):
        return "--"
    return f"[{float(low):.{digits}f}, {float(high):.{digits}f}]"


def stable_rng(seed: int, *parts: Any) -> np.random.Generator:
    import hashlib

    text = "|".join([str(seed), *[str(part) for part in parts]])
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    return np.random.default_rng(value % (2**32 - 1))


def load_learned_summaries(results_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(results_root.glob("learned_mia_seed*/attack_summary.csv")):
        frame = require_csv(path)
        directory_seed = int(path.parent.name.rsplit("seed", 1)[-1])
        if "attacker_seed" not in frame.columns:
            frame["attacker_seed"] = directory_seed
        frame["summary_source"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No learned-MIA attack_summary.csv files found under {results_root}"
        )
    return pd.concat(frames, ignore_index=True)


def prepare_factorial(
    metrics: pd.DataFrame,
    threshold: pd.DataFrame,
    resources: pd.DataFrame,
) -> pd.DataFrame:
    base = metrics[metrics["status"].astype(str).str.lower().eq("ok")].copy()
    attacks = {}
    for attack in ("loss", "confidence", "correctness", "entropy", "margin"):
        subset = threshold[threshold["attack"].astype(str).str.lower().eq(attack)][
            [
                "target_id",
                "auc",
                "balanced_accuracy_crossfit",
                "membership_advantage_crossfit",
                "tpr_at_requested_fpr_0p05",
                "tpr_at_requested_fpr_0p1",
            ]
        ].copy()
        subset = subset.rename(
            columns={
                "auc": f"{attack}_auc",
                "balanced_accuracy_crossfit": f"{attack}_balanced_accuracy",
                "membership_advantage_crossfit": f"{attack}_membership_advantage",
                "tpr_at_requested_fpr_0p05": f"{attack}_tpr_at_5pct_fpr",
                "tpr_at_requested_fpr_0p1": f"{attack}_tpr_at_10pct_fpr",
            }
        )
        attacks[attack] = subset
    output = base
    for subset in attacks.values():
        output = output.merge(subset, on="target_id", how="left")
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
            "exact_resource_counts_available",
        )
        if column in resources.columns
    ]
    output = output.merge(resources[resource_columns], on="target_id", how="left")
    for column in ("fm_kind", "architecture"):
        output[column] = output[column].astype(str).str.lower()
    return output


def factorial_cell_table(factorial: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "train_acc",
        "test_acc",
        "gap",
        "loss_auc",
        "confidence_auc",
        "correctness_auc",
        "loss_balanced_accuracy",
        "loss_membership_advantage",
        "loss_tpr_at_5pct_fpr",
        "loss_tpr_at_10pct_fpr",
    ]
    grouped = (
        factorial.groupby(["fm_kind", "reps", "depth"], dropna=False)[metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple)
        else str(column)
        for column in grouped.columns
    ]
    rows: List[Dict[str, Any]] = []
    for _, row in grouped.iterrows():
        record: Dict[str, Any] = {
            "Feature map": FM_LABELS.get(str(row["fm_kind"]), row["fm_kind"]),
            "Repetitions": int(row["reps"]),
            "Depth": int(row["depth"]),
            "Target seeds": int(row["loss_auc_count"]),
        }
        labels = {
            "train_acc": "Train accuracy",
            "test_acc": "Test accuracy",
            "gap": "Accuracy gap",
            "loss_auc": "Loss AUC",
            "confidence_auc": "Confidence AUC",
            "correctness_auc": "Correctness AUC",
            "loss_balanced_accuracy": "Loss balanced accuracy",
            "loss_membership_advantage": "Loss membership advantage",
            "loss_tpr_at_5pct_fpr": "Loss TPR@5% FPR",
            "loss_tpr_at_10pct_fpr": "Loss TPR@10% FPR",
        }
        for metric, label in labels.items():
            record[label] = mean_sd_text(row[f"{metric}_mean"], row[f"{metric}_std"])
        rows.append(record)
    return pd.DataFrame(rows)


def paired_hierarchical_effect(
    frame: pd.DataFrame,
    metric: str,
    factor: str,
    low: Any,
    high: Any,
    block_columns: Sequence[str],
    bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    index_columns = list(block_columns) + ["model_seed"]
    pivot = frame.pivot_table(
        index=index_columns,
        columns=factor,
        values=metric,
        aggfunc="mean",
    )
    if low not in pivot.columns or high not in pivot.columns:
        return {
            "mean_difference": np.nan,
            "sd_paired_units": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "n_blocks": 0,
            "n_paired_seed_units": 0,
            "valid_bootstrap_replicates": 0,
        }
    pivot = pivot.dropna(subset=[low, high]).reset_index()
    pivot["effect"] = pivot[high] - pivot[low]
    grouped_effects = [
        group["effect"].to_numpy(float)
        for _, group in pivot.groupby(list(block_columns), dropna=False)
    ]
    grouped_effects = [values for values in grouped_effects if len(values)]
    if not grouped_effects:
        return {
            "mean_difference": np.nan,
            "sd_paired_units": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "n_blocks": 0,
            "n_paired_seed_units": 0,
            "valid_bootstrap_replicates": 0,
        }
    block_means = np.asarray([values.mean() for values in grouped_effects])
    observed = float(block_means.mean())
    all_effects = np.concatenate(grouped_effects)
    rng = stable_rng(seed, factor, low, high, metric)
    values = np.empty(bootstrap, dtype=float)
    n_blocks = len(grouped_effects)
    for replicate in range(bootstrap):
        selected = rng.integers(0, n_blocks, size=n_blocks)
        sampled_blocks = []
        for index in selected:
            effects = grouped_effects[int(index)]
            sampled_blocks.append(
                float(rng.choice(effects, size=len(effects), replace=True).mean())
            )
        values[replicate] = float(np.mean(sampled_blocks))
    return {
        "mean_difference": observed,
        "sd_paired_units": (
            float(all_effects.std(ddof=1)) if len(all_effects) > 1 else np.nan
        ),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "n_blocks": n_blocks,
        "n_paired_seed_units": int(len(all_effects)),
        "valid_bootstrap_replicates": bootstrap,
    }


def factorial_effect_table(
    factorial: pd.DataFrame, bootstrap: int, seed: int
) -> pd.DataFrame:
    specifications = [
        ("Repetitions", "reps", 1, 5, ["fm_kind", "depth"], "5 − 1"),
        ("Depth", "depth", 2, 6, ["fm_kind", "reps"], "6 − 2"),
        (
            "Feature map",
            "fm_kind",
            "eff_su2",
            "z",
            ["reps", "depth"],
            "Z − EffSU2",
        ),
        (
            "Feature map",
            "fm_kind",
            "eff_su2",
            "zz",
            ["reps", "depth"],
            "ZZ − EffSU2",
        ),
        ("Feature map", "fm_kind", "z", "zz", ["reps", "depth"], "ZZ − Z"),
    ]
    metrics = {
        "test_acc": "Test accuracy",
        "gap": "Accuracy gap",
        "loss_auc": "Loss AUC",
        "confidence_auc": "Confidence AUC",
        "correctness_auc": "Correctness AUC",
        "loss_membership_advantage": "Loss membership advantage",
    }
    rows: List[Dict[str, Any]] = []
    for factor_label, factor, low, high, blocks, contrast in specifications:
        for metric, metric_label in metrics.items():
            result = paired_hierarchical_effect(
                factorial,
                metric,
                factor,
                low,
                high,
                blocks,
                bootstrap,
                seed,
            )
            rows.append(
                {
                    "Factor": factor_label,
                    "Contrast": contrast,
                    "Metric": metric_label,
                    "Mean difference": result["mean_difference"],
                    "Paired-unit SD": result["sd_paired_units"],
                    "95% CI": ci_text(result["ci95_low"], result["ci95_high"]),
                    "Structural blocks": result["n_blocks"],
                    "Paired seed units": result["n_paired_seed_units"],
                    "Valid bootstrap": result["valid_bootstrap_replicates"],
                }
            )
    return pd.DataFrame(rows)


def attack_suite_table(
    threshold: pd.DataFrame, learned: pd.DataFrame
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for attack, group in threshold.groupby("attack", sort=True):
        rows.append(
            {
                "Attack": str(attack),
                "Access": "score/probability threshold",
                "Target units": int(group["target_id"].nunique()),
                "Attacker seeds": 0,
                "AUC": mean_sd_text(group["auc"].mean(), group["auc"].std(ddof=1)),
                "TPR@5% FPR": mean_sd_text(
                    group["tpr_at_requested_fpr_0p05"].mean(),
                    group["tpr_at_requested_fpr_0p05"].std(ddof=1),
                ),
                "TPR@10% FPR": mean_sd_text(
                    group["tpr_at_requested_fpr_0p1"].mean(),
                    group["tpr_at_requested_fpr_0p1"].std(ddof=1),
                ),
            }
        )
    learned_factorial = learned[
        learned["experiment"].astype(str).str.lower().eq("multiseed_factorial")
    ].copy()
    per_target = (
        learned_factorial.groupby("target_id", dropna=False)
        .agg(
            attack_auc=("attack_auc", "mean"),
            tpr5=("tpr@fpr=0.05", "mean"),
            tpr10=("tpr@fpr=0.1", "mean"),
            attacker_seeds=("attacker_seed", "nunique"),
        )
        .reset_index()
    )
    rows.append(
        {
            "Attack": "learned prediction-vector",
            "Access": "full prediction vector + statistics",
            "Target units": int(per_target["target_id"].nunique()),
            "Attacker seeds": int(per_target["attacker_seeds"].max()),
            "AUC": mean_sd_text(
                per_target["attack_auc"].mean(), per_target["attack_auc"].std(ddof=1)
            ),
            "TPR@5% FPR": mean_sd_text(
                per_target["tpr5"].mean(), per_target["tpr5"].std(ddof=1)
            ),
            "TPR@10% FPR": mean_sd_text(
                per_target["tpr10"].mean(), per_target["tpr10"].std(ddof=1)
            ),
        }
    )
    return pd.DataFrame(rows)


def geometry_tables(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    uniqueness = (
        raw.groupby(["dataset", "fm_kind", "reps"], dropna=False)
        .agg(
            rows=("target_id", "size"),
            unique_data_seeds=("data_seed", "nunique"),
            unique_state_realizations=("state_signature", "nunique"),
        )
        .reset_index()
    )
    merged = summary.merge(
        uniqueness, on=["dataset", "fm_kind", "reps"], how="left"
    )
    rows: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "Dataset": row["dataset"],
                "Feature map": FM_LABELS.get(str(row["fm_kind"]), row["fm_kind"]),
                "Repetitions": int(row["reps"]),
                "Nominal data seeds": int(row["unique_data_seeds"]),
                "Unique state realizations": int(row["unique_state_realizations"]),
                "Class-similarity gap": mean_sd_text(
                    row["class_similarity_gap_mean"],
                    row["class_similarity_gap_std"],
                ),
                "Kernel alignment": mean_sd_text(
                    row["kernel_label_alignment_mean"],
                    row["kernel_label_alignment_std"],
                ),
                "Effective rank": mean_sd_text(
                    row["effective_rank_mean"], row["effective_rank_std"], digits=2
                ),
                "Train/test MMD²": mean_sd_text(
                    row["mmd2_train_test_mean"], row["mmd2_train_test_std"]
                ),
                "Encoder operations": mean_sd_text(
                    row["encoder_operation_count_mean"],
                    row["encoder_operation_count_std"],
                    digits=1,
                ),
            }
        )
    effect_rows: List[Dict[str, Any]] = []
    effect_metrics = [
        "class_similarity_gap",
        "kernel_label_alignment",
        "effective_rank",
        "mmd2_train_test",
        "encoder_operation_count",
    ]
    for metric in effect_metrics:
        pivot = raw.pivot_table(
            index=["dataset", "fm_kind", "data_seed"],
            columns="reps",
            values=metric,
            aggfunc="mean",
        )
        pivot = pivot.dropna(subset=[1, 5]).reset_index()
        pivot["effect"] = pivot[5] - pivot[1]
        # MNIST uses deterministic front-of-dataset selection, so its nominal
        # seeds can yield the same paired effect. Collapse exact duplicates
        # before any uncertainty calculation.
        pivot = pivot.drop_duplicates(
            subset=["dataset", "fm_kind", "effect"]
        )
        grouped_effects = [
            group["effect"].to_numpy(float)
            for _, group in pivot.groupby(["dataset", "fm_kind"], sort=False)
        ]
        block_means = np.asarray([values.mean() for values in grouped_effects])
        all_effects = np.concatenate(grouped_effects)
        rng = stable_rng(seed, "geometry_unique_effect", metric)
        boot_values = np.empty(bootstrap, dtype=float)
        for replicate in range(bootstrap):
            selected = rng.integers(
                0, len(grouped_effects), size=len(grouped_effects)
            )
            sampled_blocks = []
            for index in selected:
                values = grouped_effects[int(index)]
                sampled_blocks.append(
                    float(
                        rng.choice(values, size=len(values), replace=True).mean()
                    )
                )
            boot_values[replicate] = float(np.mean(sampled_blocks))
        effect_rows.append(
            {
                "Metric": metric,
                "Contrast": "reps 5 minus reps 1",
                "Mean difference": float(block_means.mean()),
                "Unique paired-effect SD": (
                    float(all_effects.std(ddof=1))
                    if len(all_effects) > 1
                    else np.nan
                ),
                "95% CI": ci_text(
                    np.quantile(boot_values, 0.025),
                    np.quantile(boot_values, 0.975),
                ),
                "Dataset/encoder blocks": len(grouped_effects),
                "Unique paired seed effects": len(all_effects),
                "Valid bootstrap": bootstrap,
            }
        )
    compact_effects = pd.DataFrame(effect_rows)
    audit = uniqueness.copy()
    audit["independent_seed_variation_observed"] = (
        audit["unique_state_realizations"] > 1
    )
    audit["reporting_note"] = np.where(
        audit["independent_seed_variation_observed"],
        "Data seeds produced distinct encoded-state realizations.",
        "Nominal seeds duplicated the encoded states; do not interpret zero SD as seed robustness.",
    )
    return pd.DataFrame(rows), compact_effects, audit


def noisy_tables(
    summary: pd.DataFrame, changes: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = summary[summary["attack"].astype(str).str.lower().eq("loss")].copy()
    rows: List[Dict[str, Any]] = []
    for _, row in summary.iterrows():
        mode = str(row["mode"])
        rows.append(
            {
                "Risk configuration": row["risk_role"],
                "Structural cell": row["structural_cell_id"],
                "Mode": mode,
                "Shots": "exact" if mode == "exact" else int(row["shots"]),
                "Target seeds": int(row["n_replicates"]),
                "Test accuracy": mean_sd_text(
                    row["test_accuracy_mean"], row["test_accuracy_sd"]
                ),
                "Accuracy gap": mean_sd_text(
                    row["accuracy_gap_mean"], row["accuracy_gap_sd"]
                ),
                "Loss AUC": mean_sd_text(row["roc_auc_mean"], row["roc_auc_sd"]),
                "TPR@5% FPR": mean_sd_text(
                    row["tpr_at_5pct_fpr_mean"], row["tpr_at_5pct_fpr_sd"]
                ),
                "TPR@10% FPR": mean_sd_text(
                    row["tpr_at_10pct_fpr_mean"], row["tpr_at_10pct_fpr_sd"]
                ),
            }
        )
    changes = changes[changes["attack"].astype(str).str.lower().eq("loss")].copy()
    delta_rows: List[Dict[str, Any]] = []
    for _, row in changes.iterrows():
        delta_rows.append(
            {
                "Target ID": row["target_id"],
                "Target seed": int(row["model_seed"]),
                "Risk configuration": row["risk_role"],
                "Structural cell": row["structural_cell_id"],
                "Mode": row["mode"],
                "Shots": int(row["shots"]),
                "Simulator seeds": int(row["n_replicates"]),
                "Δ test accuracy vs exact": row[
                    "delta_test_accuracy_from_exact_mean"
                ],
                "Δ test accuracy 95% CI": ci_text(
                    row["delta_test_accuracy_from_exact_ci95_low"],
                    row["delta_test_accuracy_from_exact_ci95_high"],
                ),
                "Δ loss AUC vs exact": row["delta_roc_auc_from_exact_mean"],
                "Δ loss AUC 95% CI": ci_text(
                    row["delta_roc_auc_from_exact_ci95_low"],
                    row["delta_roc_auc_from_exact_ci95_high"],
                ),
                "Valid bootstrap": int(
                    row["delta_roc_auc_from_exact_valid_bootstrap"]
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(delta_rows)


def architecture_tables(
    raw: pd.DataFrame, effects: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = raw.copy()
    raw["architecture"] = raw["architecture"].astype(str).str.lower()
    metrics = [
        "train_acc",
        "test_acc",
        "gap_acc",
        "auc",
        "balanced_accuracy_crossfit",
        "membership_advantage_crossfit",
        "trainable_parameters_total",
        "trainable_parameters_quantum",
        "trainable_parameters_classical",
        "quantum_gate_count_total",
    ]
    grouped = (
        raw.groupby(["role", "architecture"], dropna=False)[metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple)
        else str(column)
        for column in grouped.columns
    ]
    rows: List[Dict[str, Any]] = []
    for _, row in grouped.iterrows():
        rows.append(
            {
                "Structural role": row["role"],
                "Wrapper": ARCH_LABELS.get(
                    str(row["architecture"]), row["architecture"]
                ),
                "Target seeds": int(row["auc_count"]),
                "Test accuracy": mean_sd_text(
                    row["test_acc_mean"], row["test_acc_std"]
                ),
                "Accuracy gap": mean_sd_text(
                    row["gap_acc_mean"], row["gap_acc_std"]
                ),
                "Loss AUC": mean_sd_text(row["auc_mean"], row["auc_std"]),
                "Membership advantage": mean_sd_text(
                    row["membership_advantage_crossfit_mean"],
                    row["membership_advantage_crossfit_std"],
                ),
                "Total trainable parameters": int(
                    round(row["trainable_parameters_total_mean"])
                ),
                "Quantum parameters": int(
                    round(row["trainable_parameters_quantum_mean"])
                ),
                "Classical parameters": int(
                    round(row["trainable_parameters_classical_mean"])
                ),
                "Quantum gate count": int(
                    round(row["quantum_gate_count_total_mean"])
                ),
            }
        )
    compact_effects = effects.copy()
    compact_effects["95% CI"] = [
        ci_text(low, high)
        for low, high in zip(
            compact_effects["ci95_low"], compact_effects["ci95_high"]
        )
    ]
    compact_effects = compact_effects.rename(
        columns={
            "metric": "Metric",
            "architecture": "Wrapper",
            "baseline": "Baseline",
            "contrast": "Contrast",
            "mean_difference": "Mean difference",
            "sd_paired_differences": "Paired SD",
            "n_structural_roles": "Structural roles",
            "n_paired_role_seed_units": "Paired role/seed units",
            "valid_bootstrap_replicates": "Valid bootstrap",
        }
    )
    compact_effects["Wrapper"] = compact_effects["Wrapper"].map(
        lambda value: ARCH_LABELS.get(str(value).lower(), value)
    )
    compact_effects["Baseline"] = compact_effects["Baseline"].map(
        lambda value: ARCH_LABELS.get(str(value).lower(), value)
    )
    compact_effects = compact_effects[
        [
            "Metric",
            "Wrapper",
            "Baseline",
            "Mean difference",
            "Paired SD",
            "95% CI",
            "Structural roles",
            "Paired role/seed units",
            "Valid bootstrap",
        ]
    ]
    return pd.DataFrame(rows), compact_effects


def ridge_fit(
    x: np.ndarray, y: np.ndarray, alpha: float
) -> Tuple[np.ndarray, float, float]:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ y
    )
    prediction = design @ coefficients
    residual = y - prediction
    ss_res = float(np.sum(residual**2))
    ss_total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_total if ss_total > 0 else np.nan
    rmse = float(np.sqrt(np.mean(residual**2)))
    return coefficients, r2, rmse


def regression_design(
    frame: pd.DataFrame, model: str
) -> Tuple[pd.DataFrame, List[str]]:
    output = frame.copy()
    output["log_gate_count"] = np.log1p(
        pd.to_numeric(output["quantum_gate_count_total"], errors="coerce")
    )
    output["log_parameter_count"] = np.log1p(
        pd.to_numeric(output["trainable_parameters_total"], errors="coerce")
    )
    if model == "M1_gap":
        numeric = ["gap", "reps", "depth"]
    else:
        numeric = [
            "train_acc",
            "test_acc",
            "reps",
            "depth",
            "log_gate_count",
            "log_parameter_count",
        ]
    kept_numeric = []
    for column in numeric:
        values = pd.to_numeric(output[column], errors="coerce")
        if values.notna().all() and values.nunique() > 1:
            mean = float(values.mean())
            sd = float(values.std(ddof=0))
            output[f"z_{column}"] = (values - mean) / sd
            kept_numeric.append(f"z_{column}")
    dummies = pd.get_dummies(
        output["fm_kind"].astype(str),
        prefix="fm",
        drop_first=True,
        dtype=float,
    )
    output = pd.concat([output, dummies], axis=1)
    terms = kept_numeric + list(dummies.columns)
    return output, terms


def regression_tables(
    factorial: pd.DataFrame, bootstrap: int, seed: int, alpha: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficient_rows: List[Dict[str, Any]] = []
    fit_rows: List[Dict[str, Any]] = []
    correlation_rows: List[Dict[str, Any]] = []
    for model in ("M1_gap", "M2_behavior_resources"):
        data, terms = regression_design(factorial, model)
        data = data.dropna(
            subset=["loss_auc", "structural_cell_id", *terms]
        ).reset_index(drop=True)
        x = data[terms].to_numpy(float)
        y = data["loss_auc"].to_numpy(float)
        point, r2, rmse = ridge_fit(x, y, alpha)
        cells = [
            group.index.to_numpy(int)
            for _, group in data.groupby("structural_cell_id", sort=False)
        ]
        rng = stable_rng(seed, "regression", model)
        samples = np.empty((bootstrap, len(point)), dtype=float)
        r2_samples = np.empty(bootstrap, dtype=float)
        rmse_samples = np.empty(bootstrap, dtype=float)
        valid = 0
        for replicate in range(bootstrap):
            try:
                selected_cells = rng.integers(0, len(cells), size=len(cells))
                indices: List[int] = []
                for cell_index in selected_cells:
                    members = cells[int(cell_index)]
                    indices.extend(
                        rng.choice(
                            members, size=len(members), replace=True
                        ).tolist()
                    )
                fit, boot_r2, boot_rmse = ridge_fit(
                    x[indices], y[indices], alpha
                )
                samples[valid] = fit
                r2_samples[valid] = boot_r2
                rmse_samples[valid] = boot_rmse
                valid += 1
            except (ValueError, np.linalg.LinAlgError):
                continue
        samples = samples[:valid]
        r2_samples = r2_samples[:valid]
        rmse_samples = rmse_samples[:valid]
        names = ["Intercept", *terms]
        for index, name in enumerate(names):
            coefficient_rows.append(
                {
                    "Model": model,
                    "Term": name,
                    "Coefficient": point[index],
                    "95% CI": (
                        ci_text(
                            np.quantile(samples[:, index], 0.025),
                            np.quantile(samples[:, index], 0.975),
                        )
                        if valid
                        else "--"
                    ),
                    "Valid bootstrap": valid,
                }
            )
        fit_rows.append(
            {
                "Model": model,
                "Rows": len(data),
                "Structural cells": len(cells),
                "R²": r2,
                "R² 95% CI": (
                    ci_text(
                        np.quantile(r2_samples, 0.025),
                        np.quantile(r2_samples, 0.975),
                    )
                    if valid
                    else "--"
                ),
                "RMSE": rmse,
                "RMSE 95% CI": (
                    ci_text(
                        np.quantile(rmse_samples, 0.025),
                        np.quantile(rmse_samples, 0.975),
                    )
                    if valid
                    else "--"
                ),
                "Valid bootstrap": valid,
                "Interpretation": "descriptive association, not causal",
            }
        )

    cells = [
        group.index.to_numpy(int)
        for _, group in factorial.reset_index(drop=True).groupby(
            "structural_cell_id", sort=False
        )
    ]
    gap = factorial["gap"].to_numpy(float)
    auc = factorial["loss_auc"].to_numpy(float)
    for method in ("Pearson", "Spearman"):
        observed = (
            float(pearsonr(gap, auc).statistic)
            if method == "Pearson"
            else float(spearmanr(gap, auc).statistic)
        )
        rng = stable_rng(seed, "correlation", method)
        values = np.empty(bootstrap, dtype=float)
        for replicate in range(bootstrap):
            selected_cells = rng.integers(0, len(cells), size=len(cells))
            indices: List[int] = []
            for cell_index in selected_cells:
                members = cells[int(cell_index)]
                indices.extend(
                    rng.choice(members, size=len(members), replace=True).tolist()
                )
            values[replicate] = (
                pearsonr(gap[indices], auc[indices]).statistic
                if method == "Pearson"
                else spearmanr(gap[indices], auc[indices]).statistic
            )
        correlation_rows.append(
            {
                "Method": method,
                "Correlation": observed,
                "95% CI": ci_text(
                    np.quantile(values, 0.025), np.quantile(values, 0.975)
                ),
                "Structural cells": len(cells),
                "Target rows": len(factorial),
                "Valid bootstrap": bootstrap,
            }
        )
    return (
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(fit_rows),
        pd.DataFrame(correlation_rows),
    )


def learned_robustness_tables(
    learned: pd.DataFrame, threshold: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    learned = learned.copy()
    learned["architecture"] = learned["architecture"].astype(str).str.lower()
    per_target = (
        learned.groupby(
            ["experiment", "target_id", "architecture", "model_seed"],
            dropna=False,
        )
        .agg(
            learned_auc_mean=("attack_auc", "mean"),
            learned_auc_sd=("attack_auc", "std"),
            learned_tpr5_mean=("tpr@fpr=0.05", "mean"),
            learned_tpr10_mean=("tpr@fpr=0.1", "mean"),
            attacker_seeds=("attacker_seed", "nunique"),
        )
        .reset_index()
    )
    loss = threshold[
        threshold["attack"].astype(str).str.lower().eq("loss")
    ][["target_id", "auc"]].rename(columns={"auc": "loss_threshold_auc"})
    per_target = per_target.merge(loss, on="target_id", how="left")
    per_target["learned_minus_loss_threshold"] = (
        per_target["learned_auc_mean"] - per_target["loss_threshold_auc"]
    )
    grouped = (
        per_target.groupby(["experiment", "architecture"], dropna=False)
        .agg(
            targets=("target_id", "nunique"),
            attacker_seeds=("attacker_seeds", "max"),
            learned_auc_mean=("learned_auc_mean", "mean"),
            learned_auc_sd_across_targets=("learned_auc_mean", "std"),
            attacker_seed_sd_mean=("learned_auc_sd", "mean"),
            loss_threshold_auc_mean=("loss_threshold_auc", "mean"),
            learned_minus_threshold_mean=(
                "learned_minus_loss_threshold",
                "mean",
            ),
            learned_minus_threshold_sd=(
                "learned_minus_loss_threshold",
                "std",
            ),
        )
        .reset_index()
    )
    rows: List[Dict[str, Any]] = []
    for _, row in grouped.iterrows():
        rows.append(
            {
                "Experiment": row["experiment"],
                "Wrapper": ARCH_LABELS.get(
                    str(row["architecture"]), row["architecture"]
                ),
                "Targets": int(row["targets"]),
                "Attacker seeds": int(row["attacker_seeds"]),
                "Learned AUC across targets": mean_sd_text(
                    row["learned_auc_mean"],
                    row["learned_auc_sd_across_targets"],
                ),
                "Mean within-target attacker-seed SD": row[
                    "attacker_seed_sd_mean"
                ],
                "Loss-threshold AUC mean": row["loss_threshold_auc_mean"],
                "Learned − threshold AUC": mean_sd_text(
                    row["learned_minus_threshold_mean"],
                    row["learned_minus_threshold_sd"],
                ),
            }
        )
    return pd.DataFrame(rows), per_target


def completeness_table(
    factorial: pd.DataFrame,
    threshold: pd.DataFrame,
    geometry: pd.DataFrame,
    architecture: pd.DataFrame,
    learned: pd.DataFrame,
    noisy_validation: pd.DataFrame,
    noisy_consistency: pd.DataFrame,
) -> pd.DataFrame:
    learned_status = [
        {
            "Evidence block": "Learned MIA",
            "Expected units": 72,
            "Observed units": int(group["target_id"].nunique()),
            "Status": (
                "complete"
                if group["target_id"].nunique() == 72
                else "incomplete"
            ),
            "Replication unit": f"target checkpoints at attacker seed {seed}",
        }
        for seed, group in learned.groupby("attacker_seed")
    ]
    rows = [
        {
            "Evidence block": "Factorial target models",
            "Expected units": 36,
            "Observed units": int(factorial["target_id"].nunique()),
            "Status": "complete",
            "Replication unit": "target-model seed within 12 structural cells",
        },
        {
            "Evidence block": "Factorial scalar attacks",
            "Expected units": 216,
            "Observed units": len(threshold),
            "Status": "complete",
            "Replication unit": "target × scalar attack",
        },
        {
            "Evidence block": "Encoder geometry",
            "Expected units": 36,
            "Observed units": len(geometry),
            "Status": "complete_with_MNIST_duplicate_seed_caveat",
            "Replication unit": "nominal data seed; uniqueness audited separately",
        },
        {
            "Evidence block": "Architecture wrappers",
            "Expected units": 36,
            "Observed units": int(architecture["target_id"].nunique()),
            "Status": "complete",
            "Replication unit": "wrapper × structural role × target seed",
        },
        {
            "Evidence block": "Noisy targets",
            "Expected units": 15,
            "Observed units": int(noisy_validation["target_id"].nunique()),
            "Status": (
                "complete"
                if noisy_validation["passed"].astype(bool).all()
                else "failed_validation"
            ),
            "Replication unit": "target checkpoint",
        },
        {
            "Evidence block": "Noisy conditions",
            "Expected units": 915,
            "Observed units": len(noisy_consistency),
            "Status": (
                "complete"
                if noisy_consistency["matches_manifest"].astype(bool).all()
                else "failed_consistency"
            ),
            "Replication unit": "target × mode × shots × simulator seed",
        },
        *learned_status,
    ]
    return pd.DataFrame(rows)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )


def plot_factorial_interactions(factorial: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.1), sharex=True)
    metrics = [("gap", "Accuracy gap"), ("loss_auc", "Loss-threshold AUC")]
    for row_index, depth in enumerate((2, 6)):
        subset = factorial[factorial["depth"] == depth]
        for column_index, (metric, label) in enumerate(metrics):
            ax = axes[row_index, column_index]
            for fm in ("eff_su2", "z", "zz"):
                group = subset[subset["fm_kind"] == fm]
                means = group.groupby("reps")[metric].mean()
                stds = group.groupby("reps")[metric].std(ddof=1)
                x = np.asarray(means.index, dtype=float)
                ax.errorbar(
                    x,
                    means.to_numpy(),
                    yerr=stds.to_numpy(),
                    color=FM_COLORS[fm],
                    marker="o",
                    capsize=3,
                    label=FM_LABELS[fm],
                )
                for repetition, seed_group in group.groupby("reps"):
                    jitter = np.linspace(-0.09, 0.09, len(seed_group))
                    ax.scatter(
                        np.full(len(seed_group), repetition) + jitter,
                        seed_group[metric],
                        color=FM_COLORS[fm],
                        s=14,
                        alpha=0.45,
                        edgecolors="none",
                    )
            ax.set_title(f"Depth {depth}: {label}")
            ax.set_xticks([1, 5])
            ax.set_xlabel("Encoder repetitions")
            ax.set_ylabel(label)
            if metric == "loss_auc":
                ax.axhline(0.5, color="0.5", linestyle=":", linewidth=1)
    axes[0, 0].legend(frameon=False, ncol=3, loc="best")
    fig.suptitle(
        "Controlled MNIST-QNN factorial: target seeds and mean ± SD",
        y=1.01,
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def plot_gap_auc(factorial: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.3, 4.2))
    markers = {(1, 2): "o", (1, 6): "s", (5, 2): "^", (5, 6): "D"}
    for fm in ("eff_su2", "z", "zz"):
        for (reps, depth), marker in markers.items():
            group = factorial[
                (factorial["fm_kind"] == fm)
                & (factorial["reps"] == reps)
                & (factorial["depth"] == depth)
            ]
            ax.scatter(
                group["gap"],
                group["loss_auc"],
                color=FM_COLORS[fm],
                marker=marker,
                s=38,
                alpha=0.8,
                label=(
                    f"{FM_LABELS[fm]}, r={reps}, d={depth}"
                    if len(group)
                    else None
                ),
            )
    slope, intercept = np.polyfit(factorial["gap"], factorial["loss_auc"], 1)
    grid = np.linspace(factorial["gap"].min(), factorial["gap"].max(), 100)
    ax.plot(grid, intercept + slope * grid, color="0.25", linewidth=1.2)
    pearson = pearsonr(factorial["gap"], factorial["loss_auc"]).statistic
    spearman = spearmanr(factorial["gap"], factorial["loss_auc"]).statistic
    ax.text(
        0.03,
        0.97,
        f"Pearson r={pearson:.2f}\nSpearman ρ={spearman:.2f}",
        transform=ax.transAxes,
        va="top",
    )
    ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1)
    ax.set_xlabel("Retrained accuracy gap")
    ax.set_ylabel("Loss-threshold MIA AUC")
    ax.set_title("Generalization gap is informative but not deterministic")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        frameon=False,
        fontsize=6.5,
        ncol=2,
        loc="lower right",
    )
    fig.tight_layout()
    return fig


def plot_attack_suite(
    threshold: pd.DataFrame, learned: pd.DataFrame
) -> plt.Figure:
    records = []
    for attack, group in threshold.groupby("attack"):
        records.append(
            (str(attack), group["auc"].mean(), group["auc"].std(ddof=1))
        )
    learned_factorial = learned[
        learned["experiment"].astype(str).str.lower().eq("multiseed_factorial")
    ]
    learned_target = learned_factorial.groupby("target_id")["attack_auc"].mean()
    records.append(
        (
            "learned PV",
            learned_target.mean(),
            learned_target.std(ddof=1),
        )
    )
    records = sorted(records, key=lambda item: item[1], reverse=True)
    names = [record[0] for record in records]
    means = [record[1] for record in records]
    errors = [record[2] for record in records]
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    positions = np.arange(len(records))
    ax.errorbar(
        positions,
        means,
        yerr=errors,
        fmt="o",
        color="#0072B2",
        capsize=3,
    )
    ax.axhline(0.5, color="0.5", linestyle=":", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("AUC (mean ± SD across target units)")
    ax.set_title("Attack-signal decomposition across the full factorial")
    fig.tight_layout()
    return fig


def plot_geometry(raw: pd.DataFrame) -> plt.Figure:
    metrics = [
        ("effective_rank", "Effective rank"),
        ("kernel_label_alignment", "Kernel-label alignment"),
        ("class_similarity_gap", "Class-similarity gap"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1))
    for ax, (metric, label) in zip(axes, metrics):
        for dataset, linestyle in (("MNIST", "-"), ("Moons", "--")):
            for fm in ("eff_su2", "z", "zz"):
                group = raw[
                    (raw["dataset"].astype(str) == dataset)
                    & (raw["fm_kind"].astype(str) == fm)
                ]
                aggregate = group.groupby("reps")[metric].agg(["mean", "std"])
                ax.errorbar(
                    aggregate.index,
                    aggregate["mean"],
                    yerr=aggregate["std"].fillna(0),
                    color=FM_COLORS[fm],
                    linestyle=linestyle,
                    marker="o",
                    capsize=2,
                    label=f"{dataset} {FM_LABELS[fm]}",
                )
        ax.set_xticks([1, 5])
        ax.set_xlabel("Encoder repetitions")
        ax.set_ylabel(label)
        ax.set_title(label)
    axes[-1].legend(frameon=False, fontsize=6.5, bbox_to_anchor=(1.02, 1))
    fig.suptitle(
        "Direct post-encoder Hilbert-space geometry (mean ± nominal-seed SD)",
        y=1.03,
        fontsize=10.5,
    )
    fig.tight_layout()
    return fig


def plot_noise(summary: pd.DataFrame) -> plt.Figure:
    loss = summary[summary["attack"].astype(str).str.lower().eq("loss")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    cells = sorted(loss["structural_cell_id"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, len(cells)))
    for cell, color in zip(cells, palette):
        cell_data = loss[loss["structural_cell_id"] == cell]
        exact = cell_data[cell_data["mode"] == "exact"]
        for mode, linestyle in (("ideal_shot", "--"), ("noisy_shot", "-")):
            shot = cell_data[cell_data["mode"] == mode].sort_values("shots")
            if shot.empty:
                continue
            for ax, metric, sd in (
                (axes[0], "test_accuracy_mean", "test_accuracy_sd"),
                (axes[1], "roc_auc_mean", "roc_auc_sd"),
            ):
                ax.errorbar(
                    shot["shots"],
                    shot[metric],
                    yerr=shot[sd],
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    capsize=2,
                    label=f"{cell} {mode.replace('_shot', '')}",
                )
                if not exact.empty:
                    ax.axhline(
                        float(exact.iloc[0][metric]),
                        color=color,
                        alpha=0.18,
                        linewidth=0.8,
                    )
    axes[0].set_title("Target test accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Loss-threshold MIA")
    axes[1].set_ylabel("ROC AUC")
    axes[1].axhline(0.5, color="0.5", linestyle=":", linewidth=1)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([128, 512, 1024])
        ax.set_xticklabels(["128", "512", "1024"])
        ax.set_xlabel("Shots")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(
        handles,
        labels,
        frameon=False,
        fontsize=5.7,
        ncol=2,
        bbox_to_anchor=(1.02, 1),
    )
    fig.suptitle(
        "Exact references (faint) versus ideal- and backend-noisy finite shots",
        y=1.02,
        fontsize=10.5,
    )
    fig.tight_layout()
    return fig


def plot_architecture(raw: pd.DataFrame) -> plt.Figure:
    data = raw.copy()
    data["architecture"] = data["architecture"].astype(str).str.lower()
    metrics = [
        ("gap_acc", "Accuracy gap"),
        ("auc", "Loss-threshold AUC"),
        ("trainable_parameters_total", "Trainable parameters"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.3))
    for ax, (metric, label) in zip(axes, metrics):
        aggregate = data.groupby("architecture")[metric].agg(["mean", "std"])
        means = [aggregate.loc[arch, "mean"] for arch in ARCH_ORDER]
        errors = [aggregate.loc[arch, "std"] for arch in ARCH_ORDER]
        positions = np.arange(len(ARCH_ORDER))
        ax.errorbar(
            positions,
            means,
            yerr=errors,
            fmt="o",
            color="#6A3D9A",
            capsize=3,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [ARCH_LABELS[arch] for arch in ARCH_ORDER],
            rotation=25,
            ha="right",
        )
        ax.set_ylabel(label)
        ax.set_title(label)
        if metric == "auc":
            ax.axhline(0.5, color="0.5", linestyle=":", linewidth=1)
        if metric == "trainable_parameters_total":
            ax.set_yscale("log")
    fig.suptitle(
        "Complete-wrapper comparison (not a pure quantum-architecture effect)",
        y=1.03,
        fontsize=10.5,
    )
    fig.tight_layout()
    return fig


def plot_learned_vs_threshold(per_target: pd.DataFrame) -> plt.Figure:
    data = per_target[
        per_target["experiment"].astype(str).str.lower().eq(
            "multiseed_factorial"
        )
    ].dropna(subset=["loss_threshold_auc"])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(
        data["loss_threshold_auc"],
        data["learned_auc_mean"],
        color="#CC79A7",
        alpha=0.8,
        s=32,
    )
    low = min(
        data["loss_threshold_auc"].min(), data["learned_auc_mean"].min()
    )
    high = max(
        data["loss_threshold_auc"].max(), data["learned_auc_mean"].max()
    )
    ax.plot([low, high], [low, high], color="0.5", linestyle=":", linewidth=1)
    ax.axhline(0.5, color="0.8", linewidth=0.8)
    ax.axvline(0.5, color="0.8", linewidth=0.8)
    ax.set_xlabel("Loss-threshold AUC")
    ax.set_ylabel("Learned prediction-vector AUC\n(mean across attacker seeds)")
    ax.set_title("Learned and scalar attacks on the same target checkpoints")
    fig.tight_layout()
    return fig


def reviewer_evidence_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Reviewer": "Epmi",
                "Concern": "Encoder effect versus ordinary overfitting; gap is not MIA",
                "Evidence": "T01, T02, T07; F01, F02",
                "Supported response": "Report paired structural effects and gap–AUC association; interpret residual coefficients descriptively and avoid a direct causal claim.",
                "Residual limitation": "Fixed primary data split; observational mediation evidence is not causal identification.",
            },
            {
                "Reviewer": "Epmi / 1myw",
                "Concern": "Narrow attack suite and unclear driving output signal",
                "Evidence": "T03, T08, T10; F03, F07",
                "Supported response": (
                    "Separate loss, entropy, confidence, margin, correctness and "
                    "max-probability attacks; a learned prediction-vector attacker; "
                    "calibrated online/offline LiRA; and a class-label-only boundary proxy."
                ),
                "Residual limitation": (
                    "LiRA uses 16 references per structural configuration and an approximate "
                    "reference-training distribution; label-only uses a chord-boundary proxy "
                    "rather than a certified minimum boundary distance."
                ),
            },
            {
                "Reviewer": "Epmi",
                "Concern": "Hilbert-space mechanism is indirect",
                "Evidence": "T04; F04",
                "Supported response": "Report kernel alignment, class-similarity gap, train/test MMD² and effective rank immediately after the fixed encoder.",
                "Residual limitation": "MNIST nominal geometry seeds duplicate the same states; only Moons supplies genuine data-seed variability.",
            },
            {
                "Reviewer": "1myw / Area Chair",
                "Concern": "Noiseless simulation and finite-shot robustness",
                "Evidence": "T05; F05",
                "Supported response": "Compare exact, ideal finite-shot and IBM-backend-derived noisy finite-shot results with paired simulator-seed uncertainty.",
                "Residual limitation": (
                    "Backend-derived Aer noise model, not execution on quantum hardware; "
                    "five prespecified structural configurations."
                ),
            },
            {
                "Reviewer": "nVBH",
                "Concern": "No multi-seed uncertainty and post-hoc selected regimes",
                "Evidence": "T01, T02, T03, T09; F01–F03",
                "Supported response": "Report all 36 prespecified factorial targets over three target seeds with no outcome filtering and paired/hierarchical intervals.",
                "Residual limitation": "One fixed split in the primary factorial and three, not five, learned-attacker seeds.",
            },
            {
                "Reviewer": "Epmi / nVBH",
                "Concern": "Confounded architecture comparisons and missing capacity accounting",
                "Evidence": "T06; F06",
                "Supported response": "Report complete-wrapper comparisons paired within role/seed together with quantum/classical parameter and gate counts.",
                "Residual limitation": "Wrappers retain different preprocessing and heads; results are not pure causal architecture effects.",
            },
            {
                "Reviewer": "1myw / nVBH",
                "Concern": "Simple datasets and limited external validity",
                "Evidence": "Scope statement in reviewer index",
                "Supported response": "Bound claims to controlled MNIST/Moons simulation and describe sensitive-domain transfer as untested.",
                "Residual limitation": "No healthcare, finance, or public-sector dataset was added.",
            },
        ]
    )


def build_index(
    output_dir: Path,
    artifacts: Dict[str, Dict[str, str]],
    evidence: pd.DataFrame,
) -> str:
    lines = [
        "# Reviewer-response artifact index",
        "",
        "All paths are relative to this directory. Numerical tables are provided as CSV and LaTeX; figures are provided as PNG and PDF.",
        "",
        "## Concern-to-evidence map",
        "",
    ]
    for _, row in evidence.iterrows():
        lines.extend(
            [
                f"### {row['Reviewer']}: {row['Concern']}",
                "",
                f"- Evidence: {row['Evidence']}",
                f"- Supported wording: {row['Supported response']}",
                f"- Required caveat: {row['Residual limitation']}",
                "",
            ]
        )
    lines.extend(["## Generated artifacts", ""])
    for key, paths in artifacts.items():
        rendered = ", ".join(
            f"[{kind}]({Path(path).relative_to(output_dir)})"
            for kind, path in paths.items()
        )
        lines.append(f"- **{key}**: {rendered}")
    lines.extend(
        [
            "",
            "## Statistical interpretation",
            "",
            "- Factorial and architecture error bars summarize independent target-model seeds within structural cells.",
            "- Factorial main-effect intervals resample structural comparison blocks with target seeds nested.",
            "- Noisy-condition intervals resample paired simulator seeds within a fixed target checkpoint; they are not target-training uncertainty.",
            "- Learned-MIA attacker-seed SD describes attack-training sensitivity on fixed target outputs.",
            "- Record-bootstrap AUC intervals are not used as substitutes for target-model seed uncertainty.",
            "- The analyses are descriptive and controlled, but they do not establish that encoder choice causes leakage independently of overfitting.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--results-root", type=Path, default=Path("reviewer_results")
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reviewer_results/reviewer_artifacts"),
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")

    repo_root = args.repo_root.resolve()
    results_root = (
        args.results_root
        if args.results_root.is_absolute()
        else repo_root / args.results_root
    )
    output_dir = (
        args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    )
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading completed result tables...", flush=True)
    factorial_metrics = require_csv(
        results_root / "factorial_metrics/retrained_target_metrics_raw.csv"
    )
    factorial_threshold = require_csv(
        results_root / "factorial_threshold_mia/threshold_mia_raw.csv"
    )
    factorial_resources = require_csv(
        results_root / "factorial_resources/model_resources_raw.csv"
    )
    geometry_raw = require_csv(
        results_root / "geometry_multiseed/geometry_raw.csv"
    )
    geometry_summary = require_csv(
        results_root / "geometry_multiseed/geometry_summary.csv"
    )
    geometry_effects = require_csv(
        results_root / "geometry_multiseed/geometry_repetition_effects.csv"
    )
    noisy_summary = require_csv(
        results_root
        / "noisy_sanity/combined/noisy_mia_target_seed_summary.csv"
    )
    noisy_changes = require_csv(
        results_root
        / "noisy_sanity/combined/noisy_changes_simulator_summary.csv"
    )
    noisy_validation = require_csv(
        results_root / "noisy_sanity/noisy_validation_report.csv"
    )
    noisy_consistency = require_csv(
        results_root / "noisy_sanity/combined/sample_consistency_report.csv"
    )
    architecture_raw = require_csv(
        results_root / "architecture_control/architecture_control_raw.csv"
    )
    architecture_effects = require_csv(
        results_root / "architecture_control/architecture_control_effects.csv"
    )
    architecture_threshold = require_csv(
        results_root / "architecture_threshold_mia/threshold_mia_raw.csv"
    )
    learned = load_learned_summaries(results_root)
    factorial = prepare_factorial(
        factorial_metrics, factorial_threshold, factorial_resources
    )

    artifacts: Dict[str, Dict[str, str]] = {}
    print("[2/6] Building reviewer-facing compact tables...", flush=True)
    artifacts["T01 factorial cells"] = save_table(
        factorial_cell_table(factorial),
        tables_dir,
        "T01_factorial_cells",
        "Prespecified MNIST-QNN factorial cells. Values are mean $\\pm$ sample SD across three independently trained target-model seeds.",
        "tab:factorial-cells",
    )
    factorial_effects = factorial_effect_table(
        factorial, args.bootstrap, args.bootstrap_seed
    )
    artifacts["T02 factorial paired effects"] = save_table(
        factorial_effects,
        tables_dir,
        "T02_factorial_paired_effects",
        "Paired factorial effects with hierarchical percentile-bootstrap confidence intervals over structural blocks and nested target seeds.",
        "tab:factorial-effects",
    )
    artifacts["T03 attack suite"] = save_table(
        attack_suite_table(factorial_threshold, learned),
        tables_dir,
        "T03_attack_suite",
        "Attack-signal decomposition over all prespecified factorial target checkpoints.",
        "tab:attack-suite",
    )
    geometry_table, geometry_effect_table, geometry_audit = geometry_tables(
        geometry_raw,
        geometry_summary,
        geometry_effects,
        args.bootstrap,
        args.bootstrap_seed,
    )
    artifacts["T04a geometry cells"] = save_table(
        geometry_table,
        tables_dir,
        "T04a_geometry_cells",
        "Direct post-encoder Hilbert-space geometry. The unique-realization column prevents duplicated nominal seeds from being interpreted as independent evidence.",
        "tab:geometry-cells",
    )
    artifacts["T04b geometry effects"] = save_table(
        geometry_effect_table,
        tables_dir,
        "T04b_geometry_repetition_effects",
        "Paired effects of increasing encoder repetitions from one to five.",
        "tab:geometry-effects",
    )
    artifacts["T04c geometry seed audit"] = save_table(
        geometry_audit,
        tables_dir,
        "T04c_geometry_seed_audit",
        "Audit of nominal versus unique encoder-geometry realizations.",
        "tab:geometry-seed-audit",
    )
    noisy_table, noisy_delta_table = noisy_tables(
        noisy_summary, noisy_changes
    )
    artifacts["T05a noisy conditions"] = save_table(
        noisy_table,
        tables_dir,
        "T05a_noisy_conditions",
        "Exact, ideal finite-shot and IBM-backend-derived noisy finite-shot loss-MIA results.",
        "tab:noisy-conditions",
    )
    artifacts["T05b noisy changes"] = save_table(
        noisy_delta_table,
        tables_dir,
        "T05b_noisy_changes",
        "Paired finite-shot and noisy changes from exact evaluation; intervals resample simulator seeds within fixed target checkpoints.",
        "tab:noisy-changes",
    )
    architecture_table, architecture_effect_table = architecture_tables(
        architecture_raw, architecture_effects
    )
    artifacts["T06a architecture wrappers"] = save_table(
        architecture_table,
        tables_dir,
        "T06a_architecture_wrappers",
        "Complete-wrapper utility, privacy and resource accounting by structural role.",
        "tab:architecture-wrappers",
    )
    artifacts["T06b architecture effects"] = save_table(
        architecture_effect_table,
        tables_dir,
        "T06b_architecture_paired_effects",
        "Paired complete-wrapper effects relative to QNN. These are not pure causal quantum-architecture effects.",
        "tab:architecture-effects",
    )

    print("[3/6] Fitting gap–AUC descriptive regressions...", flush=True)
    coefficients, fits, correlations = regression_tables(
        factorial, args.bootstrap, args.bootstrap_seed, args.ridge_alpha
    )
    artifacts["T07a regression coefficients"] = save_table(
        coefficients,
        tables_dir,
        "T07a_gap_auc_regression_coefficients",
        "Standardized descriptive ridge-regression coefficients with structural-cell bootstrap intervals.",
        "tab:gap-auc-coefficients",
    )
    artifacts["T07b regression fit"] = save_table(
        fits,
        tables_dir,
        "T07b_gap_auc_regression_fit",
        "Descriptive regression fit and structural-cell bootstrap uncertainty.",
        "tab:gap-auc-fit",
    )
    artifacts["T07c gap correlations"] = save_table(
        correlations,
        tables_dir,
        "T07c_gap_auc_correlations",
        "Pearson and Spearman associations between retrained accuracy gap and loss-threshold MIA AUC.",
        "tab:gap-auc-correlations",
    )

    learned_table, learned_per_target = learned_robustness_tables(
        learned,
        pd.concat(
            [factorial_threshold, architecture_threshold],
            ignore_index=True,
        ),
    )
    artifacts["T08 learned MIA robustness"] = save_table(
        learned_table,
        tables_dir,
        "T08_learned_mia_attacker_seed_robustness",
        "Learned prediction-vector MIA sensitivity across attacker seeds and comparison with loss-threshold MIA.",
        "tab:learned-mia-robustness",
    )
    evidence = reviewer_evidence_map()
    artifacts["T09a reviewer evidence map"] = save_table(
        evidence,
        tables_dir,
        "T09a_reviewer_evidence_map",
        "Mapping from reviewer concerns to new evidence and remaining limitations.",
        "tab:reviewer-evidence-map",
    )
    completeness = completeness_table(
        factorial,
        factorial_threshold,
        geometry_raw,
        architecture_raw,
        learned,
        noisy_validation,
        noisy_consistency,
    )
    artifacts["T09b experiment completeness"] = save_table(
        completeness,
        tables_dir,
        "T09b_experiment_completeness",
        "Completeness and replication-unit audit for the reviewer experiments.",
        "tab:experiment-completeness",
    )

    print("[4/6] Rendering publication figures...", flush=True)
    configure_plot_style()
    artifacts["F01 factorial interactions"] = save_figure(
        plot_factorial_interactions(factorial),
        figures_dir,
        "F01_factorial_interactions",
    )
    artifacts["F02 gap versus AUC"] = save_figure(
        plot_gap_auc(factorial), figures_dir, "F02_gap_vs_loss_auc"
    )
    artifacts["F03 attack suite"] = save_figure(
        plot_attack_suite(factorial_threshold, learned),
        figures_dir,
        "F03_attack_suite",
    )
    artifacts["F04 encoder geometry"] = save_figure(
        plot_geometry(geometry_raw), figures_dir, "F04_encoder_geometry"
    )
    artifacts["F05 noisy finite shots"] = save_figure(
        plot_noise(noisy_summary), figures_dir, "F05_noisy_finite_shots"
    )
    artifacts["F06 architecture wrappers"] = save_figure(
        plot_architecture(architecture_raw),
        figures_dir,
        "F06_architecture_wrappers",
    )
    artifacts["F07 learned versus threshold"] = save_figure(
        plot_learned_vs_threshold(learned_per_target),
        figures_dir,
        "F07_learned_vs_threshold",
    )

    print("[5/6] Writing reviewer-response index and manifest...", flush=True)
    index_path = output_dir / "README.md"
    write_text(index_path, build_index(output_dir, artifacts, evidence))
    manifest = {
        "script": str(Path(__file__).resolve()),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "ridge_alpha": args.ridge_alpha,
        "artifacts": artifacts,
        "important_caveats": [
            "MNIST geometry nominal data seeds duplicate identical encoded states.",
            "The primary factorial uses one fixed data split.",
            "Architecture results compare complete wrappers, not isolated quantum architectures.",
            "Noise is IBM-backend-derived Aer simulation, not hardware execution.",
            "Only three learned-attacker seeds (41, 42, 43) are available.",
            (
                "LiRA uses 16 references per structural configuration and an approximate "
                "reference-training distribution; label-only is a chord-boundary proxy, "
                "not a certified minimum-distance method."
            ),
        ],
    }
    manifest_path = output_dir / "manifest.json"
    write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))

    print("[6/6] Validating generated artifacts...", flush=True)
    missing = []
    for paths in artifacts.values():
        for path in paths.values():
            candidate = Path(path)
            if not candidate.exists() or candidate.stat().st_size == 0:
                missing.append(str(candidate))
    if missing:
        raise RuntimeError(f"Generated artifacts are missing or empty: {missing}")
    print(
        f"[OK] tables={sum(1 for item in artifacts if item.startswith('T'))} "
        f"figures={sum(1 for item in artifacts if item.startswith('F'))} "
        f"-> {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
