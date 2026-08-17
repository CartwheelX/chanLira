#!/usr/bin/env python3
"""Generate paste-ready Markdown tables for each QuRiFT reviewer response."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    def clean(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and np.isnan(value):
            return "—"
        return str(value).replace("|", "\\|").replace("\n", " ")

    output = [
        "| " + " | ".join(clean(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend(
        "| " + " | ".join(clean(value) for value in row) + " |" for row in rows
    )
    return "\n".join(output)


def effect(mean: float, sd: float) -> str:
    return f"{float(mean):+.3f} ± {float(sd):.3f}"


def fmt(value: float) -> str:
    return f"{float(value):.3f}"


def selected_factorial_effects(frame: pd.DataFrame) -> str:
    selections = [
        ("Repetitions", "5 − 1"),
        ("Depth", "6 − 2"),
        ("Feature map", "Z − EffSU2"),
        ("Feature map", "ZZ − EffSU2"),
        ("Feature map", "ZZ − Z"),
    ]
    metrics = ["Accuracy gap", "Loss AUC"]
    rows = []
    for factor_name, contrast in selections:
        for metric in metrics:
            match = frame[
                frame["Factor"].eq(factor_name)
                & frame["Contrast"].eq(contrast)
                & frame["Metric"].eq(metric)
            ]
            if len(match) != 1:
                raise ValueError(f"Missing factorial effect: {factor_name}/{contrast}/{metric}")
            row = match.iloc[0]
            rows.append(
                [
                    factor_name,
                    contrast,
                    metric,
                    effect(row["Mean difference"], row["Paired-unit SD"]),
                    row["95% CI"],
                    int(row["Paired seed units"]),
                ]
            )
    return markdown_table(
        ["Factor", "Contrast", "Metric", "Mean difference ± SD", "95% CI", "Paired units"],
        rows,
    )


def factorial_cells(frame: pd.DataFrame) -> str:
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                row["Feature map"],
                int(row["Repetitions"]),
                int(row["Depth"]),
                int(row["Target seeds"]),
                row["Train accuracy"],
                row["Test accuracy"],
                row["Accuracy gap"],
                row["Loss AUC"],
                row["Loss TPR@5% FPR"],
                row["Loss TPR@10% FPR"],
            ]
        )
    return markdown_table(
        [
            "Feature map",
            "Reps",
            "Depth",
            "Seeds",
            "Train accuracy",
            "Test accuracy",
            "Gap",
            "Loss AUC",
            "TPR@5% FPR",
            "TPR@10% FPR",
        ],
        rows,
    )


def attack_suite(frame: pd.DataFrame) -> str:
    rows = []
    names = {
        "confidence": "Confidence threshold",
        "correctness": "Correctness",
        "entropy": "Entropy threshold",
        "loss": "Loss threshold",
        "margin": "Margin threshold",
        "max_probability": "Maximum probability",
        "learned prediction-vector": "Learned prediction vector",
    }
    for _, row in frame.iterrows():
        access = row["Access"]
        if row["Attack"] == "correctness":
            access = "predicted label + known candidate label"
        rows.append(
            [
                names.get(row["Attack"], row["Attack"]),
                access,
                int(row["Target units"]),
                int(row["Attacker seeds"]),
                row["AUC"],
                row["TPR@5% FPR"],
                row["TPR@10% FPR"],
            ]
        )
    return markdown_table(
        ["Attack", "Access", "Targets", "Attacker seeds", "AUC", "TPR@5% FPR", "TPR@10% FPR"],
        rows,
    )


def calibrated_and_label_only_attacks(frame: pd.DataFrame) -> str:
    wanted = [
        "Online LiRA, fixed variance",
        "Online LiRA, per-record variance",
        "Offline LiRA, fixed variance",
        "Label-only chord-boundary",
    ]
    selected = frame[frame["Attack"].isin(wanted)]
    rows = []
    for _, row in selected.iterrows():
        rows.append(
            [
                row["Attack"],
                row["Access"],
                int(row["Targets"]),
                row["Reference models/configuration"],
                row["AUC"],
                row["TPR@5% FPR"],
                row["TPR@10% FPR"],
            ]
        )
    return markdown_table(
        [
            "Attack",
            "Access",
            "Targets",
            "References/configuration",
            "AUC",
            "TPR@5% FPR",
            "TPR@10% FPR",
        ],
        rows,
    )


def gap_regression(coefficients: pd.DataFrame, fits: pd.DataFrame, correlations: pd.DataFrame) -> str:
    rows = []
    for _, row in correlations.iterrows():
        rows.append(
            [
                f"Gap–AUC {row['Method']} correlation",
                fmt(row["Correlation"]),
                row["95% CI"],
                f"{int(row['Structural cells'])} structural configurations / "
                f"{int(row['Target rows'])} targets",
            ]
        )
    fit = fits[fits["Model"].eq("M1_gap")].iloc[0]
    rows.append(["Gap + structure regression R²", fmt(fit["R²"]), fit["R² 95% CI"], "36 targets"])
    labels = {
        "z_gap": "Standardized gap coefficient",
        "z_reps": "Residual repetitions coefficient",
        "z_depth": "Residual depth coefficient",
        "fm_z": "Residual Z coefficient",
        "fm_zz": "Residual ZZ coefficient",
    }
    for term in labels:
        row = coefficients[
            coefficients["Model"].eq("M1_gap") & coefficients["Term"].eq(term)
        ].iloc[0]
        rows.append([labels[term], fmt(row["Coefficient"]), row["95% CI"], "5,000 bootstrap replicates"])
    return markdown_table(["Analysis", "Estimate", "95% CI", "Units/notes"], rows)


def geometry_effects(frame: pd.DataFrame) -> str:
    labels = {
        "class_similarity_gap": "Class-similarity gap",
        "kernel_label_alignment": "Kernel–label alignment",
        "effective_rank": "Effective rank",
        "mmd2_train_test": "Train/test MMD²",
        "encoder_operation_count": "Encoder operation count",
    }
    rows = []
    for metric, label in labels.items():
        row = frame[frame["Metric"].eq(metric)].iloc[0]
        rows.append(
            [
                label,
                row["Contrast"],
                effect(row["Mean difference"], row["Unique paired-effect SD"]),
                row["95% CI"],
                int(row["Unique paired seed effects"]),
            ]
        )
    return markdown_table(
        ["Geometry metric", "Contrast", "Mean difference ± SD", "95% CI", "Unique paired effects"],
        rows,
    )


def noise_table(frame: pd.DataFrame) -> str:
    low = frame[frame["Risk configuration"].eq("low_risk_anchor")]
    high = frame[frame["Risk configuration"].eq("high_risk_anchor")]
    conditions = [
        ("exact", "exact", "Exact"),
        ("ideal_shot", "128", "Ideal shot, 128"),
        ("ideal_shot", "512", "Ideal shot, 512"),
        ("ideal_shot", "1024", "Ideal shot, 1024"),
        ("noisy_shot", "128", "Backend-noisy, 128"),
        ("noisy_shot", "512", "Backend-noisy, 512"),
        ("noisy_shot", "1024", "Backend-noisy, 1024"),
    ]
    rows = []
    for mode, shots, label in conditions:
        low_row = low[low["Mode"].eq(mode) & low["Shots"].astype(str).eq(shots)].iloc[0]
        high_row = high[high["Mode"].eq(mode) & high["Shots"].astype(str).eq(shots)].iloc[0]
        low_auc = float(str(low_row["Loss AUC"]).split("±")[0])
        high_auc = float(str(high_row["Loss AUC"]).split("±")[0])
        rows.append(
            [
                label,
                low_row["Test accuracy"],
                high_row["Test accuracy"],
                low_row["Loss AUC"],
                high_row["Loss AUC"],
                f"{high_auc - low_auc:+.3f}",
            ]
        )
    return markdown_table(
        [
            "Condition",
            "Low-risk test accuracy",
            "High-risk test accuracy",
            "Low-risk loss AUC",
            "High-risk loss AUC",
            "High − low AUC",
        ],
        rows,
    )


def architecture_effects(frame: pd.DataFrame) -> str:
    labels = {"gap_acc": "Accuracy gap", "test_acc": "Test accuracy", "auc": "Loss AUC"}
    rows = []
    for wrapper in ["HQNN", "MLP-QNN", "QCNN"]:
        for metric in ["test_acc", "gap_acc", "auc"]:
            row = frame[frame["Wrapper"].eq(wrapper) & frame["Metric"].eq(metric)].iloc[0]
            rows.append(
                [
                    wrapper,
                    labels[metric],
                    effect(row["Mean difference"], row["Paired SD"]),
                    row["95% CI"],
                    int(row["Paired role/seed units"]),
                ]
            )
    return markdown_table(
        ["Wrapper vs QNN", "Metric", "Mean difference ± SD", "95% CI", "Paired units"],
        rows,
    )


def architecture_resources(frame: pd.DataFrame) -> str:
    rows = []
    for _, row in frame.iterrows():
        # Legacy resource payloads assigned the reference QNN's analytic gate
        # count to the entirely classical parameter-matched MLP.  Keep this
        # presentation correct even when it is generated from such an artifact.
        quantum_gates = 0 if row["Wrapper"] == "MLP-QNN" else int(row["Quantum gate count"])
        rows.append(
            [
                row["Structural role"],
                row["Wrapper"],
                row["Test accuracy"],
                row["Accuracy gap"],
                row["Loss AUC"],
                int(row["Total trainable parameters"]),
                int(row["Quantum parameters"]),
                int(row["Classical parameters"]),
                quantum_gates,
            ]
        )
    return markdown_table(
        [
            "Role",
            "Wrapper",
            "Test accuracy",
            "Gap",
            "Loss AUC",
            "Total params",
            "Quantum params",
            "Classical params",
            "Quantum gates",
        ],
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("reviewer_results/reviewer_artifacts"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reviewer_results/reviewer_artifacts/REVIEWER_RESPONSE_TABLES.md"),
    )
    args = parser.parse_args()
    table_dir = args.artifact_dir / "tables"
    read = lambda name: pd.read_csv(table_dir / name)
    t01 = read("T01_factorial_cells.csv")
    t02 = read("T02_factorial_paired_effects.csv")
    t03 = read("T03_attack_suite.csv")
    t04 = read("T04b_geometry_repetition_effects.csv")
    t05 = read("T05a_noisy_conditions.csv")
    t06a = read("T06a_architecture_wrappers.csv")
    t06b = read("T06b_architecture_paired_effects.csv")
    t07a = read("T07a_gap_auc_regression_coefficients.csv")
    t07b = read("T07b_gap_auc_regression_fit.csv")
    t07c = read("T07c_gap_auc_correlations.csv")
    completed_attacks = pd.read_csv(
        args.artifact_dir / "final_responses" / "T10a_attack_breadth.csv"
    )

    sections = [
        "# Paste-ready reviewer-response tables",
        "",
        (
            "All `mean ± SD` entries use the replication unit stated below each table. "
            "Bootstrap confidence intervals are shown separately and are not converted "
            "to standard deviations. The completed attack suite includes calibrated LiRA "
            "and a query-based class-label-only boundary proxy. These are focused rebuttal "
            "experiments supplementing the submission's broader exploratory sweep; they are "
            "not a multi-seed rerun of every originally swept configuration."
        ),
        "",
        "## Area Chair",
        "",
        "### AC-1. Prespecified factorial effects",
        "",
        selected_factorial_effects(t02),
        "",
        (
            "*Values are mean paired differences ± SD across paired target-seed "
            "units. The 95% intervals are hierarchical percentile-bootstrap intervals "
            "over structural blocks with target seeds nested (5,000 replicates).*"
        ),
        "",
        "### AC-2. Finite-shot and backend-noise sanity check",
        "",
        noise_table(t05),
        "",
        (
            "*Condition entries are mean ± sample SD across three independently trained "
            "target checkpoints. Shot/noise conditions additionally use ten simulator "
            "seeds per target. The final contrast column is the difference between the "
            "displayed target-seed means and does not have an independently estimated SD. "
            "The noise model is IBM-backend-derived Aer simulation, not hardware execution.*"
        ),
        "",
        "## Reviewer Epmi",
        "",
        "### Epmi-1. Structural effects on gap and loss-based MIA",
        "",
        selected_factorial_effects(t02),
        "",
        "*Mean paired difference ± paired-unit SD; 95% hierarchical bootstrap CI.*",
        "",
        "### Epmi-2. Direct gap–MIA analysis",
        "",
        gap_regression(t07a, t07b, t07c),
        "",
        (
            "*Regression predictors are standardized. Confidence intervals are cluster/"
            "hierarchical bootstrap intervals. Because these analyses provide intervals "
            "rather than a seed-level SD for each coefficient, no artificial ± value is shown.*"
        ),
        "",
        "### Epmi-3. Attack-signal decomposition",
        "",
        attack_suite(t03),
        "",
        "#### Calibrated and class-label-only baselines",
        "",
        calibrated_and_label_only_attacks(completed_attacks),
        "",
        (
            "*Attack entries are mean ± sample SD across the 36 target-model units. "
            "The learned attack is averaged over three attacker seeds per target; scalar "
            "threshold attacks have no attacker-training seed. LiRA uses 16 reference "
            "QNNs per structural configuration.*"
        ),
        "",
        "### Epmi-4. Direct encoder-geometry effects",
        "",
        geometry_effects(t04),
        "",
        (
            "*Mean paired repetition effect ± SD across unique paired effects. MNIST "
            "nominal geometry seeds duplicated encoded states, so zero MNIST configuration-level SD must "
            "not be interpreted as independent-seed robustness; Moons supplies genuine "
            "data-seed variation.*"
        ),
        "",
        "### Epmi-5. Complete-wrapper architecture effects",
        "",
        architecture_effects(t06b),
        "",
        (
            "*Mean paired wrapper-minus-QNN difference ± SD across nine role/target-seed "
            "pairs; 95% paired hierarchical bootstrap CI. Wrappers have unmatched "
            "preprocessing, heads, and parameter counts.*"
        ),
        "",
        "## Reviewer 1myw",
        "",
        "### 1myw-1. Exact, finite-shot, and backend-noisy results",
        "",
        noise_table(t05),
        "",
        (
            "*Mean ± sample SD across three target-model seeds. Each shot/noise target "
            "mean uses ten simulator seeds. Backend-derived Aer noise is not hardware execution.*"
        ),
        "",
        "### 1myw-2. Expanded currently completed attack suite",
        "",
        attack_suite(t03),
        "",
        "#### Completed calibrated and class-label-only baselines",
        "",
        calibrated_and_label_only_attacks(completed_attacks),
        "",
        (
            "*Mean ± sample SD across target models. The label-only method is a "
            "changed-label chord-boundary proxy, not a certified minimum-distance attack.*"
        ),
        "",
        "## Reviewer nVBH",
        "",
        "### nVBH-1. All configurations in the focused MNIST-QNN confirmatory factorial",
        "",
        factorial_cells(t01),
        "",
        (
            "*Every configuration in this focused factorial contains three independently "
            "initialized target models. Entries "
            "are mean ± sample SD across those target seeds. The primary factorial uses "
            "one fixed data split.*"
        ),
        "",
        "### nVBH-2. Paired factorial contrasts with uncertainty",
        "",
        selected_factorial_effects(t02),
        "",
        "*Mean paired difference ± paired-unit SD; 95% hierarchical bootstrap CI.*",
        "",
        "### nVBH-3. Wrapper performance and resource accounting",
        "",
        architecture_resources(t06a),
        "",
        (
            "*Performance entries are mean ± sample SD across three target seeds. "
            "Resource counts are deterministic for a wrapper/role configuration. These "
            "are complete-wrapper comparisons, not matched-capacity causal ablations.*"
        ),
        "",
        "### nVBH-4. Paired wrapper effects",
        "",
        architecture_effects(t06b),
        "",
        "*Mean paired wrapper-minus-QNN difference ± paired-unit SD; 95% bootstrap CI.*",
        "",
        "## Reporting statement",
        "",
        (
            "Use the tables above with the following clarification: configuration-level ± values "
            "are sample SD across independent model initializations, paired-effect ± values "
            "are SD across paired experimental units, and noisy-condition ± values are "
            "across independently trained targets. None of these should be described as "
            "standard errors or confidence intervals."
        ),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"[OK] reviewer-response Markdown -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
