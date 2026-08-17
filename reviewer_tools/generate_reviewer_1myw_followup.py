#!/usr/bin/env python3
"""Generate a paste-ready response to Reviewer 1myw's geometry/noise follow-up."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


CONDITIONS = [
    ("exact", 0, "Exact"),
    ("ideal_shot", 128, "Ideal shot, 128"),
    ("ideal_shot", 512, "Ideal shot, 512"),
    ("ideal_shot", 1024, "Ideal shot, 1024"),
    ("noisy_shot", 128, "Backend-noisy, 128"),
    ("noisy_shot", 512, "Backend-noisy, 512"),
    ("noisy_shot", 1024, "Backend-noisy, 1024"),
]

CELL_LABELS = {
    "eff_su2_r1_d2": "EffSU2, reps=1, depth=2",
    "eff_su2_r5_d2": "EffSU2, reps=5, depth=2",
    "z_r1_d6": "Z, reps=1, depth=6",
    "zz_r1_d6": "ZZ, reps=1, depth=6",
    "zz_r5_d6": "ZZ, reps=5, depth=6",
}

EXPECTED_ORDER = list(CELL_LABELS)
LOW_CELL = EXPECTED_ORDER[0]
HIGH_CELL = EXPECTED_ORDER[-1]


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def mean_sd(values: pd.Series, digits: int = 3, signed: bool = False) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    prefix = "+" if signed and float(values.mean()) >= 0 else ""
    return f"{prefix}{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"


def condition_subset(frame: pd.DataFrame, mode: str, shots: int) -> pd.DataFrame:
    return frame[frame["mode"].eq(mode) & frame["shots"].eq(shots)]


def five_cell_auc_table(target_means: pd.DataFrame) -> str:
    loss = target_means[target_means["attack"].eq("loss")]
    rows: list[list[object]] = []
    displayed = [("exact", 0, "Exact")] + [
        item for item in CONDITIONS if item[0] == "noisy_shot"
    ]
    for cell in EXPECTED_ORDER:
        row: list[object] = [CELL_LABELS[cell]]
        for mode, shots, _ in displayed:
            values = condition_subset(loss, mode, shots)
            values = values[values["structural_cell_id"].eq(cell)]["roc_auc"]
            if len(values) != 3:
                raise ValueError(f"Expected three target seeds for {cell}/{mode}/{shots}")
            row.append(mean_sd(values))
        rows.append(row)
    return md_table(
        ["Structural configuration", "Exact", "Noisy 128", "Noisy 512", "Noisy 1024"],
        rows,
    )


def hierarchy_table(long: pd.DataFrame, target_means: pd.DataFrame) -> str:
    long = long[long["attack"].eq("loss")]
    target_means = target_means[target_means["attack"].eq("loss")]
    exact_cells = condition_subset(long, "exact", 0).groupby("structural_cell_id")["roc_auc"].mean()
    rows: list[list[object]] = []
    for mode, shots, label in CONDITIONS:
        targets = condition_subset(target_means, mode, shots)
        pivot = targets.pivot(index="model_seed", columns="structural_cell_id", values="roc_auc")
        paired_difference = pivot[HIGH_CELL] - pivot[LOW_CELL]

        cell_means = targets.groupby("structural_cell_id")["roc_auc"].mean()
        observed_order = list(cell_means.sort_values().index)
        order_text = "Yes (5/5)" if observed_order == EXPECTED_ORDER else "No"

        if mode == "exact":
            rank_text = "1.000 (reference)"
            simulator_units = "—"
        else:
            ranks = []
            for _, replicate in condition_subset(long, mode, shots).groupby("simulator_seed"):
                cell_auc = replicate.groupby("structural_cell_id")["roc_auc"].mean()
                aligned = cell_auc.reindex(exact_cells.index)
                ranks.append(float(spearmanr(exact_cells, aligned).statistic))
            rank_text = mean_sd(pd.Series(ranks))
            simulator_units = str(len(ranks))

        rows.append(
            [
                label,
                mean_sd(pivot[LOW_CELL]),
                mean_sd(pivot[HIGH_CELL]),
                mean_sd(paired_difference, signed=True),
                order_text,
                rank_text,
                simulator_units,
            ]
        )
    return md_table(
        [
            "Condition",
            "Low-risk AUC",
            "High-risk AUC",
            "Paired high − low",
            "Aggregate order retained",
            "Spearman ρ ± SD",
            "Simulator seeds",
        ],
        rows,
    )


def geometry_table(effects: pd.DataFrame) -> str:
    definitions = {
        "class_similarity_gap": (
            "Class-similarity gap",
            "Mean within-class minus between-class fidelity",
        ),
        "kernel_label_alignment": (
            "Kernel–label alignment",
            "Centered fidelity-kernel/label-kernel alignment",
        ),
        "effective_rank": (
            "Effective rank",
            "Spectral effective rank of the fidelity kernel",
        ),
        "mmd2_train_test": (
            "Train/test MMD²",
            "Kernel two-sample discrepancy",
        ),
    }
    rows: list[list[object]] = []
    for metric, (label, definition) in definitions.items():
        row = effects[effects["Metric"].eq(metric)].iloc[0]
        estimate = (
            f"{float(row['Mean difference']):+.3f} ± "
            f"{float(row['Unique paired-effect SD']):.3f}"
        )
        rows.append(
            [
                label,
                definition,
                estimate,
                row["95% CI"],
                int(row["Unique paired seed effects"]),
            ]
        )
    return md_table(
        ["Post-encoder quantity", "Operationalization", "Reps 5 − 1, mean ± SD", "95% CI", "Paired effects"],
        rows,
    )


def evidence_scope_table(
    factorial: pd.DataFrame,
    correlations: pd.DataFrame,
    fits: pd.DataFrame,
) -> str:
    rep_gap = factorial[
        factorial["Factor"].eq("Repetitions")
        & factorial["Contrast"].eq("5 − 1")
        & factorial["Metric"].eq("Accuracy gap")
    ].iloc[0]
    rep_auc = factorial[
        factorial["Factor"].eq("Repetitions")
        & factorial["Contrast"].eq("5 − 1")
        & factorial["Metric"].eq("Loss AUC")
    ].iloc[0]
    spearman = correlations[correlations["Method"].eq("Spearman")].iloc[0]
    fit = fits[fits["Model"].eq("M1_gap")].iloc[0]
    return md_table(
        ["Link in proposed pathway", "Evidence", "Estimate", "What may be claimed"],
        [
            [
                "Encoder repetition → geometry",
                "Direct post-encoder fidelity-kernel measurements",
                "See geometry table",
                "Supported for the evaluated encoders/data",
            ],
            [
                "Repetition → generalization gap",
                "Paired factorial contrast",
                f"{rep_gap['Mean difference']:+.3f} ± {rep_gap['Paired-unit SD']:.3f}; CI {rep_gap['95% CI']}",
                "Supported association",
            ],
            [
                "Repetition → loss-MIA AUC",
                "Paired factorial contrast",
                f"{rep_auc['Mean difference']:+.3f} ± {rep_auc['Paired-unit SD']:.3f}; CI {rep_auc['95% CI']}",
                "Supported association",
            ],
            [
                "Gap ↔ loss-MIA AUC",
                "Hierarchical-bootstrap Spearman correlation",
                f"ρ={spearman['Correlation']:.3f}; CI {spearman['95% CI']}",
                "Strong descriptive association",
            ],
            [
                "Gap + structure → loss-MIA AUC",
                "Descriptive regression",
                f"R²={fit['R²']:.3f}; CI {fit['R² 95% CI']}",
                "Descriptive, not causal mediation",
            ],
            [
                "Geometry → gap causally",
                "No intervention on geometry independent of encoder",
                "Not identified",
                "Do not claim a causal mechanism/theorem",
            ],
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noisy-long",
        type=Path,
        default=Path("reviewer_results/noisy_sanity/combined/noisy_mia_results_long.csv"),
    )
    parser.add_argument(
        "--noisy-target-means",
        type=Path,
        default=Path("reviewer_results/noisy_sanity/combined/noisy_mia_target_condition_means.csv"),
    )
    parser.add_argument(
        "--geometry-effects",
        type=Path,
        default=Path(
            "reviewer_results/reviewer_artifacts/tables/"
            "T04b_geometry_repetition_effects.csv"
        ),
    )
    parser.add_argument(
        "--artifact-tables",
        type=Path,
        default=Path("reviewer_results/reviewer_artifacts/tables"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reviewer_results/reviewer_artifacts/REVIEWER_1MYW_FOLLOWUP.md"),
    )
    args = parser.parse_args()

    noisy_long = pd.read_csv(args.noisy_long)
    target_means = pd.read_csv(args.noisy_target_means)
    geometry = pd.read_csv(args.geometry_effects)
    factorial = pd.read_csv(args.artifact_tables / "T02_factorial_paired_effects.csv")
    correlations = pd.read_csv(args.artifact_tables / "T07c_gap_auc_correlations.csv")
    fits = pd.read_csv(args.artifact_tables / "T07b_gap_auc_regression_fit.csv")

    sections = [
        "# Response to Reviewer 1myw: geometry and noise",
        "",
        "## Suggested response",
        "",
        (
            "Thank you for identifying these two gaps. We agree that the original manuscript "
            "did not directly measure the Appendix-D geometry and that a noiseless hierarchy "
            "alone cannot establish robustness. We therefore added two targeted analyses."
        ),
        "",
        (
            "**Noise and finite shots.** We evaluated five prespecified configurations spanning "
            "the observed risk range, using three independently trained target checkpoints per "
            "configuration and ten simulator seeds at 128, 512, and 1,024 shots. We compared "
            "ideal-shot simulation with an Aer noise model derived from the `ibm_kingston` "
            "calibration snapshot (gate, readout, and thermal-relaxation errors). This comprises "
            "15 targets and 915 target/execution replicates. The five configuration means retain "
            "the exact loss-AUC ordering in every aggregate condition. Noise nevertheless has a "
            "non-uniform mitigating effect: the paired high-minus-low AUC difference decreases "
            "from 0.179 ± 0.030 exactly to 0.096 ± 0.012 at 128 noisy shots, then increases to "
            "0.124 ± 0.026 and 0.132 ± 0.036 at 512 and 1,024 shots. Individual finite-shot "
            "replicates can locally reorder nearby middle configurations, especially at 128 "
            "shots; the mean rank correlation with the exact five-configuration hierarchy is "
            "0.820 ± 0.155 under 128-shot backend noise and 0.960 ± 0.052 at 1,024 shots. Thus, "
            "noise attenuates the severity but does not reverse or eliminate the broad hierarchy "
            "in this check. We will clearly label this as backend-derived simulation rather than "
            "hardware execution and will not claim universality across devices/calibrations."
        ),
        "",
        (
            "**Direct geometry.** We now compute the pure-state Hilbert–Schmidt/fidelity kernel "
            "immediately after the fixed encoder and before the variational circuit. Across "
            "dataset/encoder blocks, increasing repetitions from 1 to 5 significantly changes "
            "class-conditioned similarity, kernel–label alignment, and effective rank; the "
            "train/test MMD² interval includes zero. These results directly verify the first link "
            "of the proposed pathway—changing the encoder changes its induced geometry."
        ),
        "",
        (
            "We also agree with the reviewer's causal distinction. Our intended hypothesis is "
            "not that geometry bypasses overfitting: overfitting is the proposed mediator through "
            "which encoder structure becomes visible to an MIA. The new measurements support an "
            "encoder → geometry association, while the factorial results support structure → gap "
            "and structure → MIA associations, and gap strongly tracks loss-AUC. They do not "
            "causally identify geometry → gap independently of encoder choice. We will therefore "
            "replace causal wording such as ‘geometric mechanism’ or ‘geometry creates separation’ "
            "with ‘empirically supported geometric pathway/association,’ explicitly state that "
            "mediation is not causally identified, and add the following tables and the corresponding "
            "geometry/noise figures to the revision."
        ),
        "",
        "## Table 1. Loss-MIA AUC across the five-configuration hierarchy",
        "",
        five_cell_auc_table(target_means),
        "",
        (
            "*Entries are mean ± sample SD across three independently trained target-model "
            "seeds after averaging ten simulator seeds for shot-based conditions.*"
        ),
        "",
        "## Table 2. Hierarchy robustness under finite shots and backend-derived noise",
        "",
        hierarchy_table(noisy_long, target_means),
        "",
        (
            "*Low/high AUCs and paired differences are mean ± sample SD across three paired "
            "target seeds. Spearman ρ is mean ± sample SD across ten simulator seeds; within "
            "each simulator seed, AUC is first averaged across the three target seeds for each "
            "of the five structural configurations. ‘Aggregate order’ uses means across target and "
            "simulator seeds. Local simulator-level reordering is therefore not hidden.*"
        ),
        "",
        "## Table 3. Direct post-encoder geometry",
        "",
        geometry_table(geometry),
        "",
        (
            "*Effects are paired reps=5 minus reps=1 contrasts. SD is across unique paired "
            "effects and the intervals are 5,000-replicate hierarchical percentile-bootstrap "
            "CIs. MNIST nominal data seeds produce identical encoded states for its fixed "
            "subset; Moons supplies genuine data-seed variation, and duplicate MNIST states "
            "are not counted as independent effects.*"
        ),
        "",
        "## Table 4. Evidence and revised claim boundary",
        "",
        evidence_scope_table(factorial, correlations, fits),
        "",
        "## Revision commitments",
        "",
        "- Add the direct post-encoder kernel measurements and paired uncertainty analysis.",
        "- Add the five-configuration finite-shot/backend-noise results and hierarchy-rank analysis.",
        "- Describe the noise experiment as IBM-backend-derived Aer simulation, not hardware.",
        "- Report attenuation and local finite-shot reorderings, not only preserved aggregate order.",
        "- Replace causal-mechanism wording with an empirically supported pathway and state that causal mediation remains unverified.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"[OK] Reviewer 1myw follow-up -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
