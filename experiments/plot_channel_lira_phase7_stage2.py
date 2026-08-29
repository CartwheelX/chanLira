#!/usr/bin/env python3
"""Plot the sealed Phase-7 Stage-2 confirmatory primary results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ATTACKS = (
    ("matched_reference_lira_online_fixed_variance", 16, "Matched noisy LiRA"),
    ("affine_channel_lira", 16, "ChannelLiRA"),
    ("latent_lira_mismatched", 16, "Mismatched LiRA"),
    ("loss_mia", 0, "Loss MIA"),
    ("target_crossfit_learned_mia", 0, "Privileged learned"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(figure: plt.Figure, base: Path, png: bool) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    if png:
        figure.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def attack_plot(summary: list[dict[str, str]], out_dir: Path, png: bool) -> None:
    indexed = {
        (row["attack"], int(row["reference_count"])): row for row in summary
    }
    labels = [label for _, _, label in ATTACKS]
    auc = [float(indexed[(attack, count)]["auc_mean"]) for attack, count, _ in ATTACKS]
    tpr = [
        100 * float(indexed[(attack, count)]["tpr_at_1pct_fpr_mean"])
        for attack, count, _ in ATTACKS
    ]
    colors = ["#4477AA", "#228833", "#CCBB44", "#EE6677", "#AA3377"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    x = np.arange(len(labels))
    axes[0].bar(x, auc, color=colors)
    axes[0].axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Mean target AUC")
    axes[0].set_ylim(min(0.45, min(auc) - 0.02), max(0.55, max(auc) + 0.02))
    axes[1].bar(x, tpr, color=colors)
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Mean target TPR@1% FPR (%)")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Phase 7 confirmatory primary: attack metrics")
    figure.text(
        0.5, 0.005,
        "Four cells × three targets; ten simulator seeds averaged within each target",
        ha="center", fontsize=9,
    )
    save_figure(figure, out_dir / "primary_attack_metrics", png)


def interval_row(
    rows: list[dict[str, str]], contrast: str, metric: str
) -> dict[str, str]:
    matches = [
        row for row in rows
        if row["contrast"] == contrast and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Missing hierarchical interval for {(contrast, metric)}")
    return matches[0]


def gate_plot(intervals: list[dict[str, str]], out_dir: Path, png: bool) -> None:
    definitions = (
        ("A: matched − mismatched", "matched_reference_minus_mismatched", 0.0),
        ("B: Channel − matched", "affine_minus_matched_reference", -0.5),
        ("C: Channel − loss", "affine_minus_loss", 0.0),
        ("Stress: Channel − learned", "affine_minus_learned", None),
    )
    rows = [interval_row(intervals, contrast, "tpr_at_1pct_fpr") for _, contrast, _ in definitions]
    point = 100 * np.asarray([float(row["point_estimate"]) for row in rows])
    low = 100 * np.asarray([float(row["ci95_lower"]) for row in rows])
    high = 100 * np.asarray([float(row["ci95_upper"]) for row in rows])
    y = np.arange(len(definitions))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
    axes[0].errorbar(
        point, y, xerr=np.vstack((point - low, high - point)),
        fmt="o", color="#225588", ecolor="#4477AA", capsize=4,
    )
    axes[0].axvline(0.0, color="#555555", linestyle="--", linewidth=1)
    axes[0].plot([-0.5], [1], marker="|", markersize=18, color="#CC3311")
    axes[0].set_yticks(y, [label for label, _, _ in definitions])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Paired TPR@1% FPR difference (percentage points)")
    axes[0].grid(axis="x", alpha=0.2)

    b_auc = interval_row(intervals, "affine_minus_matched_reference", "auc")
    auc_point = float(b_auc["point_estimate"])
    auc_low = float(b_auc["ci95_lower"])
    auc_high = float(b_auc["ci95_upper"])
    axes[1].errorbar(
        [auc_point], [0], xerr=[[auc_point - auc_low], [auc_high - auc_point]],
        fmt="o", color="#228833", ecolor="#44AA66", capsize=4,
    )
    axes[1].axvline(-0.01, color="#CC3311", linestyle="--", linewidth=1)
    axes[1].set_yticks([0], ["B: Channel − matched"])
    axes[1].set_xlabel("AUC difference\n(non-inferiority boundary −0.01)")
    axes[1].grid(axis="x", alpha=0.2)
    figure.suptitle("Frozen Phase 7 gates: hierarchical 95% intervals")
    figure.text(
        0.5, 0.01,
        "Cells and targets-within-cells resampled; 20,000 frozen bootstrap replicates",
        ha="center", fontsize=9,
    )
    save_figure(figure, out_dir / "primary_gate_intervals", png)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir", type=Path,
        default=Path("channel_lira_results/phase7/stage2_primary/analysis"),
    )
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    config = json.loads((analysis_dir / "CONFIG.json").read_text(encoding="utf-8"))
    if config.get("analysis_role") != "confirmatory_primary" or config.get("pilot_included") is not False:
        raise ValueError("Plot input is not the sealed confirmatory-primary analysis")
    summary = read_csv(analysis_dir / "metrics_summary.csv")
    intervals = read_csv(analysis_dir / "hierarchical_intervals.csv")
    plot_dir = analysis_dir / "plots"
    attack_plot(summary, plot_dir, args.png)
    gate_plot(intervals, plot_dir, args.png)
    png_lines = (
        "\n- [PNG attack metrics](plots/primary_attack_metrics.png)"
        "\n- [PNG gate intervals](plots/primary_gate_intervals.png)"
        if args.png else ""
    )
    report = f"""# Phase 7 Stage 2 plot index

## Confirmatory attack metrics

![Primary attack metrics](plots/primary_attack_metrics.svg)

Bars are target-level means after averaging ten simulator seeds within each target.
They are descriptive; frozen decisions use the hierarchical paired intervals.

## Frozen gate intervals

![Primary gate intervals](plots/primary_gate_intervals.svg)

The red marks are the frozen decision boundaries: zero for A/C, −0.5 percentage
points for B's TPR non-inferiority component, and −0.01 for B's AUC component.
The learned comparison is a privileged stress comparator and has no success gate.
{png_lines}
"""
    (analysis_dir / "PLOTS.md").write_text(report, encoding="utf-8")
    print(f"[PLOTS] {analysis_dir / 'PLOTS.md'}")


if __name__ == "__main__":
    main()
