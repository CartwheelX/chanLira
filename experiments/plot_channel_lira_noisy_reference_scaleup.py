#!/usr/bin/env python3
"""Plot the Phase-6 16-reference noisy scale-up diagnostics."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_ATTACKS = (
    "matched_reference_lira_online_fixed_variance",
    "affine_channel_lira",
    "latent_lira_mismatched",
    "loss_mia",
    "target_crossfit_learned_mia",
)
LABELS = {
    "matched_reference_lira_online_fixed_variance": "Matched noisy LiRA",
    "affine_channel_lira": "ChannelLiRA (LOTO)",
    "latent_lira_mismatched": "Mismatched LiRA",
    "loss_mia": "Loss MIA",
    "target_crossfit_learned_mia": "Learned MIA*",
}
COLORS = {
    "matched_reference_lira_online_fixed_variance": "#1b9e77",
    "affine_channel_lira": "#d95f02",
    "latent_lira_mismatched": "#7570b3",
    "loss_mia": "#666666",
    "target_crossfit_learned_mia": "#e7298a",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, path: Path, png: bool) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = path.with_suffix(".svg")
    fig.savefig(svg, bbox_inches="tight")
    outputs = [svg]
    if png:
        raster = path.with_suffix(".png")
        fig.savefig(raster, dpi=180, bbox_inches="tight")
        outputs.append(raster)
    plt.close(fig)
    return outputs


def attack_summary_plot(
    summary: list[dict[str, str]], out_dir: Path, png: bool
) -> list[Path]:
    modes = ("ideal_shot", "noisy_shot")
    x = np.arange(len(PRIMARY_ATTACKS), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    for mode_index, mode in enumerate(modes):
        means = []
        errors = []
        for attack in PRIMARY_ATTACKS:
            row = next(
                item for item in summary
                if item["mode"] == mode and item["attack"] == attack
            )
            means.append(float(row["auc_mean"]))
            errors.append(float(row["auc_sd_targets"]))
        offset = (mode_index - 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=errors,
            capsize=3,
            color="#80b1d3" if mode == "ideal_shot" else "#fb8072",
            edgecolor="black",
            linewidth=0.5,
            label=mode.replace("_", " "),
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x, [LABELS[value] for value in PRIMARY_ATTACKS], rotation=20, ha="right")
    ax.set_ylabel("Mean target AUC")
    ax.set_title("Phase 6 attack comparison (mean ± SD across 3 targets)")
    ax.set_ylim(0.35, 0.75)
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    return save_figure(fig, out_dir / "attack_auc_by_mode", png)


def noisy_target_plot(
    target_rows: list[dict[str, str]], out_dir: Path, png: bool
) -> list[Path]:
    targets = sorted({row["target_id"] for row in target_rows})
    x = np.arange(len(targets))
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for attack in PRIMARY_ATTACKS:
        values = []
        for target in targets:
            row = next(
                item for item in target_rows
                if item["mode"] == "noisy_shot"
                and item["target_id"] == target
                and item["attack"] == attack
            )
            values.append(float(row["auc_mean_over_simulator_seeds"]))
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=1.7,
            color=COLORS[attack],
            label=LABELS[attack],
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, [target.rsplit("_s", 1)[-1] for target in targets])
    ax.set_xlabel("Target model seed")
    ax.set_ylabel("AUC averaged over simulator seeds")
    ax.set_title("Noisy-shot target heterogeneity")
    ax.set_ylim(0.35, 0.75)
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    return save_figure(fig, out_dir / "noisy_target_auc", png)


def contrast_plot(
    rows: list[dict[str, str]], out_dir: Path, png: bool
) -> list[Path]:
    selected = [
        row for row in rows
        if row["mode"] == "noisy_shot" and row["contrast"] in {
            "affine_minus_matched_reference",
            "matched_reference_minus_mismatched",
            "matched_reference_minus_loss",
            "matched_reference_minus_learned",
        }
    ]
    labels = [row["contrast"].replace("_", "\n") for row in selected]
    means = np.asarray([float(row["auc_difference_mean"]) for row in selected])
    lows = np.asarray([float(row["auc_difference_min"]) for row in selected])
    highs = np.asarray([float(row["auc_difference_max"]) for row in selected])
    errors = np.vstack((means - lows, highs - means))
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    positions = np.arange(len(selected))
    ax.bar(
        positions,
        means,
        yerr=errors,
        capsize=4,
        color="#8dd3c7",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Paired target AUC difference")
    ax.set_title("Noisy-shot paired contrasts (range across 3 targets)")
    ax.grid(axis="y", alpha=0.2)
    return save_figure(fig, out_dir / "noisy_paired_contrasts", png)


def mode_delta_plot(
    target_rows: list[dict[str, str]], out_dir: Path, png: bool
) -> list[Path]:
    indexed = {
        (row["target_id"], row["mode"], row["attack"]): float(
            row["auc_mean_over_simulator_seeds"]
        )
        for row in target_rows
    }
    targets = sorted({row["target_id"] for row in target_rows})
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    x = np.arange(len(PRIMARY_ATTACKS))
    for target_index, target in enumerate(targets):
        deltas = [
            indexed[(target, "noisy_shot", attack)]
            - indexed[(target, "ideal_shot", attack)]
            for attack in PRIMARY_ATTACKS
        ]
        ax.plot(
            x,
            deltas,
            marker="o",
            linewidth=1.4,
            label=f"seed {target.rsplit('_s', 1)[-1]}",
            color=("#1f78b4", "#33a02c", "#e31a1c")[target_index],
        )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x, [LABELS[value] for value in PRIMARY_ATTACKS], rotation=20, ha="right")
    ax.set_ylabel("Noisy-shot AUC − ideal-shot AUC")
    ax.set_title("Effect of the frozen IBM-derived serving channel")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    return save_figure(fig, out_dir / "noisy_minus_ideal_auc", png)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir", type=Path,
        default=ROOT / "channel_lira_results/noisy_reference_scaleup_phase6/analysis",
    )
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    summary = read_csv(analysis_dir / "metrics_summary.csv")
    target_rows = read_csv(analysis_dir / "target_mean_metrics.csv")
    contrasts = read_csv(analysis_dir / "paired_contrasts_summary.csv")
    plot_dir = analysis_dir / "plots"
    outputs = []
    outputs.extend(attack_summary_plot(summary, plot_dir, args.png))
    outputs.extend(noisy_target_plot(target_rows, plot_dir, args.png))
    outputs.extend(contrast_plot(contrasts, plot_dir, args.png))
    outputs.extend(mode_delta_plot(target_rows, plot_dir, args.png))
    lines = [
        "# Phase-6 plot index",
        "",
        "Error bars in the attack summary are sample SD across three target checkpoints after averaging simulator seeds. Contrast bars show the observed target range; neither is a confidence interval.",
        "",
    ]
    for output in outputs:
        lines.append(f"- [`{output.name}`](plots/{output.name})")
    (analysis_dir / "PLOTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[PLOTS] {analysis_dir / 'PLOTS.md'}")


if __name__ == "__main__":
    main()
