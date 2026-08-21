#!/usr/bin/env python3
"""Plot the ChannelLiRA calibration source/count ablation."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import subprocess

from experiments.plot_channel_lira_phase2 import COLORS, LABELS, SVG, line_chart


COUNTS = [2, 4, 8, 12]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def cross_lookup(rows: list[dict[str, str]], identity: str) -> dict[tuple[int, str], dict[str, str]]:
    return {
        (int(row["auxiliary_target_count"]), row[identity]): row
        for row in rows
        if row["strategy"].startswith("cross_cell_") and int(row["shots"]) == 1024
    }


def points(
    lookup: dict[tuple[int, str], dict[str, str]],
    identity: str,
    value: str,
) -> list[dict[str, object]]:
    return [
        {
            "x": count,
            "value": float(lookup[(count, identity)][f"{value}_median"]),
            "low": float(lookup[(count, identity)][f"{value}_q05"]),
            "high": float(lookup[(count, identity)][f"{value}_q95"]),
        }
        for count in COUNTS
    ]


def auc_plot(out_dir: Path, rows: list[dict[str, str]]) -> None:
    lookup = cross_lookup(rows, "attack")
    series = []
    for attack, color, dash in (
        ("affine_channel_lira", COLORS["affine_channel_lira"], None),
        ("latent_lira_mismatched", COLORS["latent_lira_mismatched"], "7 5"),
        ("loss_mia", COLORS["loss_mia"], None),
    ):
        series.append({
            "label": LABELS[attack], "color": color, "dash": dash,
            "points": points(lookup, attack, "auc"),
        })
    line_chart(
        out_dir / "auc_vs_auxiliary_targets.svg",
        title="Cross-cell calibration-size ablation at 1024 shots",
        subtitle="Five held cells; subset-median within seed, with seed 5th–95th percentile whiskers",
        x_values=COUNTS,
        x_label="Auxiliary targets from other cells",
        y_label="Attack AUC",
        series=series,
        y_limits=(0.565, 0.61),
    )


def contrast_plot(out_dir: Path, rows: list[dict[str, str]]) -> None:
    lookup = cross_lookup(rows, "contrast")
    series = []
    for contrast, label, color in (
        ("affine_minus_mismatched_lira", "ChannelLiRA minus mismatched LiRA", "#dc2626"),
        ("affine_minus_loss", "ChannelLiRA minus loss MIA", "#d97706"),
    ):
        series.append({
            "label": label, "color": color,
            "points": points(lookup, contrast, "auc_difference"),
        })
    line_chart(
        out_dir / "paired_auc_gain_vs_auxiliary_targets.svg",
        title="Paired AUC gain versus calibration size",
        subtitle="Positive differences favor affine ChannelLiRA; intervals are descriptive over simulator seeds",
        x_values=COUNTS,
        x_label="Auxiliary targets from other cells",
        y_label="Paired AUC difference",
        series=series,
        y_limits=(-0.002, 0.036),
        y_formatter=lambda value: f"{value:+.3f}",
        reference_y=(0.0, "no difference"),
    )


def calibration_plot(
    out_dir: Path,
    rows: list[dict[str, str]],
    *,
    value: str,
    filename: str,
    title: str,
    y_label: str,
    y_limits: tuple[float, float],
    reference: bool = False,
) -> None:
    lookup = cross_lookup(rows, "attack")
    series = []
    for attack, color, dash in (
        ("affine_channel_lira", COLORS["affine_channel_lira"], None),
        ("latent_lira_mismatched", COLORS["latent_lira_mismatched"], "7 5"),
        ("loss_mia", COLORS["loss_mia"], None),
    ):
        series.append({
            "label": LABELS[attack], "color": color, "dash": dash,
            "points": points(lookup, attack, value),
        })
    line_chart(
        out_dir / filename,
        title=title,
        subtitle="Thresholds use only the selected auxiliary targets; nominal FPR is 1%",
        x_values=COUNTS,
        x_label="Auxiliary targets from other cells",
        y_label=y_label,
        series=series,
        y_limits=y_limits,
        y_formatter=lambda number: f"{100 * number:.1f}%",
        reference_y=(0.01, "nominal 1%") if reference else None,
    )


def diagnostics_plot(out_dir: Path, rows: list[dict[str, str]]) -> None:
    lookup = {
        int(row["auxiliary_target_count"]): row
        for row in rows
        if row["strategy"].startswith("cross_cell_") and int(row["shots"]) == 1024
    }
    series = [{
        "label": "Held-cell affine-channel R²",
        "color": "#2563eb",
        "points": [
            {
                "x": count,
                "value": float(lookup[count]["test_r_squared_median"]),
                "low": float(lookup[count]["test_r_squared_q05"]),
                "high": float(lookup[count]["test_r_squared_q95"]),
            }
            for count in COUNTS
        ],
    }]
    line_chart(
        out_dir / "heldout_r2_vs_auxiliary_targets.svg",
        title="Held-cell channel fit versus calibration size",
        subtitle="Whiskers combine held cells, folds, and frozen subset sensitivity; they are not confidence intervals",
        x_values=COUNTS,
        x_label="Auxiliary targets from other cells",
        y_label="Held-out R²",
        series=series,
        y_limits=(0.45, 1.0),
    )


def subset_sensitivity_plot(out_dir: Path, rows: list[dict[str, str]]) -> None:
    selected = [
        row for row in rows
        if row["strategy"] == "cross_cell_2" and int(row["shots"]) == 1024
        and row["contrast"] == "affine_minus_mismatched_lira"
    ]
    auc = sorted(float(row["auc_difference"]) for row in selected)
    tpr = sorted(float(row["calibrated_tpr_difference"]) for row in selected)
    width, height = 1180, 620
    svg = SVG(width, height)
    svg.text(85, 34, "Sensitivity to the two selected cross-cell targets", css_class="title")
    svg.text(85, 57, "Each point is one of 66 joint subset configurations, summarized over ten simulator seeds", css_class="subtitle")
    top, panel_height, panel_width = 120, 370, 430
    panels = (
        (auc, "AUC gain over mismatched LiRA", (-0.004, 0.014), "#2563eb"),
        (tpr, "Calibrated-TPR gain at nominal 1% FPR", (-0.012, 0.032), "#7c3aed"),
    )
    for panel_index, (values, title, limits, color) in enumerate(panels):
        left = 90 + panel_index * 555
        svg.text(left + panel_width / 2, 96, title, anchor="middle", size=14, weight=650)

        def xp(rank: int) -> float:
            return left + rank * panel_width / (len(values) - 1)

        def yp(value: float) -> float:
            return top + (limits[1] - value) / (limits[1] - limits[0]) * panel_height

        for tick in range(6):
            value = limits[0] + tick * (limits[1] - limits[0]) / 5
            y = yp(value)
            svg.line(left, y, left + panel_width, y, css_class="grid")
            svg.text(left - 10, y + 4, f"{value:+.3f}", anchor="end", css_class="tick")
        svg.line(left, yp(0.0), left + panel_width, yp(0.0), stroke="#374151", width=1.8)
        coordinates = [(xp(rank), yp(value)) for rank, value in enumerate(values)]
        svg.polyline(coordinates, stroke=color, width=2.3)
        for x, y in coordinates:
            svg.circle(x, y, 2.2, fill=color, width=0.7)
        svg.text(left, top + panel_height + 25, "lowest", anchor="start", css_class="tick")
        svg.text(left + panel_width, top + panel_height + 25, "highest", anchor="end", css_class="tick")
        svg.text(left + panel_width / 2, height - 37, "Ordered two-target subset", anchor="middle", size=12)
    svg.save(out_dir / "two_target_subset_sensitivity.svg")


def write_index(result_dir: Path) -> None:
    content = """# ChannelLiRA calibration-ablation figures

## Attack AUC versus auxiliary-target count

![AUC versus auxiliary targets](plots/auc_vs_auxiliary_targets.svg)

Most of the strict-transfer AUC is already present with two cross-cell targets;
twelve targets add a small paired AUC improvement.

## Paired AUC gains

![Paired gains](plots/paired_auc_gain_vs_auxiliary_targets.svg)

The ChannelLiRA gain over mismatched LiRA stays positive after subset marginalization
at every tested calibration count.

## Operational TPR

![Calibrated TPR](plots/calibrated_tpr_vs_auxiliary_targets.svg)

Two-target calibration gives higher TPR but also operates at a higher FPR than the
twelve-target condition, so it is not a free accuracy improvement.

## Realized FPR

![Realized FPR](plots/actual_fpr_vs_auxiliary_targets.svg)

Threshold calibration becomes more conservative and approaches the nominal 1% FPR
as more auxiliary targets are included.

## Held-cell channel fit

![Held-cell R squared](plots/heldout_r2_vs_auxiliary_targets.svg)

Median held-cell R² changes only modestly with calibration count; variation across
cells, folds, and target subsets remains wide.

## Two-target subset sensitivity

![Two-target subset sensitivity](plots/two_target_subset_sensitivity.svg)

The 5th–95th percentile AUC-gain range across the 66 two-target configurations is
positive, although the weakest configurations can be negative. Operational TPR is
more sensitive to which auxiliary targets set the threshold.
"""
    (result_dir / "PLOTS.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir", type=Path,
        default=Path("channel_lira_results/calibration_ablation_phase4"),
    )
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    plot_dir = result_dir / "plots"
    metrics = read_rows(result_dir / "metrics_summary.csv")
    calibration = read_rows(result_dir / "calibration_summary.csv")
    contrasts = read_rows(result_dir / "paired_contrasts_summary.csv")
    diagnostics = read_rows(result_dir / "channel_diagnostics_summary.csv")
    subset_medians = read_rows(result_dir / "paired_contrasts_subset_medians.csv")
    auc_plot(plot_dir, metrics)
    contrast_plot(plot_dir, contrasts)
    calibration_plot(
        plot_dir, calibration,
        value="calibrated_tpr",
        filename="calibrated_tpr_vs_auxiliary_targets.svg",
        title="Operational TPR versus calibration size",
        y_label="Calibrated true-positive rate",
        y_limits=(0.004, 0.034),
    )
    calibration_plot(
        plot_dir, calibration,
        value="actual_fpr",
        filename="actual_fpr_vs_auxiliary_targets.svg",
        title="Realized FPR versus calibration size",
        y_label="Actual false-positive rate",
        y_limits=(0.006, 0.032),
        reference=True,
    )
    diagnostics_plot(plot_dir, diagnostics)
    subset_sensitivity_plot(plot_dir, subset_medians)
    write_index(result_dir)
    if args.png:
        convert = shutil.which("convert")
        if convert is None:
            raise FileNotFoundError("--png requires ImageMagick's convert executable")
        for svg_path in sorted(plot_dir.glob("*.svg")):
            subprocess.run(
                [
                    convert, "-background", "white", "-density", "144",
                    str(svg_path), str(svg_path.with_suffix(".png")),
                ],
                check=True,
            )
    suffix = " plus PNG previews" if args.png else ""
    print(f"wrote six SVG figures{suffix} and {result_dir / 'PLOTS.md'}")


if __name__ == "__main__":
    main()
