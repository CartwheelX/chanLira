#!/usr/bin/env python3
"""Generate SVG and optional PNG figures for the strict transfer study."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import subprocess

from experiments.plot_channel_lira_phase2 import COLORS, LABELS, SVG, line_chart


SHOTS = [128, 512, 1024]
SCHEMES = ("leave_target_out", "leave_cell_out")
SCHEME_LABELS = {
    "leave_target_out": "Leave target out",
    "leave_cell_out": "Leave cell out",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def attack_series(
    rows: list[dict[str, str]],
    *,
    scheme: str,
    value: str,
    low: str,
    high: str,
) -> list[dict[str, object]]:
    indexed = {
        (row["scheme"], int(row["shots"]), row["attack"], int(row["reference_count"])): row
        for row in rows
        if row["scope"] == "overall"
    }
    attacks = (
        ("affine_channel_lira", 16),
        ("latent_lira_mismatched", 16),
        ("loss_mia", 0),
        ("noise_augmented_lira", 16),
    )
    output = []
    for attack, count in attacks:
        points = []
        for shots in SHOTS:
            row = indexed[(scheme, shots, attack, count)]
            points.append({
                "x": shots,
                "value": float(row[value]),
                "low": float(row[low]),
                "high": float(row[high]),
            })
        output.append({
            "label": LABELS[attack],
            "color": COLORS[attack],
            "points": points,
            "dash": "7 5" if attack == "latent_lira_mismatched" else None,
        })
    return output


def auc_plots(out_dir: Path, metrics: list[dict[str, str]]) -> None:
    for scheme in SCHEMES:
        line_chart(
            out_dir / f"attack_auc_{scheme}.svg",
            title=f"Attack AUC with {SCHEME_LABELS[scheme].lower()} channel fitting",
            subtitle="Held target records pooled across five cells; whiskers are seed 5th–95th percentiles",
            x_values=SHOTS,
            x_label="Shots",
            y_label="Attack AUC",
            series=attack_series(
                metrics, scheme=scheme, value="auc_median", low="auc_q05", high="auc_q95"
            ),
            y_limits=(0.54, 0.63),
        )


def paired_auc_plot(out_dir: Path, rows: list[dict[str, str]]) -> None:
    indexed = {
        (row["scheme"], int(row["shots"]), row["contrast"]): row
        for row in rows
    }
    specifications = (
        ("leave_target_out", "affine_minus_loss", "Target holdout: vs loss MIA", "#d97706", None),
        ("leave_target_out", "affine_minus_mismatched_lira", "Target holdout: vs mismatched LiRA", "#dc2626", None),
        ("leave_cell_out", "affine_minus_loss", "Cell holdout: vs loss MIA", "#b45309", "7 5"),
        ("leave_cell_out", "affine_minus_mismatched_lira", "Cell holdout: vs mismatched LiRA", "#991b1b", "7 5"),
    )
    series = []
    for scheme, contrast, label, color, dash in specifications:
        points = []
        for shots in SHOTS:
            row = indexed[(scheme, shots, contrast)]
            points.append({
                "x": shots,
                "value": float(row["auc_difference_median"]),
                "low": float(row["auc_difference_q05"]),
                "high": float(row["auc_difference_q95"]),
            })
        series.append({"label": label, "color": color, "dash": dash, "points": points})
    line_chart(
        out_dir / "paired_auc_transfer.svg",
        title="Paired AUC gain under strict transfer",
        subtitle="Within-simulator-seed differences; positive values favor affine ChannelLiRA",
        x_values=SHOTS,
        x_label="Shots",
        y_label="Paired AUC difference",
        series=series,
        y_limits=(-0.012, 0.036),
        y_formatter=lambda value: f"{value:+.3f}",
        reference_y=(0.0, "no difference"),
    )


def calibrated_tpr_plot(out_dir: Path, rows: list[dict[str, str]]) -> None:
    indexed = {
        (row["scheme"], int(row["shots"]), row["attack"], int(row["reference_count"])): row
        for row in rows
        if row["scope"] == "overall" and float(row["nominal_fpr"]) == 0.01
    }
    specifications = (
        ("leave_target_out", "affine_channel_lira", 16, "Target holdout: ChannelLiRA", "#047857", None),
        ("leave_target_out", "latent_lira_mismatched", 16, "Target holdout: mismatched LiRA", "#dc2626", None),
        ("leave_target_out", "loss_mia", 0, "Target holdout: loss MIA", "#d97706", None),
        ("leave_cell_out", "affine_channel_lira", 16, "Cell holdout: ChannelLiRA", "#065f46", "7 5"),
        ("leave_cell_out", "latent_lira_mismatched", 16, "Cell holdout: mismatched LiRA", "#991b1b", "7 5"),
        ("leave_cell_out", "loss_mia", 0, "Cell holdout: loss MIA", "#92400e", "7 5"),
    )
    series = []
    for scheme, attack, count, label, color, dash in specifications:
        points = []
        for shots in SHOTS:
            row = indexed[(scheme, shots, attack, count)]
            points.append({
                "x": shots,
                "value": float(row["calibrated_tpr_median"]),
                "low": float(row["calibrated_tpr_q05"]),
                "high": float(row["calibrated_tpr_q95"]),
            })
        series.append({"label": label, "color": color, "dash": dash, "points": points})
    line_chart(
        out_dir / "calibrated_tpr_at_1pct.svg",
        title="Operational TPR at a nominal 1% FPR",
        subtitle="Thresholds learned only from auxiliary targets/cells; dashed lines are leave-cell-out",
        x_values=SHOTS,
        x_label="Shots",
        y_label="Calibrated true-positive rate",
        series=series,
        y_limits=(0.004, 0.032),
        y_formatter=lambda value: f"{100 * value:.1f}%",
    )


def actual_fpr_plot(out_dir: Path, rows: list[dict[str, str]]) -> None:
    indexed = {
        (row["scheme"], int(row["shots"]), row["attack"], int(row["reference_count"])): row
        for row in rows
        if row["scope"] == "overall" and float(row["nominal_fpr"]) == 0.01
    }
    specifications = (
        ("leave_target_out", "affine_channel_lira", 16, "Target holdout: ChannelLiRA", "#047857", None),
        ("leave_target_out", "latent_lira_mismatched", 16, "Target holdout: mismatched LiRA", "#dc2626", None),
        ("leave_cell_out", "affine_channel_lira", 16, "Cell holdout: ChannelLiRA", "#065f46", "7 5"),
        ("leave_cell_out", "latent_lira_mismatched", 16, "Cell holdout: mismatched LiRA", "#991b1b", "7 5"),
    )
    series = []
    for scheme, attack, count, label, color, dash in specifications:
        points = []
        for shots in SHOTS:
            row = indexed[(scheme, shots, attack, count)]
            points.append({
                "x": shots,
                "value": float(row["actual_fpr_median"]),
                "low": float(row["actual_fpr_q05"]),
                "high": float(row["actual_fpr_q95"]),
            })
        series.append({"label": label, "color": color, "dash": dash, "points": points})
    line_chart(
        out_dir / "actual_fpr_at_1pct.svg",
        title="Realized FPR under transferred thresholds",
        subtitle="Held nonmembers only; whiskers are seed 5th–95th percentiles",
        x_values=SHOTS,
        x_label="Shots",
        y_label="Actual false-positive rate",
        series=series,
        y_limits=(0.006, 0.022),
        y_formatter=lambda value: f"{100 * value:.1f}%",
        reference_y=(0.01, "nominal 1%"),
    )


def diagnostics_plot(out_dir: Path, rows: list[dict[str, str]]) -> None:
    indexed = {(row["scheme"], int(row["shots"])): row for row in rows}
    series = []
    for scheme, color, dash in (
        ("leave_target_out", "#2563eb", None),
        ("leave_cell_out", "#7c3aed", "7 5"),
    ):
        points = []
        for shots in SHOTS:
            row = indexed[(scheme, shots)]
            points.append({
                "x": shots,
                "value": float(row["test_r_squared_median"]),
                "low": float(row["test_r_squared_q05"]),
                "high": float(row["test_r_squared_q95"]),
            })
        series.append({
            "label": SCHEME_LABELS[scheme], "color": color, "dash": dash, "points": points
        })
    line_chart(
        out_dir / "heldout_channel_r2.svg",
        title="Affine channel fit on unseen targets and structures",
        subtitle="R² evaluated on held public nonmembers; exact held outputs are diagnostic-only",
        x_values=SHOTS,
        x_label="Shots",
        y_label="Held-out R²",
        series=series,
        y_limits=(0.25, 1.0),
    )


def heterogeneity_plot(out_dir: Path, metrics: list[dict[str, str]]) -> None:
    cells = ["eff_su2_r1_d2", "eff_su2_r5_d2", "z_r1_d6", "zz_r1_d6", "zz_r5_d6"]
    lookup = {
        (row["scheme"], row["scope_id"], row["attack"], int(row["reference_count"])): float(row["auc_median"])
        for row in metrics
        if row["scope"] == "cell" and int(row["shots"]) == 1024
    }
    width, height = 1200, 620
    svg = SVG(width, height)
    svg.text(90, 34, "Cell heterogeneity after strict transfer", css_class="title")
    svg.text(90, 57, "Median 1024-shot AUC gain; positive bars favor affine ChannelLiRA", css_class="subtitle")
    top, panel_height, panel_width = 115, 390, 390
    x_limits = (-0.045, 0.115)
    panels = (("loss_mia", 0, "versus loss MIA"), ("latent_lira_mismatched", 16, "versus mismatched LiRA"))
    for panel_index, (attack, count, panel_title) in enumerate(panels):
        left = 150 + panel_index * 520

        def xp(value: float) -> float:
            return left + (value - x_limits[0]) / (x_limits[1] - x_limits[0]) * panel_width

        svg.text(left + panel_width / 2, 93, panel_title, anchor="middle", size=15, weight=650)
        for tick in range(9):
            value = x_limits[0] + tick * (x_limits[1] - x_limits[0]) / 8
            x = xp(value)
            svg.line(x, top, x, top + panel_height, css_class="grid")
            svg.text(x, top + panel_height + 25, f"{value:+.2f}", anchor="middle", css_class="tick")
        svg.line(xp(0), top, xp(0), top + panel_height, stroke="#374151", width=1.8)
        group_height = panel_height / len(cells)
        for index, cell in enumerate(cells):
            center = top + (index + 0.5) * group_height
            if panel_index == 0:
                svg.text(left - 13, center + 4, cell, anchor="end", size=11)
            for offset, scheme, color in (
                (-10, "leave_target_out", "#2563eb"),
                (10, "leave_cell_out", "#7c3aed"),
            ):
                affine = lookup[(scheme, cell, "affine_channel_lira", 16)]
                difference = affine - lookup[(scheme, cell, attack, count)]
                x0, x1 = xp(0), xp(difference)
                svg.rect(min(x0, x1), center + offset - 7, abs(x1 - x0), 14, fill=color, opacity=0.86)
        svg.text(left + panel_width / 2, height - 36, "ChannelLiRA AUC minus comparator AUC", anchor="middle", size=12)
    svg.rect(430, 546, 15, 10, fill="#2563eb")
    svg.text(453, 555, "leave target out", size=11)
    svg.rect(580, 546, 15, 10, fill="#7c3aed")
    svg.text(603, 555, "leave cell out", size=11)
    svg.save(out_dir / "cell_transfer_heterogeneity.svg")


def write_index(result_dir: Path) -> None:
    content = """# ChannelLiRA strict-transfer figures

## Leave-target-out attack AUC

![Leave-target-out AUC](plots/attack_auc_leave_target_out.svg)

The attacked checkpoint is excluded from channel fitting and threshold calibration.

## Leave-cell-out attack AUC

![Leave-cell-out AUC](plots/attack_auc_leave_cell_out.svg)

Every target with the held circuit structure is excluded from channel fitting and
threshold calibration.

## Paired transfer gains

![Paired AUC gains](plots/paired_auc_transfer.svg)

At 1024 shots, the 5th–95th percentile intervals against loss MIA and mismatched
LiRA are above zero for both holdout schemes. Target-holdout comparison with
mismatched LiRA crosses zero at 512 shots, so the evidence is not uniform by budget.

## Operational low-FPR performance

![Calibrated TPR](plots/calibrated_tpr_at_1pct.svg)

ChannelLiRA has higher transferred-threshold TPR than both primary comparators.

## FPR calibration

![Actual FPR](plots/actual_fpr_at_1pct.svg)

Actual FPR remains near 1%, though several 5th–95th percentile intervals exceed the
nominal value. These are descriptive seed intervals, not confidence intervals.

## Structural heterogeneity

![Cell heterogeneity](plots/cell_transfer_heterogeneity.svg)

Pooled superiority is not universal across the five circuit cells.

## Held-out channel diagnostics

![Held-out channel R squared](plots/heldout_channel_r2.svg)

The affine channel approximation transfers better as the shot budget grows, with
wide cell/fold variation retained in the whiskers.
"""
    (result_dir / "PLOTS.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir", type=Path, default=Path("channel_lira_results/transfer_phase3")
    )
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    plot_dir = result_dir / "plots"
    metrics = read_rows(result_dir / "metrics_summary.csv")
    calibration = read_rows(result_dir / "calibration_summary.csv")
    paired_auc = read_rows(result_dir / "paired_auc_contrasts_summary.csv")
    diagnostics = read_rows(result_dir / "channel_diagnostics_overall_summary.csv")
    auc_plots(plot_dir, metrics)
    paired_auc_plot(plot_dir, paired_auc)
    calibrated_tpr_plot(plot_dir, calibration)
    actual_fpr_plot(plot_dir, calibration)
    heterogeneity_plot(plot_dir, metrics)
    diagnostics_plot(plot_dir, diagnostics)
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
    print(f"wrote seven SVG figures{suffix} and {result_dir / 'PLOTS.md'}")


if __name__ == "__main__":
    main()
