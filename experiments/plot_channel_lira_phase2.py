#!/usr/bin/env python3
"""Generate dependency-free SVG figures from circuit-level ChannelLiRA results."""
from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path
import shutil
import subprocess
from typing import Callable, Optional


COLORS = {
    "affine_channel_lira": "#047857",
    "latent_lira_mismatched": "#dc2626",
    "loss_mia": "#d97706",
    "learned_logistic_pv_stats_target_crossfit_upper_bound": "#7c3aed",
    "empirical_channel_lira": "#0891b2",
    "noise_augmented_lira": "#2563eb",
    "exact_output_fitted_lira": "#111827",
}

LABELS = {
    "affine_channel_lira": "Affine ChannelLiRA",
    "latent_lira_mismatched": "Mismatched LiRA",
    "loss_mia": "Loss MIA",
    "learned_logistic_pv_stats_target_crossfit_upper_bound": "Learned logistic MIA",
    "empirical_channel_lira": "Empirical ChannelLiRA",
    "noise_augmented_lira": "Noise-augmented LiRA",
    "exact_output_fitted_lira": "Exact-output fitted LiRA",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def xml(value: object) -> str:
    return html.escape(str(value), quote=True)


class SVG:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            "<style>text{font-family:Inter,system-ui,-apple-system,sans-serif;fill:#1f2937}.grid{stroke:#e5e7eb;stroke-width:1}.axis{stroke:#374151;stroke-width:1.2}.tick{font-size:12px;fill:#4b5563}.title{font-size:22px;font-weight:700}.subtitle{font-size:12px;fill:#6b7280}.legend{font-size:12px}</style>",
        ]

    def line(
        self, x1: float, y1: float, x2: float, y2: float, *,
        stroke: str = "#374151", width: float = 1.0, dash: Optional[str] = None,
        opacity: float = 1.0, css_class: Optional[str] = None,
    ) -> None:
        attributes = f' class="{css_class}"' if css_class else ""
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line{attributes} x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"{dash_attribute}/>'
        )

    def rect(
        self, x: float, y: float, width: float, height: float, *,
        fill: str, opacity: float = 1.0, stroke: Optional[str] = None,
    ) -> None:
        stroke_attribute = f' stroke="{stroke}"' if stroke else ""
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
            f'fill="{fill}" opacity="{opacity}"{stroke_attribute}/>'
        )

    def circle(
        self, x: float, y: float, radius: float, *, fill: str,
        stroke: str = "#ffffff", width: float = 1.5,
    ) -> None:
        self.parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{width}"/>'
        )

    def polyline(
        self, points: list[tuple[float, float]], *, stroke: str,
        width: float = 2.5, dash: Optional[str] = None,
    ) -> None:
        joined = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<polyline points="{joined}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dash_attribute}/>'
        )

    def text(
        self, x: float, y: float, value: object, *, size: int = 12,
        anchor: str = "start", weight: Optional[int] = None,
        fill: Optional[str] = None, rotate: Optional[float] = None,
        css_class: Optional[str] = None,
    ) -> None:
        attributes = f' class="{css_class}"' if css_class else ""
        weight_attribute = f' font-weight="{weight}"' if weight else ""
        fill_attribute = f' fill="{fill}"' if fill else ""
        transform = f' transform="rotate({rotate:.1f} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.parts.append(
            f'<text{attributes} x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
            f'text-anchor="{anchor}"{weight_attribute}{fill_attribute}{transform}>{xml(value)}</text>'
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join((*self.parts, "</svg>")) + "\n", encoding="utf-8")


def rounded_limits(values: list[float], padding: float = 0.08) -> tuple[float, float]:
    low, high = min(values), max(values)
    span = max(high - low, 0.01)
    low -= padding * span
    high += padding * span
    step = 0.01 if span < 0.15 else 0.05
    return math.floor(low / step) * step, math.ceil(high / step) * step


def line_chart(
    path: Path,
    *,
    title: str,
    subtitle: str,
    x_values: list[int],
    x_label: str,
    y_label: str,
    series: list[dict[str, object]],
    y_limits: Optional[tuple[float, float]] = None,
    y_formatter: Callable[[float], str] = lambda value: f"{value:.2f}",
    reference_y: Optional[tuple[float, str]] = None,
) -> None:
    width, height = 1120, 650
    left, right, top, bottom = 90, 310, 92, 78
    plot_width, plot_height = width - left - right, height - top - bottom
    svg = SVG(width, height)
    svg.text(left, 34, title, css_class="title")
    svg.text(left, 57, subtitle, css_class="subtitle")
    values = [float(point[key]) for item in series for point in item["points"] for key in ("low", "high")]
    y_min, y_max = y_limits or rounded_limits(values)

    def x_position(value: int) -> float:
        if len(x_values) == 1:
            return left + plot_width / 2
        return left + x_values.index(value) * plot_width / (len(x_values) - 1)

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    for index in range(6):
        value = y_min + index * (y_max - y_min) / 5
        y = y_position(value)
        svg.line(left, y, left + plot_width, y, css_class="grid")
        svg.text(left - 12, y + 4, y_formatter(value), anchor="end", css_class="tick")
    svg.line(left, top, left, top + plot_height, css_class="axis")
    svg.line(left, top + plot_height, left + plot_width, top + plot_height, css_class="axis")
    for value in x_values:
        x = x_position(value)
        svg.text(x, top + plot_height + 24, value, anchor="middle", css_class="tick")
    svg.text(left + plot_width / 2, height - 18, x_label, anchor="middle", size=13)
    svg.text(22, top + plot_height / 2, y_label, anchor="middle", size=13, rotate=-90)
    if reference_y is not None:
        value, label = reference_y
        y = y_position(value)
        svg.line(left, y, left + plot_width, y, stroke="#6b7280", dash="6 5", width=1.5)
        svg.text(left + plot_width - 3, y - 7, label, anchor="end", size=11, fill="#6b7280")

    for item in series:
        color = str(item["color"])
        points = list(item["points"])
        coordinates = [(x_position(int(point["x"])), y_position(float(point["value"]))) for point in points]
        for point, (x, _) in zip(points, coordinates):
            low_y, high_y = y_position(float(point["low"])), y_position(float(point["high"]))
            svg.line(x, low_y, x, high_y, stroke=color, opacity=0.42, width=1.5)
            svg.line(x - 4, low_y, x + 4, low_y, stroke=color, opacity=0.42, width=1.5)
            svg.line(x - 4, high_y, x + 4, high_y, stroke=color, opacity=0.42, width=1.5)
        svg.polyline(coordinates, stroke=color, width=2.8, dash=item.get("dash"))
        for x, y in coordinates:
            svg.circle(x, y, 4.5, fill=color)

    legend_x, legend_y = left + plot_width + 34, top + 12
    for index, item in enumerate(series):
        y = legend_y + index * 28
        svg.line(legend_x, y, legend_x + 28, y, stroke=str(item["color"]), width=3, dash=item.get("dash"))
        svg.circle(legend_x + 14, y, 3.5, fill=str(item["color"]))
        svg.text(legend_x + 38, y + 4, item["label"], css_class="legend")
    svg.save(path)


def metric_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, int, str, int], dict[str, str]]:
    return {
        (
            row["scope"], row["scope_id"], row["mode"], int(row["shots"]),
            row["attack"], int(row["reference_count"]),
        ): row
        for row in rows
    }


def attack_auc_plot(out_dir: Path, metrics: list[dict[str, str]]) -> None:
    lookup = metric_lookup(metrics)
    shots = [128, 512, 1024]
    attacks = (
        ("affine_channel_lira", 16),
        ("latent_lira_mismatched", 16),
        ("loss_mia", 0),
        ("learned_logistic_pv_stats_target_crossfit_upper_bound", 0),
        ("noise_augmented_lira", 16),
        ("exact_output_fitted_lira", 16),
    )
    series = []
    for attack, count in attacks:
        points = []
        for shot in shots:
            mode, source_shot = ("exact", 0) if attack == "exact_output_fitted_lira" else ("noisy_shot", shot)
            row = lookup[("overall", "all_cells", mode, source_shot, attack, count)]
            points.append({
                "x": shot,
                "value": float(row["auc_median"]),
                "low": float(row["auc_q05"]),
                "high": float(row["auc_q95"]),
            })
        series.append({
            "label": LABELS[attack],
            "color": COLORS[attack],
            "points": points,
            "dash": "7 5" if attack in ("latent_lira_mismatched", "exact_output_fitted_lira") else None,
        })
    line_chart(
        out_dir / "attack_auc_noisy.svg",
        title="Membership attack AUC under IBM-derived noisy Aer",
        subtitle="Overall pooled targets; whiskers are 5th–95th percentiles across 10 simulator seeds",
        x_values=shots,
        x_label="Shots",
        y_label="Attack AUC",
        series=series,
        y_limits=(0.54, 0.625),
    )


def contrast_plot(out_dir: Path, rows: list[dict[str, str]]) -> None:
    selected = {
        "affine_minus_loss": ("vs loss MIA", "#d97706"),
        "affine_minus_learned_logistic": ("vs learned logistic", "#7c3aed"),
        "affine_minus_mismatched_lira": ("vs mismatched LiRA", "#dc2626"),
    }
    shots = [128, 512, 1024]
    indexed = {(row["mode"], int(row["shots"]), row["contrast"]): row for row in rows}
    series = []
    for contrast, (label, color) in selected.items():
        points = []
        for shot in shots:
            row = indexed[("noisy_shot", shot, contrast)]
            points.append({
                "x": shot,
                "value": float(row["auc_difference_median"]),
                "low": float(row["auc_difference_q05"]),
                "high": float(row["auc_difference_q95"]),
            })
        series.append({"label": f"Affine ChannelLiRA {label}", "color": color, "points": points})
    line_chart(
        out_dir / "paired_auc_improvement.svg",
        title="Paired AUC improvement of affine ChannelLiRA",
        subtitle="Within-simulator-seed differences; positive values favor ChannelLiRA",
        x_values=shots,
        x_label="Shots",
        y_label="Paired AUC difference",
        series=series,
        y_limits=(-0.015, 0.05),
        y_formatter=lambda value: f"{value:+.3f}",
        reference_y=(0.0, "no difference"),
    )


def reference_plot(out_dir: Path, metrics: list[dict[str, str]]) -> None:
    lookup = metric_lookup(metrics)
    width, height = 1400, 650
    svg = SVG(width, height)
    svg.text(70, 34, "Reference-count sensitivity under IBM-derived noisy Aer", css_class="title")
    svg.text(70, 57, "Overall AUC; the same exactly balanced reference subsets are used for every attack", css_class="subtitle")
    attacks = (
        "affine_channel_lira", "latent_lira_mismatched", "empirical_channel_lira", "noise_augmented_lira"
    )
    counts = [4, 8, 16]
    panel_width, panel_height = 420, 440
    top, lefts = 105, (90, 590)
    y_min, y_max = 0.51, 0.615
    for panel, (shot, left) in enumerate(zip((128, 1024), lefts)):
        svg.text(left + panel_width / 2, top - 15, f"{shot} shots", anchor="middle", size=15, weight=650)

        def xp(count: int) -> float:
            return left + counts.index(count) * panel_width / 2

        def yp(value: float) -> float:
            return top + (y_max - value) / (y_max - y_min) * panel_height

        for tick in range(6):
            value = y_min + tick * (y_max - y_min) / 5
            y = yp(value)
            svg.line(left, y, left + panel_width, y, css_class="grid")
            svg.text(left - 10, y + 4, f"{value:.2f}", anchor="end", css_class="tick")
        svg.line(left, top, left, top + panel_height, css_class="axis")
        svg.line(left, top + panel_height, left + panel_width, top + panel_height, css_class="axis")
        for count in counts:
            svg.text(xp(count), top + panel_height + 24, count, anchor="middle", css_class="tick")
        for attack in attacks:
            coordinates = []
            for count in counts:
                row = lookup[("overall", "all_cells", "noisy_shot", shot, attack, count)]
                coordinates.append((xp(count), yp(float(row["auc_median"]))))
            svg.polyline(coordinates, stroke=COLORS[attack], width=2.8, dash="7 5" if attack == "latent_lira_mismatched" else None)
            for x, y in coordinates:
                svg.circle(x, y, 4.5, fill=COLORS[attack])
        svg.text(left + panel_width / 2, height - 28, "Reference models", anchor="middle", size=13)
    legend_x, legend_y = 1060, 125
    for index, attack in enumerate(attacks):
        y = legend_y + index * 30
        svg.line(legend_x, y, legend_x + 28, y, stroke=COLORS[attack], width=3,
                 dash="7 5" if attack == "latent_lira_mismatched" else None)
        svg.circle(legend_x + 14, y, 3.5, fill=COLORS[attack])
        svg.text(legend_x + 38, y + 4, LABELS[attack], size=11)
    svg.text(20, top + panel_height / 2, "Attack AUC", anchor="middle", size=13, rotate=-90)
    svg.save(out_dir / "reference_count_ablation.svg")


def heterogeneity_plot(out_dir: Path, metrics: list[dict[str, str]]) -> None:
    lookup = metric_lookup(metrics)
    cells = ["eff_su2_r1_d2", "eff_su2_r5_d2", "z_r1_d6", "zz_r1_d6", "zz_r5_d6"]
    values = []
    for cell in cells:
        affine = float(lookup[("cell", cell, "noisy_shot", 1024, "affine_channel_lira", 16)]["auc_median"])
        loss = float(lookup[("cell", cell, "noisy_shot", 1024, "loss_mia", 0)]["auc_median"])
        mismatch = float(lookup[("cell", cell, "noisy_shot", 1024, "latent_lira_mismatched", 16)]["auc_median"])
        values.append((cell, affine - loss, affine - mismatch))
    width, height = 1080, 570
    left, right, top, bottom = 185, 85, 95, 70
    plot_width, plot_height = width - left - right, height - top - bottom
    x_min, x_max = -0.05, 0.11
    svg = SVG(width, height)
    svg.text(left, 34, "Structural heterogeneity at 1024 noisy shots", css_class="title")
    svg.text(left, 57, "Cell-level median AUC differences; positive bars favor affine ChannelLiRA", css_class="subtitle")

    def xp(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    group_height = plot_height / len(cells)
    for tick in range(9):
        value = x_min + tick * (x_max - x_min) / 8
        x = xp(value)
        svg.line(x, top, x, top + plot_height, css_class="grid")
        svg.text(x, top + plot_height + 24, f"{value:+.2f}", anchor="middle", css_class="tick")
    svg.line(xp(0), top, xp(0), top + plot_height, stroke="#374151", width=1.8)
    for index, (cell, vs_loss, vs_mismatch) in enumerate(values):
        center = top + (index + 0.5) * group_height
        svg.text(left - 14, center + 4, cell, anchor="end", size=12)
        for offset, value, color in ((-11, vs_loss, "#d97706"), (11, vs_mismatch, "#dc2626")):
            x0, x1 = xp(0), xp(value)
            svg.rect(min(x0, x1), center + offset - 7, abs(x1 - x0), 14, fill=color, opacity=0.86)
            svg.text(x1 + (7 if value >= 0 else -7), center + offset + 4, f"{value:+.3f}",
                     anchor="start" if value >= 0 else "end", size=10)
    svg.text(left + plot_width / 2, height - 18, "Affine ChannelLiRA AUC minus comparator AUC", anchor="middle", size=13)
    legend_x = left + plot_width - 220
    svg.rect(legend_x, 72, 15, 10, fill="#d97706")
    svg.text(legend_x + 22, 81, "versus loss MIA", size=11)
    svg.rect(legend_x + 120, 72, 15, 10, fill="#dc2626")
    svg.text(legend_x + 142, 81, "versus mismatched LiRA", size=11)
    svg.save(out_dir / "cell_heterogeneity.svg")


def calibration_plot(out_dir: Path, rows: list[dict[str, str]]) -> None:
    attacks = (
        ("affine_channel_lira", 16),
        ("latent_lira_mismatched", 16),
        ("loss_mia", 0),
    )
    indexed = {
        (row["mode"], int(row["shots"]), row["attack"], int(row["reference_count"])): row
        for row in rows
        if row["scope"] == "overall" and float(row["nominal_fpr"]) == 0.01
    }
    shots = [128, 512, 1024]
    series = []
    for attack, count in attacks:
        points = []
        for shot in shots:
            row = indexed[("noisy_shot", shot, attack, count)]
            points.append({
                "x": shot,
                "value": float(row["actual_fpr_median"]),
                "low": float(row["actual_fpr_q05"]),
                "high": float(row["actual_fpr_q95"]),
            })
        series.append({
            "label": LABELS[attack], "color": COLORS[attack], "points": points,
            "dash": "7 5" if attack == "latent_lira_mismatched" else None,
        })
    line_chart(
        out_dir / "calibration_at_1pct.svg",
        title="Cross-fitted operation at nominal 1% FPR",
        subtitle="Actual FPR on held-out nonmembers under IBM-derived noisy Aer",
        x_values=shots,
        x_label="Shots",
        y_label="Actual false-positive rate",
        series=series,
        y_limits=(0.005, 0.018),
        y_formatter=lambda value: f"{100 * value:.1f}%",
        reference_y=(0.01, "nominal 1%"),
    )


def diagnostics_plot(out_dir: Path, rows: list[dict[str, str]]) -> None:
    cells = ["eff_su2_r1_d2", "eff_su2_r5_d2", "z_r1_d6", "zz_r1_d6", "zz_r5_d6"]
    colors = ["#0f766e", "#2563eb", "#7c3aed", "#db2777", "#ea580c"]
    indexed = {(row["cell"], row["mode"], int(row["shots"])): row for row in rows}
    shots = [128, 512, 1024]
    series = []
    for cell, color in zip(cells, colors):
        points = []
        for shot in shots:
            row = indexed[(cell, "noisy_shot", shot)]
            points.append({
                "x": shot,
                "value": float(row["test_r_squared_median"]),
                "low": float(row["test_r_squared_q05"]),
                "high": float(row["test_r_squared_q95"]),
            })
        series.append({"label": cell, "color": color, "points": points})
    line_chart(
        out_dir / "channel_fit_r2.svg",
        title="Held-out fit of the affine serving channel",
        subtitle="Public-nonmember folds under IBM-derived noisy Aer; higher R² means a better channel approximation",
        x_values=shots,
        x_label="Shots",
        y_label="Held-out R²",
        series=series,
        y_limits=(0.35, 1.0),
    )


def write_index(out_root: Path) -> None:
    content = """# ChannelLiRA circuit-level figures

## Attack comparison

![Attack AUC under IBM-derived noisy Aer](plots/attack_auc_noisy.svg)

Affine ChannelLiRA approaches the exact-output fitted-LiRA ceiling as the shot budget
increases and stays above the loss and learned baselines in the pooled result.

## Paired improvements

![Paired AUC improvements](plots/paired_auc_improvement.svg)

The 1024-shot intervals against loss, learned logistic, and mismatched LiRA are all
above zero. At 128 shots, the interval against loss crosses zero.

## Reference-count sensitivity

![Reference-count ablation](plots/reference_count_ablation.svg)

The largest improvement comes from moving from four to eight reference models;
sixteen references provide the strongest affine result at 1024 shots.

## Structural heterogeneity

![Cell-level heterogeneity](plots/cell_heterogeneity.svg)

The pooled gain is not uniform across structural cells, which is a central gate for
the extended study.

## Fixed-FPR calibration

![Calibration at nominal one percent FPR](plots/calibration_at_1pct.svg)

Cross-fitted empirical thresholds keep realized FPR close to the nominal 1% target.

## Channel-model diagnostics

![Held-out affine channel fit](plots/channel_fit_r2.svg)

The affine approximation improves sharply with shots, but the low-shot and
`zz_r5_d6` cases motivate a heteroskedastic or nonparametric extension.
"""
    (out_root / "PLOTS.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir", type=Path, default=Path("channel_lira_results/circuit_phase2")
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also render PNG previews with ImageMagick's convert executable",
    )
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    plot_dir = result_dir / "plots"
    metrics = read_rows(result_dir / "metrics_summary.csv")
    contrasts = read_rows(result_dir / "paired_contrasts_summary.csv")
    calibration = read_rows(result_dir / "calibration_summary.csv")
    diagnostics = read_rows(result_dir / "channel_diagnostics_summary.csv")
    attack_auc_plot(plot_dir, metrics)
    contrast_plot(plot_dir, contrasts)
    reference_plot(plot_dir, metrics)
    heterogeneity_plot(plot_dir, metrics)
    calibration_plot(plot_dir, calibration)
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
    print(f"wrote six SVG figures{suffix} and {result_dir / 'PLOTS.md'}")


if __name__ == "__main__":
    main()
