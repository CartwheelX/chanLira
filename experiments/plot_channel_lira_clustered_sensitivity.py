#!/usr/bin/env python3
"""Plot target/cell-clustered ChannelLiRA sensitivity ranges."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import subprocess

from experiments.plot_channel_lira_phase2 import line_chart


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir", type=Path,
        default=Path("channel_lira_results/clustered_sensitivity_phase4"),
    )
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    with (result_dir / "clustered_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    indexed = {
        (row["scheme"], int(row["shots"]), row["contrast"]): row for row in rows
    }
    shots = [128, 512, 1024]
    specifications = (
        ("leave_cell_out", "affine_minus_mismatched_lira", "Cell out: vs mismatched LiRA", "#dc2626", None),
        ("leave_target_out", "affine_minus_mismatched_lira", "Target out: vs mismatched LiRA", "#991b1b", "7 5"),
        ("leave_cell_out", "affine_minus_loss", "Cell out: vs loss MIA", "#d97706", None),
        ("leave_target_out", "affine_minus_loss", "Target out: vs loss MIA", "#92400e", "7 5"),
    )
    series = []
    for scheme, contrast, label, color, dash in specifications:
        points = []
        for shot in shots:
            row = indexed[(scheme, shot, contrast)]
            points.append({
                "x": shot,
                "value": float(row["point_estimate"]),
                "low": float(row["hierarchical_cell_target_q025"]),
                "high": float(row["hierarchical_cell_target_q975"]),
            })
        series.append({"label": label, "color": color, "dash": dash, "points": points})
    plot_dir = result_dir / "plots"
    line_chart(
        plot_dir / "hierarchical_auc_sensitivity.svg",
        title="Target/cell-clustered AUC sensitivity",
        subtitle="Mean target-level contrasts; 20,000 hierarchical cell/target bootstrap replicates",
        x_values=shots,
        x_label="Shots",
        y_label="Mean target-level AUC difference",
        series=series,
        y_limits=(-0.025, 0.08),
        y_formatter=lambda value: f"{value:+.3f}",
        reference_y=(0.0, "no difference"),
    )
    content = """# ChannelLiRA clustered-sensitivity figure

![Hierarchical AUC sensitivity](plots/hierarchical_auc_sensitivity.svg)

Every hierarchical range crosses zero. This does not negate the fixed-target pooled
Phase-3 result; it shows that five cells and fifteen targets are insufficient for a
population-level architecture-generalization claim.
"""
    (result_dir / "PLOTS.md").write_text(content, encoding="utf-8")
    if args.png:
        convert = shutil.which("convert")
        if convert is None:
            raise FileNotFoundError("--png requires ImageMagick's convert executable")
        subprocess.run(
            [
                convert, "-background", "white", "-density", "144",
                str(plot_dir / "hierarchical_auc_sensitivity.svg"),
                str(plot_dir / "hierarchical_auc_sensitivity.png"),
            ],
            check=True,
        )
    print(f"wrote clustered sensitivity plot to {plot_dir}")


if __name__ == "__main__":
    main()
