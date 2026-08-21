#!/usr/bin/env python3
"""Target/cell-clustered sensitivity analysis for ChannelLiRA AUC contrasts."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.channel_lira_circuit_pilot import structural_cell, write_csv


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stable_offset(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def percentile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", type=Path,
        default=Path("channel_lira_results/transfer_phase3/metrics_raw.csv"),
    )
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("channel_lira_results/clustered_sensitivity_phase4"),
    )
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError("--bootstrap must be at least 100")
    metrics_path = args.metrics.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_rows(metrics_path) if row["scope"] == "target"]
    indexed = {
        (
            row["scheme"], int(row["shots"]), row["scope_id"],
            int(row["simulator_seed"]), row["attack"], int(row["reference_count"]),
        ): float(row["auc"])
        for row in rows
    }
    schemes = sorted({key[0] for key in indexed})
    shots_values = sorted({key[1] for key in indexed})
    target_ids = sorted({key[2] for key in indexed})
    simulator_seeds = sorted({key[3] for key in indexed})
    comparators = (
        ("affine_minus_mismatched_lira", "latent_lira_mismatched", 16),
        ("affine_minus_loss", "loss_mia", 0),
    )
    target_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for scheme in schemes:
        for shots in shots_values:
            for contrast, comparator, comparator_count in comparators:
                target_values: dict[str, float] = {}
                for target_id in target_ids:
                    differences = [
                        indexed[(
                            scheme, shots, target_id, simulator_seed,
                            "affine_channel_lira", 16,
                        )]
                        - indexed[(
                            scheme, shots, target_id, simulator_seed,
                            comparator, comparator_count,
                        )]
                        for simulator_seed in simulator_seeds
                    ]
                    value = float(np.mean(differences))
                    target_values[target_id] = value
                    target_rows.append({
                        "scheme": scheme,
                        "shots": shots,
                        "contrast": contrast,
                        "target_id": target_id,
                        "cell": structural_cell(target_id),
                        "n_simulator_seeds": len(differences),
                        "mean_auc_difference": value,
                    })
                by_cell: dict[str, list[float]] = {}
                for target_id, value in target_values.items():
                    by_cell.setdefault(structural_cell(target_id), []).append(value)
                cells = sorted(by_cell)
                rng = np.random.default_rng(
                    args.seed + stable_offset(f"{scheme}|{shots}|{contrast}") % 1_000_000_000
                )
                target_array = np.asarray(list(target_values.values()), dtype=np.float64)
                target_bootstrap = np.empty(args.bootstrap, dtype=np.float64)
                hierarchical_bootstrap = np.empty(args.bootstrap, dtype=np.float64)
                for replicate in range(args.bootstrap):
                    target_bootstrap[replicate] = float(np.mean(
                        rng.choice(target_array, size=len(target_array), replace=True)
                    ))
                    sampled_cells = rng.choice(cells, size=len(cells), replace=True)
                    sampled_targets = []
                    for cell in sampled_cells:
                        values = np.asarray(by_cell[str(cell)], dtype=np.float64)
                        sampled_targets.extend(
                            rng.choice(values, size=len(values), replace=True).tolist()
                        )
                    hierarchical_bootstrap[replicate] = float(np.mean(sampled_targets))
                cell_means = np.asarray([
                    float(np.mean(by_cell[cell])) for cell in cells
                ])
                record = {
                    "scheme": scheme,
                    "shots": shots,
                    "contrast": contrast,
                    "estimand": "mean target-level AUC difference after averaging simulator seeds",
                    "n_cells": len(cells),
                    "n_targets": len(target_array),
                    "n_simulator_seeds_per_target": len(simulator_seeds),
                    "point_estimate": float(np.mean(target_array)),
                    "positive_targets": int(np.sum(target_array > 0.0)),
                    "positive_cells": int(np.sum(cell_means > 0.0)),
                    "target_only_q025": percentile(target_bootstrap, 0.025),
                    "target_only_q975": percentile(target_bootstrap, 0.975),
                    "hierarchical_cell_target_q025": percentile(hierarchical_bootstrap, 0.025),
                    "hierarchical_cell_target_q975": percentile(hierarchical_bootstrap, 0.975),
                    "bootstrap_replicates": args.bootstrap,
                }
                summary_rows.append(record)
    write_csv(out_dir / "target_contrasts.csv", target_rows)
    write_csv(out_dir / "clustered_summary.csv", summary_rows)
    high = max(shots_values)
    high_rows = [row for row in summary_rows if int(row["shots"]) == high]
    table_lines = [
        f"| {row['scheme']} | {row['contrast']} | {float(row['point_estimate']):+.4f} | "
        f"[{float(row['hierarchical_cell_target_q025']):+.4f}, "
        f"{float(row['hierarchical_cell_target_q975']):+.4f}] | "
        f"{int(row['positive_targets'])}/15 | {int(row['positive_cells'])}/5 |"
        for row in high_rows
    ]
    report = f"""# ChannelLiRA clustered sensitivity analysis

## Target-average result at {high} shots

| Transfer scheme | Contrast | Mean target AUC difference | Hierarchical 95% percentile range | Positive targets | Positive cells |
|---|---|---:|---:|---:|---:|
{chr(10).join(table_lines)}

This analysis changes the estimand from the Phase-3 pooled-record AUC to the mean of
15 target-level AUC differences after averaging each target over ten simulator
seeds. It then resamples five structural cells and three targets within each sampled
cell. Consequently, these values should not numerically match the pooled Phase-3
contrasts.

## Interpretation limits

- The hierarchical ranges incorporate observed target/cell heterogeneity but still
  reuse the same records and do not constitute nested record/model confidence
  intervals.
- Five structural cells are too few for a reviewer-proof population-level
  architecture claim. These ranges are a sensitivity analysis, not definitive
  inference.
- Positive-cell counts use the mean of three target-level AUC contrasts per cell,
  whereas Phase 3 reports pooled-record cell AUC. Those are different estimands and
  can have different signs.
- Independently trained reference ensembles and more target checkpoints remain
  necessary for publication-grade uncertainty.
"""
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_source": str(metrics_path),
        "metrics_source_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        "bootstrap_replicates": args.bootstrap,
        "seed": args.seed,
        "schemes": schemes,
        "shots": shots_values,
        "n_cells": len({structural_cell(target) for target in target_ids}),
        "n_targets": len(target_ids),
        "n_simulator_seeds": len(simulator_seeds),
    }
    (out_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote clustered sensitivity artifacts to {out_dir}")


if __name__ == "__main__":
    main()
