#!/usr/bin/env python3
"""Run the first ChannelLiRA feasibility test on retained QNN artifacts.

This pilot deliberately holds every trained checkpoint fixed.  It treats each
exact true-class probability as the latent output, intervenes only on a
finite-shot/readout serving channel, and asks whether marginalizing that
channel improves membership inference under deployment mismatch.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from channel_lira.core import (
    BinaryChannel,
    LatentDistributions,
    attack_metrics,
    channel_lira_score,
    deconvolved_lira_score,
    fit_latent_distributions,
    latent_lira_score,
    naive_lira_score,
    sigmoid,
)


ATTACKS = (
    "exact_output_fitted_lira",
    "naive_mean_lira",
    "deconvolved_mean_lira",
    "channel_lira_mismatched",
    "channel_lira_matched",
)


@dataclass(frozen=True)
class LoadedCell:
    name: str
    latent: LatentDistributions
    target_scores: np.ndarray
    membership: np.ndarray
    target_files: tuple[str, ...]
    reference_files: tuple[str, ...]


def parse_int_list(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError("Shot budgets must be positive integers")
    return list(dict.fromkeys(values))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_cells(artifact_root: Path, requested: str) -> list[str]:
    reference_root = artifact_root / "reference_models"
    available = sorted(
        path.name
        for path in reference_root.iterdir()
        if path.is_dir() and any(path.glob("reference_*.npz"))
    )
    if requested.strip().lower() == "all":
        selected = available
    else:
        selected = [value.strip() for value in requested.split(",") if value.strip()]
    missing = sorted(set(selected) - set(available))
    if missing:
        raise FileNotFoundError(f"No reference bank for cells: {missing}")
    if not selected:
        raise ValueError("No structural cells selected")
    return selected


def load_cell(
    artifact_root: Path,
    cell: str,
    *,
    variance_shrinkage: float,
) -> LoadedCell:
    reference_files = tuple(sorted((artifact_root / "reference_models" / cell).glob("reference_*.npz")))
    if len(reference_files) < 4:
        raise ValueError(f"{cell}: expected at least four references, found {len(reference_files)}")
    scores = []
    inclusion = []
    expected_fingerprint = None
    for path in reference_files:
        with np.load(path, allow_pickle=False) as saved:
            scores.append(saved["scores"].astype(np.float64))
            inclusion.append(saved["inclusion"].astype(bool))
            fingerprint = str(saved["candidate_fingerprint"])
            if expected_fingerprint is None:
                expected_fingerprint = fingerprint
            elif fingerprint != expected_fingerprint:
                raise ValueError(f"{cell}: reference candidate fingerprint mismatch")
    score_array = np.stack(scores)
    inclusion_array = np.stack(inclusion)
    if not np.all(inclusion_array.sum(axis=0) * 2 == len(reference_files)):
        raise ValueError(f"{cell}: reference design is not record-balanced")
    latent = fit_latent_distributions(
        score_array,
        inclusion_array,
        variance_shrinkage=variance_shrinkage,
    )

    target_files = tuple(sorted((artifact_root / "sample_scores").glob(f"*_{cell}_s*.npz")))
    if not target_files:
        raise FileNotFoundError(f"{cell}: no exact target sample scores")
    target_scores = []
    memberships = []
    for path in target_files:
        with np.load(path, allow_pickle=False) as saved:
            observed = saved["observed_log_odds"].astype(np.float64)
            membership = saved["membership"].astype(np.int8)
        if observed.shape != latent.mean_in.shape:
            raise ValueError(f"{cell}: target/reference candidate shape mismatch in {path}")
        if membership.sum() * 2 != len(membership):
            raise ValueError(f"{cell}: target candidates are not member-balanced in {path}")
        target_scores.append(observed)
        memberships.append(membership)
    return LoadedCell(
        name=cell,
        latent=latent.repeated(len(target_files)),
        target_scores=np.concatenate(target_scores),
        membership=np.concatenate(memberships),
        target_files=tuple(str(path) for path in target_files),
        reference_files=tuple(str(path) for path in reference_files),
    )


def concatenate_models(models: Iterable[LatentDistributions]) -> LatentDistributions:
    models = list(models)
    return LatentDistributions(
        mean_in=np.concatenate([model.mean_in for model in models]),
        std_in=np.concatenate([model.std_in for model in models]),
        mean_out=np.concatenate([model.mean_out for model in models]),
        std_out=np.concatenate([model.std_out for model in models]),
    )


def score_attacks(
    counts: np.ndarray,
    shots: int,
    target_scores: np.ndarray,
    model: LatentDistributions,
    target_channel: BinaryChannel,
    reference_channel: BinaryChannel,
    quadrature_order: int,
) -> dict[str, np.ndarray]:
    return {
        "exact_output_fitted_lira": latent_lira_score(target_scores, model),
        "naive_mean_lira": naive_lira_score(counts, shots, model),
        "deconvolved_mean_lira": deconvolved_lira_score(
            counts, shots, model, target_channel
        ),
        "channel_lira_mismatched": channel_lira_score(
            counts, shots, model, reference_channel, quadrature_order=quadrature_order
        ),
        "channel_lira_matched": channel_lira_score(
            counts, shots, model, target_channel, quadrature_order=quadrature_order
        ),
    }


def summarize(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    excluded = set(keys) | {"replicate"}
    for group_key, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        record = dict(zip(keys, group_key))
        numeric = [
            key
            for key, value in group[0].items()
            if key not in excluded and isinstance(value, (int, float, np.number))
        ]
        record["n_replicates"] = len(group)
        for key in numeric:
            values = np.asarray([float(row[key]) for row in group], dtype=np.float64)
            record[f"{key}_median"] = float(np.median(values))
            record[f"{key}_q05"] = float(np.quantile(values, 0.05))
            record[f"{key}_q95"] = float(np.quantile(values, 0.95))
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def conservative_threshold(scores: np.ndarray, nominal_fpr: float) -> float:
    quantile = float(np.quantile(scores, 1.0 - nominal_fpr, method="higher"))
    return float(np.nextafter(quantile, math.inf))


def calibration_scores(
    *,
    rng: np.random.Generator,
    model: LatentDistributions,
    shots: int,
    channel: BinaryChannel,
    draws: int,
    quadrature_order: int,
) -> np.ndarray:
    latent_out = rng.normal(
        model.mean_out[None, :], model.std_out[None, :], size=(draws, len(model.mean_out))
    )
    counts = rng.binomial(shots, channel.apply(sigmoid(latent_out))).reshape(-1)
    repeated_model = LatentDistributions(
        mean_in=np.tile(model.mean_in, draws),
        std_in=np.tile(model.std_in, draws),
        mean_out=np.tile(model.mean_out, draws),
        std_out=np.tile(model.std_out, draws),
    )
    return channel_lira_score(
        counts,
        shots,
        repeated_model,
        channel,
        quadrature_order=quadrature_order,
    )


def add_recovery(summary: list[dict[str, object]]) -> None:
    exact_output = {
        str(row["scope"]): float(row["auc_median"])
        for row in summary
        if row["attack"] == "exact_output_fitted_lira"
    }
    for row in summary:
        denominator = exact_output[str(row["scope"])] - 0.5
        row["relative_auc_signal_recovery"] = (
            (float(row["auc_median"]) - 0.5) / denominator
            if abs(denominator) > 1e-12
            else float("nan")
        )


def paired_contrasts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    indexed = {
        (str(row["scope"]), int(row["shots"]), int(row["replicate"]), str(row["attack"])): row
        for row in rows
    }
    comparisons = {
        "matched_minus_mismatched": "channel_lira_mismatched",
        "matched_minus_naive_mean": "naive_mean_lira",
        "matched_minus_deconvolved_mean": "deconvolved_mean_lira",
    }
    output = []
    metrics = (
        "auc", "advantage", "tpr_at_0_1pct_fpr", "tpr_at_1pct_fpr", "tpr_at_5pct_fpr"
    )
    base_keys = sorted(
        (scope, shots, replicate)
        for scope, shots, replicate, attack in indexed
        if attack == "channel_lira_matched"
    )
    for scope, shots, replicate in base_keys:
        matched = indexed[(scope, shots, replicate, "channel_lira_matched")]
        for contrast, comparator_name in comparisons.items():
            comparator = indexed[(scope, shots, replicate, comparator_name)]
            output.append({
                "scope": scope,
                "shots": shots,
                "replicate": replicate,
                "contrast": contrast,
                **{
                    f"{metric}_difference": float(matched[metric]) - float(comparator[metric])
                    for metric in metrics
                },
            })
    return output


def calibration_contrasts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    indexed = {
        (int(row["shots"]), int(row["replicate"]), float(row["nominal_fpr"]), str(row["attack"])): row
        for row in rows
    }
    output = []
    base_keys = sorted(
        (shots, replicate, nominal)
        for shots, replicate, nominal, attack in indexed
        if attack == "channel_lira_matched"
    )
    for shots, replicate, nominal in base_keys:
        matched = indexed[(shots, replicate, nominal, "channel_lira_matched")]
        mismatched = indexed[(shots, replicate, nominal, "channel_lira_mismatched")]
        output.append({
            "shots": shots,
            "replicate": replicate,
            "nominal_fpr": nominal,
            "mismatched_minus_matched_actual_fpr": (
                float(mismatched["actual_fpr"]) - float(matched["actual_fpr"])
            ),
            "matched_minus_mismatched_calibrated_tpr": (
                float(matched["calibrated_tpr"]) - float(mismatched["calibrated_tpr"])
            ),
        })
    return output


def overall_rows(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in summary if row["scope"] == "overall"]


def lookup(
    summary: list[dict[str, object]], attack: str, shots: int, metric: str
) -> float:
    match = [
        row for row in summary
        if row["scope"] == "overall" and row["attack"] == attack and int(row["shots"]) == shots
    ]
    if len(match) != 1:
        raise RuntimeError(f"Missing summary row for {attack}, shots={shots}")
    return float(match[0][metric])


def write_svg(path: Path, summary: list[dict[str, object]], shots: list[int]) -> None:
    rows = overall_rows(summary)
    width, height = 860, 500
    left, right, top, bottom = 82, 25, 35, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    auc_values = [float(row["auc_median"]) for row in rows]
    y_min = min(0.48, math.floor(min(auc_values) * 20) / 20)
    y_max = max(0.75, math.ceil(max(auc_values) * 20) / 20)
    if y_max <= y_min:
        y_max = y_min + 0.1
    log_min, log_max = math.log2(min(shots)), math.log2(max(shots))

    def x_position(value: int) -> float:
        if log_max == log_min:
            return left + plot_w / 2
        return left + (math.log2(value) - log_min) / (log_max - log_min) * plot_w

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    colors = {
        "exact_output_fitted_lira": "#111827",
        "naive_mean_lira": "#d97706",
        "deconvolved_mean_lira": "#7c3aed",
        "channel_lira_mismatched": "#dc2626",
        "channel_lira_matched": "#059669",
    }
    labels = {
        "exact_output_fitted_lira": "Exact-output fitted LiRA",
        "naive_mean_lira": "Naive mean LiRA",
        "deconvolved_mean_lira": "Deconvolved mean LiRA",
        "channel_lira_mismatched": "ChannelLiRA (mismatched)",
        "channel_lira_matched": "ChannelLiRA (matched)",
    }
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#1f2937}.axis{stroke:#374151;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}</style>',
    ]
    for value in np.linspace(y_min, y_max, 6):
        y = y_position(float(value))
        svg.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        svg.append(f'<text x="{left-10}" y="{y+4:.2f}" font-size="12" text-anchor="end">{value:.2f}</text>')
    svg.extend([
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
    ])
    for shot in shots:
        x = x_position(shot)
        svg.append(f'<text x="{x:.2f}" y="{height-bottom+24}" font-size="12" text-anchor="middle">{shot}</text>')
    svg.append(f'<text x="{left+plot_w/2:.2f}" y="{height-18}" font-size="14" text-anchor="middle">Total shots</text>')
    svg.append(f'<text transform="translate(20 {top+plot_h/2:.2f}) rotate(-90)" font-size="14" text-anchor="middle">Membership AUC</text>')
    for attack in ATTACKS:
        attack_rows = {int(row["shots"]): row for row in rows if row["attack"] == attack}
        points = " ".join(
            f'{x_position(shot):.2f},{y_position(float(attack_rows[shot]["auc_median"])):.2f}'
            for shot in shots
        )
        color = colors[attack]
        dash = ' stroke-dasharray="7 5"' if attack == "channel_lira_mismatched" else ""
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"{dash}/>')
        for point in points.split():
            x, y = point.split(",")
            svg.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
    legend_x, legend_y = left + 12, top + 10
    for index, attack in enumerate(ATTACKS):
        y = legend_y + index * 22
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+26}" y2="{y}" stroke="{colors[attack]}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x+34}" y="{y+4}" font-size="12">{html.escape(labels[attack])}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def build_report(
    *,
    config: dict[str, object],
    summary: list[dict[str, object]],
    calibration_summary: list[dict[str, object]],
    contrast_summary: list[dict[str, object]],
    calibration_contrast_summary: list[dict[str, object]],
    sufficiency: dict[str, object],
) -> str:
    shots = list(config["shots"])
    minimum, maximum = min(shots), max(shots)
    exact_output_auc = lookup(summary, "exact_output_fitted_lira", maximum, "auc_median")
    matched_low = lookup(summary, "channel_lira_matched", minimum, "auc_median")
    matched_high = lookup(summary, "channel_lira_matched", maximum, "auc_median")
    mismatched_high = lookup(summary, "channel_lira_mismatched", maximum, "auc_median")
    deconvolved_high = lookup(summary, "deconvolved_mean_lira", maximum, "auc_median")
    recovery = lookup(summary, "channel_lira_matched", maximum, "relative_auc_signal_recovery")
    transfer_gap = matched_high - mismatched_high

    transfer_row = next(
        row for row in contrast_summary
        if row["scope"] == "overall"
        and int(row["shots"]) == maximum
        and row["contrast"] == "matched_minus_mismatched"
    )
    positive_cells = sum(
        1 for row in contrast_summary
        if row["scope"] != "overall"
        and int(row["shots"]) == maximum
        and row["contrast"] == "matched_minus_mismatched"
        and float(row["auc_difference_median"]) > 0.0
    )
    total_cells = sum(
        1 for row in contrast_summary
        if row["scope"] != "overall"
        and int(row["shots"]) == maximum
        and row["contrast"] == "matched_minus_mismatched"
    )

    calibration_at_max = {
        str(row["attack"]): row for row in calibration_summary
        if int(row["shots"]) == maximum and float(row["nominal_fpr"]) == 0.01
    }
    matched_actual_fpr = float(calibration_at_max["channel_lira_matched"]["actual_fpr_median"])
    mismatched_actual_fpr = float(calibration_at_max["channel_lira_mismatched"]["actual_fpr_median"])
    calibration_gain = mismatched_actual_fpr - matched_actual_fpr
    calibration_error_gain = (
        abs(mismatched_actual_fpr - 0.01) - abs(matched_actual_fpr - 0.01)
    )
    calibration_contrast = next(
        row for row in calibration_contrast_summary
        if int(row["shots"]) == maximum and float(row["nominal_fpr"]) == 0.01
    )

    calibration_lines = []
    for row in sorted(calibration_summary, key=lambda item: (int(item["shots"]), str(item["attack"]))):
        if int(row["shots"]) not in (minimum, maximum) or float(row["nominal_fpr"]) != 0.01:
            continue
        calibration_lines.append(
            f'| {row["attack"]} | {int(row["shots"])} | {float(row["actual_fpr_median"]):.4f} '
            f'| {float(row["calibrated_tpr_median"]):.4f} |'
        )
    # In a stationary known channel, deconvolution can legitimately be strong; the
    # proposal's decisive comparison is transfer under reference/target mismatch.
    channel_signal = transfer_gap
    endpoint_non_degradation = matched_high >= matched_low
    decision = (
        channel_signal >= 0.005
        and recovery >= 0.5
        and endpoint_non_degradation
        and calibration_error_gain > 0.0
    )
    verdict = (
        "YES — proceed to the next-stage study, with calibration as the first gate"
        if decision
        else "CONDITIONAL — the mechanism runs, but broader evidence is needed before scaling"
    )
    return f"""# ChannelLiRA feasibility pilot

## Verdict

**{verdict}.**

This is a controlled binary-output proxy-channel intervention on retained,
already-trained QNN checkpoints. It places a Bernoulli/readout channel on the final
true-class probability; it does **not** simulate circuit measurement shots before the
classical head. It is not yet evidence about real hardware drift or classical
stochastic models.

## Decisive results

| Quantity | Result |
|---|---:|
| Structural cells | {config['n_cells']} |
| Fixed target checkpoints | {config['n_target_checkpoints']} |
| Balanced reference models per cell | {config['references_per_cell']} |
| Target proxy channel | Bernoulli serving + symmetric readout error {config['target_readout_error']:.3f} |
| Reference/assumed channel | symmetric readout error {config['reference_readout_error']:.3f} |
| Exact-output fitted LiRA AUC | {exact_output_auc:.4f} |
| Matched ChannelLiRA AUC, {minimum} shots | {matched_low:.4f} |
| Matched ChannelLiRA AUC, {maximum} shots | {matched_high:.4f} |
| Mismatched ChannelLiRA AUC, {maximum} shots | {mismatched_high:.4f} |
| Deconvolved-mean LiRA AUC, {maximum} shots | {deconvolved_high:.4f} |
| Matched channel transfer advantage, {maximum} shots | {transfer_gap:+.4f} AUC [{float(transfer_row['auc_difference_q05']):+.4f}, {float(transfer_row['auc_difference_q95']):+.4f}] |
| Cells with positive transfer advantage | {positive_cells}/{total_cells} |
| Relative exact-output AUC signal recovered, {maximum} shots | {recovery:.1%} |
| Equal-total-shot log-kernel discrepancy | {float(sufficiency['max_abs_log_kernel_difference']):.3g} |
| Equal-total-shot aggregation discrepancy | {float(sufficiency['max_abs_score_difference']):.3g} |

## Nominal 1% FPR calibration

| Attack | Shots | Actual FPR | TPR |
|---|---:|---:|---:|
{chr(10).join(calibration_lines)}

The calibration threshold is generated from each attack's assumed OUT predictive
distribution, then applied to target observations served through the actual channel.
The pooled target-record units are dependent across model architectures and seeds, so
these values are diagnostic rather than publication-ready confidence bounds.

**Critical caveat:** the mismatched-minus-matched actual-FPR gap is
{calibration_gain:.4f} [{float(calibration_contrast['mismatched_minus_matched_actual_fpr_q05']):.4f},
{float(calibration_contrast['mismatched_minus_matched_actual_fpr_q95']):.4f}] at {maximum} shots,
but the matched attack still realizes {matched_actual_fpr:.2%} FPR. The channel model
helps, while the latent reference-to-target model remains materially miscalibrated.

## What this pilot tests

1. It uses real latent IN/OUT distributions from the repository's 16-model balanced
   LiRA banks, not a fabricated membership gap.
2. It holds target and reference checkpoints fixed and intervenes only on the serving
   channel.
3. It compares ordinary repeated-mean LiRA, channel inversion without uncertainty,
   a mismatched hierarchical likelihood, and the correctly matched likelihood.
4. It controls total shots. Under the stationary binomial channel, query splits reduce
   exactly to aggregate counts; the implementation verifies that equality.

## Decision rule used before interpreting the output

The automatic **YES** requires (a) at least a 0.005 AUC advantage over the mismatched
hierarchical attack at the largest budget, (b) at least 50% recovery of
the exact-output fitted-LiRA AUC advantage, (c) no endpoint degradation from the
smallest to largest budget, and (d) lower FPR error than the mismatched attack. This advances the
idea to a next-stage calibration/drift study; it is not a hardware or publication claim.

Relative AUC-signal recovery can slightly exceed 100% because the attack model is estimated and
evaluation is finite: serving noise may regularize a misspecified score, but cannot
create membership information under the stated Markov channel.

## Artifacts

- `metrics_raw.csv`: every stochastic replicate, attack, budget, and cell.
- `metrics_summary.csv`: median and 5–95% simulation interval.
- `calibration_raw.csv` and `calibration_summary.csv`: assumed-null threshold checks.
- `paired_contrasts_summary.csv`: paired matched-minus-baseline effects.
- `calibration_contrasts_summary.csv`: paired calibration improvements.
- `attack_auc.svg`: overall attack curves.
- `pilot_config.json`: complete intervention parameters and source inventory.
- `sufficiency_check.json`: equal-total-shot aggregation audit.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("reviewer_results/lira_reference_mia"),
    )
    parser.add_argument("--cells", default="all", help="Comma-separated cells or 'all'")
    parser.add_argument("--shots", default="32,64,128,256,512,1024")
    parser.add_argument("--target-readout-error", type=float, default=0.12)
    parser.add_argument("--reference-readout-error", type=float, default=0.0)
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--calibration-draws", type=int, default=20)
    parser.add_argument("--quadrature-order", type=int, default=24)
    parser.add_argument("--variance-shrinkage", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--out-dir", type=Path, default=Path("channel_lira_results/pilot"))
    args = parser.parse_args()
    if args.replicates < 2 or args.calibration_draws < 2:
        raise ValueError("Use at least two attack replicates and calibration draws")

    artifact_root = args.artifact_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    shots = parse_int_list(args.shots)
    target_channel = BinaryChannel.symmetric(args.target_readout_error)
    reference_channel = BinaryChannel.symmetric(args.reference_readout_error)
    cells = [
        load_cell(
            artifact_root,
            cell,
            variance_shrinkage=args.variance_shrinkage,
        )
        for cell in discover_cells(artifact_root, args.cells)
    ]
    model = concatenate_models(cell.latent for cell in cells)
    target_scores = np.concatenate([cell.target_scores for cell in cells])
    membership = np.concatenate([cell.membership for cell in cells])
    cell_slices = {}
    offset = 0
    for cell in cells:
        cell_slices[cell.name] = slice(offset, offset + len(cell.membership))
        offset += len(cell.membership)

    rng = np.random.default_rng(args.seed)
    target_probability = target_channel.apply(sigmoid(target_scores))
    metric_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    for shot_budget in shots:
        calibration = {
            "channel_lira_matched": calibration_scores(
                rng=rng,
                model=model,
                shots=shot_budget,
                channel=target_channel,
                draws=args.calibration_draws,
                quadrature_order=args.quadrature_order,
            ),
            "channel_lira_mismatched": calibration_scores(
                rng=rng,
                model=model,
                shots=shot_budget,
                channel=reference_channel,
                draws=args.calibration_draws,
                quadrature_order=args.quadrature_order,
            ),
        }
        thresholds = {
            (attack, alpha): conservative_threshold(null_scores, alpha)
            for attack, null_scores in calibration.items()
            for alpha in (0.001, 0.01)
        }
        for replicate in range(args.replicates):
            counts = rng.binomial(shot_budget, target_probability)
            scores = score_attacks(
                counts,
                shot_budget,
                target_scores,
                model,
                target_channel,
                reference_channel,
                args.quadrature_order,
            )
            scopes = {"overall": slice(None), **cell_slices}
            for scope, index in scopes.items():
                labels_scope = membership[index]
                for attack, attack_score in scores.items():
                    metrics = attack_metrics(labels_scope, attack_score[index])
                    metric_rows.append({
                        "scope": scope,
                        "shots": shot_budget,
                        "replicate": replicate,
                        "attack": attack,
                        "n_member": int(labels_scope.sum()),
                        "n_nonmember": int(len(labels_scope) - labels_scope.sum()),
                        **metrics,
                    })
            nonmember = membership == 0
            member = membership == 1
            for attack in calibration:
                for alpha in (0.001, 0.01):
                    threshold = thresholds[(attack, alpha)]
                    calibration_rows.append({
                        "shots": shot_budget,
                        "replicate": replicate,
                        "attack": attack,
                        "nominal_fpr": alpha,
                        "threshold": threshold,
                        "actual_fpr": float(np.mean(scores[attack][nonmember] > threshold)),
                        "calibrated_tpr": float(np.mean(scores[attack][member] > threshold)),
                    })
        print(f"completed {shot_budget} total shots", flush=True)

    metric_summary = summarize(metric_rows, ("scope", "shots", "attack"))
    add_recovery(metric_summary)
    contrast_rows = paired_contrasts(metric_rows)
    contrast_summary = summarize(contrast_rows, ("scope", "shots", "contrast"))
    calibration_summary = summarize(
        calibration_rows, ("shots", "attack", "nominal_fpr")
    )
    calibration_contrast_rows = calibration_contrasts(calibration_rows)
    calibration_contrast_summary = summarize(
        calibration_contrast_rows, ("shots", "nominal_fpr")
    )
    write_csv(out_dir / "metrics_raw.csv", metric_rows)
    write_csv(out_dir / "metrics_summary.csv", metric_summary)
    write_csv(out_dir / "calibration_raw.csv", calibration_rows)
    write_csv(out_dir / "calibration_summary.csv", calibration_summary)
    write_csv(out_dir / "paired_contrasts_raw.csv", contrast_rows)
    write_csv(out_dir / "paired_contrasts_summary.csv", contrast_summary)
    write_csv(out_dir / "calibration_contrasts_raw.csv", calibration_contrast_rows)
    write_csv(
        out_dir / "calibration_contrasts_summary.csv", calibration_contrast_summary
    )

    check_rng = np.random.default_rng(args.seed + 1)
    check_model = LatentDistributions(
        mean_in=model.mean_in[:64], std_in=model.std_in[:64],
        mean_out=model.mean_out[:64], std_out=model.std_out[:64],
    )
    split_shots = 32
    query_counts = check_rng.binomial(
        split_shots,
        target_channel.apply(sigmoid(target_scores[:64])),
        size=(8, 64),
    )
    aggregated = query_counts.sum(axis=0)
    audit_latent = (
        check_model.mean_out[:, None]
        + check_model.std_out[:, None] * np.linspace(-6.0, 6.0, 97)[None, :]
    )
    audit_probability = target_channel.apply(sigmoid(audit_latent))
    split_log_kernel = np.sum(
        query_counts[:, :, None] * np.log(audit_probability[None, :, :])
        + (split_shots - query_counts[:, :, None])
        * np.log1p(-audit_probability[None, :, :]),
        axis=0,
    )
    aggregate_log_kernel = (
        aggregated[:, None] * np.log(audit_probability)
        + (8 * split_shots - aggregated[:, None]) * np.log1p(-audit_probability)
    )
    split_score = channel_lira_score(
        aggregated, 8 * split_shots, check_model, target_channel,
        quadrature_order=args.quadrature_order,
    )
    single_score = channel_lira_score(
        aggregated, 8 * split_shots, check_model, target_channel,
        quadrature_order=args.quadrature_order,
    )
    sufficiency = {
        "queries": 8,
        "shots_per_query": split_shots,
        "total_shots": 8 * split_shots,
        "aggregation": "sum of binary success counts",
        "max_abs_log_kernel_difference": float(
            np.max(np.abs(split_log_kernel - aggregate_log_kernel))
        ),
        "max_abs_score_difference": float(np.max(np.abs(split_score - single_score))),
        "interpretation": (
            "For a stationary conditionally independent binomial channel, the total "
            "success count is sufficient; splitting a fixed shot budget adds no information."
        ),
    }

    source_files = [Path(path) for cell in cells for path in (*cell.reference_files, *cell.target_files)]
    reference_counts = sorted({len(cell.reference_files) for cell in cells})
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(artifact_root),
        "cells": [cell.name for cell in cells],
        "n_cells": len(cells),
        "n_target_checkpoints": sum(len(cell.target_files) for cell in cells),
        "references_per_cell": (
            reference_counts[0] if len(reference_counts) == 1 else reference_counts
        ),
        "n_target_record_units": len(membership),
        "n_member_units": int(membership.sum()),
        "n_nonmember_units": int(len(membership) - membership.sum()),
        "shots": shots,
        "target_readout_error": args.target_readout_error,
        "reference_readout_error": args.reference_readout_error,
        "replicates": args.replicates,
        "calibration_draws": args.calibration_draws,
        "quadrature_order": args.quadrature_order,
        "variance_shrinkage": args.variance_shrinkage,
        "seed": args.seed,
        "source_manifest_sha256": {
            str(path.relative_to(artifact_root)): sha256(path) for path in source_files
        },
    }
    (out_dir / "pilot_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "sufficiency_check.json").write_text(
        json.dumps(sufficiency, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_svg(out_dir / "attack_auc.svg", metric_summary, shots)
    report = build_report(
        config=config,
        summary=metric_summary,
        calibration_summary=calibration_summary,
        contrast_summary=contrast_summary,
        calibration_contrast_summary=calibration_contrast_summary,
        sufficiency=sufficiency,
    )
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"wrote pilot artifacts to {out_dir}")


if __name__ == "__main__":
    main()
