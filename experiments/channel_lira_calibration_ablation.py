#!/usr/bin/env python3
"""Separate calibration-set size from cross-architecture channel transfer.

The existing transfer phase compares two auxiliary targets from the victim's cell
with twelve targets from other cells.  This ablation holds the attacked records and
cells fixed while varying cross-cell auxiliary target count over 2/4/8/12.  The
two-target setting exhausts all 66 subsets; 4/8-target settings use deterministic
unique subsets.  The original same-cell/two-target protocol is retained as a source
comparison, but is reported separately because it holds out a target rather than a
complete cell.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from channel_lira.continuous import (
    AffineGaussianChannel,
    affine_channel_lira_score,
    channel_diagnostics,
)
from channel_lira.core import attack_metrics, latent_lira_score
from experiments.channel_lira_circuit_pilot import (
    CellData,
    conservative_threshold,
    discover_cells,
    load_cell,
    parse_int_list,
    sample_id_folds,
    stable_int,
    summarize,
    write_csv,
)
from experiments.channel_lira_transfer import ReferenceBundle, reference_bundles


ATTACKS = ("loss_mia", "latent_lira_mismatched", "affine_channel_lira")
NOMINAL_FPR = 0.01


@dataclass
class AblationBlock:
    strategy: str
    auxiliary_target_count: int
    subset_replicate: int
    cell: str
    target_ids: tuple[str, ...]
    shots: int
    simulator_seeds: tuple[int, ...]
    membership: np.ndarray
    scores: dict[str, np.ndarray]
    decisions: dict[str, np.ndarray]


def score_attacks(
    observed: np.ndarray,
    bundle: ReferenceBundle,
    channel: AffineGaussianChannel,
) -> dict[str, np.ndarray]:
    repetitions = observed.shape[0] * observed.shape[1]
    flattened = observed.reshape(-1)
    latent = bundle.latent.repeated(repetitions)
    return {
        "latent_lira_mismatched": latent_lira_score(flattened, latent).reshape(observed.shape),
        "affine_channel_lira": affine_channel_lira_score(
            flattened, latent, channel
        ).reshape(observed.shape),
    }


def deterministic_subsets(
    population_size: int,
    subset_size: int,
    *,
    requested_replicates: int,
    seed: int,
) -> list[tuple[int, ...]]:
    if not 1 <= subset_size <= population_size:
        raise ValueError("Invalid auxiliary subset size")
    if subset_size == population_size:
        return [tuple(range(population_size))]
    if subset_size == 2:
        pairs = list(combinations(range(population_size), 2))
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(pairs))
        return [pairs[int(index)] for index in order]
    maximum = math.comb(population_size, subset_size)
    wanted = min(requested_replicates, maximum)
    rng = np.random.default_rng(seed)
    selected: set[tuple[int, ...]] = set()
    while len(selected) < wanted:
        selected.add(tuple(sorted(
            int(index) for index in rng.choice(population_size, subset_size, replace=False)
        )))
    return sorted(selected)


def evaluate_block(
    held_cell: CellData,
    held_indices: np.ndarray,
    auxiliary_units: list[tuple[CellData, int]],
    bundles_by_cell: dict[str, dict[int, ReferenceBundle]],
    *,
    strategy: str,
    subset_replicate: int,
    shots: int,
    reference_count: int,
    folds: int,
    seed: int,
) -> tuple[AblationBlock, list[dict[str, object]]]:
    condition = held_cell.conditions[("noisy_shot", shots)]
    simulator_seeds = condition.simulator_seeds
    observed = condition.observed_scores[held_indices]
    losses = condition.losses[held_indices]
    membership = held_cell.memberships[held_indices]
    exact = held_cell.exact_scores[held_indices]
    shape = observed.shape
    assignment = sample_id_folds(held_cell.sample_ids, folds, seed)
    scores = {attack: np.full(shape, np.nan, dtype=np.float64) for attack in ATTACKS}
    decisions = {attack: np.zeros(shape, dtype=bool) for attack in ATTACKS}
    diagnostics: list[dict[str, object]] = []

    auxiliary_observed = np.stack([
        cell.conditions[("noisy_shot", shots)].observed_scores[target]
        for cell, target in auxiliary_units
    ])
    auxiliary_exact_base = np.stack([
        cell.exact_scores[target] for cell, target in auxiliary_units
    ])
    auxiliary_exact = np.broadcast_to(
        auxiliary_exact_base[:, None, :], auxiliary_observed.shape
    )
    auxiliary_membership = np.stack([
        cell.memberships[target] for cell, target in auxiliary_units
    ])
    auxiliary_membership = np.broadcast_to(
        auxiliary_membership[:, None, :], auxiliary_observed.shape
    )
    auxiliary_losses = np.stack([
        cell.conditions[("noisy_shot", shots)].losses[target]
        for cell, target in auxiliary_units
    ])
    held_exact = np.broadcast_to(exact[:, None, :], shape)
    held_membership = np.broadcast_to(membership[:, None, :], shape)

    for fold in range(folds):
        held_candidates = assignment == fold
        calibration_candidates = ~held_candidates
        calibration_mask = np.broadcast_to(
            calibration_candidates[None, None, :], auxiliary_observed.shape
        ) & (auxiliary_membership == 0)
        channel = AffineGaussianChannel.fit(
            auxiliary_exact[calibration_mask], auxiliary_observed[calibration_mask]
        )
        held_nonmember_mask = np.broadcast_to(
            held_candidates[None, None, :], shape
        ) & (held_membership == 0)
        diagnostic = {
            "strategy": strategy,
            "auxiliary_target_count": len(auxiliary_units),
            "subset_replicate": subset_replicate,
            "held_cell": held_cell.name,
            "held_target_ids": ";".join(held_cell.target_ids[index] for index in held_indices),
            "auxiliary_target_ids": ";".join(
                cell.target_ids[target] for cell, target in auxiliary_units
            ),
            "shots": shots,
            "fold": fold,
            "n_train_public_nonmember_pairs": int(calibration_mask.sum()),
            "n_test_heldout_nonmember_pairs": int(held_nonmember_mask.sum()),
            "intercept": channel.intercept,
            "slope": channel.slope,
            "scale": channel.scale,
        }
        diagnostic.update({
            f"train_{key}": value for key, value in channel_diagnostics(
                auxiliary_exact[calibration_mask],
                auxiliary_observed[calibration_mask],
                channel,
            ).items()
        })
        diagnostic.update({
            f"test_{key}": value for key, value in channel_diagnostics(
                held_exact[held_nonmember_mask], observed[held_nonmember_mask], channel
            ).items()
        })
        diagnostics.append(diagnostic)

        held_mask = np.broadcast_to(held_candidates[None, None, :], shape)
        held_scored = score_attacks(
            observed, bundles_by_cell[held_cell.name][reference_count], channel
        )
        held_scored["loss_mia"] = -losses
        for attack in ATTACKS:
            scores[attack][held_mask] = held_scored[attack][held_mask]

        auxiliary_null: dict[str, list[np.ndarray]] = {attack: [] for attack in ATTACKS}
        auxiliary_null["loss_mia"].append((-auxiliary_losses)[calibration_mask])
        for unit_index, (cell, _) in enumerate(auxiliary_units):
            unit_observed = auxiliary_observed[unit_index : unit_index + 1]
            unit_scores = score_attacks(
                unit_observed, bundles_by_cell[cell.name][reference_count], channel
            )
            unit_mask = calibration_mask[unit_index : unit_index + 1]
            for attack in ("latent_lira_mismatched", "affine_channel_lira"):
                auxiliary_null[attack].append(unit_scores[attack][unit_mask])
        for attack in ATTACKS:
            null_scores = np.concatenate(auxiliary_null[attack])
            threshold = conservative_threshold(null_scores, NOMINAL_FPR)
            decisions[attack][held_mask] = held_scored[attack][held_mask] > threshold

    if any(not np.isfinite(values).all() for values in scores.values()):
        raise RuntimeError(
            f"Incomplete ablation scores for {strategy} {held_cell.name} {shots}"
        )
    return AblationBlock(
        strategy=strategy,
        auxiliary_target_count=len(auxiliary_units),
        subset_replicate=subset_replicate,
        cell=held_cell.name,
        target_ids=tuple(held_cell.target_ids[index] for index in held_indices),
        shots=shots,
        simulator_seeds=simulator_seeds,
        membership=membership,
        scores=scores,
        decisions=decisions,
    ), diagnostics


def build_metric_rows(blocks: list[AblationBlock]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[AblationBlock]] = {}
    for block in blocks:
        key = (
            block.strategy, block.auxiliary_target_count,
            block.subset_replicate, block.shots,
        )
        grouped.setdefault(key, []).append(block)
    rows = []
    for key, parts in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        strategy, count, subset_replicate, shots = key
        for simulator_index, simulator_seed in enumerate(parts[0].simulator_seeds):
            labels = np.concatenate([part.membership.reshape(-1) for part in parts])
            for attack in ATTACKS:
                values = np.concatenate([
                    part.scores[attack][:, simulator_index, :].reshape(-1)
                    for part in parts
                ])
                rows.append({
                    "strategy": strategy,
                    "auxiliary_target_count": count,
                    "subset_replicate": subset_replicate,
                    "shots": shots,
                    "simulator_seed": simulator_seed,
                    "attack": attack,
                    "n_member": int(labels.sum()),
                    "n_nonmember": int(len(labels) - labels.sum()),
                    **attack_metrics(labels, values),
                })
    return rows


def build_calibration_rows(blocks: list[AblationBlock]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[AblationBlock]] = {}
    for block in blocks:
        key = (
            block.strategy, block.auxiliary_target_count,
            block.subset_replicate, block.shots,
        )
        grouped.setdefault(key, []).append(block)
    rows = []
    for key, parts in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        strategy, count, subset_replicate, shots = key
        for simulator_index, simulator_seed in enumerate(parts[0].simulator_seeds):
            labels = np.concatenate([part.membership.reshape(-1) for part in parts])
            nonmember = labels == 0
            member = labels == 1
            for attack in ATTACKS:
                selected = np.concatenate([
                    part.decisions[attack][:, simulator_index, :].reshape(-1)
                    for part in parts
                ])
                rows.append({
                    "strategy": strategy,
                    "auxiliary_target_count": count,
                    "subset_replicate": subset_replicate,
                    "shots": shots,
                    "simulator_seed": simulator_seed,
                    "attack": attack,
                    "nominal_fpr": NOMINAL_FPR,
                    "n_member": int(member.sum()),
                    "n_nonmember": int(nonmember.sum()),
                    "actual_fpr": float(selected[nonmember].mean()),
                    "calibrated_tpr": float(selected[member].mean()),
                })
    return rows


def build_contrast_rows(
    metrics: list[dict[str, object]],
    calibration: list[dict[str, object]],
) -> list[dict[str, object]]:
    metric_index = {
        (
            row["strategy"], row["auxiliary_target_count"], row["subset_replicate"],
            row["shots"], row["simulator_seed"], row["attack"],
        ): row
        for row in metrics
    }
    calibration_index = {
        (
            row["strategy"], row["auxiliary_target_count"], row["subset_replicate"],
            row["shots"], row["simulator_seed"], row["attack"],
        ): row
        for row in calibration
    }
    base_keys = sorted({key[:5] for key in metric_index}, key=lambda key: tuple(map(str, key)))
    output = []
    for base in base_keys:
        affine_metric = metric_index[base + ("affine_channel_lira",)]
        affine_calibration = calibration_index[base + ("affine_channel_lira",)]
        for contrast, comparator in (
            ("affine_minus_mismatched_lira", "latent_lira_mismatched"),
            ("affine_minus_loss", "loss_mia"),
        ):
            comparator_metric = metric_index[base + (comparator,)]
            comparator_calibration = calibration_index[base + (comparator,)]
            strategy, count, subset_replicate, shots, simulator_seed = base
            output.append({
                "strategy": strategy,
                "auxiliary_target_count": count,
                "subset_replicate": subset_replicate,
                "shots": shots,
                "simulator_seed": simulator_seed,
                "contrast": contrast,
                "comparator_attack": comparator,
                "auc_difference": float(affine_metric["auc"]) - float(comparator_metric["auc"]),
                "tpr_at_0_1pct_fpr_difference": (
                    float(affine_metric["tpr_at_0_1pct_fpr"])
                    - float(comparator_metric["tpr_at_0_1pct_fpr"])
                ),
                "actual_fpr_difference": (
                    float(affine_calibration["actual_fpr"])
                    - float(comparator_calibration["actual_fpr"])
                ),
                "calibrated_tpr_difference": (
                    float(affine_calibration["calibrated_tpr"])
                    - float(comparator_calibration["calibrated_tpr"])
                ),
            })
    return output


def median_over_subsets(
    rows: list[dict[str, object]],
    value_columns: tuple[str, ...],
    row_kind: str,
) -> list[dict[str, object]]:
    identity = "attack" if row_kind in ("metric", "calibration") else "contrast"
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["strategy"], row["auxiliary_target_count"], row["shots"],
            row["simulator_seed"], row[identity],
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for key, parts in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        strategy, count, shots, simulator_seed, identity_value = key
        row = {
            "strategy": strategy,
            "auxiliary_target_count": count,
            "shots": shots,
            "simulator_seed": simulator_seed,
            identity: identity_value,
            "n_subset_replicates": len(parts),
        }
        for column in value_columns:
            row[column] = float(np.median([float(part[column]) for part in parts]))
        output.append(row)
    return output


def median_over_seeds(
    rows: list[dict[str, object]],
    value_columns: tuple[str, ...],
    row_kind: str,
) -> list[dict[str, object]]:
    identity = "attack" if row_kind in ("metric", "calibration") else "contrast"
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["strategy"], row["auxiliary_target_count"], row["subset_replicate"],
            row["shots"], row[identity],
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for key, parts in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        strategy, count, subset_replicate, shots, identity_value = key
        row = {
            "strategy": strategy,
            "auxiliary_target_count": count,
            "subset_replicate": subset_replicate,
            "shots": shots,
            identity: identity_value,
            "n_simulator_seeds": len(parts),
        }
        for column in value_columns:
            row[column] = float(np.median([float(part[column]) for part in parts]))
        output.append(row)
    return output


def build_strategy_contrast_rows(
    metric_seed_medians: list[dict[str, object]],
    calibration_seed_medians: list[dict[str, object]],
    cross_cell_counts: list[int],
) -> list[dict[str, object]]:
    metric_index = {
        (row["strategy"], int(row["shots"]), int(row["simulator_seed"]), row["attack"]): row
        for row in metric_seed_medians
    }
    calibration_index = {
        (row["strategy"], int(row["shots"]), int(row["simulator_seed"]), row["attack"]): row
        for row in calibration_seed_medians
    }
    comparisons = [("cross_cell_2_minus_same_cell_2", "cross_cell_2", "same_cell_2")]
    comparisons.extend(
        (f"cross_cell_{count}_minus_cross_cell_2", f"cross_cell_{count}", "cross_cell_2")
        for count in cross_cell_counts if count != 2
    )
    output = []
    base_keys = sorted({
        (int(row["shots"]), int(row["simulator_seed"]))
        for row in metric_seed_medians
    })
    for shots, simulator_seed in base_keys:
        for contrast, treatment, comparator in comparisons:
            treatment_metric = metric_index[(
                treatment, shots, simulator_seed, "affine_channel_lira"
            )]
            comparator_metric = metric_index[(
                comparator, shots, simulator_seed, "affine_channel_lira"
            )]
            treatment_calibration = calibration_index[(
                treatment, shots, simulator_seed, "affine_channel_lira"
            )]
            comparator_calibration = calibration_index[(
                comparator, shots, simulator_seed, "affine_channel_lira"
            )]
            output.append({
                "shots": shots,
                "simulator_seed": simulator_seed,
                "contrast": contrast,
                "treatment_strategy": treatment,
                "comparator_strategy": comparator,
                "auc_difference": (
                    float(treatment_metric["auc"]) - float(comparator_metric["auc"])
                ),
                "actual_fpr_difference": (
                    float(treatment_calibration["actual_fpr"])
                    - float(comparator_calibration["actual_fpr"])
                ),
                "calibrated_tpr_difference": (
                    float(treatment_calibration["calibrated_tpr"])
                    - float(comparator_calibration["calibrated_tpr"])
                ),
            })
    return output


def lookup(
    rows: list[dict[str, object]],
    strategy: str,
    shots: int,
    identity: str,
) -> dict[str, object]:
    key = "attack" if "attack" in rows[0] else "contrast"
    matches = [
        row for row in rows
        if row["strategy"] == strategy and int(row["shots"]) == shots
        and row[key] == identity
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Missing ablation summary for {(strategy, shots, identity)}")
    return matches[0]


def build_report(
    config: dict[str, object],
    metric_summary: list[dict[str, object]],
    calibration_summary: list[dict[str, object]],
    contrast_summary: list[dict[str, object]],
    contrast_subset_summary: list[dict[str, object]],
    strategy_contrast_summary: list[dict[str, object]],
    diagnostic_summary: list[dict[str, object]],
) -> str:
    high = max(config["shots"])
    strategies = ["same_cell_2"] + [
        f"cross_cell_{count}" for count in config["cross_cell_counts"]
    ]
    result_lines = []
    contrast_lines = []
    for strategy in strategies:
        affine = lookup(metric_summary, strategy, high, "affine_channel_lira")
        mismatch = lookup(metric_summary, strategy, high, "latent_lira_mismatched")
        loss = lookup(metric_summary, strategy, high, "loss_mia")
        calibrated = lookup(calibration_summary, strategy, high, "affine_channel_lira")
        result_lines.append(
            f"| {strategy} | {int(affine['auxiliary_target_count'])} | "
            f"{float(affine['auc_median']):.4f} | {float(mismatch['auc_median']):.4f} | "
            f"{float(loss['auc_median']):.4f} | {float(calibrated['actual_fpr_median']):.4f} | "
            f"{float(calibrated['calibrated_tpr_median']):.4f} |"
        )
        contrast = lookup(contrast_summary, strategy, high, "affine_minus_mismatched_lira")
        contrast_lines.append(
            f"| {strategy} | {float(contrast['auc_difference_median']):+.4f} "
            f"[{float(contrast['auc_difference_q05']):+.4f}, {float(contrast['auc_difference_q95']):+.4f}] | "
            f"{float(contrast['calibrated_tpr_difference_median']):+.4f} "
            f"[{float(contrast['calibrated_tpr_difference_q05']):+.4f}, "
            f"{float(contrast['calibrated_tpr_difference_q95']):+.4f}] |"
        )
    cross_two_subset = lookup(
        contrast_subset_summary, "cross_cell_2", high, "affine_minus_mismatched_lira"
    )
    diagnostics = {
        (row["strategy"], int(row["shots"])): row for row in diagnostic_summary
    }
    diagnostic_lines = []
    for strategy in strategies:
        row = diagnostics[(strategy, high)]
        diagnostic_lines.append(
            f"| {strategy} | {float(row['test_r_squared_median']):.3f} | "
            f"{float(row['test_coverage_90pct_median']):.3f} |"
        )
    strategy_contrasts = {
        row["contrast"]: row for row in strategy_contrast_summary
        if int(row["shots"]) == high
    }
    direct_lines = []
    for contrast in (
        "cross_cell_2_minus_same_cell_2",
        "cross_cell_12_minus_cross_cell_2",
    ):
        row = strategy_contrasts[contrast]
        direct_lines.append(
            f"| {contrast} | {float(row['auc_difference_median']):+.4f} "
            f"[{float(row['auc_difference_q05']):+.4f}, {float(row['auc_difference_q95']):+.4f}] | "
            f"{float(row['actual_fpr_difference_median']):+.4f} | "
            f"{float(row['calibrated_tpr_difference_median']):+.4f} "
            f"[{float(row['calibrated_tpr_difference_q05']):+.4f}, "
            f"{float(row['calibrated_tpr_difference_q95']):+.4f}] |"
        )
    size_gain = float(
        strategy_contrasts["cross_cell_12_minus_cross_cell_2"]["auc_difference_median"]
    )
    source_gain = float(
        strategy_contrasts["cross_cell_2_minus_same_cell_2"]["auc_difference_median"]
    )
    low_fpr_lines = []
    for label, attack in (
        ("Affine ChannelLiRA (`cross_cell_12`)", "affine_channel_lira"),
        ("Mismatched LiRA", "latent_lira_mismatched"),
        ("Loss MIA", "loss_mia"),
    ):
        row = lookup(metric_summary, "cross_cell_12", high, attack)
        low_fpr_lines.append(
            f"| {label} | {float(row['tpr_at_0_1pct_fpr_median']):.4f} "
            f"[{float(row['tpr_at_0_1pct_fpr_q05']):.4f}, "
            f"{float(row['tpr_at_0_1pct_fpr_q95']):.4f}] |"
        )
    return f"""# ChannelLiRA calibration source/count ablation

## Answer

This experiment separates the two explanations for the stronger Phase-3
leave-cell-out result: cross-architecture calibration and the increase from two to
twelve auxiliary targets. At {high} shots, moving from cross-cell two-target
calibration to cross-cell twelve-target calibration changes affine ChannelLiRA AUC
by {size_gain:+.4f}. Changing from same-cell two-target calibration to the
cross-cell two-target median changes it by {source_gain:+.4f}.

The two-target cross-cell condition exhausts all 66 auxiliary-target pairs for every
held cell. Four/eight-target conditions use {config['subset_replicates']} frozen
unique subsets per held cell. Main intervals first take the median over calibration
subsets within each simulator seed, then report the 5th–95th percentiles over ten
seeds. They are sensitivity ranges, not confidence intervals.

## Pooled results at {high} noisy shots

| Calibration strategy | Aux targets | ChannelLiRA AUC | Mismatched AUC | Loss AUC | Actual FPR | Calibrated TPR |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(result_lines)}

## Paired comparisons with mismatched LiRA

| Calibration strategy | AUC difference [5%, 95%] | Calibrated-TPR difference [5%, 95%] |
|---|---:|---:|
{chr(10).join(contrast_lines)}

## Direct calibration-strategy contrasts

These comparisons pair strategies within simulator seed after taking the median
over their calibration subsets.

| Contrast | ChannelLiRA AUC difference [5%, 95%] | Actual-FPR difference | Calibrated-TPR difference [5%, 95%] |
|---|---:|---:|---:|
{chr(10).join(direct_lines)}

Across the 66 cross-cell two-target subsets, the seed-median ChannelLiRA-minus-
mismatched-LiRA AUC contrast has median
{float(cross_two_subset['auc_difference_median']):+.4f} and subset sensitivity range
[{float(cross_two_subset['auc_difference_q05']):+.4f},
{float(cross_two_subset['auc_difference_q95']):+.4f}]. This quantifies how strongly
the conclusion depends on which two auxiliary QNNs are chosen.

## Empirical 0.1% FPR check at {high} shots

| Attack | TPR at empirical 0.1% FPR [5%, 95%] |
|---|---:|
{chr(10).join(low_fpr_lines)}

The pooled evaluation has only {config['pooled_n_nonmember']} nonmembers, so one
false positive changes FPR by
{100.0 / int(config['pooled_n_nonmember']):.4f} percentage points. At this coarse
resolution, affine ChannelLiRA does not beat mismatched LiRA. These values are
reported for transparency and do not support a stable 0.1%-FPR claim; the candidate
population must be enlarged before publication.

## Held-out channel diagnostics at {high} shots

| Calibration strategy | Held-out R² | Held-out 90% coverage |
|---|---:|---:|
{chr(10).join(diagnostic_lines)}

## Interpretation

- Only `cross_cell_2` versus `cross_cell_12` cleanly isolates auxiliary calibration
  count under the same complete-cell holdout. Twelve targets modestly improve AUC
  but also produce a more conservative threshold: lower realized FPR and lower TPR.
- `same_cell_2` versus `cross_cell_2` holds auxiliary count fixed, but changes the
  holdout unit and calibration source. Its paired AUC interval crosses zero, and its
  TPR increase comes with a higher median FPR, so this experiment does not establish
  cross-architecture regularization.
- Mismatched LiRA AUC is channel-independent and therefore provides a flat reference
  across calibration counts. Its operational threshold can still vary with the
  selected auxiliary targets.
- All results reuse one frozen IBM-derived simulator snapshot and architecture-
  matched exact reference banks for the held cells.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--noisy-root", type=Path,
        default=Path("reviewer_results/noisy_sanity/raw_all_seeds"),
    )
    parser.add_argument(
        "--reference-root", type=Path,
        default=Path("reviewer_results/lira_reference_mia/reference_models"),
    )
    parser.add_argument("--cells", default="all")
    parser.add_argument("--shots", default="128,512,1024")
    parser.add_argument("--cross-cell-counts", default="2,4,8,12")
    parser.add_argument("--subset-replicates", type=int, default=32)
    parser.add_argument("--reference-count", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("channel_lira_results/calibration_ablation_phase4"),
    )
    args = parser.parse_args()
    shots = parse_int_list(args.shots)
    cross_cell_counts = parse_int_list(args.cross_cell_counts)
    if args.subset_replicates < 1:
        raise ValueError("--subset-replicates must be positive")
    noisy_root = args.noisy_root.resolve()
    reference_root = args.reference_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = [
        load_cell(noisy_root, reference_root, name)
        for name in discover_cells(noisy_root, args.cells)
    ]
    if len(cells) < 2:
        raise ValueError("Cross-cell ablation requires at least two cells")
    baseline_ids = cells[0].sample_ids
    if any(not np.array_equal(cell.sample_ids, baseline_ids) for cell in cells[1:]):
        raise ValueError("Calibration ablation requires aligned sample IDs")
    available_cross_targets = sum(len(cell.target_ids) for cell in cells[1:])
    if any(count > available_cross_targets for count in cross_cell_counts):
        raise ValueError(
            f"At most {available_cross_targets} cross-cell auxiliary targets are available"
        )
    for cell in cells:
        missing = {("noisy_shot", shot) for shot in shots} - set(cell.conditions)
        if missing:
            raise ValueError(f"{cell.name}: missing conditions {sorted(missing)}")
    bundles_by_cell = {
        cell.name: reference_bundles(cell, [args.reference_count], 0.15)
        for cell in cells
    }

    subset_indices: dict[tuple[str, int], list[tuple[int, ...]]] = {}
    for held_cell in cells:
        pool = [
            (cell, target)
            for cell in cells if cell.name != held_cell.name
            for target in range(len(cell.target_ids))
        ]
        for count in cross_cell_counts:
            subset_indices[(held_cell.name, count)] = deterministic_subsets(
                len(pool), count,
                requested_replicates=args.subset_replicates,
                seed=args.seed + stable_int(f"{held_cell.name}|{count}") % 1_000_000_000,
            )
    replicate_counts = {
        count: min(len(subset_indices[(cell.name, count)]) for cell in cells)
        for count in cross_cell_counts
    }

    blocks: list[AblationBlock] = []
    diagnostics: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for shot in shots:
        for cell in cells:
            for held_target in range(len(cell.target_ids)):
                auxiliary = [
                    (cell, target) for target in range(len(cell.target_ids))
                    if target != held_target
                ]
                block, rows = evaluate_block(
                    cell,
                    np.asarray([held_target], dtype=np.int64),
                    auxiliary,
                    bundles_by_cell,
                    strategy="same_cell_2",
                    subset_replicate=0,
                    shots=shot,
                    reference_count=args.reference_count,
                    folds=args.folds,
                    seed=args.seed,
                )
                blocks.append(block)
                diagnostics.extend(rows)
                selection_rows.append({
                    "strategy": "same_cell_2", "auxiliary_target_count": 2,
                    "subset_replicate": 0, "shots": shot, "held_cell": cell.name,
                    "held_target_ids": cell.target_ids[held_target],
                    "auxiliary_target_ids": ";".join(
                        auxiliary_cell.target_ids[target]
                        for auxiliary_cell, target in auxiliary
                    ),
                })
        print(f"completed same_cell_2 noisy_shot {shot}", flush=True)

        for count in cross_cell_counts:
            strategy = f"cross_cell_{count}"
            for subset_replicate in range(replicate_counts[count]):
                for held_cell in cells:
                    pool = [
                        (cell, target)
                        for cell in cells if cell.name != held_cell.name
                        for target in range(len(cell.target_ids))
                    ]
                    selected = subset_indices[(held_cell.name, count)][subset_replicate]
                    auxiliary = [pool[index] for index in selected]
                    block, rows = evaluate_block(
                        held_cell,
                        np.arange(len(held_cell.target_ids), dtype=np.int64),
                        auxiliary,
                        bundles_by_cell,
                        strategy=strategy,
                        subset_replicate=subset_replicate,
                        shots=shot,
                        reference_count=args.reference_count,
                        folds=args.folds,
                        seed=args.seed,
                    )
                    blocks.append(block)
                    diagnostics.extend(rows)
                    selection_rows.append({
                        "strategy": strategy,
                        "auxiliary_target_count": count,
                        "subset_replicate": subset_replicate,
                        "shots": shot,
                        "held_cell": held_cell.name,
                        "held_target_ids": ";".join(held_cell.target_ids),
                        "auxiliary_target_ids": ";".join(
                            auxiliary_cell.target_ids[target]
                            for auxiliary_cell, target in auxiliary
                        ),
                    })
            print(
                f"completed {strategy} ({replicate_counts[count]} subsets) noisy_shot {shot}",
                flush=True,
            )

    metric_rows = build_metric_rows(blocks)
    calibration_rows = build_calibration_rows(blocks)
    contrast_rows = build_contrast_rows(metric_rows, calibration_rows)
    metric_values = (
        "auc", "advantage", "tpr_at_0_1pct_fpr", "tpr_at_1pct_fpr", "tpr_at_5pct_fpr"
    )
    calibration_values = ("actual_fpr", "calibrated_tpr")
    contrast_values = (
        "auc_difference", "tpr_at_0_1pct_fpr_difference",
        "actual_fpr_difference", "calibrated_tpr_difference",
    )
    metric_seed_medians = median_over_subsets(metric_rows, metric_values, "metric")
    calibration_seed_medians = median_over_subsets(
        calibration_rows, calibration_values, "calibration"
    )
    contrast_seed_medians = median_over_subsets(contrast_rows, contrast_values, "contrast")
    strategy_contrast_rows = build_strategy_contrast_rows(
        metric_seed_medians, calibration_seed_medians, cross_cell_counts
    )
    metric_summary = summarize(
        metric_seed_medians,
        ("strategy", "auxiliary_target_count", "shots", "attack"),
        metric_values,
    )
    calibration_summary = summarize(
        calibration_seed_medians,
        ("strategy", "auxiliary_target_count", "shots", "attack"),
        calibration_values,
    )
    contrast_summary = summarize(
        contrast_seed_medians,
        ("strategy", "auxiliary_target_count", "shots", "contrast"),
        contrast_values,
    )
    strategy_contrast_summary = summarize(
        strategy_contrast_rows,
        ("shots", "contrast", "treatment_strategy", "comparator_strategy"),
        ("auc_difference", "actual_fpr_difference", "calibrated_tpr_difference"),
    )
    metric_subset_medians = median_over_seeds(metric_rows, metric_values, "metric")
    contrast_subset_medians = median_over_seeds(contrast_rows, contrast_values, "contrast")
    metric_subset_summary = summarize(
        metric_subset_medians,
        ("strategy", "auxiliary_target_count", "shots", "attack"),
        metric_values,
    )
    contrast_subset_summary = summarize(
        contrast_subset_medians,
        ("strategy", "auxiliary_target_count", "shots", "contrast"),
        contrast_values,
    )
    diagnostic_summary = summarize(
        diagnostics,
        ("strategy", "auxiliary_target_count", "shots"),
        ("slope", "scale", "test_r_squared", "test_coverage_90pct"),
    )
    write_csv(out_dir / "selection_manifest.csv", selection_rows)
    write_csv(out_dir / "metrics_raw.csv", metric_rows)
    write_csv(out_dir / "metrics_seed_medians.csv", metric_seed_medians)
    write_csv(out_dir / "metrics_summary.csv", metric_summary)
    write_csv(out_dir / "metrics_subset_medians.csv", metric_subset_medians)
    write_csv(out_dir / "metrics_subset_sensitivity.csv", metric_subset_summary)
    write_csv(out_dir / "calibration_raw.csv", calibration_rows)
    write_csv(out_dir / "calibration_seed_medians.csv", calibration_seed_medians)
    write_csv(out_dir / "calibration_summary.csv", calibration_summary)
    write_csv(out_dir / "paired_contrasts_raw.csv", contrast_rows)
    write_csv(out_dir / "paired_contrasts_seed_medians.csv", contrast_seed_medians)
    write_csv(out_dir / "paired_contrasts_summary.csv", contrast_summary)
    write_csv(out_dir / "strategy_contrasts_raw.csv", strategy_contrast_rows)
    write_csv(out_dir / "strategy_contrasts_summary.csv", strategy_contrast_summary)
    write_csv(out_dir / "paired_contrasts_subset_medians.csv", contrast_subset_medians)
    write_csv(out_dir / "paired_contrasts_subset_sensitivity.csv", contrast_subset_summary)
    write_csv(out_dir / "channel_diagnostics_raw.csv", diagnostics)
    write_csv(out_dir / "channel_diagnostics_summary.csv", diagnostic_summary)

    source_config = ROOT / "channel_lira_results/transfer_phase3/experiment_config.json"
    source_hash = hashlib.sha256(source_config.read_bytes()).hexdigest()
    pooled_sizes = {
        (int(row["n_member"]), int(row["n_nonmember"])) for row in metric_rows
    }
    if len(pooled_sizes) != 1:
        raise RuntimeError(f"Inconsistent pooled evaluation sizes: {sorted(pooled_sizes)}")
    pooled_n_member, pooled_n_nonmember = pooled_sizes.pop()
    config: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cells": [cell.name for cell in cells],
        "shots": shots,
        "cross_cell_counts": cross_cell_counts,
        "subset_replicates": args.subset_replicates,
        "actual_subset_replicates": replicate_counts,
        "two_target_subset_policy": "exhaustive all C(12,2)=66 pairs per held cell",
        "larger_subset_policy": "deterministic unique subsets per held cell",
        "reference_count": args.reference_count,
        "folds": args.folds,
        "nominal_fpr": NOMINAL_FPR,
        "pooled_n_member": pooled_n_member,
        "pooled_n_nonmember": pooled_n_nonmember,
        "seed": args.seed,
        "source_transfer_config": str(source_config.resolve()),
        "source_transfer_config_sha256": source_hash,
        "interval_semantics": (
            "main: median over subsets within seed, then descriptive 5th-95th percentiles over seeds; "
            "subset sensitivity: median over seeds, then descriptive 5th-95th percentiles over subsets"
        ),
    }
    (out_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = build_report(
        config,
        metric_summary,
        calibration_summary,
        contrast_summary,
        contrast_subset_summary,
        strategy_contrast_summary,
        diagnostic_summary,
    )
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"wrote calibration ablation artifacts to {out_dir}")


if __name__ == "__main__":
    main()
