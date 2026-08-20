#!/usr/bin/env python3
"""Test whether a serving channel transfers to unseen QNN targets and cells.

Two strict transfer schemes are implemented:

* ``leave_target_out`` fits the channel and fixed-FPR thresholds on the other
  two target checkpoints in the same structural cell.
* ``leave_cell_out`` fits them on every checkpoint in the other four cells.

The held target/cell is absent from channel estimation and threshold calibration.
Sample-ID folds additionally prevent the same candidate record from appearing in
auxiliary calibration while it is attacked.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from channel_lira.continuous import (
    AffineGaussianChannel,
    balanced_reference_subset,
    channel_diagnostics,
    fit_noise_augmented_distributions,
)
from channel_lira.core import LatentDistributions, attack_metrics, fit_latent_distributions
from experiments.channel_lira_circuit_pilot import (
    CellData,
    NOMINAL_FPRS,
    REFERENCE_ATTACKS,
    ConditionData,
    conservative_threshold,
    discover_cells,
    load_cell,
    parse_int_list,
    sample_id_folds,
    score_reference_attacks,
    stable_int,
    summarize,
    write_csv,
)


SCHEMES = ("leave_target_out", "leave_cell_out")


@dataclass(frozen=True)
class ReferenceBundle:
    scores: np.ndarray
    inclusion: np.ndarray
    latent: LatentDistributions


@dataclass
class TransferBlock:
    scheme: str
    cell: str
    target_ids: tuple[str, ...]
    mode: str
    shots: int
    simulator_seeds: tuple[int, ...]
    membership: np.ndarray
    scores: dict[tuple[str, int], np.ndarray]
    decisions: dict[tuple[str, int, float], np.ndarray]


def reference_bundles(
    cell: CellData,
    reference_counts: list[int],
    variance_shrinkage: float,
) -> dict[int, ReferenceBundle]:
    output = {}
    for count in reference_counts:
        selected = balanced_reference_subset(cell.inclusion, count)
        scores = cell.reference_scores[selected]
        inclusion = cell.inclusion[selected]
        output[count] = ReferenceBundle(
            scores=scores,
            inclusion=inclusion,
            latent=fit_latent_distributions(
                scores,
                inclusion,
                variance_shrinkage=variance_shrinkage,
            ),
        )
    return output


def initialize_outputs(
    shape: tuple[int, int, int], reference_counts: list[int]
) -> tuple[
    dict[tuple[str, int], np.ndarray],
    dict[tuple[str, int, float], np.ndarray],
]:
    scores: dict[tuple[str, int], np.ndarray] = {
        ("loss_mia", 0): np.full(shape, np.nan, dtype=np.float64)
    }
    decisions = {
        ("loss_mia", 0, nominal): np.zeros(shape, dtype=bool)
        for nominal in NOMINAL_FPRS
    }
    for count in reference_counts:
        for attack in REFERENCE_ATTACKS:
            scores[(attack, count)] = np.full(shape, np.nan, dtype=np.float64)
            for nominal in NOMINAL_FPRS:
                decisions[(attack, count, nominal)] = np.zeros(shape, dtype=bool)
    return scores, decisions


def broadcast_membership(membership: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    return np.broadcast_to(membership[:, None, :], shape)


def augmented_model(
    bundle: ReferenceBundle,
    channel: AffineGaussianChannel,
    *,
    draws: int,
    variance_shrinkage: float,
    seed: int,
) -> LatentDistributions:
    return fit_noise_augmented_distributions(
        bundle.scores,
        bundle.inclusion,
        channel,
        draws=draws,
        rng=np.random.default_rng(seed),
        variance_shrinkage=variance_shrinkage,
    )


def diagnostic_record(
    *,
    scheme: str,
    heldout_scope: str,
    heldout_id: str,
    calibration_ids: list[str],
    mode: str,
    shots: int,
    fold: int,
    channel: AffineGaussianChannel,
    train_exact: np.ndarray,
    train_observed: np.ndarray,
    test_exact: np.ndarray,
    test_observed: np.ndarray,
) -> dict[str, object]:
    return {
        "scheme": scheme,
        "heldout_scope": heldout_scope,
        "heldout_id": heldout_id,
        "calibration_ids": ";".join(calibration_ids),
        "mode": mode,
        "shots": shots,
        "fold": fold,
        "n_train_public_nonmember_pairs": len(train_exact),
        "n_test_heldout_nonmember_pairs": len(test_exact),
        "intercept": channel.intercept,
        "slope": channel.slope,
        "scale": channel.scale,
        **{
            f"train_{key}": value
            for key, value in channel_diagnostics(train_exact, train_observed, channel).items()
        },
        **{
            f"test_{key}": value
            for key, value in channel_diagnostics(test_exact, test_observed, channel).items()
        },
    }


def evaluate_leave_target_out(
    cell: CellData,
    condition: ConditionData,
    bundles: dict[int, ReferenceBundle],
    *,
    mode: str,
    shots: int,
    folds: int,
    noise_augmentation_draws: int,
    variance_shrinkage: float,
    seed: int,
) -> tuple[TransferBlock, list[dict[str, object]]]:
    shape = condition.observed_scores.shape
    reference_counts = sorted(bundles)
    scores, decisions = initialize_outputs(shape, reference_counts)
    assignment = sample_id_folds(cell.sample_ids, folds, seed)
    diagnostics: list[dict[str, object]] = []

    for held_target in range(shape[0]):
        auxiliary_targets = np.asarray(
            [index for index in range(shape[0]) if index != held_target], dtype=np.int64
        )
        auxiliary_observed = condition.observed_scores[auxiliary_targets]
        auxiliary_exact = np.broadcast_to(
            cell.exact_scores[auxiliary_targets, None, :], auxiliary_observed.shape
        )
        auxiliary_membership = broadcast_membership(
            cell.memberships[auxiliary_targets], auxiliary_observed.shape
        )
        held_observed = condition.observed_scores[held_target : held_target + 1]
        held_exact = np.broadcast_to(
            cell.exact_scores[held_target : held_target + 1, None, :], held_observed.shape
        )
        held_membership = broadcast_membership(
            cell.memberships[held_target : held_target + 1], held_observed.shape
        )
        for fold in range(folds):
            held_candidates = assignment == fold
            calibration_candidates = ~held_candidates
            calibration_mask = np.broadcast_to(
                calibration_candidates[None, None, :], auxiliary_observed.shape
            ) & (auxiliary_membership == 0)
            test_nonmember_mask = np.broadcast_to(
                held_candidates[None, None, :], held_observed.shape
            ) & (held_membership == 0)
            channel = AffineGaussianChannel.fit(
                auxiliary_exact[calibration_mask], auxiliary_observed[calibration_mask]
            )
            diagnostics.append(diagnostic_record(
                scheme="leave_target_out",
                heldout_scope="target",
                heldout_id=cell.target_ids[held_target],
                calibration_ids=[cell.target_ids[index] for index in auxiliary_targets],
                mode=mode,
                shots=shots,
                fold=fold,
                channel=channel,
                train_exact=auxiliary_exact[calibration_mask],
                train_observed=auxiliary_observed[calibration_mask],
                test_exact=held_exact[test_nonmember_mask],
                test_observed=held_observed[test_nonmember_mask],
            ))

            held_loss = -condition.losses[held_target : held_target + 1]
            auxiliary_loss = -condition.losses[auxiliary_targets]
            scores[("loss_mia", 0)][held_target, :, held_candidates] = (
                held_loss[0, :, held_candidates]
            )
            for nominal in NOMINAL_FPRS:
                threshold = conservative_threshold(auxiliary_loss[calibration_mask], nominal)
                decisions[("loss_mia", 0, nominal)][held_target, :, held_candidates] = (
                    held_loss[0, :, held_candidates] > threshold
                )

            for count, bundle in bundles.items():
                augmented = augmented_model(
                    bundle,
                    channel,
                    draws=noise_augmentation_draws,
                    variance_shrinkage=variance_shrinkage,
                    seed=seed + stable_int(
                        f"loto|{cell.name}|{cell.target_ids[held_target]}|{mode}|{shots}|{fold}|{count}"
                    ) % 1_000_000_000,
                )
                held_scores = score_reference_attacks(
                    held_observed,
                    bundle.latent,
                    bundle.scores,
                    bundle.inclusion,
                    channel,
                    augmented,
                )
                auxiliary_scores = score_reference_attacks(
                    auxiliary_observed,
                    bundle.latent,
                    bundle.scores,
                    bundle.inclusion,
                    channel,
                    augmented,
                )
                for attack in REFERENCE_ATTACKS:
                    scores[(attack, count)][held_target, :, held_candidates] = (
                        held_scores[attack][0, :, held_candidates]
                    )
                    for nominal in NOMINAL_FPRS:
                        threshold = conservative_threshold(
                            auxiliary_scores[attack][calibration_mask], nominal
                        )
                        decisions[(attack, count, nominal)][held_target, :, held_candidates] = (
                            held_scores[attack][0, :, held_candidates] > threshold
                        )

    if any(not np.isfinite(values).all() for values in scores.values()):
        raise RuntimeError(f"Incomplete leave-target-out scores for {cell.name} {mode} {shots}")
    return TransferBlock(
        scheme="leave_target_out",
        cell=cell.name,
        target_ids=cell.target_ids,
        mode=mode,
        shots=shots,
        simulator_seeds=condition.simulator_seeds,
        membership=cell.memberships,
        scores=scores,
        decisions=decisions,
    ), diagnostics


def evaluate_leave_cell_out(
    held_cell: CellData,
    auxiliary_cells: list[CellData],
    bundles_by_cell: dict[str, dict[int, ReferenceBundle]],
    *,
    mode: str,
    shots: int,
    folds: int,
    noise_augmentation_draws: int,
    variance_shrinkage: float,
    seed: int,
) -> tuple[TransferBlock, list[dict[str, object]]]:
    condition = held_cell.conditions[(mode, shots)]
    shape = condition.observed_scores.shape
    reference_counts = sorted(bundles_by_cell[held_cell.name])
    scores, decisions = initialize_outputs(shape, reference_counts)
    assignment = sample_id_folds(held_cell.sample_ids, folds, seed)
    diagnostics: list[dict[str, object]] = []

    for fold in range(folds):
        held_candidates = assignment == fold
        calibration_candidates = ~held_candidates
        train_exact_parts = []
        train_observed_parts = []
        for auxiliary in auxiliary_cells:
            auxiliary_condition = auxiliary.conditions[(mode, shots)]
            auxiliary_shape = auxiliary_condition.observed_scores.shape
            auxiliary_membership = broadcast_membership(
                auxiliary.memberships, auxiliary_shape
            )
            calibration_mask = np.broadcast_to(
                calibration_candidates[None, None, :], auxiliary_shape
            ) & (auxiliary_membership == 0)
            auxiliary_exact = np.broadcast_to(
                auxiliary.exact_scores[:, None, :], auxiliary_shape
            )
            train_exact_parts.append(auxiliary_exact[calibration_mask])
            train_observed_parts.append(auxiliary_condition.observed_scores[calibration_mask])
        train_exact = np.concatenate(train_exact_parts)
        train_observed = np.concatenate(train_observed_parts)
        channel = AffineGaussianChannel.fit(train_exact, train_observed)

        held_exact = np.broadcast_to(held_cell.exact_scores[:, None, :], shape)
        held_membership = broadcast_membership(held_cell.memberships, shape)
        test_nonmember_mask = np.broadcast_to(
            held_candidates[None, None, :], shape
        ) & (held_membership == 0)
        diagnostics.append(diagnostic_record(
            scheme="leave_cell_out",
            heldout_scope="cell",
            heldout_id=held_cell.name,
            calibration_ids=[cell.name for cell in auxiliary_cells],
            mode=mode,
            shots=shots,
            fold=fold,
            channel=channel,
            train_exact=train_exact,
            train_observed=train_observed,
            test_exact=held_exact[test_nonmember_mask],
            test_observed=condition.observed_scores[test_nonmember_mask],
        ))

        held_mask = np.broadcast_to(held_candidates[None, None, :], shape)
        held_loss = -condition.losses
        scores[("loss_mia", 0)][held_mask] = held_loss[held_mask]
        auxiliary_loss_null = []
        for auxiliary in auxiliary_cells:
            auxiliary_condition = auxiliary.conditions[(mode, shots)]
            auxiliary_shape = auxiliary_condition.observed_scores.shape
            null_mask = np.broadcast_to(
                calibration_candidates[None, None, :], auxiliary_shape
            ) & (broadcast_membership(auxiliary.memberships, auxiliary_shape) == 0)
            auxiliary_loss_null.append((-auxiliary_condition.losses)[null_mask])
        loss_null = np.concatenate(auxiliary_loss_null)
        for nominal in NOMINAL_FPRS:
            threshold = conservative_threshold(loss_null, nominal)
            decisions[("loss_mia", 0, nominal)][held_mask] = held_loss[held_mask] > threshold

        for count in reference_counts:
            held_bundle = bundles_by_cell[held_cell.name][count]
            held_augmented = augmented_model(
                held_bundle,
                channel,
                draws=noise_augmentation_draws,
                variance_shrinkage=variance_shrinkage,
                seed=seed + stable_int(
                    f"loco|held|{held_cell.name}|{mode}|{shots}|{fold}|{count}"
                ) % 1_000_000_000,
            )
            held_scores = score_reference_attacks(
                condition.observed_scores,
                held_bundle.latent,
                held_bundle.scores,
                held_bundle.inclusion,
                channel,
                held_augmented,
            )
            null_by_attack: dict[str, list[np.ndarray]] = {
                attack: [] for attack in REFERENCE_ATTACKS
            }
            for auxiliary in auxiliary_cells:
                auxiliary_condition = auxiliary.conditions[(mode, shots)]
                auxiliary_shape = auxiliary_condition.observed_scores.shape
                null_mask = np.broadcast_to(
                    calibration_candidates[None, None, :], auxiliary_shape
                ) & (broadcast_membership(auxiliary.memberships, auxiliary_shape) == 0)
                auxiliary_bundle = bundles_by_cell[auxiliary.name][count]
                auxiliary_augmented = augmented_model(
                    auxiliary_bundle,
                    channel,
                    draws=noise_augmentation_draws,
                    variance_shrinkage=variance_shrinkage,
                    seed=seed + stable_int(
                        f"loco|aux|{held_cell.name}|{auxiliary.name}|{mode}|{shots}|{fold}|{count}"
                    ) % 1_000_000_000,
                )
                auxiliary_scores = score_reference_attacks(
                    auxiliary_condition.observed_scores,
                    auxiliary_bundle.latent,
                    auxiliary_bundle.scores,
                    auxiliary_bundle.inclusion,
                    channel,
                    auxiliary_augmented,
                )
                for attack in REFERENCE_ATTACKS:
                    null_by_attack[attack].append(auxiliary_scores[attack][null_mask])
            for attack in REFERENCE_ATTACKS:
                scores[(attack, count)][held_mask] = held_scores[attack][held_mask]
                null_scores = np.concatenate(null_by_attack[attack])
                for nominal in NOMINAL_FPRS:
                    threshold = conservative_threshold(null_scores, nominal)
                    decisions[(attack, count, nominal)][held_mask] = (
                        held_scores[attack][held_mask] > threshold
                    )

    if any(not np.isfinite(values).all() for values in scores.values()):
        raise RuntimeError(f"Incomplete leave-cell-out scores for {held_cell.name} {mode} {shots}")
    return TransferBlock(
        scheme="leave_cell_out",
        cell=held_cell.name,
        target_ids=held_cell.target_ids,
        mode=mode,
        shots=shots,
        simulator_seeds=condition.simulator_seeds,
        membership=held_cell.memberships,
        scores=scores,
        decisions=decisions,
    ), diagnostics


def metric_record(labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    return {
        "n_member": int(labels.sum()),
        "n_nonmember": int(len(labels) - labels.sum()),
        **attack_metrics(labels, scores),
    }


def build_metric_rows(blocks: list[TransferBlock]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    overall: dict[tuple[object, ...], list[tuple[np.ndarray, np.ndarray]]] = {}
    for block in blocks:
        for (attack, reference_count), values in block.scores.items():
            for simulator_index, simulator_seed in enumerate(block.simulator_seeds):
                labels_cell = block.membership.reshape(-1)
                scores_cell = values[:, simulator_index, :].reshape(-1)
                base = {
                    "scheme": block.scheme,
                    "mode": block.mode,
                    "shots": block.shots,
                    "simulator_seed": simulator_seed,
                    "attack": attack,
                    "reference_count": reference_count,
                }
                rows.append({
                    "scope": "cell", "scope_id": block.cell, **base,
                    **metric_record(labels_cell, scores_cell),
                })
                for target, target_id in enumerate(block.target_ids):
                    rows.append({
                        "scope": "target", "scope_id": target_id, **base,
                        **metric_record(block.membership[target], values[target, simulator_index]),
                    })
                key = (
                    block.scheme, block.mode, block.shots, simulator_seed,
                    attack, reference_count,
                )
                overall.setdefault(key, []).append((labels_cell, scores_cell))
    for key, parts in sorted(overall.items(), key=lambda item: tuple(map(str, item[0]))):
        labels = np.concatenate([part[0] for part in parts])
        scores = np.concatenate([part[1] for part in parts])
        scheme, mode, shots, simulator_seed, attack, reference_count = key
        rows.append({
            "scope": "overall", "scope_id": "all_cells", "scheme": scheme,
            "mode": mode, "shots": shots, "simulator_seed": simulator_seed,
            "attack": attack, "reference_count": reference_count,
            **metric_record(labels, scores),
        })
    return rows


def build_calibration_rows(blocks: list[TransferBlock]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    overall: dict[tuple[object, ...], list[tuple[np.ndarray, np.ndarray]]] = {}
    for block in blocks:
        for (attack, reference_count, nominal), decisions in block.decisions.items():
            for simulator_index, simulator_seed in enumerate(block.simulator_seeds):
                base = {
                    "scheme": block.scheme, "mode": block.mode, "shots": block.shots,
                    "simulator_seed": simulator_seed, "attack": attack,
                    "reference_count": reference_count, "nominal_fpr": nominal,
                }
                labels_cell = block.membership.reshape(-1)
                selected_cell = decisions[:, simulator_index, :].reshape(-1)
                nonmember = labels_cell == 0
                member = labels_cell == 1
                rows.append({
                    "scope": "cell", "scope_id": block.cell, **base,
                    "n_member": int(member.sum()), "n_nonmember": int(nonmember.sum()),
                    "actual_fpr": float(selected_cell[nonmember].mean()),
                    "calibrated_tpr": float(selected_cell[member].mean()),
                })
                for target, target_id in enumerate(block.target_ids):
                    labels = block.membership[target]
                    selected = decisions[target, simulator_index]
                    target_nonmember = labels == 0
                    target_member = labels == 1
                    rows.append({
                        "scope": "target", "scope_id": target_id, **base,
                        "n_member": int(target_member.sum()),
                        "n_nonmember": int(target_nonmember.sum()),
                        "actual_fpr": float(selected[target_nonmember].mean()),
                        "calibrated_tpr": float(selected[target_member].mean()),
                    })
                key = (
                    block.scheme, block.mode, block.shots, simulator_seed,
                    attack, reference_count, nominal,
                )
                overall.setdefault(key, []).append((labels_cell, selected_cell))
    for key, parts in sorted(overall.items(), key=lambda item: tuple(map(str, item[0]))):
        labels = np.concatenate([part[0] for part in parts])
        selected = np.concatenate([part[1] for part in parts])
        nonmember = labels == 0
        member = labels == 1
        scheme, mode, shots, simulator_seed, attack, reference_count, nominal = key
        rows.append({
            "scope": "overall", "scope_id": "all_cells", "scheme": scheme,
            "mode": mode, "shots": shots, "simulator_seed": simulator_seed,
            "attack": attack, "reference_count": reference_count,
            "nominal_fpr": nominal,
            "n_member": int(member.sum()), "n_nonmember": int(nonmember.sum()),
            "actual_fpr": float(selected[nonmember].mean()),
            "calibrated_tpr": float(selected[member].mean()),
        })
    return rows


def paired_auc_rows(
    metrics: list[dict[str, object]], maximum_references: int
) -> list[dict[str, object]]:
    indexed = {
        (
            str(row["scheme"]), str(row["mode"]), int(row["shots"]),
            int(row["simulator_seed"]), str(row["attack"]), int(row["reference_count"]),
        ): row
        for row in metrics if row["scope"] == "overall"
    }
    comparators = {
        "affine_minus_loss": ("loss_mia", 0),
        "affine_minus_mismatched_lira": ("latent_lira_mismatched", maximum_references),
        "affine_minus_deconvolved_lira": ("deconvolved_lira", maximum_references),
        "affine_minus_empirical_channel": ("empirical_channel_lira", maximum_references),
        "affine_minus_noise_augmented": ("noise_augmented_lira", maximum_references),
    }
    output = []
    base_keys = sorted({key[:4] for key in indexed})
    for scheme, mode, shots, simulator_seed in base_keys:
        affine = indexed[(
            scheme, mode, shots, simulator_seed, "affine_channel_lira", maximum_references
        )]
        for contrast, (attack, count) in comparators.items():
            comparator = indexed[(scheme, mode, shots, simulator_seed, attack, count)]
            output.append({
                "scheme": scheme, "mode": mode, "shots": shots,
                "simulator_seed": simulator_seed, "contrast": contrast,
                "comparator_attack": attack, "reference_count": maximum_references,
                **{
                    f"{metric}_difference": float(affine[metric]) - float(comparator[metric])
                    for metric in ("auc", "tpr_at_1pct_fpr", "tpr_at_5pct_fpr")
                },
            })
    return output


def calibrated_contrast_rows(
    calibration: list[dict[str, object]], maximum_references: int
) -> list[dict[str, object]]:
    indexed = {
        (
            str(row["scheme"]), str(row["mode"]), int(row["shots"]),
            int(row["simulator_seed"]), str(row["attack"]), int(row["reference_count"]),
            float(row["nominal_fpr"]),
        ): row
        for row in calibration if row["scope"] == "overall"
    }
    comparators = {
        "affine_minus_loss": ("loss_mia", 0),
        "affine_minus_mismatched_lira": ("latent_lira_mismatched", maximum_references),
    }
    output = []
    base_keys = sorted({key[:4] + (key[6],) for key in indexed})
    for scheme, mode, shots, simulator_seed, nominal in base_keys:
        affine = indexed[(
            scheme, mode, shots, simulator_seed,
            "affine_channel_lira", maximum_references, nominal,
        )]
        for contrast, (attack, count) in comparators.items():
            comparator = indexed[(
                scheme, mode, shots, simulator_seed, attack, count, nominal,
            )]
            output.append({
                "scheme": scheme, "mode": mode, "shots": shots,
                "simulator_seed": simulator_seed, "nominal_fpr": nominal,
                "contrast": contrast, "comparator_attack": attack,
                "reference_count": maximum_references,
                "actual_fpr_difference": float(affine["actual_fpr"]) - float(comparator["actual_fpr"]),
                "calibrated_tpr_difference": float(affine["calibrated_tpr"]) - float(comparator["calibrated_tpr"]),
            })
    return output


def lookup_summary(
    rows: list[dict[str, object]],
    *,
    scheme: str,
    mode: str,
    shots: int,
    attack: str,
    reference_count: int,
    nominal_fpr: float = -1.0,
) -> dict[str, object]:
    matches = [
        row for row in rows
        if row.get("scope") == "overall"
        and row["scheme"] == scheme and row["mode"] == mode
        and int(row["shots"]) == shots and row["attack"] == attack
        and int(row["reference_count"]) == reference_count
        and (nominal_fpr < 0 or float(row["nominal_fpr"]) == nominal_fpr)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Missing summary for {(scheme, mode, shots, attack, reference_count, nominal_fpr)}"
        )
    return matches[0]


def build_report(
    config: dict[str, object],
    metric_summary: list[dict[str, object]],
    calibration_summary: list[dict[str, object]],
    auc_contrast_summary: list[dict[str, object]],
    calibrated_contrast_summary: list[dict[str, object]],
    diagnostic_overall: list[dict[str, object]],
) -> str:
    mode = "noisy_shot"
    high = max(config["shots"])
    count = max(config["reference_counts"])
    attacks = (
        ("loss_mia", 0),
        ("latent_lira_mismatched", count),
        ("deconvolved_lira", count),
        ("affine_channel_lira", count),
        ("empirical_channel_lira", count),
        ("noise_augmented_lira", count),
    )
    auc_lines = []
    calibration_lines = []
    contrast_lines = []
    cell_lines = []
    verdicts = {}
    for scheme in config["schemes"]:
        for attack, reference_count in attacks:
            row = lookup_summary(
                metric_summary, scheme=scheme, mode=mode, shots=high,
                attack=attack, reference_count=reference_count,
            )
            auc_lines.append(
                f"| {scheme} | {attack} | {float(row['auc_median']):.4f} | "
                f"{float(row['tpr_at_1pct_fpr_median']):.4f} | {float(row['tpr_at_5pct_fpr_median']):.4f} |"
            )
        for attack, reference_count in (
            ("loss_mia", 0),
            ("latent_lira_mismatched", count),
            ("affine_channel_lira", count),
        ):
            row = lookup_summary(
                calibration_summary, scheme=scheme, mode=mode, shots=high,
                attack=attack, reference_count=reference_count, nominal_fpr=0.01,
            )
            calibration_lines.append(
                f"| {scheme} | {attack} | {float(row['actual_fpr_median']):.4f} "
                f"[{float(row['actual_fpr_q05']):.4f}, {float(row['actual_fpr_q95']):.4f}] | "
                f"{float(row['calibrated_tpr_median']):.4f} |"
            )
        decisive_auc = {
            str(row["contrast"]): row for row in auc_contrast_summary
            if row["scheme"] == scheme and row["mode"] == mode and int(row["shots"]) == high
        }
        decisive_calibrated = {
            str(row["contrast"]): row for row in calibrated_contrast_summary
            if row["scheme"] == scheme and row["mode"] == mode
            and int(row["shots"]) == high and float(row["nominal_fpr"]) == 0.01
        }
        for contrast in ("affine_minus_loss", "affine_minus_mismatched_lira"):
            auc_row = decisive_auc[contrast]
            calibrated_row = decisive_calibrated[contrast]
            contrast_lines.append(
                f"| {scheme} | {contrast} | {float(auc_row['auc_difference_median']):+.4f} "
                f"[{float(auc_row['auc_difference_q05']):+.4f}, {float(auc_row['auc_difference_q95']):+.4f}] | "
                f"{float(calibrated_row['calibrated_tpr_difference_median']):+.4f} "
                f"[{float(calibrated_row['calibrated_tpr_difference_q05']):+.4f}, "
                f"{float(calibrated_row['calibrated_tpr_difference_q95']):+.4f}] |"
            )
        verdicts[scheme] = (
            float(decisive_auc["affine_minus_mismatched_lira"]["auc_difference_q05"]) > 0
            and float(decisive_calibrated["affine_minus_mismatched_lira"]["calibrated_tpr_difference_q05"]) > 0
        )
        for cell in config["cells"]:
            cell_rows = [
                row for row in metric_summary
                if row["scope"] == "cell" and row["scope_id"] == cell
                and row["scheme"] == scheme and row["mode"] == mode
                and int(row["shots"]) == high
            ]
            indexed = {
                (str(row["attack"]), int(row["reference_count"])): float(row["auc_median"])
                for row in cell_rows
            }
            affine = indexed[("affine_channel_lira", count)]
            cell_lines.append(
                f"| {scheme} | {cell} | {affine - indexed[('loss_mia', 0)]:+.4f} | "
                f"{affine - indexed[('latent_lira_mismatched', count)]:+.4f} |"
            )

    overall_verdict = (
        "YES — target-level transfer survives the strict holdout"
        if verdicts.get("leave_target_out", False)
        else "NO — target-level transfer does not yet survive the strict holdout"
    )
    if "leave_cell_out" in verdicts:
        overall_verdict += (
            "; cell-level transfer also passes"
            if verdicts["leave_cell_out"]
            else "; cell-level transfer remains unresolved"
        )
    diagnostic_lines = [
        f"| {row['scheme']} | {int(row['shots'])} | {float(row['slope_median']):.3f} | "
        f"{float(row['test_r_squared_median']):.3f} | {float(row['test_coverage_90pct_median']):.3f} |"
        for row in diagnostic_overall
    ]
    return f"""# ChannelLiRA unseen-target transfer report

## Verdict

**{overall_verdict}.**

This test removes the strongest Phase-2 threat-model objection. Under
`leave_target_out`, the attacked checkpoint is absent from both channel fitting and
threshold calibration. Under `leave_cell_out`, every checkpoint with the attacked
circuit structure is absent. Sample-ID folds also remove the attacked candidate ID
from auxiliary calibration records. Exact outputs of held targets are used only for
offline fit diagnostics, never for attack scoring or thresholds.

## Pooled results at {high} IBM-derived noisy shots

The ROC operating points below are descriptive empirical-ROC quantities; they are
not the cross-fitted operational thresholds in the next table.

| Transfer scheme | Attack | AUC | ROC TPR @ 1% FPR | ROC TPR @ 5% FPR |
|---|---|---:|---:|---:|
{chr(10).join(auc_lines)}

## Cross-target thresholds at nominal 1% FPR

| Transfer scheme | Attack | Actual FPR [5%, 95%] | Calibrated TPR |
|---|---|---:|---:|
{chr(10).join(calibration_lines)}

## Paired transfer contrasts

| Transfer scheme | Contrast | AUC difference [5%, 95%] | Calibrated-TPR difference [5%, 95%] |
|---|---|---:|---:|
{chr(10).join(contrast_lines)}

Intervals are 5th–95th percentiles across the ten retained simulator seeds, not
record/model-clustered confidence intervals. A transfer scheme passes the automatic
gate only when the 5th percentiles of both the AUC and calibrated-TPR contrasts over
mismatched LiRA are positive at {high} shots.

## Structural heterogeneity at {high} shots

| Transfer scheme | Cell | Affine minus loss AUC | Affine minus mismatched-LiRA AUC |
|---|---|---:|---:|
{chr(10).join(cell_lines)}

## Held-out channel diagnostics

| Transfer scheme | Shots | Median slope | Median held-out R² | Median 90% coverage |
|---|---:|---:|---:|---:|
{chr(10).join(diagnostic_lines)}

## Interpretation limits

- These are simulator-to-simulator transfers under one IBM-derived frozen noise
  snapshot, not transfers to quantum hardware or a new calibration date.
- Each structural cell has only three target checkpoints. Leave-target-out therefore
  calibrates from two auxiliary checkpoints per rotation.
- Reference IN/OUT distributions still come from the held cell's exact 16-model bank;
  this test transfers the serving channel and thresholds, not the LiRA reference bank.
- The noise-augmented baseline draws from the fitted Gaussian channel. It does not
  represent additional noisy QNN circuit executions.
- The target-cross-fitted learned MIA is omitted here because it trains on labeled
  outputs from the attacked target and therefore violates these transfer protocols.
  A matched shadow-model learned MIA remains an extended-study requirement.
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
    parser.add_argument("--schemes", default=",".join(SCHEMES))
    parser.add_argument("--modes", default="noisy_shot")
    parser.add_argument("--shots", default="128,512,1024")
    parser.add_argument("--reference-counts", default="16")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--noise-augmentation-draws", type=int, default=32)
    parser.add_argument("--variance-shrinkage", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("channel_lira_results/transfer_phase3")
    )
    args = parser.parse_args()
    schemes = [value.strip() for value in args.schemes.split(",") if value.strip()]
    unknown_schemes = sorted(set(schemes) - set(SCHEMES))
    if unknown_schemes:
        raise ValueError(f"Unknown transfer schemes: {unknown_schemes}")
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    shots = parse_int_list(args.shots)
    reference_counts = parse_int_list(args.reference_counts)
    noisy_root = args.noisy_root.resolve()
    reference_root = args.reference_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = [
        load_cell(noisy_root, reference_root, name)
        for name in discover_cells(noisy_root, args.cells)
    ]
    if len(cells) < 2 and "leave_cell_out" in schemes:
        raise ValueError("leave_cell_out requires at least two cells")
    baseline_ids = cells[0].sample_ids
    if any(not np.array_equal(cell.sample_ids, baseline_ids) for cell in cells[1:]):
        raise ValueError("Strict transfer currently requires aligned sample IDs across cells")
    requested_conditions = {(mode, shot) for mode in modes for shot in shots}
    for cell in cells:
        missing = requested_conditions - set(cell.conditions)
        if missing:
            raise ValueError(f"{cell.name}: missing conditions {sorted(missing)}")
    bundles_by_cell = {
        cell.name: reference_bundles(cell, reference_counts, args.variance_shrinkage)
        for cell in cells
    }

    blocks: list[TransferBlock] = []
    diagnostics: list[dict[str, object]] = []
    for mode in modes:
        for shots_value in shots:
            if "leave_target_out" in schemes:
                for cell in cells:
                    block, rows = evaluate_leave_target_out(
                        cell,
                        cell.conditions[(mode, shots_value)],
                        bundles_by_cell[cell.name],
                        mode=mode,
                        shots=shots_value,
                        folds=args.folds,
                        noise_augmentation_draws=args.noise_augmentation_draws,
                        variance_shrinkage=args.variance_shrinkage,
                        seed=args.seed,
                    )
                    blocks.append(block)
                    diagnostics.extend(rows)
                    print(f"completed leave_target_out {cell.name} {mode} {shots_value}", flush=True)
            if "leave_cell_out" in schemes:
                for held_cell in cells:
                    auxiliary_cells = [cell for cell in cells if cell.name != held_cell.name]
                    block, rows = evaluate_leave_cell_out(
                        held_cell,
                        auxiliary_cells,
                        bundles_by_cell,
                        mode=mode,
                        shots=shots_value,
                        folds=args.folds,
                        noise_augmentation_draws=args.noise_augmentation_draws,
                        variance_shrinkage=args.variance_shrinkage,
                        seed=args.seed,
                    )
                    blocks.append(block)
                    diagnostics.extend(rows)
                    print(f"completed leave_cell_out {held_cell.name} {mode} {shots_value}", flush=True)

    metric_rows = build_metric_rows(blocks)
    calibration_rows = build_calibration_rows(blocks)
    auc_rows = paired_auc_rows(metric_rows, max(reference_counts))
    calibrated_rows = calibrated_contrast_rows(calibration_rows, max(reference_counts))
    metric_summary = summarize(
        metric_rows,
        ("scope", "scope_id", "scheme", "mode", "shots", "attack", "reference_count"),
        ("auc", "advantage", "tpr_at_0_1pct_fpr", "tpr_at_1pct_fpr", "tpr_at_5pct_fpr"),
    )
    calibration_summary = summarize(
        calibration_rows,
        (
            "scope", "scope_id", "scheme", "mode", "shots", "attack",
            "reference_count", "nominal_fpr",
        ),
        ("actual_fpr", "calibrated_tpr"),
    )
    auc_summary = summarize(
        auc_rows,
        ("scheme", "mode", "shots", "contrast", "comparator_attack", "reference_count"),
        ("auc_difference", "tpr_at_1pct_fpr_difference", "tpr_at_5pct_fpr_difference"),
    )
    calibrated_summary = summarize(
        calibrated_rows,
        (
            "scheme", "mode", "shots", "nominal_fpr", "contrast",
            "comparator_attack", "reference_count",
        ),
        ("actual_fpr_difference", "calibrated_tpr_difference"),
    )
    diagnostic_summary = summarize(
        diagnostics,
        ("scheme", "heldout_scope", "heldout_id", "mode", "shots"),
        (
            "intercept", "slope", "scale", "train_r_squared", "test_r_squared",
            "test_rmse", "test_coverage_90pct", "test_residual_skew",
            "test_residual_excess_kurtosis",
        ),
    )
    diagnostic_overall = summarize(
        diagnostics,
        ("scheme", "mode", "shots"),
        ("intercept", "slope", "scale", "train_r_squared", "test_r_squared", "test_coverage_90pct"),
    )
    write_csv(out_dir / "metrics_raw.csv", metric_rows)
    write_csv(out_dir / "metrics_summary.csv", metric_summary)
    write_csv(out_dir / "calibration_raw.csv", calibration_rows)
    write_csv(out_dir / "calibration_summary.csv", calibration_summary)
    write_csv(out_dir / "paired_auc_contrasts_raw.csv", auc_rows)
    write_csv(out_dir / "paired_auc_contrasts_summary.csv", auc_summary)
    write_csv(out_dir / "calibrated_contrasts_raw.csv", calibrated_rows)
    write_csv(out_dir / "calibrated_contrasts_summary.csv", calibrated_summary)
    write_csv(out_dir / "channel_diagnostics_raw.csv", diagnostics)
    write_csv(out_dir / "channel_diagnostics_summary.csv", diagnostic_summary)
    write_csv(out_dir / "channel_diagnostics_overall_summary.csv", diagnostic_overall)

    source_config = ROOT / "channel_lira_results/circuit_phase2/experiment_config.json"
    source_hash = hashlib.sha256(source_config.read_bytes()).hexdigest()
    config: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schemes": schemes,
        "cells": [cell.name for cell in cells],
        "n_cells": len(cells),
        "target_ids": [target for cell in cells for target in cell.target_ids],
        "n_targets": sum(len(cell.target_ids) for cell in cells),
        "modes": modes,
        "shots": shots,
        "reference_counts": reference_counts,
        "folds": args.folds,
        "noise_augmentation_draws": args.noise_augmentation_draws,
        "variance_shrinkage": args.variance_shrinkage,
        "seed": args.seed,
        "candidate_exclusion": (
            "attacked sample-ID fold excluded from every auxiliary target/cell channel fit and threshold"
        ),
        "heldout_exact_output_usage": "diagnostics only; absent from attack scores and thresholds",
        "source_phase2_config": str(source_config.resolve()),
        "source_phase2_config_sha256": source_hash,
    }
    (out_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = build_report(
        config,
        metric_summary,
        calibration_summary,
        auc_summary,
        calibrated_summary,
        diagnostic_overall,
    )
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"wrote transfer artifacts to {out_dir}")


if __name__ == "__main__":
    main()
