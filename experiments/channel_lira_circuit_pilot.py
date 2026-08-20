#!/usr/bin/env python3
"""Evaluate ChannelLiRA on retained circuit-level ideal/noisy QNN outputs.

This stage does not rerun training or quantum simulation.  It pairs retained
exact outputs with finite-shot Aer outputs, including an IBM-Kingston-derived
noise model, and cross-fits a continuous serving channel on disjoint public
nonmember records.  Every attacked sample ID is excluded from its own channel
fit and empirical-null threshold calibration.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from channel_lira.continuous import (
    AffineGaussianChannel,
    affine_channel_lira_score,
    balanced_reference_subset,
    channel_diagnostics,
    deconvolved_continuous_lira_score,
    empirical_channel_lira_score,
    fit_noise_augmented_distributions,
)
from channel_lira.core import (
    LatentDistributions,
    attack_metrics,
    fit_latent_distributions,
    latent_lira_score,
    logit,
)


REFERENCE_ATTACKS = (
    "latent_lira_mismatched",
    "deconvolved_lira",
    "affine_channel_lira",
    "empirical_channel_lira",
    "noise_augmented_lira",
)
REFERENCE_FREE_ATTACKS = (
    "loss_mia",
    "learned_logistic_pv_stats_target_crossfit_upper_bound",
)
FEATURE_COLUMNS = (
    "p_0", "p_1", "p_2", "p_3", "loss", "entropy", "confidence", "margin", "correctness"
)
NOMINAL_FPRS = (0.01, 0.05)


@dataclass(frozen=True)
class ConditionData:
    simulator_seeds: tuple[int, ...]
    observed_scores: np.ndarray  # target, simulator seed, candidate
    losses: np.ndarray
    features: np.ndarray  # target, simulator seed, candidate, feature


@dataclass(frozen=True)
class CellData:
    name: str
    target_ids: tuple[str, ...]
    sample_ids: np.ndarray
    reference_scores: np.ndarray
    inclusion: np.ndarray
    memberships: np.ndarray
    exact_scores: np.ndarray
    exact_losses: np.ndarray
    exact_features: np.ndarray
    conditions: dict[tuple[str, int], ConditionData]
    source_files: tuple[Path, ...]


@dataclass
class EvaluationBlock:
    cell: str
    target_ids: tuple[str, ...]
    mode: str
    shots: int
    simulator_seeds: tuple[int, ...]
    membership: np.ndarray
    scores: dict[tuple[str, int], np.ndarray]
    decisions: dict[tuple[str, int, float], np.ndarray]


def parse_int_list(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError("Expected a comma-separated list of positive integers")
    return list(dict.fromkeys(values))


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def structural_cell(target_id: str) -> str:
    match = re.fullmatch(r"MNIST_QNN_(.+)_s\d+", target_id)
    if match is None:
        raise ValueError(f"Cannot derive a structural cell from {target_id}")
    return match.group(1)


def row_score(row: dict[str, str]) -> float:
    probability = float(row[f"p_{int(row['label'])}"])
    return float(logit(np.asarray([probability]))[0])


def row_features(row: dict[str, str]) -> np.ndarray:
    return np.asarray([float(row[column]) for column in FEATURE_COLUMNS], dtype=np.float64)


def read_target(path: Path) -> tuple[
    list[dict[str, str]], dict[tuple[str, int, int], list[dict[str, str]]]
]:
    exact: list[dict[str, str]] = []
    observed: dict[tuple[str, int, int], list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not str(row.get("membership_convention", "")).startswith("1=member"):
                raise ValueError(f"Unexpected membership convention in {path}")
            if row["mode"] == "exact":
                exact.append(row)
            else:
                key = (row["mode"], int(row["shots"]), int(row["simulator_seed"]))
                observed.setdefault(key, []).append(row)
    if not exact or not observed:
        raise ValueError(f"Incomplete exact/stochastic output table: {path}")
    return exact, observed


def discover_cells(noisy_root: Path, requested: str) -> list[str]:
    available = sorted({structural_cell(path.parent.name) for path in noisy_root.glob("*/per_sample_predictions.csv")})
    selected = available if requested.strip().lower() == "all" else [
        value.strip() for value in requested.split(",") if value.strip()
    ]
    missing = sorted(set(selected) - set(available))
    if missing:
        raise FileNotFoundError(f"No retained circuit-level outputs for cells: {missing}")
    if not selected:
        raise ValueError("No structural cells selected")
    return selected


def load_cell(noisy_root: Path, reference_root: Path, cell: str) -> CellData:
    prediction_files = tuple(sorted(noisy_root.glob(f"MNIST_QNN_{cell}_s*/per_sample_predictions.csv")))
    if not prediction_files:
        raise FileNotFoundError(f"{cell}: no retained per-sample circuit outputs")
    loaded_targets = [read_target(path) for path in prediction_files]
    target_ids = tuple(path.parent.name for path in prediction_files)
    sample_ids = np.asarray([row["sample_id"] for row in loaded_targets[0][0]], dtype=str)
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError(f"{cell}: duplicate exact sample IDs")

    memberships = []
    exact_scores = []
    exact_losses = []
    exact_features = []
    condition_keys: Optional[set[tuple[str, int, int]]] = None
    observed_maps = []
    for path, (exact_rows, observed) in zip(prediction_files, loaded_targets):
        exact_by_id = {row["sample_id"]: row for row in exact_rows}
        if set(exact_by_id) != set(sample_ids):
            raise ValueError(f"{cell}: target candidate IDs differ in {path}")
        memberships.append([int(exact_by_id[sample]["membership"]) for sample in sample_ids])
        exact_scores.append([row_score(exact_by_id[sample]) for sample in sample_ids])
        exact_losses.append([float(exact_by_id[sample]["loss"]) for sample in sample_ids])
        exact_features.append([row_features(exact_by_id[sample]) for sample in sample_ids])
        mapped = {
            key: {row["sample_id"]: row for row in rows}
            for key, rows in observed.items()
        }
        for key, by_id in mapped.items():
            if set(by_id) != set(sample_ids):
                raise ValueError(f"{cell}: incomplete condition {key} in {path}")
        observed_maps.append(mapped)
        condition_keys = set(mapped) if condition_keys is None else condition_keys & set(mapped)
    assert condition_keys is not None
    if any(set(mapping) != condition_keys for mapping in observed_maps):
        raise ValueError(f"{cell}: target condition grids differ")

    grouped_keys: dict[tuple[str, int], list[int]] = {}
    for mode, shots, simulator_seed in sorted(condition_keys):
        grouped_keys.setdefault((mode, shots), []).append(simulator_seed)
    conditions: dict[tuple[str, int], ConditionData] = {}
    for condition, seeds in sorted(grouped_keys.items()):
        unique_seeds = tuple(sorted(set(seeds)))
        observed_scores = np.empty((len(target_ids), len(unique_seeds), len(sample_ids)))
        losses = np.empty_like(observed_scores)
        features = np.empty((*observed_scores.shape, len(FEATURE_COLUMNS)))
        for target_index, mapping in enumerate(observed_maps):
            for seed_index, simulator_seed in enumerate(unique_seeds):
                rows = mapping[(condition[0], condition[1], simulator_seed)]
                for candidate, sample_id in enumerate(sample_ids):
                    row = rows[str(sample_id)]
                    observed_scores[target_index, seed_index, candidate] = row_score(row)
                    losses[target_index, seed_index, candidate] = float(row["loss"])
                    features[target_index, seed_index, candidate] = row_features(row)
        conditions[condition] = ConditionData(
            simulator_seeds=unique_seeds,
            observed_scores=observed_scores,
            losses=losses,
            features=features,
        )

    reference_files = tuple(sorted((reference_root / cell).glob("reference_*.npz")))
    if len(reference_files) < 4:
        raise ValueError(f"{cell}: found only {len(reference_files)} reference models")
    reference_scores = []
    inclusion = []
    expected_ids: Optional[np.ndarray] = None
    expected_fingerprint: Optional[str] = None
    for path in reference_files:
        with np.load(path, allow_pickle=False) as saved:
            ids = saved["sample_ids"].astype(str)
            fingerprint = str(saved["candidate_fingerprint"])
            if expected_ids is None:
                expected_ids = ids
                expected_fingerprint = fingerprint
            elif not np.array_equal(ids, expected_ids) or fingerprint != expected_fingerprint:
                raise ValueError(f"{cell}: inconsistent reference candidate bank")
            reference_scores.append(saved["scores"].astype(np.float64))
            inclusion.append(saved["inclusion"].astype(bool))
    assert expected_ids is not None
    id_to_reference = {sample_id: index for index, sample_id in enumerate(expected_ids)}
    try:
        candidate_indices = np.asarray([id_to_reference[sample] for sample in sample_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"{cell}: circuit output is absent from the reference candidate bank") from error
    score_array = np.stack(reference_scores)[:, candidate_indices]
    inclusion_array = np.stack(inclusion)[:, candidate_indices]
    metadata_files = tuple(
        candidate
        for prediction in prediction_files
        for candidate in (
            prediction.parent / "run_metadata.json",
            prediction.parent / "backend_noise_metadata.json",
        )
        if candidate.is_file()
    )
    return CellData(
        name=cell,
        target_ids=target_ids,
        sample_ids=sample_ids,
        reference_scores=score_array,
        inclusion=inclusion_array,
        memberships=np.asarray(memberships, dtype=np.int8),
        exact_scores=np.asarray(exact_scores, dtype=np.float64),
        exact_losses=np.asarray(exact_losses, dtype=np.float64),
        exact_features=np.asarray(exact_features, dtype=np.float64),
        conditions=conditions,
        source_files=(*prediction_files, *metadata_files, *reference_files),
    )


def sample_id_folds(sample_ids: np.ndarray, folds: int, seed: int) -> np.ndarray:
    if folds < 2 or folds > len(sample_ids):
        raise ValueError("Invalid fold count")
    order = sorted(
        range(len(sample_ids)),
        key=lambda index: stable_int(f"{seed}|{sample_ids[index]}"),
    )
    assignment = np.empty(len(sample_ids), dtype=np.int64)
    for rank, index in enumerate(order):
        assignment[index] = rank % folds
    return assignment


def stratified_folds(
    membership: np.ndarray, sample_ids: np.ndarray, folds: int, seed: int
) -> np.ndarray:
    assignment = np.empty(len(membership), dtype=np.int64)
    for label in (0, 1):
        indices = np.flatnonzero(membership == label).tolist()
        indices.sort(key=lambda index: stable_int(f"{seed}|{label}|{sample_ids[index]}"))
        for rank, index in enumerate(indices):
            assignment[index] = rank % folds
    return assignment


def fit_logistic_scores(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    *,
    l2: float = 0.05,
    iterations: int = 60,
) -> np.ndarray:
    """Small dependency-free logistic learned-MIA baseline."""
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train = (train_features - mean) / scale
    test = (test_features - mean) / scale
    train = np.column_stack((np.ones(len(train)), train))
    test = np.column_stack((np.ones(len(test)), test))
    labels = np.asarray(train_labels, dtype=np.float64)
    beta = np.zeros(train.shape[1], dtype=np.float64)
    penalty = np.eye(train.shape[1], dtype=np.float64) * float(l2)
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        linear = np.clip(train @ beta, -35.0, 35.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        gradient = train.T @ (probability - labels) / len(labels) + penalty @ beta
        weight = np.maximum(probability * (1.0 - probability), 1e-6)
        hessian = (train.T * weight) @ train / len(labels) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta -= np.clip(step, -5.0, 5.0)
        if float(np.max(np.abs(step))) < 1e-8:
            break
    test_linear = np.clip(test @ beta, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-test_linear))


def learned_crossfit_scores(
    features: np.ndarray,
    membership: np.ndarray,
    sample_ids: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> np.ndarray:
    assignment = stratified_folds(membership, sample_ids, folds, seed)
    scores = np.full(len(membership), np.nan, dtype=np.float64)
    for fold in range(folds):
        test = assignment == fold
        train = ~test
        scores[test] = fit_logistic_scores(
            features[train], membership[train], features[test]
        )
    if not np.isfinite(scores).all():
        raise RuntimeError("Learned cross-fitting did not score every record")
    return scores


def select_model(model: LatentDistributions, indices: np.ndarray) -> LatentDistributions:
    return LatentDistributions(
        mean_in=model.mean_in[indices],
        std_in=model.std_in[indices],
        mean_out=model.mean_out[indices],
        std_out=model.std_out[indices],
    )


def score_reference_attacks(
    observed: np.ndarray,
    latent: LatentDistributions,
    reference_scores: np.ndarray,
    inclusion: np.ndarray,
    channel: AffineGaussianChannel,
    augmented: LatentDistributions,
) -> dict[str, np.ndarray]:
    target_count, simulator_count, candidate_count = observed.shape
    repetitions = target_count * simulator_count
    flattened = observed.reshape(-1)
    repeated_latent = latent.repeated(repetitions)
    repeated_augmented = augmented.repeated(repetitions)
    scores = {
        "latent_lira_mismatched": latent_lira_score(flattened, repeated_latent).reshape(observed.shape),
        "deconvolved_lira": deconvolved_continuous_lira_score(
            flattened, repeated_latent, channel
        ).reshape(observed.shape),
        "affine_channel_lira": affine_channel_lira_score(
            flattened, repeated_latent, channel
        ).reshape(observed.shape),
        "noise_augmented_lira": latent_lira_score(
            flattened, repeated_augmented
        ).reshape(observed.shape),
    }
    empirical = np.empty_like(observed)
    for target in range(target_count):
        for simulator in range(simulator_count):
            empirical[target, simulator] = empirical_channel_lira_score(
                observed[target, simulator], reference_scores, inclusion, channel
            )
    scores["empirical_channel_lira"] = empirical
    if any(value.shape != (target_count, simulator_count, candidate_count) for value in scores.values()):
        raise RuntimeError("Attack score shape mismatch")
    return scores


def conservative_threshold(scores: np.ndarray, nominal_fpr: float) -> float:
    threshold = float(np.quantile(scores, 1.0 - nominal_fpr, method="higher"))
    return float(np.nextafter(threshold, math.inf))


def evaluate_condition(
    cell: CellData,
    mode: str,
    shots: int,
    condition: ConditionData,
    *,
    selected_by_count: dict[int, np.ndarray],
    folds: int,
    noise_augmentation_draws: int,
    variance_shrinkage: float,
    seed: int,
) -> tuple[EvaluationBlock, list[dict[str, object]], dict[int, list[int]]]:
    shape = condition.observed_scores.shape
    assignment = sample_id_folds(cell.sample_ids, folds, seed)
    scores: dict[tuple[str, int], np.ndarray] = {
        ("loss_mia", 0): np.full(shape, np.nan, dtype=np.float64),
        ("learned_logistic_pv_stats_target_crossfit_upper_bound", 0): np.empty(shape),
    }
    decisions: dict[tuple[str, int, float], np.ndarray] = {
        ("loss_mia", 0, nominal): np.zeros(shape, dtype=bool)
        for nominal in NOMINAL_FPRS
    }
    reference_counts = sorted(selected_by_count)
    for count in reference_counts:
        for attack in REFERENCE_ATTACKS:
            scores[(attack, count)] = np.full(shape, np.nan, dtype=np.float64)
            for nominal in NOMINAL_FPRS:
                decisions[(attack, count, nominal)] = np.zeros(shape, dtype=bool)

    for target in range(shape[0]):
        for simulator in range(shape[1]):
            learned_seed = seed + stable_int(
                f"{cell.target_ids[target]}|{mode}|{shots}|{condition.simulator_seeds[simulator]}"
            ) % 1_000_000
            scores[("learned_logistic_pv_stats_target_crossfit_upper_bound", 0)][target, simulator] = (
                learned_crossfit_scores(
                    condition.features[target, simulator],
                    cell.memberships[target],
                    cell.sample_ids,
                    folds=folds,
                    seed=learned_seed,
                )
            )

    diagnostic_rows: list[dict[str, object]] = []
    exact_broadcast = np.broadcast_to(cell.exact_scores[:, None, :], shape)
    membership_broadcast = np.broadcast_to(cell.memberships[:, None, :], shape)
    for fold in range(folds):
        held_candidates = assignment == fold
        training_candidates = ~held_candidates
        training_mask = np.broadcast_to(training_candidates[None, None, :], shape) & (
            membership_broadcast == 0
        )
        heldout_nonmember_mask = np.broadcast_to(held_candidates[None, None, :], shape) & (
            membership_broadcast == 0
        )
        channel = AffineGaussianChannel.fit(
            exact_broadcast[training_mask], condition.observed_scores[training_mask]
        )
        train_diagnostics = channel_diagnostics(
            exact_broadcast[training_mask], condition.observed_scores[training_mask], channel
        )
        test_diagnostics = channel_diagnostics(
            exact_broadcast[heldout_nonmember_mask],
            condition.observed_scores[heldout_nonmember_mask],
            channel,
        )
        diagnostic_rows.append({
            "cell": cell.name,
            "mode": mode,
            "shots": shots,
            "fold": fold,
            "n_train_public_nonmember_pairs": int(training_mask.sum()),
            "n_test_public_nonmember_pairs": int(heldout_nonmember_mask.sum()),
            "intercept": channel.intercept,
            "slope": channel.slope,
            "scale": channel.scale,
            **{f"train_{key}": value for key, value in train_diagnostics.items()},
            **{f"test_{key}": value for key, value in test_diagnostics.items()},
        })

        held_mask = np.broadcast_to(held_candidates[None, None, :], shape)
        loss_scores = -condition.losses
        scores[("loss_mia", 0)][held_mask] = loss_scores[held_mask]
        for nominal in NOMINAL_FPRS:
            threshold = conservative_threshold(loss_scores[training_mask], nominal)
            decisions[("loss_mia", 0, nominal)][held_mask] = loss_scores[held_mask] > threshold

        for count in reference_counts:
            selected = selected_by_count[count]
            reference_scores = cell.reference_scores[selected]
            inclusion = cell.inclusion[selected]
            latent = fit_latent_distributions(
                reference_scores,
                inclusion,
                variance_shrinkage=variance_shrinkage,
            )
            rng = np.random.default_rng(
                seed + stable_int(f"{cell.name}|{mode}|{shots}|{fold}|{count}") % 1_000_000_000
            )
            augmented = fit_noise_augmented_distributions(
                reference_scores,
                inclusion,
                channel,
                draws=noise_augmentation_draws,
                rng=rng,
                variance_shrinkage=variance_shrinkage,
            )
            fold_scores = score_reference_attacks(
                condition.observed_scores,
                latent,
                reference_scores,
                inclusion,
                channel,
                augmented,
            )
            for attack, values in fold_scores.items():
                scores[(attack, count)][held_mask] = values[held_mask]
                for nominal in NOMINAL_FPRS:
                    threshold = conservative_threshold(values[training_mask], nominal)
                    decisions[(attack, count, nominal)][held_mask] = values[held_mask] > threshold

    if any(not np.isfinite(values).all() for values in scores.values()):
        raise RuntimeError(f"{cell.name} {mode} {shots}: incomplete attack scores")
    block = EvaluationBlock(
        cell=cell.name,
        target_ids=cell.target_ids,
        mode=mode,
        shots=shots,
        simulator_seeds=condition.simulator_seeds,
        membership=cell.memberships,
        scores=scores,
        decisions=decisions,
    )
    return block, diagnostic_rows, {count: selected_by_count[count].tolist() for count in reference_counts}


def evaluate_exact(
    cell: CellData,
    *,
    selected_by_count: dict[int, np.ndarray],
    folds: int,
    variance_shrinkage: float,
    seed: int,
) -> EvaluationBlock:
    shape = (len(cell.target_ids), 1, len(cell.sample_ids))
    scores: dict[tuple[str, int], np.ndarray] = {
        ("loss_mia", 0): (-cell.exact_losses)[:, None, :],
        ("learned_logistic_pv_stats_target_crossfit_upper_bound", 0): np.empty(shape),
    }
    for target in range(shape[0]):
        scores[("learned_logistic_pv_stats_target_crossfit_upper_bound", 0)][target, 0] = (
            learned_crossfit_scores(
                cell.exact_features[target],
                cell.memberships[target],
                cell.sample_ids,
                folds=folds,
                seed=seed + stable_int(cell.target_ids[target]) % 1_000_000,
            )
        )
    for count, selected in sorted(selected_by_count.items()):
        model = fit_latent_distributions(
            cell.reference_scores[selected],
            cell.inclusion[selected],
            variance_shrinkage=variance_shrinkage,
        )
        scores[("exact_output_fitted_lira", count)] = latent_lira_score(
            cell.exact_scores.reshape(-1), model.repeated(len(cell.target_ids))
        ).reshape(shape)
    return EvaluationBlock(
        cell=cell.name,
        target_ids=cell.target_ids,
        mode="exact",
        shots=0,
        simulator_seeds=(-1,),
        membership=cell.memberships,
        scores=scores,
        decisions={},
    )


def metric_record(labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    return {
        "n_member": int(labels.sum()),
        "n_nonmember": int(len(labels) - labels.sum()),
        **attack_metrics(labels, scores),
    }


def build_metric_rows(blocks: list[EvaluationBlock]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    overall: dict[tuple[object, ...], list[tuple[np.ndarray, np.ndarray]]] = {}
    for block in blocks:
        for (attack, reference_count), values in block.scores.items():
            for simulator_index, simulator_seed in enumerate(block.simulator_seeds):
                labels_cell = block.membership.reshape(-1)
                scores_cell = values[:, simulator_index, :].reshape(-1)
                base = {
                    "mode": block.mode,
                    "shots": block.shots,
                    "simulator_seed": simulator_seed,
                    "attack": attack,
                    "reference_count": reference_count,
                }
                rows.append({
                    "scope": "cell",
                    "scope_id": block.cell,
                    **base,
                    **metric_record(labels_cell, scores_cell),
                })
                for target_index, target_id in enumerate(block.target_ids):
                    rows.append({
                        "scope": "target",
                        "scope_id": target_id,
                        **base,
                        **metric_record(
                            block.membership[target_index],
                            values[target_index, simulator_index],
                        ),
                    })
                key = (block.mode, block.shots, simulator_seed, attack, reference_count)
                overall.setdefault(key, []).append((labels_cell, scores_cell))
    for key, parts in sorted(overall.items(), key=lambda item: tuple(map(str, item[0]))):
        labels = np.concatenate([part[0] for part in parts])
        scores = np.concatenate([part[1] for part in parts])
        mode, shots, simulator_seed, attack, reference_count = key
        rows.append({
            "scope": "overall",
            "scope_id": "all_cells",
            "mode": mode,
            "shots": shots,
            "simulator_seed": simulator_seed,
            "attack": attack,
            "reference_count": reference_count,
            **metric_record(labels, scores),
        })
    return rows


def build_calibration_rows(blocks: list[EvaluationBlock]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    overall: dict[tuple[object, ...], list[tuple[np.ndarray, np.ndarray]]] = {}
    for block in blocks:
        for (attack, reference_count, nominal), values in block.decisions.items():
            for simulator_index, simulator_seed in enumerate(block.simulator_seeds):
                labels = block.membership.reshape(-1)
                selected = values[:, simulator_index, :].reshape(-1)
                nonmember = labels == 0
                member = labels == 1
                base = {
                    "mode": block.mode,
                    "shots": block.shots,
                    "simulator_seed": simulator_seed,
                    "attack": attack,
                    "reference_count": reference_count,
                    "nominal_fpr": nominal,
                }
                rows.append({
                    "scope": "cell",
                    "scope_id": block.cell,
                    **base,
                    "n_member": int(member.sum()),
                    "n_nonmember": int(nonmember.sum()),
                    "actual_fpr": float(selected[nonmember].mean()),
                    "calibrated_tpr": float(selected[member].mean()),
                })
                key = (block.mode, block.shots, simulator_seed, attack, reference_count, nominal)
                overall.setdefault(key, []).append((labels, selected))
    for key, parts in sorted(overall.items(), key=lambda item: tuple(map(str, item[0]))):
        labels = np.concatenate([part[0] for part in parts])
        selected = np.concatenate([part[1] for part in parts])
        nonmember = labels == 0
        member = labels == 1
        mode, shots, simulator_seed, attack, reference_count, nominal = key
        rows.append({
            "scope": "overall",
            "scope_id": "all_cells",
            "mode": mode,
            "shots": shots,
            "simulator_seed": simulator_seed,
            "attack": attack,
            "reference_count": reference_count,
            "nominal_fpr": nominal,
            "n_member": int(member.sum()),
            "n_nonmember": int(nonmember.sum()),
            "actual_fpr": float(selected[nonmember].mean()),
            "calibrated_tpr": float(selected[member].mean()),
        })
    return rows


def build_contrast_rows(
    metric_rows: list[dict[str, object]], maximum_references: int
) -> list[dict[str, object]]:
    """Build simulator-seed-paired overall contrasts against affine ChannelLiRA."""
    indexed = {
        (
            str(row["mode"]), int(row["shots"]), int(row["simulator_seed"]),
            str(row["attack"]), int(row["reference_count"]),
        ): row
        for row in metric_rows
        if row["scope"] == "overall" and row["mode"] != "exact"
    }
    comparators = {
        "affine_minus_loss": ("loss_mia", 0),
        "affine_minus_learned_logistic": (
            "learned_logistic_pv_stats_target_crossfit_upper_bound", 0
        ),
        "affine_minus_mismatched_lira": ("latent_lira_mismatched", maximum_references),
        "affine_minus_deconvolved_lira": ("deconvolved_lira", maximum_references),
        "affine_minus_empirical_channel": ("empirical_channel_lira", maximum_references),
        "affine_minus_noise_augmented": ("noise_augmented_lira", maximum_references),
    }
    output: list[dict[str, object]] = []
    base_keys = sorted({key[:3] for key in indexed})
    metrics = ("auc", "advantage", "tpr_at_1pct_fpr", "tpr_at_5pct_fpr")
    for mode, shots, simulator_seed in base_keys:
        affine = indexed[(mode, shots, simulator_seed, "affine_channel_lira", maximum_references)]
        for contrast, (attack, count) in comparators.items():
            comparator = indexed[(mode, shots, simulator_seed, attack, count)]
            output.append({
                "mode": mode,
                "shots": shots,
                "simulator_seed": simulator_seed,
                "contrast": contrast,
                "comparator_attack": attack,
                "reference_count": maximum_references,
                **{
                    f"{metric}_difference": float(affine[metric]) - float(comparator[metric])
                    for metric in metrics
                },
            })
    return output


def summarize(
    rows: list[dict[str, object]],
    keys: tuple[str, ...],
    numeric: tuple[str, ...],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for group_key, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        record = dict(zip(keys, group_key))
        record["n_replicates"] = len(group)
        for key in numeric:
            values = np.asarray([float(row[key]) for row in group], dtype=np.float64)
            record[f"{key}_median"] = float(np.median(values))
            record[f"{key}_q05"] = float(np.quantile(values, 0.05))
            record[f"{key}_q95"] = float(np.quantile(values, 0.95))
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def lookup_auc(
    summary: list[dict[str, object]], mode: str, shots: int, attack: str, reference_count: int
) -> dict[str, object]:
    matches = [
        row for row in summary
        if row["scope"] == "overall"
        and row["mode"] == mode
        and int(row["shots"]) == shots
        and row["attack"] == attack
        and int(row["reference_count"]) == reference_count
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Missing overall summary for {(mode, shots, attack, reference_count)}")
    return matches[0]


def build_report(
    config: dict[str, object],
    metric_summary: list[dict[str, object]],
    calibration_summary: list[dict[str, object]],
    diagnostic_summary: list[dict[str, object]],
    contrast_summary: list[dict[str, object]],
) -> str:
    maximum_references = max(config["reference_counts"])
    exact_lines = []
    for attack, count in (
        ("loss_mia", 0),
        ("learned_logistic_pv_stats_target_crossfit_upper_bound", 0),
        ("exact_output_fitted_lira", maximum_references),
    ):
        row = lookup_auc(metric_summary, "exact", 0, attack, count)
        exact_lines.append(f"| {attack} | {float(row['auc_median']):.4f} | {float(row['tpr_at_1pct_fpr_median']):.4f} | {float(row['tpr_at_5pct_fpr_median']):.4f} |")

    noisy_lines = []
    for mode in config["modes"]:
        for shots in config["shots"]:
            for attack in (*REFERENCE_FREE_ATTACKS, *REFERENCE_ATTACKS):
                count = maximum_references if attack in REFERENCE_ATTACKS else 0
                row = lookup_auc(metric_summary, mode, shots, attack, count)
                noisy_lines.append(
                    f"| {mode} | {shots} | {attack} | {float(row['auc_median']):.4f} "
                    f"[{float(row['auc_q05']):.4f}, {float(row['auc_q95']):.4f}] | "
                    f"{float(row['tpr_at_1pct_fpr_median']):.4f} | {float(row['tpr_at_5pct_fpr_median']):.4f} |"
                )

    reference_lines = []
    primary_mode = "noisy_shot" if "noisy_shot" in config["modes"] else config["modes"][-1]
    for shots in sorted({min(config["shots"]), max(config["shots"])}):
        for count in config["reference_counts"]:
            for attack in ("latent_lira_mismatched", "affine_channel_lira", "empirical_channel_lira", "noise_augmented_lira"):
                row = lookup_auc(metric_summary, primary_mode, shots, attack, count)
                reference_lines.append(
                    f"| {shots} | {count} | {attack} | {float(row['auc_median']):.4f} |"
                )

    calibration_lines = []
    for row in calibration_summary:
        if row["scope"] != "overall" or row["mode"] != primary_mode:
            continue
        if int(row["shots"]) not in (min(config["shots"]), max(config["shots"])):
            continue
        if float(row["nominal_fpr"]) != 0.01:
            continue
        attack = str(row["attack"])
        count = int(row["reference_count"])
        if attack not in ("loss_mia", "latent_lira_mismatched", "affine_channel_lira", "empirical_channel_lira"):
            continue
        if attack != "loss_mia" and count != maximum_references:
            continue
        calibration_lines.append(
            f"| {int(row['shots'])} | {attack} | {float(row['actual_fpr_median']):.4f} "
            f"[{float(row['actual_fpr_q05']):.4f}, {float(row['actual_fpr_q95']):.4f}] | "
            f"{float(row['calibrated_tpr_median']):.4f} |"
        )

    diagnostic_lines = []
    for row in diagnostic_summary:
        diagnostic_lines.append(
            f"| {row['cell']} | {row['mode']} | {int(row['shots'])} | "
            f"{float(row['slope_median']):.3f} | {float(row['scale_median']):.3f} | "
            f"{float(row['test_r_squared_median']):.3f} | {float(row['test_coverage_90pct_median']):.3f} |"
        )

    contrast_lines = []
    for row in contrast_summary:
        if row["mode"] != primary_mode or int(row["shots"]) not in (
            min(config["shots"]), max(config["shots"])
        ):
            continue
        contrast_lines.append(
            f"| {int(row['shots'])} | {row['contrast']} | "
            f"{float(row['auc_difference_median']):+.4f} "
            f"[{float(row['auc_difference_q05']):+.4f}, {float(row['auc_difference_q95']):+.4f}] |"
        )

    low = min(config["shots"])
    high = max(config["shots"])
    matched_low = float(lookup_auc(metric_summary, primary_mode, low, "affine_channel_lira", maximum_references)["auc_median"])
    matched_high = float(lookup_auc(metric_summary, primary_mode, high, "affine_channel_lira", maximum_references)["auc_median"])
    loss_high = float(lookup_auc(metric_summary, primary_mode, high, "loss_mia", 0)["auc_median"])
    learned_high = float(lookup_auc(metric_summary, primary_mode, high, "learned_logistic_pv_stats_target_crossfit_upper_bound", 0)["auc_median"])
    mismatch_high = float(lookup_auc(metric_summary, primary_mode, high, "latent_lira_mismatched", maximum_references)["auc_median"])
    empirical_high = float(lookup_auc(metric_summary, primary_mode, high, "empirical_channel_lira", maximum_references)["auc_median"])
    decisive_contrasts = {
        str(row["contrast"]): row
        for row in contrast_summary
        if row["mode"] == primary_mode and int(row["shots"]) == high
    }
    mechanism_supported = (
        float(decisive_contrasts["affine_minus_mismatched_lira"]["auc_difference_q05"]) > 0.0
        and float(decisive_contrasts["affine_minus_loss"]["auc_difference_q05"]) > 0.0
    )
    cell_auc = {
        (str(row["scope_id"]), str(row["attack"]), int(row["reference_count"])): float(
            row["auc_median"]
        )
        for row in metric_summary
        if row["scope"] == "cell"
        and row["mode"] == primary_mode
        and int(row["shots"]) == high
    }
    positive_vs_loss = sum(
        cell_auc[(cell, "affine_channel_lira", maximum_references)]
        > cell_auc[(cell, "loss_mia", 0)]
        for cell in config["cells"]
    )
    positive_vs_mismatch = sum(
        cell_auc[(cell, "affine_channel_lira", maximum_references)]
        > cell_auc[(cell, "latent_lira_mismatched", maximum_references)]
        for cell in config["cells"]
    )
    verdict = (
        "YES — the circuit-level extension supports continuing to the larger study"
        if mechanism_supported
        else "CONDITIONAL — retain the implementation, but resolve the failed comparison before scaling"
    )
    return f"""# Circuit-level ChannelLiRA feasibility report

## Verdict

**{verdict}.**

This is a retrospective analysis of retained Aer outputs, not a new hardware run.
The `noisy_shot` condition used the repository's IBM-Kingston-derived frozen noise
model; `ideal_shot` used finite-shot ideal Aer. All {config['n_target_checkpoints']} targets are QNNs for which the
full main quantum stack was simulated before the trained classical head.

Under `{primary_mode}` at {high} shots, affine ChannelLiRA has AUC {matched_high:.4f}, the empirical
channel mixture has {empirical_high:.4f}, mismatched LiRA has {mismatch_high:.4f},
loss MIA has {loss_high:.4f}, and the target-labeled learned logistic upper bound has
{learned_high:.4f}. The affine endpoint changes from {matched_low:.4f} at {low} shots
to {matched_high:.4f} at {high} shots; this is an endpoint comparison, not a claim
that every intermediate point is monotone. The effect is structurally heterogeneous:
at {high} shots, the cell-level median affine AUC exceeds loss MIA in
{positive_vs_loss}/{config['n_cells']} cells and mismatched LiRA in
{positive_vs_mismatch}/{config['n_cells']} cells. The pooled result therefore supports
an extended study; it does not establish uniform superiority.

## Exact-output baselines

| Attack | AUC | TPR @ 1% FPR | TPR @ 5% FPR |
|---|---:|---:|---:|
{chr(10).join(exact_lines)}

## Same-output-channel attack comparison

Intervals are the 5th–95th percentiles across the {len(config['simulator_seeds'])} retained simulator seeds.

| Mode | Shots | Attack | AUC [5%, 95%] | TPR @ 1% FPR | TPR @ 5% FPR |
|---|---:|---|---:|---:|---:|
{chr(10).join(noisy_lines)}

The learned baseline is a five-fold, target-specific logistic classifier over the
same nine `pv+stats` features used by the repository's learned MIA. It has labeled
target-output auxiliary access, so it is an explicitly marked upper-knowledge
baseline, not the same shadow-model threat model as LiRA.

## Paired AUC contrasts on the primary noisy condition

| Shots | Contrast | AUC difference [5%, 95%] |
|---:|---|---:|
{chr(10).join(contrast_lines)}

Each difference pairs attacks within the same retained simulator seed. The automatic
continuation verdict requires the 5th percentile of both `affine_minus_loss` and
`affine_minus_mismatched_lira` to be above zero at the largest shot budget.

## Reference-count ablation on the IBM-derived noisy condition

| Shots | References | Attack | AUC |
|---:|---:|---|---:|
{chr(10).join(reference_lines)}

The selected {'/'.join(str(value) for value in config['reference_counts'])}-reference subsets are exactly balanced for every evaluated
candidate. Noise-augmented LiRA uses {config['noise_augmentation_draws']} simulated
channel draws per retained reference; analytic and empirical ChannelLiRA do not
retrain or re-evaluate noisy reference models.

## Cross-fitted nominal 1% FPR calibration

| Shots | Attack | Actual FPR [5%, 95%] | TPR |
|---:|---|---:|---:|
{chr(10).join(calibration_lines)}

Channel parameters and thresholds use only public nonmember pairs from other folds.
The attacked sample ID is excluded across every target checkpoint and simulator seed.
The learned baseline is excluded from this calibration table because calibrating its
out-of-fold scores would require a nested labeled-target split.

## Channel fit diagnostics

| Cell | Mode | Shots | Slope | Residual SD | Held-out R² | Held-out 90% coverage |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(diagnostic_lines)}

Large residual skew/kurtosis values in `channel_diagnostics_raw.csv` indicate where
the affine Gaussian approximation is misspecified. They are a gate for adding a
heteroskedastic or nonparametric channel in the extended study.

## Scope and remaining publication gates

- Evidence covers {config['n_cells']} structural cells and {config['n_target_checkpoints']} fixed QNN checkpoints; the retained
  CSVs do not include noisy reference-checkpoint evaluations, classical stochastic
  models, time drift, or real hardware executions.
- The channel-calibration threat model assumes paired exact/noisy outputs for disjoint
  public nonmembers. A lower-knowledge estimator must be tested before a broad claim.
- Simulator seeds are repeated measurements of the same records and checkpoints.
  The intervals above are not record/model-clustered publication confidence intervals.
- A publication study should add noisy reference ensembles, heteroskedastic channel
  models, held-out calibration targets, hardware snapshots over time, and at least one
  classical stochastic-serving control.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--noisy-root",
        type=Path,
        default=Path("reviewer_results/noisy_sanity/raw_all_seeds"),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("reviewer_results/lira_reference_mia/reference_models"),
    )
    parser.add_argument("--cells", default="all")
    parser.add_argument("--modes", default="ideal_shot,noisy_shot")
    parser.add_argument("--shots", default="128,512,1024")
    parser.add_argument("--reference-counts", default="4,8,16")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--noise-augmentation-draws", type=int, default=32)
    parser.add_argument("--variance-shrinkage", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("channel_lira_results/circuit_phase2")
    )
    args = parser.parse_args()
    if args.folds < 2 or args.noise_augmentation_draws < 1:
        raise ValueError("Use at least two folds and one noise-augmentation draw")
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    shots = parse_int_list(args.shots)
    reference_counts = parse_int_list(args.reference_counts)
    noisy_root = args.noisy_root.resolve()
    reference_root = args.reference_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = [
        load_cell(noisy_root, reference_root, cell)
        for cell in discover_cells(noisy_root, args.cells)
    ]
    requested_conditions = {(mode, shot) for mode in modes for shot in shots}
    for cell in cells:
        missing = sorted(requested_conditions - set(cell.conditions))
        if missing:
            raise ValueError(f"{cell.name}: missing conditions {missing}")
        if max(reference_counts) > len(cell.reference_scores):
            raise ValueError(f"{cell.name}: requested more references than retained")

    blocks: list[EvaluationBlock] = []
    diagnostics: list[dict[str, object]] = []
    selected_references: dict[str, dict[str, list[int]]] = {}
    for cell in cells:
        selected_by_count = {
            count: balanced_reference_subset(cell.inclusion, count)
            for count in reference_counts
        }
        blocks.append(evaluate_exact(
            cell,
            selected_by_count=selected_by_count,
            folds=args.folds,
            variance_shrinkage=args.variance_shrinkage,
            seed=args.seed,
        ))
        selected_references[cell.name] = {}
        for mode in modes:
            for shot in shots:
                block, diagnostic_rows, selected = evaluate_condition(
                    cell,
                    mode,
                    shot,
                    cell.conditions[(mode, shot)],
                    selected_by_count=selected_by_count,
                    folds=args.folds,
                    noise_augmentation_draws=args.noise_augmentation_draws,
                    variance_shrinkage=args.variance_shrinkage,
                    seed=args.seed,
                )
                blocks.append(block)
                diagnostics.extend(diagnostic_rows)
                for count, indices in selected.items():
                    selected_references[cell.name][str(count)] = indices
                print(f"completed {cell.name} {mode} {shot} shots", flush=True)

    metric_rows = build_metric_rows(blocks)
    calibration_rows = build_calibration_rows(blocks)
    contrast_rows = build_contrast_rows(metric_rows, max(reference_counts))
    metric_summary = summarize(
        metric_rows,
        ("scope", "scope_id", "mode", "shots", "attack", "reference_count"),
        ("auc", "advantage", "tpr_at_0_1pct_fpr", "tpr_at_1pct_fpr", "tpr_at_5pct_fpr"),
    )
    calibration_summary = summarize(
        calibration_rows,
        ("scope", "scope_id", "mode", "shots", "attack", "reference_count", "nominal_fpr"),
        ("actual_fpr", "calibrated_tpr"),
    )
    diagnostic_summary = summarize(
        diagnostics,
        ("cell", "mode", "shots"),
        (
            "intercept", "slope", "scale", "train_r_squared", "test_r_squared",
            "test_rmse", "test_residual_skew", "test_residual_excess_kurtosis",
            "test_coverage_90pct",
        ),
    )
    contrast_summary = summarize(
        contrast_rows,
        ("mode", "shots", "contrast", "comparator_attack", "reference_count"),
        (
            "auc_difference", "advantage_difference", "tpr_at_1pct_fpr_difference",
            "tpr_at_5pct_fpr_difference",
        ),
    )
    write_csv(out_dir / "metrics_raw.csv", metric_rows)
    write_csv(out_dir / "metrics_summary.csv", metric_summary)
    write_csv(out_dir / "calibration_raw.csv", calibration_rows)
    write_csv(out_dir / "calibration_summary.csv", calibration_summary)
    write_csv(out_dir / "channel_diagnostics_raw.csv", diagnostics)
    write_csv(out_dir / "channel_diagnostics_summary.csv", diagnostic_summary)
    write_csv(out_dir / "paired_contrasts_raw.csv", contrast_rows)
    write_csv(out_dir / "paired_contrasts_summary.csv", contrast_summary)

    all_sources = sorted({path.resolve() for cell in cells for path in cell.source_files})
    first_metadata = json.loads((Path(cells[0].source_files[0]).parent / "run_metadata.json").read_text(encoding="utf-8"))
    backend = dict(first_metadata.get("backend", {}))
    config: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design": "cross-fitted paired exact-to-stochastic continuous serving channel",
        "threat_model": (
            "K2 paired exact/noisy public-nonmember calibration; attacked sample ID excluded "
            "across targets and simulator seeds"
        ),
        "cells": [cell.name for cell in cells],
        "n_cells": len(cells),
        "target_ids": [target for cell in cells for target in cell.target_ids],
        "n_target_checkpoints": sum(len(cell.target_ids) for cell in cells),
        "n_candidates_per_target": sorted({len(cell.sample_ids) for cell in cells}),
        "modes": modes,
        "shots": shots,
        "simulator_seeds": list(next(iter(cells[0].conditions.values())).simulator_seeds),
        "reference_counts": reference_counts,
        "selected_reference_indices": selected_references,
        "folds": args.folds,
        "noise_augmentation_draws": args.noise_augmentation_draws,
        "variance_shrinkage": args.variance_shrinkage,
        "seed": args.seed,
        "backend_name": backend.get(
            "resolved_noise_backend_name",
            backend.get("resolved_backend_name", backend.get("backend_name", backend.get("name"))),
        ),
        "calibration_timestamp": backend.get("calibration_timestamp"),
        "snapshot_manifest_sha256": backend.get("snapshot_manifest_sha256"),
        "backend_noise_metadata_sha256": sha256(
            Path(cells[0].source_files[0]).parent / "backend_noise_metadata.json"
        ),
        "quantum_execution_scope": "full main quantum stack for retained QNN targets",
        "source_manifest_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in all_sources
        },
    }
    (out_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = build_report(
        config, metric_summary, calibration_summary, diagnostic_summary, contrast_summary
    )
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"wrote circuit-level artifacts to {out_dir}")


if __name__ == "__main__":
    main()
