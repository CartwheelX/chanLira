#!/usr/bin/env python3
"""Analyze the Phase-6 true noisy-reference LiRA scale-up.

The analysis compares a matched finite-shot/noisy 16-reference LiRA bank with
strict leave-target-out ChannelLiRA, latent-reference LiRA applied directly to
served outputs, a scalar loss attack, and an explicitly stronger-access
target-cross-fitted learned MIA.  The attacked target is never used to fit the
ChannelLiRA serving channel.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.channel_lira_circuit_pilot import (  # noqa: E402
    CellData,
    ConditionData,
    learned_crossfit_scores,
    stable_int,
    write_csv,
)
from experiments.channel_lira_transfer import (  # noqa: E402
    build_metric_rows,
    evaluate_leave_target_out,
    reference_bundles,
)


EXPECTED_TARGETS = (
    "MNIST_QNN_eff_su2_r1_d2_s43",
    "MNIST_QNN_eff_su2_r1_d2_s44",
    "MNIST_QNN_eff_su2_r1_d2_s45",
)
EXPECTED_CELL = "eff_su2_r1_d2"
MATCHED_ATTACKS = (
    "lira_online",
    "lira_online_fixed_variance",
    "lira_offline",
    "lira_offline_fixed_variance",
)
PRIMARY_MATCHED = "matched_reference_lira_online_fixed_variance"
PRIMARY_ATTACKS = (
    PRIMARY_MATCHED,
    "affine_channel_lira",
    "latent_lira_mismatched",
    "loss_mia",
    "target_crossfit_learned_mia",
)
METRICS = (
    "auc",
    "advantage",
    "tpr_at_0_1pct_fpr",
    "tpr_at_1pct_fpr",
    "tpr_at_5pct_fpr",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_csv_list(text: str) -> list[str]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list")
    return list(dict.fromkeys(values))


def parse_int_list(text: str) -> list[int]:
    values = [int(value) for value in parse_csv_list(text)]
    if any(value < 0 for value in values):
        raise ValueError("Simulator seeds must be non-negative")
    return values


def read_targets(
    path: Path,
    *,
    expected_targets: tuple[str, ...] = EXPECTED_TARGETS,
    expected_cell: str = EXPECTED_CELL,
    phase_label: str = "Phase-6",
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    target_ids = tuple(row.get("target_id", "") for row in rows)
    if target_ids != expected_targets:
        raise ValueError(
            f"{phase_label} manifest must contain the frozen targets {expected_targets}; "
            f"found {target_ids}"
        )
    if any(row.get("structural_cell_id") != expected_cell for row in rows):
        raise ValueError(f"Every {phase_label} target must use cell {expected_cell}")
    if any(row.get("architecture", "").lower() != "qnn" for row in rows):
        raise ValueError(f"{phase_label} is frozen to QNN targets")
    return rows


def probability_features(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(labels):
        raise ValueError("Probability vectors and labels are not aligned")
    if probabilities.shape[1] != 4:
        raise ValueError(
            "The frozen learned-MIA baseline requires four-class probability vectors"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Probability vectors contain non-finite values")
    clipped = np.clip(probabilities, 1e-12, 1.0)
    clipped /= clipped.sum(axis=1, keepdims=True)
    true_probability = clipped[np.arange(len(labels)), labels]
    loss = -np.log(true_probability)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    ordered = np.sort(clipped, axis=1)
    confidence = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    correctness = (np.argmax(clipped, axis=1) == labels).astype(np.float64)
    features = np.column_stack(
        (clipped, loss, entropy, confidence, margin, correctness)
    )
    if features.shape[1] != 9:
        raise RuntimeError("Learned-MIA feature construction did not produce 9 features")
    return loss, features


def require_npz_fields(path: Path, saved: Any, fields: set[str]) -> None:
    missing = sorted(fields - set(saved.files))
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")


def load_scaleup_cell(
    *,
    targets_path: Path,
    reference_dir: Path,
    noisy_dir: Path,
    modes: list[str],
    shots: int,
    simulator_seeds: list[int],
    num_references: int,
    expected_targets: tuple[str, ...] = EXPECTED_TARGETS,
    expected_cell: str = EXPECTED_CELL,
    phase_label: str = "Phase-6",
) -> tuple[
    CellData,
    dict[tuple[str, int], dict[str, np.ndarray]],
    list[Path],
    dict[str, Any],
]:
    rows = read_targets(
        targets_path,
        expected_targets=expected_targets,
        expected_cell=expected_cell,
        phase_label=phase_label,
    )
    target_ids = tuple(row["target_id"] for row in rows)
    source_paths: list[Path] = [targets_path]
    sample_ids: np.ndarray | None = None
    labels: np.ndarray | None = None
    memberships = []
    exact_scores = []
    exact_losses = []
    exact_features = []

    for target_id in target_ids:
        path = reference_dir / "sample_scores" / f"{target_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing exact target score payload: {path}")
        source_paths.append(path)
        with np.load(path, allow_pickle=False) as saved:
            require_npz_fields(
                path,
                saved,
                {
                    "sample_ids", "labels", "membership", "probabilities",
                    "observed_log_odds",
                },
            )
            current_ids = saved["sample_ids"].astype(str)
            current_labels = saved["labels"].astype(np.int64)
            if sample_ids is None:
                sample_ids = current_ids
                labels = current_labels
            elif not np.array_equal(current_ids, sample_ids):
                raise ValueError(f"Candidate sample IDs differ in {path}")
            elif not np.array_equal(current_labels, labels):
                raise ValueError(f"Candidate labels differ in {path}")
            loss, features = probability_features(saved["probabilities"], current_labels)
            memberships.append(saved["membership"].astype(np.int8))
            exact_scores.append(saved["observed_log_odds"].astype(np.float64))
            exact_losses.append(loss)
            exact_features.append(features)

    assert sample_ids is not None and labels is not None
    conditions: dict[tuple[str, int], ConditionData] = {}
    matched_by_condition: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    metadata_records = []
    for mode in modes:
        observed = np.empty((len(target_ids), len(simulator_seeds), len(sample_ids)))
        losses = np.empty_like(observed)
        features = np.empty((*observed.shape, 9))
        matched = {
            attack: np.empty_like(observed) for attack in MATCHED_ATTACKS
        }
        for target_index, target_id in enumerate(target_ids):
            metadata_path = noisy_dir / "metadata" / f"{target_id}.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Missing noisy-score metadata: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_protocol = {
                "shots": shots,
                "num_reference_models": num_references,
                "modes": modes,
                "simulator_seeds": simulator_seeds,
            }
            mismatches = {
                key: (metadata.get(key), expected)
                for key, expected in expected_protocol.items()
                if metadata.get(key) != expected
            }
            if mismatches:
                raise ValueError(
                    f"Noisy-score protocol mismatch in {metadata_path}: {mismatches}"
                )
            metadata_records.append(metadata)
            source_paths.append(metadata_path)
            for seed_index, simulator_seed in enumerate(simulator_seeds):
                path = noisy_dir / "sample_scores" / (
                    f"{target_id}_{mode}_sim{simulator_seed}.npz"
                )
                if not path.is_file():
                    raise FileNotFoundError(f"Missing served-output payload: {path}")
                source_paths.append(path)
                with np.load(path, allow_pickle=False) as saved:
                    require_npz_fields(
                        path,
                        saved,
                        {
                            "sample_ids", "labels", "membership", "probabilities",
                            "observed_log_odds", *MATCHED_ATTACKS,
                        },
                    )
                    if not np.array_equal(saved["sample_ids"].astype(str), sample_ids):
                        raise ValueError(f"Candidate sample IDs differ in {path}")
                    if not np.array_equal(saved["labels"].astype(np.int64), labels):
                        raise ValueError(f"Candidate labels differ in {path}")
                    if not np.array_equal(
                        saved["membership"].astype(np.int8), memberships[target_index]
                    ):
                        raise ValueError(f"Membership labels differ in {path}")
                    loss, current_features = probability_features(
                        saved["probabilities"], labels
                    )
                    observed[target_index, seed_index] = saved[
                        "observed_log_odds"
                    ].astype(np.float64)
                    losses[target_index, seed_index] = loss
                    features[target_index, seed_index] = current_features
                    for attack in MATCHED_ATTACKS:
                        matched[attack][target_index, seed_index] = saved[attack].astype(
                            np.float64
                        )
        conditions[(mode, shots)] = ConditionData(
            simulator_seeds=tuple(simulator_seeds),
            observed_scores=observed,
            losses=losses,
            features=features,
        )
        matched_by_condition[(mode, shots)] = matched

    canonical_cell = f"{expected_cell}_wd0"
    reference_root = reference_dir / "reference_models" / canonical_cell
    reference_scores = []
    inclusion = []
    for reference_id in range(num_references):
        path = reference_root / f"reference_{reference_id:03d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing exact reference score: {path}")
        source_paths.append(path)
        with np.load(path, allow_pickle=False) as saved:
            require_npz_fields(
                path,
                saved,
                {"sample_ids", "scores", "inclusion", "num_references"},
            )
            if int(saved["num_references"]) != num_references:
                raise ValueError(f"Reference-count mismatch in {path}")
            reference_ids = saved["sample_ids"].astype(str)
            index = {sample_id: offset for offset, sample_id in enumerate(reference_ids)}
            try:
                order = np.asarray([index[value] for value in sample_ids], dtype=np.int64)
            except KeyError as error:
                raise ValueError(f"Reference candidate IDs differ in {path}") from error
            reference_scores.append(saved["scores"].astype(np.float64)[order])
            inclusion.append(saved["inclusion"].astype(bool)[order])
    reference_score_array = np.stack(reference_scores)
    inclusion_array = np.stack(inclusion)
    if not np.all(inclusion_array.sum(axis=0) == num_references // 2):
        raise ValueError("The Phase-6 reference bank is not exactly record-balanced")

    snapshot_hashes = {
        str(record.get("snapshot_manifest_sha256", "")) for record in metadata_records
    }
    if len(snapshot_hashes) != 1 or not next(iter(snapshot_hashes)):
        raise ValueError("Noisy targets do not share one frozen snapshot manifest hash")
    protocol = {
        "snapshot_manifest_sha256": next(iter(snapshot_hashes)),
        "backend_names": sorted(
            {str(record.get("backend", {}).get("resolved_backend_name", "")) for record in metadata_records}
        ),
        "calibration_timestamps": sorted(
            {str(record.get("backend", {}).get("calibration_timestamp", "")) for record in metadata_records}
        ),
    }
    cell = CellData(
        name=expected_cell,
        target_ids=target_ids,
        sample_ids=sample_ids,
        reference_scores=reference_score_array,
        inclusion=inclusion_array,
        memberships=np.asarray(memberships, dtype=np.int8),
        exact_scores=np.asarray(exact_scores, dtype=np.float64),
        exact_losses=np.asarray(exact_losses, dtype=np.float64),
        exact_features=np.asarray(exact_features, dtype=np.float64),
        conditions=conditions,
        source_files=tuple(dict.fromkeys(source_paths)),
    )
    return cell, matched_by_condition, source_paths, protocol


def add_scaleup_comparators(
    block: Any,
    cell: CellData,
    condition: ConditionData,
    matched: dict[str, np.ndarray],
    *,
    num_references: int,
    folds: int,
    seed: int,
    seed_namespace: str = "phase6-learned",
) -> None:
    learned = np.empty_like(condition.observed_scores)
    for target_index, target_id in enumerate(cell.target_ids):
        for simulator_index, simulator_seed in enumerate(condition.simulator_seeds):
            learned[target_index, simulator_index] = learned_crossfit_scores(
                condition.features[target_index, simulator_index],
                cell.memberships[target_index],
                cell.sample_ids,
                folds=folds,
                seed=seed + stable_int(
                    f"{seed_namespace}|{target_id}|{block.mode}|{block.shots}|{simulator_seed}"
                ) % 1_000_000,
            )
    block.scores[("target_crossfit_learned_mia", 0)] = learned
    for attack, values in matched.items():
        block.scores[(f"matched_reference_{attack}", num_references)] = values


def target_mean_rows(metrics: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in metrics:
        if row["scope"] != "target":
            continue
        key = (
            row["scope_id"], row["scheme"], row["mode"], row["shots"],
            row["attack"], row["reference_count"],
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        target_id, scheme, mode, shots, attack, reference_count = key
        record: dict[str, object] = {
            "target_id": target_id,
            "scheme": scheme,
            "mode": mode,
            "shots": shots,
            "attack": attack,
            "reference_count": reference_count,
            "n_simulator_seeds": len(group),
        }
        for metric in METRICS:
            record[f"{metric}_mean_over_simulator_seeds"] = float(
                np.mean([float(row[metric]) for row in group])
            )
        output.append(record)
    return output


def summarize_targets(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["scheme"], row["mode"], row["shots"], row["attack"],
            row["reference_count"],
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        scheme, mode, shots, attack, reference_count = key
        record: dict[str, object] = {
            "scheme": scheme,
            "mode": mode,
            "shots": shots,
            "attack": attack,
            "reference_count": reference_count,
            "n_targets": len(group),
            "replication_unit": "independently trained target checkpoint after averaging simulator seeds",
        }
        for metric in METRICS:
            column = f"{metric}_mean_over_simulator_seeds"
            values = np.asarray([float(row[column]) for row in group])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd_targets"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            record[f"{metric}_min"] = float(values.min())
            record[f"{metric}_max"] = float(values.max())
        output.append(record)
    return output


def paired_contrasts(
    rows: list[dict[str, object]], num_references: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    indexed = {
        (
            str(row["target_id"]), str(row["mode"]), int(row["shots"]),
            str(row["attack"]), int(row["reference_count"]),
        ): row
        for row in rows
    }
    definitions = {
        "affine_minus_matched_reference": (
            ("affine_channel_lira", num_references),
            (PRIMARY_MATCHED, num_references),
        ),
        "matched_reference_minus_mismatched": (
            (PRIMARY_MATCHED, num_references),
            ("latent_lira_mismatched", num_references),
        ),
        "matched_reference_minus_loss": (
            (PRIMARY_MATCHED, num_references), ("loss_mia", 0),
        ),
        "matched_reference_minus_learned": (
            (PRIMARY_MATCHED, num_references),
            ("target_crossfit_learned_mia", 0),
        ),
        "affine_minus_mismatched": (
            ("affine_channel_lira", num_references),
            ("latent_lira_mismatched", num_references),
        ),
        "affine_minus_loss": (
            ("affine_channel_lira", num_references), ("loss_mia", 0),
        ),
    }
    target_rows = []
    base = sorted({key[:3] for key in indexed})
    for target_id, mode, shots in base:
        for contrast, (left_key, right_key) in definitions.items():
            left = indexed[(target_id, mode, shots, *left_key)]
            right = indexed[(target_id, mode, shots, *right_key)]
            record: dict[str, object] = {
                "target_id": target_id,
                "mode": mode,
                "shots": shots,
                "contrast": contrast,
                "left_attack": left_key[0],
                "right_attack": right_key[0],
            }
            for metric in METRICS:
                column = f"{metric}_mean_over_simulator_seeds"
                record[f"{metric}_difference"] = float(left[column]) - float(right[column])
            target_rows.append(record)
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for row in target_rows:
        grouped.setdefault(
            (str(row["mode"]), int(row["shots"]), str(row["contrast"])), []
        ).append(row)
    summary = []
    for key, group in sorted(grouped.items()):
        mode, shots, contrast = key
        record: dict[str, object] = {
            "mode": mode,
            "shots": shots,
            "contrast": contrast,
            "n_targets": len(group),
            "replication_unit": "target checkpoint",
        }
        for metric in METRICS:
            column = f"{metric}_difference"
            values = np.asarray([float(row[column]) for row in group])
            record[f"{metric}_difference_mean"] = float(values.mean())
            record[f"{metric}_difference_sd_targets"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            record[f"{metric}_difference_min"] = float(values.min())
            record[f"{metric}_difference_max"] = float(values.max())
        summary.append(record)
    return target_rows, summary


def find_summary(
    rows: list[dict[str, object]], mode: str, attack: str, reference_count: int
) -> dict[str, object]:
    matches = [
        row for row in rows
        if row["mode"] == mode and row["attack"] == attack
        and int(row["reference_count"]) == reference_count
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Missing summary for {(mode, attack, reference_count)}")
    return matches[0]


def build_report(
    *,
    summary: list[dict[str, object]],
    contrasts: list[dict[str, object]],
    modes: list[str],
    shots: int,
    num_references: int,
    protocol: dict[str, Any],
) -> str:
    lines = [
        "# Phase 6: 16-reference noisy scale-up",
        "",
        "This is a one-cell scale and comparison gate. It is stronger than the four-reference canary but is not a publication-level cross-cell result.",
        "",
        "## Frozen protocol",
        "",
        f"- Targets: {len(EXPECTED_TARGETS)} independently initialized checkpoints in `{EXPECTED_CELL}`.",
        f"- References: {num_references}, with exactly {num_references // 2} IN and {num_references // 2} OUT observations per candidate.",
        f"- Modes: `{','.join(modes)}`; shots: {shots}; channel scheme: strict leave-target-out.",
        "- Channel calibration uses the other two target checkpoints and excludes the attacked sample-ID fold.",
        "- The learned baseline is target-cross-fitted on labeled attacked-target outputs and therefore has stronger auxiliary access than LiRA.",
        f"- Snapshot manifest SHA-256: `{protocol['snapshot_manifest_sha256']}`.",
        "",
        "## Target-level AUC summary",
        "",
        "Values are means across three target checkpoints after averaging simulator seeds; ± is sample SD across targets.",
        "",
        "| Mode | Attack | Mean AUC | SD across targets | Target range |",
        "|---|---|---:|---:|---:|",
    ]
    labels = {
        PRIMARY_MATCHED: "matched 16-reference LiRA",
        "affine_channel_lira": "leave-target-out ChannelLiRA",
        "latent_lira_mismatched": "mismatched latent LiRA",
        "loss_mia": "loss MIA",
        "target_crossfit_learned_mia": "target-cross-fitted learned MIA",
    }
    for mode in modes:
        for attack in PRIMARY_ATTACKS:
            count = 0 if attack in {"loss_mia", "target_crossfit_learned_mia"} else num_references
            row = find_summary(summary, mode, attack, count)
            lines.append(
                f"| {mode} | {labels[attack]} | {float(row['auc_mean']):.4f} | "
                f"{float(row['auc_sd_targets']):.4f} | "
                f"[{float(row['auc_min']):.4f}, {float(row['auc_max']):.4f}] |"
            )
    lines.extend([
        "",
        "## Paired target-level AUC contrasts",
        "",
        "Positive values favor the attack named first. Ranges are descriptive minima/maxima over three targets, not confidence intervals.",
        "",
        "| Mode | Contrast | Mean difference | Target range |",
        "|---|---|---:|---:|",
    ])
    for row in contrasts:
        if row["contrast"] not in {
            "affine_minus_matched_reference",
            "matched_reference_minus_mismatched",
            "matched_reference_minus_loss",
            "matched_reference_minus_learned",
        }:
            continue
        lines.append(
            f"| {row['mode']} | {row['contrast']} | "
            f"{float(row['auc_difference_mean']):+.4f} | "
            f"[{float(row['auc_difference_min']):+.4f}, "
            f"{float(row['auc_difference_max']):+.4f}] |"
        )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- Three checkpoints from one structural cell are a scale-up gate, not evidence of cross-model or cross-architecture generalization.",
        "- Simulator seeds quantify serving randomness but do not increase the independent model-level sample size.",
        "- The 400-candidate target pools cannot support a stable 0.1% FPR claim.",
        "- Successful completion is the go/no-go gate for the prespecified 15-target/80-reference study.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets", type=Path,
        default=ROOT / "reviewer_targets/channel_lira_noisy_reference_scaleup.csv",
    )
    parser.add_argument(
        "--reference-dir", type=Path,
        default=ROOT / "channel_lira_results/noisy_reference_scaleup_phase6/references",
    )
    parser.add_argument(
        "--noisy-dir", type=Path,
        default=ROOT / "channel_lira_results/noisy_reference_scaleup_phase6/noisy_lira",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "channel_lira_results/noisy_reference_scaleup_phase6/analysis",
    )
    parser.add_argument("--modes", default="ideal_shot,noisy_shot")
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--simulator-seeds", default="0,1")
    parser.add_argument("--num-references", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--noise-augmentation-draws", type=int, default=32)
    parser.add_argument("--variance-shrinkage", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    args.targets = args.targets.resolve()
    args.reference_dir = args.reference_dir.resolve()
    args.noisy_dir = args.noisy_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    modes = parse_csv_list(args.modes)
    if not set(modes).issubset({"ideal_shot", "noisy_shot"}):
        raise ValueError("--modes must contain ideal_shot and/or noisy_shot")
    simulator_seeds = parse_int_list(args.simulator_seeds)
    if args.num_references != 16:
        raise ValueError("The Phase-6 scale-up is frozen to 16 reference models")
    if args.shots != 128 or simulator_seeds != [0, 1]:
        raise ValueError("The Phase-6 scale gate is frozen to 128 shots and seeds 0,1")

    cell, matched, source_paths, protocol = load_scaleup_cell(
        targets_path=args.targets,
        reference_dir=args.reference_dir,
        noisy_dir=args.noisy_dir,
        modes=modes,
        shots=args.shots,
        simulator_seeds=simulator_seeds,
        num_references=args.num_references,
    )
    bundles = reference_bundles(
        cell, [args.num_references], args.variance_shrinkage
    )
    blocks = []
    diagnostics = []
    for mode in modes:
        condition = cell.conditions[(mode, args.shots)]
        block, rows = evaluate_leave_target_out(
            cell,
            condition,
            bundles,
            mode=mode,
            shots=args.shots,
            folds=args.folds,
            noise_augmentation_draws=args.noise_augmentation_draws,
            variance_shrinkage=args.variance_shrinkage,
            seed=args.seed,
        )
        add_scaleup_comparators(
            block,
            cell,
            condition,
            matched[(mode, args.shots)],
            num_references=args.num_references,
            folds=args.folds,
            seed=args.seed,
        )
        blocks.append(block)
        diagnostics.extend(rows)
        print(f"completed phase6 analysis {mode} {args.shots} shots", flush=True)

    metrics = build_metric_rows(blocks)
    target_means = target_mean_rows(metrics)
    summary = summarize_targets(target_means)
    contrast_targets, contrast_summary = paired_contrasts(
        target_means, args.num_references
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "metrics_raw.csv", metrics)
    write_csv(args.out_dir / "target_mean_metrics.csv", target_means)
    write_csv(args.out_dir / "metrics_summary.csv", summary)
    write_csv(args.out_dir / "paired_contrasts_target.csv", contrast_targets)
    write_csv(args.out_dir / "paired_contrasts_summary.csv", contrast_summary)
    write_csv(args.out_dir / "channel_diagnostics.csv", diagnostics)
    source_manifest = {
        str(path.resolve()): sha256(path)
        for path in sorted(set(source_paths), key=lambda value: str(value))
    }
    (args.out_dir / "SOURCE_MANIFEST.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "targets": list(cell.target_ids),
        "cell": cell.name,
        "modes": modes,
        "shots": args.shots,
        "simulator_seeds": simulator_seeds,
        "num_references": args.num_references,
        "folds": args.folds,
        "noise_augmentation_draws": args.noise_augmentation_draws,
        "variance_shrinkage": args.variance_shrinkage,
        "seed": args.seed,
        "channel_scheme": "leave_target_out plus sample-ID cross-fitting",
        "learned_mia_access": "labeled attacked-target outputs; stronger auxiliary access",
        "protocol": protocol,
        "real_hardware_execution": False,
        "statistical_scope": "one-cell scale gate; descriptive target-level variation",
    }
    (args.out_dir / "CONFIG.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "REPORT.md").write_text(
        build_report(
            summary=summary,
            contrasts=contrast_summary,
            modes=modes,
            shots=args.shots,
            num_references=args.num_references,
            protocol=protocol,
        ),
        encoding="utf-8",
    )
    print(f"[REPORT] {args.out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
