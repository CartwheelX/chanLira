#!/usr/bin/env python3
"""Unblind and analyze the frozen Phase-7 confirmatory primary experiment.

This module only accepts the twelve confirmatory targets and the noisy 128-shot,
seed-0..9 serving condition.  Analysis is blocked until the Stage-2 launcher has
sealed every raw target/reference/noisy-serving artifact by SHA-256.
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

from experiments.channel_lira_circuit_pilot import write_csv  # noqa: E402
from experiments.channel_lira_noisy_reference_scaleup import (  # noqa: E402
    METRICS,
    PRIMARY_ATTACKS,
    PRIMARY_MATCHED,
    add_scaleup_comparators,
    load_scaleup_cell,
    paired_contrasts,
    summarize_targets,
    target_mean_rows,
)
from experiments.channel_lira_transfer import (  # noqa: E402
    build_metric_rows,
    evaluate_leave_target_out,
    reference_bundles,
)
from experiments.check_channel_lira_phase7_readiness import (  # noqa: E402
    DEFAULT_CANDIDATE_PROBE,
    DEFAULT_PROTOCOL,
    DEFAULT_PROTOCOL_LOCK,
    inspect_candidate_probe,
    inspect_snapshot,
    read_targets as read_phase7_targets,
    resolve_repo_path,
    sha256,
    validate_design,
    validate_protocol_lock,
)


EXPECTED_CELLS = (
    "eff_su2_r5_d2",
    "z_r1_d6",
    "zz_r1_d6",
    "zz_r5_d6",
)
EXPECTED_TARGETS = (
    "MNIST_QNN_eff_su2_r5_d2_p7_s143",
    "MNIST_QNN_eff_su2_r5_d2_p7_s144",
    "MNIST_QNN_eff_su2_r5_d2_p7_s145",
    "MNIST_QNN_z_r1_d6_p7_s143",
    "MNIST_QNN_z_r1_d6_p7_s144",
    "MNIST_QNN_z_r1_d6_p7_s145",
    "MNIST_QNN_zz_r1_d6_p7_s143",
    "MNIST_QNN_zz_r1_d6_p7_s144",
    "MNIST_QNN_zz_r1_d6_p7_s145",
    "MNIST_QNN_zz_r5_d6_p7_s143",
    "MNIST_QNN_zz_r5_d6_p7_s144",
    "MNIST_QNN_zz_r5_d6_p7_s145",
)
PRIMARY_MODE = "noisy_shot"
PRIMARY_SHOTS = 128
SIMULATOR_SEEDS = tuple(range(10))
NUM_REFERENCES = 16
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20_260_822
FOLDS = 5
NOISE_AUGMENTATION_DRAWS = 32
VARIANCE_SHRINKAGE = 0.15
DEFAULT_TARGETS = ROOT / "reviewer_targets/channel_lira_phase7_stage2_confirmatory.csv"
DEFAULT_REFERENCE_DIR = ROOT / "channel_lira_results/phase7/references"
DEFAULT_NOISY_DIR = ROOT / "channel_lira_results/phase7/stage2_primary/noisy_lira"
DEFAULT_SEAL = ROOT / "channel_lira_results/phase7/stage2_primary/RAW_OUTPUT_SEAL.json"
DEFAULT_OUT = ROOT / "channel_lira_results/phase7/stage2_primary/analysis"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_stage2_manifest(
    protocol: dict[str, Any], manifest_path: Path
) -> list[dict[str, str]]:
    """Require an exact, ordered projection of the frozen confirmatory rows."""
    full_path = resolve_repo_path(protocol["provenance"]["phase7_target_manifest"])
    full_rows = read_phase7_targets(full_path)
    expected_rows = [
        row for row in full_rows if row.get("phase7_analysis_role") == "confirmatory"
    ]
    rows = read_csv(manifest_path)
    observed_ids = tuple(row.get("target_id", "") for row in rows)
    if observed_ids != EXPECTED_TARGETS:
        raise ValueError(
            f"Stage-2 manifest must contain the frozen targets {EXPECTED_TARGETS}; "
            f"found {observed_ids}"
        )
    if rows != expected_rows:
        raise ValueError("Stage-2 rows are not the exact frozen confirmatory projection")
    if tuple(dict.fromkeys(row["structural_cell_id"] for row in rows)) != EXPECTED_CELLS:
        raise ValueError("Stage-2 structural-cell order differs from the frozen protocol")
    if {row["phase7_analysis_role"] for row in rows} != {"confirmatory"}:
        raise ValueError("Stage-2 manifest contains a non-confirmatory target")
    return rows


def validate_stage2_protocol(protocol: dict[str, Any]) -> None:
    population = protocol["study_population"]
    primary = protocol["serving_protocol"]["primary"]
    endpoint = protocol["endpoint_hierarchy"]["primary_endpoint"]
    stage = next(
        value for value in protocol["execution_stages"] if int(value["stage"]) == 2
    )
    errors = []
    if protocol.get("automatic_execution") is not False:
        errors.append("automatic_execution must remain false")
    if tuple(population["confirmatory_cells"]) != EXPECTED_CELLS:
        errors.append("confirmatory cells changed")
    if int(population["confirmatory_targets"]) != len(EXPECTED_TARGETS):
        errors.append("confirmatory target count changed")
    if primary.get("mode") != PRIMARY_MODE or int(primary.get("shots", -1)) != PRIMARY_SHOTS:
        errors.append("primary serving mode or shot count changed")
    if tuple(int(value) for value in primary.get("simulator_seeds", [])) != SIMULATOR_SEEDS:
        errors.append("primary simulator seeds changed")
    if int(protocol["reference_protocol"]["references_per_cell"]) != NUM_REFERENCES:
        errors.append("reference count changed")
    if endpoint.get("metric") != "tpr_at_1pct_fpr":
        errors.append("primary endpoint changed")
    if endpoint.get("unit") != "target checkpoint after averaging simulator seeds":
        errors.append("primary replication unit changed")
    if int(endpoint.get("bootstrap_replicates", -1)) != BOOTSTRAP_REPLICATES:
        errors.append("bootstrap replicate count changed")
    if int(endpoint.get("bootstrap_seed", -1)) != BOOTSTRAP_SEED:
        errors.append("bootstrap seed changed")
    if stage.get("subset") != "four confirmatory cells":
        errors.append("Stage-2 subset changed")
    if stage.get("scientific_results_visible") is not False:
        errors.append("Stage-2 must remain blinded until sealing")
    if errors:
        raise ValueError("Invalid frozen Stage-2 protocol: " + "; ".join(errors))


def validate_frozen_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, str]]]:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    full_path = resolve_repo_path(protocol["provenance"]["phase7_target_manifest"])
    full_rows = read_phase7_targets(full_path)
    errors = validate_protocol_lock(args.protocol, args.protocol_lock)
    errors.extend(validate_design(protocol, full_rows))
    probe = inspect_candidate_probe(
        protocol, args.protocol, full_path, args.candidate_probe
    )
    errors.extend(probe["errors"])
    snapshot = inspect_snapshot(protocol)
    errors.extend(snapshot["errors"])
    if errors:
        raise ValueError("Invalid frozen Phase-7 inputs: " + "; ".join(errors))
    validate_stage2_protocol(protocol)
    return protocol, validate_stage2_manifest(protocol, args.targets)


def validate_raw_output_seal(
    seal_path: Path, *, protocol_path: Path, target_manifest: Path
) -> dict[str, Any]:
    if not seal_path.is_file():
        raise FileNotFoundError(
            f"Stage-2 raw outputs are not sealed; expected {seal_path}"
        )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("schema_version") != 1 or seal.get("scope") != "phase7_stage2_primary":
        raise ValueError("Unrecognized Stage-2 raw-output seal")
    if seal.get("protocol_sha256") != sha256(protocol_path):
        raise ValueError("Raw-output seal is bound to a different protocol")
    if seal.get("target_manifest_sha256") != sha256(target_manifest):
        raise ValueError("Raw-output seal is bound to a different target manifest")
    files = seal.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Raw-output seal has no artifact ledger")
    errors = []
    for name, expected_hash in sorted(files.items()):
        path = Path(name)
        if not path.is_file():
            errors.append(f"missing sealed file {path}")
        elif sha256(path) != expected_hash:
            errors.append(f"hash mismatch for sealed file {path}")
    if int(seal.get("artifact_count", -1)) != len(files):
        errors.append("raw-output seal artifact count is inconsistent")
    if errors:
        raise ValueError("Invalid Stage-2 raw-output seal: " + "; ".join(errors))
    return seal


def write_cell_manifest(
    directory: Path, cell: str, rows: list[dict[str, str]]
) -> Path:
    selected = [row for row in rows if row["structural_cell_id"] == cell]
    if len(selected) != 3:
        raise ValueError(f"Expected three confirmatory targets for {cell}")
    path = directory / f"{cell}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    temporary.replace(path)
    return path


def add_cell_and_learned_contrasts(
    target_means: list[dict[str, object]], target_to_cell: dict[str, str]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    target_rows, summary = paired_contrasts(target_means, NUM_REFERENCES)
    for row in target_rows:
        row["cell"] = target_to_cell[str(row["target_id"])]

    indexed = {
        (
            str(row["target_id"]), str(row["mode"]), int(row["shots"]),
            str(row["attack"]), int(row["reference_count"]),
        ): row
        for row in target_means
    }
    for target_id in EXPECTED_TARGETS:
        left = indexed[(target_id, PRIMARY_MODE, PRIMARY_SHOTS, "affine_channel_lira", 16)]
        right = indexed[
            (target_id, PRIMARY_MODE, PRIMARY_SHOTS, "target_crossfit_learned_mia", 0)
        ]
        record: dict[str, object] = {
            "target_id": target_id,
            "cell": target_to_cell[target_id],
            "mode": PRIMARY_MODE,
            "shots": PRIMARY_SHOTS,
            "contrast": "affine_minus_learned",
            "left_attack": "affine_channel_lira",
            "right_attack": "target_crossfit_learned_mia",
        }
        for metric in METRICS:
            column = f"{metric}_mean_over_simulator_seeds"
            record[f"{metric}_difference"] = float(left[column]) - float(right[column])
        target_rows.append(record)

    learned_rows = [row for row in target_rows if row["contrast"] == "affine_minus_learned"]
    learned_summary: dict[str, object] = {
        "mode": PRIMARY_MODE,
        "shots": PRIMARY_SHOTS,
        "contrast": "affine_minus_learned",
        "n_targets": len(learned_rows),
        "replication_unit": "target checkpoint",
    }
    for metric in METRICS:
        values = np.asarray(
            [float(row[f"{metric}_difference"]) for row in learned_rows]
        )
        learned_summary[f"{metric}_difference_mean"] = float(values.mean())
        learned_summary[f"{metric}_difference_sd_targets"] = float(values.std(ddof=1))
        learned_summary[f"{metric}_difference_min"] = float(values.min())
        learned_summary[f"{metric}_difference_max"] = float(values.max())
    summary.append(learned_summary)
    return sorted(target_rows, key=lambda row: (str(row["target_id"]), str(row["contrast"]))), summary


def hierarchical_intervals(
    target_rows: list[dict[str, object]], *, replicates: int, seed: int
) -> list[dict[str, object]]:
    """Resample cells, then targets within sampled cells, after seed averaging."""
    if replicates != BOOTSTRAP_REPLICATES or seed != BOOTSTRAP_SEED:
        raise ValueError("The confirmatory hierarchical bootstrap is frozen")
    output = []
    contrasts = sorted({str(row["contrast"]) for row in target_rows})
    metrics = ("tpr_at_1pct_fpr", "auc", "tpr_at_5pct_fpr")
    for contrast in contrasts:
        selected = [row for row in target_rows if row["contrast"] == contrast]
        for metric in metrics:
            cell_values = []
            for cell in EXPECTED_CELLS:
                values = np.asarray([
                    float(row[f"{metric}_difference"])
                    for row in selected if row["cell"] == cell
                ], dtype=np.float64)
                if values.shape != (3,):
                    raise ValueError(
                        f"Hierarchical bootstrap expected three targets for {cell}/{contrast}"
                    )
                cell_values.append(values)
            matrix = np.stack(cell_values)
            namespace = int.from_bytes(
                hashlib.sha256(f"{contrast}|{metric}".encode("utf-8")).digest()[:4],
                "big",
            )
            rng = np.random.default_rng(np.random.SeedSequence([seed, namespace]))
            sampled_cells = rng.integers(0, len(EXPECTED_CELLS), size=(replicates, 4))
            sampled_targets = rng.integers(0, 3, size=(replicates, 4, 3))
            boot = matrix[sampled_cells[:, :, None], sampled_targets].mean(axis=(1, 2))
            cell_means = matrix.mean(axis=1)
            flat = matrix.reshape(-1)
            output.append({
                "contrast": contrast,
                "metric": metric,
                "estimand": "mean target-level paired difference after averaging simulator seeds",
                "n_cells": len(EXPECTED_CELLS),
                "n_targets": len(flat),
                "n_simulator_seeds_per_target": len(SIMULATOR_SEEDS),
                "point_estimate": float(flat.mean()),
                "ci95_lower": float(np.quantile(boot, 0.025)),
                "ci95_upper": float(np.quantile(boot, 0.975)),
                "positive_targets": int(np.sum(flat > 0.0)),
                "positive_cells": int(np.sum(cell_means > 0.0)),
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed,
                "substream_namespace": namespace,
            })
    return output


def find_interval(
    rows: list[dict[str, object]], contrast: str, metric: str
) -> dict[str, object]:
    matches = [
        row for row in rows
        if row["contrast"] == contrast and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Missing interval for {(contrast, metric)}")
    return matches[0]


def evaluate_gates(
    protocol: dict[str, Any], intervals: list[dict[str, object]]
) -> dict[str, Any]:
    a = find_interval(
        intervals, "matched_reference_minus_mismatched", "tpr_at_1pct_fpr"
    )
    b_tpr = find_interval(
        intervals, "affine_minus_matched_reference", "tpr_at_1pct_fpr"
    )
    b_auc = find_interval(intervals, "affine_minus_matched_reference", "auc")
    c = find_interval(intervals, "affine_minus_loss", "tpr_at_1pct_fpr")
    criteria = protocol["success_criteria"]
    tpr_margin = float(
        criteria["B_efficient_recovery"]["primary_tpr_absolute_noninferiority_margin"]
    )
    auc_margin = float(
        criteria["B_efficient_recovery"]["secondary_auc_absolute_noninferiority_margin"]
    )
    amortized_cost_ratio = (
        int(protocol["cost_accounting"]["channel_calibration_models_per_cell_amortized"])
        / int(protocol["cost_accounting"]["matched_reference_models_per_cell"])
    )
    gate_a = float(a["ci95_lower"]) > 0.0
    gate_b_tpr = float(b_tpr["ci95_lower"]) > -tpr_margin
    gate_b_auc = float(b_auc["ci95_lower"]) > -auc_margin
    gate_b_cost = amortized_cost_ratio <= 0.25
    gate_b = gate_b_tpr and gate_b_auc and gate_b_cost
    gate_c = float(c["ci95_lower"]) > 0.0
    if gate_a and gate_b and gate_c:
        category = "A+B+C: strong specialized ChannelLiRA attack paper"
    elif gate_a and gate_b:
        category = "A+B only: serving-channel mismatch and privacy-auditing paper"
    elif gate_a:
        category = "A only: revise the channel model before an attack claim"
    else:
        category = "A fails: core serving-mismatch motivation not confirmed"
    return {
        "A_channel_mismatch": gate_a,
        "B_efficient_recovery": gate_b,
        "B_components": {
            "tpr_noninferiority": gate_b_tpr,
            "auc_noninferiority": gate_b_auc,
            "amortized_cost_at_most_25pct": gate_b_cost,
        },
        "C_practical_attack": gate_c,
        "tpr_noninferiority_margin": tpr_margin,
        "auc_noninferiority_margin": auc_margin,
        "amortized_channel_to_matched_cost_ratio": amortized_cost_ratio,
        "paper_decision": category,
        "secondary_execution_warranted": bool(gate_a and gate_b),
    }


def find_attack_summary(
    rows: list[dict[str, object]], attack: str, reference_count: int
) -> dict[str, object]:
    matches = [
        row for row in rows
        if row["mode"] == PRIMARY_MODE and int(row["shots"]) == PRIMARY_SHOTS
        and row["attack"] == attack and int(row["reference_count"]) == reference_count
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Missing attack summary for {(attack, reference_count)}")
    return matches[0]


def build_report(
    summary: list[dict[str, object]],
    intervals: list[dict[str, object]],
    decision: dict[str, Any],
    protocol_hash: str,
    seal_hash: str,
) -> str:
    labels = {
        PRIMARY_MATCHED: "matched 16-reference noisy LiRA",
        "affine_channel_lira": "ChannelLiRA",
        "latent_lira_mismatched": "mismatched latent LiRA",
        "loss_mia": "loss MIA",
        "target_crossfit_learned_mia": "privileged victim-crossfit learned comparator",
    }
    lines = [
        "# Phase 7 Stage 2: confirmatory primary",
        "",
        f"**Frozen decision:** {decision['paper_decision']}.",
        "",
        "## Locked provenance",
        "",
        f"- Protocol SHA-256: `{protocol_hash}`.",
        f"- Pre-unblinding raw-output seal SHA-256: `{seal_hash}`.",
        "- Confirmatory population: four cells, three target checkpoints per cell.",
        "- Serving condition: noisy 128-shot Aer; simulator seeds 0–9.",
        "- Replication unit: target checkpoint after averaging simulator seeds.",
        "- Inference: 20,000-replicate hierarchical cell/target bootstrap.",
        "",
        "## Attack results",
        "",
        "| Attack | Mean AUC | Mean TPR@1% FPR | Mean TPR@5% FPR |",
        "|---|---:|---:|---:|",
    ]
    for attack in PRIMARY_ATTACKS:
        count = 0 if attack in {"loss_mia", "target_crossfit_learned_mia"} else 16
        row = find_attack_summary(summary, attack, count)
        lines.append(
            f"| {labels[attack]} | {float(row['auc_mean']):.4f} | "
            f"{100 * float(row['tpr_at_1pct_fpr_mean']):.2f}% | "
            f"{100 * float(row['tpr_at_5pct_fpr_mean']):.2f}% |"
        )
    gate_rows = (
        ("A: matched noisy − mismatched", "matched_reference_minus_mismatched", 0.0),
        ("B: ChannelLiRA − matched noisy", "affine_minus_matched_reference", -0.005),
        ("C: ChannelLiRA − loss", "affine_minus_loss", 0.0),
        ("Stress: ChannelLiRA − learned", "affine_minus_learned", None),
    )
    lines.extend([
        "",
        "## Frozen primary contrasts",
        "",
        "| Contrast | TPR@1% difference | Hierarchical 95% interval | Decision boundary |",
        "|---|---:|---:|---:|",
    ])
    for label, contrast, boundary in gate_rows:
        row = find_interval(intervals, contrast, "tpr_at_1pct_fpr")
        boundary_text = "descriptive only" if boundary is None else f"{100 * boundary:+.2f} pp"
        lines.append(
            f"| {label} | {100 * float(row['point_estimate']):+.2f} pp | "
            f"[{100 * float(row['ci95_lower']):+.2f}, "
            f"{100 * float(row['ci95_upper']):+.2f}] pp | {boundary_text} |"
        )
    b_auc = find_interval(intervals, "affine_minus_matched_reference", "auc")
    lines.extend([
        "",
        "## Gate decision",
        "",
        f"- **A — channel mismatch:** {'PASS' if decision['A_channel_mismatch'] else 'FAIL'}.",
        f"- **B — efficient recovery:** {'PASS' if decision['B_efficient_recovery'] else 'FAIL'}. "
        f"The AUC interval is [{float(b_auc['ci95_lower']):+.4f}, "
        f"{float(b_auc['ci95_upper']):+.4f}] against the frozen −0.0100 margin; "
        f"amortized calibration cost is "
        f"{100 * float(decision['amortized_channel_to_matched_cost_ratio']):.2f}%.",
        f"- **C — practical attack:** {'PASS' if decision['C_practical_attack'] else 'FAIL'}.",
        f"- **Secondary experiment warranted:** "
        f"{'yes' if decision['secondary_execution_warranted'] else 'no'}; only A+B authorizes it.",
        "",
        "## Claim boundary",
        "",
        "The learned comparator is unchanged and privileged; this study cannot support superiority over learned MIAs generally. The four IBM-derived simulator cells do not establish universal architecture or real-hardware leakage. A failed primary gate cannot be rescued by secondary conditions.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-lock", type=Path, default=DEFAULT_PROTOCOL_LOCK)
    parser.add_argument("--candidate-probe", type=Path, default=DEFAULT_CANDIDATE_PROBE)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--noisy-dir", type=Path, default=DEFAULT_NOISY_DIR)
    parser.add_argument("--raw-output-seal", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument(
        "--noise-augmentation-draws", type=int,
        default=NOISE_AUGMENTATION_DRAWS,
    )
    parser.add_argument(
        "--variance-shrinkage", type=float, default=VARIANCE_SHRINKAGE
    )
    args = parser.parse_args()
    for name in (
        "protocol", "protocol_lock", "candidate_probe", "targets", "reference_dir",
        "noisy_dir", "raw_output_seal", "out_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if (
        args.folds != FOLDS
        or args.noise_augmentation_draws != NOISE_AUGMENTATION_DRAWS
        or not np.isclose(args.variance_shrinkage, VARIANCE_SHRINKAGE)
    ):
        raise ValueError(
            "Stage-2 analysis parameters are frozen at folds=5, "
            "noise_augmentation_draws=32, and variance_shrinkage=0.15"
        )

    protocol, rows = validate_frozen_inputs(args)
    seal = validate_raw_output_seal(
        args.raw_output_seal,
        protocol_path=args.protocol,
        target_manifest=args.targets,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_to_cell = {row["target_id"]: row["structural_cell_id"] for row in rows}
    blocks = []
    diagnostics = []
    serving_provenance = []
    source_paths: list[Path] = []
    manifest_dir = args.out_dir / "cell_manifests"
    for cell_name in EXPECTED_CELLS:
        cell_manifest = write_cell_manifest(manifest_dir, cell_name, rows)
        target_ids = tuple(
            row["target_id"] for row in rows if row["structural_cell_id"] == cell_name
        )
        cell, matched, sources, provenance = load_scaleup_cell(
            targets_path=cell_manifest,
            reference_dir=args.reference_dir,
            noisy_dir=args.noisy_dir,
            modes=[PRIMARY_MODE],
            shots=PRIMARY_SHOTS,
            simulator_seeds=list(SIMULATOR_SEEDS),
            num_references=NUM_REFERENCES,
            expected_targets=target_ids,
            expected_cell=cell_name,
            phase_label="Phase-7 Stage-2",
        )
        bundle = reference_bundles(cell, [NUM_REFERENCES], args.variance_shrinkage)
        condition = cell.conditions[(PRIMARY_MODE, PRIMARY_SHOTS)]
        block, current_diagnostics = evaluate_leave_target_out(
            cell,
            condition,
            bundle,
            mode=PRIMARY_MODE,
            shots=PRIMARY_SHOTS,
            folds=args.folds,
            noise_augmentation_draws=args.noise_augmentation_draws,
            variance_shrinkage=args.variance_shrinkage,
            seed=BOOTSTRAP_SEED,
        )
        add_scaleup_comparators(
            block,
            cell,
            condition,
            matched[(PRIMARY_MODE, PRIMARY_SHOTS)],
            num_references=NUM_REFERENCES,
            folds=args.folds,
            seed=BOOTSTRAP_SEED,
            seed_namespace="phase7-stage2-learned",
        )
        blocks.append(block)
        diagnostics.extend(current_diagnostics)
        serving_provenance.append(provenance)
        source_paths.extend([cell_manifest, *sources])
        print(f"loaded sealed confirmatory cell {cell_name}", flush=True)

    snapshot_hashes = {
        value["snapshot_manifest_sha256"] for value in serving_provenance
    }
    if snapshot_hashes != {protocol["provenance"]["snapshot_manifest_sha256"]}:
        raise ValueError("Confirmatory cells do not share the frozen backend snapshot")

    metrics = build_metric_rows(blocks)
    target_means = target_mean_rows(metrics)
    for row in target_means:
        row["cell"] = target_to_cell[str(row["target_id"])]
    summary = summarize_targets(target_means)
    contrast_targets, contrast_summary = add_cell_and_learned_contrasts(
        target_means, target_to_cell
    )
    intervals = hierarchical_intervals(
        contrast_targets,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    decision = evaluate_gates(protocol, intervals)

    write_csv(args.out_dir / "metrics_raw.csv", metrics)
    write_csv(args.out_dir / "target_mean_metrics.csv", target_means)
    write_csv(args.out_dir / "metrics_summary.csv", summary)
    write_csv(args.out_dir / "paired_contrasts_target.csv", contrast_targets)
    write_csv(args.out_dir / "paired_contrasts_summary.csv", contrast_summary)
    write_csv(args.out_dir / "hierarchical_intervals.csv", intervals)
    write_csv(args.out_dir / "channel_diagnostics.csv", diagnostics)

    protocol_hash = sha256(args.protocol)
    seal_hash = sha256(args.raw_output_seal)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "raw_output_seal": str(args.raw_output_seal),
        "raw_output_seal_sha256": seal_hash,
        "analysis_role": "confirmatory_primary",
        "pilot_included": False,
        "cells": list(EXPECTED_CELLS),
        "targets": list(EXPECTED_TARGETS),
        "mode": PRIMARY_MODE,
        "shots": PRIMARY_SHOTS,
        "simulator_seeds": list(SIMULATOR_SEEDS),
        "simulator_seed_handling": "averaged within target before inference",
        "num_references": NUM_REFERENCES,
        "folds": args.folds,
        "noise_augmentation_draws": args.noise_augmentation_draws,
        "variance_shrinkage": args.variance_shrinkage,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "learned_mia_label": "privileged victim-crossfit learned comparator",
        "real_hardware_execution": False,
    }
    (args.out_dir / "CONFIG.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "DECISION.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sources = dict(seal["files"])
    sources[str(args.raw_output_seal)] = seal_hash
    source_paths.extend([
        Path(__file__).resolve(),
        ROOT / "experiments/channel_lira_noisy_reference_scaleup.py",
        ROOT / "experiments/channel_lira_transfer.py",
        ROOT / "channel_lira/continuous.py",
        ROOT / "channel_lira/core.py",
    ])
    for path in source_paths:
        sources[str(path.resolve())] = sha256(path)
    (args.out_dir / "SOURCE_MANIFEST.json").write_text(
        json.dumps(dict(sorted(sources.items())), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "REPORT.md").write_text(
        build_report(summary, intervals, decision, protocol_hash, seal_hash),
        encoding="utf-8",
    )
    print(f"[DECISION] {decision['paper_decision']}", flush=True)
    print(f"[REPORT] {args.out_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
