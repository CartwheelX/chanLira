#!/usr/bin/env python3
"""Analyze the Phase-7 pilot-cell engineering replication.

This analysis verifies that the enlarged 1,000/1,000 candidate population works
end to end.  Its attack results remain pilot/replication evidence and are excluded
from the frozen four-cell confirmatory endpoint.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.channel_lira_noisy_reference_scaleup import (  # noqa: E402
    METRICS,
    PRIMARY_ATTACKS,
    PRIMARY_MATCHED,
    add_scaleup_comparators,
    load_scaleup_cell,
    paired_contrasts,
    parse_csv_list,
    parse_int_list,
    sha256,
    summarize_targets,
    target_mean_rows,
)
from experiments.channel_lira_transfer import (  # noqa: E402
    build_metric_rows,
    evaluate_leave_target_out,
    reference_bundles,
)
from experiments.channel_lira_circuit_pilot import write_csv  # noqa: E402
from experiments.check_channel_lira_phase7_readiness import (  # noqa: E402
    DEFAULT_CANDIDATE_PROBE,
    DEFAULT_PROTOCOL,
    DEFAULT_PROTOCOL_LOCK,
    inspect_candidate_probe,
    read_targets as read_phase7_targets,
    resolve_repo_path,
    validate_design,
    validate_protocol_lock,
)


EXPECTED_CELL = "eff_su2_r1_d2"
EXPECTED_TARGETS = (
    "MNIST_QNN_eff_su2_r1_d2_p7_s143",
    "MNIST_QNN_eff_su2_r1_d2_p7_s144",
    "MNIST_QNN_eff_su2_r1_d2_p7_s145",
)
DEFAULT_TARGETS = ROOT / "reviewer_targets/channel_lira_phase7_stage1_pilot.csv"
DEFAULT_REFERENCE_DIR = ROOT / "channel_lira_results/phase7/references"
DEFAULT_NOISY_DIR = ROOT / "channel_lira_results/phase7/stage1_pilot/noisy_lira"
DEFAULT_OUT = ROOT / "channel_lira_results/phase7/stage1_pilot/analysis"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_stage1_manifest(
    protocol: dict[str, Any], pilot_path: Path
) -> list[dict[str, str]]:
    full_path = resolve_repo_path(protocol["provenance"]["phase7_target_manifest"])
    full_rows = read_phase7_targets(full_path)
    full_index = {row["target_id"]: row for row in full_rows}
    pilot_rows = read_csv(pilot_path)
    observed_ids = tuple(row.get("target_id", "") for row in pilot_rows)
    if observed_ids != EXPECTED_TARGETS:
        raise ValueError(
            f"Stage-1 pilot manifest must contain {EXPECTED_TARGETS}; found {observed_ids}"
        )
    for row in pilot_rows:
        if row != full_index.get(row["target_id"]):
            raise ValueError(
                f"Stage-1 target row differs from the locked full manifest: {row['target_id']}"
            )
        if row.get("phase7_analysis_role") != "pilot_replication":
            raise ValueError("Stage-1 targets must be labeled pilot_replication")
    return pilot_rows


def validate_frozen_inputs(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    full_path = resolve_repo_path(protocol["provenance"]["phase7_target_manifest"])
    full_rows = read_phase7_targets(full_path)
    errors = validate_protocol_lock(args.protocol, args.protocol_lock)
    errors.extend(validate_design(protocol, full_rows))
    probe = inspect_candidate_probe(
        protocol, args.protocol, full_path, args.candidate_probe
    )
    errors.extend(probe["errors"])
    if errors:
        raise ValueError("Invalid frozen Phase-7 inputs: " + "; ".join(errors))
    validate_stage1_manifest(protocol, args.targets)
    stage = next(
        value for value in protocol["execution_stages"] if int(value["stage"]) == 1
    )
    if stage["subset"] != "pilot cell only":
        raise ValueError("Protocol Stage 1 no longer selects the pilot cell only")
    return protocol


def find_summary(
    rows: list[dict[str, object]], mode: str, attack: str, reference_count: int
) -> dict[str, object]:
    matches = [
        row for row in rows
        if row["mode"] == mode and row["attack"] == attack
        and int(row["reference_count"]) == reference_count
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Missing Stage-1 summary for {(mode, attack, reference_count)}")
    return matches[0]


def find_contrast(
    rows: list[dict[str, object]], mode: str, contrast: str
) -> dict[str, object]:
    matches = [
        row for row in rows
        if row["mode"] == mode and row["contrast"] == contrast
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Missing Stage-1 contrast for {(mode, contrast)}")
    return matches[0]


def build_report(
    summary: list[dict[str, object]],
    contrasts: list[dict[str, object]],
    protocol_hash: str,
    snapshot_hash: str,
) -> str:
    labels = {
        PRIMARY_MATCHED: "matched 16-reference LiRA",
        "affine_channel_lira": "leave-target-out ChannelLiRA",
        "latent_lira_mismatched": "mismatched latent LiRA",
        "loss_mia": "loss MIA",
        "target_crossfit_learned_mia": "privileged victim-crossfit learned comparator",
    }
    lines = [
        "# Phase 7 Stage 1: pilot-cell engineering replication",
        "",
        "This stage tests the enlarged candidate/checkpoint/noisy-serving pipeline. It is pilot replication evidence and is excluded from the four-cell confirmatory primary endpoint.",
        "",
        "## Locked provenance",
        "",
        f"- Protocol SHA-256: `{protocol_hash}`.",
        f"- Snapshot manifest SHA-256: `{snapshot_hash}`.",
        "- Candidates: 1,000 members and 1,000 nonmembers per target.",
        "- References: sixteen, exactly 8 IN / 8 OUT per candidate.",
        "- Conditions: ideal/noisy 128 shots; simulator seeds 0 and 1.",
        "- No real quantum-hardware execution.",
        "",
        "## Descriptive attack results",
        "",
        "Means first average serving seeds within each independently initialized target. SD is descriptive across three targets.",
        "",
        "| Mode | Attack | Mean AUC | Mean TPR@1% FPR | SD AUC across targets |",
        "|---|---|---:|---:|---:|",
    ]
    for mode in ("ideal_shot", "noisy_shot"):
        for attack in PRIMARY_ATTACKS:
            count = 0 if attack in {"loss_mia", "target_crossfit_learned_mia"} else 16
            row = find_summary(summary, mode, attack, count)
            lines.append(
                f"| {mode} | {labels[attack]} | {float(row['auc_mean']):.4f} | "
                f"{100 * float(row['tpr_at_1pct_fpr_mean']):.2f}% | "
                f"{float(row['auc_sd_targets']):.4f} |"
            )
    lines.extend([
        "",
        "## Noisy paired contrasts",
        "",
        "Ranges are the three observed targets, not confidence intervals.",
        "",
        "| Contrast | Mean AUC difference | Mean TPR@1% difference | Target AUC range | Target TPR@1% range |",
        "|---|---:|---:|---:|---:|",
    ])
    selected = {
        "affine_minus_matched_reference",
        "matched_reference_minus_mismatched",
        "affine_minus_mismatched",
        "affine_minus_loss",
    }
    for row in contrasts:
        if row["mode"] != "noisy_shot" or row["contrast"] not in selected:
            continue
        lines.append(
            f"| {row['contrast']} | {float(row['auc_difference_mean']):+.4f} | "
            f"{100 * float(row['tpr_at_1pct_fpr_difference_mean']):+.2f} pp | "
            f"[{float(row['auc_difference_min']):+.4f}, "
            f"{float(row['auc_difference_max']):+.4f}] | "
            f"[{100 * float(row['tpr_at_1pct_fpr_difference_min']):+.2f}, "
            f"{100 * float(row['tpr_at_1pct_fpr_difference_max']):+.2f}] pp |"
        )
    mechanism = find_contrast(
        contrasts, "noisy_shot", "matched_reference_minus_mismatched"
    )
    recovery = find_contrast(
        contrasts, "noisy_shot", "affine_minus_matched_reference"
    )
    utility = find_contrast(contrasts, "noisy_shot", "affine_minus_loss")
    lines.extend([
        "",
        "## Directional pilot check against the frozen success conditions",
        "",
        "This is a descriptive three-target check, not a hypothesis test or confirmatory decision.",
        "",
        f"- **A, channel mismatch:** matched noisy LiRA minus mismatched LiRA is "
        f"{100 * float(mechanism['tpr_at_1pct_fpr_difference_mean']):+.2f} pp "
        f"TPR@1% FPR, with observed target range "
        f"[{100 * float(mechanism['tpr_at_1pct_fpr_difference_min']):+.2f}, "
        f"{100 * float(mechanism['tpr_at_1pct_fpr_difference_max']):+.2f}] pp. "
        "The primary-metric direction is supportive, although AUC is not.",
        f"- **B, efficient recovery:** ChannelLiRA minus matched noisy LiRA is "
        f"{100 * float(recovery['tpr_at_1pct_fpr_difference_mean']):+.2f} pp, "
        f"with observed range "
        f"[{100 * float(recovery['tpr_at_1pct_fpr_difference_min']):+.2f}, "
        f"{100 * float(recovery['tpr_at_1pct_fpr_difference_max']):+.2f}] pp. "
        "That is directionally compatible with the frozen -0.5 pp margin, at 12.5% of matched-reference auxiliary shot cost per attack.",
        f"- **C, practical utility:** ChannelLiRA minus loss MIA is "
        f"{100 * float(utility['tpr_at_1pct_fpr_difference_mean']):+.2f} pp, "
        f"with observed range "
        f"[{100 * float(utility['tpr_at_1pct_fpr_difference_min']):+.2f}, "
        f"{100 * float(utility['tpr_at_1pct_fpr_difference_max']):+.2f}] pp. "
        "This pilot does not show superiority over loss MIA.",
    ])
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "These results cannot confirm or change the frozen Phase-7 primary endpoint. Stage 1 passes scientifically when artifacts, hashes, balance, cache reuse, metrics, and cost receipts are complete; attack superiority is not an engineering pass requirement.",
        "",
        "The learned comparator remains the unchanged privileged victim-crossfit implementation and does not support claims against learned MIAs generally.",
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
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--noise-augmentation-draws", type=int, default=32)
    parser.add_argument("--variance-shrinkage", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    for name in (
        "protocol", "protocol_lock", "candidate_probe", "targets",
        "reference_dir", "noisy_dir", "out_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    protocol = validate_frozen_inputs(args)
    modes = parse_csv_list("ideal_shot,noisy_shot")
    simulator_seeds = parse_int_list("0,1")
    cell, matched, source_paths, serving_provenance = load_scaleup_cell(
        targets_path=args.targets,
        reference_dir=args.reference_dir,
        noisy_dir=args.noisy_dir,
        modes=modes,
        shots=128,
        simulator_seeds=simulator_seeds,
        num_references=16,
        expected_targets=EXPECTED_TARGETS,
        expected_cell=EXPECTED_CELL,
        phase_label="Phase-7 Stage-1",
    )
    bundles = reference_bundles(cell, [16], args.variance_shrinkage)
    blocks = []
    diagnostics = []
    for mode in modes:
        condition = cell.conditions[(mode, 128)]
        block, rows = evaluate_leave_target_out(
            cell,
            condition,
            bundles,
            mode=mode,
            shots=128,
            folds=args.folds,
            noise_augmentation_draws=args.noise_augmentation_draws,
            variance_shrinkage=args.variance_shrinkage,
            seed=args.seed,
        )
        add_scaleup_comparators(
            block,
            cell,
            condition,
            matched[(mode, 128)],
            num_references=16,
            folds=args.folds,
            seed=args.seed,
            seed_namespace="phase7-stage1-learned",
        )
        blocks.append(block)
        diagnostics.extend(rows)
        print(f"completed Phase-7 Stage-1 analysis {mode}", flush=True)

    metrics = build_metric_rows(blocks)
    target_means = target_mean_rows(metrics)
    summary = summarize_targets(target_means)
    contrast_targets, contrast_summary = paired_contrasts(target_means, 16)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "metrics_raw.csv", metrics)
    write_csv(args.out_dir / "target_mean_metrics.csv", target_means)
    write_csv(args.out_dir / "metrics_summary.csv", summary)
    write_csv(args.out_dir / "paired_contrasts_target.csv", contrast_targets)
    write_csv(args.out_dir / "paired_contrasts_summary.csv", contrast_summary)
    write_csv(args.out_dir / "channel_diagnostics.csv", diagnostics)
    source_paths.extend([
        args.protocol, args.protocol_lock, args.candidate_probe,
        resolve_repo_path(protocol["provenance"]["backend_snapshot"])
        / "snapshot_manifest.json",
    ])
    source_manifest = {
        str(path.resolve()): sha256(path)
        for path in sorted(set(source_paths), key=lambda value: str(value))
    }
    (args.out_dir / "SOURCE_MANIFEST.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    protocol_hash = sha256(args.protocol)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "targets": list(cell.target_ids),
        "cell": cell.name,
        "analysis_role": "pilot_replication_engineering_canary",
        "confirmatory_primary_included": False,
        "modes": modes,
        "shots": 128,
        "simulator_seeds": simulator_seeds,
        "num_references": 16,
        "folds": args.folds,
        "learned_mia_label": "privileged victim-crossfit learned comparator",
        "serving_provenance": serving_provenance,
        "real_hardware_execution": False,
    }
    (args.out_dir / "CONFIG.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "REPORT.md").write_text(
        build_report(
            summary,
            contrast_summary,
            protocol_hash,
            serving_provenance["snapshot_manifest_sha256"],
        ),
        encoding="utf-8",
    )
    print(f"[REPORT] {args.out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
