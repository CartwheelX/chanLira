#!/usr/bin/env python3
"""Run the frozen Phase-6 three-target/16-reference noisy scale-up.

The scale-up is a resumable compute and comparison gate between the completed
four-reference canary and the full 15-target/80-reference study.  It reuses one
hash-validated frozen backend snapshot and never submits a hardware job.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.channel_lira_noisy_reference_scaleup import (  # noqa: E402
    EXPECTED_TARGETS,
    read_targets,
)
from experiments.run_channel_lira_noisy_reference_canary import (  # noqa: E402
    inspect_reference_bank,
    inspect_snapshot,
    reconstruction_diagnostic,
    sha256,
    target_artifacts,
)


DEFAULT_TARGETS = ROOT / "reviewer_targets/channel_lira_noisy_reference_scaleup.csv"
DEFAULT_OUT = ROOT / "channel_lira_results/noisy_reference_scaleup_phase6"
DEFAULT_SNAPSHOT = (
    ROOT / "channel_lira_results/noisy_reference_canary_phase5/backend_snapshot"
)


@lru_cache(maxsize=4)
def inspect_snapshot_cached(path: str) -> dict[str, Any]:
    return inspect_snapshot(Path(path))


def run(command: list[str], *, cwd: Path) -> None:
    print("[RUN] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def inspect_exact_scores(
    rows: list[dict[str, str]], reference_dir: Path
) -> dict[str, Any]:
    aggregate = reference_dir / "lira_reference_mia_raw.csv"
    records = []
    for row in rows:
        path = reference_dir / "sample_scores" / f"{row['target_id']}.npz"
        record: dict[str, Any] = {
            "target_id": row["target_id"],
            "path": str(path.resolve()),
            "ready": False,
        }
        if path.is_file() and path.stat().st_size > 0:
            try:
                with np.load(path, allow_pickle=False) as saved:
                    required = {
                        "sample_ids", "membership", "labels", "probabilities",
                        "observed_log_odds",
                    }
                    missing = sorted(required - set(saved.files))
                    if missing:
                        raise ValueError(f"missing fields {missing}")
                    if saved["probabilities"].shape != (400, 4):
                        raise ValueError("probability payload must have shape (400, 4)")
                    record.update({
                        "ready": True,
                        "candidate_count": int(len(saved["membership"])),
                        "sha256": sha256(path),
                    })
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    return {
        "ready": bool(
            aggregate.is_file() and aggregate.stat().st_size > 0
            and all(record["ready"] for record in records)
        ),
        "aggregate": str(aggregate.resolve()),
        "aggregate_ready": aggregate.is_file() and aggregate.stat().st_size > 0,
        "targets_ready": sum(bool(record["ready"]) for record in records),
        "expected": len(rows),
        "targets": records,
    }


def inspect_noisy_scores(
    rows: list[dict[str, str]],
    noisy_dir: Path,
    modes: list[str],
    simulator_seeds: list[int],
) -> dict[str, Any]:
    records = []
    for row in rows:
        target_id = row["target_id"]
        score = noisy_dir / "target_scores" / f"{target_id}.csv"
        metadata = noisy_dir / "metadata" / f"{target_id}.json"
        payloads = [
            noisy_dir / "sample_scores" / f"{target_id}_{mode}_sim{seed}.npz"
            for mode in modes for seed in simulator_seeds
        ]
        errors = []
        for path in payloads:
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing {path.name}")
                continue
            try:
                with np.load(path, allow_pickle=False) as saved:
                    required = {
                        "sample_ids", "membership", "labels", "probabilities",
                        "observed_log_odds", "lira_online",
                        "lira_online_fixed_variance", "lira_offline",
                        "lira_offline_fixed_variance",
                    }
                    missing = sorted(required - set(saved.files))
                    if missing:
                        raise ValueError(f"missing fields {missing}")
                    if saved["probabilities"].shape != (400, 4):
                        raise ValueError("probability payload must have shape (400, 4)")
            except Exception as exc:
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
        ready = bool(
            score.is_file() and score.stat().st_size > 0
            and metadata.is_file() and metadata.stat().st_size > 0
            and not errors
        )
        records.append({
            "target_id": target_id,
            "ready": ready,
            "score": str(score.resolve()),
            "metadata": str(metadata.resolve()),
            "payload_count": len(payloads) - len(errors),
            "expected_payload_count": len(payloads),
            "errors": errors,
        })
    aggregate = noisy_dir / "noisy_lira_raw.csv"
    summary = noisy_dir / "noisy_lira_summary.csv"
    aggregate_ready = all(
        path.is_file() and path.stat().st_size > 0 for path in (aggregate, summary)
    )
    return {
        "ready": bool(aggregate_ready and all(record["ready"] for record in records)),
        "targets_ready": sum(bool(record["ready"]) for record in records),
        "expected": len(rows),
        "aggregate_ready": aggregate_ready,
        "aggregate": str(aggregate.resolve()),
        "summary": str(summary.resolve()),
        "targets": records,
    }


def inspect_analysis(out_dir: Path) -> dict[str, Any]:
    analysis_dir = out_dir / "analysis"
    required = (
        "CONFIG.json",
        "SOURCE_MANIFEST.json",
        "REPORT.md",
        "metrics_raw.csv",
        "target_mean_metrics.csv",
        "metrics_summary.csv",
        "paired_contrasts_target.csv",
        "paired_contrasts_summary.csv",
        "channel_diagnostics.csv",
    )
    paths = {name: analysis_dir / name for name in required}
    present = {
        name: path.is_file() and path.stat().st_size > 0 for name, path in paths.items()
    }
    plot_names = (
        "attack_auc_by_mode.svg",
        "noisy_target_auc.svg",
        "noisy_paired_contrasts.svg",
        "noisy_minus_ideal_auc.svg",
    )
    plots = {name: analysis_dir / "plots" / name for name in plot_names}
    plot_present = {
        name: path.is_file() and path.stat().st_size > 0 for name, path in plots.items()
    }
    return {
        "ready": all(present.values()),
        "plots_ready": all(plot_present.values()),
        "directory": str(analysis_dir.resolve()),
        "present": present,
        "plot_present": plot_present,
    }


def build_status(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_targets(args.targets)
    targets = [target_artifacts(row, args.run_root) for row in rows]
    references = inspect_reference_bank(rows[0], args.reference_dir, args.num_references)
    snapshot = inspect_snapshot_cached(str(args.snapshot))
    exact = inspect_exact_scores(rows, args.reference_dir)
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    simulator_seeds = [
        int(value.strip()) for value in args.simulator_seeds.split(",") if value.strip()
    ]
    noisy = inspect_noisy_scores(rows, args.noisy_dir, modes, simulator_seeds)
    analysis = inspect_analysis(args.out_dir)
    reconstructions = [reconstruction_diagnostic(row, args.out_dir) for row in rows]
    target_count = sum(bool(item["ready"]) for item in targets)
    blockers = []
    if target_count != len(rows):
        blockers.append(f"target checkpoint bundles incomplete ({target_count}/{len(rows)})")
    if not references["ready"]:
        blockers.append(
            f"16-reference bank incomplete ({references['checkpoints_ready']}/"
            f"{args.num_references} checkpoints)"
        )
    if not snapshot["ready"]:
        blockers.append("shared hash-validated backend snapshot is unavailable")
    ready_for_exact = target_count == len(rows) and references["ready"]
    ready_to_score = ready_for_exact and snapshot["ready"] and exact["ready"]
    return {
        "protocol": "Phase-6 one-cell scale gate: 3 targets, 16 references, 128 shots, simulator seeds 0,1",
        "statistical_scope": "scale/comparison gate; not cross-cell publication evidence",
        "target_manifest": str(args.targets.resolve()),
        "target_manifest_sha256": sha256(args.targets),
        "targets_expected": len(rows),
        "target_ids": [row["target_id"] for row in rows],
        "target_bundles_ready": target_count,
        "target_artifacts": targets,
        "num_references": args.num_references,
        "reference_bank": references,
        "snapshot": snapshot,
        "exact_scores": exact,
        "noisy_scores": noisy,
        "analysis": analysis,
        "reconstruction_diagnostics": reconstructions,
        "shots": args.shots,
        "modes": modes,
        "simulator_seeds": simulator_seeds,
        "ready_for_exact_scoring": ready_for_exact,
        "ready_to_score": ready_to_score,
        "ready_to_analyze": noisy["ready"],
        "complete": bool(noisy["ready"] and analysis["ready"] and analysis["plots_ready"]),
        "blockers": blockers,
    }


def write_status(status: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "STATUS.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    state = (
        "COMPLETE" if status["complete"] else
        "READY TO ANALYZE" if status["ready_to_analyze"] else
        "READY TO SCORE" if status["ready_to_score"] else
        "READY FOR EXACT SCORING" if status["ready_for_exact_scoring"] else
        "PREPARING"
    )
    refs = status["reference_bank"]
    exact = status["exact_scores"]
    noisy = status["noisy_scores"]
    analysis = status["analysis"]
    lines = [
        "# Phase-6 noisy-reference scale-up execution",
        "",
        f"## Status: {state}",
        "",
        "| Gate | Available | Required |",
        "|---|---:|---:|",
        f"| Target checkpoint bundles | {status['target_bundles_ready']} | {status['targets_expected']} |",
        f"| Exact reference score files | {refs['scores_ready']} | {refs['expected']} |",
        f"| Reference checkpoints | {refs['checkpoints_ready']} | {refs['expected']} |",
        f"| Valid reference metadata | {refs['checkpoint_metadata_valid']} | {refs['expected']} |",
        f"| Balanced 16-reference bank | {int(refs['balanced_inclusion'])} | 1 |",
        f"| Shared frozen noise snapshot | {int(status['snapshot']['ready'])} | 1 |",
        f"| Exact target score payloads | {exact['targets_ready']} | {exact['expected']} |",
        f"| Noisy target score bundles | {noisy['targets_ready']} | {noisy['expected']} |",
        f"| Comparison analysis | {int(analysis['ready'])} | 1 |",
        f"| Plot bundle | {int(analysis['plots_ready'])} | 1 |",
        "",
        "## Protocol",
        "",
        "- One prespecified structural cell, with target seeds 43, 44, and 45.",
        "- Sixteen balanced references retained as exact-score/checkpoint pairs.",
        "- Ideal-shot and IBM-derived noisy-shot execution at 128 shots and simulator seeds 0,1.",
        "- Strict leave-target-out ChannelLiRA comparison; no hardware circuit submission.",
        "- This is a scale gate before the full 15-target/80-reference study.",
    ]
    if status["blockers"]:
        lines.extend(["", "## Current blockers", ""])
        lines.extend(f"- {value}" for value in status["blockers"])
    lines.extend([
        "", "## Artifacts", "",
        f"- Machine status: `{(out_dir / 'STATUS.json').resolve()}`",
        f"- Scientific analysis: `{(out_dir / 'analysis/REPORT.md').resolve()}`",
        f"- Plot index: `{(out_dir / 'analysis/PLOTS.md').resolve()}`",
    ])
    (out_dir / "EXECUTION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def train_targets(args: argparse.Namespace) -> None:
    run([
        args.python,
        str(ROOT / "reviewer_tools/run_target_table_dgx.py"),
        "--targets", str(args.targets),
        "--repo-root", str(ROOT),
        "--out", str(args.run_root),
        "--gpus", args.target_gpus,
        "--jobs-per-gpu", "1",
        "--cpu-threads", str(args.cpu_threads),
        "--no-largest-first",
        "--resume",
    ], cwd=ROOT)


def train_references(args: argparse.Namespace) -> None:
    run([
        args.python,
        str(ROOT / "reviewer_tools/run_lira_reference_multigpu.py"),
        "--targets", str(args.targets),
        "--repo-root", str(ROOT),
        "--run-root", str(args.run_root),
        "--out-dir", str(args.reference_dir),
        "--num-references", str(args.num_references),
        "--save-reference-checkpoints",
        "--phase", "train",
        "--seed", str(args.seed),
        "--gpus", args.reference_gpus,
        "--jobs-per-gpu", "1",
        "--cpu-threads", str(args.cpu_threads),
        "--resume",
    ], cwd=ROOT)


def exact_score(args: argparse.Namespace) -> None:
    status = build_status(args)
    if not status["ready_for_exact_scoring"]:
        raise RuntimeError(
            "Phase-6 targets/references are not ready for exact scoring: "
            + "; ".join(status["blockers"])
        )
    run([
        args.python,
        str(ROOT / "reviewer_tools/run_lira_reference_multigpu.py"),
        "--targets", str(args.targets),
        "--repo-root", str(ROOT),
        "--run-root", str(args.run_root),
        "--out-dir", str(args.reference_dir),
        "--num-references", str(args.num_references),
        "--phase", "score",
        "--bootstrap", str(args.bootstrap),
        "--seed", str(args.seed),
        "--gpus", args.target_gpus,
        "--jobs-per-gpu", "1",
        "--cpu-threads", str(args.cpu_threads),
        "--resume",
    ], cwd=ROOT)


def noisy_score(args: argparse.Namespace) -> None:
    status = build_status(args)
    if not status["ready_to_score"]:
        raise RuntimeError(
            "Phase-6 scale-up is not ready for noisy scoring: "
            + "; ".join(status["blockers"])
        )
    for target_id in EXPECTED_TARGETS:
        run([
            args.python,
            str(ROOT / "satml_tools/noisy_lira.py"),
            "--repo-root", str(ROOT),
            "--targets", str(args.targets),
            "--run-root", str(args.run_root),
            "--reference-dir", str(args.reference_dir),
            "--out-dir", str(args.noisy_dir),
            "--snapshot", str(args.snapshot),
            "--target-id", target_id,
            "--num-references", str(args.num_references),
            "--modes", args.modes,
            "--shots", str(args.shots),
            "--simulator-seeds", args.simulator_seeds,
            "--bootstrap", str(args.bootstrap),
            "--seed", str(args.seed),
            "--device", args.score_device,
            "--aer-max-parallel-threads", str(args.aer_max_parallel_threads),
            "--resume",
        ], cwd=ROOT)
    run([
        args.python,
        str(ROOT / "satml_tools/noisy_lira.py"),
        "--repo-root", str(ROOT),
        "--targets", str(args.targets),
        "--run-root", str(args.run_root),
        "--reference-dir", str(args.reference_dir),
        "--out-dir", str(args.noisy_dir),
        "--snapshot", str(args.snapshot),
        "--aggregate",
    ], cwd=ROOT)


def analyze(args: argparse.Namespace) -> None:
    status = build_status(args)
    if not status["ready_to_analyze"]:
        raise RuntimeError("Noisy target/reference score bundles are incomplete")
    run([
        args.python,
        str(ROOT / "experiments/channel_lira_noisy_reference_scaleup.py"),
        "--targets", str(args.targets),
        "--reference-dir", str(args.reference_dir),
        "--noisy-dir", str(args.noisy_dir),
        "--out-dir", str(args.out_dir / "analysis"),
        "--modes", args.modes,
        "--shots", str(args.shots),
        "--simulator-seeds", args.simulator_seeds,
        "--num-references", str(args.num_references),
        "--folds", str(args.folds),
        "--seed", str(args.seed),
    ], cwd=ROOT)


def plot(args: argparse.Namespace) -> None:
    if not inspect_analysis(args.out_dir)["ready"]:
        raise RuntimeError("Phase-6 analysis is incomplete; run the analyze stage first")
    run([
        args.python,
        str(ROOT / "experiments/plot_channel_lira_noisy_reference_scaleup.py"),
        "--analysis-dir", str(args.out_dir / "analysis"),
        "--png",
    ], cwd=ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("status", "target", "references", "exact", "score", "analyze", "plot", "all"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--noisy-dir", type=Path, default=None)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--num-references", type=int, default=16)
    parser.add_argument("--modes", default="ideal_shot,noisy_shot")
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--simulator-seeds", default="0,1")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--target-gpus", default="0,1")
    parser.add_argument("--reference-gpus", default="0,1")
    parser.add_argument("--score-device", default="cuda")
    parser.add_argument("--aer-max-parallel-threads", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    args.targets = args.targets.resolve()
    args.out_dir = args.out_dir.resolve()
    args.run_root = (args.run_root or args.out_dir / "runs").resolve()
    args.reference_dir = (args.reference_dir or args.out_dir / "references").resolve()
    args.noisy_dir = (args.noisy_dir or args.out_dir / "noisy_lira").resolve()
    args.snapshot = args.snapshot.resolve()
    if args.num_references != 16:
        parser.error("The Phase-6 scale-up is frozen to exactly 16 references")
    if args.shots != 128 or args.simulator_seeds != "0,1":
        parser.error("The Phase-6 scale gate is frozen to 128 shots and simulator seeds 0,1")
    if args.modes != "ideal_shot,noisy_shot":
        parser.error("The Phase-6 scale gate requires ideal_shot,noisy_shot")
    if args.aer_max_parallel_threads < 1:
        parser.error("--aer-max-parallel-threads must be positive")
    read_targets(args.targets)
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"target", "all"}:
        train_targets(args)
    if args.stage in {"references", "all"}:
        train_references(args)
    status = build_status(args)
    if args.stage == "exact" or (
        args.stage == "all" and status["ready_for_exact_scoring"]
        and not status["exact_scores"]["ready"]
    ):
        exact_score(args)
    status = build_status(args)
    if args.stage == "score" or (
        args.stage == "all" and status["ready_to_score"]
        and not status["noisy_scores"]["ready"]
    ):
        noisy_score(args)
    status = build_status(args)
    if args.stage == "analyze" or (
        args.stage == "all" and status["ready_to_analyze"]
        and not status["analysis"]["ready"]
    ):
        analyze(args)
    status = build_status(args)
    if args.stage == "plot" or (
        args.stage == "all" and status["analysis"]["ready"]
        and not status["analysis"]["plots_ready"]
    ):
        plot(args)
    status = build_status(args)
    write_status(status, args.out_dir)
    print(
        f"[STATUS] complete={status['complete']} "
        f"targets={status['target_bundles_ready']}/{status['targets_expected']} "
        f"references={status['reference_bank']['checkpoints_ready']}/{args.num_references}"
    )
    print(f"[REPORT] {args.out_dir / 'EXECUTION_REPORT.md'}")
    if args.require_complete and not status["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
