#!/usr/bin/env python3
"""Run the protocol-locked Phase-7 Stage-1 pilot replication.

Only the pilot cell is reachable from this runner.  It is resumable, requires an
explicit acknowledgement of the external protocol hash for every compute stage,
and never submits a real-hardware circuit job.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.channel_lira_phase7_stage1 import (  # noqa: E402
    EXPECTED_TARGETS,
    validate_stage1_manifest,
)
from experiments.check_channel_lira_phase7_readiness import (  # noqa: E402
    DEFAULT_CANDIDATE_PROBE,
    DEFAULT_PROTOCOL,
    DEFAULT_PROTOCOL_LOCK,
    inspect_candidate_probe,
    inspect_execution_artifacts,
    inspect_snapshot,
    read_targets,
    resolve_repo_path,
    sha256,
    validate_design,
    validate_protocol_lock,
)


DEFAULT_TARGETS = ROOT / "reviewer_targets/channel_lira_phase7_stage1_pilot.csv"
DEFAULT_OUT = ROOT / "channel_lira_results/phase7/stage1_pilot"
DEFAULT_RUN_ROOT = ROOT / "reviewer_runs"
DEFAULT_REFERENCE_DIR = ROOT / "channel_lira_results/phase7/references"
DEFAULT_SNAPSHOT = ROOT / "channel_lira_results/noisy_reference_canary_phase5/backend_snapshot"
MODES = ("ideal_shot", "noisy_shot")
SIMULATOR_SEEDS = (0, 1)
SHOTS = 128
NUM_REFERENCES = 16


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_locked_stage(args: argparse.Namespace, *, require_ack: bool) -> dict[str, Any]:
    protocol = read_json(args.protocol)
    full_path = resolve_repo_path(protocol["provenance"]["phase7_target_manifest"])
    full_rows = read_targets(full_path)
    errors = validate_protocol_lock(args.protocol, args.protocol_lock)
    errors.extend(validate_design(protocol, full_rows))
    probe = inspect_candidate_probe(
        protocol, args.protocol, full_path, args.candidate_probe
    )
    errors.extend(probe["errors"])
    snapshot = inspect_snapshot(protocol)
    errors.extend(snapshot["errors"])
    validate_stage1_manifest(protocol, args.targets)
    primary = protocol["serving_protocol"]["primary"]
    stage = next(
        value for value in protocol["execution_stages"] if int(value["stage"]) == 1
    )
    if protocol.get("automatic_execution") is not False:
        errors.append("protocol automatic_execution guard changed")
    if stage.get("subset") != "pilot cell only":
        errors.append("Stage 1 is no longer restricted to the pilot cell")
    if tuple(stage.get("modes", [])) != MODES:
        errors.append("Stage-1 serving modes differ from the locked protocol")
    if tuple(int(value) for value in stage.get("shots", [])) != (SHOTS,):
        errors.append("Stage-1 shot count differs from the locked protocol")
    if tuple(int(value) for value in stage.get("simulator_seeds", [])) != SIMULATOR_SEEDS:
        errors.append("Stage-1 simulator seeds differ from the locked protocol")
    frozen_snapshot = resolve_repo_path(protocol["provenance"]["backend_snapshot"]).resolve()
    if args.snapshot != frozen_snapshot:
        errors.append("Stage-1 snapshot path differs from the locked protocol")
    if int(primary["shots"]) != SHOTS:
        errors.append("the frozen primary shot count changed")
    if int(protocol["reference_protocol"]["references_per_cell"]) != NUM_REFERENCES:
        errors.append("the frozen reference count changed")
    expected_hash = sha256(args.protocol)
    if require_ack and args.acknowledge_protocol_hash != expected_hash:
        errors.append(
            "compute requires --acknowledge-protocol-hash equal to the locked protocol SHA-256"
        )
    if errors:
        raise ValueError("Phase-7 Stage-1 guard failed: " + "; ".join(errors))
    return protocol


def record_timing(out_dir: Path, label: str, seconds: float, command: list[str]) -> None:
    path = out_dir / "STAGE_TIMINGS.json"
    records = [] if not path.is_file() else read_json(path).get("records", [])
    records.append({
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "seconds": seconds,
        "command": command,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"records": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(command: list[str], *, label: str, out_dir: Path) -> None:
    print("[RUN] " + " ".join(command), flush=True)
    started = time.monotonic()
    subprocess.run(command, cwd=ROOT, check=True)
    record_timing(out_dir, label, time.monotonic() - started, command)


def inspect_exact_scores(
    target_ids: tuple[str, ...], reference_dir: Path, candidate_count: int
) -> dict[str, Any]:
    records = []
    for target_id in target_ids:
        path = reference_dir / "sample_scores" / f"{target_id}.npz"
        record: dict[str, Any] = {
            "target_id": target_id,
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
                        raise ValueError(f"missing arrays {missing}")
                    if saved["probabilities"].shape != (candidate_count, 4):
                        raise ValueError("probability shape differs from frozen candidates")
                    membership = np.asarray(saved["membership"], dtype=int)
                    if sorted(np.bincount(membership, minlength=2).tolist()) != [
                        candidate_count // 2, candidate_count // 2,
                    ]:
                        raise ValueError("membership population is not balanced")
                    record.update({"ready": True, "sha256": sha256(path)})
            except (KeyError, OSError, TypeError, ValueError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    aggregate = reference_dir / "lira_reference_mia_raw.csv"
    return {
        "ready": bool(
            aggregate.is_file() and aggregate.stat().st_size > 0
            and all(record["ready"] for record in records)
        ),
        "aggregate": str(aggregate.resolve()),
        "aggregate_ready": aggregate.is_file() and aggregate.stat().st_size > 0,
        "targets_ready": sum(bool(record["ready"]) for record in records),
        "expected": len(records),
        "targets": records,
    }


def inspect_noisy_scores(
    target_ids: tuple[str, ...], noisy_dir: Path, candidate_count: int
) -> dict[str, Any]:
    records = []
    for target_id in target_ids:
        errors = []
        payloads = []
        for mode in MODES:
            for simulator_seed in SIMULATOR_SEEDS:
                path = noisy_dir / "sample_scores" / (
                    f"{target_id}_{mode}_sim{simulator_seed}.npz"
                )
                payloads.append(path)
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
                            raise ValueError(f"missing arrays {missing}")
                        if saved["probabilities"].shape != (candidate_count, 4):
                            raise ValueError("probability shape differs from frozen candidates")
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
        score = noisy_dir / "target_scores" / f"{target_id}.csv"
        metadata = noisy_dir / "metadata" / f"{target_id}.json"
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
            "payloads_ready": len(payloads) - len(errors),
            "payloads_expected": len(payloads),
            "errors": errors,
        })
    aggregate = noisy_dir / "noisy_lira_raw.csv"
    summary = noisy_dir / "noisy_lira_summary.csv"
    aggregate_ready = all(path.is_file() and path.stat().st_size > 0 for path in (aggregate, summary))
    return {
        "ready": bool(aggregate_ready and all(record["ready"] for record in records)),
        "aggregate_ready": aggregate_ready,
        "targets_ready": sum(bool(record["ready"]) for record in records),
        "expected": len(records),
        "targets": records,
    }


def inspect_analysis(out_dir: Path) -> dict[str, Any]:
    directory = out_dir / "analysis"
    names = (
        "CONFIG.json", "SOURCE_MANIFEST.json", "REPORT.md", "metrics_raw.csv",
        "target_mean_metrics.csv", "metrics_summary.csv",
        "paired_contrasts_target.csv", "paired_contrasts_summary.csv",
        "channel_diagnostics.csv",
    )
    present = {
        name: (directory / name).is_file() and (directory / name).stat().st_size > 0
        for name in names
    }
    return {"ready": all(present.values()), "directory": str(directory.resolve()), "present": present}


def build_status(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    pilot_rows = validate_stage1_manifest(protocol, args.targets)
    artifacts = inspect_execution_artifacts(
        protocol, pilot_rows, args.run_root, args.reference_dir
    )
    snapshot = inspect_snapshot(protocol)
    candidate_count = int(
        protocol["study_population"]["candidate_protocol"]["candidates_per_target"]
    )
    exact = inspect_exact_scores(EXPECTED_TARGETS, args.reference_dir, candidate_count)
    noisy = inspect_noisy_scores(EXPECTED_TARGETS, args.noisy_dir, candidate_count)
    analysis = inspect_analysis(args.out_dir)
    target_ready = artifacts["target_bundles_ready"] == len(EXPECTED_TARGETS)
    refs_ready = artifacts["reference_banks_ready"] == 1
    ready_for_exact = target_ready and refs_ready
    ready_to_score = ready_for_exact and exact["ready"] and snapshot["ready"]
    blockers = []
    if not target_ready:
        blockers.append(
            f"pilot target bundles incomplete ({artifacts['target_bundles_ready']}/3)"
        )
    if not refs_ready:
        blockers.append(
            "pilot reference bank incomplete "
            f"({artifacts['reference_checkpoints_ready']}/16 checkpoints)"
        )
    if not snapshot["ready"]:
        blockers.append("frozen snapshot is not hash-valid")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(args.protocol),
        "analysis_role": "pilot_replication_engineering_canary",
        "confirmatory_primary_included": False,
        "target_manifest": str(args.targets),
        "target_manifest_sha256": sha256(args.targets),
        "target_ids": list(EXPECTED_TARGETS),
        "candidate_count": candidate_count,
        "members": candidate_count // 2,
        "nonmembers": candidate_count // 2,
        "modes": list(MODES),
        "shots": SHOTS,
        "simulator_seeds": list(SIMULATOR_SEEDS),
        "num_references": NUM_REFERENCES,
        "real_hardware_execution": False,
        "artifacts": artifacts,
        "snapshot": snapshot,
        "exact_scores": exact,
        "noisy_scores": noisy,
        "analysis": analysis,
        "ready_for_exact_scoring": ready_for_exact,
        "ready_for_noisy_scoring": ready_to_score,
        "ready_to_analyze": noisy["ready"],
        "complete": bool(noisy["ready"] and analysis["ready"]),
        "blockers": blockers,
    }


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def write_cost_receipt(args: argparse.Namespace, status: dict[str, Any]) -> None:
    candidates = int(status["candidate_count"])
    modes = len(MODES)
    seeds = len(SIMULATOR_SEEDS)
    target_shots_per_mode = len(EXPECTED_TARGETS) * candidates * SHOTS * seeds
    reference_shots_per_mode = NUM_REFERENCES * candidates * SHOTS * seeds
    calibration_shots_per_mode = 2 * candidates * SHOTS * seeds
    target_shots = target_shots_per_mode * modes
    reference_shots = reference_shots_per_mode * modes
    calibration_shots = calibration_shots_per_mode * modes
    timings_path = args.out_dir / "STAGE_TIMINGS.json"
    timing_records = [] if not timings_path.is_file() else read_json(timings_path).get("records", [])
    cache_metadata = sorted(args.noisy_dir.glob("reference_cache/*.json"))
    cache_reuse = []
    for path in sorted(args.noisy_dir.glob("metadata/*.json")):
        metadata = read_json(path)
        cache_reuse.append({
            "target_id": metadata.get("target_id"),
            "reference_execution_reused": metadata.get("reference_execution_reused"),
            "reference_cache_sha256": metadata.get("reference_cache_sha256"),
        })
    checkpoint_paths = sorted(args.reference_dir.glob("reference_models/**/*.pt"))
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": status["protocol_sha256"],
        "scope": "Phase-7 Stage-1 pilot replication only",
        "trained_reference_models": NUM_REFERENCES,
        "noisy_reference_model_candidate_executions": NUM_REFERENCES * candidates * seeds,
        "noisy_reference_shots": reference_shots_per_mode,
        "ideal_reference_model_candidate_executions": NUM_REFERENCES * candidates * seeds,
        "ideal_reference_shots": reference_shots_per_mode,
        "all_reference_shots": reference_shots,
        "auxiliary_calibration_models_per_attack": 2,
        "noisy_auxiliary_calibration_model_candidate_executions_per_attack": 2 * candidates * seeds,
        "noisy_auxiliary_calibration_shots_per_attack": calibration_shots_per_mode,
        "ideal_auxiliary_calibration_shots_per_attack": calibration_shots_per_mode,
        "all_auxiliary_calibration_shots_per_attack": calibration_shots,
        "noisy_target_query_shots": target_shots_per_mode,
        "ideal_target_query_shots": target_shots_per_mode,
        "all_target_query_shots": target_shots,
        "all_stage1_simulated_shots": target_shots + reference_shots,
        "channel_to_matched_reference_shot_ratio_per_attack": calibration_shots_per_mode / reference_shots_per_mode,
        "wall_clock_stage_records": timing_records,
        "wall_clock_seconds_recorded": sum(float(row["seconds"]) for row in timing_records),
        "peak_gpu_memory_bytes": None,
        "peak_host_memory_bytes": None,
        "memory_collection_status": "not instrumented in Stage-1 launcher; required for confirmatory runner",
        "checkpoint_count": len(checkpoint_paths),
        "checkpoint_bytes": sum(path.stat().st_size for path in checkpoint_paths),
        "checkpoint_sha256": {str(path.resolve()): sha256(path) for path in checkpoint_paths},
        "reference_directory_bytes": directory_bytes(args.reference_dir),
        "noisy_output_bytes": directory_bytes(args.noisy_dir),
        "reference_cache_bytes": sum(path.stat().st_size for path in args.noisy_dir.glob("reference_cache/*") if path.is_file()),
        "reference_cache_metadata": [str(path.resolve()) for path in cache_metadata],
        "reference_cache_reuse": cache_reuse,
        "cache_reuse_scope": "one pilot structural cell and complete ideal/noisy 128-shot seed-0,1 protocol",
        "real_hardware_execution": False,
    }
    path = args.out_dir / "COST_RECEIPT.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_status(args: argparse.Namespace, status: dict[str, Any]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "STATUS.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts = status["artifacts"]
    exact = status["exact_scores"]
    noisy = status["noisy_scores"]
    analysis = status["analysis"]
    state = (
        "COMPLETE" if status["complete"] else
        "READY TO ANALYZE" if status["ready_to_analyze"] else
        "READY FOR NOISY SCORING" if status["ready_for_noisy_scoring"] else
        "READY FOR EXACT SCORING" if status["ready_for_exact_scoring"] else
        "TRAINING"
    )
    lines = [
        "# Phase 7 Stage 1 execution",
        "",
        f"## Status: {state}",
        "",
        "This is the locked pilot-cell engineering replication. It is excluded from the confirmatory primary endpoint.",
        "",
        "| Gate | Available | Required |",
        "|---|---:|---:|",
        f"| Target checkpoint bundles | {artifacts['target_bundles_ready']} | 3 |",
        f"| Reference score files | {artifacts['reference_scores_ready']} | 16 |",
        f"| Reference checkpoints | {artifacts['reference_checkpoints_ready']} | 16 |",
        f"| Hash-bound checkpoint metadata | {artifacts['reference_checkpoint_metadata_valid']} | 16 |",
        f"| Complete balanced reference bank | {artifacts['reference_banks_ready']} | 1 |",
        f"| Exact target score payloads | {exact['targets_ready']} | 3 |",
        f"| Frozen snapshot | {int(status['snapshot']['ready'])} | 1 |",
        f"| Noisy target bundles | {noisy['targets_ready']} | 3 |",
        f"| Engineering analysis | {int(analysis['ready'])} | 1 |",
        "",
        "## Frozen scope",
        "",
        f"- Protocol SHA-256: `{status['protocol_sha256']}`.",
        "- Pilot cell only: `eff_su2_r1_d2`; new model seeds 143–145.",
        "- 1,000 members and 1,000 nonmembers per target.",
        "- Sixteen references, 8 IN / 8 OUT per candidate.",
        "- Ideal/noisy 128-shot Aer execution, simulator seeds 0 and 1.",
        "- Existing privileged victim-crossfit learned comparator retained unchanged.",
        "- No quantum-hardware job.",
    ]
    if status["blockers"]:
        lines.extend(["", "## Current blockers", ""])
        lines.extend(f"- {value}" for value in status["blockers"])
    lines.extend([
        "", "## Artifacts", "",
        f"- Machine status: `{(args.out_dir / 'STATUS.json').resolve()}`",
        f"- Cost receipt: `{(args.out_dir / 'COST_RECEIPT.json').resolve()}`",
        f"- Scientific/engineering report: `{(args.out_dir / 'analysis/REPORT.md').resolve()}`",
        f"- Descriptive plot index: `{(args.out_dir / 'analysis/PLOTS.md').resolve()}`",
    ])
    (args.out_dir / "EXECUTION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def refresh(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    status = build_status(args, protocol)
    write_cost_receipt(args, status)
    write_status(args, status)
    return status


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
    ], label="target_training", out_dir=args.out_dir)


def train_references(args: argparse.Namespace) -> None:
    run([
        args.python,
        str(ROOT / "reviewer_tools/run_lira_reference_multigpu.py"),
        "--targets", str(args.targets),
        "--repo-root", str(ROOT),
        "--run-root", str(args.run_root),
        "--out-dir", str(args.reference_dir),
        "--num-references", str(NUM_REFERENCES),
        "--save-reference-checkpoints",
        "--phase", "train",
        "--seed", "20260822",
        "--gpus", args.reference_gpus,
        "--jobs-per-gpu", str(args.reference_jobs_per_gpu),
        "--cpu-threads", str(args.cpu_threads),
        "--resume",
    ], label="reference_training", out_dir=args.out_dir)


def exact_score(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    status = build_status(args, protocol)
    if not status["ready_for_exact_scoring"]:
        raise RuntimeError("Stage 1 target/reference artifacts are incomplete")
    run([
        args.python,
        str(ROOT / "reviewer_tools/run_lira_reference_multigpu.py"),
        "--targets", str(args.targets),
        "--repo-root", str(ROOT),
        "--run-root", str(args.run_root),
        "--out-dir", str(args.reference_dir),
        "--num-references", str(NUM_REFERENCES),
        "--phase", "score",
        "--bootstrap", str(args.bootstrap),
        "--seed", "20260822",
        "--gpus", args.target_gpus,
        "--jobs-per-gpu", "1",
        "--cpu-threads", str(args.cpu_threads),
        "--resume",
    ], label="exact_target_scoring", out_dir=args.out_dir)


def noisy_score(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    status = build_status(args, protocol)
    if not status["ready_for_noisy_scoring"]:
        raise RuntimeError("Stage 1 is not ready for noisy scoring")
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
            "--num-references", str(NUM_REFERENCES),
            "--modes", ",".join(MODES),
            "--shots", str(SHOTS),
            "--simulator-seeds", ",".join(map(str, SIMULATOR_SEEDS)),
            "--transpiler-seed", "20260822",
            "--optimization-level", "1",
            "--qiskit-batch-size", str(args.qiskit_batch_size),
            "--aer-max-parallel-threads", str(args.aer_max_parallel_threads),
            "--bootstrap", str(args.bootstrap),
            "--seed", "20260822",
            "--device", args.score_device,
            "--resume",
        ], label=f"noisy_scoring_{target_id}", out_dir=args.out_dir)
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
    ], label="noisy_aggregate", out_dir=args.out_dir)


def analyze(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    status = build_status(args, protocol)
    if not status["ready_to_analyze"]:
        raise RuntimeError("Stage-1 noisy target/reference bundles are incomplete")
    run([
        args.python,
        str(ROOT / "experiments/channel_lira_phase7_stage1.py"),
        "--protocol", str(args.protocol),
        "--protocol-lock", str(args.protocol_lock),
        "--candidate-probe", str(args.candidate_probe),
        "--targets", str(args.targets),
        "--reference-dir", str(args.reference_dir),
        "--noisy-dir", str(args.noisy_dir),
        "--out-dir", str(args.out_dir / "analysis"),
        "--seed", "20260822",
    ], label="stage1_analysis", out_dir=args.out_dir)


def plot(args: argparse.Namespace) -> None:
    if not inspect_analysis(args.out_dir)["ready"]:
        raise RuntimeError("Stage-1 analysis is incomplete; run the analyze stage first")
    run([
        args.python,
        str(ROOT / "experiments/plot_channel_lira_noisy_reference_scaleup.py"),
        "--analysis-dir", str(args.out_dir / "analysis"),
        "--title-prefix", "Phase 7 Stage 1",
        "--index-title", "Phase 7 Stage 1 plot index",
        "--png",
    ], label="stage1_plots", out_dir=args.out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("status", "target", "references", "exact", "score", "analyze", "plot", "all"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-lock", type=Path, default=DEFAULT_PROTOCOL_LOCK)
    parser.add_argument("--candidate-probe", type=Path, default=DEFAULT_CANDIDATE_PROBE)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--noisy-dir", type=Path, default=None)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--acknowledge-protocol-hash", default="")
    parser.add_argument("--target-gpus", default="0,1")
    parser.add_argument("--reference-gpus", default="0,1")
    parser.add_argument(
        "--reference-jobs-per-gpu",
        type=int,
        default=1,
        help="Execution-only reference-training concurrency per selected GPU.",
    )
    parser.add_argument("--score-device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--qiskit-batch-size", type=int, default=16)
    parser.add_argument("--aer-max-parallel-threads", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    for name in (
        "protocol", "protocol_lock", "candidate_probe", "targets", "out_dir",
        "run_root", "reference_dir", "snapshot",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.noisy_dir = (args.noisy_dir or args.out_dir / "noisy_lira").resolve()
    if (
        args.cpu_threads < 1
        or args.reference_jobs_per_gpu < 1
        or args.qiskit_batch_size < 1
        or args.aer_max_parallel_threads < 1
    ):
        parser.error("thread and batch settings must be positive")
    return args


def main() -> None:
    args = parse_args()
    protocol = validate_locked_stage(args, require_ack=args.stage != "status")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"target", "all"}:
        train_targets(args)
    if args.stage in {"references", "all"}:
        train_references(args)
    status = refresh(args, protocol)
    if args.stage == "exact" or (
        args.stage == "all" and status["ready_for_exact_scoring"]
        and not status["exact_scores"]["ready"]
    ):
        exact_score(args, protocol)
    status = refresh(args, protocol)
    if args.stage == "score" or (
        args.stage == "all" and status["ready_for_noisy_scoring"]
        and not status["noisy_scores"]["ready"]
    ):
        noisy_score(args, protocol)
    status = refresh(args, protocol)
    if args.stage == "analyze" or (
        args.stage == "all" and status["ready_to_analyze"]
        and not status["analysis"]["ready"]
    ):
        analyze(args, protocol)
    if args.stage == "plot" or (
        args.stage == "all" and inspect_analysis(args.out_dir)["ready"]
    ):
        plot(args)
    status = refresh(args, protocol)
    print(
        f"[STATUS] complete={status['complete']} "
        f"targets={status['artifacts']['target_bundles_ready']}/3 "
        f"references={status['artifacts']['reference_checkpoints_ready']}/16 "
        f"noisy={status['noisy_scores']['targets_ready']}/3",
        flush=True,
    )
    print(f"[REPORT] {args.out_dir / 'EXECUTION_REPORT.md'}")
    if args.require_complete and not status["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
