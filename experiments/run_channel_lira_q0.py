#!/usr/bin/env python3
"""Resumable launcher for the locked ChannelLiRA Q0 falsification screen."""
from __future__ import annotations

import argparse
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

from experiments.channel_lira_q0_common import (  # noqa: E402
    DEFAULT_OUT,
    DEFAULT_PROTOCOL,
    DEFAULT_PROTOCOL_LOCK,
    DEFAULT_RUN_ROOT,
    DEFAULT_SNAPSHOT,
    DEFAULT_TARGETS,
    read_targets,
    sha256,
    validate_protocol,
)


DEFAULT_CANDIDATE_PROBE = (
    ROOT / "channel_lira_results/q0_readiness/CANDIDATE_PARTITION_PROBE.json"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str], *, label: str, out_dir: Path) -> None:
    print("[RUN] " + " ".join(command), flush=True)
    started = time.monotonic()
    subprocess.run(command, cwd=ROOT, check=True)
    timing_path = out_dir / "STAGE_TIMINGS.json"
    records = [] if not timing_path.is_file() else read_json(timing_path).get("records", [])
    records.append(
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "seconds": time.monotonic() - started,
            "command": command,
        }
    )
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = timing_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"records": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(timing_path)


def target_artifact(row: dict[str, str], run_root: Path) -> dict[str, Any]:
    directory = run_root / row["experiment"] / row["target_id"]
    model = directory / "target_model.pt"
    attack = directory / "target_attack_data.pt"
    summary = directory / "target_export_summary.json"
    files = (model, attack, summary)
    ready = all(path.is_file() and path.stat().st_size > 0 for path in files)
    return {
        "target_id": row["target_id"],
        "ready": ready,
        "directory": str(directory.resolve()),
        "files": {
            path.name: ({"sha256": sha256(path), "bytes": path.stat().st_size} if path.is_file() else None)
            for path in files
        },
    }


def acquisition_artifact(target_id: str, out_dir: Path, protocol_hash: str) -> dict[str, Any]:
    payload = out_dir / "raw" / f"{target_id}.npz"
    metadata = out_dir / "metadata" / f"{target_id}.json"
    errors = []
    if not payload.is_file() or not metadata.is_file():
        errors.append("missing payload or metadata")
    else:
        try:
            record = read_json(metadata)
            if record.get("protocol_sha256") != protocol_hash:
                errors.append("protocol hash mismatch")
            if record.get("payload_sha256") != sha256(payload):
                errors.append("payload hash mismatch")
            with np.load(payload, allow_pickle=False) as saved:
                if saved["counts_layout_a"].shape != (10, 2000, 64):
                    errors.append("layout-A count shape mismatch")
                if saved["counts_layout_b"].shape != (5, 2000, 64):
                    errors.append("layout-B count shape mismatch")
                if saved["exact_z"].shape != (2000, 6):
                    errors.append("exact-Z shape mismatch")
                if len(set(saved["content_ids"].astype(str).tolist())) != 2000:
                    errors.append("content identities are not unique")
        except (KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "target_id": target_id,
        "ready": not errors,
        "payload": str(payload.resolve()),
        "metadata": str(metadata.resolve()),
        "errors": errors,
    }


def analysis_artifact(out_dir: Path) -> dict[str, Any]:
    directory = out_dir / "analysis"
    required = (
        "REPORT.md",
        "SOURCE_MANIFEST.json",
        "FEATURES.json",
        "SCREENING_DECISION.json",
        "metrics_target.csv",
        "metrics_summary.csv",
        "contrasts_target.csv",
        "contrasts_summary.csv",
        "plots/attack_auc.png",
        "plots/attack_tpr_at_1pct.png",
        "plots/loss_conditioned_auc.png",
        "plots/paired_joint_target_differences.png",
    )
    present = {
        name: (directory / name).is_file() and (directory / name).stat().st_size > 0
        for name in required
    }
    return {
        "ready": all(present.values()),
        "directory": str(directory.resolve()),
        "present": present,
    }


def candidate_probe_artifact(
    path: Path, protocol_hash: str, target_manifest_hash: str
) -> dict[str, Any]:
    errors = []
    payload: dict[str, Any] = {}
    if not path.is_file():
        errors.append("candidate partition probe is missing")
    else:
        try:
            payload = read_json(path)
            if payload.get("protocol_sha256") != protocol_hash:
                errors.append("candidate probe protocol hash mismatch")
            if payload.get("target_manifest_sha256") != target_manifest_hash:
                errors.append("candidate probe manifest hash mismatch")
            if payload.get("ready") is not True:
                errors.append("candidate probe did not pass")
            if payload.get("total_source_identities") != 12000:
                errors.append("candidate probe source partitions are not disjoint")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "ready": not errors,
        "path": str(path.resolve()),
        "sha256": sha256(path) if path.is_file() else None,
        "errors": errors,
        "content_overlap_occurrences": payload.get(
            "cross_target_content_overlap_occurrences"
        ),
    }


def build_status(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    rows = read_targets(args.targets)
    protocol_hash = sha256(args.protocol)
    target_manifest_hash = sha256(args.targets)
    candidate_probe = candidate_probe_artifact(
        args.candidate_probe, protocol_hash, target_manifest_hash
    )
    targets = [target_artifact(row, args.run_root) for row in rows]
    acquisitions = [
        acquisition_artifact(row["target_id"], args.out_dir, protocol_hash)
        for row in rows
    ]
    analysis = analysis_artifact(args.out_dir)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "target_manifest_sha256": target_manifest_hash,
        "snapshot_manifest_sha256": sha256(args.snapshot / "snapshot_manifest.json"),
        "study_role": protocol["study_role"],
        "phase7_confirmatory_included": False,
        "real_hardware_execution": False,
        "candidate_probe": candidate_probe,
        "targets_ready": sum(bool(value["ready"]) for value in targets),
        "targets_expected": len(targets),
        "acquisitions_ready": sum(bool(value["ready"]) for value in acquisitions),
        "acquisitions_expected": len(acquisitions),
        "analysis_ready": bool(analysis["ready"]),
        "complete": bool(
            candidate_probe["ready"]
            and all(value["ready"] for value in acquisitions)
            and analysis["ready"]
        ),
        "targets": targets,
        "acquisitions": acquisitions,
        "analysis": analysis,
    }


def write_status(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    status = build_status(args, protocol)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "STATUS.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    acquisition = protocol["acquisition"]
    candidates = int(protocol["study_population"]["members_per_target"]) + int(
        protocol["study_population"]["nonmembers_per_target"]
    )
    targets = int(protocol["study_population"]["target_count"])
    shots = int(acquisition["shots_per_query"])
    layout_a_queries = len(acquisition["simulator_seeds_layout_a"])
    layout_b_queries = len(acquisition["simulator_seeds_layout_b"])
    timing_path = args.out_dir / "STAGE_TIMINGS.json"
    timings = [] if not timing_path.is_file() else read_json(timing_path).get("records", [])
    cost = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": status["protocol_sha256"],
        "trained_target_models": targets,
        "trained_reference_models": 0,
        "candidate_records_per_target": candidates,
        "shots_per_query": shots,
        "layout_a_queries_per_candidate": layout_a_queries,
        "layout_b_queries_per_candidate": layout_b_queries,
        "fixed_attack_query_budget_per_candidate": 10,
        "paired_attack_query_budget_per_candidate": 10,
        "fixed_attack_shots_per_candidate": 10 * shots,
        "paired_attack_shots_per_candidate": 10 * shots,
        "total_acquisition_simulated_shots": targets
        * candidates
        * shots
        * (layout_a_queries + layout_b_queries),
        "reference_serving_shots": 0,
        "wall_clock_stage_records": timings,
        "wall_clock_seconds_recorded": sum(float(value["seconds"]) for value in timings),
        "real_hardware_execution": False,
    }
    (args.out_dir / "COST_RECEIPT.json").write_text(
        json.dumps(cost, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    state = (
        "COMPLETE"
        if status["complete"]
        else "READY TO ANALYZE"
        if status["acquisitions_ready"] == status["acquisitions_expected"]
        else "ACQUIRING"
        if status["targets_ready"] == status["targets_expected"]
        else "TRAINING"
    )
    lines = [
        "# Q0 residual quantum leakage execution",
        "",
        f"## Status: {state}",
        "",
        "This is a bounded exploratory falsification screen. It does not alter or contribute to the frozen Phase-7 confirmatory endpoints.",
        "",
        "| Gate | Available | Required |",
        "|---|---:|---:|",
        f"| Target checkpoint bundles | {status['targets_ready']} | {status['targets_expected']} |",
        f"| Source-disjoint candidate partition probe | {int(status['candidate_probe']['ready'])} | 1 |",
        f"| Raw response acquisitions | {status['acquisitions_ready']} | {status['acquisitions_expected']} |",
        f"| Analysis and screening decision | {int(status['analysis_ready'])} | 1 |",
        "",
        f"- Protocol SHA-256: `{status['protocol_sha256']}`.",
        "- Six independent data/model seeds across two structural cells.",
        "- Equal 1,280-shot fixed-layout and paired-layout attack budgets.",
        "- Raw bitstring counts retained; zero reference models.",
        "- No quantum-hardware execution.",
        "",
        f"- Machine status: `{(args.out_dir / 'STATUS.json').resolve()}`",
        f"- Cost receipt: `{(args.out_dir / 'COST_RECEIPT.json').resolve()}`",
        f"- Results report: `{(args.out_dir / 'analysis/REPORT.md').resolve()}`",
    ]
    (args.out_dir / "EXECUTION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return status


def train_targets(args: argparse.Namespace) -> None:
    run(
        [
            args.python,
            str(ROOT / "experiments/probe_channel_lira_q0_candidates.py"),
            "--protocol",
            str(args.protocol),
            "--protocol-lock",
            str(args.protocol_lock),
            "--targets",
            str(args.targets),
            "--snapshot",
            str(args.snapshot),
            "--out",
            str(args.candidate_probe),
        ],
        label="q0_candidate_partition_probe",
        out_dir=args.out_dir,
    )
    probe = candidate_probe_artifact(
        args.candidate_probe, sha256(args.protocol), sha256(args.targets)
    )
    if not probe["ready"]:
        raise RuntimeError("Q0 target training blocked by candidate partition probe")
    run(
        [
            args.python,
            str(ROOT / "reviewer_tools/run_target_table_dgx.py"),
            "--targets",
            str(args.targets),
            "--repo-root",
            str(ROOT),
            "--out",
            str(args.run_root),
            "--gpus",
            args.gpus,
            "--jobs-per-gpu",
            str(args.target_jobs_per_gpu),
            "--cpu-threads",
            str(args.cpu_threads),
            "--no-largest-first",
            "--resume",
        ],
        label="q0_target_training",
        out_dir=args.out_dir,
    )


def acquire_targets(args: argparse.Namespace, protocol_hash: str) -> None:
    status = build_status(args, validate_protocol(args.protocol, args.protocol_lock, args.targets, args.snapshot))
    if not status["candidate_probe"]["ready"]:
        raise RuntimeError("Q0 acquisition is blocked by the candidate partition probe")
    if status["targets_ready"] != status["targets_expected"]:
        raise RuntimeError("Q0 target checkpoints are incomplete")
    for row in read_targets(args.targets):
        run(
            [
                args.python,
                str(ROOT / "experiments/channel_lira_q0_acquire.py"),
                "--target-id",
                row["target_id"],
                "--protocol",
                str(args.protocol),
                "--protocol-lock",
                str(args.protocol_lock),
                "--targets",
                str(args.targets),
                "--snapshot",
                str(args.snapshot),
                "--run-root",
                str(args.run_root),
                "--out-dir",
                str(args.out_dir),
                "--acknowledge-protocol-hash",
                protocol_hash,
                "--device",
                args.device,
                "--qiskit-batch-size",
                str(args.qiskit_batch_size),
                "--aer-max-parallel-threads",
                str(args.aer_max_parallel_threads),
                "--resume",
            ],
            label=f"q0_acquire_{row['target_id']}",
            out_dir=args.out_dir,
        )


def analyze(args: argparse.Namespace, protocol_hash: str) -> None:
    protocol = validate_protocol(args.protocol, args.protocol_lock, args.targets, args.snapshot)
    status = build_status(args, protocol)
    if status["acquisitions_ready"] != status["acquisitions_expected"]:
        raise RuntimeError("Q0 analysis is blocked until all six source bundles are complete and hash-valid")
    run(
        [
            args.python,
            str(ROOT / "experiments/channel_lira_q0_analyze.py"),
            "--protocol",
            str(args.protocol),
            "--protocol-lock",
            str(args.protocol_lock),
            "--targets",
            str(args.targets),
            "--snapshot",
            str(args.snapshot),
            "--out-dir",
            str(args.out_dir),
            "--acknowledge-protocol-hash",
            protocol_hash,
        ],
        label="q0_analysis",
        out_dir=args.out_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("status", "probe", "target", "acquire", "analyze", "all")
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-lock", type=Path, default=DEFAULT_PROTOCOL_LOCK)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--candidate-probe", type=Path, default=DEFAULT_CANDIDATE_PROBE)
    parser.add_argument("--acknowledge-protocol-hash", default="")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--target-jobs-per-gpu", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--qiskit-batch-size", type=int, default=16)
    parser.add_argument("--aer-max-parallel-threads", type=int, default=128)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    for name in (
        "protocol",
        "protocol_lock",
        "targets",
        "snapshot",
        "run_root",
        "out_dir",
        "candidate_probe",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if min(
        args.cpu_threads,
        args.target_jobs_per_gpu,
        args.qiskit_batch_size,
        args.aer_max_parallel_threads,
    ) < 1:
        parser.error("Q0 thread and batch settings must be positive")
    return args


def main() -> None:
    args = parse_args()
    protocol = validate_protocol(args.protocol, args.protocol_lock, args.targets, args.snapshot)
    protocol_hash = sha256(args.protocol)
    if args.stage not in {"status", "probe"} and args.acknowledge_protocol_hash != protocol_hash:
        raise ValueError(
            "Q0 compute requires --acknowledge-protocol-hash equal to the locked protocol SHA-256"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "probe":
        run(
            [
                args.python,
                str(ROOT / "experiments/probe_channel_lira_q0_candidates.py"),
                "--protocol",
                str(args.protocol),
                "--protocol-lock",
                str(args.protocol_lock),
                "--targets",
                str(args.targets),
                "--snapshot",
                str(args.snapshot),
                "--out",
                str(args.candidate_probe),
            ],
            label="q0_candidate_partition_probe",
            out_dir=args.out_dir,
        )
    if args.stage in {"target", "all"}:
        train_targets(args)
    status = write_status(args, protocol)
    if args.stage in {"acquire", "all"}:
        if status["targets_ready"] == status["targets_expected"]:
            acquire_targets(args, protocol_hash)
        elif args.stage == "acquire":
            raise RuntimeError("Q0 acquisition requested before target training completed")
    status = write_status(args, protocol)
    if args.stage in {"analyze", "all"}:
        if status["acquisitions_ready"] == status["acquisitions_expected"]:
            analyze(args, protocol_hash)
        elif args.stage == "analyze":
            raise RuntimeError("Q0 analysis requested before acquisition completed")
    status = write_status(args, protocol)
    print(
        f"[STATUS] complete={status['complete']} "
        f"targets={status['targets_ready']}/{status['targets_expected']} "
        f"acquisitions={status['acquisitions_ready']}/{status['acquisitions_expected']} "
        f"analysis={int(status['analysis_ready'])}/1",
        flush=True,
    )
    print(f"[REPORT] {args.out_dir / 'EXECUTION_REPORT.md'}", flush=True)
    if args.require_complete and not status["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
