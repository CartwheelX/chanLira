#!/usr/bin/env python3
"""Run the frozen Phase-7 Stage-2 confirmatory primary experiment.

The runner is resumable and can only reach the four confirmatory cells at noisy
128-shot serving with simulator seeds 0--9.  Every compute stage requires the
external protocol hash; noisy scoring additionally requires acknowledgement of
the exact 194,560,000-shot primary budget.  Aggregate ChannelLiRA comparisons are
blocked until all raw artifacts have been sealed by SHA-256.
"""
from __future__ import annotations

import argparse
from collections import deque
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

from experiments.channel_lira_phase7_stage2 import (  # noqa: E402
    EXPECTED_CELLS,
    EXPECTED_TARGETS,
    NUM_REFERENCES,
    PRIMARY_MODE,
    PRIMARY_SHOTS,
    SIMULATOR_SEEDS,
    validate_raw_output_seal,
    validate_stage2_manifest,
    validate_stage2_protocol,
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
from experiments.run_channel_lira_phase7_stage1 import (  # noqa: E402
    inspect_exact_scores,
)


DEFAULT_TARGETS = ROOT / "reviewer_targets/channel_lira_phase7_stage2_confirmatory.csv"
DEFAULT_OUT = ROOT / "channel_lira_results/phase7/stage2_primary"
DEFAULT_RUN_ROOT = ROOT / "reviewer_runs"
DEFAULT_REFERENCE_DIR = ROOT / "channel_lira_results/phase7/references"
DEFAULT_SNAPSHOT = ROOT / "channel_lira_results/noisy_reference_canary_phase5/backend_snapshot"
PRIMARY_REFERENCE_SHOTS = 163_840_000
PRIMARY_TARGET_SHOTS = 30_720_000
PRIMARY_TOTAL_SHOTS = PRIMARY_REFERENCE_SHOTS + PRIMARY_TARGET_SHOTS


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


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
    try:
        validate_stage2_protocol(protocol)
        validate_stage2_manifest(protocol, args.targets)
    except ValueError as exc:
        errors.append(str(exc))
    frozen_snapshot = resolve_repo_path(protocol["provenance"]["backend_snapshot"]).resolve()
    if args.snapshot != frozen_snapshot:
        errors.append("Stage-2 snapshot path differs from the frozen protocol")
    expected_hash = sha256(args.protocol)
    if require_ack and args.acknowledge_protocol_hash != expected_hash:
        errors.append(
            "compute requires --acknowledge-protocol-hash equal to the locked protocol SHA-256"
        )
    if args.stage in {"score", "all"} and args.acknowledge_shot_budget != PRIMARY_TOTAL_SHOTS:
        errors.append(
            f"noisy scoring requires --acknowledge-shot-budget {PRIMARY_TOTAL_SHOTS}"
        )
    if errors:
        raise ValueError("Phase-7 Stage-2 guard failed: " + "; ".join(errors))
    return protocol


def _process_tree_pids(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    try:
        proc_entries = list(Path("/proc").iterdir())
    except OSError:
        return {root_pid}
    for entry in proc_entries:
        if not entry.name.isdigit():
            continue
        status_path = entry / "status"
        try:
            pid = int(entry.name)
            parent = -1
            for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("PPid:"):
                    parent = int(line.split()[1])
                    break
            parents[pid] = parent
        except (OSError, ValueError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def _rss_bytes(pids: set[int]) -> int:
    total_kib = 0
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                if line.startswith("VmRSS:"):
                    total_kib += int(line.split()[1])
                    break
        except (OSError, ValueError):
            continue
    return total_kib * 1024


def _gpu_memory_bytes(pids: set[int]) -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    total_mib = 0
    for line in output.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            if int(fields[0]) in pids:
                total_mib += int(fields[1])
        except ValueError:
            continue
    return total_mib * 1024 * 1024


def record_timing(
    out_dir: Path,
    label: str,
    seconds: float,
    command: list[str],
    peak_host_memory_bytes: int,
    peak_gpu_memory_bytes: int | None,
    *,
    parallel_group: str | None = None,
) -> None:
    path = out_dir / "STAGE_TIMINGS.json"
    records = [] if not path.is_file() else read_json(path).get("records", [])
    record = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "seconds": seconds,
        "command": command,
        "peak_host_memory_bytes": peak_host_memory_bytes,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
    }
    if parallel_group is not None:
        record["parallel_group"] = parallel_group
    records.append(record)
    atomic_json(path, {"records": records})


def run(command: list[str], *, label: str, out_dir: Path) -> None:
    print("[RUN] " + " ".join(command), flush=True)
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT)
    peak_host = 0
    peak_gpu: int | None = None
    polls = 0
    try:
        while process.poll() is None:
            pids = _process_tree_pids(process.pid)
            peak_host = max(peak_host, _rss_bytes(pids))
            if polls % 5 == 0:
                current_gpu = _gpu_memory_bytes(pids)
                if current_gpu is not None:
                    peak_gpu = max(peak_gpu or 0, current_gpu)
            polls += 1
            time.sleep(1.0)
    except BaseException:
        process.terminate()
        process.wait()
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    record_timing(
        out_dir,
        label,
        time.monotonic() - started,
        command,
        peak_host,
        peak_gpu,
    )


def _terminate_processes(processes: list[subprocess.Popen[Any]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_cell_queues(
    commands_by_cell: dict[str, list[tuple[str, list[str]]]],
    *,
    max_workers: int,
    out_dir: Path,
    poll_interval: float = 1.0,
) -> None:
    """Run one sequential command queue per cell, with cells in parallel.

    Targets in one structural cell deliberately never overlap: they share one
    noisy-reference cache and concurrent cache construction would duplicate work
    and race the cache's atomic rename. Different cells have distinct cache keys
    and may execute safely at the same time.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    queues = {
        cell: deque(commands)
        for cell, commands in commands_by_cell.items()
        if commands
    }
    pending_cells = deque(queues)
    active: dict[str, dict[str, Any]] = {}
    parallel_group = "primary_noisy_scoring"
    parallel_started = time.monotonic()
    parallel_peak_host = 0
    parallel_peak_gpu: int | None = None
    parallel_polls = 0

    def launch(cell: str) -> None:
        label, command = queues[cell].popleft()
        print(f"[RUN cell={cell}] " + " ".join(command), flush=True)
        active[cell] = {
            "label": label,
            "command": command,
            "process": subprocess.Popen(command, cwd=ROOT),
            "started": time.monotonic(),
            "peak_host": 0,
            "peak_gpu": None,
            "polls": 0,
        }

    try:
        while pending_cells or active:
            while pending_cells and len(active) < min(max_workers, len(queues)):
                launch(pending_cells.popleft())

            completed_cells = []
            all_pids: set[int] = set()
            for cell, state in list(active.items()):
                process = state["process"]
                if process.poll() is None:
                    pids = _process_tree_pids(process.pid)
                    all_pids.update(pids)
                    state["peak_host"] = max(state["peak_host"], _rss_bytes(pids))
                    if state["polls"] % 5 == 0:
                        current_gpu = _gpu_memory_bytes(pids)
                        if current_gpu is not None:
                            state["peak_gpu"] = max(state["peak_gpu"] or 0, current_gpu)
                    state["polls"] += 1
                else:
                    completed_cells.append(cell)

            if all_pids:
                parallel_peak_host = max(parallel_peak_host, _rss_bytes(all_pids))
                if parallel_polls % 5 == 0:
                    current_gpu = _gpu_memory_bytes(all_pids)
                    if current_gpu is not None:
                        parallel_peak_gpu = max(parallel_peak_gpu or 0, current_gpu)
                parallel_polls += 1

            for cell in completed_cells:
                state = active.pop(cell)
                process = state["process"]
                if process.returncode:
                    _terminate_processes(
                        [value["process"] for value in active.values()]
                    )
                    raise subprocess.CalledProcessError(
                        process.returncode, state["command"]
                    )
                record_timing(
                    out_dir,
                    state["label"],
                    time.monotonic() - state["started"],
                    state["command"],
                    state["peak_host"],
                    state["peak_gpu"],
                    parallel_group=parallel_group,
                )
                if queues[cell]:
                    launch(cell)

            if active:
                time.sleep(poll_interval)
    except BaseException:
        _terminate_processes([state["process"] for state in active.values()])
        raise

    record_timing(
        out_dir,
        "primary_noisy_scoring_parallel_wall_clock",
        time.monotonic() - parallel_started,
        [
            "cell-aware-parallel-scheduler",
            f"--score-workers={max_workers}",
            f"--cells={','.join(commands_by_cell)}",
        ],
        parallel_peak_host,
        parallel_peak_gpu,
    )


def inspect_noisy_scores(noisy_dir: Path, candidate_count: int) -> dict[str, Any]:
    records = []
    cache_paths: set[Path] = set()
    snapshot_hashes: set[str] = set()
    for target_id in EXPECTED_TARGETS:
        errors = []
        payload_count = 0
        for simulator_seed in SIMULATOR_SEEDS:
            path = noisy_dir / "sample_scores" / (
                f"{target_id}_{PRIMARY_MODE}_sim{simulator_seed}.npz"
            )
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
                    membership = np.asarray(saved["membership"], dtype=int)
                    if np.bincount(membership, minlength=2).tolist() != [
                        candidate_count // 2, candidate_count // 2,
                    ]:
                        raise ValueError("membership population is not balanced")
                payload_count += 1
            except (KeyError, OSError, TypeError, ValueError) as exc:
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
        score = noisy_dir / "target_scores" / f"{target_id}.csv"
        metadata_path = noisy_dir / "metadata" / f"{target_id}.json"
        if not score.is_file() or score.stat().st_size == 0:
            errors.append("missing target score table")
        if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
            errors.append("missing target metadata")
        else:
            try:
                metadata = read_json(metadata_path)
                expected = {
                    "modes": [PRIMARY_MODE],
                    "simulator_seeds": list(SIMULATOR_SEEDS),
                    "shots": PRIMARY_SHOTS,
                    "num_reference_models": NUM_REFERENCES,
                    "transpiler_seed": 20_260_822,
                    "optimization_level": 1,
                }
                mismatches = {
                    key: (metadata.get(key), value)
                    for key, value in expected.items() if metadata.get(key) != value
                }
                if mismatches:
                    raise ValueError(f"metadata mismatch {mismatches}")
                hashes = metadata.get("reference_checkpoint_sha256", [])
                if len(hashes) != NUM_REFERENCES:
                    raise ValueError("reference checkpoint hash ledger is incomplete")
                cache = Path(str(metadata["reference_cache"])).resolve()
                if not cache.is_file() or not cache.with_suffix(".json").is_file():
                    raise ValueError("reference cache or metadata is missing")
                if sha256(cache) != metadata.get("reference_cache_sha256"):
                    raise ValueError("reference cache hash mismatch")
                cache_paths.add(cache)
                snapshot_hashes.add(str(metadata.get("snapshot_manifest_sha256", "")))
            except (KeyError, OSError, TypeError, ValueError) as exc:
                errors.append(f"metadata: {type(exc).__name__}: {exc}")
        records.append({
            "target_id": target_id,
            "ready": not errors,
            "payloads_ready": payload_count,
            "payloads_expected": len(SIMULATOR_SEEDS),
            "errors": errors,
        })
    aggregate_paths = (
        noisy_dir / "noisy_lira_raw.csv",
        noisy_dir / "noisy_lira_summary.csv",
    )
    aggregate_ready = all(path.is_file() and path.stat().st_size > 0 for path in aggregate_paths)
    return {
        "ready": bool(
            aggregate_ready and all(record["ready"] for record in records)
            and len(cache_paths) == len(EXPECTED_CELLS) and len(snapshot_hashes) == 1
        ),
        "aggregate_ready": aggregate_ready,
        "targets_ready": sum(bool(record["ready"]) for record in records),
        "expected": len(EXPECTED_TARGETS),
        "payloads_ready": sum(int(record["payloads_ready"]) for record in records),
        "payloads_expected": len(EXPECTED_TARGETS) * len(SIMULATOR_SEEDS),
        "reference_caches_ready": len(cache_paths),
        "reference_caches_expected": len(EXPECTED_CELLS),
        "snapshot_hashes": sorted(snapshot_hashes),
        "targets": records,
    }


def analysis_status(out_dir: Path) -> dict[str, Any]:
    directory = out_dir / "analysis"
    names = (
        "CONFIG.json", "DECISION.json", "SOURCE_MANIFEST.json", "REPORT.md",
        "metrics_raw.csv", "target_mean_metrics.csv", "metrics_summary.csv",
        "paired_contrasts_target.csv", "paired_contrasts_summary.csv",
        "hierarchical_intervals.csv", "channel_diagnostics.csv",
    )
    present = {
        name: (directory / name).is_file() and (directory / name).stat().st_size > 0
        for name in names
    }
    return {"ready": all(present.values()), "directory": str(directory), "present": present}


def plot_status(out_dir: Path) -> dict[str, Any]:
    analysis = out_dir / "analysis"
    paths = (
        analysis / "PLOTS.md",
        analysis / "plots/primary_gate_intervals.svg",
        analysis / "plots/primary_gate_intervals.png",
        analysis / "plots/primary_attack_metrics.svg",
        analysis / "plots/primary_attack_metrics.png",
    )
    return {
        "ready": all(path.is_file() and path.stat().st_size > 0 for path in paths),
        "files": {str(path): path.is_file() and path.stat().st_size > 0 for path in paths},
    }


def seal_status(args: argparse.Namespace) -> dict[str, Any]:
    path = args.out_dir / "RAW_OUTPUT_SEAL.json"
    if not path.is_file():
        return {"ready": False, "path": str(path), "error": "not sealed"}
    try:
        seal = validate_raw_output_seal(
            path, protocol_path=args.protocol, target_manifest=args.targets
        )
        return {
            "ready": True,
            "path": str(path),
            "sha256": sha256(path),
            "artifact_count": seal["artifact_count"],
        }
    except (FileNotFoundError, ValueError) as exc:
        return {"ready": False, "path": str(path), "error": str(exc)}


def build_status(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    rows = validate_stage2_manifest(protocol, args.targets)
    artifacts = inspect_execution_artifacts(
        protocol, rows, args.run_root, args.reference_dir
    )
    candidate_count = int(
        protocol["study_population"]["candidate_protocol"]["candidates_per_target"]
    )
    exact = inspect_exact_scores(EXPECTED_TARGETS, args.reference_dir, candidate_count)
    noisy = inspect_noisy_scores(args.noisy_dir, candidate_count)
    seal = seal_status(args)
    analysis = analysis_status(args.out_dir)
    plots = plot_status(args.out_dir)
    target_ready = artifacts["target_bundles_ready"] == len(EXPECTED_TARGETS)
    refs_ready = artifacts["reference_banks_ready"] == len(EXPECTED_CELLS)
    ready_for_exact = target_ready and refs_ready
    ready_for_noisy = ready_for_exact and exact["ready"] and inspect_snapshot(protocol)["ready"]
    blockers = []
    if not target_ready:
        blockers.append(
            f"confirmatory target bundles incomplete "
            f"({artifacts['target_bundles_ready']}/{len(EXPECTED_TARGETS)})"
        )
    if not refs_ready:
        blockers.append(
            f"confirmatory reference banks incomplete "
            f"({artifacts['reference_banks_ready']}/{len(EXPECTED_CELLS)} banks; "
            f"{artifacts['reference_checkpoints_ready']}/{len(EXPECTED_CELLS) * NUM_REFERENCES} checkpoints)"
        )
    if noisy["ready"] and not seal["ready"]:
        blockers.append("raw primary outputs must be sealed before aggregate unblinding")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(args.protocol),
        "analysis_role": "confirmatory_primary",
        "pilot_included": False,
        "target_manifest": str(args.targets),
        "target_manifest_sha256": sha256(args.targets),
        "target_ids": list(EXPECTED_TARGETS),
        "cells": list(EXPECTED_CELLS),
        "candidate_count": candidate_count,
        "members": candidate_count // 2,
        "nonmembers": candidate_count // 2,
        "modes": [PRIMARY_MODE],
        "shots": PRIMARY_SHOTS,
        "simulator_seeds": list(SIMULATOR_SEEDS),
        "num_references_per_cell": NUM_REFERENCES,
        "real_hardware_execution": False,
        "projected_shots": {
            "reference": PRIMARY_REFERENCE_SHOTS,
            "target": PRIMARY_TARGET_SHOTS,
            "combined": PRIMARY_TOTAL_SHOTS,
        },
        "artifacts": artifacts,
        "exact_scores": exact,
        "noisy_scores": noisy,
        "raw_output_seal": seal,
        "analysis": analysis,
        "plots": plots,
        "ready_for_exact_scoring": ready_for_exact,
        "ready_for_noisy_scoring": ready_for_noisy,
        "ready_to_seal": noisy["ready"],
        "ready_to_analyze": bool(noisy["ready"] and seal["ready"]),
        "complete": bool(noisy["ready"] and seal["ready"] and analysis["ready"] and plots["ready"]),
        "blockers": blockers,
    }


def directory_bytes(path: Path) -> int:
    return (
        sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        if path.exists() else 0
    )


def confirmatory_checkpoint_paths(reference_dir: Path) -> list[Path]:
    return [
        reference_dir / "reference_models" / f"{cell}_wd0" / f"reference_{reference_id:03d}.pt"
        for cell in EXPECTED_CELLS for reference_id in range(NUM_REFERENCES)
    ]


def write_cost_receipt(args: argparse.Namespace, status: dict[str, Any]) -> None:
    timings_path = args.out_dir / "STAGE_TIMINGS.json"
    timings = [] if not timings_path.is_file() else read_json(timings_path).get("records", [])
    accounted_timings = [row for row in timings if not row.get("parallel_group")]
    parallel_components = [row for row in timings if row.get("parallel_group")]
    checkpoint_paths = [path for path in confirmatory_checkpoint_paths(args.reference_dir) if path.is_file()]
    cache_paths = sorted(args.noisy_dir.glob("reference_cache/*.npz"))
    host_values = [int(row["peak_host_memory_bytes"]) for row in timings if row.get("peak_host_memory_bytes") is not None]
    gpu_values = [int(row["peak_gpu_memory_bytes"]) for row in timings if row.get("peak_gpu_memory_bytes") is not None]
    candidates = int(status["candidate_count"])
    seeds = len(SIMULATOR_SEEDS)
    per_attack_calibration_shots = 2 * candidates * PRIMARY_SHOTS * seeds
    per_cell_unique_calibration_shots = 3 * candidates * PRIMARY_SHOTS * seeds
    per_cell_reference_shots = NUM_REFERENCES * candidates * PRIMARY_SHOTS * seeds
    completed_cache_count = min(len(cache_paths), len(EXPECTED_CELLS))
    completed_target_count = int(status["noisy_scores"]["targets_ready"])
    actual_reference_shots = completed_cache_count * per_cell_reference_shots
    actual_target_shots = completed_target_count * candidates * PRIMARY_SHOTS * seeds
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": status["protocol_sha256"],
        "scope": "Phase-7 Stage-2 confirmatory primary only",
        "planned_reference_models": len(EXPECTED_CELLS) * NUM_REFERENCES,
        "trained_reference_models": len(checkpoint_paths),
        "planned_noisy_reference_model_candidate_executions": len(EXPECTED_CELLS) * NUM_REFERENCES * candidates * seeds,
        "noisy_reference_model_candidate_executions": completed_cache_count * NUM_REFERENCES * candidates * seeds,
        "planned_noisy_reference_shots": PRIMARY_REFERENCE_SHOTS,
        "noisy_reference_shots": actual_reference_shots,
        "auxiliary_calibration_models_per_attack": 2,
        "noisy_auxiliary_calibration_model_candidate_executions_per_attack": 2 * candidates * seeds,
        "noisy_auxiliary_calibration_shots_per_attack": per_attack_calibration_shots,
        "channel_to_matched_reference_shot_ratio_per_attack": per_attack_calibration_shots / per_cell_reference_shots,
        "unique_auxiliary_calibration_models_per_cell_amortized": 3,
        "noisy_unique_auxiliary_calibration_shots_per_cell_amortized": per_cell_unique_calibration_shots,
        "channel_to_matched_reference_shot_ratio_per_cell_amortized": per_cell_unique_calibration_shots / per_cell_reference_shots,
        "planned_target_query_shots": PRIMARY_TARGET_SHOTS,
        "target_query_shots": actual_target_shots,
        "planned_all_stage2_primary_simulated_shots": PRIMARY_TOTAL_SHOTS,
        "all_stage2_primary_simulated_shots": actual_reference_shots + actual_target_shots,
        "wall_clock_stage_records": timings,
        "wall_clock_seconds_recorded": sum(
            float(row["seconds"]) for row in accounted_timings
        ),
        "parallel_component_seconds_recorded": sum(
            float(row["seconds"]) for row in parallel_components
        ),
        "wall_clock_accounting": (
            "parallel component durations are reported but excluded from the "
            "wall-clock sum; their scheduler wall-clock record is included"
        ),
        "peak_gpu_memory_bytes": max(gpu_values) if gpu_values else None,
        "peak_host_memory_bytes": max(host_values) if host_values else None,
        "memory_collection_status": "process-tree RSS and NVIDIA compute-process memory are sampled by the Stage-2 launcher",
        "checkpoint_count": len(checkpoint_paths),
        "checkpoint_bytes": sum(path.stat().st_size for path in checkpoint_paths),
        "checkpoint_sha256": {str(path.resolve()): sha256(path) for path in checkpoint_paths},
        "reference_directory_bytes": directory_bytes(args.reference_dir),
        "noisy_output_bytes": directory_bytes(args.noisy_dir),
        "cache_count": len(cache_paths),
        "cache_bytes": sum(path.stat().st_size for path in cache_paths),
        "cache_sha256": {str(path.resolve()): sha256(path) for path in cache_paths},
        "cache_reuse_scope": "one noisy 128-shot seed-0..9 reference-oracle cache per confirmatory structural cell",
        "real_hardware_execution": False,
    }
    atomic_json(args.out_dir / "COST_RECEIPT.json", receipt)


def write_status(args: argparse.Namespace, status: dict[str, Any]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out_dir / "STATUS.json", status)
    artifacts = status["artifacts"]
    exact = status["exact_scores"]
    noisy = status["noisy_scores"]
    state = (
        "COMPLETE" if status["complete"] else
        "UNBLINDED" if status["analysis"]["ready"] else
        "SEALED / READY TO UNBLIND" if status["ready_to_analyze"] else
        "READY TO SEAL" if status["ready_to_seal"] else
        "READY FOR NOISY SCORING" if status["ready_for_noisy_scoring"] else
        "READY FOR EXACT SCORING" if status["ready_for_exact_scoring"] else
        "NOT STARTED" if (
            artifacts["target_bundles_ready"] == 0
            and artifacts["reference_checkpoints_ready"] == 0
        ) else
        "TRAINING"
    )
    lines = [
        "# Phase 7 Stage 2 confirmatory primary execution",
        "",
        f"## Status: {state}",
        "",
        "This runner is restricted to the frozen four-cell noisy 128-shot primary. The pilot and all Stage-3 secondary conditions are unreachable.",
        "",
        "| Gate | Available | Required |",
        "|---|---:|---:|",
        f"| Target checkpoint bundles | {artifacts['target_bundles_ready']} | 12 |",
        f"| Reference score files | {artifacts['reference_scores_ready']} | 64 |",
        f"| Reference checkpoints | {artifacts['reference_checkpoints_ready']} | 64 |",
        f"| Balanced reference banks | {artifacts['reference_banks_ready']} | 4 |",
        f"| Exact target score payloads | {exact['targets_ready']} | 12 |",
        f"| Noisy target bundles | {noisy['targets_ready']} | 12 |",
        f"| Noisy serving payloads | {noisy['payloads_ready']} | 120 |",
        f"| Noisy reference caches | {noisy['reference_caches_ready']} | 4 |",
        f"| Raw-output hash seal | {int(status['raw_output_seal']['ready'])} | 1 |",
        f"| Confirmatory analysis | {int(status['analysis']['ready'])} | 1 |",
        f"| Primary plots | {int(status['plots']['ready'])} | 1 |",
        "",
        "## Budget guard",
        "",
        f"- Noisy reference serving: {PRIMARY_REFERENCE_SHOTS:,} shots.",
        f"- Target serving: {PRIMARY_TARGET_SHOTS:,} shots.",
        f"- Combined primary: {PRIMARY_TOTAL_SHOTS:,} shots.",
        "- Secondary and ideal-diagnostic conditions: not implemented by this runner.",
    ]
    if status["blockers"]:
        lines.extend(["", "## Current blockers", ""])
        lines.extend(f"- {value}" for value in status["blockers"])
    lines.extend([
        "", "## Artifacts", "",
        f"- Status: `{(args.out_dir / 'STATUS.json').resolve()}`",
        f"- Cost receipt: `{(args.out_dir / 'COST_RECEIPT.json').resolve()}`",
        f"- Raw-output seal: `{(args.out_dir / 'RAW_OUTPUT_SEAL.json').resolve()}`",
        f"- Decision report: `{(args.out_dir / 'analysis/REPORT.md').resolve()}`",
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
    ], label="confirmatory_target_training", out_dir=args.out_dir)


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
    ], label="confirmatory_reference_training", out_dir=args.out_dir)


def exact_score(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    if not build_status(args, protocol)["ready_for_exact_scoring"]:
        raise RuntimeError("Stage-2 target/reference artifacts are incomplete")
    run([
        args.python,
        str(ROOT / "reviewer_tools/run_lira_reference_multigpu.py"),
        "--targets", str(args.targets),
        "--repo-root", str(ROOT),
        "--run-root", str(args.run_root),
        "--out-dir", str(args.reference_dir),
        "--num-references", str(NUM_REFERENCES),
        "--phase", "score",
        "--bootstrap", str(args.record_bootstrap),
        "--seed", "20260822",
        "--gpus", args.target_gpus,
        "--jobs-per-gpu", "1",
        "--cpu-threads", str(args.cpu_threads),
        "--resume",
    ], label="confirmatory_exact_target_scoring", out_dir=args.out_dir)


def group_targets_by_cell(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    grouped = {cell: [] for cell in EXPECTED_CELLS}
    for row in rows:
        cell = row["structural_cell_id"]
        if cell not in grouped:
            raise ValueError(f"Unexpected Stage-2 structural cell: {cell}")
        grouped[cell].append(row["target_id"])
    observed = tuple(target_id for cell in EXPECTED_CELLS for target_id in grouped[cell])
    if observed != EXPECTED_TARGETS:
        raise ValueError("Cell-aware scoring queues differ from the frozen target order")
    if any(len(targets) != 3 for targets in grouped.values()):
        raise ValueError("Each Stage-2 structural cell must contain exactly three targets")
    return grouped


def noisy_target_command(args: argparse.Namespace, target_id: str) -> list[str]:
    return [
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
        "--modes", PRIMARY_MODE,
        "--shots", str(PRIMARY_SHOTS),
        "--simulator-seeds", ",".join(map(str, SIMULATOR_SEEDS)),
        "--transpiler-seed", "20260822",
        "--optimization-level", "1",
        "--qiskit-batch-size", str(args.qiskit_batch_size),
        "--aer-max-parallel-threads", str(args.aer_max_parallel_threads),
        "--bootstrap", str(args.record_bootstrap),
        "--seed", "20260822",
        "--device", args.score_device,
        "--resume",
    ]


def noisy_score(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    status = build_status(args, protocol)
    if status["raw_output_seal"]["ready"]:
        raise RuntimeError("Stage-2 is already sealed; noisy artifacts are immutable")
    if not status["ready_for_noisy_scoring"]:
        raise RuntimeError("Stage-2 is not ready for noisy scoring")
    rows = validate_stage2_manifest(protocol, args.targets)
    target_groups = group_targets_by_cell(rows)
    commands_by_cell = {
        cell: [
            (
                f"primary_noisy_scoring_{target_id}",
                noisy_target_command(args, target_id),
            )
            for target_id in target_ids
        ]
        for cell, target_ids in target_groups.items()
    }
    print(
        f"[PARALLEL] score_workers={args.score_workers} "
        f"aer_threads_per_worker={args.aer_max_parallel_threads} "
        f"maximum_aer_threads={args.score_workers * args.aer_max_parallel_threads}",
        flush=True,
    )
    run_cell_queues(
        commands_by_cell,
        max_workers=args.score_workers,
        out_dir=args.out_dir,
    )
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
    ], label="primary_noisy_aggregate", out_dir=args.out_dir)


def collect_raw_artifacts(args: argparse.Namespace, protocol: dict[str, Any]) -> list[Path]:
    paths = [
        args.protocol,
        args.protocol_lock,
        args.candidate_probe,
        args.targets,
        args.snapshot / "snapshot_manifest.json",
        args.reference_dir / "lira_reference_mia_raw.csv",
        args.reference_dir / "lira_reference_mia_summary.csv",
        args.reference_dir / "analysis_metadata.json",
        args.reference_dir / "reference_provenance.json",
        args.noisy_dir / "noisy_lira_raw.csv",
        args.noisy_dir / "noisy_lira_summary.csv",
    ]
    for target_id in EXPECTED_TARGETS:
        target_root = args.run_root / "channel_lira_phase7" / target_id
        paths.extend([
            target_root / "target_model.pt",
            target_root / "target_attack_data.pt",
            target_root / "target_export_summary.json",
            args.reference_dir / "sample_scores" / f"{target_id}.npz",
            args.reference_dir / "target_scores" / f"{target_id}.csv",
            args.noisy_dir / "target_scores" / f"{target_id}.csv",
            args.noisy_dir / "metadata" / f"{target_id}.json",
        ])
        paths.extend(
            args.noisy_dir / "sample_scores" / f"{target_id}_{PRIMARY_MODE}_sim{seed}.npz"
            for seed in SIMULATOR_SEEDS
        )
    for cell in EXPECTED_CELLS:
        root = args.reference_dir / "reference_models" / f"{cell}_wd0"
        for reference_id in range(NUM_REFERENCES):
            paths.extend([
                root / f"reference_{reference_id:03d}.npz",
                root / f"reference_{reference_id:03d}.pt",
            ])
    for metadata_path in sorted((args.noisy_dir / "metadata").glob("*.json")):
        metadata = read_json(metadata_path)
        cache = Path(str(metadata["reference_cache"])).resolve()
        paths.extend([cache, cache.with_suffix(".json")])
    unique = sorted(set(path.resolve() for path in paths), key=str)
    missing = [path for path in unique if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            "Cannot seal incomplete Stage-2 artifacts: " + ", ".join(map(str, missing))
        )
    return unique


def seal_outputs(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    status = build_status(args, protocol)
    if not status["ready_to_seal"]:
        raise RuntimeError("All 12 noisy target bundles and four caches must complete before sealing")
    seal_path = args.out_dir / "RAW_OUTPUT_SEAL.json"
    if seal_path.is_file():
        existing = seal_status(args)
        if existing["ready"]:
            print(f"[SKIP] raw-output seal is already valid: {seal_path}", flush=True)
            return
        raise RuntimeError(f"Existing raw-output seal is invalid: {existing.get('error')}")
    paths = collect_raw_artifacts(args, protocol)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "phase7_stage2_primary",
        "protocol_sha256": sha256(args.protocol),
        "target_manifest_sha256": sha256(args.targets),
        "mode": PRIMARY_MODE,
        "shots": PRIMARY_SHOTS,
        "simulator_seeds": list(SIMULATOR_SEEDS),
        "targets": list(EXPECTED_TARGETS),
        "cells": list(EXPECTED_CELLS),
        "artifact_count": len(paths),
        "files": {str(path): sha256(path) for path in paths},
        "unblinding_authorized": True,
        "real_hardware_execution": False,
    }
    atomic_json(seal_path, payload)
    print(f"[SEALED] {len(paths)} raw artifacts -> {seal_path}", flush=True)


def analyze(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    if not build_status(args, protocol)["ready_to_analyze"]:
        raise RuntimeError("Stage-2 raw outputs are incomplete or not hash-sealed")
    run([
        args.python,
        str(ROOT / "experiments/channel_lira_phase7_stage2.py"),
        "--protocol", str(args.protocol),
        "--protocol-lock", str(args.protocol_lock),
        "--candidate-probe", str(args.candidate_probe),
        "--targets", str(args.targets),
        "--reference-dir", str(args.reference_dir),
        "--noisy-dir", str(args.noisy_dir),
        "--raw-output-seal", str(args.out_dir / "RAW_OUTPUT_SEAL.json"),
        "--out-dir", str(args.out_dir / "analysis"),
    ], label="confirmatory_primary_unblinding", out_dir=args.out_dir)


def plot(args: argparse.Namespace) -> None:
    if not analysis_status(args.out_dir)["ready"]:
        raise RuntimeError("Stage-2 analysis is incomplete; run analyze first")
    run([
        args.python,
        str(ROOT / "experiments/plot_channel_lira_phase7_stage2.py"),
        "--analysis-dir", str(args.out_dir / "analysis"),
        "--png",
    ], label="confirmatory_primary_plots", out_dir=args.out_dir)


def print_plan(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    print("Phase-7 Stage-2 frozen primary plan")
    print(f"protocol_sha256={sha256(args.protocol)}")
    print(f"targets={len(EXPECTED_TARGETS)} cells={len(EXPECTED_CELLS)} references=64")
    print(f"condition={PRIMARY_MODE} shots={PRIMARY_SHOTS} simulator_seeds=0..9")
    print(f"reference_shots={PRIMARY_REFERENCE_SHOTS}")
    print(f"target_shots={PRIMARY_TARGET_SHOTS}")
    print(f"combined_shots={PRIMARY_TOTAL_SHOTS}")
    print(
        f"score_workers={args.score_workers} "
        f"aer_threads_per_worker={args.aer_max_parallel_threads}"
    )
    print("pipeline=target -> references -> exact -> score -> seal -> analyze -> plot")
    print("secondary_conditions_reachable=false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "plan", "status", "target", "references", "exact", "score",
            "seal", "analyze", "plot", "all",
        ),
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
    parser.add_argument("--acknowledge-shot-budget", type=int, default=0)
    parser.add_argument("--target-gpus", default="0,1")
    parser.add_argument("--reference-gpus", default="0,1")
    parser.add_argument("--reference-jobs-per-gpu", type=int, default=1)
    parser.add_argument("--score-device", default="cuda")
    parser.add_argument(
        "--score-workers",
        type=int,
        default=1,
        help=(
            "parallel structural-cell scoring workers; targets within one cell "
            "remain sequential for cache safety (maximum: 4)"
        ),
    )
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--qiskit-batch-size", type=int, default=16)
    parser.add_argument("--aer-max-parallel-threads", type=int, default=128)
    parser.add_argument("--record-bootstrap", type=int, default=200)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    for name in (
        "protocol", "protocol_lock", "candidate_probe", "targets", "out_dir",
        "run_root", "reference_dir", "snapshot",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.noisy_dir = (args.noisy_dir or args.out_dir / "noisy_lira").resolve()
    if (
        args.cpu_threads < 1 or args.reference_jobs_per_gpu < 1
        or args.score_workers < 1
        or args.qiskit_batch_size < 1 or args.aer_max_parallel_threads < 1
        or args.record_bootstrap < 1
    ):
        parser.error("thread, batch, and bootstrap settings must be positive")
    if args.score_workers > len(EXPECTED_CELLS):
        parser.error(f"--score-workers cannot exceed {len(EXPECTED_CELLS)}")
    return args


def main() -> None:
    args = parse_args()
    protocol = validate_locked_stage(
        args, require_ack=args.stage not in {"plan", "status"}
    )
    if args.stage == "plan":
        print_plan(args, protocol)
        return
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
    if args.stage == "seal" or (
        args.stage == "all" and status["ready_to_seal"]
        and not status["raw_output_seal"]["ready"]
    ):
        seal_outputs(args, protocol)
    status = refresh(args, protocol)
    if args.stage == "analyze" or (
        args.stage == "all" and status["ready_to_analyze"]
        and not status["analysis"]["ready"]
    ):
        analyze(args, protocol)
    status = refresh(args, protocol)
    if args.stage == "plot" or (
        args.stage == "all" and status["analysis"]["ready"]
        and not status["plots"]["ready"]
    ):
        plot(args)
    status = refresh(args, protocol)
    print(
        f"[STATUS] complete={status['complete']} "
        f"targets={status['artifacts']['target_bundles_ready']}/12 "
        f"references={status['artifacts']['reference_checkpoints_ready']}/64 "
        f"noisy={status['noisy_scores']['targets_ready']}/12 "
        f"sealed={status['raw_output_seal']['ready']}",
        flush=True,
    )
    print(f"[REPORT] {args.out_dir / 'EXECUTION_REPORT.md'}", flush=True)
    if args.require_complete and not status["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
