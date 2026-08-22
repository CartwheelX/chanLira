#!/usr/bin/env python3
"""Run a guarded one-target/four-reference true noisy-LiRA canary.

The canary is a plumbing and reconstruction gate, not statistical evidence. It
trains one target and four balanced reference QNNs, retains every checkpoint,
requires a hash-validated frozen IBM-derived noise snapshot, and then executes the
target and references through the same ideal/noisy finite-shot oracle.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REVIEWER = ROOT / "reviewer_tools"
for directory in (ROOT, REVIEWER):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

DEFAULT_TARGETS = ROOT / "reviewer_targets/channel_lira_noisy_reference_canary.csv"
DEFAULT_OUT = ROOT / "channel_lira_results/noisy_reference_canary_phase5"
REQUIRED_MODULES = (
    "numpy", "pandas", "torch", "sklearn", "qiskit", "qiskit_aer",
    "qiskit_ibm_runtime", "torchquantum",
)
TOKEN_ENV_NAMES = ("QISKIT_IBM_TOKEN", "IBM_QUANTUM_TOKEN")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_target(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Canary manifest must contain exactly one target; found {len(rows)}")
    row = rows[0]
    required = {
        "target_id", "experiment", "architecture", "fm_kind", "n_wires",
        "reps", "depth", "model_seed", "data_seed", "structural_cell_id",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"Canary manifest is missing columns: {missing}")
    if row["target_id"] != "MNIST_QNN_eff_su2_r1_d2_s43":
        raise ValueError("Canary target is frozen to the compute-minimal Phase-3 cell/seed")
    if row["structural_cell_id"] != "eff_su2_r1_d2":
        raise ValueError("Canary structural cell does not match the frozen protocol")
    return row


def canonical_cell(row: dict[str, str]) -> str:
    base = row["structural_cell_id"].split("|", 1)[0]
    weight_decay = float(row.get("weight_decay", "0") or 0.0)
    block = str(row.get("block_id", "")).strip()
    suffix = "" if block.lower() in {"", "nan", "none"} else f"_block{block}"
    return f"{base}_wd{weight_decay:g}{suffix}"


def module_status() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES}


def target_artifacts(row: dict[str, str], run_root: Path) -> dict[str, Any]:
    directory = run_root / row["experiment"] / row["target_id"]
    paths = {
        "model": directory / "target_model.pt",
        "attack_data": directory / "target_attack_data.pt",
        "summary": directory / "target_export_summary.json",
    }
    present = {
        key: path.is_file() and path.stat().st_size > 0 for key, path in paths.items()
    }
    return {
        "ready": all(present.values()),
        "directory": str(directory.resolve()),
        "paths": {key: str(path.resolve()) for key, path in paths.items()},
        "present": present,
        "sha256": {
            key: sha256(path) for key, path in paths.items() if present[key]
        },
    }


def inspect_reference_bank(
    row: dict[str, str], reference_dir: Path, count: int
) -> dict[str, Any]:
    cell = canonical_cell(row)
    directory = reference_dir / "reference_models" / cell
    records: list[dict[str, Any]] = []
    inclusion_rows = []
    fingerprints = set()
    errors = []
    for reference_id in range(count):
        score = directory / f"reference_{reference_id:03d}.npz"
        checkpoint = score.with_suffix(".pt")
        record: dict[str, Any] = {
            "reference_id": reference_id,
            "score": str(score.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "score_ready": score.is_file() and score.stat().st_size > 0,
            "checkpoint_ready": checkpoint.is_file() and checkpoint.stat().st_size > 0,
            "metadata_valid": False,
            "checkpoint_metadata_valid": False,
        }
        score_fingerprint = None
        if record["score_ready"]:
            record["score_sha256"] = sha256(score)
            try:
                with np.load(score, allow_pickle=False) as saved:
                    if int(saved["reference_id"]) != reference_id:
                        raise ValueError("reference ID mismatch")
                    if int(saved["num_references"]) != count:
                        raise ValueError("reference count mismatch")
                    if str(saved["structural_cell"]) != cell:
                        raise ValueError("canonical cell mismatch")
                    inclusion_rows.append(saved["inclusion"].astype(bool))
                    score_fingerprint = str(saved["candidate_fingerprint"])
                    fingerprints.add(score_fingerprint)
                    record["metadata_valid"] = True
                    record["candidate_count"] = int(len(saved["inclusion"]))
                    record["reference_seed"] = int(saved["reference_seed"])
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                errors.append(record["error"])
        if record["checkpoint_ready"]:
            record["checkpoint_sha256"] = sha256(checkpoint)
            try:
                import torch

                try:
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                except TypeError:
                    payload = torch.load(checkpoint, map_location="cpu")
                if int(payload["reference_id"]) != reference_id:
                    raise ValueError("checkpoint reference ID mismatch")
                if int(payload["num_references"]) != count:
                    raise ValueError("checkpoint reference count mismatch")
                if str(payload["structural_cell"]) != cell:
                    raise ValueError("checkpoint canonical cell mismatch")
                if score_fingerprint is None or str(payload["candidate_fingerprint"]) != score_fingerprint:
                    raise ValueError("checkpoint/score candidate fingerprint mismatch")
                if not isinstance(payload.get("state_dict"), dict) or not payload["state_dict"]:
                    raise ValueError("checkpoint has no model state dictionary")
                record["checkpoint_metadata_valid"] = True
            except Exception as exc:
                record["checkpoint_error"] = f"{type(exc).__name__}: {exc}"
                errors.append(record["checkpoint_error"])
        records.append(record)
    balanced = False
    if len(inclusion_rows) == count:
        inclusion = np.stack(inclusion_rows)
        balanced = bool(np.all(inclusion.sum(axis=0) == count // 2))
        if not balanced:
            errors.append("candidate inclusion is not exactly balanced")
    ready = (
        len(records) == count
        and all(record["score_ready"] for record in records)
        and all(record["checkpoint_ready"] for record in records)
        and all(record["metadata_valid"] for record in records)
        and all(record["checkpoint_metadata_valid"] for record in records)
        and balanced
        and len(fingerprints) == 1
    )
    return {
        "ready": ready,
        "cell": cell,
        "directory": str(directory.resolve()),
        "expected": count,
        "scores_ready": sum(bool(record["score_ready"]) for record in records),
        "checkpoints_ready": sum(bool(record["checkpoint_ready"]) for record in records),
        "metadata_valid": sum(bool(record["metadata_valid"]) for record in records),
        "checkpoint_metadata_valid": sum(
            bool(record["checkpoint_metadata_valid"]) for record in records
        ),
        "balanced_inclusion": balanced,
        "candidate_fingerprints": sorted(fingerprints),
        "errors": errors,
        "references": records,
    }


def inspect_snapshot(snapshot: Path) -> dict[str, Any]:
    manifest = snapshot / "snapshot_manifest.json"
    result: dict[str, Any] = {
        "ready": False,
        "directory": str(snapshot.resolve()),
        "manifest": str(manifest.resolve()),
        "manifest_exists": manifest.is_file(),
    }
    if not manifest.is_file():
        return result
    try:
        from reviewer_tools.qurift_qiskit_bridge import load_backend_noise_snapshot

        context = load_backend_noise_snapshot(snapshot, require_noise=True)
        result.update({
            "ready": context.noise_model is not None,
            "manifest_sha256": sha256(manifest),
            "backend_name": context.metadata.resolved_backend_name,
            "calibration_timestamp": context.metadata.calibration_timestamp,
            "noise_model_loaded": context.metadata.noise_model_loaded,
            "basis_gate_count": len(context.metadata.basis_gates),
            "coupling_edge_count": len(context.metadata.coupling_map),
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_scores(row: dict[str, str], out_dir: Path) -> dict[str, Any]:
    exact = out_dir / "references/lira_reference_mia_raw.csv"
    noisy = out_dir / "noisy_lira/noisy_lira_raw.csv"
    target_score = out_dir / "noisy_lira/target_scores" / f"{row['target_id']}.csv"
    metadata = out_dir / "noisy_lira/metadata" / f"{row['target_id']}.json"
    paths = {"exact": exact, "noisy": noisy, "target_score": target_score, "metadata": metadata}
    present = {
        key: path.is_file() and path.stat().st_size > 0 for key, path in paths.items()
    }
    return {
        "ready": all(present.values()),
        "exact_ready": present["exact"],
        "noisy_ready": present["noisy"] and present["target_score"] and present["metadata"],
        "paths": {key: str(path.resolve()) for key, path in paths.items()},
        "present": present,
    }


def reconstruction_diagnostic(row: dict[str, str], out_dir: Path) -> dict[str, Any]:
    current = out_dir / "references/sample_scores" / f"{row['target_id']}.npz"
    retained = (
        ROOT / "reviewer_results/lira_reference_mia/sample_scores"
        / f"{row['target_id']}.npz"
    )
    result: dict[str, Any] = {
        "available": False,
        "current": str(current.resolve()),
        "retained_phase3": str(retained.resolve()),
    }
    if not current.is_file() or not retained.is_file():
        return result
    try:
        with np.load(current, allow_pickle=False) as new, np.load(
            retained, allow_pickle=False
        ) as old:
            new_ids = [str(value) for value in new["sample_ids"]]
            old_ids = [str(value) for value in old["sample_ids"]]
            if set(new_ids) != set(old_ids):
                raise ValueError("candidate sample-ID sets differ")
            old_index = {sample_id: index for index, sample_id in enumerate(old_ids)}
            order = np.asarray([old_index[sample_id] for sample_id in new_ids])
            new_score = new["observed_log_odds"].astype(float)
            old_score = old["observed_log_odds"][order].astype(float)
            member_equal = bool(np.array_equal(new["membership"], old["membership"][order]))
            label_equal = bool(np.array_equal(new["labels"], old["labels"][order]))
            result.update({
                "available": True,
                "candidate_count": len(new_ids),
                "sample_ids_equal": True,
                "membership_equal": member_equal,
                "labels_equal": label_equal,
                "observed_log_odds_exact_match": bool(
                    np.array_equal(new_score, old_score)
                ),
                "observed_log_odds_allclose_1e_6": bool(
                    np.allclose(new_score, old_score, rtol=1e-6, atol=1e-6)
                ),
                "observed_log_odds_mae": float(np.mean(np.abs(new_score - old_score))),
                "observed_log_odds_max_abs": float(np.max(np.abs(new_score - old_score))),
                "observed_log_odds_correlation": float(np.corrcoef(new_score, old_score)[0, 1]),
            })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_status(args: argparse.Namespace) -> dict[str, Any]:
    row = read_target(args.targets)
    modules = module_status()
    target = target_artifacts(row, args.run_root)
    references = inspect_reference_bank(row, args.reference_dir, args.num_references)
    snapshot = inspect_snapshot(args.snapshot)
    scores = inspect_scores(row, args.out_dir)
    reconstruction = reconstruction_diagnostic(row, args.out_dir)
    credentials = {
        "environment_token_present": any(bool(os.environ.get(name)) for name in TOKEN_ENV_NAMES),
        "saved_account_probe": "not_run",
        "saved_account_count": None,
    }
    if modules.get("qiskit_ibm_runtime"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                from qiskit_ibm_runtime import QiskitRuntimeService
                accounts = QiskitRuntimeService.saved_accounts()
            credentials["saved_account_probe"] = "ok"
            credentials["saved_account_count"] = len(accounts)
        except Exception as exc:
            credentials["saved_account_probe"] = f"{type(exc).__name__}: {exc}"
    blockers = []
    if not all(modules.values()):
        blockers.append("runtime modules missing: " + ", ".join(
            name for name, available in modules.items() if not available
        ))
    if not target["ready"]:
        blockers.append("canary target checkpoint/attack payload is incomplete")
    if not references["ready"]:
        blockers.append(
            f"canary reference bank is incomplete ({references['checkpoints_ready']}/"
            f"{args.num_references} checkpoints)"
        )
    if not snapshot["ready"]:
        blockers.append("hash-validated frozen backend noise snapshot is unavailable")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "protocol": "one target; one compute-minimal Phase-3 cell; four balanced references",
        "statistical_scope": "execution/reconstruction canary only; not inferential evidence",
        "target_manifest": str(args.targets.resolve()),
        "target_manifest_sha256": sha256(args.targets),
        "target": row,
        "num_references": args.num_references,
        "shots": args.shots,
        "simulator_seeds": args.simulator_seeds,
        "modules": modules,
        "credentials": credentials,
        "target_artifacts": target,
        "reference_bank": references,
        "snapshot": snapshot,
        "scores": scores,
        "reconstruction_diagnostic": reconstruction,
        "ready_for_exact_scoring": bool(target["ready"] and references["ready"]),
        "ready_to_score": bool(target["ready"] and references["ready"] and snapshot["ready"]),
        "complete": bool(scores["ready"]),
        "blockers": blockers,
    }


def summarize_auc(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["mode"], row["attack"]), []).append(float(row["auc"]))
    return [
        {
            "mode": mode,
            "attack": attack,
            "mean_auc": float(np.mean(values)),
            "n_simulator_seeds": len(values),
        }
        for (mode, attack), values in sorted(grouped.items())
    ]


def summarize_exact_auc(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {"attack": row["attack"], "auc": float(row["auc"])}
        for row in rows
    ]


def write_report(status: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if status["complete"]:
        status["auc_summary"] = summarize_auc(
            Path(status["scores"]["paths"]["noisy"])
        )
    if status["scores"]["exact_ready"]:
        status["exact_auc_summary"] = summarize_exact_auc(
            Path(status["scores"]["paths"]["exact"])
        )
    (out_dir / "STATUS.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    target = status["target_artifacts"]
    refs = status["reference_bank"]
    snapshot = status["snapshot"]
    scores = status["scores"]
    lines = [
        "# True noisy-reference LiRA canary",
        "",
        "This is an execution and reconstruction gate, not statistical evidence.",
        "",
        f"## Status: {'COMPLETE' if status['complete'] else ('READY TO SCORE' if status['ready_to_score'] else 'BLOCKED')}",
        "",
        "| Gate | Available | Required |",
        "|---|---:|---:|",
        f"| Target checkpoint bundle | {int(target['ready'])} | 1 |",
        f"| Exact reference score files | {refs['scores_ready']} | {refs['expected']} |",
        f"| Reference checkpoints | {refs['checkpoints_ready']} | {refs['expected']} |",
        f"| Valid checkpoint metadata | {refs['checkpoint_metadata_valid']} | {refs['expected']} |",
        f"| Balanced reference bank | {int(refs['balanced_inclusion'])} | 1 |",
        f"| Hash-validated noise snapshot | {int(snapshot['ready'])} | 1 |",
        f"| Exact-only score bundle | {int(scores['exact_ready'])} | 1 |",
        f"| Exact/noisy score bundle | {int(scores['ready'])} | 1 |",
        "",
        "## Frozen protocol",
        "",
        f"- Target: `{status['target']['target_id']}`.",
        f"- Structural cell: `{status['target']['structural_cell_id']}` (selected before execution as the compute-minimal Phase-3 cell).",
        f"- Reference models: {status['num_references']}, exactly balanced per candidate.",
        f"- Shots: {status['shots']}; simulator seeds: `{status['simulator_seeds']}`.",
        "- Both target and references must use the same hash-validated snapshot.",
        "- Four references are sufficient only to test plumbing; they are not a paper baseline.",
    ]
    if status["blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in status["blockers"])
    credentials = status["credentials"]
    if not snapshot["ready"] and not credentials["environment_token_present"] and not credentials["saved_account_count"]:
        lines.extend([
            "- No IBM token environment variable or saved runtime account was detected; snapshot capture needs user-provided credentials and a backend name.",
        ])
    reconstruction = status["reconstruction_diagnostic"]
    if reconstruction.get("available"):
        lines.extend([
            "", "## Retired-checkpoint comparison", "",
            f"- Candidate IDs, labels, and memberships align: `{bool(reconstruction['sample_ids_equal'] and reconstruction['labels_equal'] and reconstruction['membership_equal'])}`.",
            f"- Exact log-odds bitwise match: `{reconstruction['observed_log_odds_exact_match']}`.",
            f"- Log-odds MAE: {reconstruction['observed_log_odds_mae']:.6f}; correlation: {reconstruction['observed_log_odds_correlation']:.6f}.",
        ])
        if not reconstruction["observed_log_odds_allclose_1e_6"]:
            lines.append(
                "- The retired Phase-3 weights were not recovered. This is a newly retrained canary target and must be treated as a new experimental block."
            )
    if status["scores"]["exact_ready"]:
        lines.extend([
            "", "## Four-reference exact-only plumbing output", "",
            "| Attack | AUC |", "|---|---:|",
        ])
        for row in status["exact_auc_summary"]:
            lines.append(f"| {row['attack']} | {row['auc']:.4f} |")
        lines.extend([
            "",
            "These four-reference values are recorded to validate exact scoring only; they are not inferential results or substitutes for the 16-reference baseline.",
        ])
    if status["complete"]:
        lines.extend([
            "", "## Mean AUC by finite-shot mode", "",
            "| Mode | Attack | Mean AUC | Simulator seeds |",
            "|---|---|---:|---:|",
        ])
        for row in status["auc_summary"]:
            lines.append(
                f"| {row['mode']} | {row['attack']} | {row['mean_auc']:.4f} | "
                f"{row['n_simulator_seeds']} |"
            )
        lines.extend([
            "",
            "These values validate execution only. With one target, two simulator seeds, and four references, they must not be used as scientific evidence.",
        ])
    lines.extend([
        "", "## Artifact roots", "",
        f"- Target runs: `{target['directory']}`",
        f"- References: `{refs['directory']}`",
        f"- Snapshot: `{snapshot['directory']}`",
        f"- Machine-readable status: `{(out_dir / 'STATUS.json').resolve()}`",
    ])
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(command: list[str], *, cwd: Path) -> None:
    print("[RUN] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def train_target(args: argparse.Namespace) -> None:
    run([
        args.python, str(ROOT / "reviewer_tools/run_target_table_dgx.py"),
        "--targets", str(args.targets), "--repo-root", str(ROOT),
        "--out", str(args.run_root), "--gpus", args.target_gpu,
        "--jobs-per-gpu", "1", "--cpu-threads", str(args.cpu_threads),
        "--no-largest-first", "--resume",
    ], cwd=ROOT)


def train_references(args: argparse.Namespace) -> None:
    run([
        args.python, str(ROOT / "reviewer_tools/run_lira_reference_multigpu.py"),
        "--targets", str(args.targets), "--repo-root", str(ROOT),
        "--run-root", str(args.run_root), "--out-dir", str(args.reference_dir),
        "--num-references", str(args.num_references), "--save-reference-checkpoints",
        "--phase", "train", "--seed", str(args.seed), "--gpus", args.reference_gpus,
        "--jobs-per-gpu", "1", "--cpu-threads", str(args.cpu_threads), "--resume",
    ], cwd=ROOT)


def exact_score(args: argparse.Namespace) -> None:
    status = build_status(args)
    if not status["ready_for_exact_scoring"]:
        raise RuntimeError(
            "Canary target/reference bank is not ready for exact scoring: "
            + "; ".join(status["blockers"])
        )
    run([
        args.python, str(ROOT / "reviewer_tools/run_lira_reference_multigpu.py"),
        "--targets", str(args.targets), "--repo-root", str(ROOT),
        "--run-root", str(args.run_root), "--out-dir", str(args.reference_dir),
        "--num-references", str(args.num_references), "--phase", "score",
        "--bootstrap", str(args.bootstrap), "--seed", str(args.seed),
        "--gpus", args.target_gpu, "--jobs-per-gpu", "1",
        "--cpu-threads", str(args.cpu_threads), "--resume",
    ], cwd=ROOT)


def capture_snapshot(args: argparse.Namespace) -> None:
    if not args.backend_name:
        raise RuntimeError("Snapshot capture requires --backend-name or QURIFT_NOISE_BACKEND")
    if args.snapshot.exists() and any(args.snapshot.iterdir()):
        existing = inspect_snapshot(args.snapshot)
        if existing["ready"]:
            print(f"[SKIP] validated snapshot exists: {args.snapshot}")
            return
        raise RuntimeError(f"Refusing to overwrite incomplete snapshot directory: {args.snapshot}")
    args.snapshot.mkdir(parents=True, exist_ok=False)
    command = [
        args.python, str(ROOT / "reviewer_tools/probe_ibm_backend_noise.py"),
        "--backend-name", args.backend_name, "--require-noise",
        "--out", str(args.snapshot / "probe.json"),
        "--snapshot-dir", str(args.snapshot),
    ]
    if args.ibm_account_name:
        command.extend(["--ibm-account-name", args.ibm_account_name])
    run(command, cwd=ROOT)


def score(args: argparse.Namespace) -> None:
    status = build_status(args)
    if not status["ready_to_score"]:
        raise RuntimeError("Canary is not ready to score: " + "; ".join(status["blockers"]))
    row = status["target"]
    exact_score(args)
    run([
        args.python, str(ROOT / "satml_tools/noisy_lira.py"),
        "--repo-root", str(ROOT), "--targets", str(args.targets),
        "--run-root", str(args.run_root), "--reference-dir", str(args.reference_dir),
        "--out-dir", str(args.out_dir / "noisy_lira"), "--snapshot", str(args.snapshot),
        "--target-id", row["target_id"], "--num-references", str(args.num_references),
        "--modes", "ideal_shot,noisy_shot", "--shots", str(args.shots),
        "--simulator-seeds", args.simulator_seeds, "--bootstrap", str(args.bootstrap),
        "--seed", str(args.seed), "--device", "cuda", "--resume",
    ], cwd=ROOT)
    run([
        args.python, str(ROOT / "satml_tools/noisy_lira.py"),
        "--repo-root", str(ROOT), "--targets", str(args.targets),
        "--run-root", str(args.run_root), "--reference-dir", str(args.reference_dir),
        "--out-dir", str(args.out_dir / "noisy_lira"), "--snapshot", str(args.snapshot),
        "--aggregate",
    ], cwd=ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("status", "target", "references", "exact", "snapshot", "score", "all")
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--num-references", type=int, default=4)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--simulator-seeds", default="0,1")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--target-gpu", default="0")
    parser.add_argument("--reference-gpus", default="0,1,2")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--backend-name", default=os.environ.get("QURIFT_NOISE_BACKEND"))
    parser.add_argument("--ibm-account-name", default=os.environ.get("QURIFT_IBM_ACCOUNT_NAME"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    args.targets = args.targets.resolve()
    args.out_dir = args.out_dir.resolve()
    args.run_root = (args.run_root or args.out_dir / "runs").resolve()
    args.reference_dir = (args.reference_dir or args.out_dir / "references").resolve()
    args.snapshot = (args.snapshot or args.out_dir / "backend_snapshot").resolve()
    if args.num_references != 4:
        parser.error("The canary protocol is frozen to exactly four reference models")
    if args.shots < 1:
        parser.error("--shots must be positive")
    seeds = [part.strip() for part in args.simulator_seeds.split(",") if part.strip()]
    if len(seeds) != 2 or any(not value.isdigit() for value in seeds):
        parser.error("The canary protocol requires exactly two integer simulator seeds")
    return args


def main() -> None:
    args = parse_args()
    read_target(args.targets)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"target", "all"}:
        train_target(args)
    if args.stage in {"references", "all"}:
        train_references(args)
    current = build_status(args)
    if args.stage == "exact" or (
        args.stage == "all" and current["ready_for_exact_scoring"]
        and not current["scores"]["exact_ready"]
    ):
        exact_score(args)
    if args.stage in {"snapshot", "all"}:
        current = inspect_snapshot(args.snapshot)
        if current["ready"]:
            print(f"[SKIP] validated snapshot exists: {args.snapshot}")
        elif args.backend_name:
            capture_snapshot(args)
        elif args.stage == "snapshot":
            raise RuntimeError("Snapshot capture requires --backend-name")
        else:
            print("[BLOCKED] snapshot capture skipped: no backend name was provided")
    current = build_status(args)
    if args.stage == "score" or (args.stage == "all" and current["ready_to_score"]):
        score(args)
        current = build_status(args)
    write_report(current, args.out_dir)
    print(f"[STATUS] complete={current['complete']} ready_to_score={current['ready_to_score']}")
    print(f"[REPORT] {args.out_dir / 'REPORT.md'}")
    if args.require_complete and not current["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
