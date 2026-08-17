#!/usr/bin/env python3
"""Parallel launcher for the compact QuRiFT noisy finite-shot subset."""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def parse_int_list(text: str) -> List[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def parse_gpu_ids(text: str) -> List[int]:
    value = text.strip().lower()
    if value in {"cpu", "none", "-1"}:
        return [-1]
    if value == "auto":
        command = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"]
        output = subprocess.check_output(command, text=True)
        ids = [int(line.strip()) for line in output.splitlines() if line.strip()]
        if not ids:
            raise RuntimeError("nvidia-smi reported no GPUs")
        return ids
    ids = parse_int_list(text)
    if not ids:
        raise ValueError("--gpus must be auto, cpu, or a comma-separated GPU list")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate GPU IDs: {ids}")
    return ids


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def target_complete(
    target_out: Path,
    *,
    shots: List[int],
    simulator_seeds: List[int],
    modes: List[str],
    require_noise: bool,
) -> bool:
    status_path = target_out / "condition_status.csv"
    if not status_path.exists():
        return False
    try:
        status = pd.read_csv(status_path)
    except Exception:
        return False
    if status.empty or "status" not in status:
        return False
    bad = status[status["status"].astype(str).isin(["error", "skipped_no_noise_model"])]
    if not bad.empty:
        return False

    expected = set()
    if "exact" in modes:
        expected.add(("exact", 0, -1))
    for mode in [value for value in modes if value in {"ideal_shot", "noisy_shot"}]:
        for shot in shots:
            for seed in simulator_seeds:
                expected.add((mode, int(shot), int(seed)))
    observed = set(
        zip(
            status.get("mode", pd.Series(dtype=str)).astype(str),
            pd.to_numeric(status.get("shots", pd.Series(dtype=float)), errors="coerce").fillna(-999).astype(int),
            pd.to_numeric(status.get("simulator_seed", pd.Series(dtype=float)), errors="coerce").fillna(-999).astype(int),
        )
    )
    if not expected.issubset(observed):
        return False

    if require_noise and "noisy_shot" in modes:
        metadata = load_json(target_out / "backend_noise_metadata.json")
        if metadata.get("noise_model_loaded") is not True:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-root", type=Path, default=Path("reviewer_runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("reviewer_results/noisy_sanity/raw"))
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path("reviewer_tools/qurift_noisy_eval.py"),
    )
    parser.add_argument("--backend-name", required=True)
    parser.add_argument("--noise-backend-name", default=None)
    parser.add_argument("--ibm-account-name", default=None)
    parser.add_argument("--modes", default="exact,ideal_shot,noisy_shot")
    parser.add_argument("--shots", default="128,512,1024")
    parser.add_argument("--simulator-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--transpiler-seed", type=int, default=2026)
    parser.add_argument("--optimization-level", type=int, choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--n-member", type=int, default=100)
    parser.add_argument("--n-nonmember", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--qiskit-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--stagger-seconds", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-noise", action="store_true")
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    args.targets = args.targets.resolve()
    args.run_root = args.run_root.resolve()
    args.out_dir = args.out_dir.resolve()
    evaluator = args.evaluator
    if not evaluator.is_absolute():
        evaluator = (args.repo_root / evaluator).resolve()

    if not args.targets.exists():
        parser.error(f"Targets file not found: {args.targets}")
    if not evaluator.exists():
        parser.error(f"Evaluator not found: {evaluator}")
    if args.jobs_per_gpu < 1:
        parser.error("--jobs-per-gpu must be >= 1")

    targets = pd.read_csv(args.targets)
    required = {"target_id", "experiment", "model_seed", "structural_cell_id"}
    missing = required - set(targets.columns)
    if missing:
        parser.error(f"Targets CSV missing fields: {sorted(missing)}")

    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    shots = parse_int_list(args.shots)
    simulator_seeds = parse_int_list(args.simulator_seeds)
    require_noise = not args.allow_missing_noise

    gpu_ids = parse_gpu_ids(args.gpus)
    slots: queue.Queue[int] = queue.Queue()
    for gpu in gpu_ids:
        for _ in range(args.jobs_per_gpu):
            slots.put(gpu)
    max_workers = slots.qsize()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.out_dir / "noisy_run_status.csv"
    failure_path = args.out_dir / "noisy_run_failures.csv"

    results: List[Dict[str, object]] = []

    def build_command(target_id: str, gpu_id: int) -> List[str]:
        command = [
            sys.executable,
            str(evaluator),
            "--repo-root", str(args.repo_root),
            "--targets", str(args.targets),
            "--target-id", target_id,
            "--run-root", str(args.run_root),
            "--out-dir", str(args.out_dir),
            "--modes", args.modes,
            "--shots", args.shots,
            "--simulator-seeds", args.simulator_seeds,
            "--transpiler-seed", str(args.transpiler_seed),
            "--optimization-level", str(args.optimization_level),
            "--backend-name", args.backend_name,
            "--n-member", str(args.n_member),
            "--n-nonmember", str(args.n_nonmember),
            "--sample-seed", str(args.sample_seed),
            "--batch-size", str(args.batch_size),
            "--qiskit-batch-size", str(args.qiskit_batch_size),
            "--bootstrap", str(args.bootstrap),
            "--device", "cpu" if gpu_id < 0 else "cuda",
        ]
        if args.noise_backend_name:
            command += ["--noise-backend-name", args.noise_backend_name]
        if args.ibm_account_name:
            command += ["--ibm-account-name", args.ibm_account_name]
        if require_noise:
            command.append("--require-noise")
        if args.resume:
            command.append("--resume")
        return command

    if args.dry_run:
        preview_gpu = gpu_ids[0]
        for target_id in targets["target_id"].astype(str):
            print(" ".join(build_command(target_id, preview_gpu)))
        return

    def run_row(row: pd.Series) -> Dict[str, object]:
        target_id = str(row["target_id"])
        target_out = args.out_dir / target_id
        if args.resume and target_complete(
            target_out,
            shots=shots,
            simulator_seeds=simulator_seeds,
            modes=modes,
            require_noise=require_noise,
        ):
            return {
                "target_id": target_id,
                "structural_cell_id": row["structural_cell_id"],
                "model_seed": int(row["model_seed"]),
                "status": "skipped_complete",
                "gpu": "",
                "return_code": 0,
                "seconds": 0.0,
                "log_path": str(log_dir / f"{target_id}.log"),
                "target_out": str(target_out),
                "error": "",
            }

        gpu_id = slots.get()
        t0 = time.time()
        log_path = log_dir / f"{target_id}.log"
        try:
            if args.stagger_seconds > 0:
                time.sleep(args.stagger_seconds)
            command = build_command(target_id, gpu_id)
            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = "" if gpu_id < 0 else str(gpu_id)
            for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
                env[key] = str(args.cpu_threads)
            env["PYTHONUNBUFFERED"] = "1"
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    command,
                    cwd=args.repo_root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            complete = target_complete(
                target_out,
                shots=shots,
                simulator_seeds=simulator_seeds,
                modes=modes,
                require_noise=require_noise,
            )
            status = "ok" if process.returncode == 0 and complete else "error"
            error = "" if status == "ok" else (
                f"return_code={process.returncode}; complete={complete}; inspect {log_path}"
            )
            return {
                "target_id": target_id,
                "structural_cell_id": row["structural_cell_id"],
                "model_seed": int(row["model_seed"]),
                "status": status,
                "gpu": gpu_id,
                "return_code": process.returncode,
                "seconds": round(time.time() - t0, 3),
                "log_path": str(log_path),
                "target_out": str(target_out),
                "error": error,
            }
        except Exception as exc:
            return {
                "target_id": target_id,
                "structural_cell_id": row.get("structural_cell_id", ""),
                "model_seed": row.get("model_seed", ""),
                "status": "error",
                "gpu": gpu_id,
                "return_code": -1,
                "seconds": round(time.time() - t0, 3),
                "log_path": str(log_path),
                "target_out": str(target_out),
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            slots.put(gpu_id)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_row, row) for _, row in targets.iterrows()]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            atomic_csv(pd.DataFrame(results), status_path)
            failures = pd.DataFrame([item for item in results if item["status"] == "error"])
            atomic_csv(failures, failure_path)
            print(
                f"[{index}/{len(futures)}] {result['target_id']} -> {result['status']} "
                f"(gpu={result['gpu']}, seconds={result['seconds']})"
            )

    n_error = sum(item["status"] == "error" for item in results)
    print(f"[OK] Incremental status: {status_path}")
    print(f"[OK] Failures: {failure_path}")
    if n_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
