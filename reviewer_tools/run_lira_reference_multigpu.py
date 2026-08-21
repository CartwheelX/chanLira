#!/usr/bin/env python3
"""Train and evaluate balanced LiRA reference banks across multiple GPUs."""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from reviewer_common import atomic_write_csv


def parse_gpus(value: str, *, dry_run: bool) -> list[int]:
    value = value.strip().lower()
    if value != "auto":
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
        if not result:
            raise ValueError("--gpus must be 'auto' or a comma-separated list")
        return result
    if dry_run:
        return [0]
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        text=True,
    )
    result = [int(line.strip()) for line in output.splitlines() if line.strip()]
    if not result:
        raise RuntimeError("No GPUs detected")
    return result


def structural_cell(row: pd.Series) -> str:
    explicit = str(row.get("structural_cell_id", "")).strip()
    weight_decay = float(row.get("weight_decay", 0.0) or 0.0)
    block = str(row.get("block_id", "")).strip()
    block_suffix = "" if block.lower() in {"", "nan", "none"} else f"_block{block}"
    if explicit and explicit.lower() not in {"nan", "none"}:
        return f"{explicit.split('|', 1)[0]}_wd{weight_decay:g}{block_suffix}"
    return (
        f"{row.get('architecture', 'qnn')}_{row.get('fm_kind', 'unknown')}"
        f"_r{int(float(row.get('reps', 0)))}_d{int(float(row.get('depth', 0)))}"
        f"_wd{weight_decay:g}{block_suffix}"
    )


def structural_cell_alias(row: pd.Series) -> str:
    """Return the pre-weight-decay cell name used by retained reference banks."""
    explicit = str(row.get("structural_cell_id", "")).strip()
    if explicit and explicit.lower() not in {"nan", "none"}:
        return explicit.split("|", 1)[0]
    return (
        f"{row.get('architecture', 'qnn')}_{row.get('fm_kind', 'unknown')}"
        f"_r{int(float(row.get('reps', 0)))}_d{int(float(row.get('depth', 0)))}"
    )


def run_commands(
    tasks: list[dict[str, Any]],
    *,
    gpu_slots: queue.Queue[int],
    concurrency: int,
    logs_dir: Path,
    status_path: Path,
    dry_run: bool,
) -> pd.DataFrame:
    logs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    def worker(task: dict[str, Any]) -> dict[str, Any]:
        gpu = gpu_slots.get()
        start = time.time()
        try:
            command = task["command"]
            log_path = logs_dir / f"{task['name']}.log"
            if dry_run:
                print(
                    f"[DRY] gpu={gpu} "
                    + " ".join(str(value) for value in command),
                    flush=True,
                )
                return {
                    "name": task["name"],
                    "kind": task["kind"],
                    "target_id": task.get("target_id", ""),
                    "structural_cell_id": task.get("structural_cell_id", ""),
                    "reference_id": task.get("reference_id", ""),
                    "gpu": gpu,
                    "status": "dry_run",
                    "returncode": 0,
                    "seconds": 0.0,
                    "log": str(log_path),
                }
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            environment.setdefault("OMP_NUM_THREADS", str(task["cpu_threads"]))
            environment.setdefault("MKL_NUM_THREADS", str(task["cpu_threads"]))
            with log_path.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    command,
                    cwd=task["repo_root"],
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            return {
                "name": task["name"],
                "kind": task["kind"],
                "target_id": task.get("target_id", ""),
                "structural_cell_id": task.get("structural_cell_id", ""),
                "reference_id": task.get("reference_id", ""),
                "gpu": gpu,
                "status": "ok" if completed.returncode == 0 else "error",
                "returncode": completed.returncode,
                "seconds": time.time() - start,
                "log": str(log_path.resolve()),
            }
        finally:
            gpu_slots.put(gpu)

    if not tasks:
        return pd.DataFrame()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {executor.submit(worker, task): task for task in tasks}
        completed_count = 0
        for future in as_completed(future_map):
            result = future.result()
            rows.append(result)
            completed_count += 1
            print(
                f"[{completed_count}/{len(tasks)}] {result['name']} -> "
                f"{result['status']} (gpu={result['gpu']}, sec={result['seconds']:.1f})",
                flush=True,
            )
            atomic_write_csv(pd.DataFrame(rows), status_path)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("reviewer_targets/multiseed_factorial_targets.csv"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-root", type=Path, default=Path("reviewer_runs"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("reviewer_results/lira_reference_mia")
    )
    parser.add_argument("--num-references", type=int, default=16)
    parser.add_argument(
        "--cells",
        default="all",
        help="Comma-separated structural cells to train, or 'all'.",
    )
    parser.add_argument(
        "--save-reference-checkpoints",
        action="store_true",
        help="Retain reference weights so the same banks can be evaluated under noise.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument(
        "--phase", choices=["all", "train", "score"], default="all"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-jobs", type=int, default=None)
    parser.add_argument("--max-score-jobs", type=int, default=None)
    args = parser.parse_args()

    if args.num_references < 4 or args.num_references % 2:
        raise SystemExit("--num-references must be even and at least 4")
    repo_root = args.repo_root.resolve()
    targets_path = (
        args.targets if args.targets.is_absolute() else repo_root / args.targets
    ).resolve()
    run_root = (
        args.run_root if args.run_root.is_absolute() else repo_root / args.run_root
    ).resolve()
    out_dir = (
        args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(targets_path)
    targets = targets[
        targets["architecture"].astype(str).str.lower().eq("qnn")
    ].copy()
    if targets.empty:
        raise SystemExit("No QNN targets found")
    targets["_cell"] = targets.apply(structural_cell, axis=1)
    targets["_cell_alias"] = targets.apply(structural_cell_alias, axis=1)
    representatives = targets.drop_duplicates("_cell", keep="first")
    if args.cells.strip().lower() != "all":
        selected_cells = {
            value.strip() for value in args.cells.split(",") if value.strip()
        }
        if not selected_cells:
            raise SystemExit("--cells must contain at least one structural cell")
        available_cells = set(representatives["_cell"].astype(str)) | set(
            representatives["_cell_alias"].astype(str)
        )
        missing_cells = sorted(selected_cells - available_cells)
        if missing_cells:
            raise SystemExit(f"Requested structural cells are absent: {missing_cells}")
        target_mask = (
            targets["_cell"].astype(str).isin(selected_cells)
            | targets["_cell_alias"].astype(str).isin(selected_cells)
        )
        targets = targets[target_mask].copy()
        representatives = targets.drop_duplicates("_cell", keep="first")

    gpus = parse_gpus(args.gpus, dry_run=args.dry_run)
    concurrency = len(gpus) * args.jobs_per_gpu
    slots: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        for _ in range(args.jobs_per_gpu):
            slots.put(gpu)
    worker_script = Path(__file__).with_name("qurift_lira_attack.py").resolve()
    logs_dir = out_dir / "logs"
    print(
        f"GPUs={gpus}; jobs_per_gpu={args.jobs_per_gpu}; "
        f"concurrency={concurrency}; cells={len(representatives)}; "
        f"targets={len(targets)}",
        flush=True,
    )

    if args.phase in {"all", "train"}:
        train_tasks: list[dict[str, Any]] = []
        for _, row in representatives.iterrows():
            cell = str(row["_cell"])
            for reference_id in range(args.num_references):
                command = [
                    sys.executable,
                    str(worker_script),
                    "train-reference",
                    "--repo-root",
                    str(repo_root),
                    "--targets",
                    str(targets_path),
                    "--run-root",
                    str(run_root),
                    "--out-dir",
                    str(out_dir),
                    "--reference-dir",
                    str(out_dir),
                    "--target-id",
                    str(row["target_id"]),
                    "--reference-id",
                    str(reference_id),
                    "--num-references",
                    str(args.num_references),
                    "--seed",
                    str(args.seed),
                    "--device",
                    "cuda",
                ]
                if args.epochs is not None:
                    command.extend(["--epochs", str(args.epochs)])
                if args.save_reference_checkpoints:
                    command.append("--save-checkpoint")
                if args.resume:
                    command.append("--resume")
                train_tasks.append(
                    {
                        "name": f"train_{cell}_ref{reference_id:03d}",
                        "kind": "train_reference",
                        "target_id": str(row["target_id"]),
                        "structural_cell_id": cell,
                        "reference_id": reference_id,
                        "command": command,
                        "repo_root": str(repo_root),
                        "cpu_threads": args.cpu_threads,
                    }
                )
        if args.max_train_jobs is not None:
            train_tasks = train_tasks[: args.max_train_jobs]
        train_status = run_commands(
            train_tasks,
            gpu_slots=slots,
            concurrency=concurrency,
            logs_dir=logs_dir,
            status_path=out_dir / "reference_training_status.csv",
            dry_run=args.dry_run,
        )
        if not args.dry_run and (
            train_status.empty or not train_status["status"].eq("ok").all()
        ):
            raise SystemExit(
                f"Reference training failed; inspect {out_dir / 'reference_training_status.csv'}"
            )

    if args.phase in {"all", "score"}:
        score_tasks: list[dict[str, Any]] = []
        for _, row in targets.iterrows():
            target_id = str(row["target_id"])
            command = [
                sys.executable,
                str(worker_script),
                "score-target",
                "--repo-root",
                str(repo_root),
                "--targets",
                str(targets_path),
                "--run-root",
                str(run_root),
                "--out-dir",
                str(out_dir),
                "--reference-dir",
                str(out_dir),
                "--target-id",
                target_id,
                "--num-references",
                str(args.num_references),
                "--bootstrap",
                str(args.bootstrap),
                "--seed",
                str(args.seed),
                "--device",
                "cuda",
            ]
            if args.resume:
                command.append("--resume")
            score_tasks.append(
                {
                    "name": f"score_{target_id}",
                    "kind": "score_target",
                    "target_id": target_id,
                    "structural_cell_id": str(row["_cell"]),
                    "reference_id": "",
                    "command": command,
                    "repo_root": str(repo_root),
                    "cpu_threads": args.cpu_threads,
                }
            )
        if args.max_score_jobs is not None:
            score_tasks = score_tasks[: args.max_score_jobs]
        score_status = run_commands(
            score_tasks,
            gpu_slots=slots,
            concurrency=concurrency,
            logs_dir=logs_dir,
            status_path=out_dir / "target_scoring_status.csv",
            dry_run=args.dry_run,
        )
        if not args.dry_run and (
            score_status.empty or not score_status["status"].eq("ok").all()
        ):
            raise SystemExit(
                f"Target scoring failed; inspect {out_dir / 'target_scoring_status.csv'}"
            )
        if not args.dry_run and args.max_score_jobs is None:
            aggregate_command = [
                sys.executable,
                str(worker_script),
                "aggregate",
                "--repo-root",
                str(repo_root),
                "--targets",
                str(targets_path),
                "--run-root",
                str(run_root),
                "--out-dir",
                str(out_dir),
                "--reference-dir",
                str(out_dir),
                "--num-references",
                str(args.num_references),
                "--bootstrap",
                str(args.bootstrap),
                "--seed",
                str(args.seed),
                "--device",
                "cpu",
            ]
            raise SystemExit(subprocess.call(aggregate_command, cwd=repo_root))


if __name__ == "__main__":
    main()
