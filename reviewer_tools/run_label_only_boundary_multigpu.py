#!/usr/bin/env python3
"""Evaluate the QuRiFT class-label-only boundary MIA on multiple GPUs."""
from __future__ import annotations

import argparse
import queue
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from run_lira_reference_multigpu import parse_gpus, run_commands


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
        "--out-dir",
        type=Path,
        default=Path("reviewer_results/label_only_boundary"),
    )
    parser.add_argument("--n-member", type=int, default=None)
    parser.add_argument("--n-nonmember", type=int, default=None)
    parser.add_argument("--anchors", type=int, default=16)
    parser.add_argument("--binary-steps", type=int, default=10)
    parser.add_argument("--norm", choices=["l1", "l2", "linf"], default="l2")
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args()

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
        targets["architecture"].astype(str).str.lower().isin(["qnn", "hqnn", "qcnn"])
    ].copy()
    if targets.empty:
        raise SystemExit("No supported targets found")
    worker = Path(__file__).with_name("qurift_label_only_boundary.py").resolve()
    tasks: list[dict[str, Any]] = []
    for _, row in targets.iterrows():
        target_id = str(row["target_id"])
        command = [
            sys.executable,
            str(worker),
            "score-target",
            "--repo-root",
            str(repo_root),
            "--targets",
            str(targets_path),
            "--run-root",
            str(run_root),
            "--out-dir",
            str(out_dir),
            "--target-id",
            target_id,
            "--anchors",
            str(args.anchors),
            "--binary-steps",
            str(args.binary_steps),
            "--norm",
            args.norm,
            "--query-batch-size",
            str(args.query_batch_size),
            "--bootstrap",
            str(args.bootstrap),
            "--seed",
            str(args.seed),
            "--device",
            "cuda",
        ]
        if args.n_member is not None:
            command.extend(["--n-member", str(args.n_member)])
        if args.n_nonmember is not None:
            command.extend(["--n-nonmember", str(args.n_nonmember)])
        if args.resume:
            command.append("--resume")
        tasks.append(
            {
                "name": f"label_only_{target_id}",
                "kind": "label_only_boundary",
                "target_id": target_id,
                "structural_cell_id": row.get("structural_cell_id", ""),
                "reference_id": "",
                "command": command,
                "repo_root": str(repo_root),
                "cpu_threads": args.cpu_threads,
            }
        )
    if args.max_jobs is not None:
        tasks = tasks[: args.max_jobs]

    gpus = parse_gpus(args.gpus, dry_run=args.dry_run)
    concurrency = len(gpus) * args.jobs_per_gpu
    slots: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        for _ in range(args.jobs_per_gpu):
            slots.put(gpu)
    print(
        f"GPUs={gpus}; jobs_per_gpu={args.jobs_per_gpu}; "
        f"concurrency={concurrency}; targets={len(tasks)}",
        flush=True,
    )
    status = run_commands(
        tasks,
        gpu_slots=slots,
        concurrency=concurrency,
        logs_dir=out_dir / "logs",
        status_path=out_dir / "target_scoring_status.csv",
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return
    if status.empty or not status["status"].eq("ok").all():
        raise SystemExit(
            f"Label-only evaluation failed; inspect {out_dir / 'target_scoring_status.csv'}"
        )
    if args.max_jobs is None:
        aggregate = [
            sys.executable,
            str(worker),
            "aggregate",
            "--repo-root",
            str(repo_root),
            "--targets",
            str(targets_path),
            "--run-root",
            str(run_root),
            "--out-dir",
            str(out_dir),
            "--bootstrap",
            str(args.bootstrap),
            "--seed",
            str(args.seed),
            "--device",
            "cpu",
        ]
        raise SystemExit(subprocess.call(aggregate, cwd=repo_root))


if __name__ == "__main__":
    main()
