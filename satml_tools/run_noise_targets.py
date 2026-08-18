#!/usr/bin/env python3
"""Run a frozen-snapshot noisy evaluation over a predeclared target manifest."""
from __future__ import annotations

import argparse
import queue
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REVIEWER_TOOLS = ROOT / "reviewer_tools"
for path in (ROOT, REVIEWER_TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reviewer_tools.qurift_qiskit_bridge import load_backend_noise_snapshot
from reviewer_tools.run_lira_reference_multigpu import parse_gpus, run_commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-root", type=Path, default=Path("reviewer_runs"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--modes", default="exact,ideal_shot,noisy_shot")
    parser.add_argument("--query-shot-pairs", required=True)
    parser.add_argument("--simulator-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--n-member", type=int, default=200)
    parser.add_argument("--n-nonmember", type=int, default=200)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--transpiler-seed", type=int, default=2026)
    parser.add_argument("--optimization-level", type=int, choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--qiskit-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-targets", type=int, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    targets_path = (args.targets if args.targets.is_absolute() else repo_root / args.targets).resolve()
    run_root = (args.run_root if args.run_root.is_absolute() else repo_root / args.run_root).resolve()
    out_dir = (args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir).resolve()
    snapshot = (args.snapshot if args.snapshot.is_absolute() else repo_root / args.snapshot).resolve()
    targets = pd.read_csv(targets_path)
    if "target_id" not in targets or targets.target_id.astype(str).duplicated().any():
        raise ValueError("Target manifest must contain unique target_id values")
    if not args.dry_run:
        context = load_backend_noise_snapshot(snapshot, require_noise=True)
        print(
            f"[SNAPSHOT] backend={context.metadata.resolved_backend_name} "
            f"calibration={context.metadata.calibration_timestamp} path={snapshot}",
            flush=True,
        )

    tasks = []
    worker = repo_root / "reviewer_tools" / "qurift_noisy_eval.py"
    for _, row in targets.iterrows():
        target_id = str(row.target_id)
        model_path = run_root / "multiseed_factorial" / target_id / "target_model.pt"
        attack_path = run_root / "multiseed_factorial" / target_id / "target_attack_data.pt"
        if not args.dry_run:
            for required in (model_path, attack_path):
                if not required.is_file() or required.stat().st_size == 0:
                    raise FileNotFoundError(
                        f"Missing imported retained artifact: {required}. Run "
                        "commands/satml_import_legacy_mnist.sh first."
                    )
        command = [
            sys.executable,
            str(worker),
            "--repo-root", str(repo_root),
            "--targets", str(targets_path),
            "--target-id", target_id,
            "--run-root", str(run_root),
            "--out-dir", str(out_dir),
            "--backend-snapshot", str(snapshot),
            "--modes", args.modes,
            "--query-shot-pairs", args.query_shot_pairs,
            "--simulator-seeds", args.simulator_seeds,
            "--n-member", str(args.n_member),
            "--n-nonmember", str(args.n_nonmember),
            "--sample-seed", str(args.sample_seed),
            "--transpiler-seed", str(args.transpiler_seed),
            "--optimization-level", str(args.optimization_level),
            "--qiskit-batch-size", str(args.qiskit_batch_size),
            "--bootstrap", str(args.bootstrap),
            "--bootstrap-seed", str(args.bootstrap_seed),
            "--device", args.device,
            "--require-noise",
        ]
        if args.resume:
            command.append("--resume")
        tasks.append(
            {
                "name": f"{args.study_name}_{target_id}",
                "kind": args.study_name,
                "target_id": target_id,
                "structural_cell_id": str(row.get("structural_cell_id", "")),
                "reference_id": "",
                "command": command,
                "repo_root": str(repo_root),
                "cpu_threads": args.cpu_threads,
            }
        )
    if args.max_targets is not None:
        tasks = tasks[: args.max_targets]

    gpus = parse_gpus(args.gpus, dry_run=args.dry_run)
    slots: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        for _ in range(args.jobs_per_gpu):
            slots.put(gpu)
    concurrency = len(gpus) * args.jobs_per_gpu
    print(
        f"[START] study={args.study_name} targets={len(tasks)} GPUs={gpus} "
        f"jobs_per_gpu={args.jobs_per_gpu} concurrency={concurrency}",
        flush=True,
    )
    status = run_commands(
        tasks,
        gpu_slots=slots,
        concurrency=concurrency,
        logs_dir=out_dir / "logs",
        status_path=out_dir / "target_status.csv",
        dry_run=args.dry_run,
    )
    if not args.dry_run and (status.empty or not status.status.eq("ok").all()):
        raise SystemExit(f"Noise study failed; inspect {out_dir / 'target_status.csv'}")
    print(f"[DONE] study={args.study_name} output={out_dir}", flush=True)


if __name__ == "__main__":
    main()
