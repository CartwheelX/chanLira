# python reviewer_tools/run_reviewer_subset_dgx.py \
#   --targets reviewer_targets/matched_gap_mia_targets.csv \
#   --out reviewer_runs \
#   --gpus 0,1,2,3,4,5,6,7 \
#   --jobs-per-gpu 2 \
#   --cpu-threads 2 \
#   --resume

# You can also let it detect all GPUs:

# python reviewer_tools/run_reviewer_subset_dgx.py \
#   --targets reviewer_targets/matched_gap_mia_targets.csv \
#   --out reviewer_runs \
#   --gpus auto \
#   --jobs-per-gpu 2 \
#   --cpu-threads 2 \
#   --resume


#!/usr/bin/env python3
"""DGX-aware launcher for QuRiFT reviewer target tables."""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def detect_gpu_ids() -> List[int]:
    cmd = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"]
    try:
        output = subprocess.check_output(cmd, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not detect GPUs with nvidia-smi.") from exc
    ids = [int(line.strip()) for line in output.splitlines() if line.strip()]
    if not ids:
        raise RuntimeError("nvidia-smi returned no GPUs.")
    return ids


def gpu_ids_from_arg(value: str) -> List[int]:
    value = value.strip().lower()
    if value == "auto":
        return detect_gpu_ids()
    ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("--gpus must be 'auto' or a unique comma-separated list.")
    return ids


def as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off", "", "nan", "none"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def int_or_default(value: object, default: int) -> int:
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    return int(value)


def build_command(row: pd.Series, script: Path, out_root: Path) -> Tuple[List[str], Path, Path, Path]:
    target_id = str(row["target_id"])
    dataset = str(row["dataset"]).strip().lower()
    architecture = str(row["architecture"]).strip().lower()
    out_dir = out_root / str(row.get("experiment", "reviewer")) / target_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = (out_dir / "target_model.pt").resolve()
    attack_path = (out_dir / "target_attack_data.pt").resolve()
    log_path = out_dir / "train.log"

    cmd = [
        sys.executable, str(script.resolve()),
        "--model-type", architecture,
        "--dataset", dataset,
        "--run-id", str(int_or_default(row.get("source_run_id", -1), -1)),
        "--seed", str(int(row["seed"])),
        "--random-ops", "0",
        "--vector-train", str(int_or_default(row.get("vector_train", 200), 200)),
        "--vector-valid", str(int_or_default(row.get("vector_valid", 200), 200)),
        "--vector-test", str(int_or_default(row.get("vector_test", 200), 200)),
        "--batch-size", str(int_or_default(row.get("batch_size", 16), 16)),
        "--epochs", str(int_or_default(row.get("epochs", 100), 100)),
        "--n-wires", str(int(row["n_wires"])),
        "--depth", str(int(row["depth"])),
        "--qlayer-ent-kind", str(row["ql_ent"]),
        "--qlayer-twoq-op", str(row["ql_op"]),
        "--fm-kind", str(row["fm_kind"]),
        "--train_target",
        "--export-attack-data",
        "--attack-feature-mode", "pv+stats",
        "--target-model-path", str(model_path),
        "--attack-data-out", str(attack_path),
    ]

    if as_bool(row.get("ql_rev", False)):
        cmd.append("--qlayer-ent-wire-reverse")
    if as_bool(row.get("extra_feats", False)):
        cmd.append("--extra-feats")

    if dataset == "moons":
        cmd += ["--moons-noise", "0.3"]
    elif dataset == "circles":
        cmd += ["--circles-noise", "0.3"]
    elif dataset == "blobs":
        cmd += ["--blobs-n-features", "4", "--blobs-cluster-std", "2.1", "--blobs-center-distance", "3.5"]

    fm = str(row["fm_kind"]).strip().lower()
    reps = str(int(row["reps"]))
    pad = str(row.get("pad_mode", "wrap")).strip()
    fm_ent = str(row.get("fm_ent", "linear")).strip()
    fm_op = str(row.get("fm_op", "cx")).strip()

    if fm == "z":
        cmd += ["--fm-z-reps", reps, "--fm-z-pad-mode", pad]
    elif fm == "zz":
        cmd += ["--fm-zz-reps", reps, "--fm-zz-pad-mode", pad, "--fm-zz-entanglement", fm_ent]
    elif fm == "eff_su2":
        if fm_op.upper() in {"NA", "NAN", "NONE", ""}:
            fm_op = "cx"
        cmd += ["--fm-eff-reps", reps, "--fm-eff-pad-mod", pad, "--fm-eff-ent-kind", fm_ent, "--fm-eff-twoq-op", fm_op]
    else:
        raise ValueError(f"Unsupported feature map: {fm}")

    return cmd, log_path, model_path, attack_path


def estimated_cost(row: pd.Series) -> float:
    wires = max(1, int_or_default(row.get("n_wires", 1), 1))
    depth = max(1, int_or_default(row.get("depth", 1), 1))
    reps = max(1, int_or_default(row.get("reps", 1), 1))
    batch = max(1, int_or_default(row.get("batch_size", 1), 1))
    epochs = max(1, int_or_default(row.get("epochs", 1), 1))
    return float((2 ** wires) * depth * reps * batch * epochs)


def write_status_atomic(results: List[Dict[str, object]], path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(results).to_csv(temp, index=False)
    temp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--script", type=Path, default=Path("experiments/qurift_main.py"))
    ap.add_argument("--out", type=Path, default=Path("reviewer_runs"))
    ap.add_argument("--gpus", default="auto", help="auto or 0,1,2,3,4,5,6,7")
    ap.add_argument("--jobs-per-gpu", type=int, default=2)
    ap.add_argument("--cpu-threads", type=int, default=2)
    ap.add_argument("--stagger-seconds", type=float, default=0.25)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-largest-first", action="store_true")
    args = ap.parse_args()

    if args.jobs_per_gpu < 1 or args.cpu_threads < 1:
        ap.error("jobs-per-gpu and cpu-threads must be at least 1")
    if not args.targets.exists():
        ap.error(f"Targets CSV not found: {args.targets}")
    if not args.script.exists():
        ap.error(f"Training script not found: {args.script}")

    args.out.mkdir(parents=True, exist_ok=True)
    status_path = args.out / "run_status.csv"
    targets = pd.read_csv(args.targets)

    required = {"target_id", "dataset", "architecture", "seed", "n_wires", "depth", "reps", "ql_ent", "ql_op", "fm_kind"}
    missing = required - set(targets.columns)
    if missing:
        ap.error(f"Targets CSV missing: {sorted(missing)}")

    if not args.no_largest_first:
        targets = targets.copy()
        targets["_estimated_cost"] = targets.apply(estimated_cost, axis=1)
        targets = targets.sort_values("_estimated_cost", ascending=False)

    if args.dry_run:
        for _, row in targets.iterrows():
            cmd, _, _, _ = build_command(row, args.script, args.out)
            print(" ".join(cmd))
        print(f"[DRY RUN] Printed {len(targets)} commands.")
        return

    gpu_ids = gpu_ids_from_arg(args.gpus)
    slots: queue.Queue[int] = queue.Queue()
    for gid in gpu_ids:
        for _ in range(args.jobs_per_gpu):
            slots.put(gid)
    max_workers = slots.qsize()

    print(f"GPUs: {gpu_ids}")
    print(f"Jobs/GPU: {args.jobs_per_gpu}")
    print(f"Concurrent jobs: {max_workers}")
    print(f"Targets: {len(targets)}")

    results: List[Dict[str, object]] = []

    def run_row(row: pd.Series) -> Dict[str, object]:
        target_id = str(row["target_id"])
        try:
            cmd, log_path, model_path, attack_path = build_command(row, args.script, args.out)
            if args.resume and attack_path.exists() and attack_path.stat().st_size > 0:
                return {"target_id": target_id, "status": "skipped", "return_code": 0, "gpu": "", "seconds": 0.0,
                        "log_path": str(log_path), "model_path": str(model_path), "attack_path": str(attack_path), "error": ""}

            gid = slots.get()
            t0 = time.time()
            try:
                if args.stagger_seconds > 0:
                    time.sleep(args.stagger_seconds)
                env = os.environ.copy()
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                env["CUDA_VISIBLE_DEVICES"] = str(gid)
                env["OMP_NUM_THREADS"] = str(args.cpu_threads)
                env["MKL_NUM_THREADS"] = str(args.cpu_threads)
                env["OPENBLAS_NUM_THREADS"] = str(args.cpu_threads)
                env["NUMEXPR_NUM_THREADS"] = str(args.cpu_threads)
                env["TORCH_NUM_THREADS"] = str(args.cpu_threads)
                env["PYTHONUNBUFFERED"] = "1"
                env["QURIFT_JOB_ID"] = target_id

                with log_path.open("w", encoding="utf-8") as fh:
                    proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=Path.cwd(), check=False)

                attack_ok = attack_path.exists() and attack_path.stat().st_size > 0
                model_ok = model_path.exists() and model_path.stat().st_size > 0
                status = "ok" if proc.returncode == 0 and attack_ok and model_ok else "error"
                if proc.returncode != 0:
                    error = f"nonzero_exit={proc.returncode}"
                elif not model_ok:
                    error = "missing_or_empty_model"
                elif not attack_ok:
                    error = "missing_or_empty_attack_data"
                else:
                    error = ""
                return {"target_id": target_id, "status": status, "return_code": proc.returncode, "gpu": gid,
                        "seconds": round(time.time() - t0, 3), "log_path": str(log_path),
                        "model_path": str(model_path), "attack_path": str(attack_path), "error": error}
            finally:
                slots.put(gid)
        except Exception as exc:
            return {"target_id": target_id, "status": "error", "return_code": -1, "gpu": "", "seconds": 0.0,
                    "log_path": "", "model_path": "", "attack_path": "", "error": repr(exc)}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_row, row) for _, row in targets.iterrows()]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            write_status_atomic(results, status_path)
            print(f"[{i}/{len(futures)}] {result['target_id']} -> {result['status']} (gpu={result['gpu']}, sec={result['seconds']})")
            if result.get("error"):
                print(f"  error: {result['error']}")

    ok = sum(r["status"] == "ok" for r in results)
    skipped = sum(r["status"] == "skipped" for r in results)
    errors = sum(r["status"] == "error" for r in results)
    print(f"\nDone: ok={ok}, skipped={skipped}, errors={errors}")
    print(f"Status: {status_path.resolve()}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()