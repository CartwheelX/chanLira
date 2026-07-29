#!/usr/bin/env python3
"""
Run compact reviewer target tables against experiments/qurift_main.py.

Requires qurift_main.py to expose --seed and to correctly pass fm_eff_reps into
the Efficient-SU2 encoder constructor. Use --dry-run first.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def gpu_ids_from_arg(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_command(row: pd.Series, script: Path, out_root: Path) -> tuple[List[str], Path, Path]:
    target_id = str(row["target_id"])
    dataset = str(row["dataset"]).lower()
    architecture = str(row["architecture"]).lower()
    out_dir = out_root / str(row.get("experiment", "reviewer")) / target_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = (out_dir / "target_model.pt").resolve()
    attack_path = (out_dir / "target_attack_data.pt").resolve()

    cmd = [
        sys.executable, str(script),
        "--model-type", architecture,
        "--dataset", dataset,
        "--run-id", str(int(row.get("source_run_id", -1))),
        "--seed", str(int(row["seed"])),
        "--random-ops", "0",
        "--vector-train", str(int(row.get("vector_train", 200))),
        "--vector-valid", str(int(row.get("vector_valid", 200))),
        "--vector-test", str(int(row.get("vector_test", 200))),
        "--batch-size", str(int(row.get("batch_size", 16))),
        "--epochs", str(int(row.get("epochs", 100))),
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

    if bool(row.get("ql_rev", False)):
        cmd.append("--qlayer-ent-wire-reverse")
    if bool(row.get("extra_feats", False)):
        cmd.append("--extra-feats")

    if dataset == "moons":
        cmd += ["--moons-noise", "0.3"]
    elif dataset == "circles":
        cmd += ["--circles-noise", "0.3"]
    elif dataset == "blobs":
        cmd += [
            "--blobs-n-features", "4",
            "--blobs-cluster-std", "2.1",
            "--blobs-center-distance", "3.5",
        ]

    fm = str(row["fm_kind"]).lower()
    reps = str(int(row["reps"]))
    pad = str(row.get("pad_mode", "wrap"))
    fm_ent = str(row.get("fm_ent", "linear"))
    fm_op = str(row.get("fm_op", "cx"))
    if fm == "z":
        cmd += ["--fm-z-reps", reps, "--fm-z-pad-mode", pad]
    elif fm == "zz":
        cmd += [
            "--fm-zz-reps", reps,
            "--fm-zz-pad-mode", pad,
            "--fm-zz-entanglement", fm_ent,
        ]
    elif fm == "eff_su2":
        if fm_op.upper() in {"NA", "NAN", "NONE", ""}:
            fm_op = "cx"
        cmd += [
            "--fm-eff-reps", reps,
            "--fm-eff-pad-mod", pad,
            "--fm-eff-ent-kind", fm_ent,
            "--fm-eff-twoq-op", fm_op,
        ]
    else:
        raise ValueError(f"Unsupported feature map: {fm}")

    return cmd, out_dir / "train.log", attack_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--script", type=Path, default=Path("experiments/qurift_main.py"))
    ap.add_argument("--out", type=Path, default=Path("reviewer_runs"))
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--jobs-per-gpu", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(args.targets)
    gpus = gpu_ids_from_arg(args.gpus)
    if not gpus:
        gpus = [-1]
    slots = []
    for gid in gpus:
        slots.extend([gid] * max(1, args.jobs_per_gpu))
    lock = threading.Lock()
    available = list(slots)

    def acquire() -> int:
        while True:
            with lock:
                if available:
                    return available.pop()
            time.sleep(1)

    def release(gid: int) -> None:
        with lock:
            available.append(gid)

    def run_row(row: pd.Series) -> Dict[str, object]:
        cmd, log_path, attack_path = build_command(row, args.script, args.out)
        if args.resume and attack_path.exists():
            return {"target_id": row["target_id"], "status": "skipped", "attack_path": str(attack_path)}
        if args.dry_run:
            print(" ".join(cmd))
            return {"target_id": row["target_id"], "status": "dry_run", "attack_path": str(attack_path)}

        gid = acquire()
        t0 = time.time()
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "" if gid < 0 else str(gid)
            env["OMP_NUM_THREADS"] = "2"
            env["MKL_NUM_THREADS"] = "2"
            with log_path.open("w", encoding="utf-8") as fh:
                proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
            status = "ok" if proc.returncode == 0 and attack_path.exists() else "error"
            return {
                "target_id": row["target_id"],
                "status": status,
                "return_code": proc.returncode,
                "gpu": gid,
                "seconds": round(time.time() - t0, 3),
                "log_path": str(log_path),
                "attack_path": str(attack_path),
            }
        finally:
            release(gid)

    results = []
    with ThreadPoolExecutor(max_workers=len(slots)) as ex:
        futures = [ex.submit(run_row, row) for _, row in targets.iterrows()]
        for i, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            results.append(result)
            print(f"[{i}/{len(futures)}] {result['target_id']} -> {result['status']}")

    pd.DataFrame(results).to_csv(args.out / "run_status.csv", index=False)
    print(f"[OK] Status: {(args.out / 'run_status.csv').resolve()}")


if __name__ == "__main__":
    main()
