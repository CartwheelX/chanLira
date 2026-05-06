#!/usr/bin/env python3
# run_selected_configs_for_mia.py

import argparse
import csv
import os
import re
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd


# ---------------------------
# MASTER FILE MAPS
# ---------------------------
MNIST_MASTER_FILES = {
    "QNN":  "experiments/gen_results/qnn_extensive_results.csv",
    "HQNN": "experiments/gen_results/hqnn_extensive_results.csv",
    "QCNN": "experiments/gen_results/qcnn_extensive_results.csv",
}

SYNTHETIC_MASTER_FILES = {
    "Moons":   "experiments/gen_results/master_results_full_pipeline_moon.csv",
    "Moons_s": "experiments/gen_results/master_results_full_pipeline_moons.csv",
    "Blobs":   "experiments/gen_results/master_results_full_pipeline_blobs.csv",
    "Circles": "experiments/gen_results/master_results_full_pipeline_circles.csv",
}

DEFAULT_SCRIPT_PATH = "experiments/qurift_main.py"


# ---------------------------
# BASE ARGS
# ---------------------------
BASE_ARGS = {
    ("mnist", "qnn"): [
        "--model-type", "qnn",
        "--dataset", "mnist",
        "--random-ops", "0",
        "--vector-train", "200",
        "--vector-valid", "200",
        "--vector-test", "200",
        "--batch-size", "16",
        "--epochs", "60",
        "--train_target",
    ],
    ("mnist", "hqnn"): [
        "--model-type", "hqnn",
        "--dataset", "mnist",
        "--random-ops", "0",
        "--vector-train", "200",
        "--vector-valid", "200",
        "--vector-test", "200",
        "--batch-size", "16",
        "--epochs", "100",
        "--train_target",
    ],
    ("mnist", "qcnn"): [
        "--model-type", "qcnn",
        "--dataset", "mnist",
        "--random-ops", "0",
        "--vector-train", "100",
        "--vector-valid", "100",
        "--vector-test", "100",
        "--batch-size", "16",
        "--epochs", "100",
        "--train_target",
    ],
    ("moons", "qnn"): [
        "--model-type", "qnn",
        "--dataset", "moons",
        "--random-ops", "0",
        "--vector-train", "50",
        "--vector-valid", "50",
        "--vector-test", "50",
        "--batch-size", "8",
        "--epochs", "100",
        "--moons-noise", "0.3",
        "--train_target",
        "--extra-feats",
    ],
    ("blobs", "qnn"): [
        "--model-type", "qnn",
        "--dataset", "blobs",
        "--random-ops", "0",
        "--vector-train", "50",
        "--vector-valid", "50",
        "--vector-test", "50",
        "--batch-size", "8",
        "--epochs", "100",
        "--blobs-n-features", "4",
        "--blobs-cluster-std", "2.1",
        "--blobs-center-distance", "3.5",
        "--train_target",
        "--extra-feats",
    ],
    ("circles", "qnn"): [
        "--model-type", "qnn",
        "--dataset", "circles",
        "--random-ops", "0",
        "--vector-train", "100",
        "--vector-valid", "100",
        "--vector-test", "100",
        "--batch-size", "8",
        "--epochs", "100",
        "--circles-noise", "0.3",
        "--train_target",
        "--extra-feats",
    ],
}


# ---------------------------
# DATA STRUCT
# ---------------------------
@dataclass
class Job:
    dataset: str
    architecture: str
    role: str
    run_id: int
    config: Dict[str, Any]
    script_path: str
    out_dir: str
    extra_train_args: List[str]
    save_model: bool = False
    dry_run: bool = False


# ---------------------------
# HELPERS
# ---------------------------
def _norm(s: str) -> str:
    return str(s).strip().lower()


def parse_kv_config_string(s: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(s, str) or not s.strip():
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v in ("NA", "na", "None", "none", ""):
            out[k] = None
        elif v in ("True", "true"):
            out[k] = True
        elif v in ("False", "false"):
            out[k] = False
        else:
            if re.fullmatch(r"-?\d+", v):
                out[k] = int(v)
            elif re.fullmatch(r"-?\d+\.\d+", v):
                out[k] = float(v)
            else:
                out[k] = v
    return out


def pick_master_csv(dataset: str, architecture: str) -> Path:
    ds = _norm(dataset)
    arch = str(architecture).strip().upper()
    if ds == "mnist":
        return Path(MNIST_MASTER_FILES[arch])
    if ds == "moons":
        p = Path(SYNTHETIC_MASTER_FILES["Moons"])
        if not p.exists():
            p = Path(SYNTHETIC_MASTER_FILES["Moons_s"])
        return p
    if ds == "blobs":
        return Path(SYNTHETIC_MASTER_FILES["Blobs"])
    if ds == "circles":
        return Path(SYNTHETIC_MASTER_FILES["Circles"])
    raise ValueError(f"Unknown dataset '{dataset}' (expected mnist/moons/blobs/circles).")


def read_targets_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    colmap: Dict[str, str] = {}
    for c in df.columns:
        lc = _norm(c)
        if lc in ("arch", "architecture", "model", "target_model"):
            colmap[c] = "architecture"
        elif lc == "dataset":
            colmap[c] = "dataset"
        elif lc in ("role", "case", "category"):
            colmap[c] = "role"
        elif lc in ("run_id", "runid", "rid"):
            colmap[c] = "run_id"
        elif lc in ("config", "config_str", "configuration"):
            colmap[c] = "config"
    df = df.rename(columns=colmap)

    if "dataset" not in df.columns:
        df["dataset"] = "mnist"
    if "architecture" not in df.columns:
        df["architecture"] = "QNN"
    if "role" not in df.columns:
        df["role"] = "selected"
    if "run_id" not in df.columns:
        raise ValueError(f"Targets table must have a run_id column: {path}")
    return df


# ---------------------------
# GPU leasing helpers
# ---------------------------
def query_gpu_stats() -> List[Dict[str, int]]:
    q = (
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.free,memory.used,memory.total "
        "--format=csv,noheader,nounits"
    )
    out = subprocess.check_output(q.split()).decode("utf-8").strip()
    stats: List[Dict[str, int]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        idx, util, mem_free, mem_used, mem_total = [x.strip() for x in line.split(",")]
        stats.append({
            "index": int(idx),
            "util": int(util),
            "mem_free": int(mem_free),
            "mem_used": int(mem_used),
            "mem_total": int(mem_total),
        })
    return stats


def pick_best_gpu(
    gpu_stats: List[Dict[str, int]],
    active_counts: Dict[int, int],
    jobs_per_gpu: int,
    max_gpu_util: int,
    min_free_mem_mb: int,
    allow_gpus: Optional[List[int]],
) -> Optional[Tuple[int, Dict[str, int]]]:
    candidates: List[Dict[str, int]] = []
    for s in gpu_stats:
        gid = s["index"]
        if allow_gpus is not None and gid not in allow_gpus:
            continue
        if active_counts.get(gid, 0) >= jobs_per_gpu:
            continue
        if s["util"] > max_gpu_util:
            continue
        if s["mem_free"] < min_free_mem_mb:
            continue
        candidates.append(s)

    if not candidates:
        return None

    candidates.sort(key=lambda s: (
        active_counts.get(s["index"], 0),  # fewer active first
        s["util"],                         # lower util first
        -s["mem_free"],                    # more free mem first
    ))
    best = candidates[0]
    return best["index"], best


def acquire_gpu_slot(
    lock: threading.Lock,
    active_counts: Dict[int, int],
    jobs_per_gpu: int,
    max_gpu_util: int,
    min_free_mem_mb: int,
    allow_gpus: Optional[List[int]],
    poll_seconds: float,
) -> Tuple[int, Dict[str, int], int]:
    """
    Returns (gpu_id, snapshot_stats, active_now_after_acquire)
    """
    while True:
        stats = query_gpu_stats()
        with lock:
            picked = pick_best_gpu(
                stats,
                active_counts=active_counts,
                jobs_per_gpu=jobs_per_gpu,
                max_gpu_util=max_gpu_util,
                min_free_mem_mb=min_free_mem_mb,
                allow_gpus=allow_gpus,
            )
            if picked is not None:
                gid, snap = picked
                active_counts[gid] = active_counts.get(gid, 0) + 1
                active_now = active_counts[gid]
                return gid, snap, active_now
        time.sleep(poll_seconds)


def release_gpu_slot(lock: threading.Lock, active_counts: Dict[int, int], gid: int) -> int:
    with lock:
        active_counts[gid] = max(0, active_counts.get(gid, 1) - 1)
        return active_counts[gid]


# ---------------------------
# JOBS FROM TARGETS
# ---------------------------
def build_jobs_from_targets(
    targets_df: pd.DataFrame,
    script_path: str,
    out_dir: str,
    extra_train_args: List[str],
    save_model: bool,
    dry_run: bool,
) -> List[Job]:

    jobs: List[Job] = []
    master_cache: Dict[Tuple[str, str], pd.DataFrame] = {}

    for _, row in targets_df.iterrows():
        dataset = str(row.get("dataset", "mnist"))
        role = str(row.get("role", "selected"))
        architecture = str(row.get("architecture", "QNN"))
        run_id = int(row["run_id"])

        cfg: Dict[str, Any] = {}

        if "config" in targets_df.columns and isinstance(row.get("config", None), str):
            cfg.update(parse_kv_config_string(row["config"]))

        for k in ["fm_kind","n_wires","reps","fm_op","depth","ql_ent","ql_op","pad_mode","fm_ent","ql_rev"]:
            if k in targets_df.columns and pd.notna(row.get(k, None)):
                cfg[k] = row[k]

        need = ["fm_kind","n_wires","reps","depth","ql_ent","ql_op"]
        if any(cfg.get(k, None) is None for k in need):
            master_path = pick_master_csv(dataset, architecture)
            cache_key = (_norm(dataset), architecture.strip().upper())
            if cache_key not in master_cache:
                if not master_path.exists():
                    raise FileNotFoundError(f"Master CSV not found: {master_path}")
                master_cache[cache_key] = pd.read_csv(master_path)

            mdf = master_cache[cache_key]
            if "run_id" not in mdf.columns:
                raise ValueError(f"Master CSV missing run_id: {master_path}")

            hits = mdf[mdf["run_id"].astype(int) == int(run_id)]
            if hits.empty:
                raise ValueError(f"run_id={run_id} not found in master: {master_path}")

            mrow = hits.iloc[0].to_dict()
            for k in ["fm_kind","n_wires","reps","fm_op","depth","ql_ent","ql_op","pad_mode","fm_ent","ql_rev"]:
                if k in mrow and cfg.get(k, None) is None:
                    cfg[k] = mrow[k]

        for k, v in list(cfg.items()):
            if isinstance(v, str):
                cfg[k] = v.strip()

        jobs.append(Job(
            dataset=dataset,
            architecture=architecture,
            role=role,
            run_id=run_id,
            config=cfg,
            script_path=script_path,
            out_dir=out_dir,
            extra_train_args=list(extra_train_args),
            save_model=save_model,
            dry_run=dry_run,
        ))

    return jobs


# ---------------------------
# CMD + RUN
# ---------------------------
def build_cmd(job: Job) -> Tuple[List[str], Path, Path, Path]:
    ds = _norm(job.dataset)
    arch = _norm(job.architecture)
    role = _norm(job.role)

    key = (ds, arch)
    if key not in BASE_ARGS:
        raise ValueError(f"No BASE_ARGS for (dataset={ds}, arch={arch}). Add it to BASE_ARGS.")

    cfg = dict(job.config)

    fm_kind = cfg.get("fm_kind")
    n_wires = cfg.get("n_wires")
    reps = cfg.get("reps")
    depth = cfg.get("depth")
    ql_ent = cfg.get("ql_ent")
    ql_op = cfg.get("ql_op")

    pad_mode = cfg.get("pad_mode", "wrap")
    fm_ent = cfg.get("fm_ent", "linear")
    fm_op = cfg.get("fm_op")
    ql_rev = cfg.get("ql_rev", False)

    missing = [k for k in ["fm_kind","n_wires","reps","depth","ql_ent","ql_op"] if cfg.get(k) is None]
    if missing:
        raise ValueError(f"Missing config keys {missing} for run_id={job.run_id} ({job.dataset}/{job.architecture}/{job.role}).")

    run_dir = Path(job.out_dir) / ds / job.architecture.strip().upper() / role / f"run_{job.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "train.log"

    # absolute paths so save-model is reliable
    model_path = (run_dir / f"target_model_id{job.run_id}_{job.architecture.strip().upper()}.pt").resolve()
    attack_data_path = Path(str(model_path.with_suffix("")) + "_attack_data.pt").resolve()

    cmd = [sys.executable, job.script_path] + list(BASE_ARGS[key]) + [
        "--run-id", str(job.run_id),
        "--fm-kind", str(fm_kind),
        "--n-wires", str(int(n_wires)),
        "--depth", str(int(depth)),
        "--qlayer-ent-kind", str(ql_ent),
        "--qlayer-twoq-op", str(ql_op),
    ]

    if bool(ql_rev):
        cmd.append("--qlayer-ent-wire-reverse")

    fm_kind_s = str(fm_kind).lower()
    if fm_kind_s == "z":
        cmd += ["--fm-z-reps", str(int(reps)), "--fm-z-pad-mode", str(pad_mode)]
    elif fm_kind_s == "zz":
        cmd += [
            "--fm-zz-reps", str(int(reps)),
            "--fm-zz-pad-mode", str(pad_mode),
            "--fm-zz-entanglement", str(fm_ent),
        ]
    elif fm_kind_s == "pauli":
        cmd += ["--fm-pauli-reps", str(int(reps))]
        if pad_mode is not None:
            cmd += ["--fm-pauli-pad", str(pad_mode)]
        if fm_ent is not None:
            cmd += ["--fm-pauli-entanglement", str(fm_ent)]
    elif fm_kind_s == "eff_su2":
        if fm_op in (None, "NA", "na", ""):
            fm_op = "cx"
        cmd += [
            "--fm-eff-reps", str(int(reps)),
            "--fm-eff-pad-mod", str(pad_mode),
            "--fm-eff-ent-kind", str(fm_ent),
            "--fm-eff-twoq-op", str(fm_op),
        ]
    else:
        raise ValueError(f"Unknown fm_kind='{fm_kind}' for run_id={job.run_id}")

    # pass-through user extras
    cmd += list(job.extra_train_args)

    if job.save_model:
        cmd += ["--target-model-path", str(model_path)]
        cmd += ["--export-attack-data"]
        cmd += ["--attack-data-out", str(attack_data_path)]
    else:
        print(f"[INFO ] run_id={job.run_id} NOT saving model (save_model=False)")
        exit
    # IMPORTANT: avoid accidental empty args causing "unrecognized arguments:"
    cmd = [str(x).strip() for x in cmd if x is not None and str(x).strip() != ""]
    return cmd, log_path, model_path, attack_data_path


def run_one_with_gpu(job: Job, gpu_id: Optional[int], verbose_sched: bool, sched_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    status = "ok"
    err = ""
    out_dir_str = str(Path(job.out_dir) / _norm(job.dataset) / job.architecture.strip().upper() / _norm(job.role) / f"run_{job.run_id}")

    t0 = time.time()

    try:
        cmd, log_path, model_path, attack_data_path = build_cmd(job)

        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = "" if gpu_id is None else str(gpu_id)

        if verbose_sched:
            if gpu_id is None:
                print(f"[SCHED] run_id={job.run_id} {job.dataset}/{job.architecture}/{job.role} -> CPU")
            else:
                util = sched_meta.get("util", -1) if sched_meta else -1
                free_mb = sched_meta.get("mem_free", -1) if sched_meta else -1
                active = sched_meta.get("active", -1) if sched_meta else -1
                jpg = sched_meta.get("jobs_per_gpu", -1) if sched_meta else -1
                print(f"[SCHED] run_id={job.run_id} {job.dataset}/{job.architecture}/{job.role} -> GPU {gpu_id} "
                      f"(util={util}%, free={free_mb}MB, active={active}/{jpg})")

            # also show the command and the exact save path on terminal (not in train.log)
            if job.save_model:
                print(f"[SAVE ] run_id={job.run_id} target_model_path={model_path}")
            print(f"[CMD  ] run_id={job.run_id} " + " ".join(cmd))

        if job.dry_run:
            return {
                "dataset": job.dataset,
                "architecture": job.architecture,
                "role": job.role,
                "run_id": job.run_id,
                "status": "dry_run",
                "error_msg": "",
                "out_dir": out_dir_str,
                "model_path": str(model_path),
                "gpu_used": "" if gpu_id is None else str(gpu_id),
                "secs": 0.0,
            }

        # LOG POLICY: train.log contains ONLY the training script output
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

        if p.returncode != 0:
            status = "error"
            err = f"nonzero_exit={p.returncode}"
        else:
            # enforce that save-model truly produced the file (but DO NOT write into train.log)
            if job.save_model and not model_path.exists():
                status = "error"
                err = "model_not_saved_missing_target_model_path"
            # attack_data is optional sanity check; keep as warning only (terminal)
            if job.save_model and verbose_sched and not attack_data_path.exists():
                print(f"[WARN ] run_id={job.run_id} attack_data not found at: {attack_data_path}")

    except Exception as e:
        status = "error"
        err = str(e)

    secs = time.time() - t0

    if verbose_sched:
        print(f"[DONE ] run_id={job.run_id} status={status} gpu={'' if gpu_id is None else gpu_id} secs={secs:.3f} err={err}")

    return {
        "dataset": job.dataset,
        "architecture": job.architecture,
        "role": job.role,
        "run_id": job.run_id,
        "status": status,
        "error_msg": err,
        "out_dir": out_dir_str,
        "model_path": str(model_path) if job.save_model else "",
        "gpu_used": "" if gpu_id is None else str(gpu_id),
        "secs": round(secs, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", action="append", required=True)
    ap.add_argument("--script", default=DEFAULT_SCRIPT_PATH)
    ap.add_argument("--out", default="experiments/saved_models_for_mia")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--min-free-mem-mb", type=int, default=8000)
    ap.add_argument("--jobs-per-gpu", type=int, default=4)

    ap.add_argument("--max-cpu-workers", type=int, default=2)
    ap.add_argument("--status-csv", default=None)

    ap.add_argument("--max-gpu-util", type=int, default=75)
    ap.add_argument("--gpu-allowlist", default=None)
    ap.add_argument("--poll-seconds", type=float, default=2.0)

    ap.add_argument("--verbose-sched", action="store_true",
                    help="Print scheduling (job->GPU) decisions to TERMINAL only.")

    ap.add_argument("extra", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    out_dir = args.out
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Everything after "--" is passed to training script
    extra_train_args = args.extra
    if len(extra_train_args) > 0 and extra_train_args[0] == "--":
        extra_train_args = extra_train_args[1:]
    extra_train_args = [a.strip() for a in extra_train_args if isinstance(a, str) and a.strip() != ""]

    all_targets = []
    for t in args.targets:
        df = read_targets_table(Path(t))
        df["_targets_file"] = t
        all_targets.append(df)
    targets_df = pd.concat(all_targets, ignore_index=True)

    jobs = build_jobs_from_targets(
        targets_df=targets_df,
        script_path=args.script,
        out_dir=out_dir,
        extra_train_args=extra_train_args,
        save_model=args.save_model,
        dry_run=args.dry_run,
    )

    allow_gpus: Optional[List[int]] = None
    if args.gpu_allowlist:
        allow_gpus = [int(x.strip()) for x in args.gpu_allowlist.split(",") if x.strip()]

    have_gpu = True
    try:
        stats0 = query_gpu_stats()
        all_gpu_ids = sorted([s["index"] for s in stats0])
    except Exception:
        have_gpu = False
        all_gpu_ids = []

    if have_gpu and allow_gpus is None:
        allow_gpus = all_gpu_ids

    lock = threading.Lock()
    active_counts: Dict[int, int] = {}

    if have_gpu:
        gpu_count = len(allow_gpus) if allow_gpus else 0
        max_workers = max(1, gpu_count * max(1, args.jobs_per_gpu))
    else:
        max_workers = max(1, args.max_cpu_workers)

    print(f"Total selected configs: {len(jobs)}")
    print(f"Parallel workers: {max_workers}")
    print(f"Artifacts root: {out_dir}")
    if have_gpu:
        print(f"GPU allowlist: {allow_gpus}")
        print(f"Policy: jobs_per_gpu={args.jobs_per_gpu}, max_gpu_util={args.max_gpu_util}%, min_free_mem_mb={args.min_free_mem_mb}")
    else:
        print("No usable GPU found (or nvidia-smi unavailable). CPU-only mode.")

    def runner(job: Job) -> Dict[str, Any]:
        if not have_gpu:
            return run_one_with_gpu(job, None, args.verbose_sched, sched_meta=None)

        gid, snap, active_now = acquire_gpu_slot(
            lock=lock,
            active_counts=active_counts,
            jobs_per_gpu=args.jobs_per_gpu,
            max_gpu_util=args.max_gpu_util,
            min_free_mem_mb=args.min_free_mem_mb,
            allow_gpus=allow_gpus,
            poll_seconds=args.poll_seconds,
        )

        sched_meta = {
            "util": snap.get("util", -1),
            "mem_free": snap.get("mem_free", -1),
            "active": active_now,
            "jobs_per_gpu": args.jobs_per_gpu,
        }

        try:
            return run_one_with_gpu(job, gid, args.verbose_sched, sched_meta=sched_meta)
        finally:
            left = release_gpu_slot(lock, active_counts, gid)
            if args.verbose_sched:
                print(f"[REL  ] run_id={job.run_id} released GPU {gid} (active now {left}/{args.jobs_per_gpu})")

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(runner, job): job for job in jobs}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            print(f"[{done}/{len(jobs)}] {r['dataset']}/{r['architecture']}/{r['role']} run_id={r['run_id']} -> {r['status']} (gpu={r['gpu_used']})")

    status_csv = args.status_csv or str(Path(out_dir) / "selected_runs_status.csv")
    with open(status_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(results[0].keys()) if results else ["status"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\nDone. Status CSV: {status_csv}")


if __name__ == "__main__":
    main()
