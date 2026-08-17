#!/usr/bin/env python3
"""Run encoder-only geometry targets over multiple data seeds.

Each geometry job is independent and receives a unique output directory. The
launcher supports multiple GPUs, deterministic data seeds, resume, incremental
status/failure files, and dry-run operation without requiring a GPU.
"""
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
from typing import Any

import numpy as np
import pandas as pd

from reviewer_common import atomic_write_csv, stable_seed, write_analysis_metadata


def parse_gpu_ids(value: str, *, dry_run: bool, device: str) -> list[int]:
    value = value.strip().lower()
    if device == "cpu":
        return [-1]
    if value != "auto":
        ids = [int(item.strip()) for item in value.split(",") if item.strip()]
        if not ids:
            raise ValueError("--gpus must be 'auto' or a comma-separated list")
        return ids
    if dry_run:
        return [0]
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not auto-detect GPUs; pass --gpus explicitly") from exc
    ids = [int(line.strip()) for line in output.splitlines() if line.strip()]
    if not ids:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return ids


def hierarchical_repetition_bootstrap(
    frame: pd.DataFrame,
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, int, float, float]:
    pivot = frame.pivot_table(
        index=["dataset", "fm_kind", "data_seed"],
        columns="reps",
        values=metric,
        aggfunc="mean",
    )
    if 1 not in pivot.columns or 5 not in pivot.columns:
        return np.nan, np.nan, 0, np.nan, np.nan
    pivot = pivot.dropna(subset=[1, 5]).reset_index()
    pivot["effect"] = pivot[5] - pivot[1]
    observed = float(pivot["effect"].mean())
    observed_sd = float(pivot["effect"].std(ddof=1)) if len(pivot) > 1 else np.nan

    blocks = pivot[["dataset", "fm_kind"]].drop_duplicates().apply(tuple, axis=1).tolist()
    if not blocks:
        return np.nan, np.nan, 0, observed, observed_sd
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        selected_blocks = [blocks[index] for index in rng.integers(0, len(blocks), len(blocks))]
        block_effects: list[float] = []
        for dataset, feature_map in selected_blocks:
            group = pivot[
                (pivot["dataset"] == dataset) & (pivot["fm_kind"] == feature_map)
            ]
            seed_effects = group["effect"].dropna().to_numpy(float)
            if not len(seed_effects):
                continue
            sampled_seed_effects = rng.choice(seed_effects, len(seed_effects), replace=True)
            block_effects.append(float(sampled_seed_effects.mean()))
        if block_effects:
            values.append(float(np.mean(block_effects)))
    if not values:
        return np.nan, np.nan, 0, observed, observed_sd
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        len(values),
        observed,
        observed_sd,
    )


def summarize_geometry(
    raw: pd.DataFrame,
    out_dir: Path,
    bootstrap: int,
    bootstrap_seed: int,
) -> list[Path]:
    metric_columns = [
        "train_train_similarity",
        "train_test_similarity",
        "test_test_similarity",
        "within_class_similarity",
        "between_class_similarity",
        "class_similarity_gap",
        "mmd2_train_test",
        "kernel_label_alignment",
        "effective_rank",
        "encoder_operation_count",
    ]
    available_metrics = [column for column in metric_columns if column in raw.columns]
    summary = (
        raw.groupby(["dataset", "fm_kind", "reps"], dropna=False)[available_metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary["error_bar_type"] = "mean ± sample SD across independent data seeds"
    summary_path = out_dir / "geometry_summary.csv"
    atomic_write_csv(summary, summary_path)

    effects: list[dict[str, Any]] = []
    for metric in available_metrics:
        low, high, valid, observed, observed_sd = hierarchical_repetition_bootstrap(
            raw,
            metric,
            bootstrap,
            stable_seed(bootstrap_seed, metric, "geometry_repetition"),
        )
        effects.append(
            {
                "metric": metric,
                "contrast": "reps 5 minus reps 1",
                "mean_difference": observed,
                "sd_across_dataset_encoder_seed_effects": observed_sd,
                "ci95_low": low,
                "ci95_high": high,
                "valid_bootstrap_replicates": valid,
                "ci_method": (
                    "paired hierarchical percentile bootstrap over dataset×encoder "
                    "blocks with data seeds nested"
                ),
                "bootstrap_unit": "dataset×encoder block; data seed nested",
                "bootstrap_replicates": bootstrap,
            }
        )
    effects_path = out_dir / "geometry_repetition_effects.csv"
    atomic_write_csv(pd.DataFrame(effects), effects_path)

    integrity_rows: list[dict[str, Any]] = []
    for keys, group in raw.groupby(["dataset", "fm_kind", "data_seed"], dropna=False):
        repetitions = set(pd.to_numeric(group["reps"], errors="coerce").dropna().astype(int))
        if not {1, 5}.issubset(repetitions):
            integrity_rows.append(
                {
                    "dataset": keys[0],
                    "fm_kind": keys[1],
                    "data_seed": keys[2],
                    "status": "missing repetition",
                    "operation_count_r1": np.nan,
                    "operation_count_r5": np.nan,
                    "operation_signature_equal": np.nan,
                    "state_signature_equal": np.nan,
                    "integrity_pass": False,
                }
            )
            continue
        row1 = group[pd.to_numeric(group["reps"]) == 1].iloc[0]
        row5 = group[pd.to_numeric(group["reps"]) == 5].iloc[0]
        operation_signature_equal = (
            str(row1.get("encoder_operation_signature", ""))
            == str(row5.get("encoder_operation_signature", ""))
        )
        state_signature_equal = str(row1.get("state_signature", "")) == str(
            row5.get("state_signature", "")
        )
        count1 = float(row1.get("encoder_operation_count", np.nan))
        count5 = float(row5.get("encoder_operation_count", np.nan))
        integrity_pass = bool(
            np.isfinite(count1)
            and np.isfinite(count5)
            and count5 > count1
            and not operation_signature_equal
            and not state_signature_equal
        )
        integrity_rows.append(
            {
                "dataset": keys[0],
                "fm_kind": keys[1],
                "data_seed": keys[2],
                "status": "pass" if integrity_pass else "fail",
                "operation_count_r1": count1,
                "operation_count_r5": count5,
                "operation_count_ratio_r5_over_r1": count5 / count1 if count1 else np.nan,
                "operation_signature_equal": operation_signature_equal,
                "state_signature_equal": state_signature_equal,
                "integrity_pass": integrity_pass,
            }
        )
    integrity_path = out_dir / "repetition_integrity.csv"
    atomic_write_csv(pd.DataFrame(integrity_rows), integrity_path)
    return [summary_path, effects_path, integrity_path]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("reviewer_targets/geometry_targets.csv"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reviewer_results/geometry_multiseed"),
    )
    parser.add_argument("--seeds", default="43,44,45")
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.jobs_per_gpu < 1 or args.cpu_threads < 1:
        parser.error("--jobs-per-gpu and --cpu-threads must be at least 1")
    if not args.targets.exists():
        parser.error(f"Target table not found: {args.targets}")
    if not (args.repo_root / "experiments" / "qurift_main.py").exists() and not args.dry_run:
        parser.error("--repo-root does not contain experiments/qurift_main.py")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_targets = pd.read_csv(args.targets)
    data_seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not data_seeds:
        parser.error("No data seeds were provided")

    expanded_rows: list[dict[str, Any]] = []
    for _, base_row in base_targets.iterrows():
        for data_seed in data_seeds:
            record = base_row.to_dict()
            base_target_id = str(base_row["target_id"])
            record["base_target_id"] = base_target_id
            record["data_seed"] = data_seed
            record["target_id"] = f"{base_target_id}_ds{data_seed}"
            record["structural_cell_id"] = str(
                base_row.get("structural_cell_id", base_target_id)
            )
            expanded_rows.append(record)
    expanded = pd.DataFrame(expanded_rows)
    expanded_path = args.out_dir / "geometry_targets_expanded.csv"
    atomic_write_csv(expanded, expanded_path)

    gpu_ids = parse_gpu_ids(args.gpus, dry_run=args.dry_run, device=args.device)
    slots: queue.Queue[int] = queue.Queue()
    for gpu_id in gpu_ids:
        for _ in range(args.jobs_per_gpu):
            slots.put(gpu_id)
    max_workers = max(1, slots.qsize())

    worker = Path(__file__).with_name("encoder_geometry_worker.py")
    status_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    status_path = args.out_dir / "run_status.csv"
    failures_path = args.out_dir / "failures.csv"

    def run_one(row: pd.Series) -> dict[str, Any]:
        target_id = str(row["target_id"])
        run_dir = args.out_dir / "runs" / target_id
        run_dir.mkdir(parents=True, exist_ok=True)
        row_json = run_dir / "target.json"
        row_json.write_text(json.dumps(row.to_dict(), indent=2, default=str), encoding="utf-8")
        output_csv = run_dir / "geometry.csv"
        log_path = run_dir / "geometry.log"

        if args.resume and output_csv.exists() and output_csv.stat().st_size > 0:
            return {
                "target_id": target_id,
                "status": "skipped",
                "gpu": "",
                "seconds": 0.0,
                "output_csv": str(output_csv),
                "log_path": str(log_path),
                "return_code": 0,
                "error": "",
            }

        command = [
            sys.executable,
            str(worker.resolve()),
            "--row-json",
            str(row_json.resolve()),
            "--repo-root",
            str(args.repo_root.resolve()),
            "--out",
            str(output_csv.resolve()),
            "--n-train",
            str(args.n_train),
            "--n-test",
            str(args.n_test),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
        ]
        if args.dry_run:
            print(" ".join(command))
            return {
                "target_id": target_id,
                "status": "dry_run",
                "gpu": "",
                "seconds": 0.0,
                "output_csv": str(output_csv),
                "log_path": str(log_path),
                "return_code": 0,
                "error": "",
            }

        gpu_id = slots.get()
        started = time.time()
        try:
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": "" if gpu_id < 0 else str(gpu_id),
                    "QURIFT_DISABLE_DEBUG_EXPORTS": "1",
                    "QURIFT_DISABLE_CIRCUIT_EXPORTS": "1",
                    "OMP_NUM_THREADS": str(args.cpu_threads),
                    "MKL_NUM_THREADS": str(args.cpu_threads),
                    "OPENBLAS_NUM_THREADS": str(args.cpu_threads),
                    "NUMEXPR_NUM_THREADS": str(args.cpu_threads),
                    "PYTHONUNBUFFERED": "1",
                    "QURIFT_JOB_ID": target_id,
                }
            )
            with log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.run(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    cwd=args.repo_root.resolve(),
                    check=False,
                )
            output_ok = output_csv.exists() and output_csv.stat().st_size > 0
            success = process.returncode == 0 and output_ok
            return {
                "target_id": target_id,
                "status": "ok" if success else "error",
                "gpu": gpu_id,
                "seconds": round(time.time() - started, 3),
                "output_csv": str(output_csv),
                "log_path": str(log_path),
                "return_code": process.returncode,
                "error": "" if success else "missing output or nonzero process return code",
            }
        except Exception as exc:
            return {
                "target_id": target_id,
                "status": "error",
                "gpu": gpu_id,
                "seconds": round(time.time() - started, 3),
                "output_csv": str(output_csv),
                "log_path": str(log_path),
                "return_code": -1,
                "error": repr(exc),
            }
        finally:
            slots.put(gpu_id)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one, row) for _, row in expanded.iterrows()]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            status_rows.append(result)
            if result["status"] == "error":
                failure_rows.append(result)
            atomic_write_csv(pd.DataFrame(status_rows), status_path)
            atomic_write_csv(pd.DataFrame(failure_rows), failures_path)
            print(
                f"[{completed}/{len(futures)}] {result['target_id']} -> "
                f"{result['status']} (gpu={result['gpu']})"
            )

    if args.dry_run:
        print(f"[DRY RUN] Expanded targets: {expanded_path.resolve()}")
        return

    result_files = sorted((args.out_dir / "runs").glob("*/geometry.csv"))
    if not result_files:
        raise SystemExit("No geometry output files were produced")
    raw = pd.concat([pd.read_csv(path) for path in result_files], ignore_index=True)
    raw_path = args.out_dir / "geometry_raw.csv"
    atomic_write_csv(raw, raw_path)
    generated = summarize_geometry(
        raw,
        args.out_dir,
        args.bootstrap,
        args.bootstrap_seed,
    )

    outputs = [raw_path, status_path, failures_path, expanded_path, *generated]
    write_analysis_metadata(
        args.out_dir / "analysis_metadata.json",
        script=Path(__file__).name,
        inputs=[str(args.targets), str(args.repo_root / "experiments/qurift_main.py")],
        outputs=[str(path) for path in outputs],
        ci_method=(
            "paired hierarchical percentile bootstrap over dataset×encoder blocks "
            "with data seeds nested"
        ),
        bootstrap_unit="dataset×encoder structural block; data seed nested",
        bootstrap_replicates=args.bootstrap,
        error_bar_type="mean ± sample SD over independent data seeds",
        notes=(
            "Geometry is computed immediately after the fixed encoder. The repetition "
            "integrity table must pass before Efficient-SU2 repetition results are used."
        ),
    )
    print(f"[OK] Geometry raw results: {raw_path.resolve()}")


if __name__ == "__main__":
    main()
