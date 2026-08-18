#!/usr/bin/env python3
"""Run separate scalar threshold MIAs with correctly labelled uncertainty."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from reviewer_common import (
    CI_RECORD,
    atomic_write_csv,
    cross_fitted_threshold_metrics,
    find_attack_files,
    flatten_scalar_meta,
    membership_convention,
    normalize_membership,
    scalar_attack_scores,
    stable_seed,
    stratified_bootstrap_auc,
    stratified_bootstrap_tpr_at_fpr,
    tpr_at_resolvable_fpr,
    torch_load,
    write_analysis_metadata,
)


def infer_risk_level(role: object) -> str:
    text = str(role).lower()
    if "high" in text or "stress" in text:
        return "high"
    if "low" in text or "baseline" in text:
        return "low"
    if "hard" in text:
        return "hard"
    return ""


def hierarchical_high_low_bootstrap(
    frame: pd.DataFrame,
    metric: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, int, float]:
    subset = frame[frame["risk_level"].isin(["high", "low"])].copy()
    if subset.empty:
        return np.nan, np.nan, 0, np.nan
    block_columns = [
        column for column in ("dataset", "architecture", "fm_kind", "attack") if column in subset.columns
    ]
    subset["_block"] = subset[block_columns].astype(str).agg("|".join, axis=1)
    blocks = subset["_block"].unique()
    observed_differences = []
    for _, group in subset.groupby("_block"):
        pivot = group.pivot_table(
            index="model_seed", columns="risk_level", values=metric, aggfunc="mean"
        )
        if {"high", "low"}.issubset(pivot.columns):
            observed_differences.extend((pivot["high"] - pivot["low"]).dropna().tolist())
    observed = float(np.mean(observed_differences)) if observed_differences else np.nan

    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        selected_blocks = rng.choice(blocks, len(blocks), replace=True)
        block_effects = []
        for block in selected_blocks:
            group = subset[subset["_block"] == block]
            pivot = group.pivot_table(
                index="model_seed", columns="risk_level", values=metric, aggfunc="mean"
            )
            if {"high", "low"}.issubset(pivot.columns):
                seed_effects = (pivot["high"] - pivot["low"]).dropna().to_numpy()
                if len(seed_effects):
                    sampled = rng.choice(seed_effects, len(seed_effects), replace=True)
                    block_effects.append(float(np.mean(sampled)))
        if block_effects:
            values.append(float(np.mean(block_effects)))
    if not values:
        return np.nan, np.nan, 0, observed
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        len(values),
        observed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-data-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reviewer_results/threshold_mia"),
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument(
        "--bootstrap-chunk-size",
        type=int,
        default=2048,
        help="Bootstrap replicates processed together (larger is faster but uses more memory).",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--threshold-folds", type=int, default=5)
    parser.add_argument("--threshold-seed", type=int, default=2026)
    parser.add_argument("--fprs", default="0.05,0.10")
    parser.add_argument(
        "--attacks",
        default="all",
        help="Comma-separated scalar attacks or all. Choices include loss, entropy, confidence, margin, correctness, max_probability.",
    )
    parser.add_argument(
        "--skip-high-low-contrasts",
        action="store_true",
        help="Skip the optional role-name-based high-minus-low aggregate bootstrap.",
    )
    args = parser.parse_args()
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")
    if args.bootstrap_chunk_size <= 0:
        parser.error("--bootstrap-chunk-size must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    target_table = pd.read_csv(args.targets) if args.targets and args.targets.exists() else None
    requested_fprs = [float(value) for value in args.fprs.split(",") if value.strip()]
    requested_attacks = None if args.attacks.strip().lower() == "all" else {
        value.strip().lower() for value in args.attacks.split(",") if value.strip()
    }
    rows = []
    failures = []
    paths = find_attack_files(args.attack_data_dir)
    started = time.monotonic()
    completed_attacks = 0
    expected_attacks_per_target = (
        len(requested_attacks) if requested_attacks is not None else 6
    )
    expected_attack_jobs = len(paths) * expected_attacks_per_target
    print(
        f"[START] targets={len(paths)} attacks≈{expected_attack_jobs} "
        f"bootstrap={args.bootstrap} chunk_size={args.bootstrap_chunk_size}",
        flush=True,
    )

    for target_index, path in enumerate(paths, start=1):
        try:
            payload = torch_load(path)
            membership = normalize_membership(payload)
            meta = payload.get("meta", {}) or {}
            target_id = str(meta.get("target_id", path.parent.name))
            table_row = {}
            if target_table is not None and "target_id" in target_table.columns:
                match = target_table[target_table["target_id"].astype(str) == target_id]
                if len(match):
                    table_row = match.iloc[0].to_dict()
            scores = scalar_attack_scores(payload)
            if requested_attacks is not None:
                scores = {name: value for name, value in scores.items() if name in requested_attacks}
            print(
                f"[TARGET {target_index}/{len(paths)}] {target_id} "
                f"attacks={len(scores)}",
                flush=True,
            )
            for attack, score in scores.items():
                if len(score) != len(membership):
                    failures.append(
                        {
                            "source_file": str(path),
                            "target_id": target_id,
                            "attack": attack,
                            "error": f"length_mismatch score={len(score)} membership={len(membership)}",
                        }
                    )
                    continue
                auc = float(roc_auc_score(membership, score))
                low, high, valid = stratified_bootstrap_auc(
                    membership,
                    score,
                    args.bootstrap,
                    stable_seed(args.bootstrap_seed, target_id, attack),
                    chunk_size=args.bootstrap_chunk_size,
                )
                record = {
                    "target_id": target_id,
                    "attack": attack,
                    "auc": auc,
                    "auc_record_ci95_low": low,
                    "auc_record_ci95_high": high,
                    "valid_record_bootstrap_replicates": valid,
                    "n_member": int((membership == 1).sum()),
                    "n_nonmember": int((membership == 0).sum()),
                    "fpr_resolution": 1.0 / max(int((membership == 0).sum()), 1),
                    "membership_convention": membership_convention(payload),
                    "source_file": str(path),
                    "ci_method": CI_RECORD,
                    "bootstrap_unit": "records",
                    "bootstrap_replicates": args.bootstrap,
                }
                record.update(flatten_scalar_meta(meta, prefix=""))
                for key, value in table_row.items():
                    if key not in record or pd.isna(record.get(key)):
                        record[key] = value
                if "model_seed" not in record:
                    record["model_seed"] = record.get("seed", np.nan)
                record["risk_level"] = infer_risk_level(record.get("role", ""))
                record.update(
                    cross_fitted_threshold_metrics(
                        membership,
                        score,
                        n_splits=args.threshold_folds,
                        seed=args.threshold_seed,
                    )
                )
                for requested in requested_fprs:
                    tpr, attained = tpr_at_resolvable_fpr(membership, score, requested)
                    tag = f"{requested:.4f}".rstrip("0").rstrip(".").replace(".", "p")
                    tpr_low, tpr_high, tpr_valid = stratified_bootstrap_tpr_at_fpr(
                        membership,
                        score,
                        requested,
                        args.bootstrap,
                        stable_seed(args.bootstrap_seed, target_id, attack, "tpr", tag),
                        chunk_size=max(1, min(args.bootstrap_chunk_size, 1024)),
                    )
                    expected_false_positives = requested * record["n_nonmember"]
                    record[f"tpr_at_requested_fpr_{tag}"] = tpr
                    record[f"tpr_at_requested_fpr_{tag}_record_ci95_low"] = tpr_low
                    record[f"tpr_at_requested_fpr_{tag}_record_ci95_high"] = tpr_high
                    record[f"tpr_at_requested_fpr_{tag}_valid_bootstrap_replicates"] = tpr_valid
                    record[f"attained_fpr_{tag}"] = attained
                    record[f"expected_false_positives_{tag}"] = expected_false_positives
                    record[f"resolvable_{tag}"] = bool(expected_false_positives >= 2)
                rows.append(record)
                completed_attacks += 1
                elapsed = time.monotonic() - started
                rate = completed_attacks / elapsed if elapsed > 0 else 0.0
                remaining = max(expected_attack_jobs - completed_attacks, 0)
                eta = remaining / rate if rate > 0 else float("nan")
                print(
                    f"  [ATTACK {completed_attacks}/{expected_attack_jobs}] "
                    f"{attack} auc={auc:.4f} elapsed={elapsed:.1f}s eta≈{eta:.1f}s",
                    flush=True,
                )
        except Exception as exc:
            failures.append(
                {
                    "source_file": str(path),
                    "target_id": path.parent.name,
                    "attack": "",
                    "error": repr(exc),
                }
            )

    print("[AGGREGATE] Writing raw results and seed-level summaries...", flush=True)
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise SystemExit("No valid attack payloads were analyzed")
    raw_path = args.out_dir / "threshold_mia_raw.csv"
    failure_path = args.out_dir / "threshold_mia_failures.csv"
    atomic_write_csv(raw, raw_path)
    atomic_write_csv(pd.DataFrame(failures), failure_path)

    group_columns = [
        column
        for column in (
            "experiment",
            "dataset",
            "architecture",
            "structural_cell_id",
            "pair_id",
            "role",
            "risk_level",
            "fm_kind",
            "reps",
            "depth",
            "attack",
        )
        if column in raw.columns
    ]
    metric_columns = [
        column
        for column in (
            "auc",
            "balanced_accuracy_crossfit",
            "membership_advantage_crossfit",
            "crossfit_tpr",
            "crossfit_fpr",
        )
        if column in raw.columns
    ]
    summary = (
        raw.groupby(group_columns, dropna=False)[metric_columns]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary["error_bar_type"] = (
        "mean ± sample SD across independent target-model seeds; fixed data seed"
    )
    summary_path = args.out_dir / "threshold_mia_summary.csv"
    atomic_write_csv(summary, summary_path)

    contrast_rows = []
    if "risk_level" in raw.columns and not args.skip_high_low_contrasts:
        print("[AGGREGATE] Computing high-minus-low contrasts...", flush=True)
        for attack, attack_group in raw.groupby("attack"):
            for metric in (
                "auc",
                "balanced_accuracy_crossfit",
                "membership_advantage_crossfit",
            ):
                low, high, valid, observed = hierarchical_high_low_bootstrap(
                    attack_group,
                    metric,
                    args.bootstrap,
                    stable_seed(args.bootstrap_seed, "high_low", attack, metric),
                )
                if np.isfinite(observed):
                    contrast_rows.append(
                        {
                            "attack": attack,
                            "metric": metric,
                            "contrast": "high minus low",
                            "mean_difference": observed,
                            "ci95_low": low,
                            "ci95_high": high,
                            "valid_bootstrap_replicates": valid,
                            "ci_method": (
                                "hierarchical percentile bootstrap over dataset/architecture/encoder "
                                "blocks with target seeds nested"
                            ),
                        }
                    )
    contrast_path = args.out_dir / "threshold_mia_high_low_contrasts.csv"
    atomic_write_csv(pd.DataFrame(contrast_rows), contrast_path)

    write_analysis_metadata(
        args.out_dir / "analysis_metadata.json",
        script=Path(__file__).name,
        inputs=[str(args.attack_data_dir), str(args.targets or "")],
        outputs=[str(raw_path), str(summary_path), str(contrast_path), str(failure_path)],
        ci_method=(
            "per-target AUC: stratified percentile record bootstrap; aggregate high-low: "
            "hierarchical percentile bootstrap over structural blocks with target seeds nested"
        ),
        bootstrap_unit="records for per-target AUC; structural blocks for aggregate contrasts",
        bootstrap_replicates=args.bootstrap,
        error_bar_type="mean ± sample SD across target-model seeds in cell summaries",
        notes=(
            "TPR is requested only at empirically safer 5% and 10% FPR. The actual attained "
            "FPR, FPR resolution, and expected false-positive count are stored."
        ),
    )
    elapsed = time.monotonic() - started
    print(
        f"[OK] rows={len(raw)} targets={raw['target_id'].nunique()} "
        f"elapsed={elapsed:.1f}s -> {raw_path.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
