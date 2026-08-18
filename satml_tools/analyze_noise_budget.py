#!/usr/bin/env python3
"""Combine fixed-total-shot query-budget evaluations without pseudoreplication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_noise_metrics(root: Path) -> pd.DataFrame:
    paths = sorted(root.rglob("condition_metrics_raw.csv"))
    if not paths:
        raise FileNotFoundError(f"No condition_metrics_raw.csv files below {root}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        metadata_path = path.with_name("run_metadata.json")
        calibration = "none"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            calibration = str((metadata.get("backend") or {}).get("calibration_timestamp") or "none")
        frame["calibration_profile"] = calibration
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    for column, default in (("queries", 0), ("total_shots", 0), ("shots", 0)):
        if column not in raw:
            raw[column] = default
        raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(default).astype(int)
    return raw


def analyze_noise(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    loss = raw[
        raw.metric_scope.astype(str).eq("membership")
        & raw.metric_name.astype(str).eq("loss_auc")
    ].copy()
    group_columns = ["target_id", "calibration_profile", "mode", "queries", "shots", "total_shots", "backend_name", "noise_model_loaded"]
    summary = (
        loss.groupby(group_columns, dropna=False).value
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n_simulator_replicates", "mean": "mean_auc", "std": "sd_across_simulator_seeds"})
    )
    summary["inference_scope"] = "simulator-seed variability for a fixed target checkpoint"

    finite = loss[loss["queries"] > 0].copy()
    query_rows = []
    keys = ["target_id", "calibration_profile", "mode", "simulator_seed", "total_shots"]
    for key, group in finite.groupby(keys):
        baseline_queries = int(group.queries.min())
        baseline = group[group.queries == baseline_queries].value.mean()
        for queries, query_group in group.groupby("queries"):
            if int(queries) == baseline_queries:
                continue
            query_rows.append(
                {"target_id": key[0], "calibration_profile": key[1], "mode": key[2], "simulator_seed": key[3],
                 "total_shots": key[4], "contrast": f"queries {int(queries)} - queries {baseline_queries}",
                 "auc_difference": float(query_group.value.mean() - baseline)}
            )
    query_raw = pd.DataFrame(query_rows)
    if query_raw.empty:
        query_summary = query_raw
    else:
        query_summary = (
            query_raw.groupby(["target_id", "calibration_profile", "mode", "total_shots", "contrast"]).auc_difference
            .agg(["count", "mean", "std"]).reset_index()
            .rename(columns={"count": "n_simulator_seeds", "mean": "mean_auc_difference", "std": "sd_across_simulator_seeds"})
        )
        query_summary["inference_scope"] = "paired simulator seeds within a fixed checkpoint; not target-model replication"

    exact = summary[summary["mode"] == "exact"][["target_id", "calibration_profile", "mean_auc"]].rename(columns={"mean_auc": "exact_auc"})
    ordering_rows = []
    finite_summary = summary[summary["mode"] != "exact"]
    for keys_, group in finite_summary.groupby(["calibration_profile", "mode", "queries", "shots", "total_shots"]):
        joined = group.merge(exact, on=["target_id", "calibration_profile"], how="inner")
        if len(joined) < 3:
            rho, pvalue = np.nan, np.nan
        else:
            statistic = spearmanr(joined.exact_auc, joined.mean_auc)
            rho, pvalue = float(statistic.statistic), float(statistic.pvalue)
        ordering_rows.append(
            {"calibration_profile": keys_[0], "mode": keys_[1], "queries": keys_[2], "shots": keys_[3], "total_shots": keys_[4],
             "n_target_checkpoints": len(joined), "spearman_vs_exact": rho, "pvalue_descriptive": pvalue,
             "interpretation": "descriptive ordering across independent target checkpoints"}
        )
    return summary, query_summary, pd.DataFrame(ordering_rows)


def analyze_utility(raw: pd.DataFrame) -> pd.DataFrame:
    names = {"target_accuracy", "target_loss", "generalization_gap_accuracy"}
    utility = raw[raw.metric_name.astype(str).isin(names)].copy()
    group_columns = [
        "target_id", "calibration_profile", "mode", "queries", "shots", "total_shots",
        "backend_name", "noise_model_loaded", "metric_scope", "metric_name",
    ]
    summary = (
        utility.groupby(group_columns, dropna=False).value
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(
            columns={"count": "n_simulator_replicates", "mean": "mean_value",
                     "std": "sd_across_simulator_seeds"}
        )
    )
    summary["inference_scope"] = "prediction utility on the fixed sampled records for one target checkpoint"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("satml_results/noise_budget/combined"))
    args = parser.parse_args()
    raw = load_noise_metrics(args.root)
    summary, query, ordering = analyze_noise(raw)
    utility = analyze_utility(raw)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "noise_budget_auc_summary.csv", index=False)
    query.to_csv(args.out_dir / "equal_total_budget_query_contrasts.csv", index=False)
    ordering.to_csv(args.out_dir / "structural_ordering_vs_exact.csv", index=False)
    utility.to_csv(args.out_dir / "noise_budget_utility_summary.csv", index=False)
    print(
        f"[OK] conditions={len(summary)} query_contrasts={len(query)} "
        f"ordering={len(ordering)} utility={len(utility)}"
    )


if __name__ == "__main__":
    main()
