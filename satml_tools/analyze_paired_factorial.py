#!/usr/bin/env python3
"""Primary paired-block analysis for the SaTML structural factorial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


CONTRASTS = (
    ("repetitions", "5 - 1", "reps", 5, 1, ("fm_kind", "depth")),
    ("depth", "6 - 2", "depth", 6, 2, ("fm_kind", "reps")),
    ("feature_map", "Z - EfficientSU2", "fm_kind", "z", "eff_su2", ("reps", "depth")),
    ("feature_map", "ZZ - EfficientSU2", "fm_kind", "zz", "eff_su2", ("reps", "depth")),
    ("feature_map", "ZZ - Z", "fm_kind", "zz", "z", ("reps", "depth")),
)


def bootstrap_mean_interval(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    chunk = 2048
    means: list[np.ndarray] = []
    for start in range(0, replicates, chunk):
        size = min(chunk, replicates - start)
        sample = rng.choice(values, size=(size, len(values)), replace=True)
        means.append(sample.mean(axis=1))
    distribution = np.concatenate(means)
    return float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))


def paired_sign_flip_pvalue(values: np.ndarray) -> float:
    """Exact two-sided randomization p-value for paired block effects."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    if len(values) > 20:
        raise ValueError("Exact sign-flip inference supports at most 20 blocks")
    masks = np.arange(1 << len(values), dtype=np.uint64)[:, None]
    bits = ((masks >> np.arange(len(values), dtype=np.uint64)) & 1).astype(float)
    signs = bits * 2.0 - 1.0
    permuted = np.abs((signs * values).mean(axis=1))
    observed = abs(float(values.mean()))
    return float(np.mean(permuted >= observed - 1e-15))


def add_holm_adjustment(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    output["holm_pvalue"] = np.nan
    for _, indices in output.groupby(["outcome", "attack"], sort=False).groups.items():
        index_list = list(indices)
        valid = [index for index in index_list if np.isfinite(output.at[index, "paired_sign_flip_pvalue"])]
        ordered = sorted(valid, key=lambda index: output.at[index, "paired_sign_flip_pvalue"])
        running = 0.0
        count = len(ordered)
        for rank, index in enumerate(ordered):
            adjusted = min(1.0, (count - rank) * float(output.at[index, "paired_sign_flip_pvalue"]))
            running = max(running, adjusted)
            output.at[index, "holm_pvalue"] = running
    family_sizes = output.groupby(["outcome", "attack"])["contrast"].transform("count")
    output["multiplicity_family"] = (
        family_sizes.astype(str) + " prespecified estimable contrasts within outcome × attack"
    )
    return output


def block_contrast(
    frame: pd.DataFrame,
    outcome: str,
    column: str,
    high: object,
    low: object,
    nuisance: Iterable[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for block_id, block in frame.groupby("block_id"):
        pivot = block.pivot_table(index=list(nuisance), columns=column, values=outcome, aggfunc="mean")
        if high not in pivot.columns or low not in pivot.columns:
            continue
        differences = (pivot[high] - pivot[low]).dropna()
        if len(differences):
            records.append(
                {
                    "block_id": block_id,
                    "block_effect": float(differences.mean()),
                    "within_block_pairs": int(len(differences)),
                }
            )
    return pd.DataFrame(records)


def _cluster_ols(frame: pd.DataFrame, outcome: str) -> list[dict[str, object]]:
    """OLS with block fixed effects and CR1 block-clustered uncertainty."""
    work = frame.dropna(subset=[outcome, "block_id", "fm_kind", "reps", "depth"]).copy()
    blocks = sorted(work["block_id"].astype(str).unique())
    columns = ["intercept"]
    x_parts = [np.ones(len(work))]
    if work["reps"].nunique() > 1:
        columns.append("reps_5_minus_1")
        x_parts.append((work["reps"].astype(int) == 5).astype(float).to_numpy())
    if work["depth"].nunique() > 1:
        columns.append("depth_6_minus_2")
        x_parts.append((work["depth"].astype(int) == 6).astype(float).to_numpy())
    if work["fm_kind"].nunique() > 1:
        columns.extend(["fm_z_minus_eff", "fm_zz_minus_eff"])
        x_parts.extend(
            [
                (work["fm_kind"].astype(str) == "z").astype(float).to_numpy(),
                (work["fm_kind"].astype(str) == "zz").astype(float).to_numpy(),
            ]
        )
    structural_term_count = len(columns)
    for block in blocks[1:]:
        columns.append(f"block[{block}]")
        x_parts.append((work["block_id"].astype(str) == block).astype(float).to_numpy())
    x = np.column_stack(x_parts)
    y = work[outcome].to_numpy(float)
    xtx_inverse = np.linalg.pinv(x.T @ x)
    beta = xtx_inverse @ x.T @ y
    residual = y - x @ beta
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    block_values = work["block_id"].astype(str).to_numpy()
    for block in blocks:
        mask = block_values == block
        score = x[mask].T @ residual[mask]
        meat += np.outer(score, score)
    n, k, groups = len(y), x.shape[1], len(blocks)
    correction = (groups / (groups - 1)) * ((n - 1) / max(n - k, 1)) if groups > 1 else 1.0
    covariance = correction * xtx_inverse @ meat @ xtx_inverse
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    critical = float(student_t.ppf(0.975, df=max(groups - 1, 1)))
    rows = []
    model_terms = [term for term in columns[1:structural_term_count]]
    for index, term in enumerate(columns[:structural_term_count]):
        rows.append(
            {
                "term": term,
                "coefficient": float(beta[index]),
                "cluster_se": float(standard_errors[index]),
                "ci95_low": float(beta[index] - critical * standard_errors[index]),
                "ci95_high": float(beta[index] + critical * standard_errors[index]),
                "n_targets": n,
                "n_blocks": groups,
                "model": "outcome ~ block fixed effects + " + " + ".join(model_terms),
                "uncertainty": "CR1 standard errors clustered by paired experimental block",
            }
        )
    return rows


def prepare_frames(
    targets: pd.DataFrame,
    metrics: pd.DataFrame,
    attack_frames: list[pd.DataFrame],
) -> list[tuple[str, str, pd.DataFrame]]:
    design_columns = [
        "target_id", "block_id", "fm_kind", "reps", "depth", "data_seed", "model_seed"
    ]
    design = targets[design_columns].copy()
    frames: list[tuple[str, str, pd.DataFrame]] = []
    metrics = metrics.copy()
    if {"train_loss", "test_loss"}.issubset(metrics.columns):
        metrics["loss_gap"] = pd.to_numeric(metrics["test_loss"], errors="coerce") - pd.to_numeric(
            metrics["train_loss"], errors="coerce"
        )
    utility_columns = [
        column for column in ("train_acc", "valid_acc", "test_acc", "gap", "train_loss", "valid_loss", "test_loss", "loss_gap")
        if column in metrics
    ]
    merged_metrics = design.merge(metrics[["target_id", *utility_columns]], on="target_id", how="inner")
    for outcome in utility_columns:
        frames.append((outcome, "target_model", merged_metrics))
    for attack_frame in attack_frames:
        attack_frame = attack_frame.copy()
        if "auc" not in attack_frame and "attack_auc" in attack_frame:
            attack_frame = attack_frame.rename(columns={"attack_auc": "auc"})
        if "attack" not in attack_frame and "auc" in attack_frame:
            attack_frame["attack"] = "learned_prediction_vector_stats"
        required = {"target_id", "attack", "auc"}
        if not required.issubset(attack_frame.columns):
            raise ValueError(f"Attack table lacks {sorted(required - set(attack_frame.columns))}")
        attack_subset = attack_frame[["target_id", "attack", "auc"]].copy()
        attack_subset["attack"] = attack_subset["attack"].astype(str)
        for attack, group in attack_subset.groupby("attack"):
            merged = design.merge(group[["target_id", "auc"]], on="target_id", how="inner")
            frames.append(("auc", str(attack), merged))
    return frames


def analyze(
    targets: pd.DataFrame,
    metrics: pd.DataFrame,
    attack_frames: list[pd.DataFrame],
    *,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    regression_rows: list[dict[str, object]] = []
    for outcome_index, (outcome, attack, frame) in enumerate(prepare_frames(targets, metrics, attack_frames)):
        if frame["target_id"].duplicated().any():
            raise ValueError(f"Duplicate target rows for {attack}/{outcome}")
        for factor, contrast, column, high, low, nuisance in CONTRASTS:
            effects = block_contrast(frame, outcome, column, high, low, nuisance)
            for record in effects.to_dict("records"):
                block_rows.append(
                    {"outcome": outcome, "attack": attack, "factor": factor, "contrast": contrast, **record}
                )
            values = effects.get("block_effect", pd.Series(dtype=float)).to_numpy(float)
            if not len(values):
                continue
            ci_low, ci_high = bootstrap_mean_interval(
                values, bootstrap, seed + outcome_index * 101 + len(summaries)
            )
            summaries.append(
                {
                    "outcome": outcome,
                    "attack": attack,
                    "factor": factor,
                    "contrast": contrast,
                    "mean_difference": float(values.mean()) if len(values) else np.nan,
                    "sd_across_blocks": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "n_independent_blocks": int(len(values)),
                    "bootstrap_replicates": int(bootstrap),
                    "inference_unit": "paired split/init block",
                    "paired_sign_flip_pvalue": paired_sign_flip_pvalue(values),
                }
            )
        for row in _cluster_ols(frame, outcome):
            regression_rows.append({"outcome": outcome, "attack": attack, **row})
    return add_holm_adjustment(pd.DataFrame(summaries)), pd.DataFrame(block_rows), pd.DataFrame(regression_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--attack-results", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=Path("satml_results/credit_factorial"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")
    targets = pd.read_csv(args.targets)
    metrics = pd.read_csv(args.metrics)
    attacks = [pd.read_csv(path) for path in args.attack_results]
    summary, blocks, regression = analyze(
        targets, metrics, attacks, bootstrap=args.bootstrap, seed=args.bootstrap_seed
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "paired_contrasts.csv", index=False)
    blocks.to_csv(args.out_dir / "paired_block_effects.csv", index=False)
    regression.to_csv(args.out_dir / "fixed_block_regression.csv", index=False)
    metadata = {
        "targets": str(args.targets),
        "metrics": str(args.metrics),
        "attack_results": [str(path) for path in args.attack_results],
        "primary_inference": "paired block-level mean differences",
        "bootstrap_unit": "split/init block",
        "bootstrap_replicates": args.bootstrap,
        "regression_role": "secondary descriptive adjustment",
    }
    (args.out_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[OK] contrasts={len(summary)} block_effects={len(blocks)}")


if __name__ == "__main__":
    main()
