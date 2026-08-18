#!/usr/bin/env python3
"""Descriptive evidence for structure → geometry → generalization → leakage.

This analysis deliberately does not label the associations as causal mediation.
Independent split/initialization blocks are the resampling and clustering unit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t


STRUCTURE_COLUMNS = ["fm_kind", "reps", "depth"]
GEOMETRY_METRICS = [
    "class_similarity_gap", "kernel_label_alignment", "effective_rank",
    "mmd2_train_test", "within_class_similarity", "between_class_similarity",
]


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return np.nan
    return float(spearmanr(x[mask], y[mask]).statistic)


def _interval(values: list[float]) -> tuple[float, float, int]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return np.nan, np.nan, 0
    return float(np.quantile(clean, 0.025)), float(np.quantile(clean, 0.975)), int(len(clean))


def prepare_target_frame(targets: pd.DataFrame, metrics: pd.DataFrame, attacks: pd.DataFrame) -> pd.DataFrame:
    required = {"target_id", "block_id", *STRUCTURE_COLUMNS}
    if not required.issubset(targets.columns):
        raise ValueError(f"targets lack {sorted(required - set(targets.columns))}")
    loss = attacks[attacks.attack.astype(str).str.lower().eq("loss")][["target_id", "auc"]].copy()
    if loss.target_id.duplicated().any():
        raise ValueError("Loss attack table has duplicate target IDs")
    columns = ["target_id", "gap", "train_loss", "test_loss"]
    absent = set(columns) - set(metrics.columns)
    if absent:
        raise ValueError(f"metrics lack {sorted(absent)}")
    frame = targets[["target_id", "block_id", *STRUCTURE_COLUMNS]].merge(
        metrics[columns], on="target_id", how="inner"
    ).merge(loss, on="target_id", how="inner")
    frame["loss_gap"] = pd.to_numeric(frame.test_loss, errors="coerce") - pd.to_numeric(
        frame.train_loss, errors="coerce"
    )
    if frame.target_id.duplicated().any():
        raise ValueError("Mechanistic target merge is not one row per target")
    return frame


def block_bootstrap_target_correlations(
    target: pd.DataFrame, *, replicates: int, seed: int
) -> pd.DataFrame:
    config = target.groupby(STRUCTURE_COLUMNS, as_index=False)[["gap", "loss_gap", "auc"]].mean()
    blocks = np.asarray(sorted(target.block_id.astype(str).unique()))
    rng = np.random.default_rng(seed)
    boot = {"accuracy_gap": [], "loss_gap": []}
    for _ in range(replicates):
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        resample = pd.concat(
            [target[target.block_id.astype(str) == block] for block in sampled], ignore_index=True
        )
        means = resample.groupby(STRUCTURE_COLUMNS, as_index=False)[["gap", "loss_gap", "auc"]].mean()
        boot["accuracy_gap"].append(_rho(means.gap.to_numpy(float), means.auc.to_numpy(float)))
        boot["loss_gap"].append(_rho(means.loss_gap.to_numpy(float), means.auc.to_numpy(float)))
    rows = []
    for label, column in (("accuracy_gap", "gap"), ("loss_gap", "loss_gap")):
        low, high, valid = _interval(boot[label])
        rows.append(
            {
                "link": f"{label} → loss-MIA AUC",
                "spearman": _rho(config[column].to_numpy(float), config.auc.to_numpy(float)),
                "ci95_low": low,
                "ci95_high": high,
                "valid_bootstrap_replicates": valid,
                "n_structural_configurations": len(config),
                "n_independent_target_blocks": len(blocks),
                "inference_scope": "descriptive configuration-level association; target blocks resampled",
            }
        )
    return pd.DataFrame(rows)


def geometry_configuration_links(
    target: pd.DataFrame,
    geometry: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = [metric for metric in GEOMETRY_METRICS if metric in geometry.columns]
    if not available:
        raise ValueError("geometry table contains none of the prespecified geometry metrics")
    target_config = target.groupby(["fm_kind", "reps"], as_index=False)[["gap", "loss_gap", "auc"]].mean()
    geometry_config = geometry.groupby(["fm_kind", "reps"], as_index=False)[available].mean()
    joined = target_config.merge(geometry_config, on=["fm_kind", "reps"], how="inner")

    target_blocks = np.asarray(sorted(target.block_id.astype(str).unique()))
    geometry_seed_column = "data_seed" if "data_seed" in geometry else "seed"
    geometry_seeds = np.asarray(sorted(geometry[geometry_seed_column].unique()))
    rng = np.random.default_rng(seed)
    distributions = {metric: [] for metric in available}
    for _ in range(replicates):
        sampled_target = rng.choice(target_blocks, size=len(target_blocks), replace=True)
        sampled_geometry = rng.choice(geometry_seeds, size=len(geometry_seeds), replace=True)
        target_resample = pd.concat(
            [target[target.block_id.astype(str) == block] for block in sampled_target], ignore_index=True
        )
        geometry_resample = pd.concat(
            [geometry[geometry[geometry_seed_column] == value] for value in sampled_geometry], ignore_index=True
        )
        merged = target_resample.groupby(["fm_kind", "reps"], as_index=False).auc.mean().merge(
            geometry_resample.groupby(["fm_kind", "reps"], as_index=False)[available].mean(),
            on=["fm_kind", "reps"], how="inner",
        )
        for metric in available:
            distributions[metric].append(_rho(merged[metric].to_numpy(float), merged.auc.to_numpy(float)))
    rows = []
    for metric in available:
        low, high, valid = _interval(distributions[metric])
        rows.append(
            {
                "link": f"{metric} → loss-MIA AUC",
                "spearman": _rho(joined[metric].to_numpy(float), joined.auc.to_numpy(float)),
                "ci95_low": low,
                "ci95_high": high,
                "valid_bootstrap_replicates": valid,
                "n_structural_configurations": len(joined),
                "n_independent_target_blocks": len(target_blocks),
                "n_independent_geometry_seeds": len(geometry_seeds),
                "inference_scope": "descriptive six-configuration association; target blocks and geometry seeds resampled independently",
            }
        )
    return joined, pd.DataFrame(rows)


def _regression_design(
    frame: pd.DataFrame, mediators: list[str]
) -> tuple[np.ndarray, list[str], int]:
    blocks = sorted(frame.block_id.astype(str).unique())
    names = ["intercept"]
    parts = [np.ones(len(frame))]
    if frame.reps.nunique() > 1:
        names.append("reps_5_minus_1")
        parts.append((frame.reps.astype(int) == 5).astype(float).to_numpy())
    if frame.depth.nunique() > 1:
        names.append("depth_6_minus_2")
        parts.append((frame.depth.astype(int) == 6).astype(float).to_numpy())
    if frame.fm_kind.nunique() > 1:
        names.extend(["fm_z_minus_eff", "fm_zz_minus_eff"])
        parts.extend(
            [
                (frame.fm_kind.astype(str) == "z").astype(float).to_numpy(),
                (frame.fm_kind.astype(str) == "zz").astype(float).to_numpy(),
            ]
        )
    for mediator in mediators:
        values = frame[mediator].to_numpy(float)
        scale = values.std(ddof=1)
        parts.append((values - values.mean()) / scale if scale > 0 else np.zeros(len(values)))
        names.append(f"standardized_{mediator}")
    reported_terms = len(names)
    for block in blocks[1:]:
        parts.append((frame.block_id.astype(str) == block).astype(float).to_numpy())
        names.append(f"block[{block}]")
    return np.column_stack(parts), names, reported_terms


def explanatory_regressions(target: pd.DataFrame) -> pd.DataFrame:
    rows = []
    blocks = sorted(target.block_id.astype(str).unique())
    for model, mediators in (
        ("total_structural_association", []),
        ("plus_accuracy_gap", ["gap"]),
        ("plus_loss_gap", ["loss_gap"]),
        ("plus_both_gaps", ["gap", "loss_gap"]),
    ):
        work = target.dropna(subset=["auc", *mediators]).copy()
        x, names, reported_terms = _regression_design(work, mediators)
        y = work.auc.to_numpy(float)
        inverse = np.linalg.pinv(x.T @ x)
        beta = inverse @ x.T @ y
        residual = y - x @ beta
        meat = np.zeros((x.shape[1], x.shape[1]))
        block_values = work.block_id.astype(str).to_numpy()
        for block in blocks:
            mask = block_values == block
            score = x[mask].T @ residual[mask]
            meat += np.outer(score, score)
        n, k, groups = len(y), x.shape[1], len(blocks)
        correction = (groups / (groups - 1)) * ((n - 1) / max(n - k, 1)) if groups > 1 else 1.0
        covariance = correction * inverse @ meat @ inverse
        errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        critical = float(student_t.ppf(0.975, max(groups - 1, 1)))
        for index, term in enumerate(names[:reported_terms]):
            rows.append(
                {
                    "model": model, "term": term, "coefficient": beta[index],
                    "cluster_se": errors[index], "ci95_low": beta[index] - critical * errors[index],
                    "ci95_high": beta[index] + critical * errors[index], "n_targets": n,
                    "n_blocks": groups, "uncertainty": "CR1 clustered by split/init block",
                    "interpretation": "explanatory association; not causal mediation",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--attacks", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("satml_results/mechanistic_pathway"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")
    target = prepare_target_frame(pd.read_csv(args.targets), pd.read_csv(args.metrics), pd.read_csv(args.attacks))
    geometry = pd.read_csv(args.geometry)
    target_links = block_bootstrap_target_correlations(target, replicates=args.bootstrap, seed=args.bootstrap_seed)
    config, geometry_links = geometry_configuration_links(
        target, geometry, replicates=args.bootstrap, seed=args.bootstrap_seed + 1
    )
    correlations = pd.concat([target_links, geometry_links], ignore_index=True, sort=False)
    regressions = explanatory_regressions(target)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config.to_csv(args.out_dir / "pathway_configuration_summary.csv", index=False)
    correlations.to_csv(args.out_dir / "pathway_correlations.csv", index=False)
    regressions.to_csv(args.out_dir / "pathway_explanatory_regressions.csv", index=False)
    (args.out_dir / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "claim_scope": "associations consistent with an empirically supported pathway; not proof of causal mediation",
                "pathway": "structure -> post-encoder geometry -> generalization/loss behavior -> membership leakage",
                "target_resampling_unit": "independent paired split/initialization block",
                "geometry_resampling_unit": "independent data/encoder seed",
                "bootstrap_replicates": args.bootstrap,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"[OK] configurations={len(config)} pathway_links={len(correlations)} regressions={len(regressions)}")


if __name__ == "__main__":
    main()
