#!/usr/bin/env python3
"""Analyze SaTML N1 structural-noise and N2 API-query experiments."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reviewer_tools.reviewer_common import atomic_write_csv, atomic_write_json
from reviewer_tools.qurift_noisy_eval import API_AGGREGATION, POOLED_COUNT_AGGREGATION


def stable_seed(seed: int, *parts: object) -> int:
    text = "|".join(str(value) for value in (seed, *parts))
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def load_scalar_rows(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(root.rglob("condition_metrics_raw.csv"))
    if not paths:
        raise FileNotFoundError(f"No condition_metrics_raw.csv below {root}")
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    membership = raw[
        raw.metric_scope.astype(str).eq("membership")
        & raw.metric_name.astype(str).str.endswith("_auc")
    ].copy()
    membership["attack"] = membership.metric_name.astype(str).str.replace(
        r"_auc$", "", regex=True
    )
    membership = membership[~membership.attack.eq("max_probability")].copy()
    membership["auc"] = pd.to_numeric(membership.value, errors="coerce")
    membership["attack_family"] = "scalar_threshold"
    return membership, raw


def load_attack_rows(root: Path, learned_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    scalar, metric_raw = load_scalar_rows(root)
    columns = [
        "target_id", "structural_cell_id", "fm_kind", "reps", "depth",
        "model_seed", "data_seed", "mode", "queries", "shots", "total_shots",
        "simulator_seed", "aggregation", "calibration_timestamp",
        "snapshot_manifest_sha256", "attack", "attack_family", "auc",
    ]
    for column in columns:
        if column not in scalar:
            scalar[column] = np.nan
    frames = [scalar[columns]]
    if learned_path is not None and learned_path.is_file():
        learned = pd.read_csv(learned_path)
        learned["attack_family"] = "learned_cross_fitted"
        for column in columns:
            if column not in learned:
                learned[column] = np.nan
        frames.append(learned[columns])
    attacks = pd.concat(frames, ignore_index=True)
    for column in ("reps", "depth", "model_seed", "data_seed", "queries", "shots", "total_shots", "simulator_seed"):
        attacks[column] = pd.to_numeric(attacks[column], errors="coerce")
    attacks["auc"] = pd.to_numeric(attacks.auc, errors="coerce")
    attacks = attacks[np.isfinite(attacks.auc)].copy()
    return attacks, metric_raw


def validate_frozen_snapshot(attacks: pd.DataFrame) -> str:
    finite = attacks[~attacks["mode"].eq("exact")]
    hashes = sorted(
        value
        for value in finite.snapshot_manifest_sha256.dropna().astype(str).unique()
        if value and value.lower() not in {"nan", "none"}
    )
    if len(hashes) != 1:
        raise ValueError(
            "Noise analysis requires exactly one non-empty frozen snapshot hash; "
            f"observed {hashes}"
        )
    return hashes[0]


def cluster_bootstrap_mean(
    frame: pd.DataFrame,
    value: str,
    cluster: str,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    clusters = list(pd.unique(frame[cluster]))
    if len(clusters) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    grouped = {key: group[value].to_numpy(float) for key, group in frame.groupby(cluster)}
    for index in range(replicates):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        values[index] = np.mean(np.concatenate([grouped[key] for key in sampled]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def summarize_effects(
    raw: pd.DataFrame,
    *,
    value: str,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    group_keys = ["attack", "mode", "queries", "shots", "effect", "fm_kind", "scope"]
    rows = []
    declared = raw.copy()
    if "effect_scope" in declared:
        declared["scope"] = declared.effect_scope.astype(str)
    else:
        declared["scope"] = "encoder_specific"
    variants = [declared]
    pooled = declared[declared.scope.eq("encoder_specific")].copy()
    pooled["fm_kind"] = "pooled_across_encoders"
    pooled["scope"] = "pooled_across_encoders"
    variants.append(pooled)
    for frame in variants:
        for keys, group in frame.groupby(group_keys, dropna=False):
            values = group[value].to_numpy(float)
            low, high = cluster_bootstrap_mean(
                group,
                value,
                "model_seed",
                replicates=bootstrap,
                seed=stable_seed(seed, *keys, value),
            )
            rows.append(
                {
                    **dict(zip(group_keys, keys)),
                    "n_paired_units": int(len(values)),
                    "n_model_seed_blocks": int(group.model_seed.nunique()),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                    "ci95_low": low,
                    "ci95_high": high,
                    "ci_method": "percentile cluster bootstrap over trained model-seed blocks",
                    "bootstrap_replicates": int(bootstrap),
                }
            )
    return pd.DataFrame(rows)


def n1_analysis(
    attacks: pd.DataFrame,
    *,
    out_dir: Path,
    bootstrap: int,
    seed: int,
) -> None:
    selected = attacks[
        attacks.aggregation.astype(str).isin(["exact", API_AGGREGATION])
    ].copy()
    checkpoint_keys = [
        "target_id", "structural_cell_id", "fm_kind", "reps", "depth",
        "model_seed", "data_seed", "mode", "queries", "shots", "attack",
        "attack_family",
    ]
    checkpoint = selected.groupby(checkpoint_keys, dropna=False).auc.mean().reset_index()
    checkpoint = checkpoint.rename(columns={"auc": "mean_auc_over_simulator_seeds"})
    atomic_write_csv(checkpoint, out_dir / "n1_checkpoint_auc.csv")
    cell = (
        checkpoint.groupby(
            ["fm_kind", "reps", "depth", "mode", "queries", "shots", "attack"],
            dropna=False,
        ).mean_auc_over_simulator_seeds.agg(["count", "mean", "std"]).reset_index()
        .rename(
            columns={
                "count": "n_trained_checkpoints",
                "mean": "mean_auc",
                "std": "sd_across_trained_checkpoints",
            }
        )
    )
    atomic_write_csv(cell, out_dir / "n1_cell_summary.csv")

    effect_rows = []
    keys = ["attack", "mode", "queries", "shots", "fm_kind", "model_seed"]
    for group_keys, group in checkpoint.groupby(keys, dropna=False):
        values = {
            (int(row.reps), int(row.depth)): float(row.mean_auc_over_simulator_seeds)
            for row in group.itertuples()
        }
        if set(values) != {(1, 2), (1, 6), (5, 2), (5, 6)}:
            continue
        definitions = {
            "repetition_5_minus_1_at_depth2": values[(5, 2)] - values[(1, 2)],
            "repetition_5_minus_1_at_depth6": values[(5, 6)] - values[(1, 6)],
            "repetition_main_5_minus_1": (
                (values[(5, 2)] + values[(5, 6)])
                - (values[(1, 2)] + values[(1, 6)])
            ) / 2.0,
            "depth_6_minus_2_at_repetition1": values[(1, 6)] - values[(1, 2)],
            "depth_6_minus_2_at_repetition5": values[(5, 6)] - values[(5, 2)],
            "depth_main_6_minus_2": (
                (values[(1, 6)] + values[(5, 6)])
                - (values[(1, 2)] + values[(5, 2)])
            ) / 2.0,
            "repetition_by_depth_interaction": (
                values[(5, 6)] - values[(1, 6)]
                - values[(5, 2)] + values[(1, 2)]
            ),
        }
        base = dict(zip(keys, group_keys))
        for effect, value in definitions.items():
            effect_rows.append(
                {
                    **base, "effect": effect, "effect_auc": value,
                    "effect_scope": "encoder_specific",
                }
            )

    feature_keys = ["attack", "mode", "queries", "shots", "model_seed"]
    for group_keys, group in checkpoint.groupby(feature_keys, dropna=False):
        means = group.groupby("fm_kind").mean_auc_over_simulator_seeds.mean().to_dict()
        required = {"eff_su2", "z", "zz"}
        if not required.issubset(means):
            continue
        definitions = {
            "feature_z_minus_eff_su2": means["z"] - means["eff_su2"],
            "feature_zz_minus_eff_su2": means["zz"] - means["eff_su2"],
            "feature_zz_minus_z": means["zz"] - means["z"],
        }
        base = dict(zip(feature_keys, group_keys))
        for effect, value in definitions.items():
            effect_rows.append(
                {
                    **base,
                    "fm_kind": "paired_feature_maps",
                    "effect": effect,
                    "effect_auc": value,
                    "effect_scope": "paired_feature_map",
                }
            )
    effects = pd.DataFrame(effect_rows)
    if effects.empty:
        raise RuntimeError("N1 has no complete 2x2 repetition-depth blocks")
    atomic_write_csv(effects, out_dir / "n1_factorial_effects_raw.csv")
    atomic_write_csv(
        summarize_effects(
            effects,
            value="effect_auc",
            bootstrap=bootstrap,
            seed=seed,
        ),
        out_dir / "n1_factorial_effects_summary.csv",
    )

    exact = effects[effects["mode"].eq("exact")][
        ["attack", "fm_kind", "model_seed", "effect", "effect_scope", "effect_auc"]
    ].rename(columns={"effect_auc": "exact_effect_auc"})
    moderation = effects[~effects["mode"].eq("exact")].merge(
        exact,
        on=["attack", "fm_kind", "model_seed", "effect", "effect_scope"],
        how="inner",
        validate="many_to_one",
    )
    moderation["change_vs_exact"] = (
        moderation.effect_auc - moderation.exact_effect_auc
    )
    atomic_write_csv(moderation, out_dir / "n1_noise_moderation_raw.csv")
    atomic_write_csv(
        summarize_effects(
            moderation,
            value="change_vs_exact",
            bootstrap=bootstrap,
            seed=seed + 1,
        ),
        out_dir / "n1_noise_moderation_summary.csv",
    )

    ordering_rows = []
    cell_means = cell.copy()
    exact_cells = cell_means[cell_means["mode"].eq("exact")][
        ["fm_kind", "reps", "depth", "attack", "mean_auc"]
    ].rename(columns={"mean_auc": "exact_auc"})
    for keys_, group in cell_means[~cell_means["mode"].eq("exact")].groupby(
        ["attack", "mode", "queries", "shots"]
    ):
        joined = group.merge(exact_cells, on=["fm_kind", "reps", "depth", "attack"])
        statistic = spearmanr(joined.exact_auc, joined.mean_auc) if len(joined) >= 3 else None
        ordering_rows.append(
            {
                "attack": keys_[0], "mode": keys_[1], "queries": keys_[2], "shots": keys_[3],
                "n_structural_cells": int(len(joined)),
                "spearman_vs_exact": float(statistic.statistic) if statistic else np.nan,
                "pvalue_descriptive": float(statistic.pvalue) if statistic else np.nan,
                "scope": "descriptive ranking of 12 structural-cell means",
            }
        )
    atomic_write_csv(pd.DataFrame(ordering_rows), out_dir / "n1_ordering_descriptive.csv")


N2_CONTRASTS = (
    ("single_shot_512_minus_128", (1, 512), (1, 128)),
    ("single_shot_2560_minus_128", (1, 2560), (1, 128)),
    ("repeat_5x128_minus_1x128", (5, 128), (1, 128)),
    ("repeat_20x128_minus_1x128", (20, 128), (1, 128)),
    ("equal_total_5x512_minus_1x2560", (5, 512), (1, 2560)),
    ("equal_total_20x128_minus_1x2560", (20, 128), (1, 2560)),
)


def n2_analysis(
    attacks: pd.DataFrame,
    *,
    out_dir: Path,
    bootstrap: int,
    seed: int,
) -> None:
    selected = attacks[~attacks["mode"].eq("exact")].copy()
    condition_keys = [
        "target_id", "structural_cell_id", "fm_kind", "reps", "depth",
        "model_seed", "mode", "queries", "shots", "total_shots", "aggregation", "attack",
    ]
    checkpoint = selected.groupby(condition_keys, dropna=False).auc.agg(["count", "mean", "std"]).reset_index()
    checkpoint = checkpoint.rename(
        columns={"count": "n_simulator_seeds", "mean": "mean_auc", "std": "sd_simulator"}
    )
    atomic_write_csv(checkpoint, out_dir / "n2_condition_by_checkpoint.csv")

    contrast_rows = []
    replicate_keys = ["target_id", "mode", "aggregation", "attack", "simulator_seed"]
    for keys_, group in selected.groupby(replicate_keys, dropna=False):
        values = {
            (int(row.queries), int(row.shots)): float(row.auc)
            for row in group.itertuples()
        }
        for name, treatment, baseline in N2_CONTRASTS:
            if treatment in values and baseline in values:
                contrast_rows.append(
                    {
                        **dict(zip(replicate_keys, keys_)),
                        "contrast": name,
                        "treatment_queries": treatment[0],
                        "treatment_shots": treatment[1],
                        "baseline_queries": baseline[0],
                        "baseline_shots": baseline[1],
                        "auc_difference": values[treatment] - values[baseline],
                    }
                )
    raw = pd.DataFrame(contrast_rows)
    if raw.empty:
        raise RuntimeError("N2 has no complete query-condition contrasts")
    atomic_write_csv(raw, out_dir / "n2_query_contrasts_raw.csv")
    target_means = (
        raw.groupby(["target_id", "mode", "aggregation", "attack", "contrast"], dropna=False)
        .auc_difference.mean().reset_index()
    )
    rows = []
    for keys_, group in target_means.groupby(
        ["mode", "aggregation", "attack", "contrast"], dropna=False
    ):
        low, high = cluster_bootstrap_mean(
            group,
            "auc_difference",
            "target_id",
            replicates=bootstrap,
            seed=stable_seed(seed, *keys_),
        )
        values = group.auc_difference.to_numpy(float)
        rows.append(
            {
                "mode": keys_[0], "aggregation": keys_[1], "attack": keys_[2],
                "contrast": keys_[3], "n_target_checkpoints": len(values),
                "mean_auc_difference": float(values.mean()),
                "sd_across_target_checkpoints": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "ci95_low": low, "ci95_high": high,
                "ci_method": "percentile bootstrap over the predeclared target checkpoints",
                "scope": "targeted query-policy evidence; one trained model seed per structural cell",
            }
        )
    atomic_write_csv(pd.DataFrame(rows), out_dir / "n2_query_contrasts_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=["n1", "n2"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--learned", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    attacks, metric_raw = load_attack_rows(args.root, args.learned)
    snapshot_hash = validate_frozen_snapshot(attacks)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.study == "n1":
        n1_analysis(attacks, out_dir=args.out_dir, bootstrap=args.bootstrap, seed=args.bootstrap_seed)
    else:
        n2_analysis(attacks, out_dir=args.out_dir, bootstrap=args.bootstrap, seed=args.bootstrap_seed)
    atomic_write_json(
        {
            "study": args.study,
            "root": str(args.root.resolve()),
            "learned": str(args.learned.resolve()) if args.learned else None,
            "attacks": sorted(attacks.attack.astype(str).unique()),
            "aggregation_policy": {
                "primary": API_AGGREGATION,
                "diagnostic": POOLED_COUNT_AGGREGATION,
            },
            "snapshot_manifest_sha256": snapshot_hash,
            "simulator_seed_policy": (
                "Simulator seeds are averaged within a trained checkpoint before structural inference."
            ),
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
        },
        args.out_dir / "analysis_metadata.json",
    )
    print(f"[OK] {args.study.upper()} analysis -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
