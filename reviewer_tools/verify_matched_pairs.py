#!/usr/bin/env python3
"""Verify whether originally matched Z–ZZ targets remain matched after retraining."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reviewer_common import CI_MATCHED_PAIR, atomic_write_csv, write_analysis_metadata


def cluster_bootstrap_pair_means(
    pair_seed: pd.DataFrame,
    value: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, int]:
    pairs = pair_seed["pair_id"].dropna().unique()
    if not len(pairs):
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(pairs, len(pairs), replace=True)
        sampled_pair_means = []
        for pair_id in chosen:
            group = pair_seed[pair_seed["pair_id"] == pair_id]
            current = pd.to_numeric(group[value], errors="coerce").dropna()
            if len(current):
                sampled_pair_means.append(float(current.mean()))
        if sampled_pair_means:
            values.append(float(np.mean(sampled_pair_means)))
    if not values:
        return np.nan, np.nan, 0
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        len(values),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reviewer_results/matched_verification"),
    )
    parser.add_argument("--gap-tolerance", type=float, default=0.02)
    parser.add_argument("--acc-tolerance", type=float, default=0.02)
    parser.add_argument("--sensitivity-gap-tolerance", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()

    data = pd.read_csv(args.metrics)
    if "model_seed" not in data.columns and "seed" in data.columns:
        data["model_seed"] = data["seed"]
    required = {"pair_id", "model_seed", "fm_kind", "train_acc", "test_acc", "gap"}
    missing = required - set(data.columns)
    if missing:
        raise SystemExit(f"Metrics file missing columns: {sorted(missing)}")

    subset = data[data["fm_kind"].astype(str).str.lower().isin(["z", "zz"])].copy()
    subset["fm_kind"] = subset["fm_kind"].astype(str).str.lower()
    duplicates = subset.duplicated(["pair_id", "model_seed", "fm_kind"], keep=False)
    if duplicates.any():
        duplicate_rows = subset.loc[duplicates, ["target_id", "pair_id", "model_seed", "fm_kind"]]
        raise SystemExit(
            "Duplicate pair/model_seed/fm_kind rows detected:\n"
            + duplicate_rows.to_string(index=False)
        )

    pivot = subset.pivot(
        index=["pair_id", "model_seed"],
        columns="fm_kind",
        values=["train_acc", "test_acc", "gap"],
    )
    rows = []
    for (pair_id, model_seed), values in pivot.iterrows():
        record = {"pair_id": pair_id, "model_seed": model_seed}
        complete = all(
            (metric, encoder) in pivot.columns
            and pd.notna(values.get((metric, encoder), np.nan))
            for metric in ("train_acc", "test_acc", "gap")
            for encoder in ("z", "zz")
        )
        record["pair_complete"] = bool(complete)
        if complete:
            for metric in ("train_acc", "test_acc", "gap"):
                z_value = float(values[(metric, "z")])
                zz_value = float(values[(metric, "zz")])
                difference = zz_value - z_value
                record[f"{metric}_z"] = z_value
                record[f"{metric}_zz"] = zz_value
                record[f"delta_{metric}_zz_minus_z"] = difference
                record[f"abs_delta_{metric}"] = abs(difference)
            record["gap_matched_primary"] = (
                record["abs_delta_gap"] <= args.gap_tolerance
            )
            record["gap_matched_sensitivity"] = (
                record["abs_delta_gap"] <= args.sensitivity_gap_tolerance
            )
            record["train_matched"] = (
                record["abs_delta_train_acc"] <= args.acc_tolerance
            )
            record["test_matched"] = (
                record["abs_delta_test_acc"] <= args.acc_tolerance
            )
            record["all_matched_primary"] = bool(
                record["gap_matched_primary"]
                and record["train_matched"]
                and record["test_matched"]
            )
        rows.append(record)
    pair_seed = pd.DataFrame(rows)
    completed = pair_seed[pair_seed["pair_complete"].eq(True)].copy()

    pair_summary_rows = []
    for pair_id, group in pair_seed.groupby("pair_id", dropna=False):
        complete_group = group[group["pair_complete"].eq(True)]
        pair_summary_rows.append(
            {
                "pair_id": pair_id,
                "expected_seed_units": len(group),
                "complete_seed_units": len(complete_group),
                "primary_matched_seed_units": int(
                    complete_group.get("all_matched_primary", pd.Series(dtype=bool)).sum()
                ),
                "primary_matched_fraction": (
                    complete_group.get("all_matched_primary", pd.Series(dtype=float)).mean()
                    if len(complete_group)
                    else np.nan
                ),
                "mean_abs_delta_train_acc": complete_group.get("abs_delta_train_acc", pd.Series(dtype=float)).mean(),
                "mean_abs_delta_test_acc": complete_group.get("abs_delta_test_acc", pd.Series(dtype=float)).mean(),
                "mean_abs_delta_gap": complete_group.get("abs_delta_gap", pd.Series(dtype=float)).mean(),
            }
        )
    pair_summary = pd.DataFrame(pair_summary_rows)

    effect_rows = []
    for metric in ("train_acc", "test_acc", "gap"):
        column = f"delta_{metric}_zz_minus_z"
        if column not in completed.columns:
            continue
        pair_means = completed.groupby("pair_id")[column].mean()
        low, high, valid = cluster_bootstrap_pair_means(
            completed,
            column,
            args.bootstrap,
            args.bootstrap_seed + {"train_acc": 1, "test_acc": 2, "gap": 3}[metric],
        )
        effect_rows.append(
            {
                "contrast": f"ZZ minus Z {metric}",
                "n_pairs": int(completed["pair_id"].nunique()),
                "n_pair_seed_units": int(len(completed)),
                "mean_pair_level_difference": float(pair_means.mean()),
                "sd_across_pair_means": float(pair_means.std(ddof=1)),
                "ci95_low": low,
                "ci95_high": high,
                "valid_bootstrap_replicates": valid,
                "ci_method": CI_MATCHED_PAIR,
            }
        )
    effects = pd.DataFrame(effect_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_seed_path = args.out_dir / "matched_pair_verification_raw.csv"
    pair_summary_path = args.out_dir / "matched_pair_verification_by_pair.csv"
    effects_path = args.out_dir / "matched_pair_verification_effects.csv"
    atomic_write_csv(pair_seed, pair_seed_path)
    atomic_write_csv(pair_summary, pair_summary_path)
    atomic_write_csv(effects, effects_path)
    write_analysis_metadata(
        args.out_dir / "analysis_metadata.json",
        script=Path(__file__).name,
        inputs=[str(args.metrics)],
        outputs=[str(pair_seed_path), str(pair_summary_path), str(effects_path)],
        ci_method=CI_MATCHED_PAIR,
        bootstrap_unit="matched-pair ID; all model seeds retained inside each pair",
        bootstrap_replicates=args.bootstrap,
        error_bar_type="pair-level sample SD plus cluster-bootstrap 95% CI",
        notes=(
            f"Primary matching: |Δgap|≤{args.gap_tolerance}; "
            f"|Δtrain|, |Δtest|≤{args.acc_tolerance}. "
            f"Sensitivity gap tolerance={args.sensitivity_gap_tolerance}."
        ),
    )
    matched = int(completed.get("all_matched_primary", pd.Series(dtype=bool)).sum())
    print(
        f"[OK] complete_pair_seed_units={len(completed)}/{len(pair_seed)}; "
        f"primary_matched={matched}/{len(completed)}"
    )


if __name__ == "__main__":
    main()
