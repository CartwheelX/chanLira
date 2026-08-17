#!/usr/bin/env python3
"""Descriptive cluster-bootstrap analysis of structural variables and MIA AUC.

Two predeclared specifications are fitted:

M1: AUC ~ generalization gap + feature map + repetitions + width + depth
M2: AUC ~ train accuracy + test accuracy + feature map + repetitions + width
          + depth + log gate count + log trainable parameter count

M2 excludes generalization gap because gap = train accuracy - test accuracy.
The models are standardized ridge regressions for numerical stability and are
not interpreted causally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from reviewer_common import (
    atomic_write_csv,
    atomic_write_json,
    stable_seed,
    write_analysis_metadata,
)


def coalesce(frame: pd.DataFrame, destination: str, candidates: list[str]) -> None:
    present = [column for column in candidates if column in frame.columns]
    if not present:
        return
    value = frame[present[0]]
    for column in present[1:]:
        value = value.where(value.notna(), frame[column])
    frame[destination] = value


def prepare_input(metrics: pd.DataFrame, mia: pd.DataFrame, resources: pd.DataFrame | None) -> pd.DataFrame:
    if "attack" in mia.columns:
        mia = mia[mia["attack"].astype(str).str.lower().eq("loss")].copy()
    mia_columns = [column for column in ("target_id", "auc") if column in mia.columns]
    combined = metrics.merge(mia[mia_columns], on="target_id", how="inner", suffixes=("_metric", "_mia"))
    if resources is not None:
        resource_columns = [
            column
            for column in (
                "target_id",
                "trainable_parameters_total",
                "quantum_gate_count_total",
                "exact_resource_counts_available",
            )
            if column in resources.columns
        ]
        combined = combined.merge(resources[resource_columns], on="target_id", how="left")

    coalesce(combined, "fm_kind", ["fm_kind", "meta_fm_kind", "fm_kind_metric"])
    coalesce(combined, "reps", ["reps", "meta_reps", "reps_metric"])
    coalesce(combined, "width", ["n_wires", "width", "meta_n_wires", "n_wires_metric"])
    coalesce(combined, "depth", ["depth", "meta_depth", "depth_metric"])
    coalesce(combined, "train_accuracy", ["train_acc", "target_train_acc"])
    coalesce(combined, "test_accuracy", ["test_acc", "target_test_acc"])
    coalesce(combined, "generalization_gap", ["gap", "gap_acc", "target_gap_acc"])
    coalesce(combined, "model_seed", ["model_seed", "seed", "meta_model_seed"])
    coalesce(
        combined,
        "structural_cell_id",
        ["structural_cell_id", "role", "pair_id"],
    )
    if "structural_cell_id" not in combined.columns:
        combined["structural_cell_id"] = (
            combined[["fm_kind", "reps", "width", "depth"]]
            .astype(str)
            .agg("|".join, axis=1)
        )

    numeric = [
        "reps",
        "width",
        "depth",
        "train_accuracy",
        "test_accuracy",
        "generalization_gap",
        "trainable_parameters_total",
        "quantum_gate_count_total",
        "auc",
        "model_seed",
    ]
    for column in numeric:
        if column in combined.columns:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")
    if "trainable_parameters_total" in combined.columns:
        combined["log_parameter_count"] = np.log1p(combined["trainable_parameters_total"])
    if "quantum_gate_count_total" in combined.columns:
        combined["log_gate_count"] = np.log1p(combined["quantum_gate_count_total"])
    combined["fm_kind"] = combined["fm_kind"].astype(str).str.lower()
    return combined


def usable_features(frame: pd.DataFrame, requested: list[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for feature in requested:
        if feature not in frame.columns:
            dropped.append(f"{feature}: missing")
            continue
        values = frame[feature].dropna()
        if len(values) == 0:
            dropped.append(f"{feature}: all missing")
            continue
        if values.nunique(dropna=True) < 2:
            dropped.append(f"{feature}: zero variance")
            continue
        kept.append(feature)
    return kept, dropped


def fit_model(frame: pd.DataFrame, features: list[str], alpha: float) -> dict[str, Any]:
    categorical = [feature for feature in features if feature == "fm_kind" or frame[feature].dtype == object]
    numerical = [feature for feature in features if feature not in categorical]
    transformers = []
    if numerical:
        transformers.append(("numeric", StandardScaler(), numerical))
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("No usable predictors")
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regression", Ridge(alpha=alpha)),
        ]
    )
    target = frame["auc"].to_numpy(float)
    pipeline.fit(frame[features], target)
    prediction = pipeline.predict(frame[features])
    names = [str(name) for name in pipeline.named_steps["preprocessor"].get_feature_names_out()]
    coefficients = pipeline.named_steps["regression"].coef_.astype(float)
    return {
        "pipeline": pipeline,
        "coefficient_map": dict(zip(names, coefficients)),
        "intercept": float(pipeline.named_steps["regression"].intercept_),
        "r2": float(r2_score(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "n_rows": int(len(frame)),
    }


def hierarchical_sample(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    cells = frame["structural_cell_id"].dropna().unique()
    if not len(cells):
        raise ValueError("No structural cells")
    selected = rng.choice(cells, len(cells), replace=True)
    parts: list[pd.DataFrame] = []
    for replicate_index, cell in enumerate(selected):
        group = frame[frame["structural_cell_id"] == cell]
        if group.empty:
            continue
        seed_column = "model_seed" if "model_seed" in group.columns else None
        if seed_column is not None and group[seed_column].notna().any():
            seeds = group[seed_column].dropna().unique()
            selected_seeds = rng.choice(seeds, len(seeds), replace=True)
            sampled = pd.concat(
                [group[group[seed_column] == seed].sample(n=1, replace=True, random_state=int(rng.integers(1, 2**31 - 1))) for seed in selected_seeds],
                ignore_index=True,
            )
        else:
            sampled = group.sample(
                n=len(group),
                replace=True,
                random_state=int(rng.integers(1, 2**31 - 1)),
            )
        sampled = sampled.copy()
        sampled["_bootstrap_cell_instance"] = replicate_index
        parts.append(sampled)
    if not parts:
        raise ValueError("Hierarchical bootstrap produced no rows")
    return pd.concat(parts, ignore_index=True)


def bootstrap_regression(
    frame: pd.DataFrame,
    features: list[str],
    point_fit: dict[str, Any],
    bootstrap: int,
    seed: int,
    alpha: float,
) -> tuple[dict[str, list[float]], list[float], list[float], int, list[str]]:
    rng = np.random.default_rng(seed)
    coefficient_samples = {name: [] for name in point_fit["coefficient_map"]}
    r2_samples: list[float] = []
    rmse_samples: list[float] = []
    failures: list[str] = []
    valid = 0
    for _ in range(bootstrap):
        try:
            sample = hierarchical_sample(frame, rng)
            fit = fit_model(sample, features, alpha)
            valid += 1
            r2_samples.append(fit["r2"])
            rmse_samples.append(fit["rmse"])
            for term in coefficient_samples:
                coefficient_samples[term].append(
                    float(fit["coefficient_map"].get(term, np.nan))
                )
        except Exception as exc:
            if len(failures) < 20:
                failures.append(repr(exc))
    return coefficient_samples, r2_samples, rmse_samples, valid, failures


def cluster_bootstrap_correlation(
    frame: pd.DataFrame,
    method: str,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    usable = frame.dropna(subset=["generalization_gap", "auc"]).copy()
    if method == "pearson":
        observed = float(pearsonr(usable["generalization_gap"], usable["auc"]).statistic)
    else:
        observed = float(spearmanr(usable["generalization_gap"], usable["auc"]).statistic)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(bootstrap):
        try:
            sample = hierarchical_sample(usable, rng)
            if method == "pearson":
                value = float(pearsonr(sample["generalization_gap"], sample["auc"]).statistic)
            else:
                value = float(spearmanr(sample["generalization_gap"], sample["auc"]).statistic)
            if np.isfinite(value):
                samples.append(value)
        except Exception:
            pass
    return {
        "method": method,
        "correlation": observed,
        "ci95_low": float(np.quantile(samples, 0.025)) if samples else np.nan,
        "ci95_high": float(np.quantile(samples, 0.975)) if samples else np.nan,
        "valid_bootstrap_replicates": len(samples),
        "ci_method": "hierarchical cluster percentile bootstrap over structural cells with target seeds nested",
        "bootstrap_unit": "structural cell; target-model seed nested",
        "bootstrap_replicates": bootstrap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--mia", type=Path, required=True)
    parser.add_argument("--resources", type=Path, default=None)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("reviewer_results/mia_regression")
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(args.metrics)
    mia = pd.read_csv(args.mia)
    resources = pd.read_csv(args.resources) if args.resources is not None else None
    combined = prepare_input(metrics, mia, resources)
    input_path = args.out_dir / "regression_input.csv"
    atomic_write_csv(combined, input_path)

    requested_models = {
        "M1_gap": ["generalization_gap", "fm_kind", "reps", "width", "depth"],
        "M2_behavior_resources": [
            "train_accuracy",
            "test_accuracy",
            "fm_kind",
            "reps",
            "width",
            "depth",
            "log_gate_count",
            "log_parameter_count",
        ],
    }

    coefficient_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    specifications: dict[str, Any] = {}

    for model_name, requested_features in requested_models.items():
        required = ["auc", "structural_cell_id"]
        initial = combined.dropna(subset=[column for column in required if column in combined.columns]).copy()
        kept_features, dropped_features = usable_features(initial, requested_features)
        model_frame = initial.dropna(subset=["auc", "structural_cell_id", *kept_features]).copy()
        if not kept_features or len(model_frame) < max(8, len(kept_features) + 2):
            diagnostic_rows.append(
                {
                    "model": model_name,
                    "status": "not fitted",
                    "n_rows": len(model_frame),
                    "n_structural_cells": model_frame["structural_cell_id"].nunique() if len(model_frame) else 0,
                    "kept_features": json.dumps(kept_features),
                    "dropped_features": json.dumps(dropped_features),
                    "error": "insufficient complete rows or no varying predictors",
                }
            )
            specifications[model_name] = {
                "requested_features": requested_features,
                "kept_features": kept_features,
                "dropped_features": dropped_features,
                "status": "not fitted",
            }
            continue

        point_fit = fit_model(model_frame, kept_features, args.ridge_alpha)
        coefficient_samples, r2_samples, rmse_samples, valid, failures = bootstrap_regression(
            model_frame,
            kept_features,
            point_fit,
            args.bootstrap,
            stable_seed(args.bootstrap_seed, model_name),
            args.ridge_alpha,
        )
        for term, coefficient in point_fit["coefficient_map"].items():
            values = np.asarray(coefficient_samples.get(term, []), dtype=float)
            values = values[np.isfinite(values)]
            coefficient_rows.append(
                {
                    "model": model_name,
                    "term": term,
                    "coefficient": coefficient,
                    "ci95_low": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                    "ci95_high": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                    "valid_term_bootstrap_replicates": int(len(values)),
                    "coefficient_scale": "numeric predictors standardized; categorical terms are one-hot contrasts",
                    "ci_method": "hierarchical cluster percentile bootstrap over structural cells with target seeds nested",
                }
            )
        coefficient_rows.append(
            {
                "model": model_name,
                "term": "intercept",
                "coefficient": point_fit["intercept"],
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "valid_term_bootstrap_replicates": 0,
                "coefficient_scale": "intercept at standardized numeric means and categorical reference levels",
                "ci_method": "not bootstrapped",
            }
        )
        fit_rows.append(
            {
                "model": model_name,
                "n_rows": point_fit["n_rows"],
                "n_structural_cells": int(model_frame["structural_cell_id"].nunique()),
                "r2": point_fit["r2"],
                "r2_ci95_low": float(np.quantile(r2_samples, 0.025)) if r2_samples else np.nan,
                "r2_ci95_high": float(np.quantile(r2_samples, 0.975)) if r2_samples else np.nan,
                "rmse": point_fit["rmse"],
                "rmse_ci95_low": float(np.quantile(rmse_samples, 0.025)) if rmse_samples else np.nan,
                "rmse_ci95_high": float(np.quantile(rmse_samples, 0.975)) if rmse_samples else np.nan,
                "valid_bootstrap_replicates": valid,
                "ridge_alpha": args.ridge_alpha,
                "interpretation": "descriptive association; not causal",
            }
        )
        diagnostic_rows.append(
            {
                "model": model_name,
                "status": "fitted",
                "n_rows": len(model_frame),
                "n_structural_cells": model_frame["structural_cell_id"].nunique(),
                "kept_features": json.dumps(kept_features),
                "dropped_features": json.dumps(dropped_features),
                "valid_bootstrap_replicates": valid,
                "invalid_bootstrap_replicates": args.bootstrap - valid,
                "example_failures": json.dumps(failures),
                "error": "",
            }
        )
        specifications[model_name] = {
            "outcome": "loss-threshold MIA ROC AUC",
            "requested_features": requested_features,
            "kept_features": kept_features,
            "dropped_features": dropped_features,
            "ridge_alpha": args.ridge_alpha,
            "numeric_scaling": "z-score",
            "categorical_encoding": "one-hot with first category as reference",
            "bootstrap": "structural-cell cluster bootstrap with target seeds nested",
        }

    correlation_rows = [
        cluster_bootstrap_correlation(
            combined,
            method,
            args.bootstrap,
            stable_seed(args.bootstrap_seed, "gap_auc", method),
        )
        for method in ("pearson", "spearman")
    ]

    coefficient_path = args.out_dir / "mia_regression_coefficients.csv"
    fit_path = args.out_dir / "mia_regression_model_fit.csv"
    diagnostics_path = args.out_dir / "bootstrap_diagnostics.csv"
    correlation_path = args.out_dir / "mia_gap_correlations.csv"
    specification_path = args.out_dir / "model_specifications.json"
    atomic_write_csv(pd.DataFrame(coefficient_rows), coefficient_path)
    atomic_write_csv(pd.DataFrame(fit_rows), fit_path)
    atomic_write_csv(pd.DataFrame(diagnostic_rows), diagnostics_path)
    atomic_write_csv(pd.DataFrame(correlation_rows), correlation_path)
    atomic_write_json(specifications, specification_path)

    write_analysis_metadata(
        args.out_dir / "analysis_metadata.json",
        script=Path(__file__).name,
        inputs=[str(args.metrics), str(args.mia), str(args.resources or "")],
        outputs=[
            str(input_path),
            str(coefficient_path),
            str(fit_path),
            str(diagnostics_path),
            str(correlation_path),
            str(specification_path),
        ],
        ci_method=(
            "hierarchical cluster percentile bootstrap over structural cells with "
            "target-model seeds nested"
        ),
        bootstrap_unit="structural cell; target-model seed nested",
        bootstrap_replicates=args.bootstrap,
        error_bar_type="asymmetric 95% percentile bootstrap confidence intervals",
        notes=(
            "M1 includes generalization gap and excludes train/test accuracy. M2 includes "
            "train and test accuracy and excludes gap to avoid exact collinearity. Zero-variance "
            "or unavailable predictors are recorded in bootstrap_diagnostics.csv. Models are "
            "descriptive standardized ridge regressions and do not establish causality."
        ),
    )
    print(f"[OK] Regression coefficients: {coefficient_path.resolve()}")


if __name__ == "__main__":
    main()
