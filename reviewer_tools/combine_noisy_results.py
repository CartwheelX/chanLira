#!/usr/bin/env python3
"""Combine noisy QuRiFT predictions into utility/MIA results with explicit CIs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


ATTACK_COLUMNS = {
    "loss": "loss",
    "entropy": "entropy",
    "confidence": "confidence",
    "margin": "margin",
    "correctness": "correctness",
    "max_probability": "max_probability",
}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def stable_seed(*parts: object, base: int = 2026) -> int:
    text = "|".join(str(part) for part in parts)
    return int(base) + int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def parse_attacks(text: str) -> List[str]:
    attacks = [value.strip() for value in text.split(",") if value.strip()]
    unknown = sorted(set(attacks) - set(ATTACK_COLUMNS))
    if unknown:
        raise ValueError(f"Unsupported attacks: {unknown}")
    return attacks


def score_for_attack(group: pd.DataFrame, attack: str) -> np.ndarray:
    if attack == "loss":
        return -pd.to_numeric(group["loss"], errors="coerce").to_numpy(dtype=float)
    if attack == "entropy":
        return -pd.to_numeric(group["entropy"], errors="coerce").to_numpy(dtype=float)
    if attack in {"confidence", "margin", "correctness"}:
        return pd.to_numeric(group[attack], errors="coerce").to_numpy(dtype=float)
    if attack == "max_probability":
        probability_columns = sorted(
            [column for column in group.columns if column.startswith("p_")],
            key=lambda value: int(value.split("_")[1]),
        )
        if not probability_columns:
            return pd.to_numeric(group["confidence"], errors="coerce").to_numpy(dtype=float)
        return group[probability_columns].apply(pd.to_numeric, errors="coerce").max(axis=1).to_numpy(dtype=float)
    raise KeyError(attack)


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) != 2 or not np.all(np.isfinite(score)):
        return float("nan")
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return float("nan")


def choose_threshold(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) != 2:
        return float("inf")
    fpr, tpr, thresholds = roc_curve(y, score)
    objective = tpr - fpr
    index = int(np.nanargmax(objective))
    return float(thresholds[index])


def cross_fitted_metrics(y: np.ndarray, score: np.ndarray, seed: int, folds: int) -> Dict[str, float]:
    counts = np.bincount(y.astype(int), minlength=2)
    n_splits = min(int(folds), int(counts.min()))
    if n_splits < 2:
        return {
            "balanced_accuracy": float("nan"),
            "membership_advantage": float("nan"),
            "crossfit_tpr": float("nan"),
            "crossfit_fpr": float("nan"),
            "crossfit_folds": 0,
        }
    predictions = np.zeros(len(y), dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    for train_index, test_index in splitter.split(score.reshape(-1, 1), y):
        threshold = choose_threshold(y[train_index], score[train_index])
        predictions[test_index] = (score[test_index] >= threshold).astype(int)
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "membership_advantage": float(tpr - fpr),
        "crossfit_tpr": float(tpr),
        "crossfit_fpr": float(fpr),
        "crossfit_folds": int(n_splits),
    }


def fixed_fpr_metric(y: np.ndarray, score: np.ndarray, requested: float) -> Tuple[float, float]:
    if len(np.unique(y)) != 2:
        return float("nan"), float("nan")
    fpr, tpr, _ = roc_curve(y, score)
    valid = np.where(fpr <= float(requested) + 1e-12)[0]
    if len(valid) == 0:
        return 0.0, 0.0
    best_local = valid[int(np.argmax(tpr[valid]))]
    return float(tpr[best_local]), float(fpr[best_local])


def stratified_auc_bootstrap(
    y: np.ndarray,
    score: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> Tuple[float, float, int]:
    member = np.where(y == 1)[0]
    nonmember = np.where(y == 0)[0]
    if len(member) == 0 or len(nonmember) == 0 or replicates <= 0:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(replicates)):
        indices = np.concatenate(
            [
                rng.choice(member, size=len(member), replace=True),
                rng.choice(nonmember, size=len(nonmember), replace=True),
            ]
        )
        value = safe_auc(y[indices], score[indices])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan"), 0
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        int(len(values)),
    )


def mean_bootstrap(values: Sequence[float], replicates: int, seed: int) -> Tuple[float, float, int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2 or replicates <= 0:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(int(seed))
    output = np.empty(int(replicates), dtype=float)
    for index in range(int(replicates)):
        output[index] = float(np.mean(rng.choice(array, size=len(array), replace=True)))
    return float(np.quantile(output, 0.025)), float(np.quantile(output, 0.975)), len(output)


def target_utility(group: pd.DataFrame) -> Dict[str, float]:
    train = group[group["membership"].astype(int) == 1]
    test = group[group["membership"].astype(int) == 0]
    train_acc = pd.to_numeric(train["correctness"], errors="coerce").mean()
    test_acc = pd.to_numeric(test["correctness"], errors="coerce").mean()
    train_loss = pd.to_numeric(train["loss"], errors="coerce").mean()
    test_loss = pd.to_numeric(test["loss"], errors="coerce").mean()
    return {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "accuracy_gap": float(train_acc - test_acc),
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),
        "loss_gap": float(test_loss - train_loss),
        "n_member": int(len(train)),
        "n_nonmember": int(len(test)),
        "fpr_resolution": float(1.0 / len(test)) if len(test) else float("nan"),
    }


def summarize_over_replicates(
    frame: pd.DataFrame,
    *,
    group_columns: List[str],
    metrics: List[str],
    bootstrap: int,
    bootstrap_unit: str,
    progress_label: str = "summary",
    progress_every: int = 5,
) -> pd.DataFrame:
    rows = []
    grouped = frame.groupby(group_columns, dropna=False)
    total_groups = int(grouped.ngroups)
    started = time.monotonic()
    print(
        f"[PROGRESS] {progress_label}: 0/{total_groups} groups",
        flush=True,
    )
    for group_index, (keys, group) in enumerate(grouped, start=1):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, key_values))
        row["n_replicates"] = int(len(group))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan
            low, high, valid = mean_bootstrap(
                values,
                bootstrap,
                stable_seed(*key_values, metric, bootstrap_unit),
            )
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
            row[f"{metric}_valid_bootstrap"] = valid
        row["error_bar_type"] = "sample_sd"
        row["ci_method"] = (
            "95% percentile bootstrap over " + bootstrap_unit if len(group) >= 2 else "none_single_replicate"
        )
        row["bootstrap_unit"] = bootstrap_unit if len(group) >= 2 else "none"
        row["bootstrap_replicates"] = int(bootstrap if len(group) >= 2 else 0)
        rows.append(row)
        if (
            group_index == 1
            or group_index % max(1, progress_every) == 0
            or group_index == total_groups
        ):
            elapsed = time.monotonic() - started
            rate = group_index / elapsed if elapsed > 0 else 0.0
            remaining = (
                (total_groups - group_index) / rate if rate > 0 else float("nan")
            )
            print(
                f"[PROGRESS] {progress_label}: {group_index}/{total_groups} groups "
                f"({100.0 * group_index / max(1, total_groups):.1f}%), "
                f"elapsed={elapsed / 60:.1f} min, ETA={remaining / 60:.1f} min",
                flush=True,
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reviewer_results/noisy_sanity/combined"))
    parser.add_argument("--attacks", default="loss,confidence,correctness")
    parser.add_argument("--fpr-points", default="0.05,0.10")
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Print progress after this many conditions or summary groups.",
    )
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be at least 1")

    targets = pd.read_csv(args.targets)
    attacks = parse_attacks(args.attacks)
    fpr_points = [float(value.strip()) for value in args.fpr_points.split(",") if value.strip()]
    if any(value <= 0 or value >= 1 for value in fpr_points):
        raise ValueError("FPR points must lie strictly between zero and one")

    metadata_columns = [
        column for column in [
            "target_id", "experiment", "dataset", "architecture", "fm_kind", "reps",
            "depth", "n_wires", "model_seed", "data_seed", "structural_cell_id",
            "risk_role", "comparison_group", "selection_order", "original_acc_train",
            "original_acc_test", "original_gap", "selection_basis",
        ] if column in targets.columns
    ]
    target_metadata = targets[metadata_columns].drop_duplicates("target_id")
    metadata_lookup = target_metadata.set_index("target_id").to_dict(orient="index")

    result_rows: List[Dict[str, object]] = []
    completeness_rows: List[Dict[str, object]] = []
    consistency_rows: List[Dict[str, object]] = []
    target_ids = targets["target_id"].astype(str).tolist()
    total_conditions_hint = 0
    for target_id in target_ids:
        status_path = args.raw_root / target_id / "condition_status.csv"
        if status_path.exists():
            try:
                total_conditions_hint += len(pd.read_csv(status_path))
            except Exception:
                pass
    analysis_started = time.monotonic()
    completed_conditions = 0
    print(
        f"[PROGRESS] Combining {len(target_ids)} targets, "
        f"{total_conditions_hint or 'unknown'} conditions, "
        f"{len(attacks)} attacks, {args.bootstrap} record bootstraps per attack.",
        flush=True,
    )

    for target_index, target_id in enumerate(target_ids, start=1):
        print(
            f"[PROGRESS] Target {target_index}/{len(target_ids)}: {target_id}",
            flush=True,
        )
        target_dir = args.raw_root / target_id
        predictions_path = target_dir / "per_sample_predictions.csv"
        manifest_path = target_dir / "sample_manifest.csv"
        status_path = target_dir / "condition_status.csv"
        metadata = metadata_lookup[target_id]
        if not predictions_path.exists():
            completeness_rows.append({
                **metadata,
                "target_id": target_id,
                "status": "missing_predictions",
                "path": str(predictions_path),
            })
            continue

        predictions = pd.read_csv(predictions_path)
        manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
        status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
        if predictions.empty:
            completeness_rows.append({**metadata, "target_id": target_id, "status": "empty_predictions"})
            continue

        if "max_probability" not in predictions.columns:
            pcols = [column for column in predictions.columns if column.startswith("p_")]
            predictions["max_probability"] = predictions[pcols].max(axis=1) if pcols else predictions["confidence"]

        condition_columns = [
            "target_id", "mode", "shots", "simulator_seed", "transpiler_seed",
            "backend_name", "noise_model_loaded", "quantum_execution_scope",
        ]
        condition_columns = [column for column in condition_columns if column in predictions.columns]
        expected_samples = set(manifest["sample_id"].astype(str)) if not manifest.empty else set()

        for keys, group in predictions.groupby(condition_columns, dropna=False):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            base = dict(zip(condition_columns, key_values))
            observed_samples = set(group["sample_id"].astype(str))
            consistency_rows.append({
                **metadata,
                **base,
                "n_rows": int(len(group)),
                "n_unique_samples": int(group["sample_id"].nunique()),
                "matches_manifest": bool(not expected_samples or observed_samples == expected_samples),
                "missing_sample_count": int(len(expected_samples - observed_samples)) if expected_samples else 0,
                "extra_sample_count": int(len(observed_samples - expected_samples)) if expected_samples else 0,
            })
            utility = target_utility(group)
            y = group["membership"].astype(int).to_numpy()
            for attack in attacks:
                score = score_for_attack(group, attack)
                auc = safe_auc(y, score)
                crossfit = cross_fitted_metrics(
                    y,
                    score,
                    stable_seed(target_id, base.get("mode"), base.get("shots"), base.get("simulator_seed"), attack),
                    args.crossfit_folds,
                )
                low, high, valid = stratified_auc_bootstrap(
                    y,
                    score,
                    replicates=args.bootstrap,
                    seed=stable_seed(target_id, base.get("mode"), base.get("shots"), base.get("simulator_seed"), attack, base=args.bootstrap_seed),
                )
                row: Dict[str, object] = {
                    **metadata,
                    **base,
                    **utility,
                    "attack": attack,
                    "roc_auc": auc,
                    "auc_record_ci95_low": low,
                    "auc_record_ci95_high": high,
                    "auc_record_bootstrap_valid": valid,
                    "auc_record_ci_method": "95% stratified percentile bootstrap over member/nonmember records",
                    "auc_record_bootstrap_unit": "records_within_fixed_target_condition",
                    "auc_record_bootstrap_replicates": int(args.bootstrap),
                    **crossfit,
                }
                for requested in fpr_points:
                    tpr, attained = fixed_fpr_metric(y, score, requested)
                    suffix = str(int(round(requested * 100)))
                    row[f"tpr_at_{suffix}pct_fpr"] = tpr
                    row[f"attained_fpr_for_{suffix}pct"] = attained
                result_rows.append(row)
            completed_conditions += 1
            if (
                completed_conditions == 1
                or completed_conditions % args.progress_every == 0
                or (
                    total_conditions_hint
                    and completed_conditions == total_conditions_hint
                )
            ):
                elapsed = time.monotonic() - analysis_started
                rate = completed_conditions / elapsed if elapsed > 0 else 0.0
                if total_conditions_hint and rate > 0:
                    remaining_text = (
                        f"{(total_conditions_hint - completed_conditions) / rate / 60:.1f} min"
                    )
                    percentage = (
                        f"{100.0 * completed_conditions / total_conditions_hint:.1f}%"
                    )
                else:
                    remaining_text = "unknown"
                    percentage = "unknown"
                print(
                    f"[PROGRESS] Record bootstrap: "
                    f"{completed_conditions}/{total_conditions_hint or '?'} conditions "
                    f"({percentage}), elapsed={elapsed / 60:.1f} min, "
                    f"ETA={remaining_text}",
                    flush=True,
                )

        completeness_rows.append({
            **metadata,
            "target_id": target_id,
            "status": "ok",
            "n_prediction_rows": int(len(predictions)),
            "n_conditions": int(predictions.groupby(condition_columns, dropna=False).ngroups),
            "n_status_rows": int(len(status)),
            "has_backend_metadata": (target_dir / "backend_noise_metadata.json").exists(),
            "has_failures_file": (target_dir / "failures.csv").exists(),
        })

    results = pd.DataFrame(result_rows)
    if results.empty:
        raise RuntimeError("No noisy prediction results were available to combine")

    print("[PROGRESS] Writing condition-level results.", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(results, args.out_dir / "noisy_mia_results_long.csv")
    atomic_csv(pd.DataFrame(completeness_rows), args.out_dir / "completeness_report.csv")
    atomic_csv(pd.DataFrame(consistency_rows), args.out_dir / "sample_consistency_report.csv")

    summary_metrics = [
        "train_accuracy", "test_accuracy", "accuracy_gap", "train_loss", "test_loss", "loss_gap",
        "roc_auc", "balanced_accuracy", "membership_advantage", "crossfit_tpr", "crossfit_fpr",
    ]
    for requested in fpr_points:
        summary_metrics.append(f"tpr_at_{int(round(requested * 100))}pct_fpr")

    simulator_summary = summarize_over_replicates(
        results,
        group_columns=[
            column for column in [
                "target_id", "structural_cell_id", "model_seed", "risk_role", "fm_kind", "reps",
                "depth", "mode", "shots", "attack", "backend_name", "noise_model_loaded",
            ] if column in results.columns
        ],
        metrics=summary_metrics,
        bootstrap=args.bootstrap,
        bootstrap_unit="simulator_seeds_within_fixed_target",
        progress_label="Simulator-seed summary",
        progress_every=args.progress_every,
    )
    atomic_csv(simulator_summary, args.out_dir / "noisy_mia_simulator_summary.csv")

    # Average simulator repeats inside each independently trained target before summarizing target seeds.
    target_condition_means = results.groupby(
        [column for column in [
            "target_id", "structural_cell_id", "model_seed", "data_seed", "risk_role", "fm_kind",
            "reps", "depth", "mode", "shots", "attack",
        ] if column in results.columns],
        dropna=False,
    )[summary_metrics].mean(numeric_only=True).reset_index()
    atomic_csv(target_condition_means, args.out_dir / "noisy_mia_target_condition_means.csv")

    target_seed_summary = summarize_over_replicates(
        target_condition_means,
        group_columns=[column for column in [
            "structural_cell_id", "risk_role", "fm_kind", "reps", "depth", "mode", "shots", "attack",
        ] if column in target_condition_means.columns],
        metrics=summary_metrics,
        bootstrap=0,
        bootstrap_unit="target_model_seeds",
        progress_label="Target-model-seed summary",
        progress_every=args.progress_every,
    )
    target_seed_summary["ci_method"] = "none_for_per_cell_n3; report mean±sample_SD across target-model seeds"
    target_seed_summary["bootstrap_unit"] = "none"
    target_seed_summary["bootstrap_replicates"] = 0
    atomic_csv(target_seed_summary, args.out_dir / "noisy_mia_target_seed_summary.csv")

    # Clean-baseline changes, with exact values repeated only as a fixed reference.
    exact = results[results["mode"].astype(str) == "exact"].copy()
    exact_columns = ["target_id", "attack"] + summary_metrics
    exact = exact[exact_columns].drop_duplicates(["target_id", "attack"])
    exact = exact.rename(columns={metric: f"exact_{metric}" for metric in summary_metrics})
    shot = results[results["mode"].astype(str) != "exact"].copy()
    changes = shot.merge(exact, on=["target_id", "attack"], how="left", validate="many_to_one")
    for metric in summary_metrics:
        changes[f"delta_{metric}_from_exact"] = changes[metric] - changes[f"exact_{metric}"]
    atomic_csv(changes, args.out_dir / "noisy_changes_long.csv")

    delta_metrics = [f"delta_{metric}_from_exact" for metric in summary_metrics]
    change_simulator_summary = summarize_over_replicates(
        changes,
        group_columns=[column for column in [
            "target_id", "structural_cell_id", "model_seed", "risk_role", "fm_kind", "reps",
            "depth", "mode", "shots", "attack", "backend_name", "noise_model_loaded",
        ] if column in changes.columns],
        metrics=delta_metrics,
        bootstrap=args.bootstrap,
        bootstrap_unit="paired_simulator_seeds_within_fixed_target",
        progress_label="Paired change summary",
        progress_every=args.progress_every,
    )
    atomic_csv(change_simulator_summary, args.out_dir / "noisy_changes_simulator_summary.csv")

    change_target_means = changes.groupby(
        [column for column in [
            "target_id", "structural_cell_id", "model_seed", "data_seed", "risk_role", "fm_kind",
            "reps", "depth", "mode", "shots", "attack",
        ] if column in changes.columns],
        dropna=False,
    )[delta_metrics].mean(numeric_only=True).reset_index()
    change_target_seed_summary = summarize_over_replicates(
        change_target_means,
        group_columns=[column for column in [
            "structural_cell_id", "risk_role", "fm_kind", "reps", "depth", "mode", "shots", "attack",
        ] if column in change_target_means.columns],
        metrics=delta_metrics,
        bootstrap=0,
        bootstrap_unit="target_model_seeds",
        progress_label="Target-model-seed change summary",
        progress_every=args.progress_every,
    )
    change_target_seed_summary["ci_method"] = "none_for_per_cell_n3; report mean±sample_SD across target-model seeds"
    change_target_seed_summary["bootstrap_unit"] = "none"
    change_target_seed_summary["bootstrap_replicates"] = 0
    atomic_csv(change_target_seed_summary, args.out_dir / "noisy_changes_target_seed_summary.csv")

    consistency = pd.DataFrame(consistency_rows)
    metadata = {
        "main_output": "noisy_mia_results_long.csv",
        "membership_convention": "1=member,0=nonmember",
        "primary_attack_metric": "ROC AUC",
        "threshold_selection": "stratified cross-fitting; Youden J threshold learned on calibration folds",
        "fixed_fpr_points": fpr_points,
        "low_fpr_policy": (
            "TPR@0.1% and TPR@1% are not generated. With 100 nonmembers, FPR resolution is 1%; "
            "TPR@10% is primary and TPR@5% secondary."
        ),
        "record_auc_ci": "95% stratified percentile bootstrap over member/nonmember records within a fixed target condition",
        "simulator_uncertainty": "mean±SD and percentile bootstrap CI over simulator seeds for a fixed target checkpoint",
        "target_model_uncertainty": "mean±sample SD across independently trained target-model seeds after averaging simulator repeats",
        "pseudo_replication_warning": (
            "Simulator seeds and record bootstraps are not treated as independent target models. "
            "Target-seed summaries average simulator replicates before aggregating model seeds."
        ),
        "bootstrap_replicates": int(args.bootstrap),
        "attacks": attacks,
        "n_targets_with_results": int(results["target_id"].nunique()),
        "all_sample_manifests_consistent": bool(consistency.empty or consistency["matches_manifest"].all()),
        "scope": "backend-derived local Aer finite-shot sanity check; not hardware execution",
    }
    atomic_json(metadata, args.out_dir / "analysis_metadata.json")
    total_elapsed = time.monotonic() - analysis_started
    print(
        f"[PROGRESS] Combination complete in {total_elapsed / 60:.1f} min.",
        flush=True,
    )
    print(f"[OK] Combined long results: {args.out_dir / 'noisy_mia_results_long.csv'}")
    print(f"[OK] Simulator summary: {args.out_dir / 'noisy_mia_simulator_summary.csv'}")
    print(f"[OK] Target-seed summary: {args.out_dir / 'noisy_mia_target_seed_summary.csv'}")


if __name__ == "__main__":
    main()
