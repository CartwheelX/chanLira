#!/usr/bin/env python3
"""Analyze the complete, protocol-locked ChannelLiRA Q0 acquisition."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.channel_lira_q0_common import (  # noqa: E402
    ATTACKS,
    DEFAULT_OUT,
    DEFAULT_PROTOCOL,
    DEFAULT_PROTOCOL_LOCK,
    DEFAULT_SNAPSHOT,
    DEFAULT_TARGETS,
    build_features,
    calibration_threshold,
    empirical_tpr_at_fpr,
    loss_conditioned_auc,
    read_targets,
    sha256,
    validate_protocol,
)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_bundle(
    path: Path, protocol: dict[str, Any], protocol_hash: str
) -> dict[str, Any]:
    metadata_path = path.parent.parent / "metadata" / f"{path.stem}.json"
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete Q0 source bundle: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("protocol_sha256") != protocol_hash:
        raise ValueError(f"Q0 source protocol mismatch: {metadata_path}")
    if metadata.get("payload_sha256") != sha256(path):
        raise ValueError(f"Q0 source hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        payload = {name: saved[name] for name in saved.files}
    target_id = str(payload["target_id"])
    if target_id != path.stem or metadata.get("target_id") != target_id:
        raise ValueError(f"Q0 source target mismatch: {path}")
    membership = np.asarray(payload["membership"], dtype=int)
    if np.bincount(membership, minlength=2).tolist() != [1000, 1000]:
        raise ValueError(f"Q0 source membership imbalance: {path}")
    identities = np.asarray(payload["content_ids"]).astype(str)
    if len(set(identities.tolist())) != len(identities):
        raise ValueError(f"Q0 source contains duplicate content identities: {path}")
    features = build_features(payload, protocol)
    return {
        "target_id": target_id,
        "cell": str(payload["structural_cell_id"]),
        "membership": membership,
        "labels": np.asarray(payload["labels"], dtype=int),
        "content_ids": identities,
        "features": features,
        "metadata": metadata,
        "path": path,
        "metadata_path": metadata_path,
    }


def aggregate_training_identities(
    features: np.ndarray,
    membership: np.ndarray,
    identities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(identities)
    ordered_ids = identities[order]
    ordered_x = np.asarray(features, dtype=np.float64)[order]
    ordered_y = np.asarray(membership, dtype=int)[order]
    unique, starts, counts = np.unique(
        ordered_ids, return_index=True, return_counts=True
    )
    summed = np.add.reduceat(ordered_x, starts, axis=0)
    averaged = summed / counts[:, None]
    labels = ordered_y[starts]
    for start, count, label in zip(starts, counts, labels):
        if not np.all(ordered_y[start : start + count] == label):
            raise ValueError("A content identity has inconsistent membership labels")
    return averaged, labels, unique


def fit_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    calibration_x: np.ndarray,
    victim_x: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    specification = protocol["attack_model"]
    scaler = StandardScaler().fit(train_x)
    transformed_train = scaler.transform(train_x)
    transformed_calibration = scaler.transform(calibration_x)
    transformed_victim = scaler.transform(victim_x)
    calibration_scores = []
    victim_scores = []
    for attacker_seed in specification["attacker_seeds"]:
        classifier = MLPClassifier(
            hidden_layer_sizes=tuple(specification["hidden_layer_sizes"]),
            activation=specification["activation"],
            solver=specification["solver"],
            alpha=float(specification["alpha"]),
            batch_size=min(int(specification["batch_size"]), len(train_y)),
            learning_rate_init=float(specification["learning_rate_init"]),
            max_iter=int(specification["max_iter"]),
            early_stopping=bool(specification["early_stopping"]),
            validation_fraction=float(specification["validation_fraction"]),
            n_iter_no_change=int(specification["n_iter_no_change"]),
            random_state=int(attacker_seed),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            classifier.fit(transformed_train, train_y)
        member_column = int(np.flatnonzero(classifier.classes_ == 1)[0])
        calibration_scores.append(
            classifier.predict_proba(transformed_calibration)[:, member_column]
        )
        victim_scores.append(
            classifier.predict_proba(transformed_victim)[:, member_column]
        )
    return np.mean(calibration_scores, axis=0), np.mean(victim_scores, axis=0)


def evaluate_fold(
    victim: dict[str, Any],
    calibration: dict[str, Any],
    training: list[dict[str, Any]],
    protocol: dict[str, Any],
    score_dir: Path,
) -> list[dict[str, Any]]:
    victim_ids = set(victim["content_ids"].tolist())
    calibration_keep = np.asarray(
        [identity not in victim_ids for identity in calibration["content_ids"]],
        dtype=bool,
    )
    calibration_ids = set(calibration["content_ids"][calibration_keep].tolist())
    held_ids = victim_ids | calibration_ids
    identity_rows = []
    membership_rows = []
    feature_rows: dict[str, list[np.ndarray]] = {attack: [] for attack in ATTACKS}
    training_loss_rows = []
    raw_training_count = 0
    removed_overlap_count = 0
    for bundle in training:
        keep = np.asarray(
            [identity not in held_ids for identity in bundle["content_ids"]],
            dtype=bool,
        )
        raw_training_count += len(keep)
        removed_overlap_count += int((~keep).sum())
        identity_rows.append(bundle["content_ids"][keep])
        membership_rows.append(bundle["membership"][keep])
        training_loss_rows.append(bundle["features"]["loss_value"][keep])
        for attack in ATTACKS:
            feature_rows[attack].append(bundle["features"][attack][keep])
    training_ids = np.concatenate(identity_rows)
    training_membership = np.concatenate(membership_rows)
    raw_training_loss = np.concatenate(training_loss_rows)

    rows = []
    saved_scores: dict[str, np.ndarray] = {
        "content_ids": victim["content_ids"],
        "membership": victim["membership"].astype(np.uint8),
        "labels": victim["labels"],
        "loss_value": victim["features"]["loss_value"].reshape(-1),
    }
    nominal = float(protocol["evaluation"]["nominal_fpr"])
    for attack in ATTACKS:
        raw_training_features = np.concatenate(feature_rows[attack])
        train_x, train_y, unique_training_ids = aggregate_training_identities(
            raw_training_features, training_membership, training_ids
        )
        train_loss, loss_y, loss_ids = aggregate_training_identities(
            raw_training_loss, training_membership, training_ids
        )
        if not np.array_equal(unique_training_ids, loss_ids) or not np.array_equal(
            train_y, loss_y
        ):
            raise RuntimeError("Q0 training identity aggregation changed between features")
        calibration_x = calibration["features"][attack][calibration_keep]
        victim_x = victim["features"][attack]
        if attack == "loss_mia":
            calibration_scores = calibration_x.reshape(-1)
            victim_scores = victim_x.reshape(-1)
        else:
            calibration_scores, victim_scores = fit_scores(
                train_x,
                train_y,
                calibration_x,
                victim_x,
                protocol,
            )
        victim_y = victim["membership"]
        calibration_y = calibration["membership"][calibration_keep]
        auc = float(roc_auc_score(victim_y, victim_scores))
        empirical_tpr, empirical_fpr = empirical_tpr_at_fpr(
            victim_y, victim_scores, nominal
        )
        threshold = calibration_threshold(
            calibration_scores[calibration_y == 0], nominal
        )
        decisions = victim_scores >= threshold
        actual_fpr = float(decisions[victim_y == 0].mean())
        operational_tpr = float(decisions[victim_y == 1].mean())
        conditioned_auc, conditioned_bins, conditioned_records = loss_conditioned_auc(
            victim_y,
            victim_scores,
            victim["features"]["loss_value"],
            train_loss,
            bins=20,
        )
        saved_scores[f"score_{attack}"] = victim_scores.astype(np.float32)
        rows.append(
            {
                "victim_target_id": victim["target_id"],
                "calibration_target_id": calibration["target_id"],
                "structural_cell_id": victim["cell"],
                "attack": attack,
                "auc": auc,
                "empirical_tpr_at_1pct_fpr": empirical_tpr,
                "empirical_attained_fpr": empirical_fpr,
                "nominal_fpr": nominal,
                "calibration_threshold": threshold,
                "actual_victim_fpr": actual_fpr,
                "operational_tpr": operational_tpr,
                "loss_conditioned_auc": conditioned_auc,
                "loss_conditioned_bins": conditioned_bins,
                "loss_conditioned_records": conditioned_records,
                "victim_members": int((victim_y == 1).sum()),
                "victim_nonmembers": int((victim_y == 0).sum()),
                "calibration_members": int((calibration_y == 1).sum()),
                "calibration_nonmembers": int((calibration_y == 0).sum()),
                "training_rows_before_identity_aggregation": int(raw_training_count),
                "training_rows_removed_for_held_identity_overlap": int(
                    removed_overlap_count
                ),
                "training_unique_content_identities": len(unique_training_ids),
                "feature_count": int(train_x.shape[1]),
                "attacker": "raw score" if attack == "loss_mia" else "fixed MLP(64,32), seeds 41/42/43",
            }
        )
    score_dir.mkdir(parents=True, exist_ok=True)
    score_path = score_dir / f"{victim['target_id']}.npz"
    temporary = score_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **saved_scores)
    temporary.replace(score_path)
    return rows


def stratified_target_bootstrap(
    frame: pd.DataFrame,
    value: str,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    by_cell = {
        cell: group[value].to_numpy(dtype=float)
        for cell, group in frame.groupby("structural_cell_id", sort=True)
    }
    estimates = np.empty(int(replicates), dtype=float)
    for index in range(int(replicates)):
        cell_means = [
            rng.choice(values, size=len(values), replace=True).mean()
            for values in by_cell.values()
        ]
        estimates[index] = np.mean(cell_means)
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def summarize_metrics(frame: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    rows = []
    metrics = (
        "auc",
        "empirical_tpr_at_1pct_fpr",
        "actual_victim_fpr",
        "operational_tpr",
        "loss_conditioned_auc",
    )
    for attack, group in frame.groupby("attack", sort=False):
        row: dict[str, Any] = {"attack": attack, "targets": group.victim_target_id.nunique()}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_sd_targets"] = float(np.nanstd(values, ddof=1))
            valid = group[np.isfinite(group[metric])]
            if len(valid):
                low, high = stratified_target_bootstrap(
                    valid,
                    metric,
                    replicates=int(protocol["evaluation"]["bootstrap_replicates"]),
                    seed=int(protocol["evaluation"]["bootstrap_seed"]),
                )
            else:
                low = high = float("nan")
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def build_contrasts(frame: pd.DataFrame, protocol: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = {
        "paired_joint_minus_loss": ("paired_joint_probe", "loss_mia"),
        "paired_joint_minus_learned": ("paired_joint_probe", "learned_mia"),
        "paired_joint_minus_classical_stochastic": (
            "paired_joint_probe",
            "classical_stochastic_control",
        ),
        "fixed_joint_minus_learned": ("fixed_joint", "learned_mia"),
        "paired_joint_minus_fixed_joint": ("paired_joint_probe", "fixed_joint"),
    }
    metrics = (
        "auc",
        "empirical_tpr_at_1pct_fpr",
        "actual_victim_fpr",
        "operational_tpr",
        "loss_conditioned_auc",
    )
    indexed = frame.set_index(["victim_target_id", "attack"])
    target_rows = []
    for victim in frame.victim_target_id.unique():
        cell = str(frame[frame.victim_target_id == victim].structural_cell_id.iloc[0])
        for name, (attack, comparator) in comparisons.items():
            row: dict[str, Any] = {
                "victim_target_id": victim,
                "structural_cell_id": cell,
                "contrast": name,
                "attack": attack,
                "comparator": comparator,
            }
            for metric in metrics:
                row[f"{metric}_difference"] = float(
                    indexed.loc[(victim, attack), metric]
                    - indexed.loc[(victim, comparator), metric]
                )
            target_rows.append(row)
    target = pd.DataFrame(target_rows)
    summary_rows = []
    for contrast, group in target.groupby("contrast", sort=False):
        row = {
            "contrast": contrast,
            "attack": group.attack.iloc[0],
            "comparator": group.comparator.iloc[0],
            "targets": len(group),
        }
        for metric in metrics:
            column = f"{metric}_difference"
            row[f"{column}_mean"] = float(np.nanmean(group[column]))
            row[f"{column}_positive_targets"] = int((group[column] > 0).sum())
            valid = group[np.isfinite(group[column])]
            low, high = stratified_target_bootstrap(
                valid,
                column,
                replicates=int(protocol["evaluation"]["bootstrap_replicates"]),
                seed=int(protocol["evaluation"]["bootstrap_seed"]),
            )
            row[f"{column}_ci_low"] = low
            row[f"{column}_ci_high"] = high
        summary_rows.append(row)
    return target, pd.DataFrame(summary_rows)


def screening_decision(
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    indexed = contrasts.set_index("contrast")

    def value(contrast: str, column: str) -> float:
        return float(indexed.loc[contrast, column])

    gate = protocol["screening_gate"]
    practical = gate["practical_gain_over_loss"]
    practical_pass = bool(
        value("paired_joint_minus_loss", "auc_difference_mean")
        >= float(practical["minimum_mean_auc_difference"])
        and value(
            "paired_joint_minus_loss", "empirical_tpr_at_1pct_fpr_difference_mean"
        )
        >= float(practical["minimum_mean_tpr_at_1pct_difference"])
        and value("paired_joint_minus_loss", "auc_difference_positive_targets")
        >= int(practical["minimum_targets_positive_for_both_metrics"])
        and value(
            "paired_joint_minus_loss",
            "empirical_tpr_at_1pct_fpr_difference_positive_targets",
        )
        >= int(practical["minimum_targets_positive_for_both_metrics"])
    )
    learned = gate["gain_over_unchanged_learned_mia"]
    learned_pass = bool(
        value("paired_joint_minus_learned", "auc_difference_mean")
        >= float(learned["minimum_mean_auc_difference"])
        and value(
            "paired_joint_minus_learned", "empirical_tpr_at_1pct_fpr_difference_mean"
        )
        >= float(learned["minimum_mean_tpr_at_1pct_difference"])
        and value("paired_joint_minus_learned", "auc_difference_positive_targets")
        >= int(learned["minimum_targets_positive_for_both_metrics"])
        and value(
            "paired_joint_minus_learned",
            "empirical_tpr_at_1pct_fpr_difference_positive_targets",
        )
        >= int(learned["minimum_targets_positive_for_both_metrics"])
    )
    conditional = gate["conditional_gain"]
    conditional_pass = bool(
        value(
            "paired_joint_minus_learned", "loss_conditioned_auc_difference_mean"
        )
        >= float(conditional["minimum_mean_difference"])
        and value(
            "paired_joint_minus_learned",
            "loss_conditioned_auc_difference_positive_targets",
        )
        >= int(conditional["minimum_targets_positive"])
    )
    mechanism_fixed = bool(
        value("fixed_joint_minus_learned", "auc_difference_mean") >= 0.005
        and value("fixed_joint_minus_learned", "auc_difference_positive_targets") >= 4
    )
    mechanism_active = bool(
        value("paired_joint_minus_fixed_joint", "auc_difference_mean") >= 0.005
        and value("paired_joint_minus_fixed_joint", "auc_difference_positive_targets") >= 4
    )
    mechanism_pass = mechanism_fixed or mechanism_active
    classical_pass = bool(
        value("paired_joint_minus_classical_stochastic", "auc_difference_mean")
        >= float(gate["classical_stochastic_control"]["minimum_mean_auc_difference"])
    )
    metric_index = metrics.set_index("attack")
    paired_fpr = float(metric_index.loc["paired_joint_probe", "actual_victim_fpr_mean"])
    operational_delta = value(
        "paired_joint_minus_learned", "operational_tpr_difference_mean"
    )
    operational_pass = bool(
        paired_fpr
        <= float(gate["operational_sanity"]["maximum_mean_realized_victim_fpr"])
        and operational_delta > 0
    )
    components = {
        "practical_gain_over_loss": practical_pass,
        "gain_over_unchanged_learned_mia": learned_pass,
        "conditional_gain_within_loss_strata": conditional_pass,
        "quantum_mechanism": mechanism_pass,
        "classical_stochastic_control": classical_pass,
        "operational_sanity": operational_pass,
    }
    passed = all(components.values())
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "screen_passed": passed,
        "decision": (
            "continue_to_extended_quantum_channel_probing_study"
            if passed
            else "stop_stronger_quantum_stochastic_mia_claim_under_this_design"
        ),
        "components": components,
        "mechanism_detail": {
            "fixed_raw_joint_advantage": mechanism_fixed,
            "active_layout_advantage": mechanism_active,
        },
        "observed": {
            "paired_joint_minus_loss_mean_auc": value(
                "paired_joint_minus_loss", "auc_difference_mean"
            ),
            "paired_joint_minus_loss_mean_tpr_at_1pct": value(
                "paired_joint_minus_loss", "empirical_tpr_at_1pct_fpr_difference_mean"
            ),
            "paired_joint_minus_learned_mean_auc": value(
                "paired_joint_minus_learned", "auc_difference_mean"
            ),
            "paired_joint_minus_learned_mean_tpr_at_1pct": value(
                "paired_joint_minus_learned",
                "empirical_tpr_at_1pct_fpr_difference_mean",
            ),
            "paired_joint_minus_learned_loss_conditioned_auc": value(
                "paired_joint_minus_learned", "loss_conditioned_auc_difference_mean"
            ),
            "paired_joint_mean_realized_victim_fpr": paired_fpr,
            "paired_joint_minus_learned_operational_tpr": operational_delta,
        },
        "interpretation": (
            "Exploratory screening evidence only; passing does not establish a publication claim."
        ),
    }


def plot_results(metrics: pd.DataFrame, contrasts: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "loss_mia": "Loss",
        "learned_mia": "Learned MIA",
        "classical_stochastic_control": "Classical stochastic",
        "fixed_marginal": "Fixed marginal",
        "fixed_joint": "Fixed joint",
        "paired_probability_probe": "Paired probability",
        "paired_marginal_probe": "Paired marginal",
        "paired_joint_probe": "Paired joint",
        "clean_z_diagnostic": "Clean Z diagnostic",
    }
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    order = list(ATTACKS)
    summary = metrics.set_index("attack").loc[order]
    for metric, title, filename in (
        ("auc", "Q0 attack AUC", "attack_auc"),
        (
            "empirical_tpr_at_1pct_fpr",
            "Q0 TPR at 1% empirical FPR",
            "attack_tpr_at_1pct",
        ),
        (
            "loss_conditioned_auc",
            "Q0 loss-conditioned AUC",
            "loss_conditioned_auc",
        ),
    ):
        figure, axis = plt.subplots(figsize=(11, 5.5))
        x = np.arange(len(order))
        axis.bar(x, summary[f"{metric}_mean"], color="#31688e")
        axis.errorbar(
            x,
            summary[f"{metric}_mean"],
            yerr=summary[f"{metric}_sd_targets"],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
        axis.set_xticks(x, [labels[value] for value in order], rotation=30, ha="right")
        axis.set_ylabel(metric.replace("_", " "))
        axis.set_title(title)
        if "auc" in metric:
            axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        else:
            axis.axhline(0.01, color="black", linestyle="--", linewidth=1)
        figure.tight_layout()
        figure.savefig(plots / f"{filename}.png", dpi=180)
        figure.savefig(plots / f"{filename}.svg")
        plt.close(figure)

    selected = contrasts[
        contrasts.contrast.isin(
            ["paired_joint_minus_loss", "paired_joint_minus_learned"]
        )
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for axis, metric, title in (
        (axes[0], "auc_difference", "AUC difference"),
        (axes[1], "empirical_tpr_at_1pct_fpr_difference", "TPR@1% difference"),
    ):
        for index, (contrast, group) in enumerate(selected.groupby("contrast", sort=False)):
            x = np.arange(len(group)) + (index - 0.5) * 0.18
            axis.scatter(x, group[metric], label=contrast, s=38)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xticks(np.arange(6), [f"T{i + 1}" for i in range(6)])
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots / "paired_joint_target_differences.png", dpi=180)
    figure.savefig(plots / "paired_joint_target_differences.svg")
    plt.close(figure)


def build_report(
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
    decision: dict[str, Any],
    protocol_hash: str,
) -> str:
    metric_index = metrics.set_index("attack")
    contrast_index = contrasts.set_index("contrast")
    labels = {
        "loss_mia": "Loss MIA",
        "learned_mia": "unchanged learned MIA",
        "classical_stochastic_control": "classical stochastic control",
        "fixed_marginal": "fixed-layout marginal probe",
        "fixed_joint": "fixed-layout joint-bitstring probe",
        "paired_probability_probe": "paired-layout probability probe",
        "paired_marginal_probe": "paired-layout marginal probe",
        "paired_joint_probe": "paired-layout joint-bitstring probe",
        "clean_z_diagnostic": "privileged clean-Z diagnostic",
    }
    lines = [
        "# Q0 residual quantum leakage screen",
        "",
        f"- Protocol SHA-256: `{protocol_hash}`.",
        "- Six independently data/model-seeded targets in two structural cells.",
        "- Ten 128-shot queries per compared attack; local Aer with the frozen IBM-Kingston-derived snapshot.",
        "- No reference models and no quantum-hardware execution.",
        "- Screening evidence only; these results are excluded from Phase 7.",
        "",
        "## Results",
        "",
        "| Attack | Mean AUC | Mean TPR@1% | Mean actual FPR at transferred 1% threshold | Mean operational TPR | Loss-conditioned AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for attack in ATTACKS:
        row = metric_index.loc[attack]
        lines.append(
            f"| {labels[attack]} | {row['auc_mean']:.4f} | "
            f"{100 * row['empirical_tpr_at_1pct_fpr_mean']:.2f}% | "
            f"{100 * row['actual_victim_fpr_mean']:.2f}% | "
            f"{100 * row['operational_tpr_mean']:.2f}% | "
            f"{row['loss_conditioned_auc_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Decisive comparisons",
            "",
            "| Contrast | Mean AUC difference | Mean TPR@1% difference | Mean loss-conditioned AUC difference |",
            "|---|---:|---:|---:|",
        ]
    )
    for contrast in (
        "paired_joint_minus_loss",
        "paired_joint_minus_learned",
        "paired_joint_minus_classical_stochastic",
        "fixed_joint_minus_learned",
        "paired_joint_minus_fixed_joint",
    ):
        row = contrast_index.loc[contrast]
        lines.append(
            f"| {contrast} | {row['auc_difference_mean']:+.4f} | "
            f"{100 * row['empirical_tpr_at_1pct_fpr_difference_mean']:+.2f} pp | "
            f"{row['loss_conditioned_auc_difference_mean']:+.4f} |"
        )
    lines.extend(["", "## Locked screening decision", ""])
    for name, passed in decision["components"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`.")
    lines.extend(
        [
            "",
            f"**Overall: {'PASS' if decision['screen_passed'] else 'FAIL'} — `{decision['decision']}`.**",
            "",
            "Passing would justify a larger preregistered study, not a breakthrough claim. Failure invokes the locked stop rule for this stronger-quantum-stochastic-MIA direction under the tested access model.",
            "",
            "## Interpretation boundaries",
            "",
            "- Loss and the learned MIA use probability outputs from ten fixed-layout queries.",
            "- Raw marginal/joint attacks assume access to bitstring counts.",
            "- Paired attacks additionally assume control over the physical initial layout.",
            "- The clean-Z attack is a privileged mechanism diagnostic, not a deployable black-box baseline.",
            "- The learned estimator architecture is unchanged; only its training data are moved to independent auxiliary targets.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(args: argparse.Namespace) -> None:
    protocol = validate_protocol(
        args.protocol, args.protocol_lock, args.targets, args.snapshot
    )
    protocol_hash = sha256(args.protocol)
    if args.acknowledge_protocol_hash != protocol_hash:
        raise ValueError(
            "Q0 analysis requires --acknowledge-protocol-hash equal to the locked protocol SHA-256"
        )
    rows = read_targets(args.targets)
    expected_ids = [row["target_id"] for row in rows]
    bundles = {}
    source_rows = []
    for target_id in expected_ids:
        path = args.out_dir / "raw" / f"{target_id}.npz"
        bundle = load_bundle(path, protocol, protocol_hash)
        bundles[target_id] = bundle
        source_rows.extend(
            [
                {
                    "kind": "payload",
                    "target_id": target_id,
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                },
                {
                    "kind": "metadata",
                    "target_id": target_id,
                    "path": str(bundle["metadata_path"].resolve()),
                    "sha256": sha256(bundle["metadata_path"]),
                },
            ]
        )
    calibration_map = protocol["evaluation"]["calibration_map"]
    metric_rows = []
    for victim_id in expected_ids:
        calibration_id = calibration_map[victim_id]
        training = [
            bundle
            for target_id, bundle in bundles.items()
            if target_id not in {victim_id, calibration_id}
        ]
        metric_rows.extend(
            evaluate_fold(
                bundles[victim_id],
                bundles[calibration_id],
                training,
                protocol,
                args.out_dir / "analysis" / "scores",
            )
        )
        print(f"[ANALYZE] victim {victim_id} complete", flush=True)
    metrics_target = pd.DataFrame(metric_rows)
    metrics_summary = summarize_metrics(metrics_target, protocol)
    contrasts_target, contrasts_summary = build_contrasts(metrics_target, protocol)
    decision = screening_decision(metrics_summary, contrasts_summary, protocol)
    analysis_dir = args.out_dir / "analysis"
    atomic_csv(metrics_target, analysis_dir / "metrics_target.csv")
    atomic_csv(metrics_summary, analysis_dir / "metrics_summary.csv")
    atomic_csv(contrasts_target, analysis_dir / "contrasts_target.csv")
    atomic_csv(contrasts_summary, analysis_dir / "contrasts_summary.csv")
    atomic_json(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": protocol_hash,
            "target_manifest_sha256": sha256(args.targets),
            "snapshot_manifest_sha256": sha256(
                args.snapshot / "snapshot_manifest.json"
            ),
            "sources": source_rows,
        },
        analysis_dir / "SOURCE_MANIFEST.json",
    )
    atomic_json(
        {
            "attacks": list(ATTACKS),
            "feature_hierarchy": protocol["feature_hierarchy"],
            "feature_counts": {
                attack: int(bundles[expected_ids[0]]["features"][attack].shape[1])
                for attack in ATTACKS
            },
            "same_estimator_for_all_learned_attacks": True,
        },
        analysis_dir / "FEATURES.json",
    )
    atomic_json(decision, analysis_dir / "SCREENING_DECISION.json")
    report = build_report(metrics_summary, contrasts_summary, decision, protocol_hash)
    report_path = analysis_dir / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    plot_results(metrics_summary, contrasts_target, analysis_dir)
    print(
        f"[DECISION] pass={decision['screen_passed']} {decision['decision']}",
        flush=True,
    )
    print(f"[REPORT] {report_path.resolve()}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-lock", type=Path, default=DEFAULT_PROTOCOL_LOCK)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--acknowledge-protocol-hash", default="")
    args = parser.parse_args()
    for name in ("protocol", "protocol_lock", "targets", "snapshot", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
