#!/usr/bin/env python3
"""Shared guards and feature construction for the locked ChannelLiRA Q0 screen."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "reviewer_targets/channel_lira_q0_protocol.json"
DEFAULT_PROTOCOL_LOCK = ROOT / "reviewer_targets/channel_lira_q0_protocol.sha256"
DEFAULT_TARGETS = ROOT / "reviewer_targets/channel_lira_q0_targets.csv"
DEFAULT_SNAPSHOT = (
    ROOT / "channel_lira_results/noisy_reference_canary_phase5/backend_snapshot"
)
DEFAULT_OUT = ROOT / "channel_lira_results/q0_residual_quantum_leakage"
DEFAULT_RUN_ROOT = ROOT / "reviewer_runs"
ATTACKS = (
    "loss_mia",
    "learned_mia",
    "classical_stochastic_control",
    "fixed_marginal",
    "fixed_joint",
    "paired_probability_probe",
    "paired_marginal_probe",
    "paired_joint_probe",
    "clean_z_diagnostic",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_targets(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_protocol(
    protocol_path: Path = DEFAULT_PROTOCOL,
    lock_path: Path = DEFAULT_PROTOCOL_LOCK,
    targets_path: Path = DEFAULT_TARGETS,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    lock_path = lock_path.resolve()
    targets_path = targets_path.resolve()
    snapshot_path = snapshot_path.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock_fields = lock_path.read_text(encoding="utf-8").strip().split()
    errors: list[str] = []
    observed_protocol_hash = sha256(protocol_path)
    if len(lock_fields) != 2 or lock_fields[0] != observed_protocol_hash:
        errors.append("Q0 protocol lock does not match the protocol bytes")
    if protocol.get("automatic_execution") is not False:
        errors.append("Q0 automatic-execution guard changed")
    if protocol.get("confirmatory_claim_allowed") is not False:
        errors.append("Q0 must remain an exploratory screen")
    provenance = protocol.get("provenance", {})
    if resolve_repo_path(provenance.get("target_manifest", "")) != targets_path:
        errors.append("Q0 target-manifest path differs from the protocol")
    if provenance.get("target_manifest_sha256") != sha256(targets_path):
        errors.append("Q0 target-manifest hash differs from the protocol")
    snapshot_manifest = snapshot_path / "snapshot_manifest.json"
    if not snapshot_manifest.is_file():
        errors.append("Q0 backend snapshot manifest is missing")
    elif provenance.get("snapshot_manifest_sha256") != sha256(snapshot_manifest):
        errors.append("Q0 backend snapshot hash differs from the protocol")
    parent = protocol.get("relationship_to_phase7", {})
    parent_path = resolve_repo_path(parent.get("phase7_protocol", ""))
    if not parent_path.is_file() or parent.get("phase7_protocol_sha256") != sha256(parent_path):
        errors.append("Q0 parent Phase-7 protocol hash is invalid")

    rows = read_targets(targets_path)
    population = protocol.get("study_population", {})
    if len(rows) != int(population.get("target_count", -1)):
        errors.append("Q0 target count differs from the protocol")
    ids = [row.get("target_id", "") for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("Q0 target IDs are not unique")
    if {row.get("q0_analysis_role") for row in rows} != {"screening"}:
        errors.append("Q0 targets must all be screening targets")
    if {row.get("structural_cell_id") for row in rows} != set(
        population.get("structural_cells", [])
    ):
        errors.append("Q0 structural cells differ from the protocol")
    for cell in population.get("structural_cells", []):
        if sum(row.get("structural_cell_id") == cell for row in rows) != int(
            population.get("targets_per_cell", -1)
        ):
            errors.append(f"Q0 targets-per-cell mismatch for {cell}")
    observed_data_seeds = [int(float(row["data_seed"])) for row in rows]
    observed_model_seeds = [int(float(row["model_seed"])) for row in rows]
    if observed_data_seeds != [int(value) for value in population.get("data_seeds", [])]:
        errors.append("Q0 data seeds differ from the protocol")
    if observed_model_seeds != [int(value) for value in population.get("model_seeds", [])]:
        errors.append("Q0 model seeds differ from the protocol")
    if len(set(observed_data_seeds)) != len(rows):
        errors.append("Q0 requires one independent data seed per target")
    partition = population.get("mnist_partition_protocol", {})
    observed_partition_ids = [
        int(float(row.get("mnist_disjoint_partition_id", -1))) for row in rows
    ]
    if observed_partition_ids != [int(value) for value in partition.get("partition_ids", [])]:
        errors.append("Q0 MNIST partition IDs differ from the protocol")
    if {
        int(float(row.get("mnist_disjoint_partition_count", -1))) for row in rows
    } != {int(partition.get("partition_count", -1))}:
        errors.append("Q0 MNIST partition count differs from the protocol")
    if {
        int(float(row.get("mnist_disjoint_partition_seed", -1))) for row in rows
    } != {int(partition.get("partition_seed", -1))}:
        errors.append("Q0 MNIST partition seed differs from the protocol")

    acquisition = protocol.get("acquisition", {})
    layouts = acquisition.get("physical_initial_layouts", {})
    layout_a = [int(value) for value in layouts.get("layout_a", [])]
    layout_b = [int(value) for value in layouts.get("layout_b", [])]
    n_wires = {int(float(row.get("n_wires", 0))) for row in rows}
    if len(n_wires) != 1:
        errors.append("Q0 targets must share one wire count")
    else:
        expected_wires = next(iter(n_wires))
        if len(layout_a) != expected_wires or len(layout_b) != expected_wires:
            errors.append("Q0 physical layouts must contain one entry per logical wire")
    if len(set(layout_a)) != len(layout_a) or len(set(layout_b)) != len(layout_b):
        errors.append("Q0 physical layouts contain duplicate qubits")
    if set(layout_a) & set(layout_b):
        errors.append("Q0 physical layouts must be disjoint")
    backend_qubits = json.loads(
        (snapshot_path / "metadata.json").read_text(encoding="utf-8")
    ).get("backend_num_qubits", 0)
    if any(value < 0 or value >= int(backend_qubits) for value in layout_a + layout_b):
        errors.append("Q0 physical layout is outside the frozen backend")
    if errors:
        raise ValueError("Invalid Q0 protocol: " + "; ".join(errors))
    return protocol


def content_ids(inputs: Any, labels: Any) -> np.ndarray:
    """Content identities remain stable across independently reconstructed datasets."""
    values = np.asarray(
        inputs.detach().cpu().numpy() if hasattr(inputs, "detach") else inputs,
        dtype=np.float32,
    )
    labels = np.asarray(
        labels.detach().cpu().numpy() if hasattr(labels, "detach") else labels,
        dtype=np.int64,
    ).reshape(-1)
    if len(values) != len(labels):
        raise ValueError("Candidate tensors and labels differ in length")
    output = []
    for value, label in zip(values, labels):
        digest = hashlib.sha256()
        contiguous = np.ascontiguousarray(value, dtype=np.float32)
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
        digest.update(np.asarray([label], dtype=np.int64).tobytes())
        output.append(digest.hexdigest()[:24])
    return np.asarray(output)


def dense_counts(counts: Sequence[Mapping[Any, int]], n_wires: int, shots: int) -> np.ndarray:
    from reviewer_tools.qurift_qiskit_bridge import clean_count_key

    output = np.zeros((len(counts), 2 ** int(n_wires)), dtype=np.uint16)
    for row_index, row in enumerate(counts):
        if int(sum(int(value) for value in row.values())) != int(shots):
            raise ValueError("Bitstring counts do not sum to the locked shot count")
        for raw_key, raw_count in row.items():
            key = clean_count_key(raw_key).zfill(int(n_wires))[-int(n_wires):]
            output[row_index, int(key, 2)] += int(raw_count)
    return output


def z_sign_table(n_wires: int) -> np.ndarray:
    states = np.arange(2 ** int(n_wires), dtype=np.uint64)
    columns = []
    for wire in range(int(n_wires)):
        columns.append(1.0 - 2.0 * ((states >> wire) & 1).astype(np.float64))
    return np.stack(columns, axis=1)


def dense_z_expectations(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Dense count matrix must be two-dimensional")
    n_states = int(values.shape[1])
    n_wires = int(round(np.log2(n_states)))
    if 2 ** n_wires != n_states:
        raise ValueError("Dense count width is not a power of two")
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Dense counts contain an empty record")
    return (values / totals) @ z_sign_table(n_wires)


def z_covariance_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n_states = int(values.shape[1])
    n_wires = int(round(np.log2(n_states)))
    signs = z_sign_table(n_wires)
    probabilities = values / values.sum(axis=1, keepdims=True)
    means = probabilities @ signs
    columns = []
    for first in range(n_wires):
        for second in range(first + 1, n_wires):
            joint = probabilities @ (signs[:, first] * signs[:, second])
            columns.append(joint - means[:, first] * means[:, second])
    return np.stack(columns, axis=1)


def prediction_stats(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    true_probability = clipped[np.arange(len(labels)), labels]
    loss = -np.log(true_probability)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    confidence = probabilities.max(axis=1)
    ordered = np.sort(probabilities, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    correctness = (probabilities.argmax(axis=1) == labels).astype(float)
    return np.column_stack(
        [probabilities, loss, entropy, confidence, margin, correctness]
    )


def response_dispersion(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if values.ndim != 3 or values.shape[1] != len(labels):
        raise ValueError("Response array must have shape repetitions x records x classes")
    clipped = np.clip(values, 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=2)
    ordered = np.sort(values, axis=2)
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    modal = values.mean(axis=0).argmax(axis=1)
    flip = (values.argmax(axis=2) != modal[None, :]).mean(axis=0)
    true_probability = np.take_along_axis(
        values, labels[None, :, None], axis=2
    ).squeeze(2)
    return np.column_stack(
        [
            values.std(axis=0),
            true_probability.std(axis=0),
            entropy.std(axis=0),
            margin.std(axis=0),
            flip,
        ]
    )


def classical_multinomial_responses(
    probabilities: np.ndarray,
    *,
    repetitions: int,
    shots: int,
    seed: int,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    output = np.empty((int(repetitions), *probabilities.shape), dtype=np.float64)
    for repetition in range(int(repetitions)):
        for index, row in enumerate(probabilities):
            normalized = np.clip(row, 0.0, None)
            normalized /= normalized.sum()
            output[repetition, index] = rng.multinomial(int(shots), normalized) / float(shots)
    return output


def build_features(payload: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, np.ndarray]:
    labels = np.asarray(payload["labels"], dtype=int)
    shots = int(protocol["acquisition"]["shots_per_query"])
    probabilities_a = np.asarray(payload["probabilities_layout_a"], dtype=np.float64)
    probabilities_b = np.asarray(payload["probabilities_layout_b"], dtype=np.float64)
    z_a = np.asarray(payload["z_layout_a"], dtype=np.float64)
    z_b = np.asarray(payload["z_layout_b"], dtype=np.float64)
    counts_a = np.asarray(payload["counts_layout_a"], dtype=np.float64)
    counts_b = np.asarray(payload["counts_layout_b"], dtype=np.float64)
    if probabilities_a.shape[0] != 10 or probabilities_b.shape[0] != 5:
        raise ValueError("Q0 payload repetition counts differ from the locked protocol")

    mean_a10 = probabilities_a.mean(axis=0)
    base = prediction_stats(mean_a10, labels)
    true_probability = mean_a10[np.arange(len(labels)), labels]
    output: dict[str, np.ndarray] = {
        "loss_mia": true_probability[:, None],
        "learned_mia": base,
    }
    synthetic = classical_multinomial_responses(
        mean_a10,
        repetitions=10,
        shots=shots,
        seed=int(protocol["evaluation"]["bootstrap_seed"]),
    )
    output["classical_stochastic_control"] = np.column_stack(
        [base, response_dispersion(synthetic, labels)]
    )

    aggregate_a10 = counts_a.sum(axis=0)
    freq_a10 = aggregate_a10 / float(10 * shots)
    fixed_marginal = np.column_stack(
        [base, z_a.mean(axis=0), z_a.std(axis=0)]
    )
    output["fixed_marginal"] = fixed_marginal
    output["fixed_joint"] = np.column_stack(
        [fixed_marginal, freq_a10, z_covariance_features(aggregate_a10)]
    )

    probabilities_a5 = probabilities_a[:5]
    z_a5 = z_a[:5]
    counts_a5 = counts_a[:5]
    mean_a5 = probabilities_a5.mean(axis=0)
    mean_b5 = probabilities_b.mean(axis=0)
    paired_mean = 0.5 * (mean_a5 + mean_b5)
    paired_probability = np.column_stack(
        [
            prediction_stats(paired_mean, labels),
            mean_a5,
            mean_b5,
            mean_b5 - mean_a5,
            response_dispersion(probabilities_a5, labels),
            response_dispersion(probabilities_b, labels),
        ]
    )
    output["paired_probability_probe"] = paired_probability
    marginal_a = z_a5.mean(axis=0)
    marginal_b = z_b.mean(axis=0)
    paired_marginal = np.column_stack(
        [
            paired_probability,
            marginal_a,
            marginal_b,
            marginal_b - marginal_a,
            z_a5.std(axis=0),
            z_b.std(axis=0),
        ]
    )
    output["paired_marginal_probe"] = paired_marginal
    aggregate_a5 = counts_a5.sum(axis=0)
    aggregate_b5 = counts_b.sum(axis=0)
    freq_a5 = aggregate_a5 / float(5 * shots)
    freq_b5 = aggregate_b5 / float(5 * shots)
    covariance_a = z_covariance_features(aggregate_a5)
    covariance_b = z_covariance_features(aggregate_b5)
    output["paired_joint_probe"] = np.column_stack(
        [
            paired_marginal,
            freq_a5,
            freq_b5,
            freq_b5 - freq_a5,
            covariance_a,
            covariance_b,
            covariance_b - covariance_a,
        ]
    )
    output["clean_z_diagnostic"] = np.column_stack(
        [
            prediction_stats(np.asarray(payload["exact_probabilities"]), labels),
            np.asarray(payload["exact_z"], dtype=np.float64),
        ]
    )
    for name, values in output.items():
        values = np.asarray(values, dtype=np.float64)
        if values.shape[0] != len(labels) or not np.isfinite(values).all():
            raise ValueError(f"Invalid Q0 features for {name}: {values.shape}")
        output[name] = values
    output["loss_value"] = -np.log(np.clip(true_probability, 1e-12, 1.0))[:, None]
    return output


def empirical_tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, alpha: float) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    negatives = scores[labels == 0]
    positives = scores[labels == 1]
    threshold = calibration_threshold(negatives, alpha)
    decisions = scores >= threshold
    return float(decisions[labels == 1].mean()), float(decisions[labels == 0].mean())


def calibration_threshold(nonmember_scores: np.ndarray, alpha: float) -> float:
    values = np.asarray(nonmember_scores, dtype=float)
    allowed = int(np.floor(float(alpha) * len(values) + 1e-12))
    if allowed <= 0:
        return float("inf")
    unique, counts = np.unique(values, return_counts=True)
    unique = unique[::-1]
    cumulative = np.cumsum(counts[::-1])
    valid = np.flatnonzero(cumulative <= allowed)
    return float(unique[valid[-1]]) if len(valid) else float("inf")


def loss_conditioned_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    loss: np.ndarray,
    auxiliary_loss: np.ndarray,
    bins: int = 20,
) -> tuple[float, int, int]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    loss = np.asarray(loss, dtype=float).reshape(-1)
    auxiliary_loss = np.asarray(auxiliary_loss, dtype=float).reshape(-1)
    edges = np.unique(
        np.quantile(auxiliary_loss, np.linspace(0.0, 1.0, int(bins) + 1))
    )
    if len(edges) < 3:
        return float("nan"), 0, 0
    assignment = np.clip(np.digitize(loss, edges[1:-1], right=True), 0, len(edges) - 2)
    weighted = 0.0
    retained = 0
    used_bins = 0
    for index in range(len(edges) - 1):
        selected = assignment == index
        if selected.sum() < 2 or len(np.unique(labels[selected])) != 2:
            continue
        count = int(selected.sum())
        weighted += count * float(roc_auc_score(labels[selected], scores[selected]))
        retained += count
        used_bins += 1
    return (
        float(weighted / retained) if retained else float("nan"),
        used_bins,
        retained,
    )
