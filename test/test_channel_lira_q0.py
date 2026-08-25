from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from experiments.channel_lira_q0_analyze import (
    aggregate_training_identities,
    screening_decision,
)
from experiments.channel_lira_q0_common import (
    ATTACKS,
    DEFAULT_PROTOCOL,
    DEFAULT_PROTOCOL_LOCK,
    DEFAULT_SNAPSHOT,
    DEFAULT_TARGETS,
    build_features,
    calibration_threshold,
    content_ids,
    dense_counts,
    dense_z_expectations,
    read_targets,
    sha256,
    validate_protocol,
    z_covariance_features,
)
from reviewer_tools.qurift_qiskit_bridge import counts_to_z_expectations


ROOT = Path(__file__).resolve().parents[1]


class ChannelLiRAQ0ProtocolTests(unittest.TestCase):
    def test_locked_protocol_and_parent_are_hash_valid(self) -> None:
        protocol = validate_protocol()
        lock = DEFAULT_PROTOCOL_LOCK.read_text(encoding="utf-8").split()
        self.assertEqual(lock[0], sha256(DEFAULT_PROTOCOL))
        parent = ROOT / protocol["relationship_to_phase7"]["phase7_protocol"]
        self.assertEqual(
            sha256(parent),
            protocol["relationship_to_phase7"]["phase7_protocol_sha256"],
        )
        self.assertFalse(protocol["automatic_execution"])
        self.assertFalse(protocol["confirmatory_claim_allowed"])

    def test_manifest_has_independent_seeds_and_two_cells(self) -> None:
        rows = read_targets(DEFAULT_TARGETS)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({row["data_seed"] for row in rows}), 6)
        self.assertEqual(len({row["model_seed"] for row in rows}), 6)
        self.assertEqual(
            [int(row["mnist_disjoint_partition_id"]) for row in rows],
            list(range(6)),
        )
        self.assertEqual(
            {int(row["mnist_disjoint_partition_count"]) for row in rows}, {6}
        )
        self.assertEqual(
            {int(row["mnist_disjoint_partition_seed"]) for row in rows},
            {20260823},
        )
        self.assertEqual(
            {row["structural_cell_id"] for row in rows},
            {"eff_su2_r1_d2", "zz_r5_d6"},
        )
        for cell in {row["structural_cell_id"] for row in rows}:
            self.assertEqual(sum(row["structural_cell_id"] == cell for row in rows), 3)

    def test_protocol_rejects_changed_target_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "targets.csv"
            target.write_bytes(DEFAULT_TARGETS.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "target-manifest"):
                validate_protocol(
                    DEFAULT_PROTOCOL,
                    DEFAULT_PROTOCOL_LOCK,
                    target,
                    DEFAULT_SNAPSHOT,
                )


class ChannelLiRAQ0FeatureTests(unittest.TestCase):
    def test_content_identity_is_content_bound(self) -> None:
        inputs = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
        labels = np.asarray([0, 1])
        first = content_ids(inputs, labels)
        second = content_ids(inputs.copy(), labels.copy())
        self.assertTrue(np.array_equal(first, second))
        changed = inputs.copy()
        changed[0, 0, 0, 0] += 1
        self.assertNotEqual(first[0], content_ids(changed, labels)[0])

    def test_dense_counts_matches_bridge_z_convention(self) -> None:
        raw = [{"000000": 64, "000001": 32, "100000": 32}]
        dense = dense_counts(raw, n_wires=6, shots=128)
        observed = dense_z_expectations(dense)[0]
        expected = counts_to_z_expectations(raw[0], 6)
        np.testing.assert_allclose(observed, expected)

    def test_joint_covariance_detects_correlated_bits(self) -> None:
        counts = np.zeros((1, 4), dtype=int)
        counts[0, 0] = 50
        counts[0, 3] = 50
        covariance = z_covariance_features(counts)
        self.assertEqual(covariance.shape, (1, 1))
        self.assertAlmostEqual(float(covariance[0, 0]), 1.0)

    def test_feature_hierarchy_has_locked_dimensions(self) -> None:
        protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        records = 8
        classes = 4
        wires = 6
        states = 2**wires
        rng = np.random.default_rng(7)
        labels = np.arange(records) % classes

        def count_block(repetitions: int) -> np.ndarray:
            output = np.empty((repetitions, records, states), dtype=np.uint16)
            for repetition in range(repetitions):
                for index in range(records):
                    output[repetition, index] = rng.multinomial(
                        128, np.full(states, 1 / states)
                    )
            return output

        counts_a = count_block(10)
        counts_b = count_block(5)
        z_a = np.stack([dense_z_expectations(value) for value in counts_a])
        z_b = np.stack([dense_z_expectations(value) for value in counts_b])
        weight = rng.normal(size=(wires, classes))

        def probabilities(z: np.ndarray) -> np.ndarray:
            logits = z @ weight
            logits -= logits.max(axis=-1, keepdims=True)
            values = np.exp(logits)
            return values / values.sum(axis=-1, keepdims=True)

        exact_z = rng.uniform(-1, 1, size=(records, wires))
        payload = {
            "labels": labels,
            "counts_layout_a": counts_a,
            "counts_layout_b": counts_b,
            "z_layout_a": z_a,
            "z_layout_b": z_b,
            "probabilities_layout_a": probabilities(z_a),
            "probabilities_layout_b": probabilities(z_b),
            "exact_z": exact_z,
            "exact_probabilities": probabilities(exact_z),
        }
        features = build_features(payload, protocol)
        self.assertEqual(set(features) - {"loss_value"}, set(ATTACKS))
        self.assertEqual(features["loss_mia"].shape, (records, 1))
        self.assertGreater(
            features["paired_joint_probe"].shape[1],
            features["paired_marginal_probe"].shape[1],
        )
        self.assertGreater(
            features["fixed_joint"].shape[1],
            features["learned_mia"].shape[1],
        )
        self.assertTrue(all(np.isfinite(value).all() for value in features.values()))


class ChannelLiRAQ0AnalysisTests(unittest.TestCase):
    def test_calibration_threshold_never_exceeds_tied_fpr_budget(self) -> None:
        scores = np.asarray([0.9, 0.9, 0.8, *np.linspace(0.7, 0.0, 97)])
        threshold = calibration_threshold(scores, 0.01)
        self.assertTrue(np.isinf(threshold))
        scores = np.linspace(0.0, 1.0, 100)
        threshold = calibration_threshold(scores, 0.01)
        self.assertEqual(int((scores >= threshold).sum()), 1)

    def test_training_identities_are_averaged_once(self) -> None:
        features = np.asarray([[1.0], [3.0], [7.0]])
        membership = np.asarray([1, 1, 0])
        identities = np.asarray(["a", "a", "b"])
        values, labels, ids = aggregate_training_identities(
            features, membership, identities
        )
        np.testing.assert_allclose(values[:, 0], [2.0, 7.0])
        np.testing.assert_array_equal(labels, [1, 0])
        np.testing.assert_array_equal(ids, ["a", "b"])

    def test_training_identity_rejects_inconsistent_membership(self) -> None:
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            aggregate_training_identities(
                np.asarray([[1.0], [2.0]]),
                np.asarray([0, 1]),
                np.asarray(["same", "same"]),
            )

    def test_screening_gate_requires_every_component(self) -> None:
        protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        metrics = pd.DataFrame(
            {
                "attack": list(ATTACKS),
                "actual_victim_fpr_mean": [0.01] * len(ATTACKS),
            }
        )
        rows = []
        for contrast in (
            "paired_joint_minus_loss",
            "paired_joint_minus_learned",
            "paired_joint_minus_classical_stochastic",
            "fixed_joint_minus_learned",
            "paired_joint_minus_fixed_joint",
        ):
            rows.append(
                {
                    "contrast": contrast,
                    "auc_difference_mean": 0.02,
                    "auc_difference_positive_targets": 6,
                    "empirical_tpr_at_1pct_fpr_difference_mean": 0.02,
                    "empirical_tpr_at_1pct_fpr_difference_positive_targets": 6,
                    "loss_conditioned_auc_difference_mean": 0.02,
                    "loss_conditioned_auc_difference_positive_targets": 6,
                    "operational_tpr_difference_mean": 0.02,
                }
            )
        contrasts = pd.DataFrame(rows)
        self.assertTrue(screening_decision(metrics, contrasts, protocol)["screen_passed"])
        contrasts.loc[
            contrasts.contrast == "paired_joint_minus_loss", "auc_difference_mean"
        ] = 0.0
        self.assertFalse(screening_decision(metrics, contrasts, protocol)["screen_passed"])


if __name__ == "__main__":
    unittest.main()
