from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.channel_lira_noisy_reference_scaleup import (
    EXPECTED_CELL,
    EXPECTED_TARGETS,
    load_scaleup_cell,
    paired_contrasts,
    probability_features,
    read_targets,
)
from experiments.run_channel_lira_noisy_reference_scaleup import DEFAULT_TARGETS
from satml_tools.noisy_lira import (
    load_reference_cache,
    save_reference_cache,
)


class NoisyReferenceScaleupTests(unittest.TestCase):
    def test_frozen_manifest_has_three_target_seeds_in_one_cell(self) -> None:
        rows = read_targets(DEFAULT_TARGETS)
        self.assertEqual(tuple(row["target_id"] for row in rows), EXPECTED_TARGETS)
        self.assertEqual({row["structural_cell_id"] for row in rows}, {EXPECTED_CELL})
        self.assertEqual([int(row["model_seed"]) for row in rows], [43, 44, 45])

    def test_probability_features_match_frozen_learned_baseline_schema(self) -> None:
        probabilities = np.asarray([
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.6, 0.2, 0.1],
            [0.2, 0.1, 0.4, 0.3],
        ])
        labels = np.asarray([0, 1, 3])
        loss, features = probability_features(probabilities, labels)
        self.assertEqual(features.shape, (3, 9))
        np.testing.assert_allclose(loss, -np.log([0.7, 0.6, 0.3]))
        np.testing.assert_allclose(features[:, :4], probabilities)
        np.testing.assert_array_equal(features[:, -1], [1.0, 1.0, 0.0])

    def test_reference_cache_is_protocol_and_hash_validated(self) -> None:
        protocol = {
            "schema_version": 1,
            "structural_cell": EXPECTED_CELL,
            "candidate_fingerprint": "candidate-bank",
            "snapshot_manifest_sha256": "abc",
            "modes": ["ideal_shot", "noisy_shot"],
            "simulator_seeds": [0, 1],
            "shots": 128,
            "num_reference_models": 16,
            "transpiler_seed": 2026,
            "optimization_level": 1,
        }
        scores = {
            mode: np.arange(2 * 16 * 12, dtype=np.float32).reshape(2, 16, 12)
            for mode in protocol["modes"]
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache.npz"
            metadata = root / "cache.json"
            save_reference_cache(
                cache,
                metadata,
                scores=scores,
                sample_ids=[f"sample-{index}" for index in range(12)],
                metadata=protocol,
            )
            loaded = load_reference_cache(
                cache,
                metadata,
                expected=protocol,
                modes=protocol["modes"],
                simulator_seeds=protocol["simulator_seeds"],
                num_references=16,
                candidate_count=12,
            )
            self.assertIsNotNone(loaded)
            for mode in protocol["modes"]:
                np.testing.assert_array_equal(loaded[mode], scores[mode])
            cache.write_bytes(cache.read_bytes() + b"corrupt")
            with self.assertRaises(ValueError):
                load_reference_cache(
                    cache,
                    metadata,
                    expected=protocol,
                    modes=protocol["modes"],
                    simulator_seeds=protocol["simulator_seeds"],
                    num_references=16,
                    candidate_count=12,
                )

    def test_scaleup_loader_requires_aligned_probability_complete_payloads(self) -> None:
        candidate_count = 20
        sample_ids = np.asarray([f"sample-{index:02d}" for index in range(candidate_count)])
        labels = np.arange(candidate_count) % 4
        membership = np.arange(candidate_count) % 2
        base = np.full((candidate_count, 4), 0.1, dtype=np.float64)
        base[np.arange(candidate_count), labels] = 0.7
        observed = np.log(
            base[np.arange(candidate_count), labels]
            / (1.0 - base[np.arange(candidate_count), labels])
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_dir = root / "references"
            noisy_dir = root / "noisy"
            exact_dir = reference_dir / "sample_scores"
            exact_dir.mkdir(parents=True)
            for target_index, target_id in enumerate(EXPECTED_TARGETS):
                probabilities = base.copy()
                probabilities = np.roll(probabilities, target_index, axis=0)
                probabilities[np.arange(candidate_count), labels] = 0.7
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                true_probability = probabilities[np.arange(candidate_count), labels]
                exact_observed = np.log(true_probability / (1.0 - true_probability))
                np.savez_compressed(
                    exact_dir / f"{target_id}.npz",
                    sample_ids=sample_ids,
                    labels=labels,
                    membership=membership,
                    probabilities=probabilities,
                    observed_log_odds=exact_observed,
                )
            reference_root = reference_dir / "reference_models" / f"{EXPECTED_CELL}_wd0"
            reference_root.mkdir(parents=True)
            for reference_id in range(16):
                inclusion = np.asarray([
                    ((reference_id + candidate) % 16) < 8
                    for candidate in range(candidate_count)
                ])
                np.savez_compressed(
                    reference_root / f"reference_{reference_id:03d}.npz",
                    sample_ids=sample_ids,
                    scores=observed + 0.01 * reference_id,
                    inclusion=inclusion,
                    num_references=np.asarray(16),
                )
            (noisy_dir / "metadata").mkdir(parents=True)
            (noisy_dir / "sample_scores").mkdir(parents=True)
            metadata = {
                "shots": 128,
                "num_reference_models": 16,
                "modes": ["ideal_shot", "noisy_shot"],
                "simulator_seeds": [0, 1],
                "snapshot_manifest_sha256": "frozen-snapshot",
                "backend": {
                    "resolved_backend_name": "ibm_kingston",
                    "calibration_timestamp": "2026-08-21T21:16:41+08:00",
                },
            }
            for target_id in EXPECTED_TARGETS:
                (noisy_dir / "metadata" / f"{target_id}.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                for mode in ("ideal_shot", "noisy_shot"):
                    for simulator_seed in (0, 1):
                        shift = 0.005 * simulator_seed + (0.01 if mode == "noisy_shot" else 0.0)
                        served = observed + shift
                        payload = {
                            "sample_ids": sample_ids,
                            "labels": labels,
                            "membership": membership,
                            "probabilities": base,
                            "observed_log_odds": served,
                            "lira_online": served,
                            "lira_online_fixed_variance": served + 0.01,
                            "lira_offline": served - 0.01,
                            "lira_offline_fixed_variance": served + 0.02,
                        }
                        np.savez_compressed(
                            noisy_dir / "sample_scores" /
                            f"{target_id}_{mode}_sim{simulator_seed}.npz",
                            **payload,
                        )
            cell, matched, _, protocol = load_scaleup_cell(
                targets_path=DEFAULT_TARGETS,
                reference_dir=reference_dir,
                noisy_dir=noisy_dir,
                modes=["ideal_shot", "noisy_shot"],
                shots=128,
                simulator_seeds=[0, 1],
                num_references=16,
            )
            self.assertEqual(cell.reference_scores.shape, (16, candidate_count))
            self.assertEqual(
                cell.conditions[("noisy_shot", 128)].features.shape,
                (3, 2, candidate_count, 9),
            )
            self.assertEqual(
                matched[("noisy_shot", 128)]["lira_online"].shape,
                (3, 2, candidate_count),
            )
            self.assertEqual(protocol["snapshot_manifest_sha256"], "frozen-snapshot")

    def test_paired_contrasts_use_target_checkpoint_as_unit(self) -> None:
        attacks = {
            "matched_reference_lira_online_fixed_variance": (16, 0.60),
            "affine_channel_lira": (16, 0.62),
            "latent_lira_mismatched": (16, 0.55),
            "loss_mia": (0, 0.52),
            "target_crossfit_learned_mia": (0, 0.58),
        }
        rows = []
        for target_index, target_id in enumerate(EXPECTED_TARGETS):
            for attack, (count, auc) in attacks.items():
                row = {
                    "target_id": target_id,
                    "mode": "noisy_shot",
                    "shots": 128,
                    "attack": attack,
                    "reference_count": count,
                }
                for metric in (
                    "auc", "advantage", "tpr_at_0_1pct_fpr",
                    "tpr_at_1pct_fpr", "tpr_at_5pct_fpr",
                ):
                    row[f"{metric}_mean_over_simulator_seeds"] = auc + 0.01 * target_index
                rows.append(row)
        target, summary = paired_contrasts(rows, 16)
        self.assertEqual(len(target), 18)
        affine = next(
            row for row in summary
            if row["contrast"] == "affine_minus_matched_reference"
        )
        self.assertEqual(affine["n_targets"], 3)
        self.assertAlmostEqual(affine["auc_difference_mean"], 0.02)


if __name__ == "__main__":
    unittest.main()
