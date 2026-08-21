from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from experiments.channel_lira_circuit_pilot import CellData, ConditionData
from experiments.channel_lira_calibration_ablation import deterministic_subsets
from experiments.check_channel_lira_noisy_reference_readiness import (
    select_reference_path,
)
import experiments.channel_lira_transfer as transfer
from experiments.channel_lira_transfer import (
    evaluate_leave_cell_out,
    evaluate_leave_target_out,
    reference_bundles,
)


def synthetic_cell(
    name: str = "synthetic", *, intercept: float = 0.15, slope: float = 0.75
) -> CellData:
    candidates = 25
    target_ids = tuple(f"{name}_target_{index}" for index in range(3))
    sample_ids = np.asarray([f"sample_{index}" for index in range(candidates)])
    inclusion = np.vstack((
        np.ones(candidates, dtype=bool),
        np.ones(candidates, dtype=bool),
        np.zeros(candidates, dtype=bool),
        np.zeros(candidates, dtype=bool),
    ))
    candidate_offset = np.linspace(-0.25, 0.25, candidates)
    reference_scores = np.where(inclusion, 0.9, -0.9).astype(np.float64)
    reference_scores += np.asarray([-0.12, 0.12, -0.12, 0.12])[:, None]
    reference_scores += candidate_offset[None, :]
    memberships = np.vstack([
        (np.arange(candidates) + target) % 2 == 0 for target in range(3)
    ]).astype(np.int64)
    exact_scores = np.where(memberships == 1, 0.8, -0.8) + candidate_offset
    observed = np.stack([
        intercept + slope * exact_scores + 0.01,
        intercept + slope * exact_scores - 0.01,
    ], axis=1)
    losses = 1.0 - observed
    features = np.zeros((*observed.shape, 1), dtype=np.float64)
    condition = ConditionData(
        simulator_seeds=(10, 11),
        observed_scores=observed,
        losses=losses,
        features=features,
    )
    return CellData(
        name=name,
        target_ids=target_ids,
        sample_ids=sample_ids,
        reference_scores=reference_scores,
        inclusion=inclusion,
        memberships=memberships,
        exact_scores=exact_scores,
        exact_losses=np.zeros_like(exact_scores),
        exact_features=np.zeros((*exact_scores.shape, 1), dtype=np.float64),
        conditions={("noisy_shot", 128): condition},
        source_files=(),
    )


class TransferProtocolTests(unittest.TestCase):
    def test_readiness_prefers_new_complete_canonical_reference_pair(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "z_r1_d6" / "reference_000.npz"
            canonical = root / "z_r1_d6_wd0" / "reference_000.npz"
            legacy.parent.mkdir()
            canonical.parent.mkdir()
            legacy.touch()

            self.assertEqual(
                select_reference_path(root, "z_r1_d6", 0), legacy
            )
            canonical.touch()
            canonical.with_suffix(".pt").write_bytes(b"checkpoint")
            self.assertEqual(
                select_reference_path(root, "z_r1_d6", 0), canonical
            )

    def test_calibration_ablation_subset_schedule_is_unique_and_reproducible(self) -> None:
        pairs = deterministic_subsets(
            12, 2, requested_replicates=32, seed=23
        )
        repeated_pairs = deterministic_subsets(
            12, 2, requested_replicates=32, seed=23
        )
        quartets = deterministic_subsets(
            12, 4, requested_replicates=32, seed=29
        )

        self.assertEqual(pairs, repeated_pairs)
        self.assertEqual(len(pairs), 66)
        self.assertEqual(len(set(pairs)), 66)
        self.assertTrue(all(len(pair) == 2 for pair in pairs))
        self.assertEqual(len(quartets), 32)
        self.assertEqual(len(set(quartets)), 32)
        self.assertTrue(all(len(quartet) == 4 for quartet in quartets))

    def test_held_target_is_not_used_to_fit_its_channel(self) -> None:
        cell = synthetic_cell()
        bundles = reference_bundles(cell, [4], variance_shrinkage=0.15)
        block, diagnostics = evaluate_leave_target_out(
            cell,
            cell.conditions[("noisy_shot", 128)],
            bundles,
            mode="noisy_shot",
            shots=128,
            folds=5,
            noise_augmentation_draws=2,
            variance_shrinkage=0.15,
            seed=17,
        )
        self.assertEqual(len(diagnostics), 15)
        self.assertTrue(all(np.isfinite(values).all() for values in block.scores.values()))
        self.assertTrue(all(values.shape == (3, 2, 25) for values in block.scores.values()))

        original = cell.conditions[("noisy_shot", 128)]
        changed_observed = original.observed_scores.copy()
        changed_observed[0] = 100.0 + 7.0 * cell.exact_scores[0][None, :]
        changed = ConditionData(
            simulator_seeds=original.simulator_seeds,
            observed_scores=changed_observed,
            losses=original.losses,
            features=original.features,
        )
        changed_block, changed_diagnostics = evaluate_leave_target_out(
            cell,
            changed,
            bundles,
            mode="noisy_shot",
            shots=128,
            folds=5,
            noise_augmentation_draws=2,
            variance_shrinkage=0.15,
            seed=17,
        )
        original_fit = [
            (row["fold"], row["intercept"], row["slope"], row["scale"])
            for row in diagnostics if row["heldout_id"] == "synthetic_target_0"
        ]
        changed_fit = [
            (row["fold"], row["intercept"], row["slope"], row["scale"])
            for row in changed_diagnostics if row["heldout_id"] == "synthetic_target_0"
        ]
        self.assertEqual(original_fit, changed_fit)
        self.assertFalse(np.allclose(
            block.scores[("affine_channel_lira", 4)][0],
            changed_block.scores[("affine_channel_lira", 4)][0],
        ))

    def test_held_cell_cannot_change_channel_or_threshold_calibration(self) -> None:
        held = synthetic_cell("held", intercept=-0.4, slope=1.8)
        auxiliary = [
            synthetic_cell("aux_a", intercept=0.1, slope=0.7),
            synthetic_cell("aux_b", intercept=0.2, slope=0.8),
        ]
        cells = [held, *auxiliary]
        bundles = {
            cell.name: reference_bundles(cell, [4], variance_shrinkage=0.15)
            for cell in cells
        }

        def run(cell: CellData):
            thresholds = []
            original_threshold = transfer.conservative_threshold

            def record_threshold(values, nominal_fpr):
                value = original_threshold(values, nominal_fpr)
                thresholds.append((float(nominal_fpr), float(value)))
                return value

            with patch.object(
                transfer, "conservative_threshold", side_effect=record_threshold
            ):
                block, diagnostics = evaluate_leave_cell_out(
                    cell,
                    auxiliary,
                    bundles,
                    mode="noisy_shot",
                    shots=128,
                    folds=5,
                    noise_augmentation_draws=2,
                    variance_shrinkage=0.15,
                    seed=19,
                )
            return block, diagnostics, thresholds

        original_block, original_diagnostics, original_thresholds = run(held)
        original_condition = held.conditions[("noisy_shot", 128)]
        changed_condition = ConditionData(
            simulator_seeds=original_condition.simulator_seeds,
            observed_scores=100.0 + 9.0 * original_condition.observed_scores,
            losses=50.0 + 4.0 * original_condition.losses,
            features=original_condition.features,
        )
        changed_held = replace(
            held,
            exact_scores=-80.0 + 6.0 * held.exact_scores,
            exact_losses=30.0 + held.exact_losses,
            conditions={("noisy_shot", 128): changed_condition},
        )
        changed_block, changed_diagnostics, changed_thresholds = run(changed_held)

        original_fit = [
            (row["fold"], row["intercept"], row["slope"], row["scale"])
            for row in original_diagnostics
        ]
        changed_fit = [
            (row["fold"], row["intercept"], row["slope"], row["scale"])
            for row in changed_diagnostics
        ]
        self.assertEqual(original_fit, changed_fit)
        self.assertEqual(original_thresholds, changed_thresholds)
        self.assertFalse(np.allclose(
            original_block.scores[("affine_channel_lira", 4)],
            changed_block.scores[("affine_channel_lira", 4)],
        ))


if __name__ == "__main__":
    unittest.main()
