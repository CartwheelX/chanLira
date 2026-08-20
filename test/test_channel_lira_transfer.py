from __future__ import annotations

import unittest

import numpy as np

from experiments.channel_lira_circuit_pilot import CellData, ConditionData
from experiments.channel_lira_transfer import (
    evaluate_leave_target_out,
    reference_bundles,
)


def synthetic_cell() -> CellData:
    candidates = 25
    target_ids = ("target_0", "target_1", "target_2")
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
        0.15 + 0.75 * exact_scores + 0.01,
        0.15 + 0.75 * exact_scores - 0.01,
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
        name="synthetic",
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
            for row in diagnostics if row["heldout_id"] == "target_0"
        ]
        changed_fit = [
            (row["fold"], row["intercept"], row["slope"], row["scale"])
            for row in changed_diagnostics if row["heldout_id"] == "target_0"
        ]
        self.assertEqual(original_fit, changed_fit)
        self.assertFalse(np.allclose(
            block.scores[("affine_channel_lira", 4)][0],
            changed_block.scores[("affine_channel_lira", 4)][0],
        ))


if __name__ == "__main__":
    unittest.main()
