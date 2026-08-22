from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.run_channel_lira_noisy_reference_canary import (
    DEFAULT_TARGETS,
    canonical_cell,
    inspect_reference_bank,
    inspect_snapshot,
    read_target,
)


class NoisyReferenceCanaryTests(unittest.TestCase):
    def test_frozen_manifest_selects_compute_minimal_phase3_target(self) -> None:
        row = read_target(DEFAULT_TARGETS)
        self.assertEqual(row["target_id"], "MNIST_QNN_eff_su2_r1_d2_s43")
        self.assertEqual(canonical_cell(row), "eff_su2_r1_d2_wd0")

    def test_exact_reference_metadata_must_be_balanced_but_is_not_checkpoint_ready(self) -> None:
        row = read_target(DEFAULT_TARGETS)
        inclusion = np.asarray([
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ], dtype=np.uint8)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "reference_models" / canonical_cell(row)
            directory.mkdir(parents=True)
            for reference_id in range(4):
                np.savez_compressed(
                    directory / f"reference_{reference_id:03d}.npz",
                    reference_id=np.asarray(reference_id),
                    num_references=np.asarray(4),
                    structural_cell=np.asarray(canonical_cell(row)),
                    inclusion=inclusion[reference_id],
                    candidate_fingerprint=np.asarray("same-candidates"),
                    reference_seed=np.asarray(100 + reference_id),
                )
            status = inspect_reference_bank(row, root, 4)
            self.assertTrue(status["balanced_inclusion"])
            self.assertEqual(status["scores_ready"], 4)
            self.assertEqual(status["checkpoints_ready"], 0)
            self.assertFalse(status["ready"])

    def test_missing_snapshot_is_never_treated_as_noisy_ready(self) -> None:
        with TemporaryDirectory() as temporary:
            status = inspect_snapshot(Path(temporary) / "missing")
        self.assertFalse(status["ready"])
        self.assertFalse(status["manifest_exists"])


if __name__ == "__main__":
    unittest.main()
