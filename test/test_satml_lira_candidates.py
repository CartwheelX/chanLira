from __future__ import annotations

import unittest

import torch

from reviewer_tools.qurift_lira_attack import CandidateDataset, cell_id


class LiRACandidateTests(unittest.TestCase):
    def test_reference_bank_identity_includes_credit_block(self) -> None:
        base = {"structural_cell_id": "z_r1_d2", "weight_decay": 0.0}
        first = cell_id({**base, "block_id": "credit_b01"})
        second = cell_id({**base, "block_id": "credit_b02"})
        self.assertNotEqual(first, second)
        self.assertIn("credit_b01", first)

    def test_candidate_indices_address_selected_tensors(self) -> None:
        inputs = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        candidates = CandidateDataset(inputs, labels)
        subset = torch.utils.data.Subset(candidates, [1, 4])
        self.assertTrue(torch.equal(subset[0]["image"], inputs[1]))
        self.assertEqual(int(subset[1]["digit"]), int(labels[4]))


if __name__ == "__main__":
    unittest.main()
