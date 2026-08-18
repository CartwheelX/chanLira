from __future__ import annotations

import unittest

import pandas as pd

from satml_tools.build_fresh_selector_targets import build_fresh_targets, select_policies


class SelectorTests(unittest.TestCase):
    def test_selection_uses_validation_and_development_privacy(self) -> None:
        targets = []
        metrics = []
        attacks = []
        configs = (("z_r1_d2", "z", 1, 2, 0.80, 0.65), ("eff_r1_d2", "eff_su2", 1, 2, 0.79, 0.53))
        for block in range(1, 6):
            for cell, fm, reps, depth, accuracy, auc in configs:
                target = f"{cell}_b{block}"
                targets.append(
                    {"target_id": target, "block_id": f"b{block}", "structural_cell_id": cell,
                     "fm_kind": fm, "reps": reps, "depth": depth, "fm_ent": "linear",
                     "fm_op": "cx" if fm == "eff_su2" else "NA", "pad_mode": "wrap",
                     "data_seed": block, "model_seed": block + 10, "dataset": "credit_default"}
                )
                metrics.append({"target_id": target, "valid_acc": accuracy})
                attacks.append({"target_id": target, "attack": "loss", "auc": auc})
        target_frame = pd.DataFrame(targets)
        summary, decisions = select_policies(
            target_frame, pd.DataFrame(metrics), pd.DataFrame(attacks), accuracy_tolerance=0.02
        )
        self.assertEqual(decisions["utility_only"]["structural_cell_id"], "z_r1_d2")
        self.assertEqual(decisions["privacy_aware"]["structural_cell_id"], "eff_r1_d2")
        fresh = build_fresh_targets(target_frame, decisions, blocks=5, regularized_weight_decay=1e-3)
        self.assertEqual(len(fresh), 15)
        self.assertFalse(set(fresh.data_seed) & set(target_frame.data_seed))
        self.assertEqual(fresh.groupby("block_id").size().tolist(), [3] * 5)
        self.assertEqual(len(summary), 2)


if __name__ == "__main__":
    unittest.main()
