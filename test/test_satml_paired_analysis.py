from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from satml_tools.analyze_paired_factorial import analyze


class PairedAnalysisTests(unittest.TestCase):
    def test_recovers_prespecified_additive_effects(self) -> None:
        targets = []
        metrics = []
        attacks = []
        for block in range(1, 6):
            for fm, fm_effect in (("eff_su2", 0.0), ("z", 0.03), ("zz", 0.05)):
                for reps in (1, 5):
                    for depth in (2, 6):
                        target = f"t{block}_{fm}_{reps}_{depth}"
                        targets.append(
                            {"target_id": target, "block_id": f"b{block}", "fm_kind": fm,
                             "reps": reps, "depth": depth, "data_seed": block, "model_seed": block + 10}
                        )
                        value = 0.5 + block * 0.001 + fm_effect + (0.04 if reps == 5 else 0) + (0.02 if depth == 6 else 0)
                        metrics.append({"target_id": target, "test_acc": value, "valid_acc": value, "gap": value})
                        attacks.append({"target_id": target, "attack": "loss", "auc": value})
        summary, blocks, regression = analyze(
            pd.DataFrame(targets), pd.DataFrame(metrics), [pd.DataFrame(attacks)], bootstrap=500, seed=7
        )
        row = summary[(summary.attack == "loss") & (summary.contrast == "5 - 1")].iloc[0]
        self.assertAlmostEqual(row.mean_difference, 0.04, places=10)
        row = summary[(summary.attack == "loss") & (summary.contrast == "ZZ - EfficientSU2")].iloc[0]
        self.assertAlmostEqual(row.mean_difference, 0.05, places=10)
        self.assertEqual(row.n_independent_blocks, 5)
        self.assertTrue((blocks.within_block_pairs > 0).all())
        coefficient = regression[(regression.attack == "loss") & (regression.term == "reps_5_minus_1")].iloc[0]
        self.assertAlmostEqual(coefficient.coefficient, 0.04, places=10)

    def test_fixed_depth_targeted_design_omits_unidentifiable_depth_effect(self) -> None:
        targets = []
        metrics = []
        attacks = []
        for block in range(1, 6):
            for fm, fm_effect in (("eff_su2", 0.0), ("z", 0.02), ("zz", 0.03)):
                for reps in (1, 5):
                    target = f"w{block}_{fm}_{reps}"
                    targets.append(
                        {"target_id": target, "block_id": f"b{block}", "fm_kind": fm,
                         "reps": reps, "depth": 2, "data_seed": block, "model_seed": block + 10}
                    )
                    value = 0.5 + fm_effect + (0.04 if reps == 5 else 0)
                    metrics.append({"target_id": target, "test_acc": value, "gap": value})
                    attacks.append({"target_id": target, "attack": "loss", "auc": value})
        summary, _, regression = analyze(
            pd.DataFrame(targets), pd.DataFrame(metrics), [pd.DataFrame(attacks)], bootstrap=100, seed=3
        )
        self.assertNotIn("depth", set(summary.factor))
        self.assertNotIn("depth_6_minus_2", set(regression.term))
        repetition = summary[(summary.attack == "loss") & (summary.factor == "repetitions")].iloc[0]
        self.assertAlmostEqual(repetition.mean_difference, 0.04, places=10)


if __name__ == "__main__":
    unittest.main()
