from __future__ import annotations

import unittest

import pandas as pd

from satml_tools.analyze_mechanistic_pathway import (
    block_bootstrap_target_correlations,
    explanatory_regressions,
    geometry_configuration_links,
    prepare_target_frame,
)


class MechanisticPathwayTests(unittest.TestCase):
    def test_uses_configuration_means_and_independent_block_resampling(self) -> None:
        targets, metrics, attacks = [], [], []
        for block in range(4):
            for fm_index, fm in enumerate(("eff_su2", "z", "zz")):
                for reps in (1, 5):
                    for depth in (2, 6):
                        target_id = f"b{block}_{fm}_{reps}_{depth}"
                        gap = 0.01 * reps + 0.004 * depth + 0.003 * fm_index + 0.001 * block
                        targets.append(
                            {"target_id": target_id, "block_id": f"b{block}", "fm_kind": fm,
                             "reps": reps, "depth": depth}
                        )
                        metrics.append(
                            {"target_id": target_id, "gap": gap, "train_loss": 0.2,
                             "test_loss": 0.2 + gap * 1.5}
                        )
                        attacks.append({"target_id": target_id, "attack": "loss", "auc": 0.5 + gap})
        geometry = []
        for seed in range(4):
            for fm_index, fm in enumerate(("eff_su2", "z", "zz")):
                for reps in (1, 5):
                    value = 0.02 * reps + 0.01 * fm_index + 0.001 * seed
                    geometry.append(
                        {"data_seed": seed, "fm_kind": fm, "reps": reps,
                         "class_similarity_gap": value, "kernel_label_alignment": value,
                         "effective_rank": value, "mmd2_train_test": value,
                         "within_class_similarity": value, "between_class_similarity": value}
                    )
        target = prepare_target_frame(
            pd.DataFrame(targets), pd.DataFrame(metrics), pd.DataFrame(attacks)
        )
        target_links = block_bootstrap_target_correlations(target, replicates=100, seed=7)
        self.assertEqual(len(target_links), 2)
        self.assertTrue((target_links.n_independent_target_blocks == 4).all())
        self.assertGreater(target_links.spearman.min(), 0.99)
        config, geometry_links = geometry_configuration_links(
            target, pd.DataFrame(geometry), replicates=100, seed=8
        )
        self.assertEqual(len(config), 6)
        self.assertEqual(len(geometry_links), 6)
        regressions = explanatory_regressions(target)
        self.assertEqual(set(regressions.model), {
            "total_structural_association", "plus_accuracy_gap", "plus_loss_gap", "plus_both_gaps"
        })
        self.assertTrue((regressions.n_blocks == 4).all())
        fixed_depth = explanatory_regressions(target[target.depth.eq(2)].copy())
        self.assertNotIn("depth_6_minus_2", set(fixed_depth.term))
        self.assertIn("reps_5_minus_1", set(fixed_depth.term))


if __name__ == "__main__":
    unittest.main()
