from __future__ import annotations

import unittest

import pandas as pd

from satml_tools.analyze_capacity_controls import analyze_capacity


class CapacityControlTests(unittest.TestCase):
    def test_repetition_changes_gates_not_trainable_capacity(self) -> None:
        targets, metrics = [], []
        for block in range(2):
            for fm_index, fm in enumerate(("eff_su2", "z", "zz")):
                for reps in (1, 5):
                    for depth in (2, 6):
                        target = f"{block}_{fm}_{reps}_{depth}"
                        targets.append({"target_id": target, "block_id": block, "fm_kind": fm, "reps": reps, "depth": depth})
                        metrics.append(
                            {"target_id": target, "resource_trainable_parameters_total": 100 + depth,
                             "resource_quantum_gate_count_total": depth * 10 + reps * (fm_index + 1)}
                        )
        summary, contrasts, checks = analyze_capacity(pd.DataFrame(targets), pd.DataFrame(metrics))
        self.assertEqual(len(summary), 12)
        self.assertEqual(len(contrasts), 12)
        self.assertTrue(checks["passed"], checks)
        self.assertTrue((contrasts.trainable_parameter_difference == 0).all())
        self.assertTrue((contrasts.quantum_gate_count_difference > 0).all())


if __name__ == "__main__":
    unittest.main()
