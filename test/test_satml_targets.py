from __future__ import annotations

import unittest

import pandas as pd

from satml_tools.build_satml_targets import factorial_rows, geometry_rows, scaling_rows, validate
from satml_tools.validate_satml_study import validate_design


class SaTMLTargetTests(unittest.TestCase):
    def test_factorial_has_12_paired_cells_per_block(self) -> None:
        frame = pd.DataFrame(factorial_rows(8))
        validate(frame, 8, 12)
        self.assertEqual(len(frame), 96)
        self.assertEqual(frame["data_seed"].nunique(), 8)
        self.assertEqual(frame["model_seed"].nunique(), 8)
        checks = validate_design(frame, 8)
        self.assertTrue(all(check["passed"] for check in checks), checks)

    def test_scaling_adds_only_nonbaseline_scales(self) -> None:
        frame = pd.DataFrame(scaling_rows(5))
        validate(frame, 5, 6)
        self.assertEqual(len(frame), 60)
        self.assertEqual(set(frame["feature_angle_scale"]), {0.5, 2.0})
        self.assertEqual(set(frame["depth"]), {2})

    def test_geometry_covers_feature_maps_and_repetition(self) -> None:
        frame = pd.DataFrame(geometry_rows())
        self.assertEqual(len(frame), 6)
        self.assertEqual(set(frame.fm_kind), {"z", "zz", "eff_su2"})
        self.assertEqual(set(frame.reps), {1, 5})


if __name__ == "__main__":
    unittest.main()
