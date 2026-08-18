from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from qurift.satml_data import (
    CREDIT_TARGET_COLUMN,
    prepare_credit_default,
    sha256_file,
    stratified_credit_partition,
    write_credit_snapshot,
)


class CreditPartitionTests(unittest.TestCase):
    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "LIMIT_BAL": [1000, 2000, 3000, 4000],
                "AGE": [20, 30, 40, 50],
                "SEX": [1, 2, 1, 2],
                CREDIT_TARGET_COLUMN: [0, 1, 0, 1],
            }
        )

    def test_gzip_snapshot_is_byte_deterministic(self) -> None:
        frame = self._frame()
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv.gz"
            second = Path(directory) / "second.csv.gz"
            write_credit_snapshot(frame, first, {"provider": "test"})
            write_credit_snapshot(frame, second, {"provider": "test"})
            self.assertEqual(sha256_file(first), sha256_file(second))

    def test_partition_is_deterministic_disjoint_and_stratified(self) -> None:
        labels = np.tile([0, 0, 0, 1], 100)
        first = stratified_credit_partition(
            labels, n_train=80, n_valid=60, n_test=100, seed=17
        )
        second = stratified_credit_partition(
            labels, n_train=80, n_valid=60, n_test=100, seed=17
        )
        for split in first:
            np.testing.assert_array_equal(first[split], second[split])
        joined = np.concatenate(list(first.values()))
        self.assertEqual(len(joined), len(np.unique(joined)))
        for index in first.values():
            self.assertAlmostEqual(float(labels[index].mean()), 0.25, delta=0.03)

    def test_preprocessor_is_fit_only_on_training_records(self) -> None:
        rows = 500
        rng = np.random.default_rng(9)
        frame = pd.DataFrame(
            {
                "LIMIT_BAL": rng.normal(size=rows),
                "AGE": rng.normal(size=rows),
                "SEX": np.tile([1, 2], rows // 2),
                "EDUCATION": np.tile([1, 2, 3, 4, 1], rows // 5),
                CREDIT_TARGET_COLUMN: np.tile([0, 0, 0, 1, 0], rows // 5),
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "credit.csv.gz"
            write_credit_snapshot(frame, path, {"provider": "unit-test"})
            prepared = prepare_credit_default(
                path,
                n_train=100,
                n_valid=100,
                n_test=200,
                data_seed=5,
                n_components=2,
            )
        self.assertEqual(prepared.features["train"].shape, (100, 2))
        self.assertEqual(prepared.features["test"].shape, (200, 2))
        self.assertEqual(
            prepared.provenance["preprocessing_fit_split"], "train/member only"
        )
        self.assertLessEqual(max(value.max() for value in prepared.features.values()), 1.0)
        self.assertGreaterEqual(min(value.min() for value in prepared.features.values()), -1.0)


if __name__ == "__main__":
    unittest.main()
