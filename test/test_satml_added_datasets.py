from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer

from qurift.satml_fashion import build_fashion_mnist
from qurift.satml_wdbc import (
    WDBC_CANONICAL_CONTENT_SHA256,
    WDBC_TARGET_COLUMN,
    prepare_wdbc,
    sha256_file,
    sha256_gzip_content,
    write_wdbc_snapshot,
)
from satml_tools.build_added_dataset_targets import fashion_rows, wdbc_rows
from satml_tools.validate_added_datasets import validate_design
from torchquantum.dataset.mnist import MNIST


class FakeFashionMNIST:
    def __init__(self, root, train=True, download=False, transform=None):
        del root, download
        repeats = 100 if train else 20
        labels = np.tile(np.array([0, 1, 3, 8], dtype=np.int64), repeats)
        self.targets = torch.tensor(labels)
        self.data = torch.arange(len(labels) * 28 * 28, dtype=torch.int64).reshape(-1, 28, 28).remainder(256).byte()
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = self.data[index].float().unsqueeze(0) / 255.0
        return image, int(self.targets[index])


def canonical_wdbc_frame() -> pd.DataFrame:
    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.data.copy()
    frame.columns = [str(column).strip().replace(" ", "_") for column in frame]
    frame[WDBC_TARGET_COLUMN] = bunch.target.astype(int).to_numpy()
    return frame


class SaTMLAddedDatasetTests(unittest.TestCase):
    def test_wdbc_snapshot_and_preprocessing_are_deterministic_and_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv.gz"
            second = root / "second.csv.gz"
            frame = canonical_wdbc_frame()
            source = {"provider": "unit-test"}
            write_wdbc_snapshot(frame, first, source)
            write_wdbc_snapshot(frame, second, source)
            self.assertEqual(sha256_file(first), sha256_file(second))
            self.assertEqual(sha256_gzip_content(first), WDBC_CANONICAL_CONTENT_SHA256)

            prepared = prepare_wdbc(
                first, n_train=160, n_valid=80, n_test=329, data_seed=80261, n_components=6
            )
            repeated = prepare_wdbc(
                first, n_train=160, n_valid=80, n_test=329, data_seed=80261, n_components=6
            )
            changed = prepare_wdbc(
                first, n_train=160, n_valid=80, n_test=329, data_seed=80262, n_components=6
            )
            self.assertEqual({key: len(value) for key, value in prepared.indices.items()}, {"train": 160, "valid": 80, "test": 329})
            self.assertTrue(np.array_equal(prepared.indices["train"], repeated.indices["train"]))
            self.assertFalse(np.array_equal(prepared.indices["train"], changed.indices["train"]))
            self.assertEqual(len(np.unique(np.concatenate(list(prepared.indices.values())))), 569)
            train_raw = frame.drop(columns=[WDBC_TARGET_COLUMN]).iloc[prepared.indices["train"]].to_numpy()
            fitted_mean = prepared.preprocessor.named_steps["standardize"].mean_
            np.testing.assert_allclose(fitted_mean, train_raw.mean(axis=0))
            for values in prepared.features.values():
                self.assertTrue(np.isfinite(values).all())
                self.assertLessEqual(float(values.max()), 1.000001)
                self.assertGreaterEqual(float(values.min()), -1.000001)

    def test_fashion_blocks_change_all_splits_but_remain_balanced(self) -> None:
        with patch("torchquantum.dataset.mnist.datasets.FashionMNIST", FakeFashionMNIST):
            first, provenance_one = build_fashion_mnist(
                MNIST, root=Path("unused"), n_train=40, n_valid=20, n_test=40,
                data_seed=60261, require_source_hashes=False,
            )
            _, provenance_repeat = build_fashion_mnist(
                MNIST, root=Path("unused"), n_train=40, n_valid=20, n_test=40,
                data_seed=60261, require_source_hashes=False,
            )
            _, provenance_two = build_fashion_mnist(
                MNIST, root=Path("unused"), n_train=40, n_valid=20, n_test=40,
                data_seed=60262, require_source_hashes=False,
            )
        self.assertEqual(provenance_one["split_index_sha256"], provenance_repeat["split_index_sha256"])
        for split in ("train", "valid", "test"):
            self.assertNotEqual(
                provenance_one["split_index_sha256"][split],
                provenance_two["split_index_sha256"][split],
            )
            counts = provenance_one["class_counts"][split]
            self.assertLessEqual(max(counts) - min(counts), 1)
        train_indices = set(first["train"].source_indices.tolist())
        valid_indices = set(first["valid"].source_indices.tolist())
        self.assertFalse(train_indices & valid_indices)

    def test_added_manifests_and_fixed_depth_analysis_design(self) -> None:
        fashion = pd.DataFrame(fashion_rows())
        wdbc = pd.DataFrame(wdbc_rows())
        self.assertEqual(len(fashion), 60)
        self.assertEqual(len(wdbc), 30)
        self.assertTrue(all(item["passed"] for item in validate_design(fashion, "fashion_mnist")))
        self.assertTrue(all(item["passed"] for item in validate_design(wdbc, "breast_cancer_wdbc")))


if __name__ == "__main__":
    unittest.main()
