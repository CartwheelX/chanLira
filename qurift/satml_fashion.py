"""Seeded, balanced Fashion-MNIST construction and provenance."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torchvision


FASHION_CLASS_IDS = (0, 1, 3, 8)
FASHION_CLASS_NAMES = ("T-shirt/top", "Trouser", "Dress", "Bag")
FASHION_RAW_SHA256 = {
    "train-images-idx3-ubyte": "c59f468a2f672dc815687fe0f83887768d799fd8a3f3276145d20f83aa44d888",
    "train-labels-idx1-ubyte": "bad3541b69d912435c50bb6ba87bec294ff4f6a2e1246121d8633921760443d9",
    "t10k-images-idx3-ubyte": "5b4141f0afbad91edebe8549f8fcffe087ea10ca49f1dbef5c9a5cd8815ce37b",
    "t10k-labels-idx1-ubyte": "0402a96d92fd2663957122ceb108a494c5af83dab82d92729df917d7dec38c34",
}


def _index_hash(split: str, indices: np.ndarray) -> str:
    digest = hashlib.sha256(split.encode("utf-8"))
    digest.update(np.ascontiguousarray(indices.astype(np.int64)).tobytes())
    return digest.hexdigest()


def _validate_raw_files(root: Path) -> dict[str, str]:
    raw = root / "FashionMNIST" / "raw"
    observed: dict[str, str] = {}
    for name, expected in FASHION_RAW_SHA256.items():
        path = raw / name
        if not path.exists():
            raise FileNotFoundError(f"Fashion-MNIST raw source file is missing: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        if value != expected:
            raise RuntimeError(f"Fashion-MNIST checksum mismatch for {name}: {value}")
        observed[name] = value
    return observed


def build_fashion_mnist(
    mnist_class: Any,
    *,
    root: Path,
    n_train: int,
    n_valid: int,
    n_test: int,
    data_seed: int,
    class_ids: Sequence[int] = FASHION_CLASS_IDS,
    require_source_hashes: bool = True,
) -> tuple[Any, dict[str, Any]]:
    signature = inspect.signature(mnist_class)
    required = {"fashion", "split_seed"}
    missing = required - set(signature.parameters)
    if missing:
        raise RuntimeError(
            f"TorchQuantum MNIST wrapper lacks seeded Fashion support: missing {sorted(missing)}"
        )
    kwargs = {
        "root": str(root),
        "train_valid_split_ratio": [0.9, 0.1],
        "digits_of_interest": list(class_ids),
        "n_train_samples": int(n_train),
        "n_valid_samples": int(n_valid),
        "n_test_samples": int(n_test),
        "fashion": True,
        "split_seed": int(data_seed),
    }
    if "same_n_samples_each_class" in signature.parameters:
        kwargs["same_n_samples_each_class"] = True
    dataset = mnist_class(**kwargs)
    source_hashes = _validate_raw_files(Path(root)) if require_source_hashes else {}
    split_hashes = {}
    split_sizes = {}
    class_counts = {}
    expected_sizes = {"train": int(n_train), "valid": int(n_valid), "test": int(n_test)}
    for split in ("train", "valid", "test"):
        split_dataset = dataset[split]
        indices = torch.as_tensor(getattr(split_dataset, "source_indices", []), dtype=torch.long).numpy()
        if len(indices) != len(split_dataset):
            raise RuntimeError(f"Fashion {split} source-index provenance is incomplete")
        if len(split_dataset) != expected_sizes[split]:
            raise RuntimeError(
                f"Fashion {split} requested {expected_sizes[split]} records but obtained {len(split_dataset)}"
            )
        split_hashes[split] = _index_hash(split, indices)
        split_sizes[split] = int(len(indices))
        counts = np.zeros(len(class_ids), dtype=int)
        for index in range(len(split_dataset)):
            label = int(torch.as_tensor(split_dataset[index]["digit"]).item())
            counts[label] += 1
        if int(counts.max() - counts.min()) > 1:
            raise RuntimeError(f"Fashion {split} is not class balanced: {counts.tolist()}")
        class_counts[split] = counts.tolist()
    provenance = {
        "dataset": "fashion_mnist",
        "source": {
            "provider": "torchvision.datasets.FashionMNIST",
            "torchvision_version": torchvision.__version__,
            "upstream": "Zalando Research Fashion-MNIST",
            "raw_file_sha256": source_hashes,
        },
        "data_seed": int(data_seed),
        "split_seed": int(data_seed),
        "split_sizes": split_sizes,
        "split_index_sha256": split_hashes,
        "selected_original_class_ids": list(class_ids),
        "selected_class_names": [FASHION_CLASS_NAMES[FASHION_CLASS_IDS.index(value)] for value in class_ids],
        "remapped_labels": {str(original): new for new, original in enumerate(class_ids)},
        "class_counts": class_counts,
        "preprocessing_fit_split": "not applicable; fixed image transform",
        "preprocessing": [
            "ToTensor",
            "fixed Fashion-MNIST normalization mean=0.2860 std=0.3530",
            "adaptive average pool to 4x4 inside the target model",
        ],
    }
    return dataset, provenance
