"""Leakage-safe Breast Cancer Wisconsin Diagnostic preprocessing for SaTML."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


WDBC_UCI_ID = 17
WDBC_DOI = "10.24432/C5DW2B"
WDBC_TARGET_COLUMN = "diagnosis"
WDBC_CANONICAL_CONTENT_SHA256 = "ec5134d1f4db4e0accdbb8705285cc335eabf53785c06d4f0e75126a84c7cefc"


def _sha256_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def sha256_gzip_content(path: Path) -> str:
    with gzip.open(path, "rb") as handle:
        return _sha256_stream(handle)


def write_wdbc_snapshot(frame: pd.DataFrame, output: Path, source: Dict[str, Any]) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    csv_options = {"index": False, "lineterminator": "\n", "float_format": "%.17g"}
    if output.suffix == ".gz":
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_handle:
                    frame.to_csv(text_handle, **csv_options)
    else:
        frame.to_csv(temporary, **csv_options)
    temporary.replace(output)
    manifest = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "uci_dataset_id": WDBC_UCI_ID,
        "doi": WDBC_DOI,
        "license": "CC BY 4.0",
        "rows": int(len(frame)),
        "feature_columns": [column for column in frame if column != WDBC_TARGET_COLUMN],
        "target_column": WDBC_TARGET_COLUMN,
        "target_encoding": {"0": "malignant", "1": "benign"},
        "sha256": sha256_file(output),
        "canonical_csv_sha256": sha256_gzip_content(output) if output.suffix == ".gz" else sha256_file(output),
        "source": source,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def load_wdbc_snapshot(path: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"WDBC snapshot not found: {path}. Run satml_tools/fetch_wdbc.py first.")
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"WDBC manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_file = sha256_file(path)
    if observed_file != str(manifest.get("sha256", "")):
        raise RuntimeError("WDBC compressed snapshot checksum mismatch")
    observed_content = sha256_gzip_content(path) if path.suffix == ".gz" else observed_file
    if observed_content != str(manifest.get("canonical_csv_sha256", "")):
        raise RuntimeError("WDBC canonical CSV checksum mismatch")
    if WDBC_CANONICAL_CONTENT_SHA256 and observed_content != WDBC_CANONICAL_CONTENT_SHA256:
        raise RuntimeError(
            f"WDBC canonical content differs from the frozen protocol: {observed_content}"
        )
    frame = pd.read_csv(path)
    if frame.shape != (569, 31) or WDBC_TARGET_COLUMN not in frame:
        raise ValueError(f"Unexpected WDBC shape/columns: {frame.shape}")
    if sorted(frame[WDBC_TARGET_COLUMN].astype(int).unique().tolist()) != [0, 1]:
        raise ValueError("WDBC target is not binary 0/1")
    if frame.isna().any().any():
        raise ValueError("WDBC snapshot contains missing values")
    return frame, manifest


def stratified_wdbc_partition(
    labels: Sequence[int], *, n_train: int, n_valid: int, n_test: int, seed: int
) -> Dict[str, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    if min(n_train, n_valid, n_test) <= 0 or n_train + n_valid + n_test > len(y):
        raise ValueError("Invalid WDBC split sizes")
    all_indices = np.arange(len(y), dtype=np.int64)
    train, pool = train_test_split(
        all_indices, train_size=n_train, random_state=seed, shuffle=True, stratify=y
    )
    valid, pool = train_test_split(
        pool, train_size=n_valid, random_state=seed + 1, shuffle=True, stratify=y[pool]
    )
    if n_test == len(pool):
        test = pool
    else:
        test, _ = train_test_split(
            pool, train_size=n_test, random_state=seed + 2, shuffle=True, stratify=y[pool]
        )
    result = {"train": np.sort(train), "valid": np.sort(valid), "test": np.sort(test)}
    combined = np.concatenate(list(result.values()))
    if len(combined) != len(np.unique(combined)):
        raise AssertionError("WDBC partitions overlap")
    return result


def build_wdbc_preprocessor(n_components: int) -> Pipeline:
    return Pipeline(
        [
            ("standardize", StandardScaler()),
            ("pca", PCA(n_components=int(n_components), svd_solver="full")),
            ("angle_range", MinMaxScaler(feature_range=(-1.0, 1.0), clip=True)),
        ]
    )


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


@dataclass
class WDBCPreparedData:
    features: Dict[str, np.ndarray]
    labels: Dict[str, np.ndarray]
    indices: Dict[str, np.ndarray]
    preprocessor: Pipeline
    provenance: Dict[str, Any]


def prepare_wdbc(
    path: Path, *, n_train: int, n_valid: int, n_test: int, data_seed: int, n_components: int
) -> WDBCPreparedData:
    frame, source_manifest = load_wdbc_snapshot(path)
    labels_all = frame.pop(WDBC_TARGET_COLUMN).to_numpy(np.int64)
    partitions = stratified_wdbc_partition(
        labels_all, n_train=n_train, n_valid=n_valid, n_test=n_test, seed=data_seed
    )
    preprocessor = build_wdbc_preprocessor(n_components)
    preprocessor.fit(frame.iloc[partitions["train"]])
    transformed = {
        split: preprocessor.transform(frame.iloc[index]).astype(np.float32)
        for split, index in partitions.items()
    }
    labels = {split: labels_all[index] for split, index in partitions.items()}
    pca = preprocessor.named_steps["pca"]
    provenance = {
        "dataset": "breast_cancer_wdbc",
        "source": source_manifest,
        "data_seed": int(data_seed),
        "split_sizes": {split: int(len(index)) for split, index in partitions.items()},
        "split_index_sha256": {split: _array_hash(index) for split, index in partitions.items()},
        "class_counts": {
            split: np.bincount(values, minlength=2).astype(int).tolist()
            for split, values in labels.items()
        },
        "preprocessing_fit_split": "train/member only",
        "preprocessing": [
            "standardize all 30 numeric features",
            f"PCA({int(n_components)})",
            "min-max map PCA components to [-1, 1] with evaluation clipping",
        ],
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "feature_min": {split: values.min(axis=0).tolist() for split, values in transformed.items()},
        "feature_max": {split: values.max(axis=0).tolist() for split, values in transformed.items()},
    }
    return WDBCPreparedData(transformed, labels, partitions, preprocessor, provenance)
