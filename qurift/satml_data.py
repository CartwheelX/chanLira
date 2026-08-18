"""Leakage-safe tabular preprocessing for the SaTML experiments.

The functions in this module intentionally separate raw-data acquisition from
experimental preprocessing.  Acquisition produces a versioned local CSV.
Every experimental block then creates its own stratified partition and fits
all transformations using the member/training partition only.
"""
from __future__ import annotations

import hashlib
import gzip
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


CREDIT_DEFAULT_OPENML_ID = 42477
CREDIT_DEFAULT_UCI_ID = 350
CREDIT_DEFAULT_DOI = "10.24432/C55S3H"
CREDIT_TARGET_COLUMN = "default_payment_next_month"
CREDIT_CANONICAL_CONTENT_SHA256 = "dfb1570f223efb65c0084027570369bdff6cc291b8238b9adce17ab60da4ca83"

# These variables are integer-coded categories in the UCI data dictionary.
CREDIT_CATEGORICAL_COLUMNS = (
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_content(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_name(value: object) -> str:
    return str(value).strip().replace(" ", "_")


def normalise_credit_frame(features: pd.DataFrame, target: Iterable[Any]) -> pd.DataFrame:
    """Return the canonical numeric Credit-default frame.

    OpenML mirrors have used slightly different target spellings.  The local
    representation always uses ``default_payment_next_month`` and excludes the
    row identifier because it is not a predictive attribute.
    """
    frame = features.copy()
    frame.columns = [_normalise_name(column) for column in frame.columns]
    for identifier in ("ID", "id"):
        if identifier in frame.columns:
            frame = frame.drop(columns=[identifier])
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    labels = pd.to_numeric(pd.Series(target).reset_index(drop=True), errors="raise")
    labels = labels.astype(np.int64)
    unique = sorted(labels.unique().tolist())
    if unique != [0, 1]:
        raise ValueError(f"Credit target must be binary 0/1; observed {unique}")
    frame = frame.reset_index(drop=True)
    frame[CREDIT_TARGET_COLUMN] = labels.to_numpy()
    if frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"Credit data contain missing values in {missing}")
    return frame


def write_credit_snapshot(frame: pd.DataFrame, output: Path, source: Dict[str, Any]) -> Path:
    """Atomically write the canonical dataset and a checksum manifest."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if output.suffix == ".gz":
        # Do not let gzip timestamps or temporary filenames make the content
        # checksum differ across otherwise identical acquisitions.
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_handle:
                    frame.to_csv(text_handle, index=False, lineterminator="\n")
    else:
        frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(output)
    manifest = {
        "dataset": "UCI Default of Credit Card Clients",
        "uci_dataset_id": CREDIT_DEFAULT_UCI_ID,
        "openml_data_id": CREDIT_DEFAULT_OPENML_ID,
        "doi": CREDIT_DEFAULT_DOI,
        "license": "CC BY 4.0",
        "rows": int(len(frame)),
        "feature_columns": [column for column in frame if column != CREDIT_TARGET_COLUMN],
        "target_column": CREDIT_TARGET_COLUMN,
        "sha256": sha256_file(output),
        "canonical_csv_sha256": sha256_gzip_content(output) if output.suffix == ".gz" else sha256_file(output),
        "source": source,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def load_credit_snapshot(path: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Credit snapshot not found: {path}. Run satml_tools/fetch_credit_default.py first."
        )
    frame = pd.read_csv(path)
    if CREDIT_TARGET_COLUMN not in frame:
        raise ValueError(f"{path} has no {CREDIT_TARGET_COLUMN!r} column")
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = str(manifest.get("sha256", ""))
        observed = sha256_file(path)
        if expected and expected != observed:
            raise RuntimeError(
                f"Credit snapshot checksum mismatch: expected {expected}, observed {observed}"
            )
        expected_content = str(manifest.get("canonical_csv_sha256", ""))
        observed_content = sha256_gzip_content(path) if path.suffix == ".gz" else observed
        if expected_content and expected_content != observed_content:
            raise RuntimeError(
                f"Credit canonical-content checksum mismatch: expected {expected_content}, observed {observed_content}"
            )
    return frame, manifest


def _take_stratified(
    indices: np.ndarray,
    labels: np.ndarray,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        return np.empty(0, dtype=np.int64), indices.copy()
    if count >= len(indices):
        raise ValueError(f"Requested {count} records from a pool of {len(indices)}")
    selected, remaining = train_test_split(
        indices,
        train_size=int(count),
        random_state=int(seed),
        shuffle=True,
        stratify=labels[indices],
    )
    return np.sort(selected.astype(np.int64)), np.sort(remaining.astype(np.int64))


def stratified_credit_partition(
    labels: Sequence[int],
    *,
    n_train: int,
    n_valid: int,
    n_test: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Create disjoint, deterministic, stratified member/validation/test splits."""
    y = np.asarray(labels, dtype=np.int64)
    if min(n_train, n_valid, n_test) <= 0:
        raise ValueError("Credit train, validation, and test sizes must all be positive")
    if n_train + n_valid + n_test > len(y):
        raise ValueError("Requested Credit split sizes exceed the dataset size")
    pool = np.arange(len(y), dtype=np.int64)
    train, pool = _take_stratified(pool, y, n_train, seed)
    valid, pool = _take_stratified(pool, y, n_valid, seed + 1)
    test, _ = _take_stratified(pool, y, n_test, seed + 2)
    result = {"train": train, "valid": valid, "test": test}
    combined = np.concatenate(list(result.values()))
    if len(np.unique(combined)) != len(combined):
        raise AssertionError("Credit partitions overlap")
    return result


def _dense_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_credit_preprocessor(columns: Sequence[str], n_components: int) -> Pipeline:
    categorical = [name for name in CREDIT_CATEGORICAL_COLUMNS if name in columns]
    numeric = [name for name in columns if name not in categorical]
    if not numeric:
        raise ValueError("Credit preprocessing requires at least one numeric feature")
    features = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            ("categorical", _dense_one_hot_encoder(), categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return Pipeline(
        steps=[
            ("features", features),
            ("pca", PCA(n_components=int(n_components), svd_solver="full")),
            ("angle_range", MinMaxScaler(feature_range=(-1.0, 1.0), clip=True)),
        ]
    )


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


@dataclass
class CreditPreparedData:
    features: Dict[str, np.ndarray]
    labels: Dict[str, np.ndarray]
    indices: Dict[str, np.ndarray]
    preprocessor: Pipeline
    provenance: Dict[str, Any]


def prepare_credit_default(
    path: Path,
    *,
    n_train: int,
    n_valid: int,
    n_test: int,
    data_seed: int,
    n_components: int,
) -> CreditPreparedData:
    """Partition and transform Credit-default without evaluation-set leakage."""
    frame, source_manifest = load_credit_snapshot(path)
    labels_all = frame.pop(CREDIT_TARGET_COLUMN).to_numpy(dtype=np.int64)
    partitions = stratified_credit_partition(
        labels_all,
        n_train=n_train,
        n_valid=n_valid,
        n_test=n_test,
        seed=data_seed,
    )
    preprocessor = build_credit_preprocessor(frame.columns.tolist(), n_components)
    train_frame = frame.iloc[partitions["train"]]
    preprocessor.fit(train_frame)
    transformed = {
        split: preprocessor.transform(frame.iloc[index]).astype(np.float32)
        for split, index in partitions.items()
    }
    split_labels = {split: labels_all[index] for split, index in partitions.items()}
    pca = preprocessor.named_steps["pca"]
    index_hashes = {split: _array_hash(index) for split, index in partitions.items()}
    provenance = {
        "dataset": "credit_default",
        "source": source_manifest,
        "data_seed": int(data_seed),
        "split_sizes": {split: int(len(index)) for split, index in partitions.items()},
        "split_index_sha256": index_hashes,
        "preprocessing_fit_split": "train/member only",
        "preprocessing": [
            "standardize numeric features",
            "one-hot encode categorical features",
            f"PCA({int(n_components)})",
            "min-max map PCA components to [-1, 1] with evaluation clipping",
        ],
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "feature_min": {split: values.min(axis=0).tolist() for split, values in transformed.items()},
        "feature_max": {split: values.max(axis=0).tolist() for split, values in transformed.items()},
    }
    return CreditPreparedData(transformed, split_labels, partitions, preprocessor, provenance)
