#!/usr/bin/env python3
"""Acquire and freeze the UCI WDBC dataset with a packaged-copy fallback."""
from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer

from qurift.satml_wdbc import (
    WDBC_CANONICAL_CONTENT_SHA256,
    WDBC_TARGET_COLUMN,
    sha256_gzip_content,
    write_wdbc_snapshot,
)


def sklearn_frame() -> tuple[pd.DataFrame, dict]:
    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.data.copy()
    frame.columns = [str(column).strip().replace(" ", "_") for column in frame.columns]
    frame[WDBC_TARGET_COLUMN] = bunch.target.astype(int).to_numpy()
    return frame, {
        "provider": "scikit-learn packaged copy of UCI WDBC",
        "sklearn_version": sklearn.__version__,
        "upstream": "UCI Machine Learning Repository dataset 17",
    }


def official_frame(cache_dir: Path) -> tuple[pd.DataFrame, dict]:
    url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "wdbc_uci_17.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".zip.tmp")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
            handle.write(response.read())
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as bundle:
        candidates = [name for name in bundle.namelist() if Path(name).name.lower() == "wdbc.data"]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one wdbc.data in UCI archive; observed {candidates}")
        raw = pd.read_csv(io.BytesIO(bundle.read(candidates[0])), header=None)
    reference = load_breast_cancer(as_frame=True)
    if raw.shape != (569, 32):
        raise RuntimeError(f"Unexpected official WDBC shape: {raw.shape}")
    frame = raw.iloc[:, 2:].copy()
    frame.columns = [str(column).strip().replace(" ", "_") for column in reference.feature_names]
    frame[WDBC_TARGET_COLUMN] = raw.iloc[:, 1].map({"M": 0, "B": 1}).astype(int)
    return frame, {
        "provider": "UCI Machine Learning Repository",
        "dataset_id": 17,
        "url": url,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/wdbc/wdbc.csv.gz"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/uci_cache"))
    parser.add_argument("--source", choices=["auto", "uci", "sklearn"], default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        parser.error(f"{args.out} already exists; pass --force to replace it")
    frame = source = None
    if args.source in {"auto", "uci"}:
        try:
            frame, source = official_frame(args.cache_dir)
        except Exception as error:
            if args.source == "uci":
                raise
            print(f"[WARN] UCI acquisition failed ({type(error).__name__}); using packaged copy.")
    if frame is None:
        frame, source = sklearn_frame()
    if frame.shape != (569, 31) or frame.isna().any().any():
        raise RuntimeError(f"Invalid canonical WDBC frame: {frame.shape}")
    manifest = write_wdbc_snapshot(frame, args.out, source)
    observed = sha256_gzip_content(args.out.resolve())
    if WDBC_CANONICAL_CONTENT_SHA256 and observed != WDBC_CANONICAL_CONTENT_SHA256:
        raise RuntimeError(
            f"Canonical WDBC checksum mismatch: expected {WDBC_CANONICAL_CONTENT_SHA256}, observed {observed}"
        )
    print(f"[OK] rows=569 features=30 content_sha256={observed}")
    print(f"[OK] data={args.out.resolve()} manifest={manifest.resolve()}")


if __name__ == "__main__":
    main()
