#!/usr/bin/env python3
"""Fetch and pin the UCI Credit-default dataset through OpenML ID 42477."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from qurift.satml_data import (
    CREDIT_CANONICAL_CONTENT_SHA256,
    CREDIT_DEFAULT_OPENML_ID,
    normalise_credit_frame,
    write_credit_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/credit_default/credit_default.csv.gz"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/openml_cache"))
    parser.add_argument("--source", choices=["auto", "uci", "openml", "mirror"], default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        parser.error(f"{args.out} already exists; pass --force to replace it")
    frame = None
    source = None
    if args.source in {"auto", "openml"}:
        try:
            bunch = fetch_openml(
                data_id=CREDIT_DEFAULT_OPENML_ID,
                as_frame=True,
                data_home=str(args.cache_dir),
            )
            frame = normalise_credit_frame(bunch.data, bunch.target)
            source = {
                "provider": "OpenML",
                "data_id": CREDIT_DEFAULT_OPENML_ID,
                "details": dict(getattr(bunch, "details", {}) or {}),
            }
        except Exception as exc:
            if args.source == "openml":
                raise
            print(f"[WARN] OpenML acquisition failed ({type(exc).__name__}); trying UCI.")

    if frame is None and args.source in {"auto", "uci"}:
        try:
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            archive = args.cache_dir / "uci_credit_default_350.zip"
            url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
            if not archive.exists():
                temporary = archive.with_suffix(".zip.tmp")
                with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
                temporary.replace(archive)
            with zipfile.ZipFile(archive) as bundle:
                names = [name for name in bundle.namelist() if name.lower().endswith(".xls")]
                if len(names) != 1:
                    raise RuntimeError(f"Expected one XLS file in UCI archive; observed {names}")
                extracted = args.cache_dir / Path(names[0]).name
                if not extracted.exists():
                    with bundle.open(names[0]) as source_handle, extracted.open("wb") as target_handle:
                        shutil.copyfileobj(source_handle, target_handle)
            try:
                raw = pd.read_excel(extracted, header=1, engine="xlrd")
            except ImportError as exc:
                raise RuntimeError(
                    "Reading the official UCI XLS requires xlrd. Install requirements-satml.txt."
                ) from exc
            target_candidates = [
                column for column in raw.columns
                if str(column).strip().lower().replace(" ", "_").replace(".", "_")
                == "default_payment_next_month"
            ]
            if len(target_candidates) != 1:
                raise RuntimeError(f"Could not identify UCI target column in {list(raw.columns)}")
            target_column = target_candidates[0]
            frame = normalise_credit_frame(raw.drop(columns=[target_column]), raw[target_column])
            source = {
                "provider": "UCI Machine Learning Repository",
                "dataset_id": 350,
                "url": url,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        except Exception as exc:
            if args.source == "uci":
                raise
            print(f"[WARN] UCI acquisition failed ({type(exc).__name__}); trying pinned mirror.")

    if frame is None:
        commit = "585f77c33dff5a14dac8e5396bec24820d3db2f8"
        url = (
            "https://raw.githubusercontent.com/MatteoM95/"
            f"Default-of-Credit-Card-Clients-Dataset-Analisys/{commit}/"
            "dataset/default_of_credit_card_clients.csv"
        )
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.cache_dir / f"credit_default_mirror_{commit}.csv"
        if not csv_path.exists():
            temporary = csv_path.with_suffix(".csv.tmp")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            temporary.replace(csv_path)
        raw = pd.read_csv(csv_path)
        candidates = [
            column for column in raw
            if str(column).strip().lower().replace(" ", "_").replace(".", "_")
            == "default_payment_next_month"
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Could not identify mirror target column in {list(raw.columns)}")
        frame = normalise_credit_frame(raw.drop(columns=[candidates[0]]), raw[candidates[0]])
        source = {
            "provider": "commit-pinned GitHub mirror of UCI dataset 350",
            "upstream_repository": "MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys",
            "commit": commit,
            "url": url,
            "raw_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        }

    if len(frame) != 30000 or len(frame.columns) - 1 != 23:
        raise RuntimeError(
            f"Credit snapshot shape mismatch: rows={len(frame)}, features={len(frame.columns) - 1}"
        )

    manifest = write_credit_snapshot(
        frame,
        args.out,
        source,
    )
    content_digest = hashlib.sha256()
    with gzip.open(args.out.resolve(), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            content_digest.update(chunk)
    observed = content_digest.hexdigest()
    if observed != CREDIT_CANONICAL_CONTENT_SHA256:
        raise RuntimeError(
            f"Canonical CSV checksum mismatch: expected {CREDIT_CANONICAL_CONTENT_SHA256}, observed {observed}"
        )
    print(f"[OK] rows={len(frame)} columns={len(frame.columns) - 1}")
    print(f"[OK] data={args.out.resolve()}")
    print(f"[OK] manifest={manifest}")


if __name__ == "__main__":
    main()
