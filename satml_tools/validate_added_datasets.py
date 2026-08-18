#!/usr/bin/env python3
"""Fail-closed validation for Fashion-MNIST and WDBC SaTML studies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from qurift.satml_wdbc import WDBC_CANONICAL_CONTENT_SHA256, load_wdbc_snapshot
from qurift.satml_fashion import FASHION_RAW_SHA256


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def validate_design(targets: pd.DataFrame, dataset: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    required = {
        "target_id", "dataset", "block_id", "fm_kind", "reps", "depth",
        "data_seed", "model_seed", "split_seed", "init_seed", "vector_test",
    }
    missing = required - set(targets)
    add(checks, "required_columns", not missing, f"missing={sorted(missing)}")
    if missing:
        return checks
    add(checks, "dataset_identity", targets.dataset.eq(dataset).all(), dataset)
    add(checks, "unique_target_ids", not targets.target_id.duplicated().any(), f"rows={len(targets)}")
    expected_depths = (2, 6) if dataset == "fashion_mnist" else (2,)
    expected_cells = {
        (fm, repetitions, depth)
        for fm in ("z", "zz", "eff_su2")
        for repetitions in (1, 5)
        for depth in expected_depths
    }
    complete = True
    paired = True
    for _, block in targets.groupby("block_id"):
        observed = set(zip(block.fm_kind, block.reps.astype(int), block.depth.astype(int)))
        complete &= observed == expected_cells
        paired &= (
            block.data_seed.nunique() == 1
            and block.model_seed.nunique() == 1
            and (block.data_seed == block.split_seed).all()
            and (block.model_seed == block.init_seed).all()
        )
    add(checks, "complete_structural_blocks", complete, f"cells_per_block={len(expected_cells)}")
    add(checks, "paired_seeds_within_blocks", paired, "one split/init pair per block")
    first = targets.groupby("block_id").first()
    independent = first.data_seed.nunique() == len(first) and first.model_seed.nunique() == len(first)
    add(checks, "independent_block_seeds", independent, f"blocks={len(first)}")
    if dataset == "fashion_mnist":
        add(checks, "one_percent_fpr_design", targets.vector_test.ge(2000).all(), "2,000 nonmembers")
    else:
        add(checks, "wdbc_complete_partition", (
            targets.vector_train + targets.vector_valid + targets.vector_test
        ).eq(569).all(), "160 train + 80 validation + 329 test")
    return checks


def validate_runs(targets: pd.DataFrame, run_root: Path, dataset: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    missing: list[str] = []
    block_hashes: dict[str, list[dict[str, str]]] = {}
    source_hashes_valid = True
    for _, row in targets.iterrows():
        directory = run_root / str(row.experiment) / str(row.target_id)
        required = [
            directory / "target_model.pt",
            directory / "target_attack_data.pt",
            directory / "target_export_summary.json",
            directory / "dataset_provenance.json",
        ]
        if dataset == "breast_cancer_wdbc":
            required.append(directory / "dataset_preprocessor.joblib")
        absent = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
        if absent:
            missing.extend(absent)
            continue
        provenance = json.loads((directory / "dataset_provenance.json").read_text(encoding="utf-8"))
        block_hashes.setdefault(str(row.block_id), []).append(provenance["split_index_sha256"])
        if dataset == "fashion_mnist":
            source_hashes_valid &= provenance.get("source", {}).get("raw_file_sha256") == FASHION_RAW_SHA256
    add(checks, "complete_target_artifacts", not missing, f"missing={len(missing)}")
    if dataset == "fashion_mnist":
        add(checks, "fashion_source_checksums", source_hashes_valid and not missing, "four canonical IDX files")
    within = all(values and all(value == values[0] for value in values) for values in block_hashes.values())
    add(checks, "identical_partitions_within_block", within, "all structural configurations share records")
    training_hashes = [values[0].get("train") for values in block_hashes.values() if values]
    add(
        checks,
        "distinct_training_partitions_across_blocks",
        len(training_hashes) == len(set(training_hashes)),
        f"observed={len(training_hashes)} unique={len(set(training_hashes))}",
    )
    return checks


def validate_attacks(targets: pd.DataFrame, attacks: pd.DataFrame, dataset: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    loss = attacks[attacks.attack.astype(str).str.lower().eq("loss")]
    add(
        checks,
        "loss_attack_complete",
        set(loss.target_id.astype(str)) == set(targets.target_id.astype(str)),
        f"expected={len(targets)} observed={len(loss)}",
    )
    minimum = 2000 if dataset == "fashion_mnist" else 329
    add(
        checks,
        "nonmember_count",
        bool(len(loss) and loss.n_nonmember.ge(minimum).all()),
        f"minimum_required={minimum}",
    )
    required_fpr = "resolvable_0p01" if dataset == "fashion_mnist" else "resolvable_0p05"
    values = loss.get(required_fpr, pd.Series(dtype=bool)).astype(str).str.lower().isin({"true", "1", "yes"})
    add(checks, "primary_low_fpr_resolvable", bool(len(values) and values.all()), required_fpr)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--dataset", choices=["fashion_mnist", "breast_cancer_wdbc"], required=True)
    parser.add_argument("--wdbc-data", type=Path, default=Path("data/wdbc/wdbc.csv.gz"))
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--attacks", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    targets = pd.read_csv(args.targets)
    checks = validate_design(targets, args.dataset)
    if args.dataset == "breast_cancer_wdbc":
        try:
            _, manifest = load_wdbc_snapshot(args.wdbc_data)
            observed = str(manifest.get("canonical_csv_sha256", ""))
            add(checks, "wdbc_snapshot_checksum", observed == WDBC_CANONICAL_CONTENT_SHA256, observed)
        except Exception as error:
            add(checks, "wdbc_snapshot_checksum", False, f"{type(error).__name__}: {error}")
    if args.run_root is not None:
        checks.extend(validate_runs(targets, args.run_root, args.dataset))
    if args.attacks is not None:
        checks.extend(validate_attacks(targets, pd.read_csv(args.attacks), args.dataset))
    payload = {"passed": all(item["passed"] for item in checks), "checks": checks}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for item in checks:
        print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['check']}: {item['detail']}")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
