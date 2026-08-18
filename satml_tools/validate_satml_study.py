#!/usr/bin/env python3
"""Fail-closed protocol and artifact validation for the SaTML study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from qurift.satml_data import CREDIT_CANONICAL_CONTENT_SHA256, load_credit_snapshot


def validate_design(targets: pd.DataFrame, expected_blocks: int) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    required = {
        "target_id", "block_id", "fm_kind", "reps", "depth", "data_seed", "model_seed",
        "split_seed", "init_seed", "vector_test", "feature_angle_scale",
    }
    missing = required - set(targets.columns)
    add("required_columns", not missing, f"missing={sorted(missing)}")
    if missing:
        return checks
    add("unique_target_ids", not targets.target_id.duplicated().any(), f"rows={len(targets)}")
    add("block_count", targets.block_id.nunique() == expected_blocks, f"observed={targets.block_id.nunique()}")
    expected_cells = {(fm, reps, depth) for fm in ("z", "zz", "eff_su2") for reps in (1, 5) for depth in (2, 6)}
    complete = True
    paired = True
    for _, group in targets.groupby("block_id"):
        observed = set(zip(group.fm_kind.astype(str), group.reps.astype(int), group.depth.astype(int)))
        complete &= observed == expected_cells
        paired &= (
            group.data_seed.nunique() == 1
            and group.model_seed.nunique() == 1
            and (group.data_seed == group.split_seed).all()
            and (group.model_seed == group.init_seed).all()
        )
    add("complete_12_cell_blocks", complete, "3 feature maps × 2 repetitions × 2 depths")
    add("paired_seeds_within_blocks", paired, "one split/init pair per block")
    add("independent_block_seeds", targets.groupby("block_id").data_seed.first().nunique() == expected_blocks and targets.groupby("block_id").model_seed.first().nunique() == expected_blocks, "split and initialization seeds unique across blocks")
    add("low_fpr_nonmembers", bool((targets.vector_test >= 2000).all()), f"minimum={targets.vector_test.min()}")
    add("baseline_angle_scale", bool((targets.feature_angle_scale.astype(float) == 1.0).all()), "confirmatory factorial fixes alpha=1")
    return checks


def validate_runs(targets: pd.DataFrame, run_root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    missing = []
    provenance_by_block: dict[str, list[dict[str, Any]]] = {}
    for _, row in targets.iterrows():
        directory = run_root / str(row.experiment) / str(row.target_id)
        required = [
            directory / "target_model.pt", directory / "target_attack_data.pt",
            directory / "target_export_summary.json", directory / "dataset_preprocessor.joblib",
            directory / "dataset_provenance.json",
        ]
        absent = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
        if absent:
            missing.extend(absent)
            continue
        provenance = json.loads((directory / "dataset_provenance.json").read_text(encoding="utf-8"))
        provenance_by_block.setdefault(str(row.block_id), []).append(provenance)
    checks.append({"check": "complete_target_artifacts", "passed": not missing, "detail": f"missing={len(missing)}"})
    consistent = True
    block_hashes = []
    for records in provenance_by_block.values():
        hashes = [record.get("split_index_sha256", {}) for record in records]
        consistent &= bool(hashes) and all(value == hashes[0] for value in hashes)
        if hashes:
            block_hashes.append(hashes[0].get("train"))
    checks.append({"check": "identical_partitions_within_block", "passed": consistent, "detail": "split hashes must match across all 12 configurations"})
    checks.append({"check": "distinct_training_partitions_across_blocks", "passed": len(block_hashes) == len(set(block_hashes)), "detail": f"observed={len(block_hashes)} unique={len(set(block_hashes))}"})
    return checks


def validate_attacks(targets: pd.DataFrame, attacks: pd.DataFrame) -> list[dict[str, Any]]:
    loss = attacks[attacks.attack.astype(str).str.lower().eq("loss")].copy()
    target_ids = set(targets.target_id.astype(str))
    observed = set(loss.target_id.astype(str))
    checks = [
        {"check": "loss_attack_complete", "passed": observed == target_ids, "detail": f"expected={len(target_ids)} observed={len(observed)}"},
        {"check": "attack_nonmembers_at_least_2000", "passed": bool(len(loss) and (loss.n_nonmember >= 2000).all()), "detail": f"minimum={loss.n_nonmember.min() if len(loss) else 'none'}"},
    ]
    if "resolvable_0p01" in loss:
        resolvable_values = loss.resolvable_0p01.map(
            lambda value: value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes"}
        )
        resolvable = bool(len(loss) and resolvable_values.all())
    else:
        resolvable = False
    attained = "attained_fpr_0p01" in loss
    checks.extend(
        [
            {"check": "tpr_at_1pct_resolvable", "passed": resolvable, "detail": "requires --fprs to include 0.01"},
            {"check": "attained_1pct_fpr_recorded", "passed": attained, "detail": "empirical FPR accompanies requested FPR"},
        ]
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--expected-blocks", type=int, default=8)
    parser.add_argument("--credit-data", type=Path, default=Path("data/credit_default/credit_default.csv.gz"))
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--attacks", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("satml_results/protocol_validation.json"))
    args = parser.parse_args()
    targets = pd.read_csv(args.targets)
    checks = validate_design(targets, args.expected_blocks)
    try:
        _, manifest = load_credit_snapshot(args.credit_data)
        observed = str(manifest.get("canonical_csv_sha256", ""))
        checks.append(
            {"check": "credit_snapshot_checksum", "passed": observed == CREDIT_CANONICAL_CONTENT_SHA256,
             "detail": f"expected={CREDIT_CANONICAL_CONTENT_SHA256} observed={observed}"}
        )
    except Exception as exc:
        checks.append({"check": "credit_snapshot_checksum", "passed": False, "detail": f"{type(exc).__name__}: {exc}"})
    if args.run_root is not None:
        checks.extend(validate_runs(targets, args.run_root))
    if args.attacks is not None:
        checks.extend(validate_attacks(targets, pd.read_csv(args.attacks)))
    payload = {"passed": all(check["passed"] for check in checks), "checks": checks}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for check in checks:
        print(f"[{'PASS' if check['passed'] else 'FAIL'}] {check['check']}: {check['detail']}")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
