#!/usr/bin/env python3
"""Materialize and validate the frozen Phase-7 candidate population without training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
REVIEWER = ROOT / "reviewer_tools"
for directory in (ROOT, REVIEWER):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from experiments.check_channel_lira_phase7_readiness import (  # noqa: E402
    DEFAULT_PROTOCOL,
    DEFAULT_PROTOCOL_LOCK,
    read_targets,
    resolve_repo_path,
    sha256,
    validate_design,
    validate_protocol_lock,
)
from reviewer_tools.qurift_lira_attack import (  # noqa: E402
    balanced_inclusion_matrix,
    load_context,
    tensor_fingerprint,
)
from reviewer_tools.reviewer_common import stable_seed  # noqa: E402


DEFAULT_OUT = ROOT / "channel_lira_results/phase7_readiness/CANDIDATE_PROBE.json"


def hash_sample_ids(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def build_probe(protocol_path: Path, lock_path: Path) -> dict[str, object]:
    protocol_path = protocol_path.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    targets_path = resolve_repo_path(protocol["provenance"]["phase7_target_manifest"])
    rows = read_targets(targets_path)
    errors = validate_protocol_lock(protocol_path, lock_path.resolve())
    errors.extend(validate_design(protocol, rows))
    if errors:
        raise ValueError("Phase-7 design is invalid: " + "; ".join(errors))

    dataset_signatures = {
        (
            row["dataset"], row["data_seed"], row["vector_train"],
            row["vector_valid"], row["vector_test"],
        )
        for row in rows
    }
    if len(dataset_signatures) != 1:
        raise ValueError("Phase-7 targets do not share one controlled candidate population")

    representative = rows[0]
    _, row, dataset, _, samples = load_context(
        ROOT, targets_path, representative["target_id"], torch.device("cpu")
    )
    membership = samples.membership.detach().cpu().reshape(-1)
    values, counts = membership.unique(return_counts=True)
    membership_counts = {
        str(int(value)): int(count)
        for value, count in zip(values.tolist(), counts.tolist())
    }
    reference_seed = int(protocol["reference_protocol"]["reference_design_seed"])
    cell = str(row["structural_cell_id"])
    design = balanced_inclusion_matrix(
        int(protocol["reference_protocol"]["references_per_cell"]),
        len(samples.labels),
        stable_seed(reference_seed, cell),
    )
    expected = protocol["study_population"]["candidate_protocol"]
    observed_members = membership_counts.get("1", 0)
    observed_nonmembers = membership_counts.get("0", 0)
    errors = []
    if observed_members != int(expected["members_per_target"]):
        errors.append("constructed member count differs from protocol")
    if observed_nonmembers != int(expected["nonmembers_per_target"]):
        errors.append("constructed nonmember count differs from protocol")
    if len(set(samples.sample_ids)) != len(samples.sample_ids):
        errors.append("constructed candidate IDs are not unique")
    per_reference = sorted({int(value) for value in design.sum(axis=1).tolist()})
    per_candidate = sorted({int(value) for value in design.sum(axis=0).tolist()})
    if per_reference != [int(protocol["reference_protocol"]["reference_train_records"])]:
        errors.append("constructed per-reference train size differs from protocol")
    if per_candidate != [int(protocol["reference_protocol"]["in_references_per_candidate"])]:
        errors.append("constructed candidate inclusion is not exactly balanced")
    if errors:
        raise ValueError("Candidate probe failed: " + "; ".join(errors))

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(protocol_path),
        "target_manifest": str(targets_path.resolve()),
        "target_manifest_sha256": sha256(targets_path),
        "representative_target_id": representative["target_id"],
        "shared_dataset_signature": list(next(iter(dataset_signatures))),
        "dataset_sizes": {name: len(dataset[name]) for name in ("train", "valid", "test")},
        "candidate_count": len(samples.labels),
        "membership_counts": membership_counts,
        "candidate_fingerprint": tensor_fingerprint(samples.inputs, samples.labels),
        "sample_ids_sha256": hash_sample_ids(list(samples.sample_ids)),
        "sample_ids_unique": True,
        "reference_design_seed": reference_seed,
        "reference_design_shape": list(design.shape),
        "per_reference_train_counts": per_reference,
        "per_candidate_in_counts": per_candidate,
        "training_performed": False,
        "circuit_execution_performed": False,
        "ready": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-lock", type=Path, default=DEFAULT_PROTOCOL_LOCK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = build_probe(args.protocol, args.protocol_lock)
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(
        f"candidate_ready={payload['ready']} count={payload['candidate_count']} "
        f"membership={payload['membership_counts']} -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
