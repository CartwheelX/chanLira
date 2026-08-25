#!/usr/bin/env python3
"""Materialize and verify the locked source-disjoint Q0 candidate partitions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REVIEWER = ROOT / "reviewer_tools"
for directory in (ROOT, REVIEWER):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from experiments.channel_lira_q0_common import (  # noqa: E402
    DEFAULT_PROTOCOL,
    DEFAULT_PROTOCOL_LOCK,
    DEFAULT_SNAPSHOT,
    DEFAULT_TARGETS,
    content_ids,
    read_targets,
    sha256,
    validate_protocol,
)
from reviewer_tools.qurift_lira_attack import load_context  # noqa: E402


DEFAULT_OUT = ROOT / "channel_lira_results/q0_readiness/CANDIDATE_PARTITION_PROBE.json"


def hash_values(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def source_indices(dataset: object) -> np.ndarray:
    values = getattr(dataset, "source_indices", None)
    if values is None:
        raise ValueError("Q0 MNIST partition does not expose source indices")
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.int64).reshape(-1)


def build_probe(
    protocol_path: Path,
    lock_path: Path,
    targets_path: Path,
    snapshot_path: Path,
) -> dict[str, object]:
    protocol = validate_protocol(
        protocol_path, lock_path, targets_path, snapshot_path
    )
    rows = read_targets(targets_path)
    records = []
    global_source: set[str] = set()
    global_content: set[str] = set()
    cross_target_content_overlaps = 0
    errors: list[str] = []
    for row in rows:
        _, _, dataset, _, samples = load_context(
            ROOT, targets_path, row["target_id"], torch.device("cpu")
        )
        identities = content_ids(samples.inputs, samples.labels).astype(str).tolist()
        train_source = source_indices(dataset["train"])
        test_source = source_indices(dataset["test"])
        source_values = [f"canonical_mnist_train|{int(value)}" for value in train_source]
        source_values.extend(
            f"canonical_mnist_train|{int(value)}" for value in test_source
        )
        overlap = global_source.intersection(source_values)
        content_overlap = global_content.intersection(identities)
        cross_target_content_overlaps += len(content_overlap)
        membership = samples.membership.detach().cpu().numpy().astype(int)
        labels = samples.labels.detach().cpu().numpy().astype(int)
        target_errors = []
        if len(source_values) != 2000 or len(set(source_values)) != 2000:
            target_errors.append("source indices are not unique within target")
        if overlap:
            target_errors.append(
                f"{len(overlap)} source indices overlap an earlier target partition"
            )
        if len(identities) != 2000 or len(set(identities)) != 2000:
            target_errors.append("content identities are not unique within target")
        if np.bincount(membership, minlength=2).tolist() != [1000, 1000]:
            target_errors.append("membership counts are not 1000/1000")
        class_counts = np.bincount(labels, minlength=4).tolist()
        if class_counts != [500, 500, 500, 500]:
            target_errors.append(f"candidate class counts are not balanced: {class_counts}")
        errors.extend(f"{row['target_id']}: {value}" for value in target_errors)
        records.append(
            {
                "target_id": row["target_id"],
                "partition_id": int(float(row["mnist_disjoint_partition_id"])),
                "candidate_count": len(identities),
                "membership_counts": np.bincount(membership, minlength=2).tolist(),
                "class_counts": class_counts,
                "source_ids_sha256": hash_values(source_values),
                "content_ids_sha256": hash_values(identities),
                "source_overlap_with_earlier_targets": len(overlap),
                "content_overlap_with_earlier_targets": len(content_overlap),
                "ready": not target_errors,
            }
        )
        global_source.update(source_values)
        global_content.update(identities)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(protocol_path),
        "target_manifest_sha256": sha256(targets_path),
        "partition_seed": protocol["study_population"]["mnist_partition_protocol"][
            "partition_seed"
        ],
        "targets": records,
        "target_count": len(records),
        "total_source_identities": len(global_source),
        "expected_source_identities": 12000,
        "source_disjoint_across_targets": len(global_source) == 12000,
        "cross_target_content_overlap_occurrences": cross_target_content_overlaps,
        "content_overlap_policy": (
            "Any duplicated pixel-identical content is blocked between attack-training "
            "and held victim/calibration targets during analysis."
        ),
        "training_performed": False,
        "circuit_execution_performed": False,
        "errors": errors,
        "ready": not errors and len(global_source) == 12000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-lock", type=Path, default=DEFAULT_PROTOCOL_LOCK)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    for name in ("protocol", "protocol_lock", "targets", "snapshot", "out"):
        setattr(args, name, getattr(args, name).resolve())
    payload = build_probe(
        args.protocol, args.protocol_lock, args.targets, args.snapshot
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.out)
    print(
        f"q0_candidates_ready={payload['ready']} "
        f"source_identities={payload['total_source_identities']}/12000 "
        f"content_overlaps={payload['cross_target_content_overlap_occurrences']} "
        f"-> {args.out}",
        flush=True,
    )
    if not payload["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
