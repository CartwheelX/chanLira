#!/usr/bin/env python3
"""Acquire raw target-only response data for the locked ChannelLiRA Q0 screen."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REVIEWER = ROOT / "reviewer_tools"
for directory in (ROOT, REVIEWER):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from experiments.channel_lira_q0_common import (  # noqa: E402
    DEFAULT_OUT,
    DEFAULT_PROTOCOL,
    DEFAULT_PROTOCOL_LOCK,
    DEFAULT_RUN_ROOT,
    DEFAULT_SNAPSHOT,
    DEFAULT_TARGETS,
    content_ids,
    dense_counts,
    dense_z_expectations,
    read_targets,
    sha256,
    validate_protocol,
)
from reviewer_tools.qurift_lira_attack import load_context  # noqa: E402
from reviewer_tools.qurift_qiskit_bridge import (  # noqa: E402
    circuit_resource_counts,
    load_backend_noise_snapshot,
    run_aer_counts,
    transpile_for_backend,
)
from reviewer_tools.qurift_target_loader import (  # noqa: E402
    apply_classical_head,
    build_qiskit_circuits,
    instantiate_model,
    load_saved_model,
    preprocess_like_train,
    quantum_features_and_scope,
    resolve_target_paths,
)


SCHEMA_VERSION = 1


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def circuit_sha256(circuit: Any) -> str:
    from qiskit import qpy

    buffer = io.BytesIO()
    qpy.dump(circuit, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


@torch.no_grad()
def exact_response(
    model: torch.nn.Module,
    qmain: Any,
    architecture: str,
    config: Any,
    samples: Any,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torchquantum as tq

    model.eval()
    z_rows = []
    probability_rows = []
    for start in range(0, len(samples.labels), int(batch_size)):
        inputs = preprocess_like_train(samples.inputs[start : start + batch_size], device)
        features, _ = quantum_features_and_scope(model, architecture, inputs, config)
        qdev = tq.QuantumDevice(
            n_wires=int(config.n_wires),
            bsz=int(features.shape[0]),
            device=features.device,
            record_op=False,
        )
        model.encoder(qdev, features)
        model.vqc_circuit(qdev)
        measured = model.measure(qdev)
        probabilities = apply_classical_head(model, measured)
        z_rows.append(measured.detach().cpu().numpy().astype(np.float32))
        probability_rows.append(
            probabilities.detach().cpu().numpy().astype(np.float32)
        )
    return np.concatenate(z_rows), np.concatenate(probability_rows)


def validate_resume(
    payload_path: Path,
    metadata_path: Path,
    *,
    protocol_hash: str,
    target_id: str,
) -> bool:
    if not payload_path.is_file() or not metadata_path.is_file():
        return False
    metadata = read_json(metadata_path)
    if metadata.get("protocol_sha256") != protocol_hash:
        raise RuntimeError(f"Q0 resume protocol mismatch: {metadata_path}")
    if metadata.get("target_id") != target_id:
        raise RuntimeError(f"Q0 resume target mismatch: {metadata_path}")
    if metadata.get("payload_sha256") != sha256(payload_path):
        raise RuntimeError(f"Q0 resume payload hash mismatch: {payload_path}")
    with np.load(payload_path, allow_pickle=False) as saved:
        required = {
            "sample_ids",
            "content_ids",
            "membership",
            "labels",
            "exact_z",
            "exact_probabilities",
            "counts_layout_a",
            "counts_layout_b",
            "z_layout_a",
            "z_layout_b",
            "probabilities_layout_a",
            "probabilities_layout_b",
        }
        if int(saved["schema_version"]) != SCHEMA_VERSION or not required.issubset(
            saved.files
        ):
            raise RuntimeError(f"Q0 resume payload schema mismatch: {payload_path}")
        expected_shapes = {
            "membership": (2000,),
            "labels": (2000,),
            "exact_z": (2000, 6),
            "exact_probabilities": (2000, 4),
            "counts_layout_a": (10, 2000, 64),
            "counts_layout_b": (5, 2000, 64),
            "z_layout_a": (10, 2000, 6),
            "z_layout_b": (5, 2000, 6),
            "probabilities_layout_a": (10, 2000, 4),
            "probabilities_layout_b": (5, 2000, 4),
        }
        mismatches = {
            name: (saved[name].shape, shape)
            for name, shape in expected_shapes.items()
            if saved[name].shape != shape
        }
        if mismatches:
            raise RuntimeError(
                f"Q0 resume payload shape mismatch at {payload_path}: {mismatches}"
            )
    return True


def acquire(args: argparse.Namespace) -> Path:
    protocol = validate_protocol(
        args.protocol, args.protocol_lock, args.targets, args.snapshot
    )
    protocol_hash = sha256(args.protocol)
    if args.acknowledge_protocol_hash != protocol_hash:
        raise ValueError(
            "Q0 acquisition requires --acknowledge-protocol-hash equal to the locked protocol SHA-256"
        )
    target_rows = {row["target_id"]: row for row in read_targets(args.targets)}
    if args.target_id not in target_rows:
        raise ValueError(f"Target is outside the Q0 manifest: {args.target_id}")

    payload_path = args.out_dir / "raw" / f"{args.target_id}.npz"
    metadata_path = args.out_dir / "metadata" / f"{args.target_id}.json"
    if args.resume and validate_resume(
        payload_path,
        metadata_path,
        protocol_hash=protocol_hash,
        target_id=args.target_id,
    ):
        print(f"[SKIP] complete Q0 payload: {payload_path.resolve()}", flush=True)
        return payload_path

    requested_device = str(args.device)
    device = torch.device(
        requested_device if requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    qmain, row, _, config, samples = load_context(
        ROOT, args.targets, args.target_id, device
    )
    model, architecture = instantiate_model(qmain, row, config, device)
    model_path, _ = resolve_target_paths(row, args.run_root)
    load_report = load_saved_model(model, model_path, device)
    exact_z, exact_probabilities = exact_response(
        model,
        qmain,
        architecture,
        config,
        samples,
        device=device,
        batch_size=args.exact_batch_size,
    )
    circuits, execution_scope = build_qiskit_circuits(
        qmain,
        model,
        architecture,
        config,
        samples,
        device=device,
        batch_size=args.qiskit_batch_size,
    )
    snapshot = load_backend_noise_snapshot(args.snapshot, require_noise=True)
    acquisition = protocol["acquisition"]
    shots = int(acquisition["shots_per_query"])
    layouts = acquisition["physical_initial_layouts"]
    basis_gates = snapshot.metadata.basis_gates or snapshot.metadata.noise_basis_gates
    transpiled: dict[str, list[Any]] = {}
    layout_metadata: dict[str, Any] = {}
    for layout_name in ("layout_a", "layout_b"):
        physical_layout = [int(value) for value in layouts[layout_name]]
        current = transpile_for_backend(
            circuits,
            backend=None,
            basis_gates=basis_gates,
            coupling_map=snapshot.metadata.coupling_map,
            initial_layout=physical_layout,
            optimization_level=int(acquisition["optimization_level"]),
            seed_transpiler=int(acquisition["transpiler_seed"]),
        )
        if len(current) != len(samples.labels):
            raise RuntimeError("Q0 transpilation changed the circuit count")
        transpiled[layout_name] = current
        resources = [circuit_resource_counts(value) for value in current]
        layout_metadata[layout_name] = {
            "physical_initial_layout": physical_layout,
            "first_circuit_qpy_sha256": circuit_sha256(current[0]),
            "first_circuit_layout_sha256": hashlib.sha256(
                str(getattr(current[0], "layout", None)).encode("utf-8")
            ).hexdigest(),
            "depth_min": int(min(value["transpiled_depth"] for value in resources)),
            "depth_max": int(max(value["transpiled_depth"] for value in resources)),
            "depth_mean": float(
                np.mean([value["transpiled_depth"] for value in resources])
            ),
            "first_circuit_resources": resources[0],
        }
    if (
        layout_metadata["layout_a"]["first_circuit_qpy_sha256"]
        == layout_metadata["layout_b"]["first_circuit_qpy_sha256"]
    ):
        raise RuntimeError("The two locked Q0 physical layouts produced identical circuits")

    condition_arrays: dict[str, np.ndarray] = {}
    for layout_name, seed_key in (
        ("layout_a", "simulator_seeds_layout_a"),
        ("layout_b", "simulator_seeds_layout_b"),
    ):
        dense_rows = []
        z_rows = []
        probability_rows = []
        for simulator_seed in [int(value) for value in acquisition[seed_key]]:
            namespace = stable_int(f"{args.target_id}|{layout_name}")
            counts = run_aer_counts(
                transpiled[layout_name],
                shots=shots,
                seed_simulator=(
                    simulator_seed * 1_000_003 + namespace
                ) % (2**31 - 1),
                noise_model=snapshot.noise_model,
                max_parallel_threads=args.aer_max_parallel_threads,
            )
            dense = dense_counts(counts, int(config.n_wires), shots)
            z_values = dense_z_expectations(dense).astype(np.float32)
            measured = torch.tensor(z_values, dtype=torch.float32, device=device)
            with torch.no_grad():
                probabilities = (
                    apply_classical_head(model, measured)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            dense_rows.append(dense)
            z_rows.append(z_values)
            probability_rows.append(probabilities)
            print(
                f"[{args.target_id}] {layout_name} repetition {len(dense_rows)}/"
                f"{len(acquisition[seed_key])}",
                flush=True,
            )
        condition_arrays[f"counts_{layout_name}"] = np.stack(dense_rows)
        condition_arrays[f"z_{layout_name}"] = np.stack(z_rows)
        condition_arrays[f"probabilities_{layout_name}"] = np.stack(probability_rows)

    membership = samples.membership.detach().cpu().numpy().astype(np.uint8)
    labels = samples.labels.detach().cpu().numpy().astype(np.int16)
    identities = content_ids(samples.inputs, samples.labels)
    if len(set(identities.tolist())) != len(identities):
        raise RuntimeError("A Q0 target contains duplicate content identities")
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = payload_path.with_suffix(payload_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(SCHEMA_VERSION),
            target_id=np.asarray(args.target_id),
            structural_cell_id=np.asarray(row["structural_cell_id"]),
            model_seed=np.asarray(int(float(row["model_seed"]))),
            data_seed=np.asarray(int(float(row["data_seed"]))),
            sample_ids=np.asarray(samples.sample_ids),
            content_ids=identities,
            membership=membership,
            labels=labels,
            exact_z=exact_z,
            exact_probabilities=exact_probabilities,
            **condition_arrays,
        )
    temporary.replace(payload_path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_id": args.target_id,
        "structural_cell_id": row["structural_cell_id"],
        "model_seed": int(float(row["model_seed"])),
        "data_seed": int(float(row["data_seed"])),
        "protocol_sha256": protocol_hash,
        "target_manifest_sha256": sha256(args.targets),
        "snapshot_manifest_sha256": sha256(args.snapshot / "snapshot_manifest.json"),
        "payload": str(payload_path.resolve()),
        "payload_sha256": sha256(payload_path),
        "target_model": str(model_path.resolve()),
        "target_model_sha256": sha256(model_path),
        "target_load_report": load_report,
        "candidate_count": len(labels),
        "member_count": int(membership.sum()),
        "nonmember_count": int((membership == 0).sum()),
        "content_identity_sha256": hashlib.sha256(
            "".join(identities.tolist()).encode("utf-8")
        ).hexdigest(),
        "quantum_execution_scope": execution_scope,
        "shots_per_query": shots,
        "layout_metadata": layout_metadata,
        "calibration_timestamp": snapshot.metadata.calibration_timestamp,
        "resolved_noise_backend_name": snapshot.metadata.resolved_noise_backend_name,
        "real_hardware_execution": False,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)
    print(f"[OK] Q0 acquisition -> {payload_path.resolve()}", flush=True)
    return payload_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-lock", type=Path, default=DEFAULT_PROTOCOL_LOCK)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--acknowledge-protocol-hash", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exact-batch-size", type=int, default=64)
    parser.add_argument("--qiskit-batch-size", type=int, default=16)
    parser.add_argument("--aer-max-parallel-threads", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("protocol", "protocol_lock", "targets", "snapshot", "run_root", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if min(args.exact_batch_size, args.qiskit_batch_size, args.aer_max_parallel_threads) < 1:
        parser.error("Q0 batch and thread settings must be positive")
    return args


def main() -> None:
    acquire(parse_args())


if __name__ == "__main__":
    main()
