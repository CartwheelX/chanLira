#!/usr/bin/env python3
"""Qiskit/IBM backend bridge for QuRiFT noisy finite-shot evaluation.

This module intentionally has no top-level Qiskit imports so that repository
preflight and syntax checks can run before Qiskit is installed.

Key safety rule: a failed IBM noise load is never replaced by ideal Aer under a
"noisy" label.  The caller must either abort (`require_noise=True`) or skip the
noisy condition explicitly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


TOKEN_ENV_NAMES = ("QISKIT_IBM_TOKEN", "IBM_QUANTUM_TOKEN")
INSTANCE_ENV_NAMES = ("QISKIT_IBM_INSTANCE", "IBM_QUANTUM_INSTANCE")


def _first_nonempty_env(names: Sequence[str]) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _iso_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    try:
        return value.isoformat()
    except Exception:
        text = str(value).strip()
        return text or None


def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", None)
    if callable(name):
        name = name()
    return str(name or "unknown")


def _safe_backend_properties(backend: Any) -> Any:
    try:
        properties = backend.properties()
        return properties
    except Exception:
        return None


def _backend_calibration_timestamp(backend: Any) -> Optional[str]:
    properties = _safe_backend_properties(backend)
    for object_ in (properties, getattr(backend, "target", None)):
        if object_ is None:
            continue
        for attribute in (
            "last_update_date",
            "last_update_datetime",
            "updated_at",
            "timestamp",
        ):
            value = getattr(object_, attribute, None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    value = None
            result = _iso_or_none(value)
            if result:
                return result
    return None


def backend_basis_gates(backend: Any) -> List[str]:
    result: List[str] = []
    target = getattr(backend, "target", None)
    if target is not None:
        names = getattr(target, "operation_names", None)
        if names is not None:
            try:
                result = sorted(str(name) for name in names)
            except Exception:
                result = []
    if not result:
        configuration = None
        try:
            configuration = backend.configuration()
        except Exception:
            configuration = None
        values = getattr(configuration, "basis_gates", None) if configuration is not None else None
        if values:
            result = sorted(str(value) for value in values)
    return result


def backend_coupling_edges(backend: Any) -> List[List[int]]:
    coupling = getattr(backend, "coupling_map", None)
    if callable(coupling):
        try:
            coupling = coupling()
        except Exception:
            coupling = None
    if coupling is None:
        target = getattr(backend, "target", None)
        if target is not None:
            try:
                coupling = target.build_coupling_map()
            except Exception:
                coupling = None
    if coupling is None:
        try:
            coupling = backend.configuration().coupling_map
        except Exception:
            coupling = None
    if coupling is None:
        return []
    try:
        edges = coupling.get_edges()
    except Exception:
        edges = coupling
    output = []
    for edge in edges or []:
        try:
            a, b = edge
            output.append([int(a), int(b)])
        except Exception:
            continue
    return sorted(output)


def create_runtime_service(account_name: Optional[str] = None):
    """Create QiskitRuntimeService using a saved account or environment.

    Authentication priority:
      1. named saved account (`account_name`),
      2. token and optional instance from environment variables,
      3. default saved account.

    Credentials are never accepted as CLI arguments and never logged.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    if account_name and str(account_name).strip():
        return QiskitRuntimeService(name=str(account_name).strip()), "saved_named_account"

    token = _first_nonempty_env(TOKEN_ENV_NAMES)
    instance = _first_nonempty_env(INSTANCE_ENV_NAMES)
    if token:
        kwargs: Dict[str, Any] = {
            "channel": "ibm_quantum_platform",
            "token": token,
        }
        if instance:
            kwargs["instance"] = instance
        return QiskitRuntimeService(**kwargs), "environment_credentials"

    return QiskitRuntimeService(), "saved_default_account"


@dataclass
class BackendNoiseMetadata:
    requested_backend_name: str
    requested_noise_backend_name: str
    resolved_backend_name: Optional[str]
    resolved_noise_backend_name: Optional[str]
    authentication_mode: Optional[str]
    noise_model_loaded: bool
    noise_load_error: Optional[str]
    gate_error_enabled: bool
    readout_error_enabled: bool
    thermal_relaxation_enabled: bool
    calibration_timestamp: Optional[str]
    basis_gates: List[str]
    noise_basis_gates: List[str]
    coupling_map: List[List[int]]
    backend_num_qubits: Optional[int]
    noise_instructions: List[str]
    noise_qubits: List[List[int]]
    backend_mismatch: bool


@dataclass
class BackendNoiseContext:
    service: Any
    backend: Any
    noise_backend: Any
    noise_model: Any
    metadata: BackendNoiseMetadata


def load_backend_noise_context(
    backend_name: str,
    noise_backend_name: Optional[str] = None,
    *,
    account_name: Optional[str] = None,
    require_noise: bool = False,
    allow_backend_mismatch: bool = False,
) -> BackendNoiseContext:
    """Load backend constraints and an approximate backend-derived Aer model.

    If loading fails and `require_noise` is False, the returned context has
    `noise_model=None` and a populated `noise_load_error`.  No ideal simulator
    is created here and no result is mislabeled as noisy.
    """
    requested_noise_name = noise_backend_name or backend_name
    service = backend = noise_backend = noise_model = None
    auth_mode = None
    error_text = None

    try:
        from qiskit_aer.noise import NoiseModel

        service, auth_mode = create_runtime_service(account_name)
        backend = service.backend(backend_name)
        noise_backend = backend if requested_noise_name == backend_name else service.backend(requested_noise_name)
        if requested_noise_name != backend_name and not allow_backend_mismatch:
            raise RuntimeError(
                "Transpilation backend and noise backend differ. Use the same backend "
                "for a defensible sanity check, or pass --allow-backend-mismatch explicitly."
            )
        noise_model = NoiseModel.from_backend(
            noise_backend,
            gate_error=True,
            readout_error=True,
            thermal_relaxation=True,
        )
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        noise_model = None
        if require_noise:
            raise RuntimeError(
                f"Required IBM backend noise model could not be loaded for "
                f"'{requested_noise_name}': {error_text}"
            ) from exc

    resolved_backend = _backend_name(backend) if backend is not None else None
    resolved_noise_backend = _backend_name(noise_backend) if noise_backend is not None else None
    target_basis = backend_basis_gates(backend) if backend is not None else []
    coupling = backend_coupling_edges(backend) if backend is not None else []
    calibration = _backend_calibration_timestamp(noise_backend or backend) if (noise_backend or backend) else None
    num_qubits = getattr(backend, "num_qubits", None) if backend is not None else None
    try:
        num_qubits = int(num_qubits) if num_qubits is not None else None
    except Exception:
        num_qubits = None

    noise_basis = []
    noise_instructions = []
    noise_qubits: List[List[int]] = []
    if noise_model is not None:
        try:
            noise_basis = sorted(str(value) for value in noise_model.basis_gates)
        except Exception:
            noise_basis = []
        try:
            noise_instructions = sorted(str(value) for value in noise_model.noise_instructions)
        except Exception:
            noise_instructions = []
        try:
            noise_qubits = sorted([list(map(int, item)) for item in noise_model.noise_qubits])
        except Exception:
            noise_qubits = []

    metadata = BackendNoiseMetadata(
        requested_backend_name=str(backend_name),
        requested_noise_backend_name=str(requested_noise_name),
        resolved_backend_name=resolved_backend,
        resolved_noise_backend_name=resolved_noise_backend,
        authentication_mode=auth_mode,
        noise_model_loaded=noise_model is not None,
        noise_load_error=error_text,
        gate_error_enabled=bool(noise_model is not None),
        readout_error_enabled=bool(noise_model is not None),
        thermal_relaxation_enabled=bool(noise_model is not None),
        calibration_timestamp=calibration,
        basis_gates=target_basis,
        noise_basis_gates=noise_basis,
        coupling_map=coupling,
        backend_num_qubits=num_qubits,
        noise_instructions=noise_instructions,
        noise_qubits=noise_qubits,
        backend_mismatch=bool(
            resolved_backend and resolved_noise_backend and resolved_backend != resolved_noise_backend
        ),
    )
    return BackendNoiseContext(
        service=service,
        backend=backend,
        noise_backend=noise_backend,
        noise_model=noise_model,
        metadata=metadata,
    )


def transpile_for_backend(
    circuits: Sequence[Any],
    *,
    backend: Any = None,
    basis_gates: Optional[Sequence[str]] = None,
    coupling_map: Optional[Sequence[Sequence[int]]] = None,
    optimization_level: int = 1,
    seed_transpiler: int = 0,
) -> List[Any]:
    """Transpile circuits deterministically against a backend when available."""
    circuits = list(circuits)
    if not circuits:
        return []

    try:
        from qiskit.transpiler import generate_preset_pass_manager

        kwargs: Dict[str, Any] = {
            "optimization_level": int(optimization_level),
            "seed_transpiler": int(seed_transpiler),
        }
        if backend is not None:
            kwargs["backend"] = backend
        else:
            if basis_gates:
                kwargs["basis_gates"] = list(basis_gates)
            if coupling_map:
                kwargs["coupling_map"] = [list(map(int, edge)) for edge in coupling_map]
        pass_manager = generate_preset_pass_manager(**kwargs)
        output = pass_manager.run(circuits)
        return list(output) if isinstance(output, (list, tuple)) else [output]
    except Exception:
        from qiskit import transpile

        kwargs = {
            "optimization_level": int(optimization_level),
            "seed_transpiler": int(seed_transpiler),
        }
        if backend is not None:
            kwargs["backend"] = backend
        else:
            if basis_gates:
                kwargs["basis_gates"] = list(basis_gates)
            if coupling_map:
                kwargs["coupling_map"] = [list(map(int, edge)) for edge in coupling_map]
        output = transpile(circuits, **kwargs)
        return list(output) if isinstance(output, (list, tuple)) else [output]


def clean_count_key(value: Any) -> str:
    if isinstance(value, int):
        return bin(value)[2:]
    text = str(value).replace(" ", "")
    if text.startswith("0x"):
        try:
            return bin(int(text, 16))[2:]
        except Exception:
            pass
    return text


def counts_to_z_expectations(counts: Mapping[Any, int], n_wires: int) -> np.ndarray:
    """Convert standard q[i]→c[i] measurement counts to Pauli-Z means."""
    total = int(sum(int(value) for value in counts.values()))
    if total <= 0:
        raise ValueError("Counts dictionary is empty")
    output = np.zeros(int(n_wires), dtype=np.float64)
    for raw_key, raw_count in counts.items():
        key = clean_count_key(raw_key).zfill(int(n_wires))
        if len(key) > int(n_wires):
            key = key[-int(n_wires):]
        count = int(raw_count)
        for wire in range(int(n_wires)):
            bit = key[int(n_wires) - 1 - wire]
            output[wire] += count * (1.0 if bit == "0" else -1.0)
    return output / float(total)


def circuit_resource_counts(circuit: Any) -> Dict[str, Any]:
    one_qubit = two_qubit = multi_qubit = measurement = total = 0
    operation_counts: Dict[str, int] = {}
    for item in getattr(circuit, "data", []):
        operation = getattr(item, "operation", None)
        if operation is None and isinstance(item, tuple) and item:
            operation = item[0]
        if operation is None:
            continue
        name = str(getattr(operation, "name", "unknown"))
        nq = int(getattr(operation, "num_qubits", 0) or 0)
        operation_counts[name] = operation_counts.get(name, 0) + 1
        if name == "measure":
            measurement += 1
            continue
        if name in {"barrier", "delay"}:
            continue
        total += 1
        if nq == 1:
            one_qubit += 1
        elif nq == 2:
            two_qubit += 1
        elif nq > 2:
            multi_qubit += 1
    try:
        depth = int(circuit.depth())
    except Exception:
        depth = None
    return {
        "transpiled_depth": depth,
        "transpiled_total_gates": int(total),
        "transpiled_one_qubit_gates": int(one_qubit),
        "transpiled_two_qubit_gates": int(two_qubit),
        "transpiled_multi_qubit_gates": int(multi_qubit),
        "measurement_operations": int(measurement),
        "operation_counts": json.dumps(operation_counts, sort_keys=True),
        "num_qubits": int(getattr(circuit, "num_qubits", 0) or 0),
        "num_clbits": int(getattr(circuit, "num_clbits", 0) or 0),
    }


def run_aer_counts(
    circuits: Sequence[Any],
    *,
    shots: int,
    seed_simulator: int,
    noise_model: Any = None,
) -> List[Mapping[Any, int]]:
    """Run locally in Aer and return one counts dictionary per circuit."""
    from qiskit_aer import AerSimulator

    simulator_kwargs: Dict[str, Any] = {}
    if noise_model is not None:
        simulator_kwargs["noise_model"] = noise_model
    simulator = AerSimulator(**simulator_kwargs)
    result = simulator.run(
        list(circuits),
        shots=int(shots),
        seed_simulator=int(seed_simulator),
    ).result()
    output = []
    for index in range(len(circuits)):
        output.append(result.get_counts(index))
    return output


def write_backend_probe(context: BackendNoiseContext, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(context.metadata), indent=2), encoding="utf-8")


def write_backend_snapshot(context: BackendNoiseContext, output_dir: Path) -> Dict[str, str]:
    """Persist enough backend state to reconstruct the local Aer noise model.

    Credentials and service/account objects are deliberately excluded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads: Dict[str, Any] = {
        "metadata.json": asdict(context.metadata),
    }
    if context.noise_model is not None:
        payloads["aer_noise_model.json"] = context.noise_model.to_dict()
    properties = _safe_backend_properties(context.noise_backend or context.backend)
    if properties is not None and hasattr(properties, "to_dict"):
        payloads["backend_properties.json"] = properties.to_dict()
    try:
        configuration = (context.backend or context.noise_backend).configuration()
        if hasattr(configuration, "to_dict"):
            payloads["backend_configuration.json"] = configuration.to_dict()
    except Exception:
        pass

    hashes: Dict[str, str] = {}
    for filename, payload in payloads.items():
        path = output_dir / filename
        encoded = json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")
        path.write_bytes(encoded)
        hashes[filename] = hashlib.sha256(encoded).hexdigest()
    manifest = {
        "files_sha256": hashes,
        "credentials_recorded": False,
        "reconstruction": "qiskit_aer.noise.NoiseModel.from_dict(aer_noise_model.json)",
    }
    manifest_path = output_dir / "snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    hashes[manifest_path.name] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return hashes
