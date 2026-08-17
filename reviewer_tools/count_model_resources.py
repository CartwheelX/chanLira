#!/usr/bin/env python3
"""Count QuRiFT trainable parameters and main-stack quantum operations.

Preferred source order
----------------------
1. ``resource_counts`` exported in the patched attack payload.
2. Fresh model construction from the target table and ``--repo-root``.
3. Checkpoint tensor-element inventory as a clearly marked diagnostic fallback.

Quantum gate counts cover the fixed feature-map encoder and the downstream VQC.
The QCNN patch-wise quanvolutional frontend is reported as excluded because its
executed operation count depends on the preprocessing path and patch count.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from reviewer_common import (
    atomic_write_csv,
    find_attack_files,
    torch_load,
    write_analysis_metadata,
)

ONE_QUBIT_GATES = {
    "rx", "ry", "rz", "h", "hadamard", "x", "y", "z", "s", "sdg", "t",
    "u", "u1", "u2", "u3", "p", "phase",
}
TWO_QUBIT_GATES = {
    "cx", "cnot", "cz", "swap", "iswap", "crx", "cry", "crz", "rxx", "ryy", "rzz",
}


def clean_text(value: Any, default: str) -> str:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    return default if text.lower() in {"", "nan", "none", "na"} else text


def classify_parameter_name(name: str) -> str:
    lowered = name.lower()
    if any(
        token in lowered
        for token in (
            "vqc", "q_layer", "qlayer", "quantum", "theta", "pqc",
            "qf.", "q_filter", "quanv",
        )
    ):
        return "quantum"
    if any(
        token in lowered
        for token in (
            "cnn", "mlp", "head", "classifier", "linear", "fc", "backbone",
            "feature_extractor", "fe.", "layernorm", "norm",
        )
    ):
        return "classical"
    return "unclassified"


def exact_parameter_counts(model: Any) -> dict[str, int]:
    counts = Counter()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            counts[classify_parameter_name(name)] += int(parameter.numel())
    return {
        "trainable_parameters_total": int(sum(counts.values())),
        "trainable_parameters_quantum": int(counts["quantum"]),
        "trainable_parameters_classical": int(counts["classical"]),
        "trainable_parameters_unclassified": int(counts["unclassified"]),
    }


def checkpoint_state_dict(obj: Any) -> Mapping[str, Any]:
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, Mapping):
        for key in ("state_dict", "model_state_dict", "target_state_dict", "model"):
            value = obj.get(key)
            if hasattr(value, "state_dict"):
                return value.state_dict()
            if isinstance(value, Mapping) and any(hasattr(item, "numel") for item in value.values()):
                return value
        if any(hasattr(item, "numel") for item in obj.values()):
            return obj
    return {}


def checkpoint_tensor_inventory(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "checkpoint_tensor_elements_total": np.nan,
            "checkpoint_tensor_elements_quantum_name": np.nan,
            "checkpoint_tensor_elements_classical_name": np.nan,
            "checkpoint_tensor_elements_unclassified_name": np.nan,
        }
    state = checkpoint_state_dict(torch_load(path))
    counts = Counter()
    for name, value in state.items():
        try:
            count = int(value.numel())
        except Exception:
            continue
        counts[classify_parameter_name(str(name))] += count
    return {
        "checkpoint_tensor_elements_total": int(sum(counts.values())),
        "checkpoint_tensor_elements_quantum_name": int(counts["quantum"]),
        "checkpoint_tensor_elements_classical_name": int(counts["classical"]),
        "checkpoint_tensor_elements_unclassified_name": int(counts["unclassified"]),
    }


def normalize_resource_payload(resource: Mapping[str, Any]) -> dict[str, Any]:
    gate_counts = resource.get("gate_counts", resource.get("gates", {}))
    if not isinstance(gate_counts, Mapping):
        gate_counts = {}
    clean_gates = {
        str(gate).lower(): int(value)
        for gate, value in gate_counts.items()
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)
    }
    total_gates = int(resource.get("quantum_gate_count_total", sum(clean_gates.values())))
    return {
        "trainable_parameters_total": resource.get("trainable_parameters_total", np.nan),
        "trainable_parameters_quantum": resource.get("trainable_parameters_quantum", np.nan),
        "trainable_parameters_classical": resource.get("trainable_parameters_classical", np.nan),
        "trainable_parameters_unclassified": resource.get("trainable_parameters_unclassified", np.nan),
        "quantum_gate_count_total": total_gates,
        "quantum_one_qubit_gates": resource.get(
            "quantum_one_qubit_gates",
            sum(value for gate, value in clean_gates.items() if gate in ONE_QUBIT_GATES),
        ),
        "quantum_two_qubit_gates": resource.get(
            "quantum_two_qubit_gates",
            sum(value for gate, value in clean_gates.items() if gate in TWO_QUBIT_GATES),
        ),
        "gate_counts_json": json.dumps(clean_gates, sort_keys=True),
        "gate_count_scope": resource.get(
            "gate_count_scope", "fixed encoder plus downstream variational circuit"
        ),
        "qcnn_frontend_included": resource.get("qcnn_frontend_included", np.nan),
        "circuit_depth": resource.get("circuit_depth", np.nan),
    }


def payload_resource_counts(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    candidates = [payload.get("resource_counts")]
    meta = payload.get("meta", {}) or {}
    candidates.extend([meta.get("resource_counts"), payload.get("circuit_resources")])
    for candidate in candidates:
        if isinstance(candidate, Mapping) and (
            "trainable_parameters_total" in candidate or "gate_counts" in candidate
        ):
            return normalize_resource_payload(candidate)
    return None


def feature_dimension(row: Mapping[str, Any]) -> int:
    dataset = clean_text(row.get("dataset"), "mnist").lower()
    if dataset in {"mnist", "cifar10"}:
        return 16
    if dataset == "blobs":
        return int(row.get("blobs_n_features", 4))
    if dataset == "moons" and bool(row.get("extra_feats", False)):
        return 4
    return 2


def build_model_from_target(qmain: Any, row: Mapping[str, Any]):
    feature_map = clean_text(row.get("fm_kind"), "z").lower()
    repetitions = int(row.get("reps", 1))
    pad_mode = clean_text(row.get("pad_mode"), "wrap")
    fm_entanglement = clean_text(row.get("fm_ent"), "linear")
    fm_operation = clean_text(row.get("fm_op"), "cx")
    mapper: dict[str, Any] = {"fm_kind": feature_map}
    if feature_map == "z":
        mapper.update(fm_z_reps=repetitions, fm_z_pad_mode=pad_mode)
    elif feature_map == "zz":
        mapper.update(
            fm_zz_reps=repetitions,
            fm_zz_pad_mode=pad_mode,
            fm_zz_entanglement=fm_entanglement,
        )
    elif feature_map == "eff_su2":
        mapper.update(
            fm_eff_reps=repetitions,
            fm_eff_pad_mod=pad_mode,
            fm_eff_ent_kind=fm_entanglement,
            fm_eff_twoq_op=fm_operation,
        )
    elif feature_map == "pauli":
        mapper.update(
            fm_pauli_reps=repetitions,
            fm_pali_pad=pad_mode,
            fm_pauli_entanglement=fm_entanglement,
        )
    else:
        raise ValueError(f"Unsupported feature map: {feature_map}")

    dataset = clean_text(row.get("dataset"), "mnist").lower()
    number_classes = 4 if dataset == "mnist" else 2
    config = qmain.QFCConfig(
        n_wires=int(row.get("n_wires", 4)),
        depth=int(row.get("depth", 2)),
        n_random_ops=0,
        batch_size=int(row.get("batch_size", 16)),
        device="cpu",
        feature_dim=feature_dimension(row),
        pool_hw=4 if dataset in {"mnist", "cifar10"} else 1,
        num_classes=number_classes,
        qlayer_ent_kind=clean_text(row.get("ql_ent"), "linear"),
        qlayer_twoq_op=clean_text(row.get("ql_op"), "crz"),
        qlayer_ent_trainable=True,
        qlayer_ent_wire_reverse=bool(row.get("ql_rev", False)),
        **mapper,
    )
    architecture = clean_text(row.get("architecture"), "qnn").lower()
    if architecture == "qnn":
        model = qmain.QFCModel(config)
    elif architecture == "hqnn":
        model = qmain.HybridQNN(config)
    elif architecture == "qcnn":
        model = qmain.QCNN(config)
    elif architecture == "mlp_qnn":
        reference = qmain.QFCModel(config)
        model = qmain.build_classical_baseline(config, reference)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model, config, architecture


def main_stack_gate_counts(model: Any, config: Any, architecture: str) -> dict[str, Any]:
    if architecture == "mlp_qnn":
        return {
            "quantum_gate_count_total": 0,
            "quantum_one_qubit_gates": 0,
            "quantum_two_qubit_gates": 0,
            "gate_counts_json": "{}",
            "gate_count_scope": "classical parameter-matched MLP; no quantum main stack",
            "qcnn_frontend_included": False,
        }

    gate_counts = Counter()
    seen_operation_lists: set[int] = set()
    for module in model.modules():
        operations = getattr(module, "func_list", None)
        if not isinstance(operations, list) or id(operations) in seen_operation_lists:
            continue
        seen_operation_lists.add(id(operations))
        for operation in operations:
            if isinstance(operation, Mapping):
                gate_counts[clean_text(operation.get("func"), "unknown").lower()] += 1

    n_wires = int(config.n_wires)
    depth = int(config.depth)
    topology = str(config.qlayer_ent_kind).lower()
    if topology in {"ring", "circular"}:
        entanglers_per_layer = n_wires
    elif topology == "full":
        entanglers_per_layer = n_wires * (n_wires - 1) // 2
    elif topology == "pairwise":
        entanglers_per_layer = n_wires // 2
    else:
        entanglers_per_layer = max(n_wires - 1, 0)
    gate_counts["rx"] += depth * n_wires
    gate_counts["ry"] += depth * n_wires
    gate_counts["rz"] += depth * n_wires
    gate_counts[str(config.qlayer_twoq_op).lower()] += depth * entanglers_per_layer

    return {
        "quantum_gate_count_total": int(sum(gate_counts.values())),
        "quantum_one_qubit_gates": int(
            sum(value for gate, value in gate_counts.items() if gate in ONE_QUBIT_GATES)
        ),
        "quantum_two_qubit_gates": int(
            sum(value for gate, value in gate_counts.items() if gate in TWO_QUBIT_GATES)
        ),
        "gate_counts_json": json.dumps(dict(sorted(gate_counts.items())), sort_keys=True),
        "gate_count_scope": "fixed encoder plus downstream variational circuit",
        "qcnn_frontend_included": False if architecture == "qcnn" else np.nan,
    }


def locate_artifacts(run_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    attack_files = {path.parent.name: path for path in find_attack_files(run_root)}
    model_files = {
        path.parent.name: path
        for path in run_root.rglob("target_model.pt")
        if path.is_file()
    }
    return attack_files, model_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("reviewer_results/model_resources")
    )
    parser.add_argument("--fail-on-missing-exact", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(args.targets)
    if "target_id" not in targets.columns:
        parser.error("Target table must include target_id")
    attack_files, model_files = locate_artifacts(args.run_root)

    qmain = None
    if args.repo_root is not None:
        qmain_path = args.repo_root / "experiments" / "qurift_main.py"
        if not qmain_path.exists():
            parser.error(f"Could not find {qmain_path}")
        os.environ.setdefault("QURIFT_DISABLE_DEBUG_EXPORTS", "1")
        os.environ.setdefault("QURIFT_DISABLE_CIRCUIT_EXPORTS", "1")
        sys.path[:0] = [
            str(args.repo_root.resolve()),
            str((args.repo_root / "experiments").resolve()),
        ]
        import qurift_main as qmain_module

        qmain = qmain_module

    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for _, target in targets.iterrows():
        target_id = str(target["target_id"])
        attack_path = attack_files.get(target_id)
        model_path = model_files.get(target_id)
        payload = torch_load(attack_path) if attack_path else {}
        exported = payload_resource_counts(payload) if isinstance(payload, Mapping) else None

        record = target.to_dict()
        record.update(
            {
                "target_id": target_id,
                "attack_path": str(attack_path) if attack_path else "",
                "model_path": str(model_path) if model_path else "",
            }
        )
        record.update(checkpoint_tensor_inventory(model_path))

        error = ""
        if exported is not None:
            record.update(exported)
            record["parameter_count_method"] = "exact model.named_parameters count exported during target run"
            record["gate_count_method"] = "main-stack operation metadata exported during target run"
            record["resource_count_source"] = "attack payload resource_counts"
            # Payloads produced before the MLP resource-export fix inherited
            # the analytic gate count of the reference QNN configuration.  The
            # parameter-matched MLP is entirely classical, so normalize those
            # legacy exports during the resource audit.
            if clean_text(target.get("architecture"), "qnn").lower() == "mlp_qnn":
                record.update(main_stack_gate_counts(None, None, "mlp_qnn"))
                record["gate_count_method"] = "classical MLP override; no quantum execution stack"
        elif qmain is not None:
            try:
                model, config, architecture = build_model_from_target(qmain, target.to_dict())
                record.update(exact_parameter_counts(model))
                record.update(main_stack_gate_counts(model, config, architecture))
                record["parameter_count_method"] = "exact fresh model.named_parameters reconstruction"
                record["gate_count_method"] = "fresh encoder operation-list count plus analytic downstream VQC count"
                record["resource_count_source"] = "reconstructed from target table and repository"
            except Exception as exc:
                error = repr(exc)
        else:
            error = "no exported resource_counts and --repo-root not supplied"

        exact_available = bool(
            np.isfinite(pd.to_numeric(record.get("trainable_parameters_total"), errors="coerce"))
            and np.isfinite(pd.to_numeric(record.get("quantum_gate_count_total"), errors="coerce"))
        )
        record["exact_resource_counts_available"] = exact_available
        record["resource_error"] = error
        rows.append(record)
        if not exact_available:
            missing_rows.append(
                {
                    "target_id": target_id,
                    "attack_path": str(attack_path) if attack_path else "",
                    "model_path": str(model_path) if model_path else "",
                    "error": error,
                }
            )

    raw = pd.DataFrame(rows)
    raw_path = args.out_dir / "model_resources_raw.csv"
    missing_path = args.out_dir / "model_resources_missing.csv"
    atomic_write_csv(raw, raw_path)
    atomic_write_csv(pd.DataFrame(missing_rows), missing_path)

    metrics = [
        column
        for column in (
            "trainable_parameters_total",
            "trainable_parameters_quantum",
            "trainable_parameters_classical",
            "trainable_parameters_unclassified",
            "quantum_gate_count_total",
            "quantum_one_qubit_gates",
            "quantum_two_qubit_gates",
        )
        if column in raw.columns
    ]
    group_columns = [
        column
        for column in (
            "experiment", "dataset", "architecture", "role", "fm_kind", "reps", "depth"
        )
        if column in raw.columns
    ]
    if group_columns and metrics and not raw.empty:
        summary = (
            raw.groupby(group_columns, dropna=False)[metrics]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        summary.columns = [
            "_".join(str(value) for value in column if str(value))
            if isinstance(column, tuple)
            else str(column)
            for column in summary.columns
        ]
    else:
        summary = pd.DataFrame()
    summary_path = args.out_dir / "model_resources_summary.csv"
    atomic_write_csv(summary, summary_path)

    write_analysis_metadata(
        args.out_dir / "analysis_metadata.json",
        script=Path(__file__).name,
        inputs=[str(args.run_root), str(args.targets), str(args.repo_root or "")],
        outputs=[str(raw_path), str(summary_path), str(missing_path)],
        ci_method="none; deterministic descriptive resource accounting",
        bootstrap_unit="none",
        bootstrap_replicates=0,
        error_bar_type="none for deterministic model resources",
        notes=(
            "Trainable parameter counts are exact only when exported or reconstructed. "
            "Checkpoint tensor-element counts are diagnostic and are not labeled trainable. "
            "Gate counts cover the fixed encoder and downstream VQC; QCNN patch-wise frontend "
            "execution is excluded and explicitly marked."
        ),
    )
    print(f"[OK] Resource table: {raw_path.resolve()}")
    if args.fail_on_missing_exact and missing_rows:
        raise SystemExit(f"Exact resource counts missing for {len(missing_rows)} targets")


if __name__ == "__main__":
    main()
