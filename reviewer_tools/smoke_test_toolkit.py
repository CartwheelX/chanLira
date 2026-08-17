#!/usr/bin/env python3
"""End-to-end smoke test for the non-noise reviewer toolkit.

The test generates small synthetic QuRiFT-like attack payloads and exercises:
metric extraction, matched verification, missing-run inventory, scalar MIAs,
correctness-only label-output MIA, resource accounting, factorial regression,
architecture analysis, DGX runner dry-run, and geometry launcher dry-run.
It does not import TorchQuantum or train a quantum model.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def build_payload(
    rng: np.random.Generator,
    *,
    target_id: str,
    metadata: dict[str, Any],
    train_acc: float,
    valid_acc: float,
    test_acc: float,
    member_signal: float,
    num_classes: int,
    gate_count: int,
    parameter_count: int,
) -> dict[str, Any]:
    n_member = 30
    n_nonmember = 30
    membership = np.concatenate(
        [np.zeros(n_member, dtype=np.int64), np.ones(n_nonmember, dtype=np.int64)]
    )
    is_member = membership == 0
    labels = rng.integers(0, num_classes, n_member + n_nonmember)
    true_probability = np.clip(
        rng.normal(0.54 + member_signal * is_member, 0.09, len(membership)),
        1.0 / num_classes + 0.02,
        0.97,
    )
    probabilities = np.empty((len(membership), num_classes), dtype=np.float32)
    for index, (label, p_true) in enumerate(zip(labels, true_probability)):
        probabilities[index] = (1.0 - p_true) / (num_classes - 1)
        probabilities[index, label] = p_true
    prediction = probabilities.argmax(axis=1)
    correctness = (prediction == labels).astype(np.int64)
    loss = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0))
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)
    confidence = probabilities.max(axis=1)
    top2 = np.sort(probabilities, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]

    quantum_params = int(parameter_count * 0.6)
    classical_params = parameter_count - quantum_params
    resource_counts = {
        "trainable_parameters_total": parameter_count,
        "trainable_parameters_quantum": quantum_params,
        "trainable_parameters_classical": classical_params,
        "trainable_parameters_unclassified": 0,
        "gate_counts": {"ry": gate_count // 3, "rz": gate_count // 3, "cx": gate_count - 2 * (gate_count // 3)},
        "quantum_gate_count_total": gate_count,
        "quantum_one_qubit_gates": 2 * (gate_count // 3),
        "quantum_two_qubit_gates": gate_count - 2 * (gate_count // 3),
        "gate_count_scope": "fixed encoder plus downstream variational circuit",
        "qcnn_frontend_included": False if metadata.get("architecture") == "qcnn" else None,
    }
    return {
        "X": torch.tensor(probabilities),
        "pv": torch.tensor(probabilities),
        "pv_dim": num_classes,
        "y_true": torch.tensor(labels),
        "y_pred": torch.tensor(prediction),
        "correct": torch.tensor(correctness),
        "membership": torch.tensor(membership),
        "split": torch.tensor(membership),
        "meta": {
            **metadata,
            "target_id": target_id,
            "membership_convention": "0=member",
        },
        "stats": {
            "loss": torch.tensor(loss, dtype=torch.float32),
            "entropy": torch.tensor(entropy, dtype=torch.float32),
            "conf": torch.tensor(confidence, dtype=torch.float32),
            "margin": torch.tensor(margin, dtype=torch.float32),
        },
        "target_metrics": {
            "train": {"acc": train_acc, "loss": 1.0 - train_acc, "N": n_member},
            "valid": {"acc": valid_acc, "loss": 1.0 - valid_acc, "N": n_nonmember},
            "test": {"acc": test_acc, "loss": 1.0 - test_acc, "N": n_nonmember},
        },
        "resource_counts": resource_counts,
    }


def save_run(root: Path, target: dict[str, Any], payload: dict[str, Any]) -> None:
    run_dir = root / str(target["experiment"]) / str(target["target_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, run_dir / "target_attack_data.pt")
    torch.save(
        {
            "vqc_circuit.theta": torch.randn(8),
            "head.linear.weight": torch.randn(2, 4),
        },
        run_dir / "target_model.pt",
    )
    (run_dir / "train.log").write_text("synthetic smoke run completed\n", encoding="utf-8")
    (run_dir / "target_export_summary.json").write_text(
        json.dumps(
            {
                "meta": payload["meta"],
                "target_metrics": payload["target_metrics"],
                "resource_counts": payload["resource_counts"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def create_matched(root: Path, rng: np.random.Generator) -> Path:
    targets: list[dict[str, Any]] = []
    for pair_index, risk in enumerate(("low", "high"), start=1):
        pair_id = f"SmokePair{pair_index}"
        gap = 0.02 if risk == "low" else 0.20
        for feature_map in ("z", "zz"):
            for model_seed in (43, 44, 45):
                target_id = f"{pair_id}_{feature_map}_s{model_seed}"
                target = {
                    "target_id": target_id,
                    "experiment": "matched_gap_mia",
                    "pair_id": pair_id,
                    "role": f"{risk}_gap_{feature_map}",
                    "dataset": "Smoke",
                    "architecture": "qnn",
                    "fm_kind": feature_map,
                    "n_wires": 4,
                    "reps": 1,
                    "depth": 2,
                    "ql_ent": "linear",
                    "ql_op": "crz",
                    "model_seed": model_seed,
                    "data_seed": 43,
                    "seed": model_seed,
                    "structural_cell_id": f"{pair_id}|{feature_map}",
                }
                train_acc = 0.78 + gap + 0.003 * (model_seed - 44)
                test_acc = 0.78 + (0.002 if feature_map == "zz" else 0.0)
                payload = build_payload(
                    rng,
                    target_id=target_id,
                    metadata=target,
                    train_acc=train_acc,
                    valid_acc=test_acc - 0.01,
                    test_acc=test_acc,
                    member_signal=0.01 if risk == "low" else 0.13,
                    num_classes=2,
                    gate_count=40,
                    parameter_count=70,
                )
                save_run(root, target, payload)
                targets.append(target)
    path = root.parent / "matched_targets.csv"
    pd.DataFrame(targets).to_csv(path, index=False)
    return path


def create_factorial(root: Path, rng: np.random.Generator) -> Path:
    targets: list[dict[str, Any]] = []
    fm_effect = {"z": 0.06, "zz": 0.05, "eff_su2": 0.01}
    for feature_map in ("z", "zz"):
        for repetitions in (1, 5):
            for depth in (2, 6):
                role = f"{feature_map}_r{repetitions}_d{depth}"
                for model_seed in (43, 44, 45):
                    target_id = f"SmokeFactorial_{role}_s{model_seed}"
                    target = {
                        "target_id": target_id,
                        "experiment": "multiseed_factorial",
                        "role": role,
                        "dataset": "MNIST",
                        "architecture": "qnn",
                        "fm_kind": feature_map,
                        "n_wires": 6,
                        "reps": repetitions,
                        "depth": depth,
                        "ql_ent": "linear",
                        "ql_op": "crz",
                        "model_seed": model_seed,
                        "data_seed": 43,
                        "seed": model_seed,
                        "structural_cell_id": role,
                    }
                    gap = (
                        fm_effect[feature_map]
                        + 0.025 * (repetitions == 5)
                        + 0.012 * (depth == 6)
                        + 0.002 * (model_seed - 44)
                    )
                    train_acc = 0.82 + 0.02 * (repetitions == 5)
                    test_acc = train_acc - gap
                    gate_count = 30 + repetitions * 12 + depth * 10 + (10 if feature_map == "zz" else 0)
                    parameter_count = 60 + depth * 18
                    payload = build_payload(
                        rng,
                        target_id=target_id,
                        metadata=target,
                        train_acc=train_acc,
                        valid_acc=test_acc - 0.01,
                        test_acc=test_acc,
                        member_signal=0.02 + 0.7 * gap,
                        num_classes=4,
                        gate_count=gate_count,
                        parameter_count=parameter_count,
                    )
                    save_run(root, target, payload)
                    targets.append(target)
    path = root.parent / "factorial_targets.csv"
    pd.DataFrame(targets).to_csv(path, index=False)
    return path


def create_architecture(root: Path, rng: np.random.Generator) -> Path:
    targets: list[dict[str, Any]] = []
    architectures = ("qnn", "hqnn", "qcnn", "mlp_qnn")
    architecture_gap = {"qnn": 0.08, "hqnn": 0.04, "qcnn": 0.05, "mlp_qnn": 0.03}
    for role_index, role in enumerate(("low_reupload", "high_reupload")):
        for architecture in architectures:
            for model_seed in (43, 44, 45):
                target_id = f"SmokeArch_{role}_{architecture}_s{model_seed}"
                target = {
                    "target_id": target_id,
                    "experiment": "architecture_control",
                    "role": role,
                    "dataset": "MNIST",
                    "architecture": architecture,
                    "fm_kind": "z" if role != "high_reupload" else "zz",
                    "n_wires": 6,
                    "reps": 5 if role == "high_reupload" else 1,
                    "depth": 6 if role == "high_depth" else 2,
                    "ql_ent": "linear",
                    "ql_op": "crz",
                    "model_seed": model_seed,
                    "data_seed": 43,
                    "seed": model_seed,
                    "structural_cell_id": f"{role}|{architecture}",
                }
                gap = architecture_gap[architecture] + 0.025 * role_index + 0.002 * (model_seed - 44)
                train_acc = 0.88
                test_acc = train_acc - gap
                is_classical = architecture == "mlp_qnn"
                payload = build_payload(
                    rng,
                    target_id=target_id,
                    metadata=target,
                    train_acc=train_acc,
                    valid_acc=test_acc - 0.01,
                    test_acc=test_acc,
                    member_signal=0.02 + gap,
                    num_classes=4,
                    gate_count=0 if is_classical else 60 + role_index * 20,
                    parameter_count=110 + architectures.index(architecture) * 20,
                )
                save_run(root, target, payload)
                targets.append(target)
    path = root.parent / "architecture_targets.csv"
    pd.DataFrame(targets).to_csv(path, index=False)
    return path


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("reviewer_smoke"))
    parser.add_argument("--bootstrap", type=int, default=5)
    args = parser.parse_args()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    tools = Path(__file__).resolve().parent

    matched_root = work / "matched_runs"
    factorial_root = work / "factorial_runs"
    architecture_root = work / "architecture_runs"
    matched_targets = create_matched(matched_root, rng)
    factorial_targets = create_factorial(factorial_root, rng)
    architecture_targets = create_architecture(architecture_root, rng)

    matched_results = work / "matched_results"
    factorial_results = work / "factorial_results"
    architecture_results = work / "architecture_results"

    run(
        [
            sys.executable,
            str(tools / "extract_retrained_metrics.py"),
            "--attack-data-dir",
            str(matched_root),
            "--targets",
            str(matched_targets),
            "--out-dir",
            str(matched_results / "metrics"),
        ]
    )
    run(
        [
            sys.executable,
            str(tools / "verify_matched_pairs.py"),
            "--metrics",
            str(matched_results / "metrics" / "retrained_target_metrics_raw.csv"),
            "--out-dir",
            str(matched_results / "matching"),
            "--bootstrap",
            str(args.bootstrap),
        ]
    )
    run(
        [
            sys.executable,
            str(tools / "identify_missing_runs.py"),
            "--targets",
            str(matched_targets),
            "--run-root",
            str(matched_root),
            "--out-dir",
            str(matched_results / "inventory"),
        ]
    )
    run(
        [
            sys.executable,
            str(tools / "threshold_mia_analysis.py"),
            "--attack-data-dir",
            str(matched_root),
            "--targets",
            str(matched_targets),
            "--out-dir",
            str(matched_results / "mia"),
            "--bootstrap",
            str(args.bootstrap),
            "--threshold-folds",
            "2",
        ]
    )
    run(
        [
            sys.executable,
            str(tools / "label_only_correctness_attack.py"),
            "--attack-data-dir",
            str(matched_root),
            "--targets",
            str(matched_targets),
            "--out-dir",
            str(matched_results / "label_only"),
            "--bootstrap",
            str(args.bootstrap),
        ]
    )

    run(
        [
            sys.executable,
            str(tools / "extract_retrained_metrics.py"),
            "--attack-data-dir",
            str(factorial_root),
            "--targets",
            str(factorial_targets),
            "--out-dir",
            str(factorial_results / "metrics"),
        ]
    )
    run(
        [
            sys.executable,
            str(tools / "threshold_mia_analysis.py"),
            "--attack-data-dir",
            str(factorial_root),
            "--targets",
            str(factorial_targets),
            "--out-dir",
            str(factorial_results / "mia"),
            "--bootstrap",
            str(args.bootstrap),
            "--attacks",
            "loss",
            "--threshold-folds",
            "2",
        ]
    )
    run(
        [
            sys.executable,
            str(tools / "count_model_resources.py"),
            "--run-root",
            str(factorial_root),
            "--targets",
            str(factorial_targets),
            "--out-dir",
            str(factorial_results / "resources"),
            "--fail-on-missing-exact",
        ]
    )
    run(
        [
            sys.executable,
            str(tools / "mia_regression_analysis.py"),
            "--metrics",
            str(factorial_results / "metrics" / "retrained_target_metrics_raw.csv"),
            "--mia",
            str(factorial_results / "mia" / "threshold_mia_raw.csv"),
            "--resources",
            str(factorial_results / "resources" / "model_resources_raw.csv"),
            "--out-dir",
            str(factorial_results / "regression"),
            "--bootstrap",
            str(args.bootstrap),
        ]
    )

    run(
        [
            sys.executable,
            str(tools / "extract_retrained_metrics.py"),
            "--attack-data-dir",
            str(architecture_root),
            "--targets",
            str(architecture_targets),
            "--out-dir",
            str(architecture_results / "metrics"),
        ]
    )
    run(
        [
            sys.executable,
            str(tools / "threshold_mia_analysis.py"),
            "--attack-data-dir",
            str(architecture_root),
            "--targets",
            str(architecture_targets),
            "--out-dir",
            str(architecture_results / "mia"),
            "--bootstrap",
            str(args.bootstrap),
            "--attacks",
            "loss",
            "--threshold-folds",
            "2",
        ]
    )
    run(
        [
            sys.executable,
            str(tools / "count_model_resources.py"),
            "--run-root",
            str(architecture_root),
            "--targets",
            str(architecture_targets),
            "--out-dir",
            str(architecture_results / "resources"),
            "--fail-on-missing-exact",
        ]
    )
    run(
        [
            sys.executable,
            str(tools / "architecture_control_analysis.py"),
            "--metrics",
            str(architecture_results / "metrics" / "retrained_target_metrics_raw.csv"),
            "--mia",
            str(architecture_results / "mia" / "threshold_mia_raw.csv"),
            "--resources",
            str(architecture_results / "resources" / "model_resources_raw.csv"),
            "--out-dir",
            str(architecture_results / "analysis"),
            "--bootstrap",
            str(args.bootstrap),
        ]
    )

    dummy_repo = work / "dummy_repo"
    (dummy_repo / "experiments").mkdir(parents=True, exist_ok=True)
    (dummy_repo / "experiments" / "qurift_main.py").write_text(
        "print('dry-run placeholder')\n", encoding="utf-8"
    )
    run(
        [
            sys.executable,
            str(tools / "run_target_table_dgx.py"),
            "--targets",
            str(factorial_targets),
            "--repo-root",
            str(dummy_repo),
            "--out",
            str(work / "runner_dry_run"),
            "--gpus",
            "0,1",
            "--max-jobs",
            "2",
            "--dry-run",
        ]
    )
    geometry_targets = tools.parent / "reviewer_targets" / "geometry_targets.csv"
    run(
        [
            sys.executable,
            str(tools / "run_multiseed_geometry.py"),
            "--targets",
            str(geometry_targets),
            "--repo-root",
            str(dummy_repo),
            "--out-dir",
            str(work / "geometry_dry_run"),
            "--gpus",
            "0,1",
            "--seeds",
            "43",
            "--dry-run",
        ]
    )

    required_outputs = [
        matched_results / "matching" / "matched_pair_verification_raw.csv",
        matched_results / "mia" / "threshold_mia_raw.csv",
        factorial_results / "regression" / "mia_regression_coefficients.csv",
        architecture_results / "analysis" / "architecture_control_effects.csv",
    ]
    missing = [str(path) for path in required_outputs if not path.exists()]
    if missing:
        raise SystemExit(f"Smoke test missing outputs: {missing}")
    print(f"[OK] Non-noise toolkit smoke test passed: {work}")


if __name__ == "__main__":
    main()
