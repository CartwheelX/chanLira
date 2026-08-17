#!/usr/bin/env python3
"""Offline synthetic test for Checkpoint 5 selection and aggregation scripts."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make_predictions(target_id: str, model_seed: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    labels = np.tile([0, 1], 20)
    membership = np.concatenate([np.ones(20, dtype=int), np.zeros(20, dtype=int)])
    labels = labels[:40]
    for mode, shots, sim_seed, shift in [
        ("exact", 0, -1, 0.00),
        ("ideal_shot", 128, 0, -0.03),
        ("ideal_shot", 128, 1, -0.01),
        ("noisy_shot", 128, 0, -0.08),
        ("noisy_shot", 128, 1, -0.06),
    ]:
        base_true = 0.75 + 0.12 * membership + shift + rng.normal(0, 0.04, size=40)
        base_true = np.clip(base_true, 0.51, 0.98)
        for index in range(40):
            p_true = base_true[index]
            if labels[index] == 0:
                p0, p1 = p_true, 1 - p_true
            else:
                p0, p1 = 1 - p_true, p_true
            prediction = int(p1 > p0)
            confidence = max(p0, p1)
            loss = -np.log(max(1e-12, p_true))
            entropy = -(p0 * np.log(max(p0, 1e-12)) + p1 * np.log(max(p1, 1e-12)))
            rows.append({
                "target_id": target_id,
                "mode": mode,
                "shots": shots,
                "simulator_seed": sim_seed,
                "transpiler_seed": 2026,
                "backend_name": "synthetic_backend",
                "noise_model_loaded": mode == "noisy_shot",
                "quantum_execution_scope": "synthetic_test",
                "sample_id": f"sample_{index}",
                "source_split": "train" if membership[index] else "test",
                "source_index": index,
                "label": int(labels[index]),
                "membership": int(membership[index]),
                "prediction": prediction,
                "loss": float(loss),
                "entropy": float(entropy),
                "confidence": float(confidence),
                "margin": float(abs(p0 - p1)),
                "correctness": int(prediction == labels[index]),
                "p_0": float(p0),
                "p_1": float(p1),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("checkpoint5_smoke"))
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    raw = args.work_dir / "raw"
    raw.mkdir(exist_ok=True)

    targets = pd.DataFrame([
        {
            "target_id": "synthetic_target_s43",
            "experiment": "multiseed_factorial",
            "dataset": "MNIST",
            "architecture": "QNN",
            "fm_kind": "zz",
            "reps": 5,
            "depth": 6,
            "n_wires": 6,
            "model_seed": 43,
            "data_seed": 43,
            "structural_cell_id": "zz_r5_d6",
            "risk_role": "high_risk_anchor",
            "comparison_group": "zz_repetition_d6",
            "selection_order": 5,
            "original_acc_train": 0.765,
            "original_acc_test": 0.340,
            "original_gap": 0.425,
            "selection_basis": "synthetic_test",
        },
        {
            "target_id": "synthetic_target_s44",
            "experiment": "multiseed_factorial",
            "dataset": "MNIST",
            "architecture": "QNN",
            "fm_kind": "zz",
            "reps": 5,
            "depth": 6,
            "n_wires": 6,
            "model_seed": 44,
            "data_seed": 43,
            "structural_cell_id": "zz_r5_d6",
            "risk_role": "high_risk_anchor",
            "comparison_group": "zz_repetition_d6",
            "selection_order": 5,
            "original_acc_train": 0.765,
            "original_acc_test": 0.340,
            "original_gap": 0.425,
            "selection_basis": "synthetic_test",
        },
    ])
    targets_path = args.work_dir / "targets.csv"
    targets.to_csv(targets_path, index=False)
    rng = np.random.default_rng(2026)
    for _, row in targets.iterrows():
        directory = raw / row["target_id"]
        directory.mkdir(parents=True, exist_ok=True)
        predictions = make_predictions(row["target_id"], int(row["model_seed"]), rng)
        predictions.to_csv(directory / "per_sample_predictions.csv", index=False)
        manifest = predictions[predictions["mode"] == "exact"][[
            "target_id", "sample_id", "source_split", "source_index", "label", "membership"
        ]]
        manifest.to_csv(directory / "sample_manifest.csv", index=False)
        pd.DataFrame({"status": ["ok"]}).to_csv(directory / "condition_status.csv", index=False)

    script = Path(__file__).with_name("combine_noisy_results.py")
    output = args.work_dir / "combined"
    subprocess.run([
        sys.executable, str(script),
        "--targets", str(targets_path),
        "--raw-root", str(raw),
        "--out-dir", str(output),
        "--bootstrap", "100",
    ], check=True)
    required = [
        "noisy_mia_results_long.csv",
        "noisy_mia_simulator_summary.csv",
        "noisy_mia_target_seed_summary.csv",
        "noisy_changes_long.csv",
        "analysis_metadata.json",
    ]
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise RuntimeError(f"Smoke test missing outputs: {missing}")
    print(f"[OK] Checkpoint 5 synthetic smoke test passed: {output.resolve()}")


if __name__ == "__main__":
    main()
