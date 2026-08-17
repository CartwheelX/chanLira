#!/usr/bin/env python3
"""Offline unit smoke tests for Checkpoint 4 helpers.

This test does not require Qiskit, IBM credentials, TorchQuantum, or a GPU.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from qurift_qiskit_bridge import counts_to_z_expectations
from qurift_noisy_eval import probability_statistics, simulator_seed_bootstrap


def main() -> None:
    # 2-qubit counts: 00 and 11 equally likely -> both Z expectations are 0.
    values = counts_to_z_expectations({"00": 50, "11": 50}, 2)
    assert np.allclose(values, [0.0, 0.0])

    # q0=0 always, q1 balanced. Qiskit keys are c1c0.
    values = counts_to_z_expectations({"00": 50, "10": 50}, 2)
    assert np.allclose(values, [1.0, 0.0])

    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float32)
    labels = torch.tensor([0, 1])
    stats = probability_statistics(probs, labels)
    assert stats["correctness"].tolist() == [1.0, 1.0]
    assert torch.all(stats["loss"] > 0)
    assert torch.allclose(stats["confidence"], torch.tensor([0.9, 0.8]))

    low, high, valid = simulator_seed_bootstrap(np.array([0.5, 0.6, 0.7]), 100, 1)
    assert valid == 100
    assert np.isfinite(low) and np.isfinite(high) and low <= high

    print("[OK] Checkpoint 4 offline smoke tests passed.")


if __name__ == "__main__":
    main()
