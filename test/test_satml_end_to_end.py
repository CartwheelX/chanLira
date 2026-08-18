from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qurift.satml_data import CREDIT_TARGET_COLUMN, write_credit_snapshot


class SaTMLEndToEndTests(unittest.TestCase):
    def test_credit_target_trains_and_exports_provenance(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        rng = np.random.default_rng(31)
        rows = 160
        frame = pd.DataFrame(
            {
                "LIMIT_BAL": rng.normal(size=rows),
                "AGE": rng.normal(size=rows),
                "BILL_AMT1": rng.normal(size=rows),
                "SEX": np.tile([1, 2], rows // 2),
                "EDUCATION": np.tile([1, 2, 3, 4], rows // 4),
                CREDIT_TARGET_COLUMN: np.tile([0, 0, 0, 1], rows // 4),
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "credit.csv.gz"
            model = root / "model.pt"
            attack = root / "attack.pt"
            metrics = root / "metrics.json"
            preprocessor = root / "preprocessor.joblib"
            provenance = root / "provenance.json"
            write_credit_snapshot(frame, data, {"provider": "integration-test"})
            command = [
                sys.executable, str(repo / "experiments" / "qurift_main.py"),
                "--dataset", "credit_default", "--model-type", "qnn",
                "--target-id", "integration_credit", "--experiment-id", "integration",
                "--data-seed", "17", "--model-seed", "29",
                "--vector-train", "32", "--vector-valid", "16", "--vector-test", "48",
                "--credit-data-path", str(data), "--credit-pca-components", "2",
                "--n-wires", "2", "--depth", "1", "--batch-size", "8", "--epochs", "1",
                "--fm-kind", "z", "--fm-z-reps", "1", "--fm-z-alpha", "0.5",
                "--qlayer-ent-kind", "linear", "--qlayer-twoq-op", "crz",
                "--train_target", "--export-attack-data", "--attack-feature-mode", "pv+stats",
                "--target-model-path", str(model), "--attack-data-out", str(attack),
                "--attack-metrics-out", str(metrics), "--preprocessor-out", str(preprocessor),
                "--dataset-provenance-out", str(provenance),
            ]
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": "",
                    "QURIFT_DISABLE_DEBUG_EXPORTS": "1",
                    "QURIFT_DISABLE_CIRCUIT_EXPORTS": "1",
                    "PYTHONPATH": str(repo),
                }
            )
            completed = subprocess.run(
                command, cwd=repo, env=environment, capture_output=True, text=True, timeout=180
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            for path in (model, attack, metrics, preprocessor, provenance):
                self.assertTrue(path.exists() and path.stat().st_size > 0, path)
            try:
                payload = torch.load(attack, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(attack, map_location="cpu")
            self.assertEqual(payload["meta"]["dataset"], "credit_default")
            self.assertEqual(payload["meta"]["feature_angle_scale"], 0.5)
            self.assertEqual(int((payload["membership"] == 0).sum()), 32)
            self.assertEqual(int((payload["membership"] == 1).sum()), 48)


if __name__ == "__main__":
    unittest.main()
