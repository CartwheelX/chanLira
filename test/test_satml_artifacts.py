from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from satml_tools.generate_satml_artifacts import DEFAULT_INPUTS, generate


class SaTMLArtifactTests(unittest.TestCase):
    def test_generates_tables_figures_and_missing_input_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factorial = root / "paired.csv"
            pd.DataFrame(
                [
                    {
                        "outcome": "auc", "attack": "loss", "factor": "repetitions",
                        "contrast": "5 - 1", "mean_difference": 0.04,
                        "sd_across_blocks": 0.02, "ci95_low": 0.01, "ci95_high": 0.07,
                        "n_independent_blocks": 8,
                    }
                ]
            ).to_csv(factorial, index=False)
            inputs = {name: root / f"missing_{name}.csv" for name in DEFAULT_INPUTS}
            inputs["factorial"] = factorial
            output = root / "artifacts"
            manifest = generate(inputs, output)
            self.assertEqual(set(manifest["inputs_loaded"]), {"factorial"})
            self.assertIn("geometry", manifest["inputs_missing"])
            self.assertFalse(manifest["family_errors"])
            self.assertTrue((output / "tables" / "satml_tables.md").exists())
            self.assertTrue((output / "tables" / "satml_tables.tex").exists())
            self.assertTrue((output / "figures" / "factorial_attack_effects.png").exists())
            self.assertTrue((output / "figures" / "factorial_attack_effects.pdf").exists())
            text = (output / "tables" / "satml_tables.md").read_text(encoding="utf-8")
            self.assertIn("0.040 ± 0.020 [0.010, 0.070]", text)

    def test_generates_n1_n2_n3_noise_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cells = pd.DataFrame([{
                "fm_kind": "z", "reps": 1, "depth": 2, "mode": "exact",
                "queries": 0, "shots": 0, "attack": "loss",
                "n_trained_checkpoints": 3, "mean_auc": 0.60,
                "sd_across_trained_checkpoints": 0.02,
            }])
            effect = pd.DataFrame([{
                "attack": "loss", "mode": "noisy_shot", "queries": 1,
                "shots": 512, "effect": "repetition_5_minus_1_at_depth2",
                "fm_kind": "z", "scope": "encoder_specific", "n_paired_units": 3,
                "n_model_seed_blocks": 3, "mean": 0.04, "sd": 0.01,
                "ci95_low": 0.02, "ci95_high": 0.06,
            }])
            query = pd.DataFrame([{
                "mode": "noisy_shot", "aggregation": "mean_api_probabilities",
                "attack": "loss", "contrast": "equal_total_5x512_minus_1x2560",
                "n_target_checkpoints": 6, "mean_auc_difference": 0.02,
                "sd_across_target_checkpoints": 0.01, "ci95_low": 0.00,
                "ci95_high": 0.04,
            }])
            lira = pd.DataFrame([{
                "mode": "noisy_shot", "shots": 512, "attack": "online_fixed",
                "contrast": "zz_r5_d6 - eff_su2_r1_d6", "n_paired_model_seeds": 3,
                "mean_auc_difference": 0.05, "sd_across_model_seeds": 0.02,
                "ci95_low": 0.01, "ci95_high": 0.08,
            }])
            paths = {
                "noise_n1_cells": cells,
                "noise_n1_effects": effect,
                "noise_n1_moderation": effect,
                "noise_n2_queries": query,
                "noise_n3_lira": lira,
            }
            inputs = {name: root / f"missing_{name}.csv" for name in DEFAULT_INPUTS}
            for name, frame in paths.items():
                inputs[name] = root / f"{name}.csv"
                frame.to_csv(inputs[name], index=False)
            output = root / "artifacts"
            manifest = generate(inputs, output)
            self.assertFalse(manifest["family_errors"])
            for name in paths:
                self.assertTrue((output / "tables" / f"{name}.csv").is_file())
            self.assertTrue((output / "figures" / "noise_n1_structural_ordering.png").is_file())
            self.assertTrue((output / "figures" / "noise_n2_query_policy.pdf").is_file())
            self.assertTrue((output / "figures" / "noise_n3_lira.png").is_file())


if __name__ == "__main__":
    unittest.main()
