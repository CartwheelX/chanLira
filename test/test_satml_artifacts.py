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


if __name__ == "__main__":
    unittest.main()
