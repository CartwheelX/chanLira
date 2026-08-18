from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from satml_tools.status_progress import summarize


class StatusProgressTests(unittest.TestCase):
    def test_reports_observed_remaining_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.csv"
            pd.DataFrame(
                [{"name": "a", "status": "ok", "log": "a.log"},
                 {"name": "b", "status": "error", "log": "b.log"}]
            ).to_csv(path, index=False)
            text = summarize(path, expected=5)
            self.assertIn("observed=2", text)
            self.assertIn("remaining=3", text)
            self.assertIn("error=1", text)
            self.assertIn("b.log", text)


if __name__ == "__main__":
    unittest.main()
