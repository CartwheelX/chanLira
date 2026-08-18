from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from satml_tools.import_legacy_mnist import ARTIFACT_NAMES, import_targets, sha256


class LegacyImportTests(unittest.TestCase):
    def test_import_is_byte_exact_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "old", root / "new"
            target = "target_a"
            source_dir = source / "reviewer_runs" / "multiseed_factorial" / target
            source_dir.mkdir(parents=True)
            for index, name in enumerate(ARTIFACT_NAMES):
                (source_dir / name).write_bytes(f"artifact-{index}".encode())
            first = import_targets(source, destination, [target])
            second = import_targets(source, destination, [target])
            self.assertTrue(all(row["status"] == "copied" for row in first))
            self.assertTrue(all(row["status"] == "already_identical" for row in second))
            for name in ARTIFACT_NAMES:
                source_file = source_dir / name
                destination_file = destination / "reviewer_runs" / "multiseed_factorial" / target / name
                self.assertEqual(sha256(source_file), sha256(destination_file))


if __name__ == "__main__":
    unittest.main()
