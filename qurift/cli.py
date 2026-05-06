"""Command line entry point for the QuRiFT audit experiment."""

from pathlib import Path
import runpy
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "mnist" / "mnist_2qubit_4class.py"
    if not script.exists():
        raise FileNotFoundError(
            "QuRiFT main experiment was not found. Run from a source checkout "
            "or use: python examples/mnist/mnist_2qubit_4class.py"
        )

    sys.path.insert(0, str(script.parent))
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")
