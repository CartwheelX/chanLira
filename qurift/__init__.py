"""QuRiFT research framework."""

from pathlib import Path


_version_file = Path(__file__).resolve().parents[1] / "torchquantum" / "__version__.py"
_version_ns = {}
exec(_version_file.read_text(encoding="utf-8"), _version_ns)
__version__ = _version_ns["version"]

__all__ = ["__version__"]
