"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors
QuRiFT additions copyright (c) 2026 QuRiFT contributors

QuRiFT uses TorchQuantum primitives as its upstream quantum programming layer.
"""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
VERSION = {}

with open(ROOT / "torchquantum" / "__version__.py", "r", encoding="utf-8") as version_file:
    exec(version_file.read(), VERSION)


def read_requirements():
    requirements = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


if __name__ == "__main__":
    setup(
        name="qurift",
        version=VERSION["version"],
        description="QuRiFT: Quantum Risk and Inference Fault-line Tracer for structural privacy analysis in QML",
        long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        url="https://github.com/CartwheelX/QuRiFT",
        author="QuRiFT contributors",
        license="MIT",
        install_requires=read_requirements(),
        extras_require={"doc": ["nbsphinx", "recommonmark"]},
        python_requires=">=3.7,<3.10",
        include_package_data=True,
        packages=find_packages(),
        entry_points={
            "console_scripts": [
                "qurift=qurift.cli:main",
            ],
        },
    )
