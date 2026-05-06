"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors
QuRiFT additions copyright (c) 2026 QuRiFT contributors

This project is a research fork of TorchQuantum. The Python import namespace
`torchquantum` is intentionally preserved for compatibility with upstream code.
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
        description="QuRiFT: quantum representation and privacy audit framework built on TorchQuantum",
        long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        url="https://github.com/CartwheelX/QuRiFT",
        author="QuRiFT contributors; based on TorchQuantum by the MIT HAN Lab",
        license="MIT",
        install_requires=read_requirements(),
        extras_require={"doc": ["nbsphinx", "recommonmark"]},
        python_requires=">=3.8",
        include_package_data=True,
        packages=find_packages(),
        entry_points={
            "console_scripts": [
                "qurift=qurift.cli:main",
            ],
        },
    )
