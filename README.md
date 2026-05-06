# QuRiFT

**QuRiFT** (**Quantum Risk and Inference Fault-line Tracer**) is a controlled
audit framework for structural privacy analysis in quantum machine learning
(QML).

This repository accompanies the paper:

```text
Structural Privacy Vulnerabilities in Quantum Neural Networks
```

QuRiFT studies how architectural choices in quantum neural networks influence
privacy-relevant behavior, including generalization gaps and
membership-inference risk. The framework keeps the training protocol controlled
while sweeping feature maps, circuit width, variational depth, entanglement
topology, two-qubit gate families, and model families such as QNN, HQNN, and
QCNN.

QuRiFT uses [TorchQuantum](https://github.com/mit-han-lab/torchquantum)
primitives for quantum devices, gates, encoders, measurements, and PyTorch
integration. TorchQuantum is the upstream quantum programming layer; QuRiFT is
the audit and experiment framework built on top of those primitives.

## Main Entry Point

The central QuRiFT experiment driver is:

```bash
examples/mnist/qurift_main.py
```

## Sweep Drivers

The main sweep scripts call `examples/mnist/qurift_main.py` and reproduce the
large controlled sweeps used by the project:

```text
examples/mnist/full_sweep_qnn_moons.py
examples/mnist/full_sweep_qnn_circles.py
examples/mnist/full_sweep_qnn_blobs.py
examples/mnist/run_mnist_sweep_qnn.py
examples/mnist/run_mnist_sweep_hqnn.py
examples/mnist/run_mnist_sweep_qcnn.py
```

## Installation

```bash
git clone https://github.com/CartwheelX/QuRiFT.git
cd QuRiFT
pip install --editable .
```

The editable install exposes the QuRiFT command:

```bash
qurift --help
```

The experiment code uses TorchQuantum primitives through:

```python
import torchquantum as tq
```

## Quick Start

Run a small synthetic moons experiment:

```bash
python examples/mnist/qurift_main.py --dataset moons --model-type qnn --n-wires 4 --depth 2 --epochs 1 --train_target --extra-feats
```

The same entry point is available through the installed console command:

```bash
qurift --dataset moons --model-type qnn --n-wires 4 --depth 2 --epochs 1 --train_target --extra-feats
```

Example with probability-vector export for membership-inference analysis:

```bash
python examples/mnist/qurift_main.py ^
  --dataset moons ^
  --model-type qnn ^
  --n-wires 4 ^
  --depth 6 ^
  --vector-train 50 ^
  --vector-valid 50 ^
  --vector-test 50 ^
  --batch-size 8 ^
  --epochs 100 ^
  --moons-noise 0.3 ^
  --fm-kind z ^
  --fm-z-pad-mode wrap ^
  --fm-z-reps 1 ^
  --train_target ^
  --extra-feats ^
  --export-attack-data ^
  --target-model-path checkpoints/moons_qnn.pt ^
  --attack-data-out audit_outputs/moons_qnn_attack_data.pt
```

On Linux/macOS, replace PowerShell line continuations `^` with `\`.

## What QuRiFT Sweeps

QuRiFT is designed to isolate structural privacy drivers by varying architectural
parameters while holding the optimizer and data protocol fixed. The main
structural controls include:

- feature-map family: `z`, `zz`, `pauli`, `eff_su2`
- feature-map repetitions and padding mode
- feature-map entanglement topology
- circuit width via `--n-wires`
- variational depth via `--depth`
- q-layer entanglement topology
- q-layer two-qubit operation
- model family: `qnn`, `hqnn`, `qcnn`, `mlp_qnn`

## Repository Notes

- The installable distribution name is `qurift`.
- The main QuRiFT driver is `examples/mnist/qurift_main.py`.
- TorchQuantum provides the upstream quantum primitives used by the framework.
- Generated datasets, model checkpoints, circuits, logs, plots, and audit
  outputs should not be committed unless they are intentionally curated for a
  paper artifact.
- Large result artifacts are better stored in GitHub Releases, Zenodo, or an
  institutional data repository.

## Paper Citation Stability

For a stable paper artifact, create a release tag after committing:

```bash
git tag -a v0.2.0 -m "QuRiFT v0.2.0"
git push origin v0.2.0
```

Then create a GitHub release for `v0.2.0`. For a DOI, archive the GitHub release
with Zenodo.

## Attribution

QuRiFT builds on TorchQuantum by the MIT HAN Lab and contributors:

https://github.com/mit-han-lab/torchquantum

TorchQuantum is distributed under the MIT License. Preserve upstream license and
attribution notices when redistributing code derived from or bundled with
TorchQuantum.
