# QuRiFT

**QuRiFT** (**Quantum Risk and Inference Fault-line Tracer**) is a controlled
audit framework for structural privacy analysis in quantum machine learning
(QML).

This repository accompanies the paper:

```text
Structural Privacy Vulnerabilities in Quantum Neural Networks
```

QuRiFT studies whether membership-inference risk in quantum neural networks is
shaped by circuit structure, especially the non-trainable classical-to-quantum
encoder. The framework performs controlled interventions over QML architecture
while keeping the training protocol fixed, then records utility, overfitting,
and membership-inference signals.

## Central Hypothesis

QuRiFT audits the pathway:

```text
encoder U_phi(x)
  -> encoded state rho_phi(x)
  -> Hilbert-Schmidt kernel geometry K_phi
  -> train-test asymmetry
  -> membership-inference signal
```

The core claim is not that the encoder reveals membership directly. Instead,
the encoder shapes the Hilbert-space representation on which the trainable
ansatz operates. Some feature maps and data-reuploading-style repetitions can
make training samples easier to fit than unseen samples, producing a larger
generalization gap and stronger output-based membership signals.

## Role of TorchQuantum

QuRiFT uses [TorchQuantum](https://github.com/mit-han-lab/torchquantum) as the
upstream quantum primitive layer. TorchQuantum provides PyTorch-native quantum
devices, gates, differentiable circuit execution, measurements, and GPU-backed
simulation. QuRiFT provides the audit framework, sweep orchestration, model
wrappers, feature-map experiments, logging, and privacy-analysis pipeline built
around those primitives.

All reported experiments in the paper use controlled noiseless TorchQuantum
simulation to isolate encoder-induced representation effects before introducing
backend-specific hardware noise.

## Main Entry Point

The central QuRiFT experiment driver is:

```bash
examples/mnist/qurift_main.py
```

Installable console entry point:

```bash
qurift
```

## Sweep Drivers

The main sweep scripts call `examples/mnist/qurift_main.py`:

```text
examples/mnist/full_sweep_qnn_moons.py
examples/mnist/full_sweep_qnn_circles.py
examples/mnist/full_sweep_qnn_blobs.py
examples/mnist/run_mnist_sweep_qnn.py
examples/mnist/run_mnist_sweep_hqnn.py
examples/mnist/run_mnist_sweep_qcnn.py
```

These scripts launch controlled sweeps over synthetic datasets and MNIST model
families.

## What QuRiFT Sweeps

QuRiFT varies structural QML components while holding the data protocol and
optimizer fixed within each experimental family:

- feature-map family: `z`, `zz`, `pauli`, `eff_su2`
- feature-map repetitions via `--fm-*-reps`
- feature-map padding mode
- feature-map entanglement topology
- circuit width via `--n-wires`
- variational depth via `--depth`
- q-layer entanglement topology via `--qlayer-ent-kind`
- q-layer two-qubit operation via `--qlayer-twoq-op`
- model family: `qnn`, `hqnn`, `qcnn`, `mlp_qnn`

The paper emphasizes the distinction between feature-map repetitions and
variational depth: repetitions repeatedly inject input-dependent structure into
the fixed encoder, while depth primarily increases trainable ansatz capacity.

## Model Families

QuRiFT supports:

- `qnn`: dense QNN with quantum encoder, variational circuit, measurement, and
  classical classifier.
- `hqnn`: hybrid CNN-QNN model with a trainable classical bottleneck before the
  quantum encoder and an MLP head after measurement.
- `qcnn`: quantum-filter/quanvolutional front end with local quantum processing
  before the downstream encoder and classifier.
- `mlp_qnn`: optional classical/MLP-style comparison path.

Synthetic benchmarks include Moons, Circles, and Blobs. The MNIST setting uses
a four-class subset over digits `{0, 1, 3, 8}` with compact `1x16`
representations before the main quantum encoder.

## Metrics and Privacy Evaluation

For every configuration, QuRiFT records:

- train, validation, and test loss
- train, validation, and test accuracy
- train-test accuracy gap
- probability vectors for member and non-member samples
- output-derived attack features such as loss, entropy, confidence, margin, and
  correctness

Membership inference is evaluated in a black-box setting: the attacker observes
the target model's prediction vector but does not access parameters, gradients,
quantum states, training data, or optimizer state.

## Installation

```bash
git clone https://github.com/CartwheelX/QuRiFT.git
cd QuRiFT
pip install --editable .
```

The experiment code uses TorchQuantum primitives through:

```python
import torchquantum as tq
```

The installable distribution name is:

```text
qurift
```

Current package version:

```text
0.1.0
```

## Quick Start

Run a small synthetic Moons experiment:

```bash
python examples/mnist/qurift_main.py --dataset moons --model-type qnn --n-wires 4 --depth 2 --epochs 1 --train_target --extra-feats
```

Equivalent installed command:

```bash
qurift --dataset moons --model-type qnn --n-wires 4 --depth 2 --epochs 1 --train_target --extra-feats
```

Example with prediction-vector export for membership-inference analysis:

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

## Repository Notes

- The main QuRiFT driver is `examples/mnist/qurift_main.py`.
- Generated datasets, checkpoints, circuits, logs, plots, and attack outputs
  should not be committed unless intentionally curated as paper artifacts.
- Large result artifacts should be stored in GitHub Releases, Zenodo, or an
  institutional data repository.

## Paper Citation Stability

For a stable paper artifact, tag the version used in the paper:

```bash
git tag -a v0.1.0 -m "QuRiFT v0.1.0"
git push origin v0.1.0
```

Then create a GitHub release for `v0.1.0`. For a DOI, archive the GitHub release
with Zenodo.

## Attribution

QuRiFT builds on TorchQuantum by the MIT HAN Lab and contributors:

https://github.com/mit-han-lab/torchquantum

TorchQuantum is distributed under the MIT License. Preserve upstream license and
attribution notices when redistributing code derived from or bundled with
TorchQuantum.
