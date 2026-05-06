# QuRiFT

QuRiFT is a research framework for auditing quantum representation learning
and membership-inference behavior in quantum neural network experiments. It is
built as a fork of [TorchQuantum](https://github.com/mit-han-lab/torchquantum)
and keeps the `torchquantum` Python namespace so existing TorchQuantum layers,
datasets, plugins, and examples continue to work.

The current main experiment is:

```bash
examples/mnist/mnist_2qubit_4class.py
```

It supports MNIST and synthetic vector datasets, multiple quantum feature maps,
entanglement settings, QNN/HQNN/QCNN-style model choices, target-model training,
and probability-vector export for membership-inference audit data.

## Installation

```bash
git clone https://github.com/CartwheelX/QuRiFT.git
cd QuRiFT
pip install --editable .
```

The editable install exposes both:

```python
import torchquantum as tq
import qurift
```

## Quick Start

Run a small synthetic-data experiment:

```bash
qurift --dataset moons --model-type qnn --n-wires 2 --depth 2 --epochs 1 --train_target
```

Run the main file directly:

```bash
python examples/mnist/mnist_2qubit_4class.py --dataset moons --model-type qnn --n-wires 2 --depth 2 --epochs 1 --train_target
```

Export attack data after target-model training:

```bash
python examples/mnist/mnist_2qubit_4class.py ^
  --dataset moons ^
  --model-type qnn ^
  --n-wires 2 ^
  --depth 2 ^
  --epochs 5 ^
  --train_target ^
  --export_attack_data ^
  --target-model-path checkpoints/moons_qnn.pt ^
  --attack-data-out audit_outputs/moons_qnn_attack_data.pt
```

On Linux/macOS, replace `^` with `\` for multi-line commands.

## Repository Notes

- The installable distribution name is `qurift`.
- The upstream package namespace remains `torchquantum`.
- Generated datasets, model checkpoints, circuits, and audit outputs should not
  be committed unless they are intentionally curated for the paper.
- Large result artifacts are best uploaded as GitHub Releases, Zenodo records,
  or an institutional data repository and referenced from the paper separately.

## Uploading for a Paper Link

1. Create an empty GitHub repository named `QuRiFT`.
2. Update the placeholder URL in `setup.py` and this README from
   `CartwheelX` to your GitHub username or organization.
3. Review `git status` and commit the source files you want to publish.
4. Push:

```bash
git remote set-url origin https://github.com/CartwheelX/QuRiFT.git
git add .
git commit -m "Release QuRiFT research framework"
git push -u origin main
```

If you want a stable paper citation, create a GitHub release and archive it with
Zenodo to obtain a DOI.

## Attribution

QuRiFT is based on TorchQuantum by the MIT HAN Lab and contributors. The
original TorchQuantum project is available at:

https://github.com/mit-han-lab/torchquantum

TorchQuantum and QuRiFT are distributed under the MIT License. Preserve the
license and attribution notices when redistributing this fork.
