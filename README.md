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
experiments/qurift_main.py
```

Installable console entry point:

```bash
qurift
```

## Sweep Drivers

The main sweep scripts call `experiments/qurift_main.py`:

```text
experiments/full_sweep_qnn_moons.py
experiments/full_sweep_qnn_circles.py
experiments/full_sweep_qnn_blobs.py
experiments/run_mnist_sweep_qnn.py
experiments/run_mnist_sweep_hqnn.py
experiments/run_mnist_sweep_qcnn.py
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

The repository includes a small MNIST cache under:

```text
data/MNIST/raw
```

MNIST experiments use `root="./data"`, so fresh clones can run the MNIST smoke
tests without downloading MNIST again. If the cache is removed, TorchVision will
attempt to download the dataset.

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

This is the same editable-install workflow as TorchQuantum:

```bash
pip install --editable .
```

The difference is that this repository installs the QuRiFT distribution
(`qurift`) and uses the bundled TorchQuantum-compatible source tree as the
upstream quantum primitive layer. Dependencies are read from `requirements.txt`
through `setup.py`.

## Dependencies

- Python `>=3.7, <=3.9` is recommended. Python 3.10 may have a `concurrent`
  package issue with some Qiskit/TorchQuantum dependency combinations.
- PyTorch `>=1.8.0`
- `configargparse >= 0.14`
- GPU model training requires NVIDIA GPUs.

Install dependencies together with QuRiFT:

```bash
pip install --editable .
```

If you need to install dependencies separately:

```bash
pip install -r requirements.txt
pip install --editable . --no-deps
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
python experiments/qurift_main.py --dataset moons --model-type qnn --n-wires 4 --depth 2 --epochs 1 --train_target --extra-feats
```

Equivalent installed command:

```bash
qurift --dataset moons --model-type qnn --n-wires 4 --depth 2 --epochs 1 --train_target --extra-feats
```

Example with prediction-vector export for membership-inference analysis:

```bash
python experiments/qurift_main.py ^
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

## MIA Target Selection and Attack Workflow

The `experiments/gen_results/` directory intentionally tracks only the
source scripts needed for paper-table generation and membership-inference
attack training. Generated CSVs, plots, saved models, and attack outputs remain
ignored by Git.

Generate a run-id grid for a selected synthetic setup:

```bash
python experiments/gen_results/make_runid_tables_for_mia.py \
  --dataset Moons --arch QNN \
  --out-dir experiments/gen_results/paper_arch_compare/retrain_grid \
  --fix "fm_kind=zz,fm_op_eff=rzz,n_wires=3,ql_ent=full,ql_op=crz,pad_mode=wrap,fm_ent=linear" \
  --reps "1,2,3,4,5" \
  --depths "2,3,4,5,6" \
  --train-min 0.99 --gap-lo 0.25 --gap-hi 0.30 \
  --prefer-low-test \
  --fallback-if-empty
```

Generate matched target-configuration CSVs for the synthetic QNN and MNIST
architecture comparisons:

```bash
python experiments/gen_results/qnn_qcnn_hqnn_models_comp_mnist.py
```

Before running this step, place the MNIST sweep summaries in
`experiments/gen_results/` with consistent architecture names:

```text
experiments/gen_results/qnn_extensive_results.csv
experiments/gen_results/hqnn_extensive_results.csv
experiments/gen_results/qcnn_extensive_results.csv
```

This produces target tables such as:

```text
experiments/gen_results/paper_arch_compare/synthetic_qnn_targets_table.csv
experiments/gen_results/paper_arch_compare/mnist_matched_runids_table.csv
```

Train/export selected target models and prediction-vector attack data:

```bash
python experiments/gen_results/run_selected_configs_for_mia.py \
  --targets experiments/gen_results/paper_arch_compare/synthetic_qnn_targets_table.csv \
  --out experiments/gen_results/paper_arch_compare/saved_models_for_mia \
  --save-model

python experiments/gen_results/run_selected_configs_for_mia.py \
  --targets experiments/gen_results/paper_arch_compare/mnist_matched_runids_table.csv \
  --out experiments/gen_results/paper_arch_compare/saved_models_for_mia \
  --save-model
```

Train MLP membership-inference attacks on a single GPU:

```bash
python experiments/gen_results/train_mia_attack.py \
  --attack-data-dir experiments/gen_results/paper_arch_compare/saved_models_for_mia \
  --out experiments/gen_results/paper_arch_compare/mia_results \
  --test-ratio 0.2 --cv-folds 5 \
  --tune --n-trials 30 --max-epochs 200 --patience 15 \
  --device cuda --seed 42
```

Train MLP membership-inference attacks with the multi-GPU launcher:

```bash
python experiments/gen_results/run_train_mia_attack_cvholdout_multigpu.py \
  --attack-data-dir experiments/gen_results/paper_arch_compare/saved_models_for_mia \
  --out experiments/gen_results/paper_arch_compare/mia_results_multiGPU \
  --launcher \
  --device cuda \
  --tune --n-trials 120 --max-epochs 300 --patience 25 \
  --test-ratio 0.2 --cv-folds 5 \
  --jobs-per-gpu 4 \
  --cpu-threads 1 \
  --resume \
  --gpus 2,3,4,5,6 \
  --summary-only
```

These scripts expect the corresponding sweep summary CSVs to be present under
`experiments/gen_results/` before target-table generation.

## Repository Notes

- The main QuRiFT driver is `experiments/qurift_main.py`.
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
