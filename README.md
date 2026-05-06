
<p align="center">
  <img src="qurift_logo_1.png" alt="QuRiFT logo" width="420"/>
</p>

**QuRiFT** (**Quantum Risk and Inference Fault-line Tracer**) is a controlled audit framework for studying **structural privacy leakage in quantum machine learning (QML)**.

This repository accompanies the paper:

```text
Structural Privacy Vulnerabilities in Quantum Neural Networks
```

QuRiFT is designed to answer a specific question: **how much of membership-inference risk in QML is induced by circuit structure, especially the non-trainable classical-to-quantum encoder?**

Rather than treating privacy leakage only as a consequence of trainable model capacity, QuRiFT performs controlled interventions over QML design choices, keeps the training protocol fixed within each experimental family, and records utility, overfitting, and membership-inference signals.

---

## What QuRiFT Provides

QuRiFT provides an end-to-end experimental pipeline for:

- running controlled QML architecture sweeps,
- varying encoder and ansatz design factors,
- training QNN, HQNN, and QCNN-style models,
- exporting prediction vectors for member and non-member samples,
- selecting stress, baseline, and hard target configurations,
- training black-box membership-inference attacks,
- generating CSV summaries for paper tables and analysis.

The framework is intended for controlled privacy auditing, not for claiming hardware-level leakage. The reported experiments use noiseless simulation to isolate representation effects before hardware noise or backend-specific artifacts are introduced.

---

## Relationship to TorchQuantum

QuRiFT builds around [TorchQuantum](https://github.com/mit-han-lab/torchquantum) as the quantum primitive layer. TorchQuantum provides PyTorch-native quantum devices, gates, differentiable circuit execution, measurements, and GPU-backed simulation.

QuRiFT adds the privacy-audit layer on top of those primitives:

- experiment drivers,
- feature-map and ansatz configuration logic,
- QNN/HQNN/QCNN model wrappers,
- sweep orchestration,
- result logging,
- target-table construction,
- prediction-vector export,
- membership-inference attack training.

TorchQuantum should be credited as the upstream quantum simulation and circuit-execution foundation. QuRiFT is the audit and analysis framework built around it.

---

## Repository Structure

```text
QuRiFT/
├── experiments/
│   ├── qurift_main.py
│   ├── full_sweep_qnn_moons.py
│   ├── full_sweep_qnn_circles.py
│   ├── full_sweep_qnn_blobs.py
│   ├── run_mnist_sweep_qnn.py
│   ├── run_mnist_sweep_hqnn.py
│   ├── run_mnist_sweep_qcnn.py
│   └── gen_results/
│       ├── make_runid_tables_for_mia.py
│       ├── qnn_qcnn_hqnn_models_comp_mnist.py
│       ├── run_selected_configs_for_mia.py
│       ├── train_mia_attack.py
│       └── run_train_mia_attack_cvholdout_multigpu.py
├── data/
│   └── MNIST/raw/
├── requirements.txt
├── setup.py
└── README.md
```

Generated outputs such as checkpoints, sweep folders, CSV summaries, plots, and attack outputs are intentionally kept out of Git unless they are curated paper artifacts.

---

## Main Entry Point

The central experiment driver is:

```bash
python experiments/qurift_main.py
```

After installation, the same driver can also be called through the console command:

```bash
qurift
```

Most sweep scripts in `experiments/` are wrappers around `experiments/qurift_main.py` with pre-defined experimental grids.

---

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/CartwheelX/QuRiFT.git
cd QuRiFT
pip install --editable .
```

This installs the QuRiFT package and exposes the console command:

```bash
qurift
```

If you prefer to install dependencies separately:

```bash
pip install -r requirements.txt
pip install --editable . --no-deps
```

The installable distribution name is:

```text
qurift
```

Current package version:

```text
0.1.0
```

---

## Dependencies

Recommended environment:

- Python `>=3.7, <=3.9`
- PyTorch `>=1.8.0`
- `configargparse >= 0.14`
- CUDA-enabled NVIDIA GPU for larger sweeps and target-model retraining

Python 3.10 may cause compatibility issues with some TorchQuantum/Qiskit dependency combinations, including issues around the `concurrent` package in older stacks. Python 3.8 or 3.9 is recommended for reproducibility.

The experiment code imports TorchQuantum primitives through:

```python
import torchquantum as tq
```

---

## Quick Start

Run a small synthetic Moons experiment:

```bash
python experiments/qurift_main.py \
  --dataset moons \
  --model-type qnn \
  --n-wires 4 \
  --depth 2 \
  --epochs 1 \
  --train_target \
  --extra-feats
```

Equivalent installed command:

```bash
qurift \
  --dataset moons \
  --model-type qnn \
  --n-wires 4 \
  --depth 2 \
  --epochs 1 \
  --train_target \
  --extra-feats
```

On Windows PowerShell, replace Linux/macOS line continuations `\` with `^`.

---

## Example: Export Prediction Vectors for MIA

The following example trains a QNN target model on Moons and exports prediction-vector data for membership-inference analysis:

```bash
python experiments/qurift_main.py \
  --dataset moons \
  --model-type qnn \
  --n-wires 4 \
  --depth 6 \
  --vector-train 50 \
  --vector-valid 50 \
  --vector-test 50 \
  --batch-size 8 \
  --epochs 100 \
  --moons-noise 0.3 \
  --fm-kind z \
  --fm-z-pad-mode wrap \
  --fm-z-reps 1 \
  --train_target \
  --extra-feats \
  --export-attack-data \
  --target-model-path checkpoints/moons_qnn.pt \
  --attack-data-out audit_outputs/moons_qnn_attack_data.pt
```

The exported attack data contains prediction vectors and labels needed to train black-box membership-inference attacks.

---

## Sweep Drivers

The main sweep drivers are:

```text
experiments/full_sweep_qnn_moons.py
experiments/full_sweep_qnn_circles.py
experiments/full_sweep_qnn_blobs.py
experiments/run_mnist_sweep_qnn.py
experiments/run_mnist_sweep_hqnn.py
experiments/run_mnist_sweep_qcnn.py
```

These scripts launch controlled sweeps across synthetic datasets and MNIST model families. They call `experiments/qurift_main.py` with different structural configurations.

---

## Structural Factors Swept by QuRiFT

QuRiFT varies QML structure while keeping the data and training protocol fixed within each experimental family. The main factors are:

| Factor | Example values / flags | Purpose |
|---|---|---|
| Feature-map family | `z`, `zz`, `pauli`, `eff_su2` | Tests encoder-induced representation effects |
| Feature-map repetitions | `--fm-*-reps` | Repeatedly injects input-dependent structure |
| Padding mode | e.g., `wrap` | Controls feature-to-qubit mapping when dimensions do not match |
| Feature-map entanglement | e.g., `linear`, `full` | Controls encoder entanglement topology |
| Circuit width | `--n-wires` | Changes number of qubits/wires |
| Variational depth | `--depth` | Changes trainable ansatz capacity |
| Q-layer entanglement | `--qlayer-ent-kind` | Controls trainable-layer connectivity |
| Q-layer two-qubit gate | `--qlayer-twoq-op` | Tests trainable entangling operation choice |
| Model family | `qnn`, `hqnn`, `qcnn`, `mlp_qnn` | Compares QML architecture families |

A key distinction in the paper is between **feature-map repetitions** and **variational depth**. Feature-map repetitions re-inject the input through the fixed encoder, similar in spirit to data re-uploading. Variational depth mainly increases the number of trainable operations after encoding.

---

## Model Families

QuRiFT supports the following model families:

- **`qnn`**: Dense quantum neural network with a quantum encoder, variational circuit, measurement, and classical classifier.
- **`hqnn`**: Hybrid CNN-QNN model with a trainable classical bottleneck before the quantum encoder and an MLP head after measurement.
- **`qcnn`**: Quantum-filter/quanvolutional front end with local quantum processing before the downstream encoder and classifier.
- **`mlp_qnn`**: Optional classical/MLP-style comparison path.

Synthetic benchmarks include:

```text
Moons, Circles, Blobs
```

The MNIST experiments use a four-class subset:

```text
{0, 1, 3, 8}
```

MNIST inputs are represented using compact `1x16` features before the main quantum encoder.

---

## MNIST Data Cache

The repository includes a small MNIST cache under:

```text
data/MNIST/raw
```

MNIST experiments use:

```python
root="./data"
```

Fresh clones can therefore run MNIST smoke tests without downloading MNIST again. If the cache is removed, TorchVision will attempt to download the dataset.

---

## Metrics Recorded

For every configuration, QuRiFT records utility and privacy-relevant signals, including:

- train, validation, and test loss,
- train, validation, and test accuracy,
- train-test accuracy gap,
- prediction vectors for member and non-member samples,
- output-derived attack features such as loss, entropy, confidence, margin, and correctness.

The train-test accuracy gap is used as a structural proxy for memorization pressure. Membership inference is then evaluated directly using exported prediction vectors.

---

## Membership-Inference Threat Model

QuRiFT evaluates membership inference in a strict black-box setting. The attacker observes only the target model's prediction vector for a queried sample.

The attacker does **not** access:

- model parameters,
- gradients,
- optimizer state,
- quantum states,
- circuit internals at inference time,
- target training data.

The attack objective is to distinguish member samples from non-member samples using output-derived signals.

---

## Result and MIA Workflow

The `experiments/gen_results/` directory tracks the scripts needed for paper-table generation and membership-inference attack training. Generated CSVs, plots, checkpoints, and attack outputs are ignored by Git unless intentionally curated.

A typical workflow is:

```text
1. Run QML sweeps.
2. Copy or collect the resulting sweep summary CSVs into experiments/gen_results/.
3. Generate target-configuration tables for MIA.
4. Retrain/export selected target models and prediction vectors.
5. Train membership-inference attacks.
6. Aggregate attack results for paper tables and plots.
```

---

## Step 1: Run Sweeps

Run the desired synthetic and MNIST sweep drivers, for example:

```bash
python experiments/full_sweep_qnn_moons.py
python experiments/full_sweep_qnn_circles.py
python experiments/full_sweep_qnn_blobs.py
python experiments/run_mnist_sweep_qnn.py
python experiments/run_mnist_sweep_hqnn.py
python experiments/run_mnist_sweep_qcnn.py
```

Each sweep creates a timestamped output directory. Examples include:

```text
sweep_full_pipeline_moons_<timestamp>/
sweep_full_pipeline_circles_<timestamp>/
sweep_full_pipeline_blobs_<timestamp>/
mnist_extensive_sweep_qnn_<timestamp>/
hqnn_sweep_<timestamp>/
qcnn_sweep_100_<timestamp>/
```

The corresponding CSV summaries should be copied into `experiments/gen_results/` before target-table generation.

Expected MNIST architecture summary names:

```text
experiments/gen_results/qnn_extensive_results.csv
experiments/gen_results/hqnn_extensive_results.csv
experiments/gen_results/qcnn_extensive_results.csv
```

Synthetic sweep summaries are similarly collected from their generated sweep directories.

---

## Step 2: Generate Run-ID Tables for Selected MIA Targets

For a selected synthetic setup, use:

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

This script filters sweep results and constructs run-id tables for target-model retraining and MIA export.

---

## Step 3: Generate Matched Target Tables

Generate matched target-configuration CSVs for synthetic QNN and MNIST architecture comparisons:

```bash
python experiments/gen_results/qnn_qcnn_hqnn_models_comp_mnist.py
```

This step expects the sweep summary CSVs to be present under `experiments/gen_results/` with consistent architecture names.

Typical output tables include:

```text
experiments/gen_results/paper_arch_compare/synthetic_qnn_targets_table.csv
experiments/gen_results/paper_arch_compare/mnist_matched_runids_table.csv
```

These tables are later consumed by the target retraining and prediction-vector export script.

---

## Step 4: Retrain Selected Targets and Export Attack Data

Train/export selected target models and prediction-vector attack data for synthetic QNN targets:

```bash
python experiments/gen_results/run_selected_configs_for_mia.py \
  --targets experiments/gen_results/paper_arch_compare/synthetic_qnn_targets_table.csv \
  --out experiments/gen_results/paper_arch_compare/saved_models_for_mia \
  --save-model
```

Train/export selected target models and prediction-vector attack data for matched MNIST targets:

```bash
python experiments/gen_results/run_selected_configs_for_mia.py \
  --targets experiments/gen_results/paper_arch_compare/mnist_matched_runids_table.csv \
  --out experiments/gen_results/paper_arch_compare/saved_models_for_mia \
  --save-model
```

The output directory stores trained target checkpoints and exported attack-data files.

---

## Step 5: Train Membership-Inference Attacks

Train MLP membership-inference attacks on a single GPU:

```bash
python experiments/gen_results/train_mia_attack.py \
  --attack-data-dir experiments/gen_results/paper_arch_compare/saved_models_for_mia \
  --out experiments/gen_results/paper_arch_compare/mia_results \
  --test-ratio 0.2 --cv-folds 5 \
  --tune --n-trials 30 --max-epochs 200 --patience 15 \
  --device cuda --seed 42
```

Train attacks with the multi-GPU launcher:

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

---

## Important Generated Files

The following files are commonly used in the paper analysis pipeline:

```text
experiments/gen_results/paper_arch_compare/synthetic_qnn_targets_table.csv
experiments/gen_results/paper_arch_compare/mnist_matched_runids_table.csv
```

They are generated from sweep summary CSVs and are used as target lists for `run_selected_configs_for_mia.py`.

The target retraining/export step then produces saved models and attack-data files under:

```text
experiments/gen_results/paper_arch_compare/saved_models_for_mia/
```

The MIA training step produces attack results under directories such as:

```text
experiments/gen_results/paper_arch_compare/mia_results/
experiments/gen_results/paper_arch_compare/mia_results_multiGPU/
```

---


## Attribution

QuRiFT builds on TorchQuantum by the MIT HAN Lab and contributors:

```text
https://github.com/mit-han-lab/torchquantum
```

TorchQuantum is distributed under the MIT License. Preserve upstream license and attribution notices when redistributing code derived from or bundled with TorchQuantum.

---

## Citation

If you use QuRiFT, please cite the accompanying paper:

```bibtex
@misc{qurift2026,
  title        = {Structural Privacy Vulnerabilities in Quantum Neural Networks},
  author       = {Anonymous Authors},
  year         = {2026},
  note         = {QuRiFT: Quantum Risk and Inference Fault-line Tracer}
}
```

Update the BibTeX entry with the final author list, venue, and DOI once the paper is public.
