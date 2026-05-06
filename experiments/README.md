# QuRiFT Experiment Drivers

This directory contains the main experiment drivers for **QuRiFT**, a framework for running controlled structural privacy experiments on quantum machine learning models. The experiments cover both synthetic datasets and MNIST-based quantum architectures.

The central QuRiFT entry point is:

```bash
python experiments/qurift_main.py
```

The sweep driver scripts in this directory call `qurift_main.py` with different experiment configurations. These sweeps generate result CSV files that are later used by the analysis scripts and membership inference attack pipelines.

---

## 1. Sweep Driver Scripts

### 1.1 Synthetic QNN Sweeps

The following scripts run full QNN sweeps on synthetic datasets:

```text
full_sweep_qnn_moons.py
full_sweep_qnn_circles.py
full_sweep_qnn_blobs.py
```

These scripts execute controlled sweeps over synthetic datasets such as Moons, Circles, and Blobs.

### 1.2 MNIST Sweeps

The following scripts run MNIST experiments for different model architectures:

```text
run_mnist_sweep_qnn.py
run_mnist_sweep_hqnn.py
run_mnist_sweep_qcnn.py
```

These scripts generate results for:

```text
QNN
HQNN
QCNN
```

---

## 2. Running Experiments

To run an experiment, execute the corresponding sweep driver from the project root.

For example:

```bash
python experiments/full_sweep_qnn_moons.py
python experiments/full_sweep_qnn_circles.py
python experiments/full_sweep_qnn_blobs.py
```

For MNIST sweeps:

```bash
python experiments/run_mnist_sweep_qnn.py
python experiments/run_mnist_sweep_hqnn.py
python experiments/run_mnist_sweep_qcnn.py
```

Each sweep script internally calls:

```bash
python experiments/qurift_main.py
```

with the appropriate dataset, architecture, and structural configuration settings.

---

## 3. Generated Output Directories

After a sweep is completed, QuRiFT creates a timestamped output directory containing the generated results.

### 3.1 MNIST Output Directories

For MNIST experiments, the generated directories may look like:

```text
qcnn_sweep_100_xxxxxxxx/
hqnn_sweep_xxxxxxxx/
mnist_extensive_sweep_qnn_xxxxxxxx/
```

The exact directory name may vary because it usually includes a timestamp or unique run identifier.

These directories contain architecture-specific result files such as:

```text
qnn_extensive_results.csv
hqnn_extensive_results.csv
qcnn_extensive_results.csv
```

### 3.2 Synthetic Dataset Output Directories

For synthetic datasets, the generated directories may look like:

```text
sweep_full_pipeline_moons_xxxxxxxx/
sweep_full_pipeline_blobs_xxxxxxxx/
sweep_full_pipeline_circles_xxxxxxxx/
```

These directories contain dataset-specific master result files such as:

```text
master_results_full_pipeline_moon.csv
master_results_full_pipeline_blobs.csv
master_results_full_pipeline_circles.csv
```

---

## 4. Required CSV Files for Downstream Analysis

The downstream analysis scripts and MIA attack scripts expect the final CSV files to be available inside:

```text
experiments/gen_results/
```

After each sweep finishes, manually copy the corresponding CSV files from the generated sweep directories into this folder.

---

## 5. Files to Copy

### 5.1 MNIST Files

Copy the following MNIST result files:

```text
qnn_extensive_results.csv
hqnn_extensive_results.csv
qcnn_extensive_results.csv
```

from their generated directories into:

```text
experiments/gen_results/
```

For example:

```text
experiments/gen_results/qnn_extensive_results.csv
experiments/gen_results/hqnn_extensive_results.csv
experiments/gen_results/qcnn_extensive_results.csv
```

### 5.2 Synthetic Dataset Files

Copy the following synthetic dataset result files:

```text
master_results_full_pipeline_moon.csv
master_results_full_pipeline_blobs.csv
master_results_full_pipeline_circles.csv
```

from their generated sweep directories into:

```text
experiments/gen_results/
```

For example:

```text
experiments/gen_results/master_results_full_pipeline_moon.csv
experiments/gen_results/master_results_full_pipeline_blobs.csv
experiments/gen_results/master_results_full_pipeline_circles.csv
```

---

## 6. Expected Final Directory Structure

After copying the required files, the `experiments/gen_results/` directory should contain files similar to the following:

```text
experiments/gen_results/
├── master_results_full_pipeline_moon.csv
├── master_results_full_pipeline_blobs.csv
├── master_results_full_pipeline_circles.csv
├── qnn_extensive_results.csv
├── hqnn_extensive_results.csv
└── qcnn_extensive_results.csv
```

These files are then visible to the downstream analysis scripts and membership inference attack scripts.

---

## 7. Relationship Between Sweeps, Results, and MIA Attacks

The workflow is:

```text
Sweep driver scripts
        ↓
experiments/qurift_main.py
        ↓
Generated sweep output directories
        ↓
Result CSV files
        ↓
Copy CSV files into experiments/gen_results/
        ↓
Analysis scripts and MIA attack scripts
```

In other words, the sweep scripts first generate the raw experiment result CSV files. These CSV files must then be copied into `experiments/gen_results/`, where they become available to the analysis and MIA attack pipelines.

---

## 8. Summary

Use the sweep driver scripts to generate structural privacy experiment results. Once the sweeps are complete, copy the generated CSV files into:

```text
experiments/gen_results/
```

The most important files are:

```text
master_results_full_pipeline_moon.csv
master_results_full_pipeline_blobs.csv
master_results_full_pipeline_circles.csv
qnn_extensive_results.csv
hqnn_extensive_results.csv
qcnn_extensive_results.csv
```

These files serve as the shared input for downstream QuRiFT analysis and membership inference attack experiments.
