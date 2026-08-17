# Paste-ready reviewer-response tables

All `mean ± SD` entries use the replication unit stated below each table. Bootstrap confidence intervals are shown separately and are not converted to standard deviations. The completed attack suite includes calibrated LiRA and a query-based class-label-only boundary proxy. These are focused rebuttal experiments supplementing the submission's broader exploratory sweep; they are not a multi-seed rerun of every originally swept configuration.

## Area Chair

### AC-1. Prespecified factorial effects

| Factor | Contrast | Metric | Mean difference ± SD | 95% CI | Paired units |
| --- | --- | --- | --- | --- | --- |
| Repetitions | 5 − 1 | Accuracy gap | +0.123 ± 0.041 | [0.094, 0.147] | 18 |
| Repetitions | 5 − 1 | Loss AUC | +0.069 ± 0.029 | [0.049, 0.088] | 18 |
| Depth | 6 − 2 | Accuracy gap | +0.087 ± 0.059 | [0.049, 0.128] | 18 |
| Depth | 6 − 2 | Loss AUC | +0.042 ± 0.026 | [0.024, 0.060] | 18 |
| Feature map | Z − EffSU2 | Accuracy gap | +0.092 ± 0.048 | [0.050, 0.124] | 12 |
| Feature map | Z − EffSU2 | Loss AUC | +0.057 ± 0.032 | [0.027, 0.080] | 12 |
| Feature map | ZZ − EffSU2 | Accuracy gap | +0.122 ± 0.057 | [0.071, 0.170] | 12 |
| Feature map | ZZ − EffSU2 | Loss AUC | +0.052 ± 0.027 | [0.030, 0.074] | 12 |
| Feature map | ZZ − Z | Accuracy gap | +0.030 ± 0.053 | [-0.011, 0.066] | 12 |
| Feature map | ZZ − Z | Loss AUC | -0.006 ± 0.030 | [-0.033, 0.017] | 12 |

*Values are mean paired differences ± SD across paired target-seed units. The 95% intervals are hierarchical percentile-bootstrap intervals over structural blocks with target seeds nested (5,000 replicates).*

### AC-2. Finite-shot and backend-noise sanity check

| Condition | Low-risk test accuracy | High-risk test accuracy | Low-risk loss AUC | High-risk loss AUC | High − low AUC |
| --- | --- | --- | --- | --- | --- |
| Exact | 0.750 ± 0.040 | 0.550 ± 0.010 | 0.521 ± 0.018 | 0.700 ± 0.037 | +0.179 |
| Ideal shot, 128 | 0.622 ± 0.016 | 0.516 ± 0.007 | 0.518 ± 0.015 | 0.671 ± 0.033 | +0.153 |
| Ideal shot, 512 | 0.699 ± 0.032 | 0.532 ± 0.009 | 0.520 ± 0.012 | 0.691 ± 0.031 | +0.171 |
| Ideal shot, 1024 | 0.724 ± 0.040 | 0.539 ± 0.013 | 0.521 ± 0.015 | 0.696 ± 0.037 | +0.175 |
| Backend-noisy, 128 | 0.611 ± 0.022 | 0.495 ± 0.019 | 0.526 ± 0.020 | 0.622 ± 0.031 | +0.096 |
| Backend-noisy, 512 | 0.686 ± 0.030 | 0.525 ± 0.048 | 0.527 ± 0.013 | 0.651 ± 0.034 | +0.124 |
| Backend-noisy, 1024 | 0.710 ± 0.029 | 0.539 ± 0.044 | 0.525 ± 0.019 | 0.657 ± 0.034 | +0.132 |

*Condition entries are mean ± sample SD across three independently trained target checkpoints. Shot/noise conditions additionally use ten simulator seeds per target. The final contrast column is the difference between the displayed target-seed means and does not have an independently estimated SD. The noise model is IBM-backend-derived Aer simulation, not hardware execution.*

## Reviewer Epmi

### Epmi-1. Structural effects on gap and loss-based MIA

| Factor | Contrast | Metric | Mean difference ± SD | 95% CI | Paired units |
| --- | --- | --- | --- | --- | --- |
| Repetitions | 5 − 1 | Accuracy gap | +0.123 ± 0.041 | [0.094, 0.147] | 18 |
| Repetitions | 5 − 1 | Loss AUC | +0.069 ± 0.029 | [0.049, 0.088] | 18 |
| Depth | 6 − 2 | Accuracy gap | +0.087 ± 0.059 | [0.049, 0.128] | 18 |
| Depth | 6 − 2 | Loss AUC | +0.042 ± 0.026 | [0.024, 0.060] | 18 |
| Feature map | Z − EffSU2 | Accuracy gap | +0.092 ± 0.048 | [0.050, 0.124] | 12 |
| Feature map | Z − EffSU2 | Loss AUC | +0.057 ± 0.032 | [0.027, 0.080] | 12 |
| Feature map | ZZ − EffSU2 | Accuracy gap | +0.122 ± 0.057 | [0.071, 0.170] | 12 |
| Feature map | ZZ − EffSU2 | Loss AUC | +0.052 ± 0.027 | [0.030, 0.074] | 12 |
| Feature map | ZZ − Z | Accuracy gap | +0.030 ± 0.053 | [-0.011, 0.066] | 12 |
| Feature map | ZZ − Z | Loss AUC | -0.006 ± 0.030 | [-0.033, 0.017] | 12 |

*Mean paired difference ± paired-unit SD; 95% hierarchical bootstrap CI.*

### Epmi-2. Direct gap–MIA analysis

| Analysis | Estimate | 95% CI | Units/notes |
| --- | --- | --- | --- |
| Gap–AUC Pearson correlation | 0.948 | [0.864, 0.982] | 12 structural configurations / 36 targets |
| Gap–AUC Spearman correlation | 0.931 | [0.710, 0.974] | 12 structural configurations / 36 targets |
| Gap + structure regression R² | 0.935 | [0.852, 0.986] | 36 targets |
| Standardized gap coefficient | 0.039 | [0.010, 0.053] | 5,000 bootstrap replicates |
| Residual repetitions coefficient | 0.010 | [-0.001, 0.032] | 5,000 bootstrap replicates |
| Residual depth coefficient | 0.004 | [-0.005, 0.020] | 5,000 bootstrap replicates |
| Residual Z coefficient | 0.020 | [-0.003, 0.061] | 5,000 bootstrap replicates |
| Residual ZZ coefficient | 0.003 | [-0.024, 0.048] | 5,000 bootstrap replicates |

*Regression predictors are standardized. Confidence intervals are cluster/hierarchical bootstrap intervals. Because these analyses provide intervals rather than a seed-level SD for each coefficient, no artificial ± value is shown.*

### Epmi-3. Attack-signal decomposition

| Attack | Access | Targets | Attacker seeds | AUC | TPR@5% FPR | TPR@10% FPR |
| --- | --- | --- | --- | --- | --- | --- |
| Confidence threshold | score/probability threshold | 36 | 0 | 0.520 ± 0.025 | 0.055 ± 0.020 | 0.117 ± 0.029 |
| Correctness | predicted label + known candidate label | 36 | 0 | 0.582 ± 0.049 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Entropy threshold | score/probability threshold | 36 | 0 | 0.521 ± 0.023 | 0.053 ± 0.020 | 0.116 ± 0.029 |
| Loss threshold | score/probability threshold | 36 | 0 | 0.596 ± 0.052 | 0.061 ± 0.024 | 0.135 ± 0.039 |
| Margin threshold | score/probability threshold | 36 | 0 | 0.521 ± 0.026 | 0.053 ± 0.019 | 0.116 ± 0.029 |
| Maximum probability | score/probability threshold | 36 | 0 | 0.520 ± 0.025 | 0.055 ± 0.020 | 0.117 ± 0.029 |
| Learned prediction vector | full prediction vector + statistics | 36 | 3 | 0.574 ± 0.065 | 0.099 ± 0.044 | 0.157 ± 0.061 |

#### Calibrated and class-label-only baselines

| Attack | Access | Targets | References/configuration | AUC | TPR@5% FPR | TPR@10% FPR |
| --- | --- | --- | --- | --- | --- | --- |
| Online LiRA, fixed variance | true-label probability + calibrated reference QNNs | 36 | 16 | 0.609 ± 0.063 | 0.128 ± 0.056 | 0.192 ± 0.069 |
| Online LiRA, per-record variance | true-label probability + calibrated reference QNNs | 36 | 16 | 0.594 ± 0.057 | 0.086 ± 0.034 | 0.155 ± 0.063 |
| Offline LiRA, fixed variance | true-label probability + calibrated reference QNNs | 36 | 16 | 0.517 ± 0.029 | 0.066 ± 0.027 | 0.128 ± 0.035 |
| Label-only chord-boundary | predicted class labels only + held-out anchors | 36 | — | 0.582 ± 0.052 | 0.077 ± 0.027 | 0.139 ± 0.030 |

*Attack entries are mean ± sample SD across the 36 target-model units. The learned attack is averaged over three attacker seeds per target; scalar threshold attacks have no attacker-training seed. LiRA uses 16 reference QNNs per structural configuration.*

### Epmi-4. Direct encoder-geometry effects

| Geometry metric | Contrast | Mean difference ± SD | 95% CI | Unique paired effects |
| --- | --- | --- | --- | --- |
| Class-similarity gap | reps 5 minus reps 1 | -0.124 ± 0.061 | [-0.158, -0.079] | 12 |
| Kernel–label alignment | reps 5 minus reps 1 | -0.208 ± 0.132 | [-0.285, -0.109] | 12 |
| Effective rank | reps 5 minus reps 1 | +49.747 ± 37.448 | [15.791, 83.870] | 12 |
| Train/test MMD² | reps 5 minus reps 1 | -0.003 ± 0.008 | [-0.010, 0.003] | 12 |
| Encoder operation count | reps 5 minus reps 1 | +113.333 ± 110.321 | [50.000, 204.667] | 6 |

*Mean paired repetition effect ± SD across unique paired effects. MNIST nominal geometry seeds duplicated encoded states, so zero MNIST configuration-level SD must not be interpreted as independent-seed robustness; Moons supplies genuine data-seed variation.*

### Epmi-5. Complete-wrapper architecture effects

| Wrapper vs QNN | Metric | Mean difference ± SD | 95% CI | Paired units |
| --- | --- | --- | --- | --- |
| HQNN | Test accuracy | +0.201 ± 0.237 | [-0.021, 0.333] | 9 |
| HQNN | Accuracy gap | -0.071 ± 0.096 | [-0.167, 0.009] | 9 |
| HQNN | Loss AUC | -0.014 ± 0.051 | [-0.068, 0.030] | 9 |
| MLP-QNN | Test accuracy | +0.146 ± 0.093 | [0.062, 0.259] | 9 |
| MLP-QNN | Accuracy gap | +0.016 ± 0.098 | [-0.087, 0.096] | 9 |
| MLP-QNN | Loss AUC | +0.011 ± 0.044 | [-0.038, 0.045] | 9 |
| QCNN | Test accuracy | +0.190 ± 0.052 | [0.148, 0.249] | 9 |
| QCNN | Accuracy gap | -0.041 ± 0.042 | [-0.083, -0.007] | 9 |
| QCNN | Loss AUC | -0.019 ± 0.025 | [-0.040, -0.001] | 9 |

*Mean paired wrapper-minus-QNN difference ± SD across nine role/target-seed pairs; 95% paired hierarchical bootstrap CI. Wrappers have unmatched preprocessing, heads, and parameter counts.*

## Reviewer 1myw

### 1myw-1. Exact, finite-shot, and backend-noisy results

| Condition | Low-risk test accuracy | High-risk test accuracy | Low-risk loss AUC | High-risk loss AUC | High − low AUC |
| --- | --- | --- | --- | --- | --- |
| Exact | 0.750 ± 0.040 | 0.550 ± 0.010 | 0.521 ± 0.018 | 0.700 ± 0.037 | +0.179 |
| Ideal shot, 128 | 0.622 ± 0.016 | 0.516 ± 0.007 | 0.518 ± 0.015 | 0.671 ± 0.033 | +0.153 |
| Ideal shot, 512 | 0.699 ± 0.032 | 0.532 ± 0.009 | 0.520 ± 0.012 | 0.691 ± 0.031 | +0.171 |
| Ideal shot, 1024 | 0.724 ± 0.040 | 0.539 ± 0.013 | 0.521 ± 0.015 | 0.696 ± 0.037 | +0.175 |
| Backend-noisy, 128 | 0.611 ± 0.022 | 0.495 ± 0.019 | 0.526 ± 0.020 | 0.622 ± 0.031 | +0.096 |
| Backend-noisy, 512 | 0.686 ± 0.030 | 0.525 ± 0.048 | 0.527 ± 0.013 | 0.651 ± 0.034 | +0.124 |
| Backend-noisy, 1024 | 0.710 ± 0.029 | 0.539 ± 0.044 | 0.525 ± 0.019 | 0.657 ± 0.034 | +0.132 |

*Mean ± sample SD across three target-model seeds. Each shot/noise target mean uses ten simulator seeds. Backend-derived Aer noise is not hardware execution.*

### 1myw-2. Expanded currently completed attack suite

| Attack | Access | Targets | Attacker seeds | AUC | TPR@5% FPR | TPR@10% FPR |
| --- | --- | --- | --- | --- | --- | --- |
| Confidence threshold | score/probability threshold | 36 | 0 | 0.520 ± 0.025 | 0.055 ± 0.020 | 0.117 ± 0.029 |
| Correctness | predicted label + known candidate label | 36 | 0 | 0.582 ± 0.049 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Entropy threshold | score/probability threshold | 36 | 0 | 0.521 ± 0.023 | 0.053 ± 0.020 | 0.116 ± 0.029 |
| Loss threshold | score/probability threshold | 36 | 0 | 0.596 ± 0.052 | 0.061 ± 0.024 | 0.135 ± 0.039 |
| Margin threshold | score/probability threshold | 36 | 0 | 0.521 ± 0.026 | 0.053 ± 0.019 | 0.116 ± 0.029 |
| Maximum probability | score/probability threshold | 36 | 0 | 0.520 ± 0.025 | 0.055 ± 0.020 | 0.117 ± 0.029 |
| Learned prediction vector | full prediction vector + statistics | 36 | 3 | 0.574 ± 0.065 | 0.099 ± 0.044 | 0.157 ± 0.061 |

#### Completed calibrated and class-label-only baselines

| Attack | Access | Targets | References/configuration | AUC | TPR@5% FPR | TPR@10% FPR |
| --- | --- | --- | --- | --- | --- | --- |
| Online LiRA, fixed variance | true-label probability + calibrated reference QNNs | 36 | 16 | 0.609 ± 0.063 | 0.128 ± 0.056 | 0.192 ± 0.069 |
| Online LiRA, per-record variance | true-label probability + calibrated reference QNNs | 36 | 16 | 0.594 ± 0.057 | 0.086 ± 0.034 | 0.155 ± 0.063 |
| Offline LiRA, fixed variance | true-label probability + calibrated reference QNNs | 36 | 16 | 0.517 ± 0.029 | 0.066 ± 0.027 | 0.128 ± 0.035 |
| Label-only chord-boundary | predicted class labels only + held-out anchors | 36 | — | 0.582 ± 0.052 | 0.077 ± 0.027 | 0.139 ± 0.030 |

*Mean ± sample SD across target models. The label-only method is a changed-label chord-boundary proxy, not a certified minimum-distance attack.*

## Reviewer nVBH

### nVBH-1. All configurations in the focused MNIST-QNN confirmatory factorial

| Feature map | Reps | Depth | Seeds | Train accuracy | Test accuracy | Gap | Loss AUC | TPR@5% FPR | TPR@10% FPR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EffSU2 | 1 | 2 | 3 | 0.748 ± 0.014 | 0.725 ± 0.028 | 0.023 ± 0.029 | 0.530 ± 0.010 | 0.067 ± 0.016 | 0.108 ± 0.003 |
| EffSU2 | 1 | 6 | 3 | 0.810 ± 0.018 | 0.772 ± 0.012 | 0.038 ± 0.019 | 0.536 ± 0.006 | 0.053 ± 0.003 | 0.117 ± 0.013 |
| EffSU2 | 5 | 2 | 3 | 0.858 ± 0.018 | 0.735 ± 0.023 | 0.123 ± 0.036 | 0.571 ± 0.006 | 0.042 ± 0.010 | 0.112 ± 0.028 |
| EffSU2 | 5 | 6 | 3 | 0.962 ± 0.015 | 0.780 ± 0.013 | 0.182 ± 0.028 | 0.603 ± 0.017 | 0.087 ± 0.024 | 0.153 ± 0.013 |
| Z | 1 | 2 | 3 | 0.818 ± 0.023 | 0.758 ± 0.020 | 0.060 ± 0.041 | 0.541 ± 0.017 | 0.048 ± 0.023 | 0.125 ± 0.009 |
| Z | 1 | 6 | 3 | 0.912 ± 0.012 | 0.745 ± 0.025 | 0.167 ± 0.016 | 0.599 ± 0.015 | 0.060 ± 0.025 | 0.153 ± 0.024 |
| Z | 5 | 2 | 3 | 0.638 ± 0.016 | 0.427 ± 0.026 | 0.212 ± 0.029 | 0.646 ± 0.012 | 0.078 ± 0.020 | 0.172 ± 0.028 |
| Z | 5 | 6 | 3 | 0.743 ± 0.003 | 0.445 ± 0.005 | 0.298 ± 0.003 | 0.684 ± 0.012 | 0.095 ± 0.020 | 0.195 ± 0.017 |
| ZZ | 1 | 2 | 3 | 0.820 ± 0.030 | 0.703 ± 0.031 | 0.117 ± 0.025 | 0.562 ± 0.012 | 0.045 ± 0.010 | 0.112 ± 0.021 |
| ZZ | 1 | 6 | 3 | 0.927 ± 0.010 | 0.722 ± 0.013 | 0.205 ± 0.017 | 0.604 ± 0.011 | 0.037 ± 0.008 | 0.110 ± 0.028 |
| ZZ | 5 | 2 | 3 | 0.703 ± 0.015 | 0.520 ± 0.010 | 0.183 ± 0.025 | 0.601 ± 0.025 | 0.075 ± 0.023 | 0.153 ± 0.081 |
| ZZ | 5 | 6 | 3 | 0.888 ± 0.042 | 0.538 ± 0.016 | 0.350 ± 0.035 | 0.679 ± 0.018 | 0.045 ± 0.013 | 0.112 ± 0.048 |

*Every configuration in this focused factorial contains three independently initialized target models. Entries are mean ± sample SD across those target seeds. The primary factorial uses one fixed data split.*

### nVBH-2. Paired factorial contrasts with uncertainty

| Factor | Contrast | Metric | Mean difference ± SD | 95% CI | Paired units |
| --- | --- | --- | --- | --- | --- |
| Repetitions | 5 − 1 | Accuracy gap | +0.123 ± 0.041 | [0.094, 0.147] | 18 |
| Repetitions | 5 − 1 | Loss AUC | +0.069 ± 0.029 | [0.049, 0.088] | 18 |
| Depth | 6 − 2 | Accuracy gap | +0.087 ± 0.059 | [0.049, 0.128] | 18 |
| Depth | 6 − 2 | Loss AUC | +0.042 ± 0.026 | [0.024, 0.060] | 18 |
| Feature map | Z − EffSU2 | Accuracy gap | +0.092 ± 0.048 | [0.050, 0.124] | 12 |
| Feature map | Z − EffSU2 | Loss AUC | +0.057 ± 0.032 | [0.027, 0.080] | 12 |
| Feature map | ZZ − EffSU2 | Accuracy gap | +0.122 ± 0.057 | [0.071, 0.170] | 12 |
| Feature map | ZZ − EffSU2 | Loss AUC | +0.052 ± 0.027 | [0.030, 0.074] | 12 |
| Feature map | ZZ − Z | Accuracy gap | +0.030 ± 0.053 | [-0.011, 0.066] | 12 |
| Feature map | ZZ − Z | Loss AUC | -0.006 ± 0.030 | [-0.033, 0.017] | 12 |

*Mean paired difference ± paired-unit SD; 95% hierarchical bootstrap CI.*

### nVBH-3. Wrapper performance and resource accounting

| Role | Wrapper | Test accuracy | Gap | Loss AUC | Total params | Quantum params | Classical params | Quantum gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eff_control | HQNN | 0.968 ± 0.008 | 0.032 ± 0.008 | 0.548 ± 0.010 | 19930 | 46 | 19884 | 63 |
| eff_control | MLP-QNN | 0.783 ± 0.008 | 0.120 ± 0.010 | 0.567 ± 0.005 | 96 | 0 | 96 | 0 |
| eff_control | QCNN | 0.832 ± 0.033 | 0.002 ± 0.028 | 0.514 ± 0.004 | 592 | 394 | 198 | 67 |
| eff_control | QNN | 0.667 ± 0.013 | 0.023 ± 0.018 | 0.530 ± 0.029 | 74 | 46 | 28 | 63 |
| high_reupload | HQNN | 0.822 ± 0.188 | 0.082 ± 0.023 | 0.563 ± 0.018 | 20022 | 138 | 19884 | 543 |
| high_reupload | MLP-QNN | 0.800 ± 0.009 | 0.152 ± 0.040 | 0.592 ± 0.011 | 188 | 0 | 188 | 0 |
| high_reupload | QCNN | 0.792 ± 0.008 | 0.168 ± 0.010 | 0.595 ± 0.005 | 684 | 486 | 198 | 547 |
| high_reupload | QNN | 0.538 ± 0.015 | 0.253 ± 0.028 | 0.633 ± 0.020 | 166 | 138 | 28 | 543 |
| low_reupload | HQNN | 0.742 ± 0.348 | 0.018 ± 0.069 | 0.542 ± 0.021 | 19930 | 46 | 19884 | 82 |
| low_reupload | MLP-QNN | 0.783 ± 0.008 | 0.120 ± 0.010 | 0.567 ± 0.005 | 96 | 0 | 96 | 0 |
| low_reupload | QCNN | 0.877 ± 0.008 | 0.052 ± 0.008 | 0.528 ± 0.014 | 592 | 394 | 198 | 86 |
| low_reupload | QNN | 0.725 ± 0.018 | 0.068 ± 0.026 | 0.531 ± 0.019 | 74 | 46 | 28 | 82 |

*Performance entries are mean ± sample SD across three target seeds. Resource counts are deterministic for a wrapper/role configuration. These are complete-wrapper comparisons, not matched-capacity causal ablations.*

### nVBH-4. Paired wrapper effects

| Wrapper vs QNN | Metric | Mean difference ± SD | 95% CI | Paired units |
| --- | --- | --- | --- | --- |
| HQNN | Test accuracy | +0.201 ± 0.237 | [-0.021, 0.333] | 9 |
| HQNN | Accuracy gap | -0.071 ± 0.096 | [-0.167, 0.009] | 9 |
| HQNN | Loss AUC | -0.014 ± 0.051 | [-0.068, 0.030] | 9 |
| MLP-QNN | Test accuracy | +0.146 ± 0.093 | [0.062, 0.259] | 9 |
| MLP-QNN | Accuracy gap | +0.016 ± 0.098 | [-0.087, 0.096] | 9 |
| MLP-QNN | Loss AUC | +0.011 ± 0.044 | [-0.038, 0.045] | 9 |
| QCNN | Test accuracy | +0.190 ± 0.052 | [0.148, 0.249] | 9 |
| QCNN | Accuracy gap | -0.041 ± 0.042 | [-0.083, -0.007] | 9 |
| QCNN | Loss AUC | -0.019 ± 0.025 | [-0.040, -0.001] | 9 |

*Mean paired wrapper-minus-QNN difference ± paired-unit SD; 95% bootstrap CI.*

## Reporting statement

Use the tables above with the following clarification: configuration-level ± values are sample SD across independent model initializations, paired-effect ± values are SD across paired experimental units, and noisy-condition ± values are across independently trained targets. None of these should be described as standard errors or confidence intervals.
