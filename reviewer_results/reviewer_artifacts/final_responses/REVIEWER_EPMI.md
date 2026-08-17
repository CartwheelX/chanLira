# Response to Reviewer Epmi

Thank you for the detailed evaluation requests. The submitted paper reports a broad exploratory sweep across datasets, QNN/HQNN/QCNN wrappers, feature-map families, repetitions, widths, variational depths, entanglers, gates, and padding. For the rebuttal, we did not rerun that entire sweep with multiple seeds. Instead, we defined a focused 3×2×2 MNIST-QNN confirmatory factorial over feature-map family, repetition, and variational depth, trained three target-model seeds for every configuration (36 targets), and ran the same MIA suite on every one without attack-outcome selection. We also added targeted geometry, finite-shot/noise, and complete-wrapper architecture controls.

## Attack breadth on the focused confirmatory factorial

The expanded attacks separate the information channels requested in the review. Among the uncalibrated scalar signals, loss is the strongest by mean AUC (0.596 ± 0.052). Correctness reaches 0.582 ± 0.049 AUC but has zero TPR at 5% and 10% FPR because its binary score cannot realize those low operating points. Confidence, entropy, margin, and maximum probability are all close to chance (0.520–0.521 mean AUC), and the learned feature combination reaches 0.574 ± 0.065. Thus, loss is the clearest raw output feature in this factorial; combining the prediction-vector features does not improve its mean AUC. Fixed-variance online LiRA is numerically strongest overall at 0.609 ± 0.063 and improves low-FPR TPR relative to raw loss, while the class-label-only boundary proxy reaches 0.582 ± 0.052.

| Attack | Information access | AUC | TPR@5% FPR | TPR@10% FPR |
| --- | --- | --- | --- | --- |
| Loss threshold | true-label probability + known candidate label | 0.596 ± 0.052 | 0.061 ± 0.024 | 0.135 ± 0.039 |
| Confidence threshold | maximum predicted-class probability | 0.520 ± 0.025 | 0.055 ± 0.020 | 0.117 ± 0.029 |
| Entropy threshold | entropy of the prediction vector | 0.521 ± 0.023 | 0.053 ± 0.020 | 0.116 ± 0.029 |
| Margin threshold | top-1 minus top-2 predicted probability | 0.521 ± 0.026 | 0.053 ± 0.019 | 0.116 ± 0.029 |
| Maximum-probability threshold | maximum predicted-class probability | 0.520 ± 0.025 | 0.055 ± 0.020 | 0.117 ± 0.029 |
| Correctness | predicted label + known candidate label | 0.582 ± 0.049 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Learned prediction-vector | full prediction vector + statistics | 0.574 ± 0.065 | 0.099 ± 0.044 | 0.157 ± 0.061 |
| Online LiRA, fixed variance | true-label probability + calibrated reference QNNs | 0.609 ± 0.063 | 0.128 ± 0.056 | 0.192 ± 0.069 |
| Online LiRA, per-record variance | true-label probability + calibrated reference QNNs | 0.594 ± 0.057 | 0.086 ± 0.034 | 0.155 ± 0.063 |
| Offline LiRA, fixed variance | true-label probability + calibrated reference QNNs | 0.517 ± 0.029 | 0.066 ± 0.027 | 0.128 ± 0.035 |
| Label-only chord-boundary | predicted class labels only + held-out anchors | 0.582 ± 0.052 | 0.077 ± 0.027 | 0.139 ± 0.030 |

In our exported statistics, confidence is defined as the largest predicted-class probability, so the confidence and maximum-probability rows are the same scalar baseline under two commonly used names; they are not independent attacks.

LiRA uses 16 same-architecture references per structural configuration; every candidate is IN for exactly eight references. Its reference-training pool is an explicit approximation formed from the canonical 200 target-train and 200 target-test candidates. The label-only method uses no probability, loss, logit, gradient, or model parameter and is described only as a chord-boundary proxy.

## Paired structural effects across attacks

| Attack | Factor | Contrast | AUC difference ± SD | 95% CI | Paired units |
| --- | --- | --- | --- | --- | --- |
| Online LiRA, fixed variance | Repetitions | 5 − 1 | +0.044 ± 0.045 | [0.013, 0.072] | 18 |
| Online LiRA, fixed variance | Depth | 6 − 2 | +0.077 ± 0.057 | [0.038, 0.118] | 18 |
| Online LiRA, fixed variance | Feature map | Z − EffSU2 | +0.057 ± 0.030 | [0.038, 0.076] | 12 |
| Online LiRA, fixed variance | Feature map | ZZ − EffSU2 | +0.070 ± 0.059 | [0.021, 0.117] | 12 |
| Online LiRA, fixed variance | Feature map | ZZ − Z | +0.014 ± 0.044 | [-0.026, 0.048] | 12 |
| Label-only chord-boundary | Repetitions | 5 − 1 | +0.068 ± 0.026 | [0.049, 0.085] | 18 |
| Label-only chord-boundary | Depth | 6 − 2 | +0.047 ± 0.033 | [0.021, 0.068] | 18 |
| Label-only chord-boundary | Feature map | Z − EffSU2 | +0.038 ± 0.035 | [0.004, 0.061] | 12 |
| Label-only chord-boundary | Feature map | ZZ − EffSU2 | +0.054 ± 0.039 | [0.022, 0.087] | 12 |
| Label-only chord-boundary | Feature map | ZZ − Z | +0.017 ± 0.026 | [-0.004, 0.035] | 12 |

The calibrated attacks support the same encoder-induced pathway. Repetition raises online LiRA AUC in five of six fixed-depth comparisons and has a positive pooled contrast; Z and ZZ exceed EffSU2 under both LiRA and label-only access, while ZZ−Z is unresolved. Variational depth also modulates the trained model's output signal, so we do not claim that the encoder is the only contributor. The supported claim is that encoder family and repetition alter the pre-training representation and systematically condition the downstream asymmetry exploited by MIAs.

## Gap, causal scope, and geometry

Across all 36 targets, loss-AUC and accuracy gap have Pearson r=0.948 ([0.864, 0.982]) and Spearman ρ=0.931 ([0.710, 0.974]). A descriptive model containing standardized gap plus feature map, repetition, and depth obtains R²=0.935 ([0.852, 0.986]); after conditioning on gap, all residual structural coefficient intervals cross zero. We therefore agree with the reviewer's causal distinction: the evidence supports structural choices as antecedents of overfitting, which then supplies membership signal, not a gap-independent causal leakage effect.

We directly measured K(x,x′)=|⟨ψ(x)|ψ(x′)⟩|² immediately after the fixed encoder and before the trainable variational circuit. The class-similarity gap summarizes the within-class and between-class kernel-similarity distributions; kernel–label alignment measures task alignment; and effective rank measures the kernel spectrum. Their paired repetition contrasts are:

| Geometry measurement | Definition | Reps 5 − reps 1 ± SD | 95% CI |
| --- | --- | --- | --- |
| Class-similarity gap | mean within-class minus between-class fidelity | −0.124 ± 0.061 | [−0.158, −0.079] |
| Kernel–label alignment | centered fidelity-kernel/label-kernel alignment | −0.208 ± 0.132 | [−0.285, −0.109] |
| Effective rank | spectral effective rank of the fidelity kernel | +49.747 ± 37.448 | [15.791, 83.870] |
| Train/test MMD² | kernel discrepancy between train and test encoded states | −0.003 ± 0.008 | [−0.010, 0.003] |

Increasing repetition therefore produces a higher-rank but less class-aligned post-encoder representation. The train/test MMD² interval includes zero, so the fixed encoder does not itself measurably separate members from non-members. These measurements directly verify encoder→geometry association, but they do not causally identify geometry→generalization-gap mediation. We will accordingly replace ‘geometric mechanism’ with ‘empirically supported geometric pathway.’

## Scope and remaining limitations

The primary factorial uses one fixed MNIST split, so three model seeds are not three independent data splits. Architecture controls are complete-wrapper comparisons with resource accounting and a classical MLP with a comparable small parameter budget to the QNN roles; preprocessing and heads remain unmatched. Backend-noise results are Aer simulations from one `ibm_kingston` calibration snapshot, not hardware execution. We will state all three limitations and avoid independent-causal or hardware-general claims.
