# Response to Reviewer Epmi

Thank you. The submission reports a broad exploratory sweep across datasets, architectures, encoders, repetitions, depths, and lower-level circuit choices. For the rebuttal, we added a focused MNIST-QNN factorial (3 feature maps × 2 repetitions × 2 depths × 3 target seeds = 36 targets) and applied the same attack suite to every target. Full protocols, tables, and figures are in our [detailed response](https://github.com/CartwheelX/QuRiFT/blob/main/docs/Reviewer-Epmi.md).

**Which output signal drives MIA?**

We evaluated every requested scalar separately. Mean AUC ± sample SD across 36 targets is: loss 0.596 ± 0.052; correctness 0.582 ± 0.049; entropy 0.521 ± 0.023; margin 0.521 ± 0.026; and confidence/maximum probability 0.520 ± 0.025. Confidence and maximum probability are the same exported scalar, not independent attacks. The learned prediction-vector-plus-statistics attacker obtains 0.574 ± 0.065. Thus, loss is the clearest raw signal, and combining features does not improve its mean AUC. Correctness has zero TPR at 5% and 10% FPR because its score is binary.

Fixed-variance online LiRA is numerically strongest overall (AUC 0.609 ± 0.063; TPR@5% FPR 0.128 ± 0.056). It uses 16 same-architecture references per structural configuration; each candidate is IN for eight. The class-label-only chord-boundary proxy obtains 0.582 ± 0.052 AUC and 0.077 ± 0.027 TPR@5% FPR.

**Paired structural effects across the calibrated and label-only attacks.**

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

Repetition has a positive pooled effect under both attacks, and Z and ZZ exceed EffSU2; ZZ−Z remains unresolved. LiRA rises with repetition in five of six fixed-depth comparisons. Variational depth also modulates the downstream signal, but does not remove the encoder-family/repetition association.

**Direct Hilbert-space geometry.**

We compute the fidelity kernel K(x,x′)=|⟨ψ(x)|ψ(x′)⟩|² immediately after the fixed encoder and before the trainable variational circuit. Paired reps=5 minus reps=1 effects are:

| Geometry measurement | Definition | Reps 5 − reps 1 ± SD | 95% CI |
| --- | --- | --- | --- |
| Class-similarity gap | mean within-class minus between-class fidelity | −0.124 ± 0.061 | [−0.158, −0.079] |
| Kernel–label alignment | centered fidelity-kernel/label-kernel alignment | −0.208 ± 0.132 | [−0.285, −0.109] |
| Effective rank | spectral effective rank of the fidelity kernel | +49.747 ± 37.448 | [15.791, 83.870] |
| Train/test MMD² | kernel discrepancy between train and test encoded states | −0.003 ± 0.008 | [−0.010, 0.003] |

Increasing repetition therefore produces a higher-rank but less class-aligned post-encoder representation. The MMD² interval includes zero, so the fixed encoder does not itself measurably separate members from non-members before training.

**Interpretation and scope.**

Across the 36 targets, accuracy gap strongly tracks loss-AUC (Spearman ρ=0.931, 95% CI [0.710, 0.974]). After conditioning descriptively on gap, the residual structural coefficient intervals cross zero. We therefore refine the claim to an empirically supported, overfitting-mediated pathway: encoder design → post-encoder geometry → downstream generalization asymmetry → membership signal. The geometry measurements verify encoder→geometry association but do not causally identify geometry→gap mediation. The factorial uses one fixed MNIST split, and it is a focused multi-seed confirmation rather than a rerun of every configuration in the original broad sweep. We will state these limitations and revise causal wording accordingly.
