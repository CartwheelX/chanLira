# Final reviewer responses

# Response to the Area Chair

We thank the Area Chair for consolidating the discussion. In response, we completed: (i) a focused 36-target multi-seed MNIST-QNN factorial; (ii) threshold, learned-vector, 16-reference online/offline LiRA, and class-label-only attacks; (iii) direct post-encoder fidelity-kernel geometry; (iv) QNN/HQNN/QCNN/classical-MLP wrapper controls with resource accounting; and (v) finite-shot and ibm_kingston-derived Aer-noise evaluation. Full tables and protocols are provided in the detailed responses linked in our reviewer comments.

**Statistical robustness, selection, and attack breadth.**

The confirmatory factorial covers 3 feature maps × 2 repetitions × 2 depths × 3 target initializations. Every target receives the same attacks; the earlier post-hoc baseline/stress/hard labels are not used for confirmation. For loss-MIA, reps=5−1 is +0.069 ± 0.029 AUC (95% CI [0.049,0.088]), depth=6−2 is +0.042 ± 0.026 ([0.024,0.060]), Z−EffSU2 is +0.057 ± 0.032 ([0.027,0.080]), and ZZ−EffSU2 is +0.052 ± 0.027 ([0.030,0.074]). The learned attacker is repeated over three training seeds and LiRA uses 16 references/configuration. The data split remains fixed, so these results establish initialization robustness rather than multi-split generalization.

Across attacks with different access assumptions, the same aggregate feature-map and repetition directions recur. Fixed-variance online LiRA has the highest mean performance (AUC 0.609 ± 0.063; TPR@10% FPR 0.192 ± 0.069), followed by loss-threshold (0.596 ± 0.052; 0.135 ± 0.039); label-only obtains 0.582 ± 0.052 AUC using predicted labels alone. Repetition has positive pooled LiRA and label-only effects, and Z/ZZ exceed EffSU2 under both. Attack choice changes magnitude—LiRA is also depth-modulated—but the principal encoder associations are not specific to one attacker.

**Overfitting, proxy validity, and direct geometry.**

Across all 36 prespecified targets, gap and loss-AUC have Spearman ρ=0.931 ([0.710,0.974]); after conditioning descriptively on gap, residual structural coefficient intervals cross zero. We therefore do not claim gap-independent causation. Directly after the fixed encoder, reps=5−1 changes within-minus-between-class fidelity by −0.124 ± 0.061 ([−0.158,−0.079]), kernel–label alignment by −0.208 ± 0.132 ([−0.285,−0.109]), and effective rank by +49.747 ± 37.448 ([15.791,83.870]); the train/test MMD² interval includes zero. Together these results support an empirically measured pathway—encoder design → post-encoder geometry → downstream generalization asymmetry → membership signal—rather than a geometry-only or causally identified mechanism.

**Finite shots and backend-derived noise.**

We evaluated five representative configurations, 15 independently trained checkpoints, three shot counts, and ten simulator seeds under exact inference, ideal finite shots, and an ibm_kingston-derived Aer model (915 target/execution replicates). The exact high-minus-low loss-AUC difference 0.179 ± 0.030 is attenuated to 0.096 ± 0.012, 0.124 ± 0.026, and 0.132 ± 0.036 at 128, 512, and 1,024 noisy shots. Aggregate ordering remains, while nearby configurations can reorder in individual runs. This is a backend-derived robustness check, not hardware execution or evidence of device universality.

**Architecture, attribution, and scope.**

Relative to paired QNN roles, QCNN improves accuracy by +0.190 ± 0.052 ([0.148,0.249]) and reduces gap by −0.041 ± 0.042 ([−0.083,−0.007]) and loss-AUC by −0.019 ± 0.025 ([−0.040,−0.001]); MLP and HQNN gap/AUC intervals are unresolved. These are complete-wrapper controls: preprocessing and heads remain unmatched. We will define Fig. 8 percentages precisely as dataset-normalized factor-association shares and supplement them with paired MIA intervals, rather than interpret them as causal allocations.

The submission's broad sweep supplies exploratory coverage; the rebuttal confirms feature-map family, repetition, and depth in the focused factorial, not every original width/gate/entangler/padding configuration. Dataset conclusions remain bounded to synthetic tasks and compressed four-class MNIST. Sensitive-domain examples motivate why membership privacy matters but are not deployment-validation claims. We will expand related work on QML MIAs, differential privacy, unlearning, and noise-aware QML, and revise the paper's causal, architectural, hardware, and deployment wording accordingly.

---

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

---

# Follow-up response to Reviewer 1myw: completed MIA baselines

This is a follow-up to the already-posted geometry/noise response; it does not repeat that material.

Following up on the attack-breadth paragraph in our previous response, the reference-model LiRA and class-label-only experiments are now complete. Both were evaluated on all 36 targets in our focused confirmatory MNIST-QNN factorial (12 structural configurations × three independently initialized target models). This is the multi-seed follow-up subset, not a claim that we reran every configuration in the submission's broader architecture/dataset sweep.

For LiRA, we trained 16 same-architecture reference QNNs per structural configuration (192 references total). Every one of the 400 candidate records was included in exactly eight references, and every reference was trained on 200 candidates. The primary fixed-variance online LiRA result is 0.609 ± 0.063 AUC, with TPR 0.128 ± 0.056 at 5% FPR and 0.192 ± 0.069 at 10% FPR. The per-record-variance online result is 0.594 ± 0.057 AUC, showing that the result is not specific to one variance model. The offline result is much weaker, which we report rather than selecting only the strongest LiRA variant.

The label-only attack consumes only returned class labels and estimates input-space boundary distance by changed-label searches toward held-out validation anchors. It obtains 0.582 ± 0.052 AUC, with TPR 0.077 ± 0.027 at 5% FPR and 0.139 ± 0.030 at 10% FPR. We describe this as a chord-boundary proxy, not HopSkipJump/QEBA or a certified minimum boundary distance.

| Attack | Information access | AUC | TPR@5% FPR | TPR@10% FPR |
| --- | --- | --- | --- | --- |
| Loss threshold | true-label probability + known candidate label | 0.596 ± 0.052 | 0.061 ± 0.024 | 0.135 ± 0.039 |
| Learned prediction-vector | full prediction vector + statistics | 0.574 ± 0.065 | 0.099 ± 0.044 | 0.157 ± 0.061 |
| Online LiRA, fixed variance | true-label probability + calibrated reference QNNs | 0.609 ± 0.063 | 0.128 ± 0.056 | 0.192 ± 0.069 |
| Online LiRA, per-record variance | true-label probability + calibrated reference QNNs | 0.594 ± 0.057 | 0.086 ± 0.034 | 0.155 ± 0.063 |
| Offline LiRA, fixed variance | true-label probability + calibrated reference QNNs | 0.517 ± 0.029 | 0.066 ± 0.027 | 0.128 ± 0.035 |
| Label-only chord-boundary | predicted class labels only + held-out anchors | 0.582 ± 0.052 | 0.077 ± 0.027 | 0.139 ± 0.030 |

*Entries are mean ± sample SD across 36 target models. The learned attack is first averaged across three attacker seeds per target. LiRA uses one 16-reference bank per structural configuration.*

The paired analysis supports the encoder-induced pathway across access settings. Repetition increases label-only AUC by 0.068 ± 0.026 (95% CI [0.049, 0.085]) and pooled LiRA AUC by 0.044 ± 0.045 ([0.013, 0.072]). In fixed-depth comparisons, LiRA AUC rises with repetition in five of six configurations; the single ZZ/depth=2 exception is unresolved. Z and ZZ remain higher than EffSU2 for both attacks, while the ZZ−Z intervals include zero. These calibrated attacks therefore support the claim that encoder family and repetition condition the downstream membership signal. Variational depth can modulate the signal produced by the trained model, but that does not remove the upstream encoder association.

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

---

# Response to Reviewer nVBH

Thank you. The submission provides broad exploratory coverage across datasets, QNN/HQNN/QCNN wrappers, and circuit choices. The rebuttal adds a focused complete 3×2×2 MNIST-QNN factorial over feature-map family, repetition, and depth, with three independently initialized targets per configuration (36 targets). This tests the central structural claim with replication while retaining the original sweep as exploratory breadth.

## Q1. Multi-seed robustness of the structural and attack claims

The three target seeds quantify initialization sensitivity; the learned attacker is also repeated over three training seeds, and LiRA uses 16 references per structural configuration. Mean ± SD and paired hierarchical-bootstrap intervals are reported. For loss-MIA, reps=5 minus reps=1 is +0.069 ± 0.029 AUC (95% CI [0.049, 0.088]), depth=6 minus depth=2 is +0.042 ± 0.026 ([0.024, 0.060]), Z−EffSU2 is +0.057 ± 0.032 ([0.027, 0.080]), and ZZ−EffSU2 is +0.052 ± 0.027 ([0.030, 0.074]). These results support the claim that encoder family and repetition systematically condition membership leakage across target initializations.

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

The access-model comparison adds nuance to the hierarchy: repetition has a positive pooled effect for LiRA and label-only attacks; Z and ZZ exceed EffSU2 under both; and ZZ−Z remains unresolved. Under loss and label-only attacks the repetition contrast exceeds the depth contrast, whereas LiRA is also strongly modulated by depth. We therefore claim that encoder choice and repetition are upstream privacy-relevant factors, not that they dominate every downstream attack statistic. The primary data seed remains fixed, so model seeds are not independent data-split replications.

## Q2. All-configuration evaluation and post-hoc selection

We do not reuse the submission's post-hoc baseline/stress/hard labels for confirmatory inference. Every one of the 36 prespecified targets receives the same loss-threshold, learned-vector, online LiRA, and label-only analyses; complete results follow.

| Structural configuration | Target seeds | Loss threshold AUC | Learned-vector AUC | Online LiRA AUC | Label-only AUC |
| --- | --- | --- | --- | --- | --- |
| EffSU2, reps=1, depth=2 | 3 | 0.530 ± 0.010 | 0.558 ± 0.008 | 0.530 ± 0.032 | 0.528 ± 0.010 |
| EffSU2, reps=1, depth=6 | 3 | 0.536 ± 0.006 | 0.510 ± 0.027 | 0.542 ± 0.020 | 0.518 ± 0.011 |
| EffSU2, reps=5, depth=2 | 3 | 0.571 ± 0.006 | 0.517 ± 0.039 | 0.562 ± 0.029 | 0.557 ± 0.015 |
| EffSU2, reps=5, depth=6 | 3 | 0.603 ± 0.017 | 0.568 ± 0.023 | 0.632 ± 0.037 | 0.604 ± 0.014 |
| Z, reps=1, depth=2 | 3 | 0.541 ± 0.017 | 0.481 ± 0.030 | 0.578 ± 0.014 | 0.515 ± 0.020 |
| Z, reps=1, depth=6 | 3 | 0.599 ± 0.015 | 0.542 ± 0.028 | 0.618 ± 0.023 | 0.578 ± 0.013 |
| Z, reps=5, depth=2 | 3 | 0.646 ± 0.012 | 0.602 ± 0.010 | 0.604 ± 0.025 | 0.613 ± 0.008 |
| Z, reps=5, depth=6 | 3 | 0.684 ± 0.012 | 0.638 ± 0.019 | 0.693 ± 0.023 | 0.651 ± 0.007 |
| ZZ, reps=1, depth=2 | 3 | 0.562 ± 0.012 | 0.557 ± 0.017 | 0.583 ± 0.007 | 0.539 ± 0.015 |
| ZZ, reps=1, depth=6 | 3 | 0.604 ± 0.011 | 0.606 ± 0.034 | 0.671 ± 0.004 | 0.612 ± 0.030 |
| ZZ, reps=5, depth=2 | 3 | 0.601 ± 0.025 | 0.595 ± 0.020 | 0.566 ± 0.023 | 0.601 ± 0.022 |
| ZZ, reps=5, depth=6 | 3 | 0.679 ± 0.018 | 0.714 ± 0.040 | 0.727 ± 0.008 | 0.671 ± 0.025 |

*Entries are mean ± sample SD across three target seeds. Learned-vector AUC is first averaged across three attacker seeds.*

Across all 36 targets, accuracy gap and loss-AUC have Pearson r=0.948 (95% CI [0.864, 0.982]) and Spearman ρ=0.931 ([0.710, 0.974]). This answers the circular-selection concern within the focused factorial and supports gap as a strong descriptive proxy, not a deterministic or externally validated predictor. It is not a claim that every configuration in the original thousands-run sweep was retrained.

## Q3. Mathematical definition of the factor-attribution figure

Fig. 8 uses a dataset-wise normalized factor-association score for the prespecified privacy-risk proxy $y=\Delta_{\mathrm{gen}}$. Let $N$ be the number of runs and $g_j$ the number of levels of categorical factor j. Its one-way ANOVA statistic $F_j$ is transformed as:

$$s_j^{(\mathrm{cat})} = F_j / (F_j + N - g_j).$$

For a numeric factor and displayed numeric product term, respectively:

$$s_j^{(\mathrm{num})} = |\operatorname{corr}(X_j,y)|,  \qquad s_{jk} = |\operatorname{corr}(X_j X_k,y)|.$$

The plotted allocation is:

$$A_j = 100s_j / \sum_k s_k,  \qquad \sum_j A_j = 100.$$

The component scores are nonnegative, dimensionless, and bounded by one. Thus, 22.5% means 22.5% of the aggregate factor-association score over the displayed terms within that dataset. This hybrid ANOVA/correlation diagnostic is a broad-sweep descriptive ranking, not a Sobol decomposition or conditional joint-model coefficient. Direct MIA validation is separate: the balanced factorial's paired loss-AUC intervals corroborate the feature-map and repetition associations without relying on this normalization. We will add the equations and use ‘normalized factor-association share’ in the revision.

## Q4. Architecture controls

We added QNN, HQNN, QCNN, and a classical MLP with a comparable small parameter budget to the QNN roles, evaluated over three structural roles and three target seeds with exact parameter and main-stack gate counts. Relative to QNN, MLP improves test accuracy by +0.146 ± 0.093 ([0.062, 0.259]) while its gap and AUC differences are unresolved. QCNN improves test accuracy by +0.190 ± 0.052 ([0.148, 0.249]) and reduces gap by −0.041 ± 0.042 ([−0.083, −0.007]) and loss-AUC by −0.019 ± 0.025 ([−0.040, −0.001]); HQNN intervals cross zero. These results support architecture as a moderator of the encoder-induced signal. Because preprocessing and heads remain different, they are complete-wrapper controls rather than matched causal ablations.

## Q5. Stronger baselines and bounded conclusions

Beyond the classical MLP model control, we broadened the adversaries to separate scalar thresholds, a learned prediction-vector attacker, calibrated online/offline LiRA, and a class-label-only boundary proxy. This tests whether the structural signal is specific to one attack or access model.

| Attack | Information access | AUC | TPR@5% FPR | TPR@10% FPR |
| --- | --- | --- | --- | --- |
| Loss threshold | true-label probability + known candidate label | 0.596 ± 0.052 | 0.061 ± 0.024 | 0.135 ± 0.039 |
| Learned prediction-vector | full prediction vector + statistics | 0.574 ± 0.065 | 0.099 ± 0.044 | 0.157 ± 0.061 |
| Online LiRA, fixed variance | true-label probability + calibrated reference QNNs | 0.609 ± 0.063 | 0.128 ± 0.056 | 0.192 ± 0.069 |
| Online LiRA, per-record variance | true-label probability + calibrated reference QNNs | 0.594 ± 0.057 | 0.086 ± 0.034 | 0.155 ± 0.063 |
| Offline LiRA, fixed variance | true-label probability + calibrated reference QNNs | 0.517 ± 0.029 | 0.066 ± 0.027 | 0.128 ± 0.035 |
| Label-only chord-boundary | predicted class labels only + held-out anchors | 0.582 ± 0.052 | 0.077 ± 0.027 | 0.139 ± 0.030 |

We did not add classical CNN/kernel, quantum-kernel, regularized/early-stopped, differentially private, or calibration-defense baselines. The confirmatory factorial also does not provide new multi-seed validation of width, entangler, gate, or padding trends. Dataset scope remains synthetic tasks and compressed four-class MNIST. Sensitive domains motivate why membership matters under established classical-ML threats and proposed QML use cases; they are not deployment-validation claims. We will bound conclusions to the evaluated simulated datasets and avoid claims of sensitive-domain deployment readiness.

---
