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
