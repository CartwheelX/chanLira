# Response to Reviewer nVBH

Thank you. We added a focused complete MNIST-QNN factorial (3 feature maps × 2 repetitions × 2 depths × 3 target seeds = 36 targets) and applied the same attacks to every target without outcome selection. Full protocols, per-configuration results, attribution equations, resource counts, and limitations are in our [detailed response](https://github.com/CartwheelX/QuRiFT/blob/main/docs/Reviewer-nVBH.md).

**1. Multi-seed robustness.**

Values below are paired AUC differences with 95% hierarchical-bootstrap CIs.

| Attack | Reps 5−1 | Depth 6−2 | Z−EffSU2 | ZZ−EffSU2 |
| --- | --- | --- | --- | --- |
| Loss | +.069 [.049,.088] | +.042 [.024,.060] | +.057 [.027,.080] | +.052 [.030,.074] |
| Online LiRA | +.044 [.013,.072] | +.077 [.038,.118] | +.057 [.038,.076] | +.070 [.021,.117] |
| Label-only | +.068 [.049,.085] | +.047 [.021,.068] | +.038 [.004,.061] | +.054 [.022,.087] |

These results support the claim that encoder family and repetition condition leakage across target initializations and access models. Repetition exceeds depth for loss and label-only; LiRA is also strongly depth-modulated, so we do not claim encoder factors dominate every attack statistic. Each configuration has three target seeds; the learned attacker also has three training seeds and LiRA has 16 references/configuration. The data split is fixed, so these are not independent split replications.

**2. Post-hoc selection and gap validation.**

We do not reuse baseline/stress/hard labels for confirmation. Across all 36 prespecified targets, gap and loss-AUC have Pearson r=.948 (CI [.864,.982]) and Spearman ρ=.931 ([.710,.974]). This removes attack-outcome selection within the focused factorial and supports gap as a strong descriptive proxy, not a deterministic predictor. It is not a multi-seed rerun of every configuration in the original broad sweep.

**3. Factor attribution.**

Fig. 8 is a dataset-wise normalized factor-association analysis of the generalization-gap proxy. Categorical terms use an ANOVA-derived score; numeric and displayed product terms use absolute Pearson correlation; scores are normalized to sum to 100. Hence a reported percentage is a share of the aggregate displayed association score, not a Sobol or causal allocation. The detailed response gives the exact equations. The new paired MIA intervals separately corroborate the feature-map/repetition associations.

**4. Architecture controls.**

Across three roles × three seeds, we evaluated QNN/HQNN/QCNN and a classical MLP with a comparable small parameter budget to the QNN roles, reporting exact parameter/gate counts. Relative to QNN, MLP test accuracy changes by +.146 ± .093 (CI [.062,.259]) with unresolved gap/AUC effects; QCNN changes accuracy by +.190 ± .052 ([.148,.249]), gap by −.041 ± .042 ([−.083,−.007]), and loss-AUC by −.019 ± .025 ([−.040,−.001]); HQNN intervals cross zero. This supports architecture as a moderator, but wrappers retain different preprocessing/heads and are not matched causal ablations.

**5. Baselines and scope.**

We added the classical MLP control and attacks spanning scalar thresholds, learned prediction vectors, calibrated LiRA, and label-only access. We did not add every proposed model/defense baseline. The focused factorial does not revalidate width, gates, entanglers, or padding with multiple seeds, and datasets remain synthetic plus compressed four-class MNIST. Sensitive domains motivate why membership matters; they are not deployment-validation claims. We will state these boundaries explicitly.
