# Response to Reviewer 1myw: geometry and noise

## Suggested response

Thank you for identifying these two gaps. We agree that the original manuscript did not directly measure the Appendix-D geometry and that a noiseless hierarchy alone cannot establish robustness. We therefore added two targeted analyses.

**Noise and finite shots.** We evaluated five prespecified configurations spanning the observed risk range, using three independently trained target checkpoints per configuration and ten simulator seeds at 128, 512, and 1,024 shots. We compared ideal-shot simulation with an Aer noise model derived from the `ibm_kingston` calibration snapshot (gate, readout, and thermal-relaxation errors). This comprises 15 targets and 915 target/execution replicates. The five configuration means retain the exact loss-AUC ordering in every aggregate condition. Noise nevertheless has a non-uniform mitigating effect: the paired high-minus-low AUC difference decreases from 0.179 ± 0.030 exactly to 0.096 ± 0.012 at 128 noisy shots, then increases to 0.124 ± 0.026 and 0.132 ± 0.036 at 512 and 1,024 shots. Individual finite-shot replicates can locally reorder nearby middle configurations, especially at 128 shots; the mean rank correlation with the exact five-configuration hierarchy is 0.820 ± 0.155 under 128-shot backend noise and 0.960 ± 0.052 at 1,024 shots. Thus, noise attenuates the severity but does not reverse or eliminate the broad hierarchy in this check. We will clearly label this as backend-derived simulation rather than hardware execution and will not claim universality across devices/calibrations.

**Direct geometry.** We now compute the pure-state Hilbert–Schmidt/fidelity kernel immediately after the fixed encoder and before the variational circuit. Across dataset/encoder blocks, increasing repetitions from 1 to 5 significantly changes class-conditioned similarity, kernel–label alignment, and effective rank; the train/test MMD² interval includes zero. These results directly verify the first link of the proposed pathway—changing the encoder changes its induced geometry.

We also agree with the reviewer's causal distinction. Our intended hypothesis is not that geometry bypasses overfitting: overfitting is the proposed mediator through which encoder structure becomes visible to an MIA. The new measurements support an encoder → geometry association, while the factorial results support structure → gap and structure → MIA associations, and gap strongly tracks loss-AUC. They do not causally identify geometry → gap independently of encoder choice. We will therefore replace causal wording such as ‘geometric mechanism’ or ‘geometry creates separation’ with ‘empirically supported geometric pathway/association,’ explicitly state that mediation is not causally identified, and add the following tables and the corresponding geometry/noise figures to the revision.

## Table 1. Loss-MIA AUC across the five-configuration hierarchy

| Structural configuration | Exact | Noisy 128 | Noisy 512 | Noisy 1024 |
| --- | --- | --- | --- | --- |
| EffSU2, reps=1, depth=2 | 0.521 ± 0.018 | 0.526 ± 0.020 | 0.527 ± 0.013 | 0.525 ± 0.019 |
| EffSU2, reps=5, depth=2 | 0.559 ± 0.014 | 0.544 ± 0.016 | 0.552 ± 0.024 | 0.554 ± 0.022 |
| Z, reps=1, depth=6 | 0.568 ± 0.010 | 0.550 ± 0.017 | 0.554 ± 0.011 | 0.557 ± 0.014 |
| ZZ, reps=1, depth=6 | 0.585 ± 0.013 | 0.572 ± 0.015 | 0.582 ± 0.022 | 0.587 ± 0.021 |
| ZZ, reps=5, depth=6 | 0.700 ± 0.037 | 0.622 ± 0.031 | 0.651 ± 0.034 | 0.657 ± 0.034 |

*Entries are mean ± sample SD across three independently trained target-model seeds after averaging ten simulator seeds for shot-based conditions.*

## Table 2. Hierarchy robustness under finite shots and backend-derived noise

| Condition | Low-risk AUC | High-risk AUC | Paired high − low | Aggregate order retained | Spearman ρ ± SD | Simulator seeds |
| --- | --- | --- | --- | --- | --- | --- |
| Exact | 0.521 ± 0.018 | 0.700 ± 0.037 | +0.179 ± 0.030 | Yes (5/5) | 1.000 (reference) | — |
| Ideal shot, 128 | 0.518 ± 0.015 | 0.671 ± 0.033 | +0.153 ± 0.018 | Yes (5/5) | 0.900 ± 0.115 | 10 |
| Ideal shot, 512 | 0.520 ± 0.012 | 0.691 ± 0.031 | +0.171 ± 0.021 | Yes (5/5) | 0.990 ± 0.032 | 10 |
| Ideal shot, 1024 | 0.521 ± 0.015 | 0.696 ± 0.037 | +0.175 ± 0.027 | Yes (5/5) | 0.990 ± 0.032 | 10 |
| Backend-noisy, 128 | 0.526 ± 0.020 | 0.622 ± 0.031 | +0.096 ± 0.012 | Yes (5/5) | 0.820 ± 0.155 | 10 |
| Backend-noisy, 512 | 0.527 ± 0.013 | 0.651 ± 0.034 | +0.124 ± 0.026 | Yes (5/5) | 0.920 ± 0.092 | 10 |
| Backend-noisy, 1024 | 0.525 ± 0.019 | 0.657 ± 0.034 | +0.132 ± 0.036 | Yes (5/5) | 0.960 ± 0.052 | 10 |

*Low/high AUCs and paired differences are mean ± sample SD across three paired target seeds. Spearman ρ is mean ± sample SD across ten simulator seeds; within each simulator seed, AUC is first averaged across the three target seeds for each of the five structural configurations. ‘Aggregate order’ uses means across target and simulator seeds. Local simulator-level reordering is therefore not hidden.*

## Table 3. Direct post-encoder geometry

| Post-encoder quantity | Operationalization | Reps 5 − 1, mean ± SD | 95% CI | Paired effects |
| --- | --- | --- | --- | --- |
| Class-similarity gap | Mean within-class minus between-class fidelity | -0.124 ± 0.061 | [-0.158, -0.079] | 12 |
| Kernel–label alignment | Centered fidelity-kernel/label-kernel alignment | -0.208 ± 0.132 | [-0.285, -0.109] | 12 |
| Effective rank | Spectral effective rank of the fidelity kernel | +49.747 ± 37.448 | [15.791, 83.870] | 12 |
| Train/test MMD² | Kernel two-sample discrepancy | -0.003 ± 0.008 | [-0.010, 0.003] | 12 |

*Effects are paired reps=5 minus reps=1 contrasts. SD is across unique paired effects and the intervals are 5,000-replicate hierarchical percentile-bootstrap CIs. MNIST nominal data seeds produce identical encoded states for its fixed subset; Moons supplies genuine data-seed variation, and duplicate MNIST states are not counted as independent effects.*

## Table 4. Evidence and revised claim boundary

| Link in proposed pathway | Evidence | Estimate | What may be claimed |
| --- | --- | --- | --- |
| Encoder repetition → geometry | Direct post-encoder fidelity-kernel measurements | See geometry table | Supported for the evaluated encoders/data |
| Repetition → generalization gap | Paired factorial contrast | +0.123 ± 0.041; CI [0.094, 0.147] | Supported association |
| Repetition → loss-MIA AUC | Paired factorial contrast | +0.069 ± 0.029; CI [0.049, 0.088] | Supported association |
| Gap ↔ loss-MIA AUC | Hierarchical-bootstrap Spearman correlation | ρ=0.931; CI [0.710, 0.974] | Strong descriptive association |
| Gap + structure → loss-MIA AUC | Descriptive regression | R²=0.935; CI [0.852, 0.986] | Descriptive, not causal mediation |
| Geometry → gap causally | No intervention on geometry independent of encoder | Not identified | Do not claim a causal mechanism/theorem |

## Revision commitments

- Add the direct post-encoder kernel measurements and paired uncertainty analysis.
- Add the five-configuration finite-shot/backend-noise results and hierarchy-rank analysis.
- Describe the noise experiment as IBM-backend-derived Aer simulation, not hardware.
- Report attenuation and local finite-shot reorderings, not only preserved aggregate order.
- Replace causal-mechanism wording with an empirically supported pathway and state that causal mediation remains unverified.
