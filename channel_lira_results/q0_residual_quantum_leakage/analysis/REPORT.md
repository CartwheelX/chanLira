# Q0 residual quantum leakage screen

- Protocol SHA-256: `5936a137b84ee6c175f76077c9a61e9227e6882066b855483a6b357d03aef050`.
- Six independently data/model-seeded targets in two structural cells.
- Ten 128-shot queries per compared attack; local Aer with the frozen IBM-Kingston-derived snapshot.
- No reference models and no quantum-hardware execution.
- Screening evidence only; these results are excluded from Phase 7.

## Results

| Attack | Mean AUC | Mean TPR@1% | Mean actual FPR at transferred 1% threshold | Mean operational TPR | Loss-conditioned AUC |
|---|---:|---:|---:|---:|---:|
| Loss MIA | 0.5254 | 1.32% | 0.95% | 1.43% | 0.5036 |
| unchanged learned MIA | 0.5244 | 1.67% | 0.93% | 1.50% | 0.5120 |
| classical stochastic control | 0.5177 | 1.28% | 0.98% | 1.25% | 0.5089 |
| fixed-layout marginal probe | 0.5099 | 1.08% | 1.73% | 1.88% | 0.5068 |
| fixed-layout joint-bitstring probe | 0.5103 | 1.10% | 0.93% | 1.22% | 0.5032 |
| paired-layout probability probe | 0.5136 | 1.05% | 0.65% | 0.68% | 0.4986 |
| paired-layout marginal probe | 0.5026 | 0.92% | 1.42% | 1.68% | 0.4960 |
| paired-layout joint-bitstring probe | 0.5163 | 0.98% | 0.43% | 0.50% | 0.5131 |
| privileged clean-Z diagnostic | 0.5269 | 1.07% | 7.88% | 8.65% | 0.5165 |

## Decisive comparisons

| Contrast | Mean AUC difference | Mean TPR@1% difference | Mean loss-conditioned AUC difference |
|---|---:|---:|---:|
| paired_joint_minus_loss | -0.0092 | -0.33 pp | +0.0095 |
| paired_joint_minus_learned | -0.0082 | -0.68 pp | +0.0011 |
| paired_joint_minus_classical_stochastic | -0.0014 | -0.30 pp | +0.0043 |
| fixed_joint_minus_learned | -0.0141 | -0.57 pp | -0.0088 |
| paired_joint_minus_fixed_joint | +0.0059 | -0.12 pp | +0.0099 |

## Locked screening decision

- FAIL: `practical_gain_over_loss`.
- FAIL: `gain_over_unchanged_learned_mia`.
- FAIL: `conditional_gain_within_loss_strata`.
- PASS: `quantum_mechanism`.
- FAIL: `classical_stochastic_control`.
- FAIL: `operational_sanity`.

**Overall: FAIL — `stop_stronger_quantum_stochastic_mia_claim_under_this_design`.**

Passing would justify a larger preregistered study, not a breakthrough claim. Failure invokes the locked stop rule for this stronger-quantum-stochastic-MIA direction under the tested access model.

## Interpretation boundaries

- Loss and the learned MIA use probability outputs from ten fixed-layout queries.
- Raw marginal/joint attacks assume access to bitstring counts.
- Paired attacks additionally assume control over the physical initial layout.
- The clean-Z attack is a privileged mechanism diagnostic, not a deployable black-box baseline.
- The learned estimator architecture is unchanged; only its training data are moved to independent auxiliary targets.
