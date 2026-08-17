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
