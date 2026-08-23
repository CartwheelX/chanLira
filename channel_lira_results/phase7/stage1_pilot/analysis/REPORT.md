# Phase 7 Stage 1: pilot-cell engineering replication

This stage tests the enlarged candidate/checkpoint/noisy-serving pipeline. It is pilot replication evidence and is excluded from the four-cell confirmatory primary endpoint.

## Locked provenance

- Protocol SHA-256: `ab360b674c3843670cd4a2213924098a5a9ffef43607fd5ee6510f3c56a78d18`.
- Snapshot manifest SHA-256: `7af4abc6763339615cc42bd00fafb532e3c44a31c14bc574c88456a496136a2f`.
- Candidates: 1,000 members and 1,000 nonmembers per target.
- References: sixteen, exactly 8 IN / 8 OUT per candidate.
- Conditions: ideal/noisy 128 shots; simulator seeds 0 and 1.
- No real quantum-hardware execution.

## Descriptive attack results

Means first average serving seeds within each independently initialized target. SD is descriptive across three targets.

| Mode | Attack | Mean AUC | Mean TPR@1% FPR | SD AUC across targets |
|---|---|---:|---:|---:|
| ideal_shot | matched 16-reference LiRA | 0.4985 | 0.95% | 0.0086 |
| ideal_shot | leave-target-out ChannelLiRA | 0.4929 | 0.82% | 0.0097 |
| ideal_shot | mismatched latent LiRA | 0.4895 | 0.68% | 0.0036 |
| ideal_shot | loss MIA | 0.5223 | 0.87% | 0.0070 |
| ideal_shot | privileged victim-crossfit learned comparator | 0.5146 | 1.17% | 0.0155 |
| noisy_shot | matched 16-reference LiRA | 0.4923 | 1.05% | 0.0068 |
| noisy_shot | leave-target-out ChannelLiRA | 0.4993 | 1.05% | 0.0141 |
| noisy_shot | mismatched latent LiRA | 0.4960 | 0.78% | 0.0083 |
| noisy_shot | loss MIA | 0.5245 | 1.20% | 0.0067 |
| noisy_shot | privileged victim-crossfit learned comparator | 0.5129 | 0.78% | 0.0110 |

## Noisy paired contrasts

Ranges are the three observed targets, not confidence intervals.

| Contrast | Mean AUC difference | Mean TPR@1% difference | Target AUC range | Target TPR@1% range |
|---|---:|---:|---:|---:|
| affine_minus_loss | -0.0252 | -0.15 pp | [-0.0336, -0.0198] | [-0.50, +0.15] pp |
| affine_minus_matched_reference | +0.0071 | +0.00 pp | [-0.0020, +0.0186] | [-0.35, +0.50] pp |
| affine_minus_mismatched | +0.0033 | +0.27 pp | [-0.0055, +0.0100] | [-0.20, +0.60] pp |
| matched_reference_minus_mismatched | -0.0038 | +0.27 pp | [-0.0132, +0.0054] | [+0.10, +0.55] pp |

## Directional pilot check against the frozen success conditions

This is a descriptive three-target check, not a hypothesis test or confirmatory decision.

- **A, channel mismatch:** matched noisy LiRA minus mismatched LiRA is +0.27 pp TPR@1% FPR, with observed target range [+0.10, +0.55] pp. The primary-metric direction is supportive, although AUC is not.
- **B, efficient recovery:** ChannelLiRA minus matched noisy LiRA is +0.00 pp, with observed range [-0.35, +0.50] pp. That is directionally compatible with the frozen -0.5 pp margin, at 12.5% of matched-reference auxiliary shot cost per attack.
- **C, practical utility:** ChannelLiRA minus loss MIA is -0.15 pp, with observed range [-0.50, +0.15] pp. This pilot does not show superiority over loss MIA.

## Interpretation boundary

These results cannot confirm or change the frozen Phase-7 primary endpoint. Stage 1 passes scientifically when artifacts, hashes, balance, cache reuse, metrics, and cost receipts are complete; attack superiority is not an engineering pass requirement.

The learned comparator remains the unchanged privileged victim-crossfit implementation and does not support claims against learned MIAs generally.
