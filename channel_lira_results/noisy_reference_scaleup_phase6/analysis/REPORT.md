# Phase 6: 16-reference noisy scale-up

This is a one-cell scale and comparison gate. It is stronger than the four-reference canary but is not a publication-level cross-cell result.

## Frozen protocol

- Targets: 3 independently initialized checkpoints in `eff_su2_r1_d2`.
- References: 16, with exactly 8 IN and 8 OUT observations per candidate.
- Modes: `ideal_shot,noisy_shot`; shots: 128; channel scheme: strict leave-target-out.
- Channel calibration uses the other two target checkpoints and excludes the attacked sample-ID fold.
- The learned baseline is target-cross-fitted on labeled attacked-target outputs and therefore has stronger auxiliary access than LiRA.
- Snapshot manifest SHA-256: `7af4abc6763339615cc42bd00fafb532e3c44a31c14bc574c88456a496136a2f`.

## Target-level AUC summary

Values are means across three target checkpoints after averaging simulator seeds; ± is sample SD across targets.

| Mode | Attack | Mean AUC | SD across targets | Target range |
|---|---|---:|---:|---:|
| ideal_shot | matched 16-reference LiRA | 0.4924 | 0.0256 | [0.4672, 0.5184] |
| ideal_shot | leave-target-out ChannelLiRA | 0.5091 | 0.0112 | [0.4972, 0.5194] |
| ideal_shot | mismatched latent LiRA | 0.4859 | 0.0131 | [0.4730, 0.4992] |
| ideal_shot | loss MIA | 0.5302 | 0.0170 | [0.5119, 0.5455] |
| ideal_shot | target-cross-fitted learned MIA | 0.4921 | 0.0064 | [0.4847, 0.4960] |
| noisy_shot | matched 16-reference LiRA | 0.5157 | 0.0013 | [0.5143, 0.5169] |
| noisy_shot | leave-target-out ChannelLiRA | 0.5199 | 0.0082 | [0.5105, 0.5256] |
| noisy_shot | mismatched latent LiRA | 0.4978 | 0.0140 | [0.4830, 0.5108] |
| noisy_shot | loss MIA | 0.5333 | 0.0104 | [0.5254, 0.5450] |
| noisy_shot | target-cross-fitted learned MIA | 0.4991 | 0.0396 | [0.4534, 0.5246] |

## Paired target-level AUC contrasts

Positive values favor the attack named first. Ranges are descriptive minima/maxima over three targets, not confidence intervals.

| Mode | Contrast | Mean difference | Target range |
|---|---|---:|---:|
| ideal_shot | affine_minus_matched_reference | +0.0167 | [-0.0212, +0.0432] |
| ideal_shot | matched_reference_minus_learned | +0.0003 | [-0.0283, +0.0224] |
| ideal_shot | matched_reference_minus_loss | -0.0378 | [-0.0783, +0.0065] |
| ideal_shot | matched_reference_minus_mismatched | +0.0065 | [-0.0184, +0.0454] |
| noisy_shot | affine_minus_matched_reference | +0.0043 | [-0.0063, +0.0098] |
| noisy_shot | matched_reference_minus_learned | +0.0166 | [-0.0103, +0.0635] |
| noisy_shot | matched_reference_minus_loss | -0.0176 | [-0.0292, -0.0112] |
| noisy_shot | matched_reference_minus_mismatched | +0.0179 | [+0.0051, +0.0339] |

## Interpretation limits

- Three checkpoints from one structural cell are a scale-up gate, not evidence of cross-model or cross-architecture generalization.
- Simulator seeds quantify serving randomness but do not increase the independent model-level sample size.
- The 400-candidate target pools cannot support a stable 0.1% FPR claim.
- Successful completion is the go/no-go gate for the prespecified 15-target/80-reference study.
