# ChannelLiRA feasibility pilot

## Verdict

**YES — proceed to the next-stage study, with calibration as the first gate.**

This is a controlled binary-output proxy-channel intervention on retained,
already-trained QNN checkpoints. It places a Bernoulli/readout channel on the final
true-class probability; it does **not** simulate circuit measurement shots before the
classical head. It is not yet evidence about real hardware drift or classical
stochastic models.

## Decisive results

| Quantity | Result |
|---|---:|
| Structural cells | 12 |
| Fixed target checkpoints | 36 |
| Balanced reference models per cell | 16 |
| Target proxy channel | Bernoulli serving + symmetric readout error 0.120 |
| Reference/assumed channel | symmetric readout error 0.000 |
| Latent-oracle AUC | 0.6028 |
| Matched ChannelLiRA AUC, 32 shots | 0.5995 |
| Matched ChannelLiRA AUC, 1024 shots | 0.6053 |
| Mismatched ChannelLiRA AUC, 1024 shots | 0.5790 |
| Deconvolved-mean LiRA AUC, 1024 shots | 0.5968 |
| Matched channel transfer advantage, 1024 shots | +0.0262 AUC [+0.0241, +0.0279] |
| Cells with positive transfer advantage | 10/12 |
| Latent leakage recovered, 1024 shots | 102.4% |
| Equal-total-shot log-kernel discrepancy | 1.14e-13 |
| Equal-total-shot aggregation discrepancy | 0 |

## Nominal 1% FPR calibration

| Attack | Shots | Actual FPR | TPR |
|---|---:|---:|---:|
| channel_lira_matched | 32 | 0.0335 | 0.0755 |
| channel_lira_mismatched | 32 | 0.0546 | 0.0826 |
| channel_lira_matched | 1024 | 0.0539 | 0.0970 |
| channel_lira_mismatched | 1024 | 0.0758 | 0.1047 |

The calibration threshold is generated from each attack's assumed OUT predictive
distribution, then applied to target observations served through the actual channel.
The pooled target-record units are dependent across model architectures and seeds, so
these values are diagnostic rather than publication-ready confidence bounds.

**Critical caveat:** the mismatched-minus-matched actual-FPR gap is
0.0219 [0.0206,
0.0244] at 1024 shots,
but the matched attack still realizes 5.39% FPR. The channel model
helps, while the latent reference-to-target model remains materially miscalibrated.

## What this pilot tests

1. It uses real latent IN/OUT distributions from the repository's 16-model balanced
   LiRA banks, not a fabricated membership gap.
2. It holds target and reference checkpoints fixed and intervenes only on the serving
   channel.
3. It compares ordinary repeated-mean LiRA, channel inversion without uncertainty,
   a mismatched hierarchical likelihood, and the correctly matched likelihood.
4. It controls total shots. Under the stationary binomial channel, query splits reduce
   exactly to aggregate counts; the implementation verifies that equality.

## Decision rule used before interpreting the output

The automatic **YES** requires (a) at least a 0.005 AUC advantage over the mismatched
hierarchical attack at the largest budget, (b) at least 50% recovery of
the latent-oracle AUC advantage, (c) non-decreasing matched AUC from the smallest to
largest budget, and (d) lower FPR error than the mismatched attack. This advances the
idea to a next-stage calibration/drift study; it is not a hardware or publication claim.

Leakage recovery can slightly exceed 100% because the attack model is estimated and
evaluation is finite: serving noise may regularize a misspecified score, but cannot
create membership information under the stated Markov channel.

## Artifacts

- `metrics_raw.csv`: every stochastic replicate, attack, budget, and cell.
- `metrics_summary.csv`: median and 5–95% simulation interval.
- `calibration_raw.csv` and `calibration_summary.csv`: assumed-null threshold checks.
- `paired_contrasts_summary.csv`: paired matched-minus-baseline effects.
- `calibration_contrasts_summary.csv`: paired calibration improvements.
- `leakage_recovery.svg`: overall attack curves.
- `pilot_config.json`: complete intervention parameters and source inventory.
- `sufficiency_check.json`: equal-total-shot aggregation audit.
