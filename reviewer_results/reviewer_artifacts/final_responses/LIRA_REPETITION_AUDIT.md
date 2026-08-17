# LiRA repetition and reference-bank audit

## Fixed-depth repetition comparison

| Feature map | Fixed depth | Gap, reps=1 | Gap, reps=5 | Paired Δ gap | LiRA AUC, reps=1 | LiRA AUC, reps=5 | Paired Δ LiRA AUC | Δ LiRA 95% seed-bootstrap CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EffSU2 | 2 | 0.023 ± 0.029 | 0.123 ± 0.036 | +0.100 ± 0.065 | 0.530 ± 0.032 | 0.562 ± 0.029 | +0.032 ± 0.048 | [-0.013, 0.083] |
| EffSU2 | 6 | 0.038 ± 0.019 | 0.182 ± 0.028 | +0.143 ± 0.018 | 0.542 ± 0.020 | 0.632 ± 0.037 | +0.090 ± 0.046 | [0.037, 0.122] |
| Z | 2 | 0.060 ± 0.041 | 0.212 ± 0.029 | +0.152 ± 0.013 | 0.578 ± 0.014 | 0.604 ± 0.025 | +0.025 ± 0.017 | [0.009, 0.044] |
| Z | 6 | 0.167 ± 0.016 | 0.298 ± 0.003 | +0.132 ± 0.015 | 0.618 ± 0.023 | 0.693 ± 0.023 | +0.075 ± 0.028 | [0.052, 0.106] |
| ZZ | 2 | 0.117 ± 0.025 | 0.183 ± 0.025 | +0.067 ± 0.025 | 0.583 ± 0.007 | 0.566 ± 0.023 | -0.017 ± 0.025 | [-0.036, 0.011] |
| ZZ | 6 | 0.205 ± 0.017 | 0.350 ± 0.035 | +0.145 ± 0.018 | 0.671 ± 0.004 | 0.727 ± 0.008 | +0.056 ± 0.005 | [0.053, 0.062] |

*Every comparison fixes feature-map family and variational depth and pairs reps=1 with reps=5 by target-model seed. Entries are mean ± sample SD across three paired target seeds. The displayed intervals resample the three paired seed effects and are necessarily coarse at n=3.*

## Fixed-depth repetition comparison using attack balanced accuracy

| Feature map | Fixed depth | Paired Δ gap | LiRA balanced accuracy, reps=1 | LiRA balanced accuracy, reps=5 | Paired Δ LiRA balanced accuracy | Δ balanced accuracy 95% seed-bootstrap CI |
| --- | --- | --- | --- | --- | --- | --- |
| EffSU2 | 2 | +0.100 ± 0.065 | 0.515 ± 0.026 | 0.565 ± 0.016 | +0.050 ± 0.039 | [0.022, 0.095] |
| EffSU2 | 6 | +0.143 ± 0.018 | 0.513 ± 0.017 | 0.584 ± 0.028 | +0.071 ± 0.036 | [0.030, 0.095] |
| Z | 2 | +0.152 ± 0.013 | 0.547 ± 0.011 | 0.573 ± 0.013 | +0.025 ± 0.003 | [0.023, 0.028] |
| Z | 6 | +0.132 ± 0.015 | 0.565 ± 0.020 | 0.604 ± 0.027 | +0.039 ± 0.044 | [0.010, 0.090] |
| ZZ | 2 | +0.067 ± 0.025 | 0.562 ± 0.001 | 0.526 ± 0.008 | -0.036 ± 0.009 | [-0.045, -0.028] |
| ZZ | 6 | +0.145 ± 0.018 | 0.607 ± 0.004 | 0.662 ± 0.033 | +0.055 ± 0.035 | [0.015, 0.075] |

*Balanced attack accuracy uses the five-fold cross-fitted LiRA threshold. ROC AUC remains the primary threshold-independent metric.*

## Interpretation

Repetition increases the accuracy gap in all six matched comparisons and increases both fixed-variance online LiRA AUC and cross-fitted balanced attack accuracy in five of six. The exception is ZZ at depth 2: ΔAUC = −0.017 ± 0.025 with a seed-bootstrap interval [−0.036, 0.011], while cross-fitted balanced accuracy decreases by −0.036 ± 0.009. Thus, the repetition effect is non-uniform for this configuration even though the pooled repetition effect remains positive (+0.044 ± 0.045; hierarchical-bootstrap CI [0.013, 0.072]).

The LiRA results do not indicate that the implementation ignored repetition: the pooled repetition contrast is positive and five of six matched directions are positive. Variational depth also changes LiRA AUC, particularly for the deep ZZ configurations. LiRA compares each target score to calibrated IN/OUT reference distributions, so it need not be a monotone transformation of the target's aggregate accuracy gap. The geometry measurements are pre-ansatz; they establish that repetition changes the encoded representation but do not require repetition to dominate every post-training attack statistic.

## Reference-bank integrity audit

| Structural configuration | References | Candidate fingerprints | IN references/candidate | Training records/reference | Epochs | Learning rate | Audit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eff_su2_r1_d2 | 16 | 1 | 8 | 200 | 100 | 0.05 | PASS |
| eff_su2_r1_d6 | 16 | 1 | 8 | 200 | 100 | 0.05 | PASS |
| eff_su2_r5_d2 | 16 | 1 | 8 | 200 | 100 | 0.05 | PASS |
| eff_su2_r5_d6 | 16 | 1 | 8 | 200 | 100 | 0.05 | PASS |
| z_r1_d2 | 16 | 1 | 8 | 200 | 100 | 0.05 | PASS |
| z_r1_d6 | 16 | 1 | 8 | 200 | 100 | 0.05 | PASS |
| z_r5_d2 | 16 | 1 | 8 | 200 | 100 | 0.05 | PASS |
| z_r5_d6 | 16 | 1 | 8 | 200 | 100 | 0.05 | PASS |
| zz_r1_d2 | 16 | 1 | 8 | 200 | 100 | 0.05 | PASS |
| zz_r1_d6 | 16 | 1 | 8 | 200 | 100 | 0.05 | PASS |
| zz_r5_d2 | 16 | 1 | 8 | 200 | 100 | 0.05 | PASS |
| zz_r5_d6 | 16 | 1 | 8 | 200 | 100 | 0.05 | PASS |

*All 12 banks pass: the structural identifier matches the directory, each bank has one consistent candidate fingerprint, every candidate is IN in exactly 8/16 references, and every reference trains for 100 epochs on 200 records.*
