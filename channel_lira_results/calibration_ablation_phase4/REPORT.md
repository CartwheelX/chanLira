# ChannelLiRA calibration source/count ablation

## Answer

This experiment separates the two explanations for the stronger Phase-3
leave-cell-out result: cross-architecture calibration and the increase from two to
twelve auxiliary targets. At 1024 shots, moving from cross-cell two-target
calibration to cross-cell twelve-target calibration changes affine ChannelLiRA AUC
by +0.0016. Changing from same-cell two-target calibration to the
cross-cell two-target median changes it by +0.0018.

The two-target cross-cell condition exhausts all 66 auxiliary-target pairs for every
held cell. Four/eight-target conditions use 32 frozen
unique subsets per held cell. Main intervals first take the median over calibration
subsets within each simulator seed, then report the 5th–95th percentiles over ten
seeds. They are sensitivity ranges, not confidence intervals.

## Pooled results at 1024 noisy shots

| Calibration strategy | Aux targets | ChannelLiRA AUC | Mismatched AUC | Loss AUC | Actual FPR | Calibrated TPR |
|---|---:|---:|---:|---:|---:|---:|
| same_cell_2 | 2 | 0.6013 | 0.5966 | 0.5775 | 0.0123 | 0.0250 |
| cross_cell_2 | 2 | 0.6013 | 0.5966 | 0.5775 | 0.0133 | 0.0295 |
| cross_cell_4 | 4 | 0.6022 | 0.5966 | 0.5775 | 0.0117 | 0.0282 |
| cross_cell_8 | 8 | 0.6026 | 0.5966 | 0.5775 | 0.0110 | 0.0263 |
| cross_cell_12 | 12 | 0.6029 | 0.5966 | 0.5775 | 0.0103 | 0.0267 |

## Paired comparisons with mismatched LiRA

| Calibration strategy | AUC difference [5%, 95%] | Calibrated-TPR difference [5%, 95%] |
|---|---:|---:|
| same_cell_2 | +0.0049 [+0.0008, +0.0094] | +0.0043 [+0.0019, +0.0088] |
| cross_cell_2 | +0.0072 [+0.0016, +0.0106] | +0.0078 [+0.0068, +0.0104] |
| cross_cell_4 | +0.0085 [+0.0025, +0.0122] | +0.0087 [+0.0070, +0.0138] |
| cross_cell_8 | +0.0085 [+0.0030, +0.0123] | +0.0077 [+0.0060, +0.0122] |
| cross_cell_12 | +0.0087 [+0.0032, +0.0124] | +0.0080 [+0.0059, +0.0132] |

## Direct calibration-strategy contrasts

These comparisons pair strategies within simulator seed after taking the median
over their calibration subsets.

| Contrast | ChannelLiRA AUC difference [5%, 95%] | Actual-FPR difference | Calibrated-TPR difference [5%, 95%] |
|---|---:|---:|---:|
| cross_cell_2_minus_same_cell_2 | +0.0018 [-0.0007, +0.0032] | +0.0013 | +0.0038 [+0.0015, +0.0064] |
| cross_cell_12_minus_cross_cell_2 | +0.0016 [+0.0009, +0.0018] | -0.0027 | -0.0032 [-0.0045, -0.0016] |

Across the 66 cross-cell two-target subsets, the seed-median ChannelLiRA-minus-
mismatched-LiRA AUC contrast has median
+0.0070 and subset sensitivity range
[+0.0030,
+0.0100]. This quantifies how strongly
the conclusion depends on which two auxiliary QNNs are chosen.

## Empirical 0.1% FPR check at 1024 shots

| Attack | TPR at empirical 0.1% FPR [5%, 95%] |
|---|---:|
| Affine ChannelLiRA (`cross_cell_12`) | 0.0033 [0.0020, 0.0053] |
| Mismatched LiRA | 0.0047 [0.0033, 0.0068] |
| Loss MIA | 0.0020 [0.0010, 0.0027] |

The pooled evaluation has only 1500 nonmembers, so one
false positive changes FPR by
0.0667 percentage points. At this coarse
resolution, affine ChannelLiRA does not beat mismatched LiRA. These values are
reported for transparency and do not support a stable 0.1%-FPR claim; the candidate
population must be enlarged before publication.

## Held-out channel diagnostics at 1024 shots

| Calibration strategy | Held-out R² | Held-out 90% coverage |
|---|---:|---:|
| same_cell_2 | 0.876 | 0.896 |
| cross_cell_2 | 0.852 | 0.871 |
| cross_cell_4 | 0.855 | 0.894 |
| cross_cell_8 | 0.862 | 0.905 |
| cross_cell_12 | 0.867 | 0.910 |

## Interpretation

- Only `cross_cell_2` versus `cross_cell_12` cleanly isolates auxiliary calibration
  count under the same complete-cell holdout. Twelve targets modestly improve AUC
  but also produce a more conservative threshold: lower realized FPR and lower TPR.
- `same_cell_2` versus `cross_cell_2` holds auxiliary count fixed, but changes the
  holdout unit and calibration source. Its paired AUC interval crosses zero, and its
  TPR increase comes with a higher median FPR, so this experiment does not establish
  cross-architecture regularization.
- Mismatched LiRA AUC is channel-independent and therefore provides a flat reference
  across calibration counts. Its operational threshold can still vary with the
  selected auxiliary targets.
- All results reuse one frozen IBM-derived simulator snapshot and architecture-
  matched exact reference banks for the held cells.
