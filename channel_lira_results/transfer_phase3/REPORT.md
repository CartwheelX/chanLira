# ChannelLiRA unseen-target transfer report

## Verdict

**YES — target-level transfer survives the strict holdout; cell-level transfer also passes.**

This test removes the strongest Phase-2 threat-model objection. Under
`leave_target_out`, the attacked checkpoint is absent from both channel fitting and
threshold calibration. Under `leave_cell_out`, every checkpoint with the attacked
circuit structure is absent. Sample-ID folds also remove the attacked candidate ID
from auxiliary calibration records. Exact outputs of held targets are used only for
offline fit diagnostics, never for attack scoring or thresholds.

## Pooled results at 1024 IBM-derived noisy shots

The ROC operating points below are descriptive empirical-ROC quantities; they are
not the cross-fitted operational thresholds in the next table.

| Transfer scheme | Attack | AUC | ROC TPR @ 1% FPR | ROC TPR @ 5% FPR |
|---|---|---:|---:|---:|
| leave_target_out | loss_mia | 0.5775 | 0.0080 | 0.0403 |
| leave_target_out | latent_lira_mismatched | 0.5966 | 0.0147 | 0.0727 |
| leave_target_out | deconvolved_lira | 0.5896 | 0.0213 | 0.0813 |
| leave_target_out | affine_channel_lira | 0.6013 | 0.0303 | 0.1007 |
| leave_target_out | empirical_channel_lira | 0.5954 | 0.0270 | 0.0997 |
| leave_target_out | noise_augmented_lira | 0.6018 | 0.0287 | 0.1060 |
| leave_cell_out | loss_mia | 0.5775 | 0.0080 | 0.0403 |
| leave_cell_out | latent_lira_mismatched | 0.5966 | 0.0147 | 0.0727 |
| leave_cell_out | deconvolved_lira | 0.5939 | 0.0180 | 0.0803 |
| leave_cell_out | affine_channel_lira | 0.6029 | 0.0243 | 0.0960 |
| leave_cell_out | empirical_channel_lira | 0.6005 | 0.0217 | 0.1003 |
| leave_cell_out | noise_augmented_lira | 0.6069 | 0.0240 | 0.0937 |

## Cross-target thresholds at nominal 1% FPR

| Transfer scheme | Attack | Actual FPR [5%, 95%] | Calibrated TPR |
|---|---|---:|---:|
| leave_target_out | loss_mia | 0.0130 [0.0110, 0.0144] | 0.0130 |
| leave_target_out | latent_lira_mismatched | 0.0127 [0.0106, 0.0140] | 0.0207 |
| leave_target_out | affine_channel_lira | 0.0123 [0.0092, 0.0154] | 0.0250 |
| leave_cell_out | loss_mia | 0.0130 [0.0116, 0.0147] | 0.0087 |
| leave_cell_out | latent_lira_mismatched | 0.0107 [0.0093, 0.0148] | 0.0180 |
| leave_cell_out | affine_channel_lira | 0.0103 [0.0090, 0.0120] | 0.0267 |

## Paired transfer contrasts

| Transfer scheme | Contrast | AUC difference [5%, 95%] | Calibrated-TPR difference [5%, 95%] |
|---|---|---:|---:|
| leave_target_out | affine_minus_loss | +0.0236 [+0.0144, +0.0294] | +0.0117 [+0.0088, +0.0150] |
| leave_target_out | affine_minus_mismatched_lira | +0.0049 [+0.0008, +0.0094] | +0.0043 [+0.0019, +0.0088] |
| leave_cell_out | affine_minus_loss | +0.0255 [+0.0182, +0.0327] | +0.0177 [+0.0142, +0.0193] |
| leave_cell_out | affine_minus_mismatched_lira | +0.0087 [+0.0032, +0.0124] | +0.0080 [+0.0059, +0.0132] |

Intervals are 5th–95th percentiles across the ten retained simulator seeds, not
record/model-clustered confidence intervals. A transfer scheme passes the automatic
gate only when the 5th percentiles of both the AUC and calibrated-TPR contrasts over
mismatched LiRA are positive at 1024 shots.

## Structural heterogeneity at 1024 shots

| Transfer scheme | Cell | Affine minus loss AUC | Affine minus mismatched-LiRA AUC |
|---|---|---:|---:|
| leave_target_out | eff_su2_r1_d2 | +0.0139 | +0.0003 |
| leave_target_out | eff_su2_r5_d2 | -0.0099 | +0.0074 |
| leave_target_out | z_r1_d6 | +0.0412 | +0.0182 |
| leave_target_out | zz_r1_d6 | +0.1006 | -0.0061 |
| leave_target_out | zz_r5_d6 | -0.0374 | -0.0130 |
| leave_cell_out | eff_su2_r1_d2 | +0.0105 | -0.0031 |
| leave_cell_out | eff_su2_r5_d2 | -0.0111 | +0.0063 |
| leave_cell_out | z_r1_d6 | +0.0430 | +0.0201 |
| leave_cell_out | zz_r1_d6 | +0.1085 | +0.0018 |
| leave_cell_out | zz_r5_d6 | -0.0274 | -0.0030 |

## Held-out channel diagnostics

| Transfer scheme | Shots | Median slope | Median held-out R² | Median 90% coverage |
|---|---:|---:|---:|---:|
| leave_cell_out | 1024 | 0.745 | 0.867 | 0.910 |
| leave_cell_out | 128 | 0.744 | 0.519 | 0.887 |
| leave_cell_out | 512 | 0.744 | 0.781 | 0.901 |
| leave_target_out | 1024 | 0.817 | 0.876 | 0.896 |
| leave_target_out | 128 | 0.806 | 0.571 | 0.900 |
| leave_target_out | 512 | 0.809 | 0.806 | 0.900 |

## Interpretation limits

- These are simulator-to-simulator transfers under one IBM-derived frozen noise
  snapshot, not transfers to quantum hardware or a new calibration date.
- Each structural cell has only three target checkpoints. Leave-target-out therefore
  calibrates from two auxiliary checkpoints per rotation.
- Reference IN/OUT distributions still come from the held cell's exact 16-model bank;
  this test transfers the serving channel and thresholds, not the LiRA reference bank.
- The noise-augmented baseline draws from the fitted Gaussian channel. It does not
  represent additional noisy QNN circuit executions.
- The target-cross-fitted learned MIA is omitted here because it trains on labeled
  outputs from the attacked target and therefore violates these transfer protocols.
  A matched shadow-model learned MIA remains an extended-study requirement.
