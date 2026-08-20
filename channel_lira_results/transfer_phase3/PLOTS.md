# ChannelLiRA strict-transfer figures

## Leave-target-out attack AUC

![Leave-target-out AUC](plots/attack_auc_leave_target_out.svg)

The attacked checkpoint is excluded from channel fitting and threshold calibration.

## Leave-cell-out attack AUC

![Leave-cell-out AUC](plots/attack_auc_leave_cell_out.svg)

Every target with the held circuit structure is excluded from channel fitting and
threshold calibration.

## Paired transfer gains

![Paired AUC gains](plots/paired_auc_transfer.svg)

At 1024 shots, the 5th–95th percentile intervals against loss MIA and mismatched
LiRA are above zero for both holdout schemes. Target-holdout comparison with
mismatched LiRA crosses zero at 512 shots, so the evidence is not uniform by budget.

## Operational low-FPR performance

![Calibrated TPR](plots/calibrated_tpr_at_1pct.svg)

ChannelLiRA has higher transferred-threshold TPR than both primary comparators.

## FPR calibration

![Actual FPR](plots/actual_fpr_at_1pct.svg)

Actual FPR remains near 1%, though several 5th–95th percentile intervals exceed the
nominal value. These are descriptive seed intervals, not confidence intervals.

## Structural heterogeneity

![Cell heterogeneity](plots/cell_transfer_heterogeneity.svg)

Pooled superiority is not universal across the five circuit cells.

## Held-out channel diagnostics

![Held-out channel R squared](plots/heldout_channel_r2.svg)

The affine channel approximation transfers better as the shot budget grows, with
wide cell/fold variation retained in the whiskers.
