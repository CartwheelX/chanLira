# ChannelLiRA calibration-ablation figures

## Attack AUC versus auxiliary-target count

![AUC versus auxiliary targets](plots/auc_vs_auxiliary_targets.svg)

Most of the strict-transfer AUC is already present with two cross-cell targets;
twelve targets add a small paired AUC improvement.

## Paired AUC gains

![Paired gains](plots/paired_auc_gain_vs_auxiliary_targets.svg)

The ChannelLiRA gain over mismatched LiRA stays positive after subset marginalization
at every tested calibration count.

## Operational TPR

![Calibrated TPR](plots/calibrated_tpr_vs_auxiliary_targets.svg)

Two-target calibration gives higher TPR but also operates at a higher FPR than the
twelve-target condition, so it is not a free accuracy improvement.

## Realized FPR

![Realized FPR](plots/actual_fpr_vs_auxiliary_targets.svg)

Threshold calibration becomes more conservative and approaches the nominal 1% FPR
as more auxiliary targets are included.

## Held-cell channel fit

![Held-cell R squared](plots/heldout_r2_vs_auxiliary_targets.svg)

Median held-cell R² changes only modestly with calibration count; variation across
cells, folds, and target subsets remains wide.

## Two-target subset sensitivity

![Two-target subset sensitivity](plots/two_target_subset_sensitivity.svg)

The 5th–95th percentile AUC-gain range across the 66 two-target configurations is
positive, although the weakest configurations can be negative. Operational TPR is
more sensitive to which auxiliary targets set the threshold.
