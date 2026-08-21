# ChannelLiRA clustered sensitivity analysis

## Target-average result at 1024 shots

| Transfer scheme | Contrast | Mean target AUC difference | Hierarchical 95% percentile range | Positive targets | Positive cells |
|---|---|---:|---:|---:|---:|
| leave_cell_out | affine_minus_mismatched_lira | +0.0060 | [-0.0060, +0.0183] | 10/15 | 2/5 |
| leave_cell_out | affine_minus_loss | +0.0260 | [-0.0118, +0.0705] | 9/15 | 3/5 |
| leave_target_out | affine_minus_mismatched_lira | +0.0027 | [-0.0131, +0.0168] | 10/15 | 3/5 |
| leave_target_out | affine_minus_loss | +0.0227 | [-0.0164, +0.0648] | 9/15 | 3/5 |

This analysis changes the estimand from the Phase-3 pooled-record AUC to the mean of
15 target-level AUC differences after averaging each target over ten simulator
seeds. It then resamples five structural cells and three targets within each sampled
cell. Consequently, these values should not numerically match the pooled Phase-3
contrasts.

## Interpretation limits

- The hierarchical ranges incorporate observed target/cell heterogeneity but still
  reuse the same records and do not constitute nested record/model confidence
  intervals.
- Five structural cells are too few for a reviewer-proof population-level
  architecture claim. These ranges are a sensitivity analysis, not definitive
  inference.
- Positive-cell counts use the mean of three target-level AUC contrasts per cell,
  whereas Phase 3 reports pooled-record cell AUC. Those are different estimands and
  can have different signs.
- Independently trained reference ensembles and more target checkpoints remain
  necessary for publication-grade uncertainty.
