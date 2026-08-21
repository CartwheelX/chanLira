# ChannelLiRA Phase 4: calibration and inference stress tests

Phase 4 tests two possible overinterpretations of the strict-transfer result:
whether leave-cell-out improved because it used cross-architecture calibration or
simply because it used more auxiliary targets, and whether the pooled-record result
generalizes across target checkpoints and structural cells.

It does not execute new quantum circuits. It reuses the frozen Phase-3 target
outputs and exact, architecture-matched LiRA reference banks. All percentile ranges
below are descriptive sensitivity ranges, not confidence intervals.

## Calibration source/count ablation

The ablation compares five calibration strategies while keeping the attacked
records, five sample-ID folds, 16-reference LiRA bank, and nominal 1% FPR rule
fixed:

- `same_cell_2`: the two other targets in the held target's structural cell.
- `cross_cell_2`: two targets from the other four cells, exhausting all 66 pairs
  for every held cell.
- `cross_cell_4` and `cross_cell_8`: 32 deterministic unique subsets per held cell.
- `cross_cell_12`: all twelve targets in the other four cells.

At 1024 noisy shots, affine ChannelLiRA AUC is 0.6013 with `cross_cell_2` and
0.6029 with `cross_cell_12`. The paired twelve-minus-two difference is +0.0016
[+0.0009, +0.0018]. The larger calibration set brings realized FPR closer to the
nominal value (1.03% rather than 1.33%), but lowers TPR (2.67% rather than 2.95%).

Holding the count at two, `cross_cell_2` minus `same_cell_2` gives a paired AUC
difference of +0.0018 [-0.0007, +0.0032]. Its TPR is higher, but its realized FPR
is also 0.13 percentage points higher. Therefore this experiment does not establish
that cross-architecture calibration itself regularizes or improves the attack.

At empirical 0.1% FPR, the twelve-target affine attack has TPR 0.0033, versus
0.0047 for mismatched LiRA and 0.0020 for loss MIA. With only 1,500 pooled
nonmembers, one false positive changes FPR by 0.0667 percentage points. This is too
coarse for a stable ultra-low-FPR claim and does not show ChannelLiRA beating
mismatched LiRA there. A larger attack population remains essential.

The complete numerical report is
[`channel_lira_results/calibration_ablation_phase4/REPORT.md`](../channel_lira_results/calibration_ablation_phase4/REPORT.md),
with figures indexed in
[`PLOTS.md`](../channel_lira_results/calibration_ablation_phase4/PLOTS.md).

## Clustered sensitivity

The clustered analysis changes the estimand from a pooled-record AUC to the mean of
15 target-level AUC differences. It first averages each target over ten simulator
seeds, then resamples structural cells and targets within cells.

At 1024 shots, leave-cell-out affine ChannelLiRA minus mismatched LiRA is +0.0060,
with a hierarchical 95% percentile range of [-0.0060, +0.0183]. The affine-minus-
loss result is +0.0260 [-0.0118, +0.0705]. Both ranges cross zero. With only five
cells and reused records, these are heterogeneity stress tests—not population-level
confidence intervals. More independently trained targets, cells, and reference
ensembles are required for a publication-grade generalization claim.

See the full
[`clustered sensitivity report`](../channel_lira_results/clustered_sensitivity_phase4/REPORT.md)
and its [`plot index`](../channel_lira_results/clustered_sensitivity_phase4/PLOTS.md).

## True noisy-reference readiness

The audit validates all 80 exact reference-output files for the five Phase-3 cells,
but finds 0/80 retained reference checkpoints and no reconstructable backend
snapshot manifest. Summary metadata naming the IBM calibration date is not enough
to rebuild the noise model. Gaussian channel draws are explicitly not accepted as
true circuit-executed noisy references.

Consequently, true noisy-reference LiRA is blocked until either the original full
snapshot is recovered or a new snapshot is captured and both targets and retained
reference checkpoints are evaluated under it. The scoped retraining command and
machine-readable checks are in the
[`readiness report`](../channel_lira_results/noisy_reference_phase5/READINESS.md).
The retraining runner accepts both the retained base cell names and the current
canonical `*_wd0` names; the audit recognizes either layout but only counts a
paired score file and checkpoint as checkpoint-ready.

## Reproduce

```bash
python3 experiments/channel_lira_calibration_ablation.py \
  --cells all \
  --shots 128,512,1024 \
  --cross-cell-counts 2,4,8,12 \
  --subset-replicates 32 \
  --reference-count 16 \
  --folds 5 \
  --seed 20260820 \
  --out-dir channel_lira_results/calibration_ablation_phase4

python3 experiments/plot_channel_lira_calibration_ablation.py --png

python3 experiments/channel_lira_clustered_sensitivity.py \
  --bootstrap 20000 \
  --seed 20260820

python3 experiments/plot_channel_lira_clustered_sensitivity.py --png

python3 experiments/check_channel_lira_noisy_reference_readiness.py
```

Use the environment from `requirements-channel-lira.txt`. Passing `--require-ready`
to the readiness audit makes missing checkpoints or snapshot metadata a nonzero-exit
condition suitable for an automated pipeline.
