# ChannelLiRA strict-transfer phase

This phase tests whether the Phase-2 serving-channel correction transfers to a QNN
checkpoint or circuit structure that was not used to estimate the channel or the
fixed-FPR threshold. It reuses the frozen circuit outputs from Phase 2; it does not
retrain a QNN, execute a new circuit, or claim quantum-hardware evaluation.

## Holdout protocols

`leave_target_out` rotates over all 15 target checkpoints. For each attacked target,
the affine channel and operational thresholds are fitted using only the other two
targets in the same structural cell.

`leave_cell_out` rotates over all five structural cells. For each attacked cell,
channel fitting and threshold calibration use the 12 targets in the other four
cells. No checkpoint sharing the held circuit structure is available to either
operation.

Both protocols use five deterministic sample-ID folds. The attacked candidate's ID
is removed from all auxiliary channel-fitting and threshold-calibration records.
The exact output of a held target is used only after scoring to measure channel-fit
R² and coverage; it is never used in an attack score or threshold.

The exact 16-model LiRA reference bank for the held structural cell remains
available. Consequently, this study establishes transfer of the serving-channel
model and operating threshold, not transfer of the LiRA reference distributions.

The Phase-2 target-cross-fitted learned MIA is not included in the transfer table:
it trains on labeled outputs from the target being attacked, which violates both
holdout protocols. A learned MIA trained on matched auxiliary shadow models remains
part of the extended-study plan.

## Reproduce

```bash
python3 -m pip install -r requirements-channel-lira.txt

python3 experiments/channel_lira_transfer.py \
  --cells all \
  --schemes leave_target_out,leave_cell_out \
  --modes noisy_shot \
  --shots 128,512,1024 \
  --reference-counts 16 \
  --folds 5 \
  --noise-augmentation-draws 32 \
  --variance-shrinkage 0.15 \
  --seed 20260820 \
  --out-dir channel_lira_results/transfer_phase3

python3 experiments/plot_channel_lira_transfer.py --png
```

The numerical report is
`channel_lira_results/transfer_phase3/REPORT.md`; the figure index is
`channel_lira_results/transfer_phase3/PLOTS.md`.

## Result

At 1024 IBM-derived noisy shots, leave-target-out affine ChannelLiRA reaches pooled
AUC 0.6013, versus 0.5966 for mismatched LiRA and 0.5775 for loss MIA. The paired
seed differences are +0.0049 [+0.0008, +0.0094] and +0.0236
[+0.0144, +0.0294], respectively.

The stricter leave-cell-out result reaches AUC 0.6029, versus the same 0.5966 and
0.5775 comparators. Its paired differences are +0.0087 [+0.0032, +0.0124] and
+0.0255 [+0.0182, +0.0327]. The brackets are descriptive 5th–95th percentiles
over ten reused simulator seeds, not confidence intervals.

At the transferred nominal 1% FPR threshold, leave-target-out ChannelLiRA has
median actual FPR 1.23% and TPR 2.50%; leave-cell-out has actual FPR 1.03% and TPR
2.67%. The corresponding TPR gains over mismatched LiRA are +0.43 percentage
points [+0.19, +0.88] and +0.80 points [+0.59, +1.32].

This clears the predefined pooled transfer gate at 1024 shots. It does not establish
universal dominance: the target-holdout AUC interval over mismatched LiRA crosses
zero at 512 shots, and cell-level 1024-shot gains over mismatched LiRA are positive
in only 3/5 cells for each transfer protocol.

## What remains before a publication claim

Phase 4 now addresses calibration-set size/source and adds a target/cell clustered
sensitivity analysis. See [the Phase-4 protocol and results](CHANNEL_LIRA_PHASE4.md).
The clustered ranges cross zero, so the following publication gates remain:

1. Retain or retrain noisy reference checkpoints and evaluate their serving outputs
   directly instead of channel-augmenting an exact reference bank. The readiness
   audit currently finds 0/80 checkpoints and no reconstructable noise snapshot.
2. Repeat across multiple calibration dates and real quantum-hardware snapshots to
   test temporal drift and simulator-to-hardware transfer.
3. Add heteroskedastic and nonparametric channel models, especially for the circuit
   cells where affine-Gaussian transfer is weak.
4. Add classical stochastic-serving controls and stronger shadow-model learned-MIA
   baselines under matched auxiliary knowledge.
5. Propagate channel-parameter uncertainty and collect enough independent targets,
   cells, reference ensembles, and records for nested inferential intervals. The
   current hierarchical resampling is a sensitivity analysis over only five cells.

## Generated artifacts

- `metrics_raw.csv` and `metrics_summary.csv`: target, cell, and pooled ROC metrics.
- `calibration_raw.csv` and `calibration_summary.csv`: transferred fixed-threshold
  FPR and TPR.
- `paired_auc_contrasts_*`: within-seed ROC comparisons.
- `calibrated_contrasts_*`: within-seed operational FPR/TPR comparisons.
- `channel_diagnostics_*`: held-target/cell affine fit and residual diagnostics.
- `experiment_config.json`: frozen protocol and the source Phase-2 config hash.
- `REPORT.md`, `PLOTS.md`, and `plots/`: concise numerical and visual summaries.
