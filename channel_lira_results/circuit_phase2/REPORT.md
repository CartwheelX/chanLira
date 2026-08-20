# Circuit-level ChannelLiRA feasibility report

## Verdict

**YES — the circuit-level extension supports continuing to the larger study.**

This is a retrospective analysis of retained Aer outputs, not a new hardware run.
The `noisy_shot` condition used the repository's IBM-Kingston-derived frozen noise
model; `ideal_shot` used finite-shot ideal Aer. All 15 targets are QNNs for which the
full main quantum stack was simulated before the trained classical head.

Under `noisy_shot` at 1024 shots, affine ChannelLiRA has AUC 0.6013, the empirical
channel mixture has 0.5958, mismatched LiRA has 0.5966,
loss MIA has 0.5775, and the target-labeled learned logistic upper bound has
0.5645. The affine endpoint changes from 0.5742 at 128 shots
to 0.6013 at 1024 shots; this is an endpoint comparison, not a claim
that every intermediate point is monotone. The effect is structurally heterogeneous:
at 1024 shots, the cell-level median affine AUC exceeds loss MIA in
3/5 cells and mismatched LiRA in
2/5 cells. The pooled result therefore supports
an extended study; it does not establish uniform superiority.

## Exact-output baselines

| Attack | AUC | TPR @ 1% FPR | TPR @ 5% FPR |
|---|---:|---:|---:|
| loss_mia | 0.5923 | 0.0093 | 0.0473 |
| learned_logistic_pv_stats_target_crossfit_upper_bound | 0.5803 | 0.0187 | 0.0847 |
| exact_output_fitted_lira | 0.6146 | 0.0247 | 0.0860 |

## Same-output-channel attack comparison

Intervals are the 5th–95th percentiles across the 10 retained simulator seeds.

| Mode | Shots | Attack | AUC [5%, 95%] | TPR @ 1% FPR | TPR @ 5% FPR |
|---|---:|---|---:|---:|---:|
| ideal_shot | 128 | loss_mia | 0.5797 [0.5690, 0.5852] | 0.0093 | 0.0477 |
| ideal_shot | 128 | learned_logistic_pv_stats_target_crossfit_upper_bound | 0.5645 [0.5560, 0.5823] | 0.0137 | 0.0733 |
| ideal_shot | 128 | latent_lira_mismatched | 0.5790 [0.5665, 0.5904] | 0.0103 | 0.0610 |
| ideal_shot | 128 | deconvolved_lira | 0.5750 [0.5622, 0.5889] | 0.0107 | 0.0603 |
| ideal_shot | 128 | affine_channel_lira | 0.5940 [0.5820, 0.6068] | 0.0310 | 0.1063 |
| ideal_shot | 128 | empirical_channel_lira | 0.5918 [0.5835, 0.6045] | 0.0313 | 0.1023 |
| ideal_shot | 128 | noise_augmented_lira | 0.5968 [0.5862, 0.6082] | 0.0327 | 0.1077 |
| ideal_shot | 512 | loss_mia | 0.5884 [0.5832, 0.5900] | 0.0083 | 0.0483 |
| ideal_shot | 512 | learned_logistic_pv_stats_target_crossfit_upper_bound | 0.5850 [0.5730, 0.6019] | 0.0137 | 0.0830 |
| ideal_shot | 512 | latent_lira_mismatched | 0.6008 [0.5946, 0.6038] | 0.0130 | 0.0773 |
| ideal_shot | 512 | deconvolved_lira | 0.5998 [0.5936, 0.6036] | 0.0130 | 0.0823 |
| ideal_shot | 512 | affine_channel_lira | 0.6121 [0.6052, 0.6142] | 0.0317 | 0.1017 |
| ideal_shot | 512 | empirical_channel_lira | 0.6065 [0.5997, 0.6093] | 0.0307 | 0.1057 |
| ideal_shot | 512 | noise_augmented_lira | 0.6138 [0.6055, 0.6157] | 0.0363 | 0.1067 |
| ideal_shot | 1024 | loss_mia | 0.5902 [0.5867, 0.5929] | 0.0080 | 0.0447 |
| ideal_shot | 1024 | learned_logistic_pv_stats_target_crossfit_upper_bound | 0.5845 [0.5752, 0.5892] | 0.0140 | 0.0790 |
| ideal_shot | 1024 | latent_lira_mismatched | 0.6026 [0.5997, 0.6130] | 0.0170 | 0.0850 |
| ideal_shot | 1024 | deconvolved_lira | 0.6022 [0.5992, 0.6125] | 0.0187 | 0.0867 |
| ideal_shot | 1024 | affine_channel_lira | 0.6120 [0.6083, 0.6185] | 0.0307 | 0.0993 |
| ideal_shot | 1024 | empirical_channel_lira | 0.6022 [0.5987, 0.6124] | 0.0320 | 0.1063 |
| ideal_shot | 1024 | noise_augmented_lira | 0.6124 [0.6080, 0.6178] | 0.0293 | 0.1047 |
| noisy_shot | 128 | loss_mia | 0.5650 [0.5493, 0.5750] | 0.0097 | 0.0470 |
| noisy_shot | 128 | learned_logistic_pv_stats_target_crossfit_upper_bound | 0.5503 [0.5368, 0.5601] | 0.0100 | 0.0640 |
| noisy_shot | 128 | latent_lira_mismatched | 0.5681 [0.5575, 0.5703] | 0.0100 | 0.0590 |
| noisy_shot | 128 | deconvolved_lira | 0.5585 [0.5480, 0.5710] | 0.0093 | 0.0620 |
| noisy_shot | 128 | affine_channel_lira | 0.5742 [0.5635, 0.5896] | 0.0250 | 0.0873 |
| noisy_shot | 128 | empirical_channel_lira | 0.5739 [0.5654, 0.5920] | 0.0250 | 0.0897 |
| noisy_shot | 128 | noise_augmented_lira | 0.5726 [0.5618, 0.5857] | 0.0220 | 0.0863 |
| noisy_shot | 512 | loss_mia | 0.5757 [0.5701, 0.5771] | 0.0073 | 0.0430 |
| noisy_shot | 512 | learned_logistic_pv_stats_target_crossfit_upper_bound | 0.5731 [0.5491, 0.5787] | 0.0113 | 0.0623 |
| noisy_shot | 512 | latent_lira_mismatched | 0.5889 [0.5784, 0.5977] | 0.0143 | 0.0723 |
| noisy_shot | 512 | deconvolved_lira | 0.5820 [0.5685, 0.5943] | 0.0167 | 0.0733 |
| noisy_shot | 512 | affine_channel_lira | 0.5937 [0.5851, 0.6037] | 0.0293 | 0.0950 |
| noisy_shot | 512 | empirical_channel_lira | 0.5904 [0.5836, 0.6021] | 0.0260 | 0.0957 |
| noisy_shot | 512 | noise_augmented_lira | 0.5946 [0.5850, 0.6049] | 0.0310 | 0.0960 |
| noisy_shot | 1024 | loss_mia | 0.5775 [0.5738, 0.5806] | 0.0080 | 0.0403 |
| noisy_shot | 1024 | learned_logistic_pv_stats_target_crossfit_upper_bound | 0.5645 [0.5573, 0.5678] | 0.0107 | 0.0600 |
| noisy_shot | 1024 | latent_lira_mismatched | 0.5966 [0.5884, 0.6038] | 0.0147 | 0.0727 |
| noisy_shot | 1024 | deconvolved_lira | 0.5898 [0.5785, 0.5973] | 0.0203 | 0.0800 |
| noisy_shot | 1024 | affine_channel_lira | 0.6013 [0.5943, 0.6069] | 0.0310 | 0.0997 |
| noisy_shot | 1024 | empirical_channel_lira | 0.5958 [0.5904, 0.6019] | 0.0267 | 0.0997 |
| noisy_shot | 1024 | noise_augmented_lira | 0.5990 [0.5926, 0.6035] | 0.0310 | 0.0930 |

The learned baseline is a five-fold, target-specific logistic classifier over the
same nine `pv+stats` features used by the repository's learned MIA. It has labeled
target-output auxiliary access, so it is an explicitly marked upper-knowledge
baseline, not the same shadow-model threat model as LiRA.

## Paired AUC contrasts on the primary noisy condition

| Shots | Contrast | AUC difference [5%, 95%] |
|---:|---|---:|
| 1024 | affine_minus_deconvolved_lira | +0.0117 [+0.0086, +0.0171] |
| 1024 | affine_minus_empirical_channel | +0.0046 [+0.0010, +0.0070] |
| 1024 | affine_minus_learned_logistic | +0.0365 [+0.0308, +0.0439] |
| 1024 | affine_minus_loss | +0.0236 [+0.0145, +0.0297] |
| 1024 | affine_minus_mismatched_lira | +0.0050 [+0.0009, +0.0096] |
| 1024 | affine_minus_noise_augmented | +0.0020 [-0.0004, +0.0038] |
| 128 | affine_minus_deconvolved_lira | +0.0166 [+0.0092, +0.0216] |
| 128 | affine_minus_empirical_channel | -0.0010 [-0.0035, +0.0020] |
| 128 | affine_minus_learned_logistic | +0.0296 [+0.0036, +0.0419] |
| 128 | affine_minus_loss | +0.0144 [-0.0076, +0.0260] |
| 128 | affine_minus_mismatched_lira | +0.0069 [+0.0020, +0.0209] |
| 128 | affine_minus_noise_augmented | +0.0016 [-0.0017, +0.0074] |

Each difference pairs attacks within the same retained simulator seed. The automatic
continuation verdict requires the 5th percentile of both `affine_minus_loss` and
`affine_minus_mismatched_lira` to be above zero at the largest shot budget.

## Reference-count ablation on the IBM-derived noisy condition

| Shots | References | Attack | AUC |
|---:|---:|---|---:|
| 128 | 4 | latent_lira_mismatched | 0.5284 |
| 128 | 4 | affine_channel_lira | 0.5509 |
| 128 | 4 | empirical_channel_lira | 0.5521 |
| 128 | 4 | noise_augmented_lira | 0.5515 |
| 128 | 8 | latent_lira_mismatched | 0.5667 |
| 128 | 8 | affine_channel_lira | 0.5679 |
| 128 | 8 | empirical_channel_lira | 0.5719 |
| 128 | 8 | noise_augmented_lira | 0.5638 |
| 128 | 16 | latent_lira_mismatched | 0.5681 |
| 128 | 16 | affine_channel_lira | 0.5742 |
| 128 | 16 | empirical_channel_lira | 0.5739 |
| 128 | 16 | noise_augmented_lira | 0.5726 |
| 1024 | 4 | latent_lira_mismatched | 0.5381 |
| 1024 | 4 | affine_channel_lira | 0.5620 |
| 1024 | 4 | empirical_channel_lira | 0.5665 |
| 1024 | 4 | noise_augmented_lira | 0.5601 |
| 1024 | 8 | latent_lira_mismatched | 0.5845 |
| 1024 | 8 | affine_channel_lira | 0.5914 |
| 1024 | 8 | empirical_channel_lira | 0.5937 |
| 1024 | 8 | noise_augmented_lira | 0.5972 |
| 1024 | 16 | latent_lira_mismatched | 0.5966 |
| 1024 | 16 | affine_channel_lira | 0.6013 |
| 1024 | 16 | empirical_channel_lira | 0.5958 |
| 1024 | 16 | noise_augmented_lira | 0.5990 |

The selected 4/8/16-reference subsets are exactly balanced for every evaluated
candidate. Noise-augmented LiRA uses 32 simulated
channel draws per retained reference; analytic and empirical ChannelLiRA do not
retrain or re-evaluate noisy reference models.

## Cross-fitted nominal 1% FPR calibration

| Shots | Attack | Actual FPR [5%, 95%] | TPR |
|---:|---|---:|---:|
| 1024 | affine_channel_lira | 0.0120 [0.0107, 0.0127] | 0.0243 |
| 1024 | empirical_channel_lira | 0.0123 [0.0113, 0.0150] | 0.0267 |
| 1024 | latent_lira_mismatched | 0.0117 [0.0106, 0.0141] | 0.0213 |
| 1024 | loss_mia | 0.0133 [0.0116, 0.0150] | 0.0103 |
| 128 | affine_channel_lira | 0.0110 [0.0089, 0.0138] | 0.0240 |
| 128 | empirical_channel_lira | 0.0127 [0.0100, 0.0144] | 0.0230 |
| 128 | latent_lira_mismatched | 0.0110 [0.0075, 0.0155] | 0.0127 |
| 128 | loss_mia | 0.0113 [0.0090, 0.0144] | 0.0093 |

Channel parameters and thresholds use only public nonmember pairs from other folds.
The attacked sample ID is excluded across every target checkpoint and simulator seed.
The learned baseline is excluded from this calibration table because calibrating its
out-of-fold scores would require a nested labeled-target split.

## Channel fit diagnostics

| Cell | Mode | Shots | Slope | Residual SD | Held-out R² | Held-out 90% coverage |
|---|---|---:|---:|---:|---:|---:|
| eff_su2_r1_d2 | ideal_shot | 1024 | 1.005 | 0.574 | 0.889 | 0.903 |
| eff_su2_r1_d2 | ideal_shot | 128 | 1.006 | 1.580 | 0.502 | 0.900 |
| eff_su2_r1_d2 | ideal_shot | 512 | 1.016 | 0.812 | 0.813 | 0.888 |
| eff_su2_r1_d2 | noisy_shot | 1024 | 0.843 | 0.611 | 0.823 | 0.892 |
| eff_su2_r1_d2 | noisy_shot | 128 | 0.831 | 1.559 | 0.409 | 0.894 |
| eff_su2_r1_d2 | noisy_shot | 512 | 0.844 | 0.819 | 0.733 | 0.896 |
| eff_su2_r5_d2 | ideal_shot | 1024 | 0.998 | 0.608 | 0.939 | 0.895 |
| eff_su2_r5_d2 | ideal_shot | 128 | 1.000 | 1.754 | 0.618 | 0.895 |
| eff_su2_r5_d2 | ideal_shot | 512 | 1.000 | 0.859 | 0.886 | 0.892 |
| eff_su2_r5_d2 | noisy_shot | 1024 | 0.821 | 0.724 | 0.880 | 0.902 |
| eff_su2_r5_d2 | noisy_shot | 128 | 0.823 | 1.804 | 0.512 | 0.900 |
| eff_su2_r5_d2 | noisy_shot | 512 | 0.816 | 0.954 | 0.792 | 0.898 |
| z_r1_d6 | ideal_shot | 1024 | 0.995 | 0.377 | 0.981 | 0.912 |
| z_r1_d6 | ideal_shot | 128 | 0.996 | 1.054 | 0.867 | 0.902 |
| z_r1_d6 | ideal_shot | 512 | 0.999 | 0.540 | 0.964 | 0.909 |
| z_r1_d6 | noisy_shot | 1024 | 0.846 | 0.623 | 0.932 | 0.895 |
| z_r1_d6 | noisy_shot | 128 | 0.840 | 1.153 | 0.803 | 0.902 |
| z_r1_d6 | noisy_shot | 512 | 0.852 | 0.737 | 0.910 | 0.905 |
| zz_r1_d6 | ideal_shot | 1024 | 1.000 | 0.451 | 0.977 | 0.904 |
| zz_r1_d6 | ideal_shot | 128 | 0.993 | 1.240 | 0.848 | 0.900 |
| zz_r1_d6 | ideal_shot | 512 | 1.001 | 0.631 | 0.954 | 0.902 |
| zz_r1_d6 | noisy_shot | 1024 | 0.734 | 0.647 | 0.936 | 0.880 |
| zz_r1_d6 | noisy_shot | 128 | 0.728 | 1.311 | 0.730 | 0.895 |
| zz_r1_d6 | noisy_shot | 512 | 0.737 | 0.775 | 0.896 | 0.897 |
| zz_r5_d6 | ideal_shot | 1024 | 0.994 | 0.471 | 0.972 | 0.885 |
| zz_r5_d6 | ideal_shot | 128 | 0.992 | 1.287 | 0.817 | 0.904 |
| zz_r5_d6 | ideal_shot | 512 | 0.991 | 0.659 | 0.945 | 0.892 |
| zz_r5_d6 | noisy_shot | 1024 | 0.558 | 0.804 | 0.767 | 0.907 |
| zz_r5_d6 | noisy_shot | 128 | 0.559 | 1.477 | 0.524 | 0.895 |
| zz_r5_d6 | noisy_shot | 512 | 0.557 | 0.940 | 0.699 | 0.909 |

Large residual skew/kurtosis values in `channel_diagnostics_raw.csv` indicate where
the affine Gaussian approximation is misspecified. They are a gate for adding a
heteroskedastic or nonparametric channel in the extended study.

## Scope and remaining publication gates

- Evidence covers 5 structural cells and 15 fixed QNN checkpoints; the retained
  CSVs do not include noisy reference-checkpoint evaluations, classical stochastic
  models, time drift, or real hardware executions.
- The channel-calibration threat model assumes paired exact/noisy outputs for disjoint
  public nonmembers. A lower-knowledge estimator must be tested before a broad claim.
- Simulator seeds are repeated measurements of the same records and checkpoints.
  The intervals above are not record/model-clustered publication confidence intervals.
- A publication study should add noisy reference ensembles, heteroskedastic channel
  models, held-out calibration targets, hardware snapshots over time, and at least one
  classical stochastic-serving control.
