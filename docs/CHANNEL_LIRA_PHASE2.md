# Circuit-level ChannelLiRA phase

This phase tests the serving-channel idea on retained QNN circuit simulations. It
uses 15 fixed targets from five structural cells, exact outputs, ideal finite-shot
Aer outputs, and Aer outputs generated with the repository's frozen
IBM-Kingston-derived noise model. It does not retrain a model, rerun a quantum
circuit, or claim execution on IBM hardware.

## Reproduce

Install the isolated dependency:

```bash
python3 -m pip install -r requirements-channel-lira.txt
```

Run the frozen analysis:

```bash
python3 experiments/channel_lira_circuit_pilot.py \
  --cells all \
  --modes ideal_shot,noisy_shot \
  --shots 128,512,1024 \
  --reference-counts 4,8,16 \
  --folds 5 \
  --noise-augmentation-draws 32 \
  --variance-shrinkage 0.15 \
  --seed 20260820 \
  --out-dir channel_lira_results/circuit_phase2
```

Run the numerical tests:

```bash
python3 -W error::DeprecationWarning -m unittest \
  test.test_channel_lira test.test_channel_lira_continuous -v
```

The result is in `channel_lira_results/circuit_phase2/REPORT.md`.

Generate the figure set with:

```bash
python3 experiments/plot_channel_lira_phase2.py --png
```

The visual index is `channel_lira_results/circuit_phase2/PLOTS.md`.

## Threat models and leakage controls

The reference attacks use the retained exact 16-model LiRA banks. For the channel
model, the attacker has paired exact and stochastic outputs for public nonmembers
(`K2` calibration access). Five folds are assigned by sample ID. A record's sample
ID is excluded from its own channel fit and threshold calibration across all target
checkpoints and simulator seeds.

The learned baseline is intentionally separate: a target-specific five-fold
logistic classifier is trained on labeled target outputs using the same nine
probability-vector-plus-statistics features as the repository's learned MIA. It is a
higher-knowledge baseline; every reported record is out of fold for that classifier.
Its scores are not used for the empirical-null calibration table because that would
require a nested labeled-target split.

## Attacks compared on the same stochastic outputs

- `loss_mia`: negative observed cross-entropy loss.
- `learned_logistic_pv_stats_target_crossfit_upper_bound`: learned target-output
  baseline with labeled auxiliary access.
- `latent_lira_mismatched`: ordinary exact-reference LiRA applied directly to noisy
  log odds.
- `deconvolved_lira`: invert the fitted channel mean and ignore residual uncertainty.
- `affine_channel_lira`: analytically convolve Gaussian IN/OUT reference models with
  the fitted affine-Gaussian serving channel.
- `empirical_channel_lira`: convolve the channel with the finite empirical IN/OUT
  reference mixtures.
- `noise_augmented_lira`: draw 32 noisy observations per exact reference and refit
  ordinary Gaussian LiRA.

The exact-output table uses the neutral name `exact_output_fitted_lira`; it is a
fitted reference-model attack, not an oracle.

## Main feasibility result

At 1024 shots under the IBM-derived noisy condition, affine ChannelLiRA reaches AUC
0.6013, versus 0.5775 for loss MIA, 0.5645 for learned logistic MIA, and 0.5966 for
mismatched LiRA. Simulator-seed-paired AUC differences are respectively +0.0236
[+0.0145, +0.0297], +0.0365 [+0.0308, +0.0439], and +0.0050
[+0.0009, +0.0096] over the 5th–95th percentiles.

At 128 shots, the channel-vs-loss paired interval crosses zero, so this phase does
not claim uniform superiority. The 16-reference affine result increases from 0.5742
at 128 shots to 0.6013 at 1024 shots; this is an endpoint observation, not a claim
that every intermediate value must be monotone.

The result is also heterogeneous across structural cells. At 1024 noisy shots,
affine ChannelLiRA's cell-level median AUC exceeds loss MIA in 3/5 cells and
mismatched LiRA in 2/5 cells. The positive pooled comparison is a reason to run the
extended study, not evidence of universal dominance.

Cross-fitted nominal 1% FPR calibration yields median realized FPR 1.10% at 128
shots and 1.20% at 1024 shots for affine ChannelLiRA. These are useful feasibility
checks, not publication confidence intervals, because simulator seeds reuse the same
records and checkpoints.

## Publication gates that remain

1. Evaluate noisy reference checkpoints directly, rather than approximating their
   serving outputs from an exact bank and a fitted target-side channel.
2. Hold out complete target models for channel and FPR calibration, and propagate
   channel-parameter uncertainty.
3. Add heteroskedastic or nonparametric channels where residual diagnostics reject
   the affine-Gaussian approximation.
4. Collect multiple real-hardware calibration snapshots and test temporal drift and
   simulator-to-hardware transfer.
5. Add classical stochastic-serving controls and stronger learned/shadow-model
   baselines.
6. Use record- and model-clustered uncertainty for inferential claims.

## Generated artifacts

- `metrics_raw.csv` and `metrics_summary.csv`: AUC, advantage, and low-FPR metrics.
- `paired_contrasts_raw.csv` and `paired_contrasts_summary.csv`: within-seed attack
  comparisons.
- `calibration_raw.csv` and `calibration_summary.csv`: cross-fitted fixed-threshold
  operation.
- `channel_diagnostics_raw.csv` and `channel_diagnostics_summary.csv`: channel fit,
  residual shape, held-out R², and coverage.
- `experiment_config.json`: threat model, source hashes, selected reference subsets,
  and frozen parameters.
- `REPORT.md`: concise interpretation and gates.
- `PLOTS.md` and `plots/`: visual summary and six reproducible SVG figures.
