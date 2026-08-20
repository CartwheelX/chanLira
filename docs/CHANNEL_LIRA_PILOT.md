# ChannelLiRA pilot protocol

This repository now contains a minimal feasibility study for membership inference
through stochastic serving channels. The pilot reuses the retained balanced LiRA
reference banks and exact QNN target scores. It does not retrain models, and it does
not use the copied, untracked `qurift_satML/` tree.

## Reproduce

Install the one additional pilot dependency in an isolated environment:

```bash
python3 -m pip install -r requirements-channel-lira.txt
```

Run the frozen all-cell intervention:

```bash
python3 experiments/channel_lira_pilot.py \
  --cells all \
  --shots 32,64,128,256,512,1024 \
  --target-readout-error 0.12 \
  --reference-readout-error 0.0 \
  --replicates 30 \
  --calibration-draws 20 \
  --quadrature-order 32 \
  --seed 20260820 \
  --out-dir channel_lira_results/pilot
```

Run the focused tests:

```bash
python3 -m unittest test.test_channel_lira -v
```

The result report is at `channel_lira_results/pilot/REPORT.md`.

The circuit-level follow-up has also been completed. Its frozen protocol and command
are in [CHANNEL_LIRA_PHASE2.md](CHANNEL_LIRA_PHASE2.md), and its results are in
`channel_lira_results/circuit_phase2/REPORT.md`.

## Threat model represented by this pilot

For each candidate, exact reference-model true-class log odds define Gaussian latent
IN and OUT distributions. The target exposes a finite number of binary observations
after an asymmetric-capable readout confusion channel. ChannelLiRA integrates the
binomial likelihood over each latent distribution. The matched attack knows the target
channel; the mismatched attack incorrectly assumes its ideal reference channel.

The symmetric synthetic readout intervention is deliberately simpler than an IBM
noise model. In particular, its Bernoulli observations are drawn from the final
true-class probability; they are not quantum measurement shots propagated through the
QNN's classical head. It tests the factorization and transfer mechanism before
spending compute or hardware quota.

## Gates for the next study

1. The second stage adds sample-ID-cross-fitted empirical-null calibration. A
   publication study must still move calibration to entirely held-out target models.
2. Add time-varying and drifting channels. A stationary binomial channel has aggregate
   counts as a sufficient statistic, so repeated query splits cannot add information at
   a fixed total shot budget.
3. The second stage estimates channel parameters from disjoint public calibration
   records. The extended study must propagate estimator uncertainty and reduce the
   assumed paired exact/noisy access.
4. The second stage uses IBM-derived noisy Aer outputs. Real-hardware transfer remains
   untested.
5. Add at least one classical stochastic predictor before framing the work as a general
   serving-channel contribution.
6. Use record/model-clustered uncertainty; the current pooled target-record units are
   dependent across architectures and seeds.
