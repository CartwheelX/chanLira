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

1. Fix low-FPR reference-to-target calibration using held-out targets or conformal/
   empirical null calibration. Do not claim calibrated attack operation until nominal
   and realized FPR align with uncertainty intervals.
2. Add time-varying and drifting channels. A stationary binomial channel has aggregate
   counts as a sufficient statistic, so repeated query splits cannot add information at
   a fixed total shot budget.
3. Estimate channel parameters from a disjoint public calibration set and propagate
   parameter uncertainty.
4. Reproduce simulator-to-hardware transfer on a small set of fixed QNN checkpoints.
5. Add at least one classical stochastic predictor before framing the work as a general
   serving-channel contribution.
6. Use record/model-clustered uncertainty; the current pooled target-record units are
   dependent across architectures and seeds.
