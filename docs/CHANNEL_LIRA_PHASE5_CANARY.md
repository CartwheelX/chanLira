# ChannelLiRA Phase 5: true noisy-reference canary

This canary validates the execution path needed for true circuit-executed noisy
reference LiRA before committing to the full 15-target/80-reference study. It is a
plumbing and reconstruction gate, not statistical evidence.

## Frozen scope

- One target: `MNIST_QNN_eff_su2_r1_d2_s43`.
- One structural cell: `eff_su2_r1_d2`, selected before execution because it is the
  compute-minimal Phase-3 cell.
- Four reference models, the minimum that gives two IN and two OUT observations per
  candidate.
- 128 shots and two simulator seeds.
- One hash-validated IBM-derived snapshot shared by target and reference execution.
- Ideal-shot and noisy-shot reference LiRA are both evaluated; no hardware circuit
  is submitted by this canary.

Four references and one target cannot support comparative or inferential claims.
Successful completion only establishes that checkpoints, circuit conversion,
snapshot reconstruction, matched noisy execution, and scoring work together.

## Completed local gates

The target model, target attack payload, and all four reference checkpoints have
been reconstructed and retained. The reference metadata validates that all 400
candidates occur in exactly two of the four reference training sets. Checkpoint
IDs, structural-cell metadata, state dictionaries, and candidate fingerprints all
match their exact score files.

The reconstructed target was also scored exactly. Its candidate IDs, labels,
memberships, and all 400 exact target log-odds match the retained Phase-3 target
outputs bit-for-bit. This confirms deterministic target recovery in the current
environment. The four-reference attack AUCs are recorded only as pipeline outputs;
they are not comparable to the 16-reference Phase-3 results.

The live status and reconstruction diagnostics are in
[`REPORT.md`](../channel_lira_results/noisy_reference_canary_phase5/REPORT.md) and
[`STATUS.json`](../channel_lira_results/noisy_reference_canary_phase5/STATUS.json).

## Completed frozen-noise gate

The complete `ibm_kingston` snapshot was captured on 21 August 2026. Its schema-2
manifest hashes the Aer noise model, backend configuration, backend properties, and
metadata; no credentials are recorded. Credential-free reconstruction produced
1,286 Aer noise errors, and a local noisy smoke circuit completed successfully.

The matched target/reference scorer then completed both ideal-shot and noisy-shot
conditions at 128 shots for simulator seeds 0 and 1. All execution gates pass with
no recorded failures. The resulting AUCs remain plumbing outputs because two IN and
two OUT reference observations per candidate cannot stably estimate LiRA densities.
The canary therefore authorizes scaling the protocol; it does not establish an
attack comparison.

The completed report is
[`REPORT.md`](../channel_lira_results/noisy_reference_canary_phase5/REPORT.md), and
the next frozen experiment is the
[three-target/16-reference Phase-6 scale-up](CHANNEL_LIRA_PHASE6_SCALEUP.md).

## Rebuild from scratch

The stages are independently resumable:

```bash
PYTHON_BIN=/home/najeeb/miniconda3/envs/tq39_vv2/bin/python

"${PYTHON_BIN}" experiments/run_channel_lira_noisy_reference_canary.py target \
  --python "${PYTHON_BIN}"

"${PYTHON_BIN}" experiments/run_channel_lira_noisy_reference_canary.py references \
  --python "${PYTHON_BIN}"

"${PYTHON_BIN}" experiments/run_channel_lira_noisy_reference_canary.py exact \
  --python "${PYTHON_BIN}"

"${PYTHON_BIN}" experiments/run_channel_lira_noisy_reference_canary.py snapshot \
  --python "${PYTHON_BIN}"

"${PYTHON_BIN}" experiments/run_channel_lira_noisy_reference_canary.py score \
  --python "${PYTHON_BIN}"

"${PYTHON_BIN}" experiments/run_channel_lira_noisy_reference_canary.py status
```

The five small canary `.pt` checkpoints are explicitly retained as curated,
hash-audited evidence so this reconstruction gate cannot silently lose its weights.
Training logs and future bulk checkpoints remain ignored; a full study should store
large checkpoint banks in durable artifact storage with published hashes.
