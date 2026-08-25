# Q0 residual quantum leakage screen

Q0 is a bounded exploratory falsification experiment for the stronger research
question raised after Phase 7 Stage 1:

> Does controlled quantum execution expose membership information beyond loss and
> the unchanged prediction-vector-plus-statistics learned MIA?

It is not an amendment to Phase 7. The frozen Phase-7 protocol and endpoints remain
unchanged, and Q0 results cannot enter the Phase-7 confirmatory analysis.

## Why the design differs from the first suggestion

The three Phase-7 pilot targets contain the same 2,000 candidate identities and the
same membership labels. Merely leaving one target model out would therefore not make
a learned response attack independent. Q0 trains six new targets on six class-balanced,
source-disjoint partitions of the filtered canonical MNIST training corpus. Every
partition contains 1,000 target-training members, 200 validation records, and 1,000
nonmembers. Every attack fold additionally removes any duplicate content identity,
and repeated auxiliary identities are averaged to one training row.

The first Q0 lock (`cf20475...`) was invalidated during preflight, before any target
checkpoint or response acquisition completed: the historical TorchQuantum MNIST path
ignored `data_seed` for record selection and reconstructed six identical pools. Q0 v2
adds the explicit source-disjoint partition mechanism; the attacks, endpoints,
channels, and stopping gates are unchanged. This revision is recorded in the protocol.

A stationary finite-shot query also has aggregate counts as a sufficient statistic.
Q0 consequently does not claim that splitting a fixed shot budget manufactures new
information. It compares equal 1,280-shot budgets:

- fixed channel: ten 128-shot queries on physical layout A;
- active probe: five 128-shot queries on layout A and five on disjoint layout B.

Layout A uses physical qubits 0–5 and layout B uses 8–13 in the frozen
IBM-Kingston-derived snapshot. These are local Aer executions, not hardware jobs.

## Threat models

The probability-only baselines receive final prediction vectors from one fixed layout.
Raw marginal and joint attacks assume that the serving interface returns bitstring
counts. Paired-layout attacks additionally assume the caller may choose a physical
initial layout. Results from the latter threat model must not be presented as attacks
on an ordinary probability-only prediction API.

The learned attack remains the existing `MLPClassifier(64, 32)` with the same fixed
training parameters and attacker seeds 41–43. All learned feature ablations use that
same estimator; there is no attack-specific tuning.

## Locked inputs

- Protocol: `reviewer_targets/channel_lira_q0_protocol.json`
- Protocol SHA-256: `5936a137b84ee6c175f76077c9a61e9227e6882066b855483a6b357d03aef050`
- Targets: `reviewer_targets/channel_lira_q0_targets.csv`
- Frozen snapshot manifest SHA-256:
  `7af4abc6763339615cc42bd00fafb532e3c44a31c14bc574c88456a496136a2f`

Any change to a target, channel, feature, estimator, endpoint, threshold, or screening
gate requires a new protocol version.

## Run

Use the QuRiFT environment and explicitly acknowledge the protocol hash for every
compute stage:

```bash
Q0_HASH=5936a137b84ee6c175f76077c9a61e9227e6882066b855483a6b357d03aef050
PYTHON=/home/najeeb/miniconda3/envs/tq39_vv2/bin/python3.9

$PYTHON experiments/run_channel_lira_q0.py status

$PYTHON experiments/run_channel_lira_q0.py probe \
  --python "$PYTHON"

$PYTHON experiments/run_channel_lira_q0.py target \
  --python "$PYTHON" \
  --gpus 0 \
  --target-jobs-per-gpu 2 \
  --acknowledge-protocol-hash "$Q0_HASH"

$PYTHON experiments/run_channel_lira_q0.py acquire \
  --python "$PYTHON" \
  --device cuda \
  --acknowledge-protocol-hash "$Q0_HASH"

$PYTHON experiments/run_channel_lira_q0.py analyze \
  --python "$PYTHON" \
  --acknowledge-protocol-hash "$Q0_HASH"
```

The `all` stage is resumable and performs the same sequence. Analysis is blocked until
all six raw payloads and their hashes validate.

## Cost and outputs

Q0 trains six targets and zero reference models. Acquisition uses 23,040,000 local
simulated shots in total. Each fixed-versus-paired attack comparison uses the same
1,280 shots per candidate.

Outputs are written below `channel_lira_results/q0_residual_quantum_leakage/`:

- `raw/`: compressed raw bitstring counts and response arrays;
- `metadata/`: checkpoint, snapshot, layout, and payload hashes;
- `analysis/metrics_target.csv`: victim-level metrics;
- `analysis/contrasts_summary.csv`: paired comparisons;
- `analysis/SCREENING_DECISION.json`: locked go/stop decision;
- `analysis/REPORT.md`: concise result and interpretation report;
- `analysis/plots/`: AUC, low-FPR, conditional, and target-difference plots.

## Decision boundary

Q0 continues only if the paired joint probe clears every locked condition: practical
gain over loss, gain over the unchanged learned MIA, gain within loss strata, evidence
for a raw-joint or active-layout mechanism, separation from the classical stochastic
control, and acceptable transferred-threshold FPR. Passing justifies a larger
preregistered study; it does not itself establish a breakthrough. Failure invokes the
stop rule for this stronger quantum-stochastic/channel-probing MIA under the tested
access model.
