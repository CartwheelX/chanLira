# ChannelLiRA Phase 7: frozen confirmatory protocol

Phase 7 is the confirmatory study that follows the completed Phase-6 scale gate.
It is frozen before target/reference training or circuit execution. The executable
protocol is
[`channel_lira_phase7_protocol.json`](../reviewer_targets/channel_lira_phase7_protocol.json),
and the new target population is
[`channel_lira_phase7_targets.csv`](../reviewer_targets/channel_lira_phase7_targets.csv).

This phase does not redefine ChannelLiRA as a universally stronger MIA. It separates
three questions:

1. Does stochastic quantum serving create a damaging mismatch for ordinary LiRA?
2. Is ChannelLiRA non-inferior to an expensive matched noisy-reference LiRA while
   using substantially less noisy auxiliary-model serving?
3. Does ChannelLiRA beat the cheap loss attack at a predefined security-relevant
   operating point?

## Pilot and confirmatory separation

The Phase-6 `eff_su2_r1_d2` result selected the noisy 128-shot TPR@1% FPR
hypothesis. It is therefore pilot evidence, not prior confirmation. Phase 7 keeps
that cell as a separately reported engineering replication and excludes it from
the primary interval.

The confirmatory subset contains twelve newly initialized targets from four cells:

- `eff_su2_r5_d2`;
- `z_r1_d6`;
- `zz_r1_d6`; and
- `zz_r5_d6`.

Each cell has model seeds 143, 144, and 145. All cells use the same newly frozen
data seed 143 so that circuit structure is not confounded with a different MNIST
sample population.

## Low-FPR population gate

The old 200-member/200-nonmember pool is prohibited for the Phase-7 primary
endpoint. Every new target uses 1,000 members and 1,000 nonmembers. This gives:

- ten false positives at 1% FPR;
- 0.1-percentage-point empirical FPR resolution; and
- 0.1-percentage-point empirical TPR resolution.

The corresponding 2,000-candidate reference design retains sixteen models and is
exactly balanced: every candidate is IN in eight references and OUT in eight, and
every reference trains on 1,000 records, matching the target training size.

Additional simulator seeds are repeated serving realizations, not new membership
decisions. Even this enlarged population does not support a stable 0.1%-FPR claim,
which remains explicitly out of scope.

## Frozen endpoints

The primary condition is IBM-derived noisy Aer execution at 128 shots with
simulator seeds 0–9. The primary endpoint is TPR@1% FPR on the four confirmatory
cells after averaging simulator seeds within each target. The primary practical
comparison is ChannelLiRA minus loss MIA.

The mechanism comparison is matched noisy-reference LiRA minus mismatched latent
LiRA. Efficient recovery is ChannelLiRA minus matched noisy-reference LiRA. The
frozen non-inferiority margins are an absolute 0.5 percentage point for primary
TPR@1% FPR and 0.01 for secondary AUC. Inference uses a hierarchical bootstrap over
structural cells and targets within cells; the selected four-cell scope must still
be reported and must not be described as universal architecture generalization.

Secondary endpoints are AUC and TPR@5% FPR. Secondary serving conditions are noisy
512/1024 shots and ideal finite-shot 128/512/1024. Exploratory associations with
train–test gap, channel fit, exact LiRA strength, loss leakage, and cell are labeled
exploratory and cannot replace a failed primary endpoint.

## Retained learned comparator

The learned MIA is intentionally not redesigned. Phase 7 retains
`target_crossfit_learned_mia` with the same frozen probability-feature schema used
in Phase 6. Reports must call it the **privileged victim-crossfit learned
comparator**, because it uses membership-labeled cross-fitted outputs from the
attacked target. It is a stress comparison and does not justify claims against
learned MIAs in general.

## Success conditions and paper decision

- **A — channel mismatch:** the confirmatory interval for matched noisy LiRA minus
  mismatched LiRA is above zero at the primary endpoint.
- **B — efficient recovery:** ChannelLiRA is non-inferior to matched noisy LiRA
  under the frozen margin, while its amortized noisy calibration-model cost is no
  more than 25% of matched-reference serving cost.
- **C — practical attack utility:** the confirmatory interval for ChannelLiRA minus
  loss MIA is above zero for noisy 128-shot TPR@1% FPR.

A+B+C supports a specialized ChannelLiRA attack paper. A+B without C supports a
serving-channel mismatch and privacy-auditing paper. A without B requires a method
revision, while failure of A undermines the central serving-mismatch motivation.

The pilot cell, secondary conditions, and exploratory subgroups cannot be used to
rescue a failed confirmatory primary endpoint.

## Cost accounting

Every attack report must record trained references, noisy reference executions and
shots, auxiliary calibration executions and shots, target-query shots, wall-clock
time, peak GPU/host memory, checkpoint/cache bytes, and cache reuse scope.

At the primary condition for one attacked target:

| Comparator | Noisy auxiliary models | Auxiliary shots |
|---|---:|---:|
| Matched noisy-reference LiRA | 16 references | 40,960,000 |
| ChannelLiRA | 2 calibration targets | 5,120,000 |

Thus the per-attack calibration/reference ratio is 0.125. Across all three attacks
in a cell, outputs are reused and the amortized ratio is 3/16 = 0.1875. ChannelLiRA
has a real calibration cost; it does not have zero quantum-serving cost.

The confirmatory primary reference execution projects to 163,840,000 noisy shots.
The complete frozen primary/secondary/diagnostic matrix projects to approximately
3.794 billion simulated circuit shots, so it must be executed in resumable stages.

## Staging and unblinding

1. Validate the design without computation.
2. Run the pilot-cell engineering replication at ideal/noisy 128 shots and seeds
   0–1.
3. Complete and hash all four-cell primary outputs before unblinding aggregate
   attack comparisons.
4. Run the already frozen secondary and ideal-diagnostic matrix.

Observing an engineering failure may justify repairing code without viewing attack
comparisons. Any scientific method or endpoint change creates a new protocol
version and makes the changed analysis exploratory.

### Stage 1 runner

The guarded runner can only access the three pilot-replication targets. Every
compute stage requires the exact externally locked protocol hash:

```bash
PYTHON_BIN=/home/najeeb/miniconda3/envs/tq39_vv2/bin/python
PHASE7_HASH=$(cut -d ' ' -f 1 reviewer_targets/channel_lira_phase7_protocol.sha256)

"${PYTHON_BIN}" experiments/run_channel_lira_phase7_stage1.py status \
  --python "${PYTHON_BIN}"

"${PYTHON_BIN}" experiments/run_channel_lira_phase7_stage1.py all \
  --python "${PYTHON_BIN}" \
  --acknowledge-protocol-hash "${PHASE7_HASH}"
```

The target, reference, exact-score, noisy-score, and analysis stages are
independently resumable. Stage 1 writes a checkpoint-hash ledger and shot/runtime
receipt under `channel_lira_results/phase7/stage1_pilot`. It does not implement or
authorize the four-cell confirmatory run.

### Completed Stage 1 outcome

Stage 1 completed on 2026-08-22 with three new targets, 1,000 members and 1,000
nonmembers per target, all sixteen balanced reference checkpoints, and all twelve
target serving payloads (ideal/noisy, seeds 0/1). The validator reports no
blockers. The shared noisy-reference oracle was executed once and hash-reused for
the second and third targets. The recorded total is 19,456,000 simulated shots.

At the frozen noisy 128-shot endpoint, the descriptive target means are:

| Attack | AUC | TPR@1% FPR |
|---|---:|---:|
| Matched 16-reference LiRA | 0.4923 | 1.05% |
| ChannelLiRA | 0.4993 | 1.05% |
| Mismatched latent LiRA | 0.4960 | 0.78% |
| Loss MIA | 0.5245 | 1.20% |
| Privileged victim-crossfit learned comparator | 0.5129 | 0.78% |

The pilot primary-metric directions are therefore: A (matched minus mismatched)
`+0.27` percentage points, B (ChannelLiRA minus matched) `+0.00` points, and C
(ChannelLiRA minus loss) `-0.15` points. This is supportive of the mismatch and
efficient-recovery questions at the selected low-FPR metric, but it does not show
practical superiority over loss MIA. These are three-target ranges, not
confidence intervals, and they neither confirm nor alter the frozen four-cell
endpoint.

See the [Stage 1 report](../channel_lira_results/phase7/stage1_pilot/analysis/REPORT.md),
[plot index](../channel_lira_results/phase7/stage1_pilot/analysis/PLOTS.md), and
[cost receipt](../channel_lira_results/phase7/stage1_pilot/COST_RECEIPT.json).

## Readiness command

The audit performs no training and submits no circuit jobs:

```bash
PYTHON_BIN=/home/najeeb/miniconda3/envs/tq39_vv2/bin/python

"${PYTHON_BIN}" experiments/probe_channel_lira_phase7_candidates.py

python3 experiments/check_channel_lira_phase7_readiness.py \
  --require-design-ready
```

It writes
`channel_lira_results/phase7_readiness/READINESS.json` and `READINESS.md`.
The protocol JSON is independently locked by
`reviewer_targets/channel_lira_phase7_protocol.sha256`; silent changes to an
endpoint, margin, population, or baseline make the design audit fail.
The candidate probe materializes the controlled MNIST population and inclusion
matrix without training or circuit execution. Its fingerprint and balance receipt
are required by the design audit.
Design readiness, training readiness, noisy-scoring readiness, and
publication-artifact readiness are reported separately. Missing target/reference
checkpoints correctly block scoring but not the training that creates them. Before artifact submission, checkpoints must be
placed in an immutable archive with recorded content hashes; Git-ignored local
weights alone are insufficient.
