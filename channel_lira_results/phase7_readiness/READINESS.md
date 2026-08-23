# ChannelLiRA Phase 7 readiness

## Design status: READY

## Training status: READY

## Noisy-scoring status: BLOCKED

## Publication-artifact status: PENDING

This audit does not train models or execute circuits. Phase 6 is explicitly pilot
evidence. The Phase-7 primary analysis is restricted to four confirmatory cells
and uses noisy 128-shot TPR@1% FPR. The existing learned attack is retained without
redesign and is labeled the privileged victim-crossfit learned comparator.

| Design gate | Frozen value |
|---|---:|
| Pilot replication targets | 3 |
| Confirmatory targets | 12 |
| References per cell | 16 |
| Total reference models | 80 |
| Members per target | 1,000 |
| Nonmembers per target | 1,000 |
| False positives at 1% FPR | 10 |
| FPR/TPR empirical resolution | 0.1 percentage points |
| Stable 0.1% FPR supported | false |
| Materialized candidate probe | 1 |

## Artifact availability

| Item | Available | Required |
|---|---:|---:|
| Target checkpoint bundles | 0 | 15 |
| Reference score files | 0 | 80 |
| Reference checkpoints | 0 | 80 |
| Hash-bound checkpoint metadata | 0 | 80 |
| Complete balanced reference banks | 0 | 5 |
| Frozen snapshot | 1 | 1 |

## Projected serving cost

| Scope | Simulated circuit shots |
|---|---:|
| Confirmatory primary matched-reference execution | 163,840,000 |
| Confirmatory primary target execution | 30,720,000 |
| Full frozen noisy reference matrix | 2,662,400,000 |
| Full frozen ideal reference matrix | 532,480,000 |
| Full frozen target matrix | 599,040,000 |
| Full frozen total | 3,793,920,000 |

At the primary condition for one attacked target, matched noisy LiRA requires
40,960,000 reference shots. ChannelLiRA
requires 5,120,000 auxiliary
calibration shots under the frozen two-model leave-target-out threat model, a ratio
of 0.125. This is not zero-cost calibration.

## Design errors

- none

## Training blockers

- none

## Noisy-scoring blockers

- 15/15 Phase-7 target bundles are missing
- 80/80 Phase-7 reference checkpoints are missing

The absent target/reference artifacts are expected at this stage. The passed design
and snapshot gates permit a later explicitly requested training stage; noisy scoring
remains blocked until those artifacts validate. An immutable checkpoint archive and
hashes are required before artifact submission, but do not block training.
