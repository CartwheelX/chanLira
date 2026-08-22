# True noisy-reference LiRA canary

This is an execution and reconstruction gate, not statistical evidence.

## Status: COMPLETE

| Gate | Available | Required |
|---|---:|---:|
| Target checkpoint bundle | 1 | 1 |
| Exact reference score files | 4 | 4 |
| Reference checkpoints | 4 | 4 |
| Valid checkpoint metadata | 4 | 4 |
| Balanced reference bank | 1 | 1 |
| Hash-validated noise snapshot | 1 | 1 |
| Exact-only score bundle | 1 | 1 |
| Exact/noisy score bundle | 1 | 1 |

## Frozen protocol

- Target: `MNIST_QNN_eff_su2_r1_d2_s43`.
- Structural cell: `eff_su2_r1_d2` (selected before execution as the compute-minimal Phase-3 cell).
- Reference models: 4, exactly balanced per candidate.
- Shots: 128; simulator seeds: `0,1`.
- Both target and references must use the same hash-validated snapshot.
- Four references are sufficient only to test plumbing; they are not a paper baseline.

## Retired-checkpoint comparison

- Candidate IDs, labels, and memberships align: `True`.
- Exact log-odds bitwise match: `True`.
- Log-odds MAE: 0.000000; correlation: 1.000000.

## Four-reference exact-only plumbing output

| Attack | AUC |
|---|---:|
| lira_online | 0.4692 |
| lira_online_fixed_variance | 0.5230 |
| lira_offline | 0.5163 |
| lira_offline_fixed_variance | 0.5261 |
| global_true_class_log_odds | 0.5392 |

These four-reference values are recorded to validate exact scoring only; they are not inferential results or substitutes for the 16-reference baseline.

## Mean AUC by finite-shot mode

| Mode | Attack | Mean AUC | Simulator seeds |
|---|---|---:|---:|
| ideal_shot | global_true_class_log_odds | 0.5331 | 2 |
| ideal_shot | lira_offline | 0.4830 | 2 |
| ideal_shot | lira_offline_fixed_variance | 0.4875 | 2 |
| ideal_shot | lira_online | 0.4949 | 2 |
| ideal_shot | lira_online_fixed_variance | 0.5085 | 2 |
| noisy_shot | global_true_class_log_odds | 0.5293 | 2 |
| noisy_shot | lira_offline | 0.5016 | 2 |
| noisy_shot | lira_offline_fixed_variance | 0.5047 | 2 |
| noisy_shot | lira_online | 0.5246 | 2 |
| noisy_shot | lira_online_fixed_variance | 0.5251 | 2 |

These values validate execution only. With one target, two simulator seeds, and four references, they must not be used as scientific evidence.

## Artifact roots

- Target runs: `/home/najeeb/chanLira/channel_lira_results/noisy_reference_canary_phase5/runs/multiseed_factorial/MNIST_QNN_eff_su2_r1_d2_s43`
- References: `/home/najeeb/chanLira/channel_lira_results/noisy_reference_canary_phase5/references/reference_models/eff_su2_r1_d2_wd0`
- Snapshot: `/home/najeeb/chanLira/channel_lira_results/noisy_reference_canary_phase5/backend_snapshot`
- Machine-readable status: `/home/najeeb/chanLira/channel_lira_results/noisy_reference_canary_phase5/STATUS.json`
