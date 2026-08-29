# Phase 7 Stage 2 confirmatory primary execution

## Status: COMPLETE

This runner is restricted to the frozen four-cell noisy 128-shot primary. The pilot and all Stage-3 secondary conditions are unreachable.

| Gate | Available | Required |
|---|---:|---:|
| Target checkpoint bundles | 12 | 12 |
| Reference score files | 64 | 64 |
| Reference checkpoints | 64 | 64 |
| Balanced reference banks | 4 | 4 |
| Exact target score payloads | 12 | 12 |
| Noisy target bundles | 12 | 12 |
| Noisy serving payloads | 120 | 120 |
| Noisy reference caches | 4 | 4 |
| Raw-output hash seal | 1 | 1 |
| Confirmatory analysis | 1 | 1 |
| Primary plots | 1 | 1 |

## Budget guard

- Noisy reference serving: 163,840,000 shots.
- Target serving: 30,720,000 shots.
- Combined primary: 194,560,000 shots.
- Secondary and ideal-diagnostic conditions: not implemented by this runner.

## Artifacts

- Status: `/home/najeeb/chanLira/channel_lira_results/phase7/stage2_primary/STATUS.json`
- Cost receipt: `/home/najeeb/chanLira/channel_lira_results/phase7/stage2_primary/COST_RECEIPT.json`
- Raw-output seal: `/home/najeeb/chanLira/channel_lira_results/phase7/stage2_primary/RAW_OUTPUT_SEAL.json`
- Decision report: `/home/najeeb/chanLira/channel_lira_results/phase7/stage2_primary/analysis/REPORT.md`
