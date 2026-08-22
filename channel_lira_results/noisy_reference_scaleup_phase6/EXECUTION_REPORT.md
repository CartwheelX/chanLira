# Phase-6 noisy-reference scale-up execution

## Status: COMPLETE

| Gate | Available | Required |
|---|---:|---:|
| Target checkpoint bundles | 3 | 3 |
| Exact reference score files | 16 | 16 |
| Reference checkpoints | 16 | 16 |
| Valid reference metadata | 16 | 16 |
| Balanced 16-reference bank | 1 | 1 |
| Shared frozen noise snapshot | 1 | 1 |
| Exact target score payloads | 3 | 3 |
| Noisy target score bundles | 3 | 3 |
| Comparison analysis | 1 | 1 |
| Plot bundle | 1 | 1 |

## Protocol

- One prespecified structural cell, with target seeds 43, 44, and 45.
- Sixteen balanced references retained as exact-score/checkpoint pairs.
- Ideal-shot and IBM-derived noisy-shot execution at 128 shots and simulator seeds 0,1.
- Strict leave-target-out ChannelLiRA comparison; no hardware circuit submission.
- This is a scale gate before the full 15-target/80-reference study.

## Artifacts

- Machine status: `/home/najeeb/chanLira/channel_lira_results/noisy_reference_scaleup_phase6/STATUS.json`
- Scientific analysis: `/home/najeeb/chanLira/channel_lira_results/noisy_reference_scaleup_phase6/analysis/REPORT.md`
- Plot index: `/home/najeeb/chanLira/channel_lira_results/noisy_reference_scaleup_phase6/analysis/PLOTS.md`
