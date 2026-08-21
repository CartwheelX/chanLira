# True noisy-reference LiRA readiness

## Status: BLOCKED

The existing exact reference banks are validated independently from trained model
weights. True noisy-reference LiRA must execute retained reference QNN checkpoints
through a reconstructable backend noise snapshot; Gaussian channel draws do not
satisfy this requirement.

| Item | Available | Required |
|---|---:|---:|
| Valid exact reference files | 80 | 80 |
| Trained reference checkpoints | 0 | 80 |
| Frozen backend snapshot manifest | 0 | 1 |

## Blockers

- 80/80 trained reference checkpoints are missing
- a reconstructable frozen backend snapshot is missing

The Phase-3 target outputs were generated from an IBM calibration dated
2026-07-30, but their summary metadata is insufficient to reconstruct the full
noise model. For a controlled comparison, capture a new frozen snapshot and rerun
both targets and reference checkpoints under that same snapshot, or recover the
original complete snapshot.

## New reference-ensemble retraining command

```bash
python3 reviewer_tools/run_lira_reference_multigpu.py --targets reviewer_targets/multiseed_factorial_targets.csv --out-dir reviewer_results/lira_reference_mia --cells eff_su2_r1_d2,eff_su2_r5_d2,z_r1_d6,zz_r1_d6,zz_r5_d6 --num-references 16 --save-reference-checkpoints --phase train --resume
```

This command is deliberately scoped to the five Phase-3 cells and requests saved
weights. The runner accepts the retained base cell names even though newly trained
banks use the canonical `*_wd0` layout. The readiness audit prefers a complete
score/checkpoint pair and will detect that layout on its next run. The command does
not start automatically from this readiness audit.
