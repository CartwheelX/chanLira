# True noisy-reference LiRA readiness

## Status: BLOCKED

The existing exact reference banks are validated independently from trained model
weights. True noisy-reference LiRA must execute retained reference QNN checkpoints
through a reconstructable backend noise snapshot; Gaussian channel draws do not
satisfy this requirement.

| Item | Available | Required |
|---|---:|---:|
| Target checkpoints | 0 | 15 |
| Target attack payloads | 0 | 15 |
| Valid exact reference files | 80 | 80 |
| Trained reference checkpoints | 0 | 80 |
| Frozen backend snapshot manifest | 1 | 1 |

## Blockers

- 15/15 target checkpoints are missing
- 15/15 target attack payloads are missing
- 80/80 trained reference checkpoints are missing

The completed Phase-5 canary supplies a hash-validated, credential-free
`ibm_kingston` snapshot. The remaining full-study work is to reconstruct and retain
all target/reference checkpoints, then execute both sides under that same frozen
snapshot. The older Phase-3 served outputs remain a separate July-30 calibration
block and are not silently mixed with the new snapshot.

## New reference-ensemble retraining command

```bash
python3 reviewer_tools/run_lira_reference_multigpu.py --targets reviewer_targets/multiseed_factorial_targets.csv --out-dir reviewer_results/lira_reference_mia --cells eff_su2_r1_d2,eff_su2_r5_d2,z_r1_d6,zz_r1_d6,zz_r5_d6 --num-references 16 --save-reference-checkpoints --phase train --resume
```

This command is deliberately scoped to the five Phase-3 cells and requests saved
weights. The runner accepts the retained base cell names even though newly trained
banks use the canonical `*_wd0` layout. The readiness audit prefers a complete
score/checkpoint pair and will detect that layout on its next run. The command does
not start automatically from this readiness audit.

## Staged scale-up first

The guarded one-cell/four-reference canary is complete. Before any 80-reference
launch, run `experiments/run_channel_lira_noisy_reference_scaleup.py` to test three
target checkpoints against a complete 16-reference bank. Its leave-target-out
analysis is the final compute and comparison gate before the five-cell study.
