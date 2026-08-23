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

## Superseded full-study launch guidance

The guarded canary and the subsequent three-target/16-reference Phase-6 scale-up
are complete. Do not launch the legacy 400-candidate full-study command from this
audit. The low-FPR population and confirmatory split are now governed by
`reviewer_targets/channel_lira_phase7_protocol.json`; validate them with
`experiments/check_channel_lira_phase7_readiness.py` before any further compute.
