# ChannelLiRA Phase 6: 16-reference noisy scale-up

Phase 6 is the controlled bridge between the completed four-reference execution
canary and the full five-cell study. It asks whether the complete target/reference
pipeline remains tractable and produces internally coherent comparisons when LiRA
has eight IN and eight OUT observations per candidate.

It is deliberately frozen before execution. Results must not be used to change the
target cell, checkpoint seeds, reference count, shot count, or simulator seeds.

## Frozen scope

- Structural cell: `eff_su2_r1_d2`, selected before the Phase-5 canary as the
  compute-minimal retained Phase-3 cell.
- Targets: independently initialized model seeds 43, 44, and 45.
- References: 16 newly trained models, exactly balanced so every candidate is IN in
  eight and OUT in eight references. Every exact score file retains its paired
  checkpoint.
- Serving modes: ideal finite-shot Aer and the frozen IBM-derived noisy Aer model.
- Shot count: 128; simulator seeds: 0 and 1.
- Snapshot: the completed Phase-5 schema-2 `ibm_kingston` snapshot. Phase 6 does not
  query IBM or update the calibration.
- Channel transfer: leave-target-out. The attacked checkpoint is excluded from
  channel fitting, and sample-ID folds exclude the attacked record from calibration.
- Baselines: matched finite-shot/noisy reference LiRA, mismatched latent LiRA, loss
  MIA, and target-cross-fitted learned MIA.

The learned baseline trains on labeled outputs from the attacked target and has
stronger auxiliary access than LiRA. It is included as a stress comparator and is
not described as the same shadow-model threat model.

## Reference-oracle reuse

All three targets share the same candidate pool, reference checkpoints, backend
snapshot, modes, shots, and simulator seeds. The noisy reference executions are
therefore identical. The scorer stores them in a hash-keyed cache whose protocol
includes the candidate fingerprint, snapshot-manifest hash, reference count,
transpiler settings, modes, shots, and simulator seeds. Later targets reuse the
cache only after protocol and file-hash validation. Target circuits are always
executed independently.

This reuse changes compute cost, not the attack information: every target is still
scored against the same 16 served reference distributions.

## Runbook

Use the project environment; IBM credentials are not required because the snapshot
is local and hash validated.

```bash
PYTHON_BIN=/home/najeeb/miniconda3/envs/tq39_vv2/bin/python

"${PYTHON_BIN}" experiments/run_channel_lira_noisy_reference_scaleup.py status \
  --python "${PYTHON_BIN}"

"${PYTHON_BIN}" experiments/run_channel_lira_noisy_reference_scaleup.py target \
  --python "${PYTHON_BIN}"

"${PYTHON_BIN}" experiments/run_channel_lira_noisy_reference_scaleup.py references \
  --python "${PYTHON_BIN}"

"${PYTHON_BIN}" experiments/run_channel_lira_noisy_reference_scaleup.py exact \
  --python "${PYTHON_BIN}"

"${PYTHON_BIN}" experiments/run_channel_lira_noisy_reference_scaleup.py score \
  --python "${PYTHON_BIN}"

"${PYTHON_BIN}" experiments/run_channel_lira_noisy_reference_scaleup.py analyze \
  --python "${PYTHON_BIN}"

"${PYTHON_BIN}" experiments/run_channel_lira_noisy_reference_scaleup.py plot \
  --python "${PYTHON_BIN}"
```

Every compute stage is resumable. `all` runs every missing stage in order, while
`status` performs no training or circuit execution. The machine-readable status is
written to `channel_lira_results/noisy_reference_scaleup_phase6/STATUS.json`. The
runner defaults to GPUs 0 and 1; `--target-gpus` and `--reference-gpus` can override
that selection when the machine allocation changes. Aer is capped at 128 parallel
threads by default; use `--aer-max-parallel-threads` to lower the cap on a shared
host. The thread cap is part of noisy-score and reference-cache provenance.

## Outputs and go/no-go rule

The analysis reports target-level AUCs after averaging simulator seeds, descriptive
SD/ranges across the three independent target checkpoints, and paired target-level
contrasts. It generates:

- attack AUCs by ideal/noisy mode;
- per-target noisy-shot heterogeneity;
- matched-reference/ChannelLiRA/baseline paired contrasts; and
- noisy-minus-ideal AUC changes.

Three checkpoints from one cell do not support population-level confidence
intervals, cross-cell generalization, or stable 0.1% FPR claims. The scale gate is a
go when all three target bundles, all sixteen checkpoint/score pairs, the shared
snapshot, matched noisy outputs, comparison tables, and plots pass their validators.
Scientific superiority is not required for the pipeline go decision and must not be
declared from this phase alone.

After a clean execution, the next confirmatory phase expands to the five frozen
cells, 15 targets, and 80 references at 128/512/1024 shots and simulator seeds 0–9.
Publication-level low-FPR analysis additionally requires a much larger candidate
population and more independent target structures/checkpoints.
