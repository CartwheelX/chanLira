# QuRiFT reference-model and label-only MIA baselines

This extension adds the two attack families that were still missing from the
reviewer response:

1. calibrated reference-model LiRA; and
2. a genuine class-label-only, query-based decision-boundary proxy.

The implementations are adapted to QuRiFT's saved QNN targets and fixed
MNIST candidate pool. They do not require installing or importing any of the
reference repositories.

## Reference implementations studied

The protocol was checked against these repository revisions:

- `orientino/lira-pytorch` at
  `50dc2a3fc5e66628d48bf07e05c8c33f9703c789` (Apache-2.0);
- `antibloch/mia_attacks` at
  `9c09fe9d9be982e203df303fb450375f2333987b` (MIT);
- `Pierre-Joly/Membership-Inference-Attacks` at
  `9182ed809d9fa3d5141d50816b7e83a06590371b` (MIT); and
- `zhenglisec/Label-Only-MIA` at
  `af3e8146279d595389aecf8eb6e47245129d6021`.

The studied `zhenglisec/Label-Only-MIA` revision does not contain a license
file. No code was copied from it. The QuRiFT label-only implementation is
independent and records that fact in its output provenance.

## LiRA protocol

For each of the 12 QNN structural cells:

- construct the canonical pool of 200 target-training and 200 target-test
  candidates;
- train 16 same-architecture reference QNNs;
- include every candidate in exactly 8 reference models;
- train every reference model on exactly 200 candidates, matching target
  training-set size and optimization settings;
- score each candidate with true-class log-odds; and
- report online/offline LiRA with per-record and fixed-variance Gaussian fits.

One reference bank is reused for the three independently initialized target
models in the same structural cell. The pool combines the canonical target
train and target test candidates, so the paper must disclose that the
reference-training distribution is an approximation rather than claiming
perfect target/reference distribution identity.

## Label-only protocol

The attacker receives only predicted class labels. For each correctly
classified candidate it:

- finds nearby held-out validation anchors assigned a different predicted
  class;
- performs label-only binary search on the segment between the attacked
  candidate and the validation anchor; and
- uses the minimum observed input-space boundary distance as membership score.

Initially misclassified candidates receive distance zero, as in conventional
label-only robustness attacks. This is a query-based decision-boundary proxy,
not a gradient-refined HopSkipJump/QEBA attack and not a certified global
minimum distance.

## Full run

Run both baselines:

```bash
bash commands/run_missing_mia_baselines.sh
```

Or run LiRA alone:

```bash
python reviewer_tools/run_lira_reference_multigpu.py \
  --targets reviewer_targets/multiseed_factorial_targets.csv \
  --repo-root . \
  --run-root reviewer_runs \
  --out-dir reviewer_results/lira_reference_mia \
  --num-references 16 \
  --bootstrap 5000 \
  --seed 2026 \
  --gpus 0,1,2,3,4,5,6,7 \
  --jobs-per-gpu 1 \
  --cpu-threads 2 \
  --phase all \
  --resume
```

Run the label-only attack alone:

```bash
python reviewer_tools/run_label_only_boundary_multigpu.py \
  --targets reviewer_targets/multiseed_factorial_targets.csv \
  --repo-root . \
  --run-root reviewer_runs \
  --out-dir reviewer_results/label_only_boundary \
  --anchors 16 \
  --binary-steps 10 \
  --norm l2 \
  --query-batch-size 64 \
  --bootstrap 5000 \
  --seed 2026 \
  --gpus 0,1,2,3,4,5,6,7 \
  --jobs-per-gpu 1 \
  --cpu-threads 2 \
  --resume
```

## Monitoring

Both launchers print one line when each job completes. Detailed progress and
tracebacks are written under:

- `reviewer_results/lira_reference_mia/logs/`;
- `reviewer_results/label_only_boundary/logs/`;
- `reviewer_results/lira_reference_mia/reference_training_status.csv`;
- `reviewer_results/lira_reference_mia/target_scoring_status.csv`; and
- `reviewer_results/label_only_boundary/target_scoring_status.csv`.

When the command is also piped through `tee`, the console remains live while a
combined log is saved.

## Principal outputs

LiRA:

- `lira_reference_mia_raw.csv`;
- `lira_reference_mia_summary.csv`;
- per-target scores under `target_scores/`; and
- per-sample scores under `sample_scores/`.

Label-only:

- `label_only_boundary_raw.csv`;
- `label_only_boundary_summary.csv`;
- per-target scores under `target_scores/`; and
- per-sample query/distance records under `sample_scores/`.

## Reviewer-safe wording

LiRA supports:

> We added calibrated online and offline LiRA using 16 same-architecture
> reference models per structural cell, with every candidate included in
> exactly half of the reference models.

The label-only experiment supports:

> We added a class-label-only query attack that estimates decision-boundary
> distance using changed-label anchor searches. The attack consumes no
> probability vector, logit, loss, gradient, or model parameter.

Do not describe the chord-boundary attack as HopSkipJump, QEBA, or a certified
minimum distance, and do not describe 16 references as exhaustive LiRA.
