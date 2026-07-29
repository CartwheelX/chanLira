# QuRiFT reviewer-results runbook

## Priority order

1. Apply `REQUIRED_FIXES.md`.
2. Audit the existing master/extensive CSVs.
3. Select exact/near gap-matched Z–ZZ pairs.
4. Rerun only the compact reviewer target tables across three seeds.
5. Run scalar threshold MIAs and the existing learned MLP attack.
6. Run direct encoder-geometry analysis.
7. Add a small finite-shot/readout-noise sanity study only after the above.

## Copy the existing results

Place these files under `experiments/gen_results/`:

```text
master_results_full_pipeline_moon.csv
master_results_full_pipeline_blobs.csv
master_results_full_pipeline_circles.csv
mnist_extensive_results.csv
hqnn_extensive_results.csv
qcnn_extensive_results.csv
```

## A. Audit current results

```bash
python reviewer_tools/audit_existing_sweeps.py \
  --data-dir experiments/gen_results \
  --out-dir reviewer_audit
```

Inspect:

```text
reviewer_audit/dataset_inventory.csv
reviewer_audit/eff_su2_repetition_integrity.csv
reviewer_audit/paired_structural_effects.csv
reviewer_audit/matched_gap_pairs.csv
```

Do not use Efficient-SU2 repetition results until the required fix is applied and
the affected configurations are rerun.

## B. Build compact reviewer target tables

```bash
python reviewer_tools/build_reviewer_targets.py \
  --data-dir experiments/gen_results \
  --audit-dir reviewer_audit \
  --out-dir reviewer_targets \
  --seeds 43,44,45
```

Generated tables:

- `matched_gap_mia_targets.csv`: direct test of different encoders at similar gap.
- `multiseed_factorial_targets.csv`: 12 MNIST QNN cells × 3 seeds.
- `architecture_control_targets.csv`: common protocol across wrappers.
- `geometry_targets.csv`: encoder-only Hilbert-space analysis.

## C. Dry-run all target commands

```bash
python reviewer_tools/run_reviewer_subset.py \
  --targets reviewer_targets/matched_gap_mia_targets.csv \
  --out reviewer_runs \
  --gpus 0 \
  --dry-run
```

## D. Run matched-gap MIA targets

```bash
python reviewer_tools/run_reviewer_subset.py \
  --targets reviewer_targets/matched_gap_mia_targets.csv \
  --out reviewer_runs \
  --gpus 0,1 \
  --jobs-per-gpu 1 \
  --resume
```

## E. Run the multi-seed validation subset

```bash
python reviewer_tools/run_reviewer_subset.py \
  --targets reviewer_targets/multiseed_factorial_targets.csv \
  --out reviewer_runs \
  --gpus 0,1 \
  --jobs-per-gpu 1 \
  --resume
```

## F. Run scalar threshold MIAs

```bash
python reviewer_tools/threshold_mia_bootstrap.py \
  --attack-data-dir reviewer_runs \
  --out reviewer_results/threshold_mia_results.csv \
  --bootstrap 3000
```

This gives loss, entropy, confidence, margin, correctness, and max-probability
attacks without training shadow/reference models.

## G. Run the existing learned attack

```bash
python experiments/gen_results/train_mia_attack.py \
  --attack-data-dir reviewer_runs \
  --out reviewer_results/learned_mia \
  --test-ratio 0.2 \
  --cv-folds 5 \
  --tune \
  --n-trials 30 \
  --max-epochs 200 \
  --patience 15 \
  --device cuda \
  --seed 42
```

For uncertainty from the attack learner, repeat with seeds 42, 43, and 44 into
separate output directories. Do not substitute attack-seed repetitions for
target-model seed repetitions.

## H. Run direct encoder geometry

```bash
python reviewer_tools/encoder_geometry_audit.py \
  --targets reviewer_targets/geometry_targets.csv \
  --repo-root . \
  --out reviewer_results/encoder_geometry.csv \
  --n-train 100 \
  --n-test 100 \
  --device cuda
```

## I. Architecture controls

Run only after standardizing protocols and clarifying that the comparison is
between complete wrappers:

```bash
python reviewer_tools/run_reviewer_subset.py \
  --targets reviewer_targets/architecture_control_targets.csv \
  --out reviewer_runs \
  --gpus 0,1 \
  --jobs-per-gpu 1 \
  --resume
```

## What can be obtained without any target retraining

From the existing CSVs alone:

- exact paired change in gap from repetitions 1→5;
- exact paired change in gap from minimum→maximum variational depth;
- bootstrap confidence intervals over matched structural cells;
- candidate Z–ZZ pairs with equal/near-equal train accuracy, test accuracy, and gap;
- integrity checks and accurate compute/configuration accounting.

## What requires limited reruns

- MIA AUC for matched-gap pairs;
- target-seed error bars;
- corrected Efficient-SU2 repetition results;
- standardized architecture controls.

## What should not be attempted as a quick rebuttal experiment

- rerunning the full 116,640-configuration grid across five seeds;
- full hardware execution;
- a comprehensive LiRA/reference-model benchmark;
- a large full-factorial noise sweep.

Those are suitable for a full resubmission, not the shortest path to reviewer-facing evidence.
