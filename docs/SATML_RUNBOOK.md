# SaTML Runbook

Run every command from the repository root in the `tq39_vv2` environment.

## 1. Install and prepare

```bash
pip install -r requirements-satml.txt
bash commands/satml_prepare.sh
```

If the dataset provider temporarily resets the connection, rerun the command.
The fetcher reuses `data/openml_cache/uci_credit_default_350.zip` if present.
The preparation stage pins both tabular dataset checksums; creates the Credit,
Fashion-MNIST, and WDBC manifests; and validates every paired design.

## 2. Confirmatory Credit factorial

```bash
export QURIFT_GPUS=0,1,2,3,4,5,6,7
export QURIFT_JOBS_PER_GPU=1
bash commands/satml_run_credit_factorial.sh
```

Monitor it from another terminal:

```bash
watch -n 15 python satml_tools/progress.py \
  --targets satml_targets/credit_factorial_targets.csv \
  --run-root satml_runs
```

Inspect an individual live/partial target with:

```bash
tail -f satml_runs/satml_credit_factorial/CREDIT_QNN_z_r1_d2_b01/train.log
```

The launcher is resumable. Re-running the same command with `--resume` skips
complete model-plus-attack exports.

## 3. Primary metrics, threshold attacks, and paired inference

```bash
bash commands/satml_analyze_credit_factorial.sh
```

This stage extracts utility/generalization metrics, runs all scalar threshold
signals with 10,000 record bootstraps, calculates TPR at 1%, 5%, and 10% FPR,
verifies that repetition changes encoder gate count without changing trainable
parameter count, runs paired block inference, and performs fail-closed protocol
validation.

## 4. Direct Credit geometry

```bash
bash commands/satml_run_credit_geometry.sh
```

Do not interpret repetition geometry unless
`satml_results/credit_geometry/repetition_integrity.csv` passes.

After the geometry and threshold results both exist, quantify the proposed
pathway without treating it as proof of causal mediation:

```bash
bash commands/satml_analyze_mechanism.sh
```

The script resamples independent target blocks and geometry seeds, reports
configuration-level associations, and fits secondary block-clustered
explanatory regressions with accuracy and loss gaps.

## 5. Fashion-MNIST and WDBC replications

Run the 60-target Fashion-MNIST factorial and 30-target fixed-depth WDBC study:

```bash
bash commands/satml_run_fashion_factorial.sh
bash commands/satml_run_wdbc_targeted.sh
```

Monitor either target table:

```bash
watch -n 15 python satml_tools/progress.py \
  --targets satml_targets/fashion_factorial_targets.csv --run-root satml_runs

watch -n 15 python satml_tools/progress.py \
  --targets satml_targets/wdbc_targeted_targets.csv --run-root satml_runs
```

Then extract metrics, run threshold attacks, calculate paired contrasts, audit
resources, and validate provenance:

```bash
bash commands/satml_analyze_fashion.sh
bash commands/satml_analyze_wdbc.sh
```

Fashion-MNIST treats TPR at 1%, 5%, and 10% FPR as planned endpoints. WDBC
treats 5% and 10% as primary; its 1% output is exploratory because only 329
nonmembers are available.

## 6. Added-domain geometry and pathway analysis

```bash
bash commands/satml_run_added_geometry.sh
bash commands/satml_analyze_added_mechanisms.sh
```

Check each `repetition_integrity.csv` before interpreting geometry. These
analyses are dataset-specific and are not automatically pooled.

## 7. LiRA and label-only robustness attacks

```bash
bash commands/satml_run_credit_attacks.sh
```

This is intentionally separate because it is substantially more expensive. It
trains the cross-validated learned prediction-vector/statistics attacker, LiRA
and the label-only boundary attack. LiRA trains 16 references for each of the
96 structural-configuration × split-block candidate populations (1,536
references total), then scores its matching target. Banks cannot be reused
across blocks because their candidate records differ. The final command
regenerates paired structural contrasts across all attack families.

After the Fashion-MNIST and WDBC threshold analyses finish, run their full
learned and label-only attacks plus the prespecified representative LiRA subset:

```bash
bash commands/satml_run_added_attacks.sh
```

The added-domain LiRA subset covers all six depth-2 configurations in the first
three independent blocks (18 targets and 288 reference models per dataset).
Its paired uncertainty therefore uses three blocks and is robustness evidence,
not the primary endpoint.

```bash
watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/fashion_factorial/lira_representative/reference_training_status.csv \
  --expected 288

watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/wdbc_targeted/lira_representative/reference_training_status.csv \
  --expected 288

watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/fashion_factorial/label_only/target_scoring_status.csv \
  --expected 60

watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/wdbc_targeted/label_only/target_scoring_status.csv \
  --expected 30
```

Monitor each long launcher from another terminal without hiding its output:

```bash
watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/credit_factorial/lira/reference_training_status.csv \
  --expected 1536

watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/credit_factorial/lira/target_scoring_status.csv \
  --expected 96

watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/credit_factorial/label_only/target_scoring_status.csv \
  --expected 96
```

## 8. Targeted encoding-scale experiment

```bash
bash commands/satml_run_encoding_scale.sh
```

The `alpha=1` baseline is reused from the confirmatory factorial. The script
trains only `alpha=0.5` and `alpha=2` targets and calculates within-block scale
contrasts.

## 9. Freeze and evaluate the privacy selector

Only after the development factorial and loss-threshold outputs are complete:

```bash
bash commands/satml_build_selector.sh
git diff -- satml_targets/selector
```

The command writes the policy decision and five fresh blocks. Commit or archive
the decision files before training the fresh targets. Then run:

```bash
bash commands/satml_run_fresh_selector.sh
```

Do not regenerate the selector decision after inspecting fresh results.

## 10. IBM calibration snapshot and fixed query budget

The retained MNIST checkpoints are ignored by Git. Import only the five
prespecified noise-study targets from the completed NeurIPS workspace, without
retraining or pooling its result tables:

```bash
export QURIFT_LEGACY_REPO='/absolute/path/to/quarift_neurips_rebutal_2'
bash commands/satml_import_legacy_mnist.sh
```

The importer copies files byte-for-byte and records SHA-256 values under
`satml_results/imported_mnist_manifest.json`. It refuses to overwrite a
different destination artifact.

Credentials remain environment-only:

```bash
export QISKIT_IBM_TOKEN='YOUR_NEW_TOKEN'
export QISKIT_IBM_INSTANCE='YOUR_INSTANCE_CRN'
export QURIFT_NOISE_BACKEND='ibm_kingston'
bash commands/satml_noise_budget.sh
```

Never commit or print the token. Rotate any token that has appeared in chat,
terminal history, logs, or screenshots. Each noise run saves a timestamped,
credential-free reconstruction snapshot under `satml_results/backend_snapshots`.

By default, the command evaluates the five prespecified representative
checkpoints in `reviewer_targets/noisy_sanity_targets_core.csv` under one
calibration profile. To override that set:

```bash
QURIFT_NOISE_TARGET_IDS='MNIST_QNN_eff_su2_r1_d2_s43,MNIST_QNN_zz_r5_d6_s43' \
  bash commands/satml_noise_budget.sh
```

Repeat on two or three calibration dates/backends when access permits. The
combiner keys every result by calibration timestamp and never pools distinct
profiles as simulator replicates.

## 11. Generate submission artifacts

After all available result families finish, generate submission tables and
figures directly from the analysis CSVs:

```bash
bash commands/satml_generate_artifacts.sh
```

The artifact manifest lists every loaded and missing result family. Missing
analyses are omitted rather than rendered as zero-valued results. Markdown and
LaTeX tables are written below `satml_results/paper_artifacts/tables`, with PNG
and PDF figures below `satml_results/paper_artifacts/figures`.

## 12. Full verification

```bash
PYTHONPATH=.:reviewer_tools python -m unittest \
  test.test_satml_data \
  test.test_satml_capacity \
  test.test_satml_targets \
  test.test_satml_paired_analysis \
  test.test_satml_selector \
  test.test_satml_noise_budget \
  test.test_satml_lira_candidates \
  test.test_satml_import \
  test.test_satml_mechanism \
  test.test_satml_progress \
  test.test_satml_artifacts \
  test.test_satml_added_datasets \
  test.test_satml_end_to_end -v

python -m py_compile satml_tools/*.py reviewer_tools/*.py experiments/qurift_main.py
git diff --check
```

The end-to-end test constructs a temporary Credit-like snapshot, fits the
training-only preprocessing, trains a QNN for one epoch, exports attack data,
and verifies the model, preprocessing, provenance, membership counts, and
feature-angle scale.
