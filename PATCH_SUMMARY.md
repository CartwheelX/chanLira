# Checkpoint 3 patch summary

This checkpoint adds the non-noise reviewer experiments and analysis pipeline. It does **not** add finite-shot or backend-noise simulation; that remains Checkpoint 4.

## Required `experiments/qurift_main.py` patch

Run `reviewer_tools/apply_qurift_main_reviewer_patch.py` in dry-run mode first. The patch targets the current QuRiFT driver whose GitHub blob SHA is `e63df98e687335d8573b2c923b8bcb8b38ea6c0a`. It uses guarded source patterns and validates the patched source with Python's compiler before writing it.

The patch makes the following changes:

1. **Independent deterministic seeds**
   - adds `--model-seed` and `--data-seed`;
   - retains `--seed` as a legacy model-seed alias;
   - constructs/subsamples the dataset under `data_seed`;
   - resets RNG state before model initialization and training using `model_seed`;
   - supplies a seeded `torch.Generator` to the shuffled training loader.

2. **Efficient-SU2 repetition correction**
   - restores `fm_eff_reps: int = 1` as a real `QFCConfig` field;
   - stores `args.fm_eff_reps` in the mapper configuration;
   - passes `reps=cfg.fm_eff_reps` to every QNN/HQNN/QCNN Efficient-SU2 operation-list constructor;
   - the geometry launcher produces `repetition_integrity.csv`, which must pass before repetition results are used.

3. **Controlled architecture protocol**
   - adds `--learning-rate`;
   - preserves the old architecture-specific defaults when this option is omitted;
   - the supplied architecture-control table sets a common Adam learning rate of `0.01` for QNN, HQNN, QCNN, and MLP-QNN.

4. **Concurrency safety**
   - fixed-name operation-list debug exports require explicit opt-in with `QURIFT_ENABLE_DEBUG_EXPORTS=1`;
   - circuit-image exports are disabled when `QURIFT_DISABLE_CIRCUIT_EXPORTS=1`;
   - when enabled, circuit filenames include target ID, model seed, and data seed;
   - every reviewer target writes to a unique directory.

5. **Complete attack payloads**
   - exports train, validation, and test accuracy/loss;
   - records model/data seed, structural settings, learning rate, and the explicit membership convention `0=member`;
   - exports exact trainable-parameter counts and fixed-encoder/downstream-VQC gate counts;
   - writes a small JSON export summary beside the `.pt` payload.

## Target-table changes

The supplied CSVs now include:

- `model_seed`;
- `data_seed`;
- `structural_cell_id`;
- `learning_rate`.

The factorial and matched QNN studies use `0.05`, consistent with their existing QNN protocol. The controlled architecture comparison uses `0.01` for all wrappers.

## Analysis additions

- retrained train/validation/test metric extraction;
- matched-gap verification with explicit primary and sensitivity tolerances;
- missing/failed seed inventory;
- scalar threshold MIAs with record-bootstrap AUC CIs;
- cross-fitted balanced accuracy and membership advantage;
- empirically resolvable TPR at 5% and 10% FPR;
- correctness-only label-output baseline;
- multi-seed encoder geometry with data-seed SD and hierarchical repetition-effect CIs;
- exact/reconstructed parameter and gate accounting;
- complete-wrapper architecture analysis with paired hierarchical bootstrap;
- descriptive cluster-bootstrap AUC regression.

## Statistical scope safeguards

- Record-bootstrap confidence intervals are not labeled as target-seed uncertainty.
- Target-seed summaries use individual values and mean ± sample SD.
- Matched-pair and structural effects use paired/cluster or hierarchical bootstrap.
- TPR@0.1% and TPR@1% are not used as primary metrics with only 50–100 non-members.
- The correctness-only attack is not described as a boundary-distance/query-augmentation label-only attack.
- Architecture effects are described as complete-wrapper comparisons, not pure quantum-architecture causal effects.
- Regression coefficients are descriptive associations, not causal estimates.
- QCNN patch-wise frontend gate execution is excluded from the main-stack gate count and marked explicitly.
