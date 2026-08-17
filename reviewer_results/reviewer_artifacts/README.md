# Reviewer-response artifact index

All paths are relative to this directory. Numerical tables are provided as CSV and LaTeX; figures are provided as PNG and PDF.

## Concern-to-evidence map

### Epmi: Encoder effect versus ordinary overfitting; gap is not MIA

- Evidence: T01, T02, T07; F01, F02
- Supported wording: Report paired structural effects and gap–AUC association; interpret residual coefficients descriptively and avoid a direct causal claim.
- Required caveat: Fixed primary data split; observational mediation evidence is not causal identification.

### Epmi / 1myw: Narrow attack suite and unclear driving output signal

- Evidence: T03, T08, T10; F03, F07
- Supported wording: Separate loss, entropy, confidence, margin, correctness and max-probability attacks; a learned prediction-vector attacker; calibrated online/offline LiRA; and a class-label-only boundary proxy.
- Required caveat: LiRA uses 16 references per structural configuration and an approximate reference-training distribution; label-only uses a chord-boundary proxy rather than a certified minimum boundary distance.

### Epmi: Hilbert-space mechanism is indirect

- Evidence: T04; F04
- Supported wording: Report kernel alignment, class-similarity gap, train/test MMD² and effective rank immediately after the fixed encoder.
- Required caveat: MNIST nominal geometry seeds duplicate the same states; only Moons supplies genuine data-seed variability.

### 1myw / Area Chair: Noiseless simulation and finite-shot robustness

- Evidence: T05; F05
- Supported wording: Compare exact, ideal finite-shot and IBM-backend-derived noisy finite-shot results with paired simulator-seed uncertainty.
- Required caveat: Backend-derived Aer noise model, not execution on quantum hardware; five prespecified structural configurations.

### nVBH: No multi-seed uncertainty and post-hoc selected regimes

- Evidence: T01, T02, T03, T09; F01–F03
- Supported wording: Report all 36 prespecified factorial targets over three target seeds with no outcome filtering and paired/hierarchical intervals.
- Required caveat: One fixed split in the primary factorial and three, not five, learned-attacker seeds.

### Epmi / nVBH: Confounded architecture comparisons and missing capacity accounting

- Evidence: T06; F06
- Supported wording: Report complete-wrapper comparisons paired within role/seed together with quantum/classical parameter and gate counts.
- Required caveat: Wrappers retain different preprocessing and heads; results are not pure causal architecture effects.

### 1myw / nVBH: Simple datasets and limited external validity

- Evidence: Scope statement in reviewer index
- Supported wording: Bound claims to controlled MNIST/Moons simulation and describe sensitive-domain transfer as untested.
- Required caveat: No healthcare, finance, or public-sector dataset was added.

## Generated artifacts

- **T01 factorial cells**: [csv](tables/T01_factorial_cells.csv), [tex](tables/T01_factorial_cells.tex)
- **T02 factorial paired effects**: [csv](tables/T02_factorial_paired_effects.csv), [tex](tables/T02_factorial_paired_effects.tex)
- **T03 attack suite**: [csv](tables/T03_attack_suite.csv), [tex](tables/T03_attack_suite.tex)
- **T04a geometry cells**: [csv](tables/T04a_geometry_cells.csv), [tex](tables/T04a_geometry_cells.tex)
- **T04b geometry effects**: [csv](tables/T04b_geometry_repetition_effects.csv), [tex](tables/T04b_geometry_repetition_effects.tex)
- **T04c geometry seed audit**: [csv](tables/T04c_geometry_seed_audit.csv), [tex](tables/T04c_geometry_seed_audit.tex)
- **T05a noisy conditions**: [csv](tables/T05a_noisy_conditions.csv), [tex](tables/T05a_noisy_conditions.tex)
- **T05b noisy changes**: [csv](tables/T05b_noisy_changes.csv), [tex](tables/T05b_noisy_changes.tex)
- **T06a architecture wrappers**: [csv](tables/T06a_architecture_wrappers.csv), [tex](tables/T06a_architecture_wrappers.tex)
- **T06b architecture effects**: [csv](tables/T06b_architecture_paired_effects.csv), [tex](tables/T06b_architecture_paired_effects.tex)
- **T07a regression coefficients**: [csv](tables/T07a_gap_auc_regression_coefficients.csv), [tex](tables/T07a_gap_auc_regression_coefficients.tex)
- **T07b regression fit**: [csv](tables/T07b_gap_auc_regression_fit.csv), [tex](tables/T07b_gap_auc_regression_fit.tex)
- **T07c gap correlations**: [csv](tables/T07c_gap_auc_correlations.csv), [tex](tables/T07c_gap_auc_correlations.tex)
- **T08 learned MIA robustness**: [csv](tables/T08_learned_mia_attacker_seed_robustness.csv), [tex](tables/T08_learned_mia_attacker_seed_robustness.tex)
- **T09a reviewer evidence map**: [csv](tables/T09a_reviewer_evidence_map.csv), [tex](tables/T09a_reviewer_evidence_map.tex)
- **T09b experiment completeness**: [csv](tables/T09b_experiment_completeness.csv), [tex](tables/T09b_experiment_completeness.tex)
- **F01 factorial interactions**: [png](figures/F01_factorial_interactions.png), [pdf](figures/F01_factorial_interactions.pdf)
- **F02 gap versus AUC**: [png](figures/F02_gap_vs_loss_auc.png), [pdf](figures/F02_gap_vs_loss_auc.pdf)
- **F03 attack suite**: [png](figures/F03_attack_suite.png), [pdf](figures/F03_attack_suite.pdf)
- **F04 encoder geometry**: [png](figures/F04_encoder_geometry.png), [pdf](figures/F04_encoder_geometry.pdf)
- **F05 noisy finite shots**: [png](figures/F05_noisy_finite_shots.png), [pdf](figures/F05_noisy_finite_shots.pdf)
- **F06 architecture wrappers**: [png](figures/F06_architecture_wrappers.png), [pdf](figures/F06_architecture_wrappers.pdf)
- **F07 learned versus threshold**: [png](figures/F07_learned_vs_threshold.png), [pdf](figures/F07_learned_vs_threshold.pdf)

## Statistical interpretation

- Factorial and architecture error bars summarize independent target-model seeds within structural cells.
- Factorial main-effect intervals resample structural comparison blocks with target seeds nested.
- Noisy-condition intervals resample paired simulator seeds within a fixed target checkpoint; they are not target-training uncertainty.
- Learned-MIA attacker-seed SD describes attack-training sensitivity on fixed target outputs.
- Record-bootstrap AUC intervals are not used as substitutes for target-model seed uncertainty.
- The analyses are descriptive and controlled, but they do not establish that encoder choice causes leakage independently of overfitting.
