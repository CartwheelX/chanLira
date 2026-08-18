# Frozen SaTML Experimental Protocol

## Research claim

The confirmatory claim is:

> Quantum data encoding is not privacy-neutral: feature-map family and
> repetition systematically alter post-encoding geometry and downstream
> membership leakage, and this structural information can support
> privacy-aware circuit selection.

The study tests an empirically supported, overfitting-mediated pathway. It does
not claim that encoder geometry directly reveals membership, that geometry is
independent of generalization, or that the reported associations are causal.

## Evidence layers

1. The completed 36-target MNIST QNN factorial is retained without rerunning or
   retrospectively changing its protocol.
2. A new Credit-default factorial is the large tabular cross-domain replication.
3. A Fashion-MNIST factorial tests transfer to a harder image domain, while a
   targeted WDBC study tests a sensitive biomedical tabular domain.
4. Post-encoder fidelity-kernel measurements test the proposed representation
   mechanism before the trainable circuit.
5. Threshold, learned/posterior, LiRA reference-model, and label-only attacks
   cover distinct adversary access assumptions.
6. Finite-shot, backend-derived noise and fixed query-budget conditions test
   operational robustness.
7. A structural privacy selector is developed on the factorial targets and
   evaluated once on entirely fresh split/initialization blocks.

## Dataset and leakage-safe preprocessing

The second domain is the UCI **Default of Credit Card Clients** dataset: 30,000
records, 23 attributes, binary default outcome, UCI dataset 350, DOI
`10.24432/C55S3H`. The repository fetcher pins OpenML data ID 42477, falls
back to the official UCI archive, and finally to a commit-pinned mirror if the
official providers are unavailable. It writes deterministic gzip and requires
canonical uncompressed-CSV SHA-256
`dfb1570f223efb65c0084027570369bdff6cc291b8238b9adce17ab60da4ca83`.

Each experimental block contains:

- 200 target-training records, which are MIA members;
- 200 validation records, used for training monitoring and selector utility;
- 2,000 target-test records, which are MIA nonmembers.

The partition is stratified and determined only by `split_seed`. Numeric
standardization, categorical one-hot encoding, PCA to six components, and the
final mapping to `[-1,1]` are fitted on the 200 target-training records only.
The fitted preprocessor, split hashes, PCA variance, source checksum, and range
diagnostics are saved beside every target. Evaluation values outside the
training-derived range are explicitly clipped by the fitted range transform.

Fashion-MNIST uses the original train and test partitions and classes 0, 1, 3,
and 8 (T-shirt/top, Trouser, Dress, and Bag), remapped to four labels. Each of
five independent blocks uses 200 balanced training members, 200 balanced
validation records, and 2,000 balanced test nonmembers. The train/validation
split and balanced test subset are determined by the block's `split_seed`.
Normalization uses fixed Fashion-MNIST constants `(0.2860, 0.3530)` and is not
estimated from evaluation data. The four canonical IDX source checksums, split
hashes, and class counts are validated and saved.

WDBC is UCI dataset 17 (DOI `10.24432/C5DW2B`), with 569 records, 30 numeric
features, and a binary diagnosis. Its deterministic snapshot has canonical
CSV SHA-256 `ec5134d1f4db4e0accdbb8705285cc335eabf53785c06d4f0e75126a84c7cefc`.
Each of five blocks partitions all records into 160 training members, 80
validation records, and 329 test nonmembers. Standardization, PCA to six
components, and mapping to `[-1,1]` are fitted on training members only.

## Paired factorial

The confirmatory Credit experiment uses eight independent blocks. A block is
one `(split_seed, init_seed)` pair. All 12 structural configurations in a block
share both seeds:

| Factor | Values |
| --- | --- |
| Feature map | Z, ZZ, EfficientSU2 |
| Encoder repetitions | 1, 5 |
| Variational depth | 2, 6 |

Width, optimizer, learning rate, epochs, batch size, encoder padding,
feature-map entanglement, variational entanglement, and measurement design are
fixed. This yields `8 × 12 = 96` target models.

The Fashion-MNIST replication uses the same 12 configurations over five paired
blocks (`60` targets). WDBC is deliberately targeted because of its smaller
sample size: it fixes variational depth at 2 and tests three feature maps by two
repetition levels over five paired blocks (`30` targets). Consequently, WDBC
supports repetition and feature-map contrasts, not a depth effect.

The primary endpoint is loss-threshold MIA AUC. The primary structural contrast
is repetitions `5 − 1`. Confirmatory secondary contrasts are depth `6 − 2`,
`Z − EfficientSU2`, and `ZZ − EfficientSU2`. `ZZ − Z` is reported as a
secondary feature-map comparison.

For every outcome, contrasts are calculated inside each block while averaging
over the other factorial dimensions. The eight block effects are the
independent observations. Reported uncertainty is a percentile bootstrap over
those block effects. Exact paired sign-flip tests are Holm-adjusted across the
five prespecified contrasts within each outcome/attack family. A block-fixed
additive regression with CR1 standard errors clustered by block is secondary
and descriptive. Both accuracy gap and `test_loss - train_loss` are reported.

For feature-map and repetition contrasts at a fixed variational depth, the
trainable parameter count is held constant. A fail-closed resource check
verifies that repetition changes fixed-encoder gate count while leaving
trainable capacity unchanged. Depth is analyzed as a separate, intentionally
capacity-changing factor.

## Attack endpoints and access models

Separate scalar threshold attacks use loss, confidence/maximum probability,
entropy, margin, and correctness. Threshold calibration is cross-fitted. The
analysis reports AUC, balanced accuracy, membership advantage, TPR at requested
FPRs of 1%, 5%, and 10%, and the actually attained empirical FPR.

TPR@1% FPR is confirmatory for the loss attack. Every target has at least 2,000
nonmembers, giving an empirical FPR resolution of at most `1/2000 = 0.0005`.
Record-level bootstrap confidence intervals accompany both AUC and fixed-FPR
TPR.

Fashion-MNIST retains 1%, 5%, and 10% FPR endpoints. For WDBC, 5% and 10% are
primary; 1% is displayed only as exploratory because 329 nonmembers leave very
few false-positive observations at that operating point.

Online/offline LiRA uses a balanced candidate subset: all target-training
members and an equal-size deterministic subset of nonmembers (400 records for
Credit/Fashion-MNIST and 320 for WDBC). Sixteen references
are trained per structural-configuration × split-block candidate population,
with each candidate included in exactly half. Reference banks are never shared
across different data splits. Credit uses full LiRA coverage. For the added
datasets, learned and label-only attacks cover every target, while LiRA is a
prespecified representative analysis of all six depth-2 structural
configurations in the first three paired blocks (18 targets per dataset). This
is a reference-model robustness attack,
not the source of the 1% FPR claim. The label-only boundary attack uses
predicted labels and held-out validation anchors; its query count and
boundary-score definition are reported.

## Direct geometry

The pure-state fidelity kernel is measured immediately after the fixed encoder
and before the variational circuit. For Z, ZZ, and EfficientSU2 at repetitions
1 and 5, the study reports:

- within-class and between-class similarity;
- their difference;
- centered kernel-label alignment;
- effective rank and kernel spectrum summaries;
- train-test MMD²;
- encoder-operation and state-signature integrity checks.

Geometry is evaluated over the same eight Credit split seeds. Repetition
operation counts and state signatures must differ in the expected direction
before repetition results are accepted.

The pathway analysis connects configuration-level geometry, accuracy/loss
gaps, and loss-MIA AUC using independent block/geometry-seed resampling. Its
block-clustered regressions are explanatory associations only; they are not
presented as causal mediation estimates.

## Encoding-scale robustness

Angle scale is targeted robustness evidence rather than another full
factorial. At depth 2, all three feature maps and both repetition levels are
evaluated at `alpha ∈ {0.5, 1, 2}` over five paired blocks. The `alpha=1`
targets are reused from the main factorial; only `0.5` and `2` are additionally
trained. The preprocessing is unchanged, isolating the intervention
`theta = alpha × f(x)`.

## Fresh privacy-selector evaluation

Development uses only the 96 factorial targets. Three policies are frozen:

1. `utility_only`: highest mean development validation accuracy;
2. `privacy_aware`: lowest mean loss-MIA AUC among configurations whose mean
   validation accuracy is within 0.02 of the utility-only maximum;
3. `utility_regularized`: the utility-only configuration trained with
   prespecified Adam weight decay `0.001`.

Ties are resolved deterministically by privacy, utility, and structural ID as
recorded by the selector script. After the decision JSON is written, all three
policies are trained on five new split/initialization blocks: 15 targets. Their
utility, gap, and leakage differences are evaluated as fresh paired contrasts.
The fresh seeds do not overlap development seeds.

## Noise and query budget

The noise study uses local Aer simulation with a noise model derived from a
named IBM backend calibration. It is not described as hardware execution. Each
run serializes the full Aer noise model, backend properties/configuration,
calibration timestamp, and SHA-256 manifest without credentials.

The fixed total budget is 2,560 shots per record, compared as:

- 1 query × 2,560 shots;
- 5 queries × 512 shots;
- 20 queries × 128 shots.

Counts from independent repeated calls are aggregated before the classical
head. Ideal-shot and noisy-shot conditions use identical circuits, samples,
transpilation controls, total budgets, and ten simulator seeds. Simulator seeds
are repeated measurements of a checkpoint, not independent target models.
Five prespecified representative MNIST checkpoints cover feature-map and
repetition extremes. Distinct backend calibration timestamps remain separate
profiles; two or three profiles are collected when IBM access permits and are
never pooled as independent target-model replication.
Loss-MIA AUC is reported beside sampled train/test accuracy, loss, and accuracy
gap for every query/shot condition.

## Exclusions and interpretation

- A failed IBM noise-model load is never relabeled as ideal noise.
- Incomplete paired blocks are not used in confirmatory contrasts.
- The Credit snapshot checksum, preprocessing provenance, member convention,
  target count, and low-FPR resolution must pass the fail-closed validator.
- Fashion-MNIST and WDBC split provenance, class balance, target counts, and
  dataset-specific FPR claims must pass their fail-closed validator.
- Results from the old NeurIPS folders and new `satml_*` folders are never
  automatically pooled.
- Scaling and noise-query analyses remain targeted robustness checks.
- Real-device execution and broader architectures remain future extensions;
  the cross-domain evidence now spans MNIST, Fashion-MNIST, Credit-default,
  WDBC, and the retained synthetic tasks.
