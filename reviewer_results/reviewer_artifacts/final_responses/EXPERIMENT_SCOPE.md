# Scope of the submitted sweep and rebuttal experiments

The rebuttal experiments supplement rather than replace the submitted paper's broad sweep. The distinction below should be stated explicitly whenever the new multi-seed results are described.

| Evidence block | Scope | Purpose | Claim boundary |
| --- | --- | --- | --- |
| Submitted broad sweep | Multiple datasets; QNN/HQNN/QCNN; three feature-map families; repetitions, widths, depths, entanglers, gates, and padding | Exploratory breadth and discovery of the reported structural hierarchy | Not rerun in full with the new multi-seed/attack protocol |
| Focused confirmatory factorial | MNIST QNN; 3 feature maps × 2 repetitions × 2 depths × 3 model seeds = 36 targets; fixed data split | Initialization robustness and paired uncertainty for the central encoder/repetition/depth claim | Supports the core claim in this factorial, not every submitted configuration |
| Expanded MIA suite | Every one of the 36 confirmatory targets; thresholds, learned vector, LiRA, and label-only | Remove attack/regime selection within the confirmatory factorial and test multiple access models | Does not apply LiRA/label-only to the entire original sweep |
| Direct geometry | MNIST and Moons; three feature maps; reps 1/5; nominal data seeds 43–45 | Directly measure post-encoder fidelity-kernel geometry | MNIST nominal seeds duplicate states; causal mediation is not identified |
| Finite-shot/noise sanity check | Five representative MNIST-QNN configurations; 15 checkpoints; 128/512/1024 shots; ten simulator seeds | Test whether the broad leakage ordering survives one backend-derived noise model | Aer simulation from one backend snapshot, not hardware/general noise validation |
| Architecture controls | QNN/HQNN/QCNN/MLP-QNN; three structural roles × three model seeds | Complete-wrapper performance, leakage, and resource comparison | Preprocessing and heads are not matched causal ablations |

Recommended description: ‘The original submission provides broad exploratory coverage. The rebuttal adds a focused multi-seed confirmatory factorial and targeted geometry, attack-breadth, architecture, and noise checks for the central claim.’
