# Response to the Area Chair

We thank the Area Chair for consolidating the discussion. In response, we completed: (i) a focused 36-target multi-seed MNIST-QNN factorial; (ii) threshold, learned-vector, 16-reference online/offline LiRA, and class-label-only attacks; (iii) direct post-encoder fidelity-kernel geometry; (iv) QNN/HQNN/QCNN/classical-MLP wrapper controls with resource accounting; and (v) finite-shot and ibm_kingston-derived Aer-noise evaluation. Full tables and protocols are provided in the detailed responses linked in our reviewer comments.

**Statistical robustness, selection, and attack breadth.**

The confirmatory factorial covers 3 feature maps × 2 repetitions × 2 depths × 3 target initializations. Every target receives the same attacks; the earlier post-hoc baseline/stress/hard labels are not used for confirmation. For loss-MIA, reps=5−1 is +0.069 ± 0.029 AUC (95% CI [0.049,0.088]), depth=6−2 is +0.042 ± 0.026 ([0.024,0.060]), Z−EffSU2 is +0.057 ± 0.032 ([0.027,0.080]), and ZZ−EffSU2 is +0.052 ± 0.027 ([0.030,0.074]). The learned attacker is repeated over three training seeds and LiRA uses 16 references/configuration. The data split remains fixed, so these results establish initialization robustness rather than multi-split generalization.

Across attacks with different access assumptions, the same aggregate feature-map and repetition directions recur. Fixed-variance online LiRA has the highest mean performance (AUC 0.609 ± 0.063; TPR@10% FPR 0.192 ± 0.069), followed by loss-threshold (0.596 ± 0.052; 0.135 ± 0.039); label-only obtains 0.582 ± 0.052 AUC using predicted labels alone. Repetition has positive pooled LiRA and label-only effects, and Z/ZZ exceed EffSU2 under both. Attack choice changes magnitude—LiRA is also depth-modulated—but the principal encoder associations are not specific to one attacker.

**Overfitting, proxy validity, and direct geometry.**

Across all 36 prespecified targets, gap and loss-AUC have Spearman ρ=0.931 ([0.710,0.974]); after conditioning descriptively on gap, residual structural coefficient intervals cross zero. We therefore do not claim gap-independent causation. Directly after the fixed encoder, reps=5−1 changes within-minus-between-class fidelity by −0.124 ± 0.061 ([−0.158,−0.079]), kernel–label alignment by −0.208 ± 0.132 ([−0.285,−0.109]), and effective rank by +49.747 ± 37.448 ([15.791,83.870]); the train/test MMD² interval includes zero. Together these results support an empirically measured pathway—encoder design → post-encoder geometry → downstream generalization asymmetry → membership signal—rather than a geometry-only or causally identified mechanism.

**Finite shots and backend-derived noise.**

We evaluated five representative configurations, 15 independently trained checkpoints, three shot counts, and ten simulator seeds under exact inference, ideal finite shots, and an ibm_kingston-derived Aer model (915 target/execution replicates). The exact high-minus-low loss-AUC difference 0.179 ± 0.030 is attenuated to 0.096 ± 0.012, 0.124 ± 0.026, and 0.132 ± 0.036 at 128, 512, and 1,024 noisy shots. Aggregate ordering remains, while nearby configurations can reorder in individual runs. This is a backend-derived robustness check, not hardware execution or evidence of device universality.

**Architecture, attribution, and scope.**

Relative to paired QNN roles, QCNN improves accuracy by +0.190 ± 0.052 ([0.148,0.249]) and reduces gap by −0.041 ± 0.042 ([−0.083,−0.007]) and loss-AUC by −0.019 ± 0.025 ([−0.040,−0.001]); MLP and HQNN gap/AUC intervals are unresolved. These are complete-wrapper controls: preprocessing and heads remain unmatched. We will define Fig. 8 percentages precisely as dataset-normalized factor-association shares and supplement them with paired MIA intervals, rather than interpret them as causal allocations.

The submission's broad sweep supplies exploratory coverage; the rebuttal confirms feature-map family, repetition, and depth in the focused factorial, not every original width/gate/entangler/padding configuration. Dataset conclusions remain bounded to synthetic tasks and compressed four-class MNIST. Sensitive-domain examples motivate why membership privacy matters but are not deployment-validation claims. We will expand related work on QML MIAs, differential privacy, unlearning, and noise-aware QML, and revise the paper's causal, architectural, hardware, and deployment wording accordingly.
