# Phase 7 Stage 2: confirmatory primary

**Frozen decision:** A+B only: serving-channel mismatch and privacy-auditing paper.

## Locked provenance

- Protocol SHA-256: `ab360b674c3843670cd4a2213924098a5a9ffef43607fd5ee6510f3c56a78d18`.
- Pre-unblinding raw-output seal SHA-256: `7dd18b5bf4a9460659dd463e5fdc03006eadb6822d846b03703f8d4b770e1857`.
- Confirmatory population: four cells, three target checkpoints per cell.
- Serving condition: noisy 128-shot Aer; simulator seeds 0–9.
- Replication unit: target checkpoint after averaging simulator seeds.
- Inference: 20,000-replicate hierarchical cell/target bootstrap.

## Attack results

| Attack | Mean AUC | Mean TPR@1% FPR | Mean TPR@5% FPR |
|---|---:|---:|---:|
| matched 16-reference noisy LiRA | 0.5136 | 1.42% | 5.73% |
| ChannelLiRA | 0.5111 | 1.25% | 5.61% |
| mismatched latent LiRA | 0.5171 | 1.01% | 4.91% |
| loss MIA | 0.5339 | 1.52% | 6.18% |
| privileged victim-crossfit learned comparator | 0.5227 | 1.22% | 5.69% |

## Frozen primary contrasts

| Contrast | TPR@1% difference | Hierarchical 95% interval | Decision boundary |
|---|---:|---:|---:|
| A: matched noisy − mismatched | +0.42 pp | [+0.12, +0.70] pp | +0.00 pp |
| B: ChannelLiRA − matched noisy | -0.17 pp | [-0.44, +0.10] pp | -0.50 pp |
| C: ChannelLiRA − loss | -0.27 pp | [-0.58, +0.06] pp | +0.00 pp |
| Stress: ChannelLiRA − learned | +0.03 pp | [-0.18, +0.26] pp | descriptive only |

## Gate decision

- **A — channel mismatch:** PASS.
- **B — efficient recovery:** PASS. The AUC interval is [-0.0080, +0.0026] against the frozen −0.0100 margin; amortized calibration cost is 18.75%.
- **C — practical attack:** FAIL.
- **Secondary experiment warranted:** yes; only A+B authorizes it.

## Claim boundary

The learned comparator is unchanged and privileged; this study cannot support superiority over learned MIAs generally. The four IBM-derived simulator cells do not establish universal architecture or real-hardware leakage. A failed primary gate cannot be rescued by secondary conditions.
