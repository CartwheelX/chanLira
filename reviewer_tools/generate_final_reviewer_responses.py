#!/usr/bin/env python3
"""Generate final paste-ready reviewer responses after LiRA/label-only completion."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_reviewer_artifacts import paired_hierarchical_effect  # noqa: E402


def clean(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def mean_sd(values: pd.Series, digits: int = 3, signed: bool = False) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    prefix = "+" if signed and float(values.mean()) >= 0 else ""
    return f"{prefix}{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"


def attack_row(
    name: str,
    access: str,
    frame: pd.DataFrame,
    reference_models: object = "—",
    note: str = "",
) -> dict[str, object]:
    return {
        "Attack": name,
        "Access": access,
        "Targets": int(frame["target_id"].nunique()),
        "Reference models/configuration": reference_models,
        "AUC": mean_sd(frame["auc"]),
        "TPR@5% FPR": mean_sd(frame["tpr_at_0_05_fpr"]),
        "TPR@10% FPR": mean_sd(frame["tpr_at_0_10_fpr"]),
        "Interpretation": note,
    }


def build_attack_overview(
    threshold: pd.DataFrame,
    learned: pd.DataFrame,
    lira: pd.DataFrame,
    label: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    names = {
        "confidence": "Confidence threshold",
        "correctness": "Correctness",
        "entropy": "Entropy threshold",
        "loss": "Loss threshold",
        "margin": "Margin threshold",
        "max_probability": "Maximum-probability threshold",
    }
    access = {
        "loss": "true-label probability + known candidate label",
        "confidence": "maximum predicted-class probability",
        "entropy": "entropy of the prediction vector",
        "margin": "top-1 minus top-2 predicted probability",
        "max_probability": "maximum predicted-class probability",
        "correctness": "predicted label + known candidate label",
    }
    for attack in ["loss", "confidence", "entropy", "margin", "max_probability", "correctness"]:
        group = threshold[threshold["attack"].eq(attack)]
        rows.append(
            attack_row(
                names[attack],
                access[attack],
                group.rename(
                    columns={
                        "tpr_at_requested_fpr_0p05": "tpr_at_0_05_fpr",
                        "tpr_at_requested_fpr_0p1": "tpr_at_0_10_fpr",
                    }
                ),
            )
        )

    learned_factorial = learned[
        learned["experiment"].astype(str).str.lower().eq("multiseed_factorial")
    ]
    learned_target = (
        learned_factorial.groupby("target_id", as_index=False)
        .agg(
            auc=("attack_auc", "mean"),
            tpr_at_0_05_fpr=("tpr@fpr=0.05", "mean"),
            tpr_at_0_10_fpr=("tpr@fpr=0.1", "mean"),
        )
    )
    rows.append(
        attack_row(
            "Learned prediction-vector",
            "full prediction vector + statistics",
            learned_target,
            note="mean over three attacker seeds per target",
        )
    )

    lira_specs = [
        (
            "Online LiRA, fixed variance",
            "lira_online_fixed_variance",
            "primary calibrated reference-model result",
        ),
        (
            "Online LiRA, per-record variance",
            "lira_online",
            "variance-model sensitivity check",
        ),
        (
            "Offline LiRA, fixed variance",
            "lira_offline_fixed_variance",
            "offline sensitivity check",
        ),
    ]
    for name, attack, note in lira_specs:
        rows.append(
            attack_row(
                name,
                "true-label probability + calibrated reference QNNs",
                lira[lira["attack"].eq(attack)],
                reference_models=16,
                note=note,
            )
        )
    rows.append(
        attack_row(
            "Label-only chord-boundary",
            "predicted class labels only + held-out anchors",
            label,
            note="query-based boundary proxy; not certified minimum distance",
        )
    )
    return pd.DataFrame(rows)


def build_structural_effects(
    lira: pd.DataFrame,
    label: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    attacks = [
        (
            "Online LiRA, fixed variance",
            lira[lira["attack"].eq("lira_online_fixed_variance")],
        ),
        ("Label-only chord-boundary", label),
    ]
    specifications = [
        ("Repetitions", "reps", 1, 5, ["fm_kind", "depth"], "5 − 1"),
        ("Depth", "depth", 2, 6, ["fm_kind", "reps"], "6 − 2"),
        (
            "Feature map",
            "fm_kind",
            "eff_su2",
            "z",
            ["reps", "depth"],
            "Z − EffSU2",
        ),
        (
            "Feature map",
            "fm_kind",
            "eff_su2",
            "zz",
            ["reps", "depth"],
            "ZZ − EffSU2",
        ),
        (
            "Feature map",
            "fm_kind",
            "z",
            "zz",
            ["reps", "depth"],
            "ZZ − Z",
        ),
    ]
    rows = []
    for attack_name, frame in attacks:
        for factor, column, low, high, blocks, contrast in specifications:
            result = paired_hierarchical_effect(
                frame,
                "auc",
                column,
                low,
                high,
                blocks,
                bootstrap,
                seed,
            )
            rows.append(
                {
                    "Attack": attack_name,
                    "Factor": factor,
                    "Contrast": contrast,
                    "Mean difference": result["mean_difference"],
                    "Paired-unit SD": result["sd_paired_units"],
                    "95% CI low": result["ci95_low"],
                    "95% CI high": result["ci95_high"],
                    "Paired target-seed units": result["n_paired_seed_units"],
                    "Bootstrap replicates": result["valid_bootstrap_replicates"],
                }
            )
    return pd.DataFrame(rows)


def build_configuration_table(
    threshold: pd.DataFrame,
    learned: pd.DataFrame,
    lira: pd.DataFrame,
    label: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["fm_kind", "reps", "depth"]
    base = threshold[threshold["attack"].eq("loss")][keys + ["target_id", "auc"]].rename(
        columns={"auc": "Loss threshold"}
    )
    lira_primary = lira[lira["attack"].eq("lira_online_fixed_variance")][
        ["target_id", "auc"]
    ].rename(columns={"auc": "Online LiRA"})
    label_primary = label[["target_id", "auc"]].rename(columns={"auc": "Label-only"})
    learned_factorial = learned[
        learned["experiment"].astype(str).str.lower().eq("multiseed_factorial")
    ]
    learned_target = (
        learned_factorial.groupby("target_id", as_index=False)["attack_auc"]
        .mean()
        .rename(columns={"attack_auc": "Learned vector"})
    )
    merged = base.merge(lira_primary, on="target_id").merge(label_primary, on="target_id")
    merged = merged.merge(learned_target, on="target_id")
    rows = []
    labels = {"eff_su2": "EffSU2", "z": "Z", "zz": "ZZ"}
    for values, group in merged.groupby(keys, sort=True):
        fm_kind, reps, depth = values
        rows.append(
            {
                "Structural configuration": f"{labels.get(fm_kind, fm_kind)}, reps={int(reps)}, depth={int(depth)}",
                "Target seeds": int(group["target_id"].nunique()),
                "Loss threshold AUC": mean_sd(group["Loss threshold"]),
                "Learned-vector AUC": mean_sd(group["Learned vector"]),
                "Online LiRA AUC": mean_sd(group["Online LiRA"]),
                "Label-only AUC": mean_sd(group["Label-only"]),
            }
        )
    return pd.DataFrame(rows)


def frame_markdown(frame: pd.DataFrame) -> str:
    return md_table(list(frame.columns), frame.itertuples(index=False, name=None))


def effect_markdown(effects: pd.DataFrame) -> str:
    rows = []
    for _, row in effects.iterrows():
        rows.append(
            [
                row["Attack"],
                row["Factor"],
                row["Contrast"],
                f"{row['Mean difference']:+.3f} ± {row['Paired-unit SD']:.3f}",
                f"[{row['95% CI low']:.3f}, {row['95% CI high']:.3f}]",
                int(row["Paired target-seed units"]),
            ]
        )
    return md_table(
        ["Attack", "Factor", "Contrast", "AUC difference ± SD", "95% CI", "Paired units"],
        rows,
    )


def compact_attack_markdown(overview: pd.DataFrame) -> str:
    wanted = [
        "Loss threshold",
        "Learned prediction-vector",
        "Online LiRA, fixed variance",
        "Online LiRA, per-record variance",
        "Offline LiRA, fixed variance",
        "Label-only chord-boundary",
    ]
    frame = overview[overview["Attack"].isin(wanted)].copy()
    return md_table(
        ["Attack", "Information access", "AUC", "TPR@5% FPR", "TPR@10% FPR"],
        (
            [row["Attack"], row["Access"], row["AUC"], row["TPR@5% FPR"], row["TPR@10% FPR"]]
            for _, row in frame.iterrows()
        ),
    )


def full_attack_markdown(overview: pd.DataFrame) -> str:
    """Render every requested scalar signal together with the learned baselines."""
    return md_table(
        ["Attack", "Information access", "AUC", "TPR@5% FPR", "TPR@10% FPR"],
        (
            [row["Attack"], row["Access"], row["AUC"], row["TPR@5% FPR"], row["TPR@10% FPR"]]
            for _, row in overview.iterrows()
        ),
    )


def geometry_effect_markdown() -> str:
    """Render the direct post-encoder repetition contrasts used in the response."""
    return md_table(
        ["Geometry measurement", "Definition", "Reps 5 − reps 1 ± SD", "95% CI"],
        [
            [
                "Class-similarity gap",
                "mean within-class minus between-class fidelity",
                "−0.124 ± 0.061",
                "[−0.158, −0.079]",
            ],
            [
                "Kernel–label alignment",
                "centered fidelity-kernel/label-kernel alignment",
                "−0.208 ± 0.132",
                "[−0.285, −0.109]",
            ],
            [
                "Effective rank",
                "spectral effective rank of the fidelity kernel",
                "+49.747 ± 37.448",
                "[15.791, 83.870]",
            ],
            [
                "Train/test MMD²",
                "kernel discrepancy between train and test encoded states",
                "−0.003 ± 0.008",
                "[−0.010, 0.003]",
            ],
        ],
    )


def write(path: Path, parts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("reviewer_results"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reviewer_results/reviewer_artifacts/final_responses"),
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()

    threshold = pd.read_csv(args.results_root / "factorial_threshold_mia/threshold_mia_raw.csv")
    lira = pd.read_csv(args.results_root / "lira_reference_mia/lira_reference_mia_raw.csv")
    label = pd.read_csv(args.results_root / "label_only_boundary/label_only_boundary_raw.csv")
    learned_frames = []
    for path in sorted(args.results_root.glob("learned_mia_seed*/attack_summary.csv")):
        learned_frames.append(pd.read_csv(path))
    if not learned_frames:
        raise FileNotFoundError("No learned-MIA summaries found")
    learned = pd.concat(learned_frames, ignore_index=True)

    overview = build_attack_overview(threshold, learned, lira, label)
    effects = build_structural_effects(lira, label, args.bootstrap, args.bootstrap_seed)
    configurations = build_configuration_table(threshold, learned, lira, label)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overview.to_csv(args.out_dir / "T10a_attack_breadth.csv", index=False)
    effects.to_csv(args.out_dir / "T10b_lira_label_structural_effects.csv", index=False)
    configurations.to_csv(args.out_dir / "T10c_attack_structural_configurations.csv", index=False)

    attack_table = compact_attack_markdown(overview)
    full_attack_table = full_attack_markdown(overview)
    structural_table = effect_markdown(effects)
    configuration_table = frame_markdown(configurations)
    geometry_table = geometry_effect_markdown()

    posted_note = (
        "This is a follow-up to the already-posted geometry/noise response; it does not "
        "repeat that material."
    )

    one_mia = [
        "# Follow-up response to Reviewer 1myw: completed MIA baselines",
        "",
        posted_note,
        "",
        (
            "Following up on the attack-breadth paragraph in our previous response, the "
            "reference-model LiRA and class-label-only experiments are now complete. Both "
            "were evaluated on all 36 targets in our focused confirmatory MNIST-QNN "
            "factorial (12 structural configurations × three independently initialized "
            "target models). This is the multi-seed follow-up subset, not a claim that we "
            "reran every configuration in the submission's broader architecture/dataset sweep."
        ),
        "",
        (
            "For LiRA, we trained 16 same-architecture reference QNNs per structural "
            "configuration (192 references total). Every one of the 400 candidate records "
            "was included in exactly eight references, and every reference was trained on "
            "200 candidates. The primary fixed-variance online LiRA result is 0.609 ± 0.063 "
            "AUC, with TPR 0.128 ± 0.056 at 5% FPR and 0.192 ± 0.069 at 10% FPR. The "
            "per-record-variance online result is 0.594 ± 0.057 AUC, showing that the result "
            "is not specific to one variance model. The offline result is much weaker, which "
            "we report rather than selecting only the strongest LiRA variant."
        ),
        "",
        (
            "The label-only attack consumes only returned class labels and estimates input-space "
            "boundary distance by changed-label searches toward held-out validation anchors. It "
            "obtains 0.582 ± 0.052 AUC, with TPR 0.077 ± 0.027 at 5% FPR and 0.139 ± 0.030 "
            "at 10% FPR. We describe this as a chord-boundary proxy, not HopSkipJump/QEBA or "
            "a certified minimum boundary distance."
        ),
        "",
        attack_table,
        "",
        (
            "*Entries are mean ± sample SD across 36 target models. The learned attack is "
            "first averaged across three attacker seeds per target. LiRA uses one 16-reference "
            "bank per structural configuration.*"
        ),
        "",
        (
            "The paired analysis supports the encoder-induced pathway across access settings. "
            "Repetition increases "
            "label-only AUC by 0.068 ± 0.026 (95% CI [0.049, 0.085]) and pooled LiRA AUC "
            "by 0.044 ± 0.045 ([0.013, 0.072]). In fixed-depth comparisons, LiRA AUC rises "
            "with repetition in five of six configurations; the single ZZ/depth=2 exception "
            "is unresolved. Z and ZZ remain higher than EffSU2 for both attacks, while the "
            "ZZ−Z intervals include zero. These calibrated attacks therefore support the claim "
            "that encoder family and repetition condition the downstream membership signal. "
            "Variational depth can modulate the signal produced by the trained model, but that "
            "does not remove the upstream encoder association."
        ),
        "",
        structural_table,
    ]

    epmi = [
        "# Response to Reviewer Epmi",
        "",
        (
            "Thank you for the detailed evaluation requests. The submitted paper reports a "
            "broad exploratory sweep across datasets, QNN/HQNN/QCNN wrappers, feature-map "
            "families, repetitions, widths, variational depths, entanglers, gates, and padding. "
            "For the rebuttal, we did not rerun that entire sweep with multiple seeds. Instead, "
            "we defined a focused 3×2×2 MNIST-QNN confirmatory factorial over feature-map family, "
            "repetition, and variational depth, trained three target-model seeds for every "
            "configuration (36 targets), and ran the same MIA suite on every one without "
            "attack-outcome selection. We also added targeted geometry, finite-shot/noise, and "
            "complete-wrapper architecture controls."
        ),
        "",
        "## Attack breadth on the focused confirmatory factorial",
        "",
        (
            "The expanded attacks separate the information channels requested in the review. "
            "Among the uncalibrated scalar signals, loss is the strongest by mean AUC "
            "(0.596 ± 0.052). Correctness reaches 0.582 ± 0.049 AUC but has zero TPR at "
            "5% and 10% FPR because its binary score cannot realize those low operating "
            "points. Confidence, entropy, margin, and maximum probability are all close to "
            "chance (0.520–0.521 mean AUC), and the learned feature combination reaches "
            "0.574 ± 0.065. Thus, loss is the clearest raw output feature in this factorial; "
            "combining the prediction-vector features does not improve its mean AUC. "
            "Fixed-variance online LiRA is numerically strongest overall at 0.609 ± 0.063 "
            "and improves low-FPR TPR relative to raw loss, while the class-label-only "
            "boundary proxy reaches 0.582 ± 0.052."
        ),
        "",
        full_attack_table,
        "",
        (
            "In our exported statistics, confidence is defined as the largest predicted-class "
            "probability, so the confidence and maximum-probability rows are the same scalar "
            "baseline under two commonly used names; they are not independent attacks."
        ),
        "",
        (
            "LiRA uses 16 same-architecture references per structural configuration; every "
            "candidate is IN for exactly eight references. Its reference-training pool is an "
            "explicit approximation formed from the canonical 200 target-train and 200 "
            "target-test candidates. The label-only method uses no probability, loss, logit, "
            "gradient, or model parameter and is described only as a chord-boundary proxy."
        ),
        "",
        "## Paired structural effects across attacks",
        "",
        structural_table,
        "",
        (
            "The calibrated attacks support the same encoder-induced pathway. Repetition raises "
            "online LiRA AUC in five of six fixed-depth comparisons and has a positive pooled "
            "contrast; Z and ZZ exceed EffSU2 under both LiRA and label-only access, while "
            "ZZ−Z is unresolved. Variational depth also modulates the trained model's output "
            "signal, so we do not claim that the encoder is the only contributor. The supported "
            "claim is that encoder family and repetition alter the pre-training representation "
            "and systematically condition the downstream asymmetry exploited by MIAs."
        ),
        "",
        "## Gap, causal scope, and geometry",
        "",
        (
            "Across all 36 targets, loss-AUC and accuracy gap have Pearson r=0.948 "
            "([0.864, 0.982]) and Spearman ρ=0.931 ([0.710, 0.974]). A descriptive model "
            "containing standardized gap plus feature map, repetition, and depth obtains "
            "R²=0.935 ([0.852, 0.986]); after conditioning on gap, all residual structural "
            "coefficient intervals cross zero. We therefore agree with the reviewer's causal "
            "distinction: the evidence supports structural choices as antecedents of overfitting, "
            "which then supplies membership signal, not a gap-independent causal leakage effect."
        ),
        "",
        (
            "We directly measured K(x,x′)=|⟨ψ(x)|ψ(x′)⟩|² immediately after the fixed "
            "encoder and before the trainable variational circuit. The class-similarity gap "
            "summarizes the within-class and between-class kernel-similarity distributions; "
            "kernel–label alignment measures task alignment; and effective rank measures the "
            "kernel spectrum. Their paired repetition contrasts are:"
        ),
        "",
        geometry_table,
        "",
        (
            "Increasing repetition therefore produces a higher-rank but less class-aligned "
            "post-encoder representation. The train/test MMD² interval includes zero, so the "
            "fixed encoder does not itself measurably separate members from non-members. These "
            "measurements directly verify encoder→geometry association, but they do not causally "
            "identify geometry→generalization-gap mediation. We will accordingly replace "
            "‘geometric mechanism’ with ‘empirically supported geometric pathway.’"
        ),
        "",
        "## Scope and remaining limitations",
        "",
        (
            "The primary factorial uses one fixed MNIST split, so three model seeds are not "
            "three independent data splits. Architecture controls are complete-wrapper comparisons "
            "with resource accounting and a classical MLP with a comparable small parameter "
            "budget to the QNN roles; preprocessing and "
            "heads remain unmatched. Backend-noise results are Aer simulations from one "
            "`ibm_kingston` calibration snapshot, not hardware execution. We will state all three "
            "limitations and avoid independent-causal or hardware-general claims."
        ),
    ]

    epmi_openreview = [
        "# Response to Reviewer Epmi",
        "",
        (
            "Thank you. The submission reports a broad exploratory sweep across datasets, "
            "architectures, encoders, repetitions, depths, and lower-level circuit choices. "
            "For the rebuttal, we added a focused MNIST-QNN factorial (3 feature maps × 2 "
            "repetitions × 2 depths × 3 target seeds = 36 targets) and applied the same attack "
            "suite to every target. Full protocols, tables, and figures are in our "
            "[detailed response](https://github.com/CartwheelX/QuRiFT/blob/main/docs/Reviewer-Epmi.md)."
        ),
        "",
        "**Which output signal drives MIA?**",
        "",
        (
            "We evaluated every requested scalar separately. Mean AUC ± sample SD across 36 "
            "targets is: loss 0.596 ± 0.052; correctness 0.582 ± 0.049; entropy "
            "0.521 ± 0.023; margin 0.521 ± 0.026; and confidence/maximum probability "
            "0.520 ± 0.025. Confidence and maximum probability are the same exported scalar, "
            "not independent attacks. The learned prediction-vector-plus-statistics attacker "
            "obtains 0.574 ± 0.065. Thus, loss is the clearest raw signal, and combining features "
            "does not improve its mean AUC. Correctness has zero TPR at 5% and 10% FPR because "
            "its score is binary."
        ),
        "",
        (
            "Fixed-variance online LiRA is numerically strongest overall (AUC 0.609 ± 0.063; "
            "TPR@5% FPR 0.128 ± 0.056). It uses 16 same-architecture references per structural "
            "configuration; each candidate is IN for eight. The class-label-only "
            "chord-boundary proxy obtains 0.582 ± 0.052 AUC and 0.077 ± 0.027 TPR@5% FPR."
        ),
        "",
        "**Paired structural effects across the calibrated and label-only attacks.**",
        "",
        structural_table,
        "",
        (
            "Repetition has a positive pooled effect under both attacks, and Z and ZZ exceed "
            "EffSU2; ZZ−Z remains unresolved. LiRA rises with repetition in five of six "
            "fixed-depth comparisons. Variational depth also modulates the downstream signal, "
            "but does not remove the encoder-family/repetition association."
        ),
        "",
        "**Direct Hilbert-space geometry.**",
        "",
        (
            "We compute the fidelity kernel K(x,x′)=|⟨ψ(x)|ψ(x′)⟩|² immediately after the fixed "
            "encoder and before the trainable variational circuit. Paired reps=5 minus reps=1 "
            "effects are:"
        ),
        "",
        geometry_table,
        "",
        (
            "Increasing repetition therefore produces a higher-rank but less class-aligned "
            "post-encoder representation. The MMD² interval includes zero, so the fixed encoder "
            "does not itself measurably separate members from non-members before training."
        ),
        "",
        "**Interpretation and scope.**",
        "",
        (
            "Across the 36 targets, accuracy gap strongly tracks loss-AUC (Spearman ρ=0.931, "
            "95% CI [0.710, 0.974]). After conditioning descriptively on gap, the residual "
            "structural coefficient intervals cross zero. We therefore refine the claim to an "
            "empirically supported, overfitting-mediated pathway: encoder design → post-encoder "
            "geometry → downstream generalization asymmetry → membership signal. The geometry "
            "measurements verify encoder→geometry association but do not causally identify "
            "geometry→gap mediation. The factorial uses one fixed MNIST split, and it is a focused "
            "multi-seed confirmation rather than a rerun of every configuration in the original "
            "broad sweep. We will state these limitations and revise causal wording accordingly."
        ),
    ]

    nvbh = [
        "# Response to Reviewer nVBH",
        "",
        (
            "Thank you. The submission provides broad exploratory coverage across datasets, "
            "QNN/HQNN/QCNN wrappers, and circuit choices. The rebuttal adds a focused complete "
            "3×2×2 MNIST-QNN factorial over feature-map family, repetition, and depth, with three "
            "independently initialized targets per configuration (36 targets). This tests the "
            "central structural claim with replication while retaining the original sweep as "
            "exploratory breadth."
        ),
        "",
        "## Q1. Multi-seed robustness of the structural and attack claims",
        "",
        (
            "The three target seeds quantify initialization sensitivity; the learned attacker is "
            "also repeated over three training seeds, and LiRA uses 16 references per structural "
            "configuration. Mean ± SD and paired hierarchical-bootstrap intervals are reported. "
            "For loss-MIA, reps=5 minus reps=1 is +0.069 ± 0.029 AUC (95% CI [0.049, 0.088]), "
            "depth=6 minus depth=2 is +0.042 ± 0.026 ([0.024, 0.060]), Z−EffSU2 is "
            "+0.057 ± 0.032 ([0.027, 0.080]), and ZZ−EffSU2 is +0.052 ± 0.027 "
            "([0.030, 0.074]). These results support the claim that encoder family and repetition "
            "systematically condition membership leakage across target initializations."
        ),
        "",
        structural_table,
        "",
        (
            "The access-model comparison adds nuance to the hierarchy: repetition has a positive "
            "pooled effect for LiRA and label-only attacks; Z and ZZ exceed EffSU2 under both; and "
            "ZZ−Z remains unresolved. Under loss and label-only attacks the repetition contrast "
            "exceeds the depth contrast, whereas LiRA is also strongly modulated by depth. We "
            "therefore claim that encoder choice and repetition are upstream privacy-relevant "
            "factors, not that they dominate every downstream attack statistic. The primary data "
            "seed remains fixed, so model seeds are not independent data-split replications."
        ),
        "",
        "## Q2. All-configuration evaluation and post-hoc selection",
        "",
        (
            "We do not reuse the submission's post-hoc baseline/stress/hard labels for confirmatory "
            "inference. Every one of the 36 prespecified targets receives the same loss-threshold, "
            "learned-vector, online LiRA, and label-only analyses; complete results follow."
        ),
        "",
        configuration_table,
        "",
        (
            "*Entries are mean ± sample SD across three target seeds. Learned-vector AUC is first "
            "averaged across three attacker seeds.*"
        ),
        "",
        (
            "Across all 36 targets, accuracy gap and loss-AUC have Pearson r=0.948 (95% CI "
            "[0.864, 0.982]) and Spearman ρ=0.931 ([0.710, 0.974]). This answers the circular-"
            "selection concern within the focused factorial and supports gap as a strong descriptive "
            "proxy, not a deterministic or externally validated predictor. It is not a claim that "
            "every configuration in the original thousands-run sweep was retrained."
        ),
        "",
        "## Q3. Mathematical definition of the factor-attribution figure",
        "",
        (
            "Fig. 8 uses a dataset-wise normalized factor-association score for the prespecified "
            "privacy-risk proxy $y=\\Delta_{\\mathrm{gen}}$. Let $N$ be the number of runs and "
            "$g_j$ the number of levels of categorical factor j. Its one-way ANOVA statistic "
            "$F_j$ is transformed as:"
        ),
        "",
        "$$s_j^{(\\mathrm{cat})} = F_j / (F_j + N - g_j).$$",
        "",
        "For a numeric factor and displayed numeric product term, respectively:",
        "",
        "$$s_j^{(\\mathrm{num})} = |\\operatorname{corr}(X_j,y)|,  \\qquad "
        "s_{jk} = |\\operatorname{corr}(X_j X_k,y)|.$$",
        "",
        "The plotted allocation is:",
        "",
        "$$A_j = 100s_j / \\sum_k s_k,  \\qquad \\sum_j A_j = 100.$$",
        "",
        (
            "The component scores are nonnegative, dimensionless, and bounded by one. Thus, 22.5% "
            "means 22.5% of the aggregate factor-association score over the displayed terms within "
            "that dataset. This hybrid ANOVA/correlation diagnostic is a broad-sweep descriptive "
            "ranking, not a Sobol decomposition or conditional joint-model coefficient. Direct MIA "
            "validation is separate: the balanced factorial's paired loss-AUC intervals corroborate "
            "the feature-map and repetition associations without relying on this normalization. We "
            "will add the equations and use ‘normalized factor-association share’ in the revision."
        ),
        "",
        "## Q4. Architecture controls",
        "",
        (
            "We added QNN, HQNN, QCNN, and a classical MLP with a comparable small parameter budget "
            "to the QNN roles, evaluated over three structural roles and three target seeds with exact "
            "parameter and main-stack gate counts. Relative to QNN, MLP improves test accuracy by "
            "+0.146 ± 0.093 ([0.062, 0.259]) while its gap and AUC differences are unresolved. QCNN "
            "improves test accuracy by +0.190 ± 0.052 ([0.148, 0.249]) and reduces gap by "
            "−0.041 ± 0.042 ([−0.083, −0.007]) and loss-AUC by −0.019 ± 0.025 "
            "([−0.040, −0.001]); HQNN intervals cross zero. These results support architecture as a "
            "moderator of the encoder-induced signal. Because preprocessing and heads remain "
            "different, they are complete-wrapper controls rather than matched causal ablations."
        ),
        "",
        "## Q5. Stronger baselines and bounded conclusions",
        "",
        (
            "Beyond the classical MLP model control, we broadened the adversaries to separate scalar "
            "thresholds, a learned prediction-vector attacker, calibrated online/offline LiRA, and a "
            "class-label-only boundary proxy. This tests whether the structural signal is specific to "
            "one attack or access model."
        ),
        "",
        attack_table,
        "",
        (
            "We did not add classical CNN/kernel, quantum-kernel, regularized/early-stopped, "
            "differentially private, or calibration-defense baselines. The confirmatory factorial "
            "also does not provide new multi-seed validation of width, entangler, gate, or padding "
            "trends. Dataset scope remains synthetic tasks and compressed four-class MNIST. Sensitive "
            "domains motivate why membership matters under established classical-ML threats and "
            "proposed QML use cases; they are not deployment-validation claims. We will bound "
            "conclusions to the evaluated simulated datasets and avoid claims of sensitive-domain "
            "deployment readiness."
        ),
    ]

    nvbh_openreview_effects = md_table(
        ["Attack", "Reps 5−1", "Depth 6−2", "Z−EffSU2", "ZZ−EffSU2"],
        [
            ["Loss", "+.069 [.049,.088]", "+.042 [.024,.060]", "+.057 [.027,.080]", "+.052 [.030,.074]"],
            ["Online LiRA", "+.044 [.013,.072]", "+.077 [.038,.118]", "+.057 [.038,.076]", "+.070 [.021,.117]"],
            ["Label-only", "+.068 [.049,.085]", "+.047 [.021,.068]", "+.038 [.004,.061]", "+.054 [.022,.087]"],
        ],
    )
    nvbh_openreview = [
        "# Response to Reviewer nVBH",
        "",
        (
            "Thank you. We added a focused complete MNIST-QNN factorial (3 feature maps × 2 "
            "repetitions × 2 depths × 3 target seeds = 36 targets) and applied the same attacks to "
            "every target without outcome selection. Full protocols, per-configuration results, "
            "attribution equations, resource counts, and limitations are in our "
            "[detailed response](https://github.com/CartwheelX/QuRiFT/blob/main/docs/Reviewer-nVBH.md)."
        ),
        "",
        "**1. Multi-seed robustness.**",
        "",
        "Values below are paired AUC differences with 95% hierarchical-bootstrap CIs.",
        "",
        nvbh_openreview_effects,
        "",
        (
            "These results support the claim that encoder family and repetition condition leakage "
            "across target initializations and access models. Repetition exceeds depth for loss and "
            "label-only; LiRA is also strongly depth-modulated, so we do not claim encoder factors "
            "dominate every attack statistic. Each configuration has three target seeds; the learned "
            "attacker also has three training seeds and LiRA has 16 references/configuration. The "
            "data split is fixed, so these are not independent split replications."
        ),
        "",
        "**2. Post-hoc selection and gap validation.**",
        "",
        (
            "We do not reuse baseline/stress/hard labels for confirmation. Across all 36 prespecified "
            "targets, gap and loss-AUC have Pearson r=.948 (CI [.864,.982]) and Spearman ρ=.931 "
            "([.710,.974]). This removes attack-outcome selection within the focused factorial and "
            "supports gap as a strong descriptive proxy, not a deterministic predictor. It is not a "
            "multi-seed rerun of every configuration in the original broad sweep."
        ),
        "",
        "**3. Factor attribution.**",
        "",
        (
            "Fig. 8 is a dataset-wise normalized factor-association analysis of the generalization-"
            "gap proxy. Categorical terms use an ANOVA-derived score; numeric and displayed product "
            "terms use absolute Pearson correlation; scores are normalized to sum to 100. Hence a "
            "reported percentage is a share of the aggregate displayed association score, not a "
            "Sobol or causal allocation. The detailed response gives the exact equations. The new "
            "paired MIA intervals separately corroborate the feature-map/repetition associations."
        ),
        "",
        "**4. Architecture controls.**",
        "",
        (
            "Across three roles × three seeds, we evaluated QNN/HQNN/QCNN and a classical MLP with "
            "a comparable small parameter budget to the QNN roles, reporting exact parameter/gate "
            "counts. Relative to QNN, MLP test accuracy changes by +.146 ± .093 (CI [.062,.259]) "
            "with unresolved gap/AUC effects; QCNN changes accuracy by +.190 ± .052 ([.148,.249]), "
            "gap by −.041 ± .042 ([−.083,−.007]), and loss-AUC by −.019 ± .025 ([−.040,−.001]); "
            "HQNN intervals cross zero. This supports architecture as a moderator, but wrappers retain "
            "different preprocessing/heads and are not matched causal ablations."
        ),
        "",
        "**5. Baselines and scope.**",
        "",
        (
            "We added the classical MLP control and attacks spanning scalar thresholds, learned "
            "prediction vectors, calibrated LiRA, and label-only access. We did not add every proposed "
            "model/defense baseline. The focused factorial does not revalidate width, gates, "
            "entanglers, or padding with multiple seeds, and datasets remain synthetic plus compressed "
            "four-class MNIST. Sensitive domains motivate why membership matters; they are not "
            "deployment-validation claims. We will state these boundaries explicitly."
        ),
    ]

    ac = [
        "# Response to the Area Chair",
        "",
        (
            "We thank the Area Chair for consolidating the discussion. In response, we completed: "
            "(i) a focused 36-target multi-seed MNIST-QNN factorial; (ii) threshold, learned-vector, "
            "16-reference online/offline LiRA, and class-label-only attacks; (iii) direct post-encoder "
            "fidelity-kernel geometry; (iv) QNN/HQNN/QCNN/classical-MLP wrapper controls with resource "
            "accounting; and (v) finite-shot and ibm_kingston-derived Aer-noise evaluation. Full "
            "tables and protocols are provided in the detailed responses linked in our reviewer comments."
        ),
        "",
        "**Statistical robustness, selection, and attack breadth.**",
        "",
        (
            "The confirmatory factorial covers 3 feature maps × 2 repetitions × 2 depths × 3 target "
            "initializations. Every target receives the same attacks; the earlier post-hoc "
            "baseline/stress/hard labels are not used for confirmation. For loss-MIA, reps=5−1 is "
            "+0.069 ± 0.029 AUC (95% CI [0.049,0.088]), depth=6−2 is +0.042 ± 0.026 "
            "([0.024,0.060]), Z−EffSU2 is +0.057 ± 0.032 ([0.027,0.080]), and ZZ−EffSU2 is "
            "+0.052 ± 0.027 ([0.030,0.074]). The learned attacker is repeated over three training "
            "seeds and LiRA uses 16 references/configuration. The data split remains fixed, so these "
            "results establish initialization robustness rather than multi-split generalization."
        ),
        "",
        (
            "Across attacks with different access assumptions, the same aggregate feature-map and "
            "repetition directions recur. Fixed-variance online LiRA has the highest mean performance "
            "(AUC 0.609 ± 0.063; TPR@10% FPR 0.192 ± 0.069), followed by loss-threshold "
            "(0.596 ± 0.052; 0.135 ± 0.039); label-only obtains 0.582 ± 0.052 AUC using predicted "
            "labels alone. Repetition has positive pooled LiRA and label-only effects, and Z/ZZ exceed "
            "EffSU2 under both. Attack choice changes magnitude—LiRA is also depth-modulated—but the "
            "principal encoder associations are not specific to one attacker."
        ),
        "",
        "**Overfitting, proxy validity, and direct geometry.**",
        "",
        (
            "Across all 36 prespecified targets, gap and loss-AUC have Spearman ρ=0.931 "
            "([0.710,0.974]); after conditioning descriptively on gap, residual structural coefficient "
            "intervals cross zero. We therefore do not claim gap-independent causation. Directly after "
            "the fixed encoder, reps=5−1 changes within-minus-between-class fidelity by "
            "−0.124 ± 0.061 ([−0.158,−0.079]), kernel–label alignment by −0.208 ± 0.132 "
            "([−0.285,−0.109]), and effective rank by +49.747 ± 37.448 ([15.791,83.870]); the "
            "train/test MMD² interval includes zero. Together these results support an empirically "
            "measured pathway—encoder design → post-encoder geometry → downstream generalization "
            "asymmetry → membership signal—rather than a geometry-only or causally identified mechanism."
        ),
        "",
        "**Finite shots and backend-derived noise.**",
        "",
        (
            "We evaluated five representative configurations, 15 independently trained checkpoints, "
            "three shot counts, and ten simulator seeds under exact inference, ideal finite shots, and "
            "an ibm_kingston-derived Aer model (915 target/execution replicates). The exact high-minus-"
            "low loss-AUC difference 0.179 ± 0.030 is attenuated to 0.096 ± 0.012, 0.124 ± 0.026, "
            "and 0.132 ± 0.036 at 128, 512, and 1,024 noisy shots. Aggregate ordering remains, while "
            "nearby configurations can reorder in individual runs. This is a backend-derived robustness "
            "check, not hardware execution or evidence of device universality."
        ),
        "",
        "**Architecture, attribution, and scope.**",
        "",
        (
            "Relative to paired QNN roles, QCNN improves accuracy by +0.190 ± 0.052 "
            "([0.148,0.249]) and reduces gap by −0.041 ± 0.042 ([−0.083,−0.007]) and loss-AUC by "
            "−0.019 ± 0.025 ([−0.040,−0.001]); MLP and HQNN gap/AUC intervals are unresolved. "
            "These are complete-wrapper controls: preprocessing and heads remain unmatched. We will "
            "define Fig. 8 percentages precisely as dataset-normalized factor-association shares and "
            "supplement them with paired MIA intervals, rather than interpret them as causal allocations."
        ),
        "",
        (
            "The submission's broad sweep supplies exploratory coverage; the rebuttal confirms "
            "feature-map family, repetition, and depth in the focused factorial, not every original "
            "width/gate/entangler/padding configuration. Dataset conclusions remain bounded to synthetic "
            "tasks and compressed four-class MNIST. Sensitive-domain examples motivate why membership "
            "privacy matters but are not deployment-validation claims. We will expand related work on "
            "QML MIAs, differential privacy, unlearning, and noise-aware QML, and revise the paper's "
            "causal, architectural, hardware, and deployment wording accordingly."
        ),
    ]

    scope = [
        "# Scope of the submitted sweep and rebuttal experiments",
        "",
        (
            "The rebuttal experiments supplement rather than replace the submitted paper's "
            "broad sweep. The distinction below should be stated explicitly whenever the new "
            "multi-seed results are described."
        ),
        "",
        md_table(
            ["Evidence block", "Scope", "Purpose", "Claim boundary"],
            [
                [
                    "Submitted broad sweep",
                    "Multiple datasets; QNN/HQNN/QCNN; three feature-map families; repetitions, widths, depths, entanglers, gates, and padding",
                    "Exploratory breadth and discovery of the reported structural hierarchy",
                    "Not rerun in full with the new multi-seed/attack protocol",
                ],
                [
                    "Focused confirmatory factorial",
                    "MNIST QNN; 3 feature maps × 2 repetitions × 2 depths × 3 model seeds = 36 targets; fixed data split",
                    "Initialization robustness and paired uncertainty for the central encoder/repetition/depth claim",
                    "Supports the core claim in this factorial, not every submitted configuration",
                ],
                [
                    "Expanded MIA suite",
                    "Every one of the 36 confirmatory targets; thresholds, learned vector, LiRA, and label-only",
                    "Remove attack/regime selection within the confirmatory factorial and test multiple access models",
                    "Does not apply LiRA/label-only to the entire original sweep",
                ],
                [
                    "Direct geometry",
                    "MNIST and Moons; three feature maps; reps 1/5; nominal data seeds 43–45",
                    "Directly measure post-encoder fidelity-kernel geometry",
                    "MNIST nominal seeds duplicate states; causal mediation is not identified",
                ],
                [
                    "Finite-shot/noise sanity check",
                    "Five representative MNIST-QNN configurations; 15 checkpoints; 128/512/1024 shots; ten simulator seeds",
                    "Test whether the broad leakage ordering survives one backend-derived noise model",
                    "Aer simulation from one backend snapshot, not hardware/general noise validation",
                ],
                [
                    "Architecture controls",
                    "QNN/HQNN/QCNN/MLP-QNN; three structural roles × three model seeds",
                    "Complete-wrapper performance, leakage, and resource comparison",
                    "Preprocessing and heads are not matched causal ablations",
                ],
            ],
        ),
        "",
        (
            "Recommended description: ‘The original submission provides broad exploratory "
            "coverage. The rebuttal adds a focused multi-seed confirmatory factorial and targeted "
            "geometry, attack-breadth, architecture, and noise checks for the central claim.’"
        ),
    ]

    files = {
        "AREA_CHAIR.md": ac,
        "REVIEWER_EPMI.md": epmi,
        "REVIEWER_EPMI_OPENREVIEW.md": epmi_openreview,
        "REVIEWER_1MYW_BASELINE_FOLLOWUP.md": one_mia,
        "REVIEWER_NVBH.md": nvbh,
        "REVIEWER_NVBH_OPENREVIEW.md": nvbh_openreview,
        "EXPERIMENT_SCOPE.md": scope,
    }
    for name, parts in files.items():
        write(args.out_dir / name, parts)

    combined = ["# Final reviewer responses", ""]
    for name in ["AREA_CHAIR.md", "REVIEWER_EPMI.md", "REVIEWER_1MYW_BASELINE_FOLLOWUP.md", "REVIEWER_NVBH.md"]:
        combined.extend(files[name])
        combined.extend(["", "---", ""])
    write(args.out_dir / "ALL_RESPONSES.md", combined)
    print(f"[OK] Final reviewer responses -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
