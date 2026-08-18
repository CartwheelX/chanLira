#!/usr/bin/env python3
"""Generate paper-ready SaTML tables and figures from completed analyses.

Missing result families are recorded in the manifest instead of being silently
represented as zero or empty evidence. At least one supported input is required.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUTS = {
    "factorial": Path("satml_results/credit_factorial/paired_all_attacks/paired_contrasts.csv"),
    "geometry": Path("satml_results/credit_geometry/geometry_repetition_effects.csv"),
    "scaling": Path("satml_results/encoding_scale/paired_analysis/encoding_scale_contrasts.csv"),
    "selector_summary": Path("satml_results/selector_fresh/paired_analysis/selector_policy_summary.csv"),
    "selector_contrasts": Path("satml_results/selector_fresh/paired_analysis/selector_paired_contrasts.csv"),
    "pathway": Path("satml_results/mechanistic_pathway/pathway_correlations.csv"),
    "capacity": Path("satml_results/credit_factorial/capacity_controls/structural_resource_summary.csv"),
    "fashion_factorial": Path("satml_results/fashion_factorial/paired_all_attacks/paired_contrasts.csv"),
    "fashion_geometry": Path("satml_results/fashion_geometry/geometry_repetition_effects.csv"),
    "fashion_pathway": Path("satml_results/fashion_mechanism/pathway_correlations.csv"),
    "fashion_capacity": Path("satml_results/fashion_factorial/capacity_controls/structural_resource_summary.csv"),
    "wdbc_factorial": Path("satml_results/wdbc_targeted/paired_all_attacks/paired_contrasts.csv"),
    "wdbc_geometry": Path("satml_results/wdbc_geometry/geometry_repetition_effects.csv"),
    "wdbc_pathway": Path("satml_results/wdbc_mechanism/pathway_correlations.csv"),
    "wdbc_capacity": Path("satml_results/wdbc_targeted/capacity_controls/structural_resource_summary.csv"),
    "noise_n1_cells": Path("satml_results/noise/n1_structural/analysis/n1_cell_summary.csv"),
    "noise_n1_effects": Path("satml_results/noise/n1_structural/analysis/n1_factorial_effects_summary.csv"),
    "noise_n1_moderation": Path("satml_results/noise/n1_structural/analysis/n1_noise_moderation_summary.csv"),
    "noise_n2_queries": Path("satml_results/noise/n2_query_policy/analysis/n2_query_contrasts_summary.csv"),
    "noise_n3_lira": Path("satml_results/noise/n3_attack_breadth/analysis/n3_endpoint_contrasts.csv"),
}


def _display(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value).replace("|", "\\|")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(str(column) for column in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_display(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def latex_table(frame: pd.DataFrame, caption: str, label: str) -> str:
    escaped = frame.copy()
    for column in escaped:
        escaped[column] = escaped[column].map(_display)
    return escaped.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label=label,
        column_format="l" + "r" * max(len(escaped.columns) - 1, 0),
    )


def _effect_text(frame: pd.DataFrame, sd_column: str) -> pd.Series:
    return frame.apply(
        lambda row: (
            f"{row['mean_difference']:.3f} ± {row[sd_column]:.3f} "
            f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
        ),
        axis=1,
    )


def factorial_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "outcome", "attack", "factor", "contrast", "mean_difference",
        "sd_across_blocks", "ci95_low", "ci95_high", "n_independent_blocks",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"factorial table lacks {sorted(required - set(frame.columns))}")
    selected = frame[(frame.outcome == "auc") | frame.outcome.isin(["test_acc", "gap"])].copy()
    selected["Effect ± SD [95% CI]"] = _effect_text(selected, "sd_across_blocks")
    return selected.rename(
        columns={"outcome": "Outcome", "attack": "Attack", "factor": "Factor",
                 "contrast": "Contrast", "n_independent_blocks": "Blocks"}
    )[["Outcome", "Attack", "Factor", "Contrast", "Effect ± SD [95% CI]", "Blocks"]]


def geometry_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"metric", "mean_difference", "ci95_low", "ci95_high"}
    if not required.issubset(frame.columns):
        raise ValueError(f"geometry table lacks {sorted(required - set(frame.columns))}")
    selected = frame.copy()
    sd = "sd_across_dataset_encoder_seed_effects"
    if sd not in selected:
        selected[sd] = np.nan
    selected["Reps 5 − 1, mean ± SD [95% CI]"] = _effect_text(selected, sd)
    return selected.rename(columns={"metric": "Post-encoder metric"})[
        ["Post-encoder metric", "Reps 5 − 1, mean ± SD [95% CI]"]
    ]


def scaling_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"outcome", "attack", "fm_kind", "reps", "contrast", "mean_difference",
                "sd_across_blocks", "ci95_low", "ci95_high", "n_independent_blocks"}
    if not required.issubset(frame.columns):
        raise ValueError(f"scaling table lacks {sorted(required - set(frame.columns))}")
    selected = frame.copy()
    selected["Effect ± SD [95% CI]"] = _effect_text(selected, "sd_across_blocks")
    return selected.rename(
        columns={"outcome": "Outcome", "attack": "Attack", "fm_kind": "Feature map",
                 "reps": "Reps", "contrast": "Contrast", "n_independent_blocks": "Blocks"}
    )[["Outcome", "Attack", "Feature map", "Reps", "Contrast", "Effect ± SD [95% CI]", "Blocks"]]


def selector_table(summary: pd.DataFrame) -> pd.DataFrame:
    required = {"outcome", "attack", "selector_policy", "mean_value", "sd_across_blocks", "n_blocks"}
    if not required.issubset(summary.columns):
        raise ValueError(f"selector table lacks {sorted(required - set(summary.columns))}")
    selected = summary.copy()
    selected["Mean ± SD"] = selected.apply(
        lambda row: f"{row.mean_value:.3f} ± {row.sd_across_blocks:.3f}", axis=1
    )
    return selected.rename(
        columns={"outcome": "Outcome", "attack": "Attack", "selector_policy": "Policy", "n_blocks": "Fresh blocks"}
    )[["Outcome", "Attack", "Policy", "Mean ± SD", "Fresh blocks"]]


def noise_n1_cells_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fm_kind", "reps", "depth", "mode", "queries", "shots", "attack",
        "n_trained_checkpoints", "mean_auc", "sd_across_trained_checkpoints",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"N1 cell table lacks {sorted(required - set(frame.columns))}")
    preferred = {"loss", "learned_mlp_pv_plus_stats"}
    selected = frame[frame.attack.astype(str).isin(preferred)].copy()
    if selected.empty:
        selected = frame.copy()
    selected["AUC mean ± checkpoint SD"] = selected.apply(
        lambda row: f"{row.mean_auc:.3f} ± {row.sd_across_trained_checkpoints:.3f}",
        axis=1,
    )
    return selected.rename(
        columns={
            "fm_kind": "Feature map", "reps": "Reps", "depth": "Depth",
            "mode": "Mode", "queries": "Queries", "shots": "Shots/query",
            "attack": "Attack", "n_trained_checkpoints": "Checkpoints",
        }
    )[["Feature map", "Reps", "Depth", "Mode", "Queries", "Shots/query",
       "Attack", "AUC mean ± checkpoint SD", "Checkpoints"]]


def noise_n1_effect_table(frame: pd.DataFrame, *, moderation: bool = False) -> pd.DataFrame:
    required = {
        "attack", "mode", "queries", "shots", "effect", "fm_kind", "scope",
        "n_paired_units", "n_model_seed_blocks", "mean", "sd", "ci95_low", "ci95_high",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"N1 effect table lacks {sorted(required - set(frame.columns))}")
    selected = frame[~frame.scope.eq("pooled_across_encoders")].copy()
    selected["Effect ± SD [95% CI]"] = selected.apply(
        lambda row: (
            f"{row['mean']:.3f} ± {row.sd:.3f} "
            f"[{row.ci95_low:.3f}, {row.ci95_high:.3f}]"
        ),
        axis=1,
    )
    effect_label = "Noise change in AUC effect" if moderation else "AUC effect"
    selected = selected.rename(
        columns={
            "attack": "Attack", "mode": "Mode", "queries": "Queries",
            "shots": "Shots/query", "effect": "Contrast", "fm_kind": "Feature map",
            "n_model_seed_blocks": "Model-seed blocks",
            "Effect ± SD [95% CI]": f"{effect_label} ± SD [95% CI]",
        }
    )
    return selected[["Attack", "Mode", "Queries", "Shots/query", "Contrast",
                     "Feature map", f"{effect_label} ± SD [95% CI]", "Model-seed blocks"]]


def noise_n2_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "mode", "aggregation", "attack", "contrast", "n_target_checkpoints",
        "mean_auc_difference", "sd_across_target_checkpoints", "ci95_low", "ci95_high",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"N2 query table lacks {sorted(required - set(frame.columns))}")
    selected = frame.copy()
    selected["AUC difference ± SD [95% CI]"] = selected.apply(
        lambda row: (
            f"{row.mean_auc_difference:.3f} ± {row.sd_across_target_checkpoints:.3f} "
            f"[{row.ci95_low:.3f}, {row.ci95_high:.3f}]"
        ),
        axis=1,
    )
    return selected.rename(
        columns={
            "mode": "Mode", "aggregation": "Aggregation", "attack": "Attack",
            "contrast": "Query-policy contrast", "n_target_checkpoints": "Checkpoints",
        }
    )[["Mode", "Aggregation", "Attack", "Query-policy contrast",
       "AUC difference ± SD [95% CI]", "Checkpoints"]]


def noise_n3_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "mode", "shots", "attack", "contrast", "n_paired_model_seeds",
        "mean_auc_difference", "sd_across_model_seeds", "ci95_low", "ci95_high",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"N3 LiRA table lacks {sorted(required - set(frame.columns))}")
    selected = frame.copy()
    selected["AUC difference ± SD [95% CI]"] = selected.apply(
        lambda row: (
            f"{row.mean_auc_difference:.3f} ± {row.sd_across_model_seeds:.3f} "
            f"[{row.ci95_low:.3f}, {row.ci95_high:.3f}]"
        ),
        axis=1,
    )
    return selected.rename(
        columns={
            "mode": "Mode", "shots": "Shots/query", "attack": "LiRA variant",
            "contrast": "Endpoint contrast", "n_paired_model_seeds": "Paired model seeds",
        }
    )[["Mode", "Shots/query", "LiRA variant", "Endpoint contrast",
       "AUC difference ± SD [95% CI]", "Paired model seeds"]]


def pathway_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"link", "spearman", "ci95_low", "ci95_high", "n_structural_configurations", "inference_scope"}
    if not required.issubset(frame.columns):
        raise ValueError(f"pathway table lacks {sorted(required - set(frame.columns))}")
    selected = frame.copy()
    selected["Spearman ρ [95% CI]"] = selected.apply(
        lambda row: f"{row.spearman:.3f} [{row.ci95_low:.3f}, {row.ci95_high:.3f}]", axis=1
    )
    return selected.rename(
        columns={"link": "Descriptive pathway link", "n_structural_configurations": "Configurations",
                 "inference_scope": "Inference scope"}
    )[["Descriptive pathway link", "Spearman ρ [95% CI]", "Configurations", "Inference scope"]]


def capacity_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"fm_kind", "reps", "depth", "n_targets", "trainable_parameters_mean", "quantum_gate_count_mean"}
    if not required.issubset(frame.columns):
        raise ValueError(f"capacity table lacks {sorted(required - set(frame.columns))}")
    selected = frame.copy()
    selected["trainable_parameters_mean"] = selected.trainable_parameters_mean.round().astype(int)
    selected["quantum_gate_count_mean"] = selected.quantum_gate_count_mean.round(1)
    return selected.rename(
        columns={"fm_kind": "Feature map", "reps": "Reps", "depth": "Depth", "n_targets": "Targets",
                 "trainable_parameters_mean": "Trainable parameters",
                 "quantum_gate_count_mean": "Main-stack gates"}
    )[["Feature map", "Reps", "Depth", "Trainable parameters", "Main-stack gates", "Targets"]]


def _forest(
    frame: pd.DataFrame,
    *,
    label_columns: Iterable[str],
    title: str,
    x_label: str,
    output_stem: Path,
) -> None:
    plot = frame.dropna(subset=["mean_difference", "ci95_low", "ci95_high"]).copy()
    if plot.empty:
        return
    plot["label"] = plot[list(label_columns)].astype(str).agg(" · ".join, axis=1)
    height = max(3.2, 0.34 * len(plot) + 1.4)
    figure, axis = plt.subplots(figsize=(8.5, height))
    y = np.arange(len(plot))
    means = plot.mean_difference.to_numpy(float)
    errors = np.vstack([means - plot.ci95_low.to_numpy(float), plot.ci95_high.to_numpy(float) - means])
    axis.errorbar(means, y, xerr=errors, fmt="o", color="#1f4e79", ecolor="#527da3", capsize=3)
    axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_yticks(y, labels=plot.label)
    axis.invert_yaxis()
    axis.set_xlabel(x_label)
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _selector_plot(summary: pd.DataFrame, output_stem: Path) -> None:
    plot = summary[(summary.outcome == "auc")].copy()
    if plot.empty:
        return
    attacks = list(dict.fromkeys(plot.attack.astype(str)))
    policies = list(dict.fromkeys(plot.selector_policy.astype(str)))
    x = np.arange(len(attacks), dtype=float)
    width = 0.8 / max(len(policies), 1)
    figure, axis = plt.subplots(figsize=(max(7.0, 1.15 * len(attacks)), 4.4))
    for index, policy in enumerate(policies):
        values = plot[plot.selector_policy.astype(str) == policy].set_index("attack")
        means = np.array([values.mean_value.get(attack, np.nan) for attack in attacks], dtype=float)
        errors = np.array([values.sd_across_blocks.get(attack, np.nan) for attack in attacks], dtype=float)
        axis.bar(x + (index - (len(policies) - 1) / 2) * width, means, width,
                 yerr=errors, capsize=2, label=policy.replace("_", " "))
    axis.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axis.set_xticks(x, labels=attacks, rotation=25, ha="right")
    axis.set_ylabel("Membership-inference AUC")
    axis.set_title("Fresh-block selector evaluation")
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _interval_forest(
    frame: pd.DataFrame,
    *,
    mean_column: str,
    label_columns: Iterable[str],
    title: str,
    x_label: str,
    output_stem: Path,
) -> None:
    required = {mean_column, "ci95_low", "ci95_high", *label_columns}
    if not required.issubset(frame.columns):
        return
    plot = frame.dropna(subset=[mean_column, "ci95_low", "ci95_high"]).copy()
    if plot.empty:
        return
    plot["label"] = plot[list(label_columns)].astype(str).agg(" · ".join, axis=1)
    height = max(3.2, 0.34 * len(plot) + 1.4)
    figure, axis = plt.subplots(figsize=(9.0, height))
    y = np.arange(len(plot))
    means = plot[mean_column].to_numpy(float)
    errors = np.vstack(
        [means - plot.ci95_low.to_numpy(float), plot.ci95_high.to_numpy(float) - means]
    )
    axis.errorbar(means, y, xerr=errors, fmt="o", color="#1f4e79", ecolor="#527da3", capsize=3)
    axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_yticks(y, labels=plot.label)
    axis.invert_yaxis()
    axis.set_xlabel(x_label)
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _noise_n1_cell_plot(frame: pd.DataFrame, output_stem: Path) -> None:
    plot = frame[frame.attack.astype(str).eq("loss")].copy()
    if plot.empty:
        return
    plot["configuration"] = plot.apply(
        lambda row: f"{row.fm_kind}\nr{int(row.reps)} d{int(row.depth)}", axis=1
    )
    configurations = list(dict.fromkeys(plot.configuration))
    figure, axis = plt.subplots(figsize=(11.0, 4.8))
    x = np.arange(len(configurations))
    for mode, group in plot.groupby("mode"):
        indexed = group.set_index("configuration")
        means = np.array([indexed.mean_auc.get(item, np.nan) for item in configurations], dtype=float)
        errors = np.array(
            [indexed.sd_across_trained_checkpoints.get(item, np.nan) for item in configurations],
            dtype=float,
        )
        axis.errorbar(x, means, yerr=np.nan_to_num(errors), marker="o", capsize=2,
                      label=str(mode).replace("_", " "))
    axis.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axis.set_xticks(x, labels=configurations)
    axis.set_ylabel("Loss-threshold MIA AUC")
    axis.set_title("N1 structural ordering under frozen finite-shot noise")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def generate(inputs: dict[str, Path], out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir, figure_dir = out_dir / "tables", out_dir / "figures"
    table_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    loaded: dict[str, pd.DataFrame] = {}
    missing: dict[str, str] = {}
    for name, path in inputs.items():
        if path.exists():
            loaded[name] = pd.read_csv(path)
        else:
            missing[name] = str(path)
    if not loaded:
        raise FileNotFoundError("None of the supported SaTML analysis inputs exists")

    tables: list[tuple[str, str, pd.DataFrame]] = []
    errors: dict[str, str] = {}
    builders = {
        "factorial": ("Paired Credit-default structural effects", factorial_table),
        "geometry": ("Direct post-encoder geometry effects", geometry_table),
        "scaling": ("Encoder-angle scale sensitivity", scaling_table),
        "selector_summary": ("Fresh-block selector outcomes", selector_table),
        "noise_n1_cells": ("N1 structural leakage under a frozen calibration", noise_n1_cells_table),
        "noise_n1_effects": ("N1 repetition, depth, and interaction effects", noise_n1_effect_table),
        "noise_n1_moderation": (
            "N1 noise moderation of structural effects",
            lambda frame: noise_n1_effect_table(frame, moderation=True),
        ),
        "noise_n2_queries": ("N2 API-query policy contrasts", noise_n2_table),
        "noise_n3_lira": ("N3 noisy LiRA endpoint contrasts", noise_n3_table),
        "pathway": ("Descriptive mechanistic-pathway associations", pathway_table),
        "capacity": ("Capacity and circuit-resource controls", capacity_table),
        "fashion_factorial": ("Paired Fashion-MNIST structural effects", factorial_table),
        "fashion_geometry": ("Fashion-MNIST post-encoder geometry effects", geometry_table),
        "fashion_pathway": ("Fashion-MNIST mechanistic-pathway associations", pathway_table),
        "fashion_capacity": ("Fashion-MNIST capacity controls", capacity_table),
        "wdbc_factorial": ("Paired WDBC structural effects", factorial_table),
        "wdbc_geometry": ("WDBC post-encoder geometry effects", geometry_table),
        "wdbc_pathway": ("WDBC mechanistic-pathway associations", pathway_table),
        "wdbc_capacity": ("WDBC capacity controls", capacity_table),
    }
    for name, (title, builder) in builders.items():
        if name not in loaded:
            continue
        try:
            tables.append((name, title, builder(loaded[name])))
        except Exception as error:  # fail this family visibly; preserve other completed families
            errors[name] = f"{type(error).__name__}: {error}"

    markdown_parts = ["# SaTML result tables", "", "Generated directly from analysis CSVs; no values are hand-entered."]
    latex_parts = []
    for name, title, frame in tables:
        frame.to_csv(table_dir / f"{name}.csv", index=False)
        markdown_parts.extend(["", f"## {title}", "", markdown_table(frame)])
        latex_parts.append(latex_table(frame, title, f"tab:satml_{name}"))
    (table_dir / "satml_tables.md").write_text("\n".join(markdown_parts) + "\n", encoding="utf-8")
    (table_dir / "satml_tables.tex").write_text("\n\n".join(latex_parts) + "\n", encoding="utf-8")

    if "factorial" in loaded:
        factorial = loaded["factorial"]
        _forest(factorial[factorial.outcome == "auc"], label_columns=("attack", "contrast"),
                title="Paired structural effects on membership leakage",
                x_label="AUC difference (paired block mean)",
                output_stem=figure_dir / "factorial_attack_effects")
    if "geometry" in loaded:
        geometry = loaded["geometry"]
        _forest(geometry, label_columns=("metric",), title="Repetition-induced post-encoder geometry changes",
                x_label="Reps 5 − reps 1", output_stem=figure_dir / "geometry_repetition_effects")
    if "scaling" in loaded:
        scale = loaded["scaling"]
        _forest(scale, label_columns=("outcome", "fm_kind", "reps", "contrast"),
                title="Encoder-angle scale sensitivity", x_label="Paired difference from α=1",
                output_stem=figure_dir / "encoding_scale_effects")
    if "selector_summary" in loaded:
        _selector_plot(loaded["selector_summary"], figure_dir / "selector_fresh_auc")
    if "noise_n1_cells" in loaded:
        _noise_n1_cell_plot(loaded["noise_n1_cells"], figure_dir / "noise_n1_structural_ordering")
    if "noise_n1_effects" in loaded:
        effects = loaded["noise_n1_effects"]
        effects = effects[
            ~effects.scope.eq("pooled_across_encoders")
            & effects.attack.astype(str).eq("loss")
            & effects.effect.astype(str).isin(
                [
                    "repetition_5_minus_1_at_depth2",
                    "repetition_5_minus_1_at_depth6",
                    "feature_z_minus_eff_su2",
                    "feature_zz_minus_eff_su2",
                    "repetition_by_depth_interaction",
                ]
            )
        ]
        _interval_forest(
            effects,
            mean_column="mean",
            label_columns=("attack", "mode", "fm_kind", "effect"),
            title="N1 structural effects under exact and finite-shot inference",
            x_label="AUC effect",
            output_stem=figure_dir / "noise_n1_factorial_effects",
        )
    if "noise_n2_queries" in loaded:
        queries = loaded["noise_n2_queries"]
        queries = queries[
            queries.attack.astype(str).eq("loss")
            & queries.aggregation.astype(str).eq("mean_api_probabilities")
        ]
        _interval_forest(
            queries,
            mean_column="mean_auc_difference",
            label_columns=("attack", "mode", "aggregation", "contrast"),
            title="N2 repeated-query and shot-allocation contrasts",
            x_label="AUC difference",
            output_stem=figure_dir / "noise_n2_query_policy",
        )
    if "noise_n3_lira" in loaded:
        _interval_forest(
            loaded["noise_n3_lira"],
            mean_column="mean_auc_difference",
            label_columns=("attack", "mode", "contrast"),
            title="N3 LiRA structural endpoint contrast",
            x_label="High-leakage minus low-leakage endpoint AUC",
            output_stem=figure_dir / "noise_n3_lira",
        )
    for prefix in ("fashion", "wdbc"):
        factorial_name = f"{prefix}_factorial"
        geometry_name = f"{prefix}_geometry"
        if factorial_name in loaded:
            frame = loaded[factorial_name]
            _forest(
                frame[frame.outcome == "auc"],
                label_columns=("attack", "contrast"),
                title=f"{prefix.upper()} paired structural effects on membership leakage",
                x_label="AUC difference (paired block mean)",
                output_stem=figure_dir / f"{prefix}_attack_effects",
            )
        if geometry_name in loaded:
            _forest(
                loaded[geometry_name],
                label_columns=("metric",),
                title=f"{prefix.upper()} repetition-induced geometry changes",
                x_label="Reps 5 − reps 1",
                output_stem=figure_dir / f"{prefix}_geometry_repetition_effects",
            )

    generated = sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())
    manifest = {
        "inputs_loaded": {name: str(inputs[name]) for name in sorted(loaded)},
        "inputs_missing": missing,
        "family_errors": errors,
        "generated_files": generated,
        "note": "Missing families are omissions, never zero-valued results.",
    }
    (out_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("satml_results/paper_artifacts"))
    for name, default in DEFAULT_INPUTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default)
    args = parser.parse_args()
    inputs = {name: getattr(args, name) for name in DEFAULT_INPUTS}
    manifest = generate(inputs, args.out_dir)
    message = (
        f"loaded={len(manifest['inputs_loaded'])} missing={len(manifest['inputs_missing'])} "
        f"errors={len(manifest['family_errors'])} files={len(manifest['generated_files'])}"
    )
    if manifest["family_errors"]:
        print(f"[FAIL] {message}")
        for family, error in manifest["family_errors"].items():
            print(f"[ERROR] {family}: {error}")
        raise SystemExit(1)
    print(f"[OK] {message}")


if __name__ == "__main__":
    main()
