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
    "noise_auc": Path("satml_results/noise_budget/combined/noise_budget_auc_summary.csv"),
    "noise_queries": Path("satml_results/noise_budget/combined/equal_total_budget_query_contrasts.csv"),
    "noise_ordering": Path("satml_results/noise_budget/combined/structural_ordering_vs_exact.csv"),
    "noise_utility": Path("satml_results/noise_budget/combined/noise_budget_utility_summary.csv"),
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


def noise_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"target_id", "mode", "queries", "shots", "total_shots", "mean_auc", "sd_across_simulator_seeds",
                "n_simulator_replicates", "calibration_profile"}
    if not required.issubset(frame.columns):
        raise ValueError(f"noise table lacks {sorted(required - set(frame.columns))}")
    selected = frame.copy()
    selected["AUC mean ± simulator-seed SD"] = selected.apply(
        lambda row: f"{row.mean_auc:.3f} ± {row.sd_across_simulator_seeds:.3f}", axis=1
    )
    return selected.rename(
        columns={"mode": "Mode", "queries": "Queries", "shots": "Shots/query",
                 "total_shots": "Total shots", "n_simulator_replicates": "Simulator seeds",
                 "calibration_profile": "Calibration"}
    )[["Mode", "Queries", "Shots/query", "Total shots", "AUC mean ± simulator-seed SD", "Simulator seeds", "Calibration"]]


def noise_utility_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"mode", "queries", "shots", "total_shots", "metric_scope", "metric_name",
                "mean_value", "sd_across_simulator_seeds", "n_simulator_replicates", "calibration_profile"}
    if not required.issubset(frame.columns):
        raise ValueError(f"noise utility table lacks {sorted(required - set(frame.columns))}")
    selected = frame.copy()
    selected["Mean ± simulator-seed SD"] = selected.apply(
        lambda row: (
            f"{row.mean_value:.3f}" if pd.isna(row.sd_across_simulator_seeds)
            else f"{row.mean_value:.3f} ± {row.sd_across_simulator_seeds:.3f}"
        ), axis=1
    )
    return selected.rename(
        columns={"mode": "Mode", "queries": "Queries", "shots": "Shots/query",
                 "total_shots": "Total shots", "metric_scope": "Split", "metric_name": "Utility metric",
                 "n_simulator_replicates": "Simulator seeds", "calibration_profile": "Calibration"}
    )[["Mode", "Queries", "Shots/query", "Total shots", "Split", "Utility metric",
       "Mean ± simulator-seed SD", "Simulator seeds", "Calibration"]]


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


def _noise_plot(frame: pd.DataFrame, output_stem: Path) -> None:
    plot = frame[frame["mode"].isin(["ideal_shot", "noisy_shot"])].copy()
    if plot.empty:
        return
    aggregate = plot.groupby(["mode", "queries", "shots", "total_shots"], as_index=False).agg(
        mean_auc=("mean_auc", "mean"), target_sd=("mean_auc", "std"), n_targets=("target_id", "nunique")
    )
    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    for mode, group in aggregate.groupby("mode"):
        group = group.sort_values("queries")
        label = mode.replace("_", " ")
        axis.errorbar(group.queries, group.mean_auc, yerr=group.target_sd.fillna(0), marker="o", capsize=3, label=label)
    axis.set_xscale("log")
    axis.set_xlabel("Independent queries (equal total-shot budget)")
    axis.set_ylabel("Mean loss-MIA AUC across target checkpoints")
    axis.set_title("Noise and query-budget robustness")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
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
        "noise_auc": ("IBM-calibrated noise and query-budget outcomes", noise_table),
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
        "noise_utility": ("Prediction utility under noise and query budgets", noise_utility_table),
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
    if "noise_auc" in loaded:
        _noise_plot(loaded["noise_auc"], figure_dir / "noise_query_budget")
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
