#!/usr/bin/env python3
"""Build the predeclared compact noisy finite-shot target subset.

The selection is based on the existing noiseless sweep before any noisy outcomes
are inspected. It deliberately includes all three feature-map families, clear
low/medium/high risk anchors, and matched repetition controls for ZZ and the
corrected Efficient-SU2 encoder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


CELL_SPECS: List[Dict[str, object]] = [
    {
        "structural_cell_id": "eff_su2_r1_d2",
        "risk_role": "low_risk_anchor",
        "selection_order": 1,
        "comparison_group": "eff_su2_repetition_d2",
        "selection_reason": (
            "Lowest-risk encoder-family anchor in the pre-specified MNIST factorial: "
            "Efficient-SU2, one repetition, shallow depth, original sweep gap 0.155."
        ),
    },
    {
        "structural_cell_id": "eff_su2_r5_d2",
        "risk_role": "corrected_eff_su2_repetition_control",
        "selection_order": 2,
        "comparison_group": "eff_su2_repetition_d2",
        "selection_reason": (
            "Corrected Efficient-SU2 r5 control paired with r1 at the same width, depth, "
            "ansatz and data split. Historical r5 sweep values are not used as evidence "
            "because the old repetition argument was ignored."
        ),
    },
    {
        "structural_cell_id": "z_r1_d6",
        "risk_role": "medium_risk_z_anchor",
        "selection_order": 3,
        "comparison_group": "encoder_family_r1_d6",
        "selection_reason": (
            "Representative Z encoder at one repetition and depth six; original sweep "
            "gap 0.270, providing a middle-risk encoder-family anchor."
        ),
    },
    {
        "structural_cell_id": "zz_r1_d6",
        "risk_role": "medium_risk_zz_repetition_baseline",
        "selection_order": 4,
        "comparison_group": "zz_repetition_d6",
        "selection_reason": (
            "ZZ repetition baseline matched to the high-risk r5 setting on all listed "
            "structural factors; original sweep gap 0.275."
        ),
    },
    {
        "structural_cell_id": "zz_r5_d6",
        "risk_role": "high_risk_anchor",
        "selection_order": 5,
        "comparison_group": "zz_repetition_d6",
        "selection_reason": (
            "High-risk repeated-ZZ anchor from the pre-specified MNIST factorial; "
            "original sweep gap 0.425, the largest among the selected cells."
        ),
    },
]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factorial-targets",
        type=Path,
        default=Path("reviewer_targets/multiseed_factorial_targets.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("reviewer_targets"))
    parser.add_argument("--model-seeds", default="43,44,45")
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.model_seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one model seed is required")

    source = pd.read_csv(args.factorial_targets)
    required = {
        "target_id", "experiment", "dataset", "architecture", "fm_kind", "reps",
        "depth", "n_wires", "model_seed", "data_seed", "structural_cell_id",
        "original_acc_train", "original_acc_test", "original_gap",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Factorial target table is missing columns: {sorted(missing)}")

    selected_parts = []
    rationale_rows = []
    for spec in CELL_SPECS:
        cell = str(spec["structural_cell_id"])
        rows = source[
            (source["structural_cell_id"].astype(str) == cell)
            & (source["model_seed"].astype(int).isin(seeds))
        ].copy()
        found_seeds = sorted(rows["model_seed"].astype(int).unique().tolist())
        if found_seeds != sorted(seeds):
            raise ValueError(
                f"Cell {cell!r} has model seeds {found_seeds}, expected {sorted(seeds)}"
            )

        for key, value in spec.items():
            rows[key] = value
        rows["selection_basis"] = "predeclared_from_existing_noiseless_sweep_before_noisy_outcomes"
        rows["old_eff_su2_repetition_rows_used_as_evidence"] = False
        rows["noise_study_scope"] = "backend_derived_aer_sanity_check_not_hardware"
        rows["primary_subset"] = True
        selected_parts.append(rows)

        first = rows.iloc[0]
        rationale_rows.append(
            {
                **spec,
                "dataset": first["dataset"],
                "architecture": first["architecture"],
                "fm_kind": first["fm_kind"],
                "reps": int(first["reps"]),
                "depth": int(first["depth"]),
                "n_wires": int(first["n_wires"]),
                "original_acc_train": float(first["original_acc_train"]),
                "original_acc_test": float(first["original_acc_test"]),
                "original_gap": float(first["original_gap"]),
                "model_seeds": ",".join(str(seed) for seed in seeds),
                "mia_selection_note": (
                    "The prior matched threshold-MIA study supports using risk tiers overall, "
                    "but no noisy outcome or target-specific MNIST MIA was used to select this cell."
                ),
            }
        )

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values(["selection_order", "model_seed"]).reset_index(drop=True)
    core_seed = min(seeds)
    core = selected[selected["model_seed"].astype(int) == core_seed].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(pd.DataFrame(rationale_rows), args.out_dir / "noisy_sanity_selection_rationale.csv")
    atomic_csv(core, args.out_dir / "noisy_sanity_targets_core.csv")
    atomic_csv(selected, args.out_dir / "noisy_sanity_targets_all_seeds.csv")

    metadata = {
        "selection_is_predeclared": True,
        "selection_time_relative_to_noisy_results": "before",
        "core_model_seed": core_seed,
        "all_model_seeds": seeds,
        "n_structural_cells": len(CELL_SPECS),
        "n_core_targets": int(len(core)),
        "n_all_seed_targets": int(len(selected)),
        "feature_maps": sorted(selected["fm_kind"].astype(str).unique().tolist()),
        "repetitions": sorted(selected["reps"].astype(int).unique().tolist()),
        "risk_roles": selected["risk_role"].drop_duplicates().tolist(),
        "important_limitation": (
            "Historical Efficient-SU2 r5 metrics were produced before the repetition bug was fixed; "
            "they are not used to assign risk. The r5 cell is included only as a corrected, predeclared control."
        ),
    }
    (args.out_dir / "noisy_sanity_selection_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"[OK] Core targets: {len(core)} -> {args.out_dir / 'noisy_sanity_targets_core.csv'}")
    print(f"[OK] All-seed targets: {len(selected)} -> {args.out_dir / 'noisy_sanity_targets_all_seeds.csv'}")
    print(f"[OK] Rationale: {args.out_dir / 'noisy_sanity_selection_rationale.csv'}")


if __name__ == "__main__":
    main()
