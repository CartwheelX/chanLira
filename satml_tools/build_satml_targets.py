#!/usr/bin/env python3
"""Generate the frozen SaTML Credit factorial and targeted scaling manifests."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


DATA_SEEDS = (20261, 20262, 20263, 20264, 20265, 20266, 20267, 20268)
MODEL_SEEDS = (30261, 30262, 30263, 30264, 30265, 30266, 30267, 30268)


def base_row(block: int, *, experiment: str) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "dataset": "credit_default",
        "architecture": "qnn",
        "source_run_id": -1,
        "n_wires": 6,
        "pad_mode": "wrap",
        "fm_ent": "linear",
        "ql_ent": "linear",
        "ql_op": "crz",
        "ql_rev": False,
        "vector_train": 200,
        "vector_valid": 200,
        "vector_test": 2000,
        "batch_size": 16,
        "epochs": 100,
        "extra_feats": False,
        "learning_rate": 0.05,
        "weight_decay": 0.0,
        "credit_data_path": "data/credit_default/credit_default.csv.gz",
        "credit_pca_components": 6,
        "block_id": f"credit_b{block:02d}",
        "seed": MODEL_SEEDS[block - 1],
        "model_seed": MODEL_SEEDS[block - 1],
        "init_seed": MODEL_SEEDS[block - 1],
        "data_seed": DATA_SEEDS[block - 1],
        "split_seed": DATA_SEEDS[block - 1],
    }


def factorial_rows(blocks: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in range(1, blocks + 1):
        for feature_map in ("z", "zz", "eff_su2"):
            for repetitions in (1, 5):
                for depth in (2, 6):
                    cell = f"{feature_map}_r{repetitions}_d{depth}"
                    rows.append(
                        {
                            **base_row(block, experiment="satml_credit_factorial"),
                            "target_id": f"CREDIT_QNN_{cell}_b{block:02d}",
                            "role": cell,
                            "structural_cell_id": cell,
                            "fm_kind": feature_map,
                            "fm_op": "cx" if feature_map == "eff_su2" else "NA",
                            "reps": repetitions,
                            "depth": depth,
                            "feature_angle_scale": 1.0,
                        }
                    )
    return rows


def scaling_rows(blocks: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in range(1, blocks + 1):
        for feature_map in ("z", "zz", "eff_su2"):
            for repetitions in (1, 5):
                for scale in (0.5, 2.0):
                    cell = f"{feature_map}_r{repetitions}_d2"
                    scale_tag = str(scale).replace(".", "p")
                    rows.append(
                        {
                            **base_row(block, experiment="satml_credit_scaling"),
                            "target_id": f"CREDIT_SCALE_{cell}_a{scale_tag}_b{block:02d}",
                            "role": f"{cell}_alpha{scale}",
                            "structural_cell_id": cell,
                            "fm_kind": feature_map,
                            "fm_op": "cx" if feature_map == "eff_su2" else "NA",
                            "reps": repetitions,
                            "depth": 2,
                            "feature_angle_scale": scale,
                        }
                    )
    return rows


def geometry_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_map in ("z", "zz", "eff_su2"):
        for repetitions in (1, 5):
            cell = f"{feature_map}_r{repetitions}_d2"
            row = base_row(1, experiment="satml_credit_geometry")
            rows.append(
                {
                    **row,
                    "target_id": f"GEOM_CREDIT_{feature_map}_r{repetitions}",
                    "role": cell,
                    "structural_cell_id": cell,
                    "fm_kind": feature_map,
                    "fm_op": "cx" if feature_map == "eff_su2" else "NA",
                    "reps": repetitions,
                    "depth": 2,
                    "feature_angle_scale": 1.0,
                }
            )
    return rows


def validate(frame: pd.DataFrame, blocks: int, expected_cells: int) -> None:
    if frame["target_id"].duplicated().any():
        raise AssertionError("Target IDs are not unique")
    counts = frame.groupby("block_id")["structural_cell_id"].nunique()
    if len(counts) != blocks or not counts.eq(expected_cells).all():
        raise AssertionError(f"Incomplete paired blocks: {counts.to_dict()}")
    for _, group in frame.groupby("block_id"):
        if group["data_seed"].nunique() != 1 or group["model_seed"].nunique() != 1:
            raise AssertionError("Seeds vary inside a paired block")
        if not (group["data_seed"] == group["split_seed"]).all():
            raise AssertionError("data_seed and split_seed disagree")
        if not (group["model_seed"] == group["init_seed"]).all():
            raise AssertionError("model_seed and init_seed disagree")
    if (frame["vector_test"] < 2000).any():
        raise AssertionError("Low-FPR evaluation requires at least 2,000 nonmembers")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("satml_targets"))
    parser.add_argument("--blocks", type=int, choices=range(5, 9), default=8)
    parser.add_argument("--scaling-blocks", type=int, choices=range(1, 9), default=5)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    factorial = pd.DataFrame(factorial_rows(args.blocks))
    scaling = pd.DataFrame(scaling_rows(args.scaling_blocks))
    geometry = pd.DataFrame(geometry_rows())
    validate(factorial, args.blocks, expected_cells=12)
    validate(scaling, args.scaling_blocks, expected_cells=6)
    factorial_path = args.out_dir / "credit_factorial_targets.csv"
    scaling_path = args.out_dir / "credit_scaling_targets.csv"
    geometry_path = args.out_dir / "credit_geometry_targets.csv"
    factorial.to_csv(factorial_path, index=False)
    scaling.to_csv(scaling_path, index=False)
    geometry.to_csv(geometry_path, index=False)
    print(f"[OK] factorial={len(factorial)} -> {factorial_path}")
    print(f"[OK] scaling={len(scaling)} -> {scaling_path}")
    print(f"[OK] geometry={len(geometry)} -> {geometry_path}")


if __name__ == "__main__":
    main()
