#!/usr/bin/env python3
"""Build frozen Fashion-MNIST and WDBC SaTML target manifests."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


FASHION_DATA_SEEDS = (60261, 60262, 60263, 60264, 60265)
FASHION_MODEL_SEEDS = (70261, 70262, 70263, 70264, 70265)
WDBC_DATA_SEEDS = (80261, 80262, 80263, 80264, 80265)
WDBC_MODEL_SEEDS = (90261, 90262, 90263, 90264, 90265)


def common_row(
    block: int,
    *,
    dataset: str,
    experiment: str,
    data_seeds: tuple[int, ...],
    model_seeds: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "dataset": dataset,
        "architecture": "qnn",
        "source_run_id": -1,
        "n_wires": 6,
        "pad_mode": "wrap",
        "fm_ent": "linear",
        "ql_ent": "linear",
        "ql_op": "crz",
        "ql_rev": False,
        "batch_size": 16,
        "epochs": 100,
        "extra_feats": False,
        "learning_rate": 0.05,
        "weight_decay": 0.0,
        "seed": model_seeds[block - 1],
        "model_seed": model_seeds[block - 1],
        "init_seed": model_seeds[block - 1],
        "data_seed": data_seeds[block - 1],
        "split_seed": data_seeds[block - 1],
    }


def fashion_rows(blocks: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in range(1, blocks + 1):
        base = common_row(
            block,
            dataset="fashion_mnist",
            experiment="satml_fashion_factorial",
            data_seeds=FASHION_DATA_SEEDS,
            model_seeds=FASHION_MODEL_SEEDS,
        )
        base.update(
            block_id=f"fashion_b{block:02d}",
            vector_train=200,
            vector_valid=200,
            vector_test=2000,
        )
        for feature_map in ("z", "zz", "eff_su2"):
            for repetitions in (1, 5):
                for depth in (2, 6):
                    cell = f"{feature_map}_r{repetitions}_d{depth}"
                    rows.append(
                        {
                            **base,
                            "target_id": f"FASHION_QNN_{cell}_b{block:02d}",
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


def wdbc_rows(blocks: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in range(1, blocks + 1):
        base = common_row(
            block,
            dataset="breast_cancer_wdbc",
            experiment="satml_wdbc_targeted",
            data_seeds=WDBC_DATA_SEEDS,
            model_seeds=WDBC_MODEL_SEEDS,
        )
        base.update(
            block_id=f"wdbc_b{block:02d}",
            vector_train=160,
            vector_valid=80,
            vector_test=329,
            wdbc_data_path="data/wdbc/wdbc.csv.gz",
            wdbc_pca_components=6,
        )
        for feature_map in ("z", "zz", "eff_su2"):
            for repetitions in (1, 5):
                depth = 2
                cell = f"{feature_map}_r{repetitions}_d{depth}"
                rows.append(
                    {
                        **base,
                        "target_id": f"WDBC_QNN_{cell}_b{block:02d}",
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


def geometry_rows(frame: pd.DataFrame, *, prefix: str, experiment: str) -> pd.DataFrame:
    rows = frame[frame["depth"].eq(2)].drop_duplicates(
        ["fm_kind", "reps"], keep="first"
    ).copy()
    rows["experiment"] = experiment
    rows["target_id"] = [
        f"GEOM_{prefix}_{fm}_r{int(repetitions)}"
        for fm, repetitions in zip(rows["fm_kind"], rows["reps"])
    ]
    return rows.reset_index(drop=True)


def lira_representatives(frame: pd.DataFrame, blocks: int = 3) -> pd.DataFrame:
    selected_blocks = sorted(frame["block_id"].unique())[:blocks]
    return frame[
        frame["block_id"].isin(selected_blocks) & frame["depth"].eq(2)
    ].reset_index(drop=True)


def validate(frame: pd.DataFrame, *, blocks: int, cells: int, expected_test: int) -> None:
    if frame["target_id"].duplicated().any():
        raise AssertionError("Target IDs are not unique")
    grouped = frame.groupby("block_id")
    if len(grouped) != blocks or not grouped["structural_cell_id"].nunique().eq(cells).all():
        raise AssertionError("A paired block is incomplete")
    for _, block in grouped:
        if block["data_seed"].nunique() != 1 or block["model_seed"].nunique() != 1:
            raise AssertionError("Seeds vary within a paired block")
    if not frame["vector_test"].eq(expected_test).all():
        raise AssertionError("Unexpected nonmember evaluation size")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("satml_targets"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fashion = pd.DataFrame(fashion_rows())
    wdbc = pd.DataFrame(wdbc_rows())
    validate(fashion, blocks=5, cells=12, expected_test=2000)
    validate(wdbc, blocks=5, cells=6, expected_test=329)

    outputs = {
        "fashion_factorial_targets.csv": fashion,
        "fashion_geometry_targets.csv": geometry_rows(
            fashion, prefix="FASHION", experiment="satml_fashion_geometry"
        ),
        "fashion_lira_targets.csv": lira_representatives(fashion),
        "wdbc_targeted_targets.csv": wdbc,
        "wdbc_geometry_targets.csv": geometry_rows(
            wdbc, prefix="WDBC", experiment="satml_wdbc_geometry"
        ),
        "wdbc_lira_targets.csv": lira_representatives(wdbc),
    }
    for name, frame in outputs.items():
        path = args.out_dir / name
        frame.to_csv(path, index=False)
        print(f"[OK] {name}: {len(frame)} rows -> {path}")


if __name__ == "__main__":
    main()
