#!/usr/bin/env python3
"""
Build compact, pre-specified reviewer target tables from QuRiFT master CSVs.

Input:
  reviewer_audit/matched_gap_pairs.csv
  the original master/extensive CSVs

Outputs:
  matched_gap_mia_targets.csv
  multiseed_factorial_targets.csv
  architecture_control_targets.csv
  geometry_targets.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


FILE_MAP = {
    "Moons_QNN": "master_results_full_pipeline_moon.csv",
    "Blobs_QNN": "master_results_full_pipeline_blobs.csv",
    "Circles_QNN": "master_results_full_pipeline_circles.csv",
    "MNIST_QNN": "mnist_extensive_results.csv",
    "MNIST_HQNN": "hqnn_extensive_results.csv",
    "MNIST_QCNN": "qcnn_extensive_results.csv",
}


def load_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("ok")].copy()
    if "pad_mode" not in df.columns:
        df["pad_mode"] = "wrap"
    if "fm_ent" not in df.columns:
        df["fm_ent"] = "linear"
    if "fm_op" not in df.columns:
        df["fm_op"] = "NA"
    if "ql_rev" not in df.columns:
        df["ql_rev"] = False
    df["fm_op"] = df["fm_op"].fillna("NA")
    df["gap_acc"] = df["acc_train"] - df["acc_test"]
    return df


def dataset_arch(label: str) -> tuple[str, str]:
    ds, arch = label.split("_", 1)
    return ds, arch


def base_protocol(dataset: str, architecture: str) -> Dict[str, object]:
    # Reviewer subsets use a common protocol within each comparison.
    if dataset == "MNIST":
        return {
            "vector_train": 200, "vector_valid": 200, "vector_test": 200,
            "batch_size": 16, "epochs": 100, "extra_feats": False,
        }
    if dataset == "Circles":
        return {
            "vector_train": 100, "vector_valid": 100, "vector_test": 100,
            "batch_size": 8, "epochs": 100, "extra_feats": True,
        }
    return {
        "vector_train": 50, "vector_valid": 50, "vector_test": 50,
        "batch_size": 8, "epochs": 100, "extra_feats": True,
    }


def config_from_row(row: pd.Series, dataset: str, architecture: str) -> Dict[str, object]:
    p = base_protocol(dataset, architecture)
    return {
        "dataset": dataset,
        "architecture": architecture,
        "source_run_id": int(row["run_id"]),
        "fm_kind": str(row["fm_kind"]),
        "n_wires": int(row["n_wires"]),
        "reps": int(row["reps"]),
        "pad_mode": str(row.get("pad_mode", "wrap")),
        "fm_ent": str(row.get("fm_ent", "linear")),
        "fm_op": str(row.get("fm_op", "NA")),
        "depth": int(row["depth"]),
        "ql_ent": str(row["ql_ent"]),
        "ql_op": str(row["ql_op"]),
        "ql_rev": bool(row.get("ql_rev", False)),
        "original_acc_train": float(row["acc_train"]),
        "original_acc_test": float(row["acc_test"]),
        "original_gap": float(row["gap_acc"]),
        **p,
    }


def choose_low_high_pairs(pairs: pd.DataFrame, per_regime: int) -> pd.DataFrame:
    chosen = []
    for label, g in pairs.groupby("source_label"):
        # Low-gap: accurate and close to zero gap.
        low_pool = g[g["mean_gap"].abs() <= 0.05].copy()
        if low_pool.empty:
            low_pool = g.copy()
        low = low_pool.sort_values(
            ["mean_test_acc", "delta_gap_abs", "delta_test_abs"],
            ascending=[False, True, True],
        ).head(per_regime).assign(regime="low_gap")

        # High-gap: high gap while retaining nontrivial utility.
        test_floor = 0.55
        high_pool = g[g["mean_test_acc"] >= test_floor].copy()
        if high_pool.empty:
            high_pool = g.copy()
        high = high_pool.sort_values(
            ["mean_gap", "mean_test_acc", "delta_gap_abs"],
            ascending=[False, False, True],
        ).head(per_regime).assign(regime="high_gap")
        chosen.extend([low, high])
    return pd.concat(chosen, ignore_index=True).drop_duplicates(
        ["source_label", "z_run_id", "zz_run_id"]
    )


def find_row(df: pd.DataFrame, **kwargs) -> pd.Series:
    hit = df.copy()
    for key, value in kwargs.items():
        if key == "fm_op" and str(value).upper() == "NA":
            hit = hit[hit[key].astype(str).str.upper().isin(["NA", "NAN", "NONE"])]
        else:
            hit = hit[hit[key].astype(str) == str(value)]
    if hit.empty:
        raise RuntimeError(f"No master row matches {kwargs}")
    # Prefer best test accuracy only to obtain a valid source configuration;
    # the rerun protocol and seeds are pre-specified below.
    return hit.sort_values("acc_test", ascending=False).iloc[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("experiments/gen_results"))
    ap.add_argument("--audit-dir", type=Path, default=Path("reviewer_audit"))
    ap.add_argument("--out-dir", type=Path, default=Path("reviewer_targets"))
    ap.add_argument("--seeds", default="43,44,45")
    ap.add_argument("--pairs-per-regime", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    masters = {
        label: load_master(args.data_dir / filename)
        for label, filename in FILE_MAP.items()
        if (args.data_dir / filename).exists()
    }

    # 1) Matched-gap direct MIA targets.
    pairs = pd.read_csv(args.audit_dir / "matched_gap_pairs.csv")
    selected = choose_low_high_pairs(pairs, args.pairs_per_regime)
    long_rows: List[Dict[str, object]] = []
    for _, pair in selected.iterrows():
        label = str(pair["source_label"])
        dataset, arch = dataset_arch(label)
        master = masters[label]
        for fm_name, rid_col in [("z", "z_run_id"), ("zz", "zz_run_id")]:
            rid = int(pair[rid_col])
            row = master[master["run_id"].astype(int) == rid].iloc[0]
            cfg = config_from_row(row, dataset, arch)
            for seed in seeds:
                long_rows.append({
                    "target_id": f"{pair['pair_id']}_{fm_name}_s{seed}",
                    "experiment": "matched_gap_mia",
                    "role": f"{pair['regime']}_{fm_name}",
                    "pair_id": pair["pair_id"],
                    "seed": seed,
                    **cfg,
                })
    pd.DataFrame(long_rows).to_csv(
        args.out_dir / "matched_gap_mia_targets.csv", index=False
    )

    # 2) Small factorial validation: QNN, same ansatz/protocol, varying FM/reps/depth.
    # Use MNIST because the reviewer-facing attack evidence is strongest there.
    factorial_rows = []
    if "MNIST_QNN" in masters:
        master = masters["MNIST_QNN"]
        for fm in ["z", "zz", "eff_su2"]:
            for reps in [1, 5]:
                for depth in [2, 6]:
                    query = dict(
                        fm_kind=fm, n_wires=6, reps=reps, depth=depth,
                        ql_ent="linear", ql_op="crz",
                    )
                    if fm == "eff_su2":
                        query["fm_op"] = "cx"
                    row = find_row(master, **query)
                    cfg = config_from_row(row, "MNIST", "QNN")
                    # Standardize FM-side controls.
                    cfg["pad_mode"] = "wrap"
                    cfg["fm_ent"] = "linear"
                    if fm in {"z", "zz"}:
                        cfg["fm_op"] = "NA"
                    for seed in seeds:
                        factorial_rows.append({
                            "target_id": f"MNIST_QNN_{fm}_r{reps}_d{depth}_s{seed}",
                            "experiment": "multiseed_factorial",
                            "role": f"{fm}_r{reps}_d{depth}",
                            "seed": seed,
                            **cfg,
                        })
    pd.DataFrame(factorial_rows).to_csv(
        args.out_dir / "multiseed_factorial_targets.csv", index=False
    )

    # 3) Architecture control: same data count, epochs, seeds, and structural settings.
    # These are complete-wrapper comparisons, not pure quantum-architecture isolation.
    arch_rows = []
    structural_configs = [
        dict(name="low_reupload", fm_kind="z", n_wires=6, reps=1, depth=2, ql_ent="linear", ql_op="crz", fm_op="NA"),
        dict(name="high_reupload", fm_kind="zz", n_wires=6, reps=5, depth=6, ql_ent="linear", ql_op="crz", fm_op="NA"),
        dict(name="eff_control", fm_kind="eff_su2", n_wires=6, reps=1, depth=2, ql_ent="linear", ql_op="crz", fm_op="cx"),
    ]
    for cfg_spec in structural_configs:
        for arch in ["QNN", "HQNN", "QCNN", "mlp_qnn"]:
            source_label = "MNIST_QNN" if arch == "mlp_qnn" else f"MNIST_{arch}"
            if source_label not in masters:
                continue
            query = {k: v for k, v in cfg_spec.items() if k != "name"}
            row = find_row(masters[source_label], **query)
            cfg = config_from_row(row, "MNIST", arch)
            cfg.update(base_protocol("MNIST", arch))
            cfg["pad_mode"] = "wrap"
            cfg["fm_ent"] = "linear"
            for seed in seeds:
                arch_rows.append({
                    "target_id": f"ARCH_{cfg_spec['name']}_{arch}_s{seed}",
                    "experiment": "architecture_control",
                    "role": cfg_spec["name"],
                    "seed": seed,
                    **cfg,
                })
    pd.DataFrame(arch_rows).to_csv(
        args.out_dir / "architecture_control_targets.csv", index=False
    )

    # 4) Encoder-only geometry targets: no target training required.
    geom_rows = []
    for dataset in ["Moons", "MNIST"]:
        for fm in ["z", "zz", "eff_su2"]:
            for reps in [1, 5]:
                geom_rows.append({
                    "target_id": f"GEOM_{dataset}_{fm}_r{reps}",
                    "dataset": dataset,
                    "architecture": "QNN",
                    "fm_kind": fm,
                    "n_wires": 4 if dataset == "Moons" else 6,
                    "reps": reps,
                    "pad_mode": "wrap",
                    "fm_ent": "linear",
                    "fm_op": "cx" if fm == "eff_su2" else "NA",
                    "depth": 2,
                    "ql_ent": "linear",
                    "ql_op": "crz",
                    "seed": 43,
                })
    pd.DataFrame(geom_rows).to_csv(
        args.out_dir / "geometry_targets.csv", index=False
    )

    print(f"[OK] Wrote target tables to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
