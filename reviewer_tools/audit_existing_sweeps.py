#!/usr/bin/env python3
"""
Audit QuRiFT master/extensive CSVs and extract reviewer-facing analyses.

Outputs:
  dataset_inventory.csv
  eff_su2_repetition_integrity.csv
  paired_structural_effects.csv
  matched_gap_pairs.csv
  audit_report.md

The paired analyses deliberately exclude eff_su2 until fm_eff_reps is confirmed
to affect the constructed encoder.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_FILES = {
    "Moons_QNN": "master_results_full_pipeline_moon.csv",
    "Blobs_QNN": "master_results_full_pipeline_blobs.csv",
    "Circles_QNN": "master_results_full_pipeline_circles.csv",
    "MNIST_QNN": "mnist_extensive_results.csv",
    "MNIST_HQNN": "hqnn_extensive_results.csv",
    "MNIST_QCNN": "qcnn_extensive_results.csv",
}

METRICS = [
    "acc_train", "acc_test", "acc_val",
    "loss_train", "loss_val", "test_loss",
]


def normalize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    if "status" in out.columns:
        out = out[out["status"].astype(str).str.lower().eq("ok")].copy()

    aliases = {
        "train_acc": "acc_train",
        "test_acc": "acc_test",
        "qlayer_ent": "ql_ent",
        "qlayer_op": "ql_op",
        "qlayer_rev": "ql_rev",
        "n_qubits": "n_wires",
    }
    for src, dst in aliases.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]

    defaults = {
        "pad_mode": "wrap",
        "fm_ent": "linear",
        "fm_op": "NA",
        "ql_rev": False,
    }
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value
        out[col] = out[col].fillna(value)

    for col in ["run_id", "n_wires", "depth", "reps"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in METRICS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["gap_acc"] = out["acc_train"] - out["acc_test"]
    out["source_label"] = label
    return out


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def config_key_columns(df: pd.DataFrame, exclude: Iterable[str]) -> List[str]:
    candidates = [
        "n_wires", "depth", "reps", "pad_mode", "fm_ent", "fm_op",
        "ql_ent", "ql_op", "ql_rev",
    ]
    excluded = set(exclude)
    return [c for c in candidates if c in df.columns and c not in excluded]


def eff_su2_integrity(df: pd.DataFrame, label: str) -> Dict[str, object]:
    sub = df[df["fm_kind"].astype(str).str.lower().eq("eff_su2")].copy()
    if sub.empty:
        return {
            "source_label": label, "n_eff_rows": 0, "n_rep_groups": 0,
            "groups_identical_across_reps": 0, "fraction_identical": np.nan,
        }

    keys = config_key_columns(sub, exclude={"reps"})
    total = 0
    identical = 0
    for _, g in sub.groupby(keys, dropna=False):
        if g["reps"].nunique(dropna=True) < 2:
            continue
        total += 1
        cols = [m for m in METRICS if m in g.columns]
        if cols and all(g[c].nunique(dropna=False) == 1 for c in cols):
            identical += 1

    return {
        "source_label": label,
        "n_eff_rows": int(len(sub)),
        "n_rep_groups": int(total),
        "groups_identical_across_reps": int(identical),
        "fraction_identical": (identical / total) if total else np.nan,
    }


def paired_effect(
    df: pd.DataFrame,
    label: str,
    variable: str,
    low: int,
    high: int,
    n_boot: int,
    seed: int,
) -> Dict[str, object]:
    # Exclude eff_su2 because current repository code may ignore fm_eff_reps.
    sub = df[df["fm_kind"].astype(str).str.lower().isin(["z", "zz"])].copy()
    sub = sub[sub[variable].isin([low, high])].copy()
    keys = ["fm_kind"] + config_key_columns(sub, exclude={variable})
    rows = []
    for _, g in sub.groupby(keys, dropna=False):
        lo = g[g[variable] == low]
        hi = g[g[variable] == high]
        if len(lo) != 1 or len(hi) != 1:
            continue
        rows.append(float(hi.iloc[0]["gap_acc"] - lo.iloc[0]["gap_acc"]))
    arr = np.asarray(rows, dtype=float)
    mean, lo_ci, hi_ci = bootstrap_mean_ci(arr, n_boot=n_boot, seed=seed)
    return {
        "source_label": label,
        "comparison": f"{variable}:{low}->{high}",
        "feature_maps": "z,zz",
        "n_exact_pairs": int(len(arr)),
        "mean_delta_gap": mean,
        "ci95_low": lo_ci,
        "ci95_high": hi_ci,
        "fraction_positive": float(np.mean(arr > 0)) if len(arr) else np.nan,
        "median_delta_gap": float(np.median(arr)) if len(arr) else np.nan,
    }


def nearest_matched_pairs(
    df: pd.DataFrame,
    label: str,
    train_tol: float,
    test_tol: float,
    gap_tol: float,
) -> pd.DataFrame:
    """
    Vectorized matching of Z and linear-ZZ while holding
    width/depth/reps/padding/ansatz fixed.
    """
    z = df[df["fm_kind"].astype(str).str.lower().eq("z")].copy()
    zz = df[
        df["fm_kind"].astype(str).str.lower().eq("zz")
        & df["fm_ent"].astype(str).str.lower().eq("linear")
    ].copy()

    base_keys = [
        c for c in ["n_wires", "depth", "reps", "pad_mode", "ql_ent", "ql_op", "ql_rev"]
        if c in df.columns
    ]
    value_cols = ["run_id", "acc_train", "acc_test", "gap_acc"]
    merged = z[base_keys + value_cols].merge(
        zz[base_keys + value_cols],
        on=base_keys,
        how="inner",
        suffixes=("_z", "_zz"),
    )
    if merged.empty:
        return pd.DataFrame()

    merged["delta_train_abs"] = (merged["acc_train_zz"] - merged["acc_train_z"]).abs()
    merged["delta_test_abs"] = (merged["acc_test_zz"] - merged["acc_test_z"]).abs()
    merged["delta_gap_abs"] = (merged["gap_acc_zz"] - merged["gap_acc_z"]).abs()
    merged = merged[
        (merged["delta_train_abs"] <= train_tol)
        & (merged["delta_test_abs"] <= test_tol)
        & (merged["delta_gap_abs"] <= gap_tol)
    ].copy()
    if merged.empty:
        return pd.DataFrame()

    merged["distance"] = (
        merged["delta_train_abs"]
        + merged["delta_test_abs"]
        + merged["delta_gap_abs"]
    )
    # Keep the nearest ZZ candidate for each Z configuration.
    merged = merged.sort_values(
        ["run_id_z", "distance", "delta_gap_abs", "delta_test_abs"]
    ).drop_duplicates(["run_id_z"], keep="first")

    out = pd.DataFrame({
        "source_label": label,
        "pair_id": [
            f"{label}_z{int(zid)}_zz{int(zzid)}"
            for zid, zzid in zip(merged["run_id_z"], merged["run_id_zz"])
        ],
        "z_run_id": merged["run_id_z"].astype(int),
        "zz_run_id": merged["run_id_zz"].astype(int),
        "n_wires": merged["n_wires"].astype(int),
        "depth": merged["depth"].astype(int),
        "reps": merged["reps"].astype(int),
        "pad_mode": merged["pad_mode"],
        "ql_ent": merged["ql_ent"],
        "ql_op": merged["ql_op"],
        "ql_rev": merged["ql_rev"].astype(bool),
        "z_train": merged["acc_train_z"].astype(float),
        "zz_train": merged["acc_train_zz"].astype(float),
        "z_test": merged["acc_test_z"].astype(float),
        "zz_test": merged["acc_test_zz"].astype(float),
        "z_gap": merged["gap_acc_z"].astype(float),
        "zz_gap": merged["gap_acc_zz"].astype(float),
        "delta_train_abs": merged["delta_train_abs"].astype(float),
        "delta_test_abs": merged["delta_test_abs"].astype(float),
        "delta_gap_abs": merged["delta_gap_abs"].astype(float),
        "mean_test_acc": ((merged["acc_test_z"] + merged["acc_test_zz"]) / 2).astype(float),
        "mean_gap": ((merged["gap_acc_z"] + merged["gap_acc_zz"]) / 2).astype(float),
    })
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("experiments/gen_results"))
    ap.add_argument("--out-dir", type=Path, default=Path("reviewer_audit"))
    ap.add_argument("--train-tol", type=float, default=0.01)
    ap.add_argument("--test-tol", type=float, default=0.01)
    ap.add_argument("--gap-tol", type=float, default=0.01)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loaded: Dict[str, pd.DataFrame] = {}
    inventory = []
    for label, filename in DEFAULT_FILES.items():
        path = args.data_dir / filename
        if not path.exists():
            print(f"[WARN] Missing {path}")
            continue
        raw = pd.read_csv(path)
        ok = normalize(raw, label)
        loaded[label] = ok
        config_cols = [
            c for c in [
                "fm_kind", "n_wires", "depth", "reps", "pad_mode", "fm_ent",
                "fm_op", "ql_ent", "ql_op", "ql_rev",
            ] if c in ok.columns
        ]
        inventory.append({
            "source_label": label,
            "path": str(path),
            "raw_rows": int(len(raw)),
            "ok_rows": int(len(ok)),
            "unique_configurations": int(ok[config_cols].drop_duplicates().shape[0]),
            "error_rows": int(len(raw) - len(ok)),
        })

    inv_df = pd.DataFrame(inventory)
    inv_df.to_csv(args.out_dir / "dataset_inventory.csv", index=False)

    integrity_rows = [eff_su2_integrity(df, label) for label, df in loaded.items()]
    integrity_df = pd.DataFrame(integrity_rows)
    integrity_df.to_csv(args.out_dir / "eff_su2_repetition_integrity.csv", index=False)

    effect_rows = []
    for i, (label, df) in enumerate(loaded.items()):
        if not label.endswith("_QNN"):
            continue
        effect_rows.append(paired_effect(
            df, label, "reps", 1, 5, args.bootstrap, args.seed + i
        ))
        available_depths = sorted(pd.to_numeric(df["depth"], errors="coerce").dropna().unique())
        if len(available_depths) >= 2:
            effect_rows.append(paired_effect(
                df, label, "depth", int(min(available_depths)), int(max(available_depths)),
                args.bootstrap, args.seed + 100 + i,
            ))
    effects_df = pd.DataFrame(effect_rows)
    effects_df.to_csv(args.out_dir / "paired_structural_effects.csv", index=False)

    pair_frames = []
    for label, df in loaded.items():
        if not label.endswith("_QNN"):
            continue
        pair_frames.append(nearest_matched_pairs(
            df, label, args.train_tol, args.test_tol, args.gap_tol
        ))
    pairs_df = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(
            ["source_label", "delta_gap_abs", "delta_test_abs", "mean_test_acc"],
            ascending=[True, True, True, False],
        )
    pairs_df.to_csv(args.out_dir / "matched_gap_pairs.csv", index=False)

    report = []
    report.append("# QuRiFT sweep audit\n")
    report.append(f"- Loaded successful rows: **{int(inv_df['ok_rows'].sum()) if len(inv_df) else 0:,}**")
    report.append(f"- Exact/near matched Z–ZZ pairs: **{len(pairs_df):,}**")
    if len(integrity_df):
        flagged = integrity_df[
            (integrity_df["n_rep_groups"] > 0)
            & (integrity_df["fraction_identical"] >= 0.99)
        ]
        report.append(f"- eff_su2 datasets with ≥99% repetition-invariant groups: **{len(flagged)}**")
    report.append("\n## Interpretation\n")
    report.append(
        "The paired structural analysis uses only Z and ZZ until the Efficient-SU2 "
        "repetition parameter is verified to alter the constructed encoder. "
        "The matched-pair file is intended to select target models for direct MIA, "
        "not to infer privacy solely from the generalization gap."
    )
    (args.out_dir / "audit_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"[OK] Wrote reviewer audit to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
