#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# =========================
# INPUT FILES (Linux paths)
# =========================
MNIST_FILES = {
    "QNN":  "experiments/gen_results/qnn_extensive_results.csv",
    "HQNN": "experiments/gen_results/hqnn_extensive_results.csv",
    "QCNN": "experiments/gen_results/qcnn_extensive_results.csv",
}

SYNTHETIC_FILES = {
    "Moons":   "experiments/gen_results/master_results_full_pipeline_moon.csv",
    "Blobs":   "experiments/gen_results/master_results_full_pipeline_blobs.csv",
    "Circles": "experiments/gen_results/master_results_full_pipeline_circles.csv",
}

# MNIST fixed FM settings (as per your sweep)
MNIST_FIXED = {"pad_mode": "wrap", "fm_ent": "linear"}

# This is the CLI flag you described (store_true => present => True)
QL_REV_CLI_FLAG = "--qlayer-ent-wire-reverse"

# Column aliases across CSVs
ALIASES = {
    # acc
    "train_acc": "acc_train",
    "test_acc": "acc_test",
    "accuracy_train": "acc_train",
    "accuracy_test": "acc_test",
    "val_acc": "acc_test",
    "acc_val": "acc_test",

    # widths
    "n_qubits": "n_wires",
    "num_wires": "n_wires",
    "wires": "n_wires",

    # qlayer
    "qlayer_ent": "ql_ent",
    "qlayer_op": "ql_op",
    "qlayer_twoq_op": "ql_op",

    # reverse flag variants seen in different logs
    "qlayer_rev": "ql_rev",
    "qlayer_reverse": "ql_rev",
    "qlayer_ent_wire_reverse": "ql_rev",
    "qlayer_ent_wire_rev": "ql_rev",
    "ql_rev": "ql_rev",

    # padding
    "padding_mode": "pad_mode",
}

BASE_CONFIG_COLS = [
    "fm_kind", "n_wires", "reps", "pad_mode", "fm_ent", "fm_op",
    "depth", "ql_ent", "ql_op", "ql_rev", "run_id"
]


def resolve_existing(p: str) -> Path:
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {p}\nResolved: {path}\nCWD: {Path.cwd()}")
    return path


def apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for src, dst in ALIASES.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    return df


def ensure_cols(df: pd.DataFrame, cols: List[str], fill: str = "NA") -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df


def to_bool(x) -> bool:
    """Robust boolean parsing for CSV values."""
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def normalize_schema(df: pd.DataFrame, is_mnist: bool) -> pd.DataFrame:
    df = apply_aliases(df)
    df = ensure_cols(df, BASE_CONFIG_COLS + ["acc_train", "acc_test"])

    if is_mnist:
        df["pad_mode"] = MNIST_FIXED["pad_mode"]
        df["fm_ent"] = MNIST_FIXED["fm_ent"]

    # numeric
    df["acc_train"] = pd.to_numeric(df["acc_train"], errors="coerce")
    df["acc_test"]  = pd.to_numeric(df["acc_test"], errors="coerce")
    df["gap_acc"]   = df["acc_train"] - df["acc_test"]

    # normalize strings
    for c in ["fm_kind", "pad_mode", "fm_ent", "fm_op", "ql_ent", "ql_op"]:
        df[c] = df[c].astype(str).replace({"nan": "NA", "None": "NA", "none": "NA", "": "NA"})

    # reps/depth/n_wires as numeric-ish for matching
    for c in ["reps", "depth", "n_wires"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # ql_rev as bool (default False if missing/NA)
    df["ql_rev"] = df["ql_rev"].apply(to_bool)

    # derive effective fm op
    fm_op_eff = []
    for k, op in zip(df["fm_kind"].tolist(), df["fm_op"].tolist()):
        k_low = str(k).strip().lower()
        op_low = str(op).strip().lower()
        if k_low == "zz":
            fm_op_eff.append("rzz" if op_low in {"na", "nan", "none", ""} else str(op))
        elif k_low == "z":
            fm_op_eff.append("NA")
        elif k_low == "eff_su2":
            fm_op_eff.append("NA" if op_low in {"na", "nan", "none", ""} else str(op))
        else:
            fm_op_eff.append("NA" if op_low in {"na", "nan", "none", ""} else str(op))
    df["fm_op_eff"] = fm_op_eff

    return df


def parse_kv_list(s: str) -> Dict[str, str]:
    """
    --fix "n_wires=8,ql_ent=full,ql_op=crz,fm_kind=zz,fm_op_eff=rzz,ql_rev=true"
    """
    out: Dict[str, str] = {}
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad --fix item: {p} (expected key=value)")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    vals: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    return vals


def gap_distance(g: float, lo: float, hi: float) -> float:
    if pd.isna(g):
        return 1e9
    if lo <= g <= hi:
        return 0.0
    return (lo - g) if g < lo else (g - hi)


def build_config_string(r: pd.Series) -> str:
    return (
        f"fm_kind={r.get('fm_kind','NA')}, n_wires={r.get('n_wires','NA')}, reps={r.get('reps','NA')}, "
        f"pad_mode={r.get('pad_mode','NA')}, fm_ent={r.get('fm_ent','NA')}, fm_op_eff={r.get('fm_op_eff','NA')}, "
        f"depth={r.get('depth','NA')}, ql_ent={r.get('ql_ent','NA')}, ql_op={r.get('ql_op','NA')}, "
        f"ql_rev={bool(r.get('ql_rev', False))}"
    )


def select_one_for_cell(
    cell_df: pd.DataFrame,
    train_min: float,
    gap_lo: float,
    gap_hi: float,
    prefer_low_test: bool,
    fallback_if_empty: bool
) -> Tuple[Optional[pd.Series], str, int]:
    """
    Returns: (selected_row or None, reason, n_candidates)
    Strategy:
      - primary: acc_train>=train_min
      - rank: acc_train desc, gap_dist asc, (optional) acc_test asc, gap_acc desc
      - fallback: if no candidates and fallback_if_empty=True, ignore train_min but keep ranking.
    """
    df = cell_df.dropna(subset=["run_id", "acc_train", "acc_test", "gap_acc"]).copy()
    n_all = int(len(df))
    if df.empty:
        return None, "no_candidates", 0

    df["gap_dist"] = df["gap_acc"].apply(lambda g: gap_distance(g, gap_lo, gap_hi))

    prim = df[df["acc_train"] >= float(train_min)].copy()

    used = prim
    reason = "primary_train_filtered"
    if used.empty:
        if not fallback_if_empty:
            return None, "no_candidates_after_train_filter", n_all
        used = df.copy()
        reason = "fallback_ignored_train_min"

    sort_cols = ["acc_train", "gap_dist"]
    asc = [False, True]
    if prefer_low_test:
        sort_cols.append("acc_test")
        asc.append(True)
    sort_cols.append("gap_acc")
    asc.append(False)

    used = used.sort_values(sort_cols, ascending=asc)
    return used.iloc[0], reason, n_all


def main():
    ap = argparse.ArgumentParser(
        description="Build a reps×depth GRID run_id table (one run per cell) for retraining + MIA."
    )
    ap.add_argument("--dataset", required=True, choices=["MNIST", "Moons", "Blobs", "Circles"])
    ap.add_argument("--arch", required=True, help="MNIST: QNN/HQNN/QCNN; Synthetic: QNN")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fix", default="",
                    help='Fixed knobs, e.g. "n_wires=8,ql_ent=full,ql_op=crz,fm_kind=zz,fm_op_eff=rzz,ql_rev=false"')
    ap.add_argument("--reps", default=None, help="Comma list reps to include (e.g. 1,2,3,4,5). Default: all present.")
    ap.add_argument("--depths", default=None, help="Comma list depths to include (e.g. 2,4,6). Default: all present.")
    ap.add_argument("--train-min", type=float, default=0.99)
    ap.add_argument("--gap-lo", type=float, default=0.25)
    ap.add_argument("--gap-hi", type=float, default=0.30)
    ap.add_argument("--prefer-low-test", action="store_true", help="Tie-break by lower test (more memorization).")
    ap.add_argument("--fallback-if-empty", action="store_true",
                    help="If a cell has no acc_train>=train_min, pick best available anyway (marked as fallback).")
    args = ap.parse_args()

    # pick file
    if args.dataset == "MNIST":
        if args.arch not in MNIST_FILES:
            raise ValueError(f"MNIST arch must be one of {list(MNIST_FILES.keys())}")
        csv_path = MNIST_FILES[args.arch]
        is_mnist = True
        dataset_label = "MNIST"
        arch_label = args.arch
    else:
        if args.arch != "QNN":
            raise ValueError("Synthetic datasets only support arch=QNN in your sweep.")
        csv_path = SYNTHETIC_FILES[args.dataset]
        is_mnist = False
        dataset_label = args.dataset
        arch_label = "QNN"

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(resolve_existing(csv_path))
    df = normalize_schema(df, is_mnist=is_mnist)

    # apply fixed constraints
    fixes = parse_kv_list(args.fix)
    for k, v in fixes.items():
        if k not in df.columns:
            raise ValueError(f"--fix key '{k}' not in columns. Available: {list(df.columns)[:80]}")
        if k in {"n_wires", "reps", "depth"}:
            df = df[pd.to_numeric(df[k], errors="coerce") == int(v)]
        elif k == "ql_rev":
            want = to_bool(v)
            df = df[df["ql_rev"] == want]
        else:
            df = df[df[k].astype(str) == str(v)]

    if df.empty:
        raise RuntimeError("No rows left after applying --fix constraints.")

    # reps/depth grid to cover
    reps_list = parse_int_list(args.reps)
    depths_list = parse_int_list(args.depths)

    if reps_list is None:
        reps_list = sorted([int(x) for x in df["reps"].dropna().unique().tolist()])
    if depths_list is None:
        depths_list = sorted([int(x) for x in df["depth"].dropna().unique().tolist()])

    if not reps_list or not depths_list:
        raise RuntimeError("No reps/depth values found after filters.")

    # build table: one row per (reps, depth)
    rows = []
    for r in reps_list:
        for d in depths_list:
            cell = df[(df["reps"] == r) & (df["depth"] == d)].copy()

            sel, reason, n_cand = select_one_for_cell(
                cell_df=cell,
                train_min=float(args.train_min),
                gap_lo=float(args.gap_lo),
                gap_hi=float(args.gap_hi),
                prefer_low_test=bool(args.prefer_low_test),
                fallback_if_empty=bool(args.fallback_if_empty),
            )

            if sel is None:
                rows.append({
                    "run_id": "NA",
                    "dataset": dataset_label,
                    "architecture": arch_label,
                    "reps": r,
                    "depth": d,
                    "acc_train": float("nan"),
                    "acc_test": float("nan"),
                    "gap_acc": float("nan"),
                    "reason": reason,
                    "n_candidates_in_cell": n_cand,
                    "ql_rev": False,
                    "ql_rev_flag": "",
                    "config": "NA",
                })
            else:
                ql_rev_bool = bool(sel.get("ql_rev", False))
                rows.append({
                    "run_id": sel.get("run_id", "NA"),
                    "dataset": dataset_label,
                    "architecture": arch_label,
                    "reps": int(sel.get("reps")),
                    "depth": int(sel.get("depth")),
                    "acc_train": float(sel.get("acc_train")),
                    "acc_test": float(sel.get("acc_test")),
                    "gap_acc": float(sel.get("gap_acc")),
                    "reason": reason,
                    "n_candidates_in_cell": n_cand,
                    "fm_kind": sel.get("fm_kind", "NA"),
                    "fm_op_eff": sel.get("fm_op_eff", "NA"),
                    "n_wires": sel.get("n_wires", "NA"),
                    "ql_ent": sel.get("ql_ent", "NA"),
                    "ql_op": sel.get("ql_op", "NA"),
                    "ql_rev": ql_rev_bool,
                    "ql_rev_flag": (QL_REV_CLI_FLAG if ql_rev_bool else ""),
                    "pad_mode": sel.get("pad_mode", "NA"),
                    "fm_ent": sel.get("fm_ent", "NA"),
                    "fm_op": sel.get("fm_op", "NA"),
                    "config": build_config_string(sel),
                })

    out_df = pd.DataFrame(rows).sort_values(["reps", "depth"])

    base = out_dir / f"runid_grid_{dataset_label}_{arch_label}"
    out_df.to_csv(base.with_suffix(".csv"), index=False)
    out_df.to_latex(base.with_suffix(".tex"), index=False, escape=False)

    print("\n✅ Wrote grid run_id table:")
    print(f"  {base.with_suffix('.csv')}")
    print(f"  {base.with_suffix('.tex')}")
    print("\nGrid:")
    print(f"  reps={reps_list}")
    print(f"  depths={depths_list}")
    if fixes:
        print(f"\nFixed constraints: {fixes}")
    missing = int((out_df["run_id"] == "NA").sum())
    if missing > 0:
        print(f"\n⚠️ Missing cells: {missing} (run_id=NA). Consider relaxing --fix or using --fallback-if-empty.")


if __name__ == "__main__":
    main()
