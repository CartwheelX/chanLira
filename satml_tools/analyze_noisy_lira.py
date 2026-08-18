#!/usr/bin/env python3
"""Paired low/high endpoint analysis for SaTML N3 LiRA evidence."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reviewer_tools.reviewer_common import atomic_write_csv, atomic_write_json


LOW_CELL = "eff_su2_r1_d6"
HIGH_CELL = "zz_r5_d6"


def bootstrap(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def stable_seed(seed: int, *parts: object) -> int:
    text = "|".join(str(value) for value in (seed, *parts))
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def normalize_exact(path: Path, target_ids: set[str]) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw[raw.target_id.astype(str).isin(target_ids)].copy()
    output = raw[
        ["target_id", "structural_cell_id", "model_seed", "attack", "auc"]
    ].assign(mode="exact", shots=0)
    output["structural_cell_id"] = output.structural_cell_id.astype(str).str.replace(
        r"_wd[^_]+(?:_block.*)?$", "", regex=True
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--noisy", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    target_table = pd.read_csv(args.targets)
    target_ids = set(target_table.target_id.astype(str))
    noisy = pd.read_csv(args.noisy)
    noisy = noisy[noisy.target_id.astype(str).isin(target_ids)].copy()
    snapshot_hashes = sorted(
        value
        for value in noisy.snapshot_manifest_sha256.dropna().astype(str).unique()
        if value and value.lower() not in {"nan", "none"}
    )
    if len(snapshot_hashes) != 1:
        raise ValueError(
            "N3 analysis requires exactly one frozen snapshot hash; "
            f"observed {snapshot_hashes}"
        )
    noisy["structural_cell_id"] = noisy.structural_cell_id.astype(str).str.replace(
        r"_wd[^_]+(?:_block.*)?$", "", regex=True
    )
    noisy_checkpoint = (
        noisy.groupby(
            ["target_id", "structural_cell_id", "model_seed", "mode", "shots", "attack"],
            dropna=False,
        ).auc.mean().reset_index()
    )
    exact = normalize_exact(args.exact, target_ids)
    combined = pd.concat([exact, noisy_checkpoint], ignore_index=True)
    atomic_write_csv(combined, args.out_dir / "n3_checkpoint_auc.csv")

    rows = []
    for keys, group in combined.groupby(["mode", "shots", "attack"], dropna=False):
        pivot = group.pivot_table(
            index="model_seed", columns="structural_cell_id", values="auc", aggfunc="mean"
        )
        if LOW_CELL not in pivot or HIGH_CELL not in pivot:
            continue
        effect = (pivot[HIGH_CELL] - pivot[LOW_CELL]).dropna()
        if effect.empty:
            continue
        low, high = bootstrap(
            effect.to_numpy(float),
            args.bootstrap,
            stable_seed(args.bootstrap_seed, *keys),
        )
        rows.append(
            {
                "mode": keys[0], "shots": keys[1], "attack": keys[2],
                "contrast": f"{HIGH_CELL} - {LOW_CELL}",
                "n_paired_model_seeds": int(len(effect)),
                "mean_auc_difference": float(effect.mean()),
                "sd_across_model_seeds": float(effect.std(ddof=1)) if len(effect) > 1 else np.nan,
                "ci95_low": low, "ci95_high": high,
                "ci_method": "paired percentile bootstrap over trained model seeds",
            }
        )
    summary = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(summary, args.out_dir / "n3_endpoint_contrasts.csv")
    atomic_write_json(
        {
            "targets": str(args.targets.resolve()),
            "noisy": str(args.noisy.resolve()),
            "exact": str(args.exact.resolve()),
            "endpoint_contrast": f"{HIGH_CELL} - {LOW_CELL}",
            "scope": "selected attack-breadth confirmation, not factorial inference",
            "simulator_seeds_averaged_within_checkpoint": True,
            "snapshot_manifest_sha256": snapshot_hashes[0],
        },
        args.out_dir / "analysis_metadata.json",
    )
    print(f"[OK] N3 LiRA contrasts -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
