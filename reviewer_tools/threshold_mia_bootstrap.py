#!/usr/bin/env python3
"""
Evaluate inexpensive scalar threshold MIAs from QuRiFT *_attack_data.pt files.

Known member-score directions:
  loss       -> -loss
  entropy    -> -entropy
  confidence -> +confidence
  margin     -> +margin
  correctness-> +correctness

Reports AUC, bootstrap 95% CI, and TPR at selected FPRs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def stratified_bootstrap_auc(
    y: np.ndarray, score: np.ndarray, n_boot: int, seed: int
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        vals.append(roc_auc_score(y[idx], score[idx]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def tpr_at_fpr(y: np.ndarray, score: np.ndarray, target: float) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    eligible = np.flatnonzero(fpr <= target)
    return float(tpr[eligible[-1]]) if len(eligible) else 0.0


def scalar_scores(payload: Dict) -> Dict[str, np.ndarray]:
    stats = payload["stats"]
    correct = payload.get("correct")
    result = {
        "loss": -torch.as_tensor(stats["loss"]).float().numpy(),
        "entropy": -torch.as_tensor(stats["entropy"]).float().numpy(),
        "confidence": torch.as_tensor(stats["conf"]).float().numpy(),
        "margin": torch.as_tensor(stats["margin"]).float().numpy(),
    }
    if correct is not None:
        result["correctness"] = torch.as_tensor(correct).float().numpy()
    pv = torch.as_tensor(payload["pv"]).float()
    result["max_probability"] = pv.max(dim=1).values.numpy()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack-data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("threshold_mia_results.csv"))
    ap.add_argument("--bootstrap", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    files = sorted(args.attack_data_dir.rglob("*_attack_data.pt"))
    if not files:
        raise SystemExit(f"No *_attack_data.pt files under {args.attack_data_dir}")

    rows = []
    for file_index, path in enumerate(files):
        payload = torch.load(path, map_location="cpu")
        # Stored convention: membership 0=member, 1=non-member.
        membership = torch.as_tensor(payload["membership"]).long().numpy()
        y_member = (membership == 0).astype(int)
        meta = payload.get("meta", {})
        for attack, score in scalar_scores(payload).items():
            auc = float(roc_auc_score(y_member, score))
            lo, hi = stratified_bootstrap_auc(
                y_member, score, n_boot=args.bootstrap, seed=args.seed + file_index
            )
            rows.append({
                "source_file": str(path),
                "attack": attack,
                "auc": auc,
                "auc_ci95_low": lo,
                "auc_ci95_high": hi,
                "tpr_at_fpr_0.001": tpr_at_fpr(y_member, score, 0.001),
                "tpr_at_fpr_0.01": tpr_at_fpr(y_member, score, 0.01),
                "tpr_at_fpr_0.05": tpr_at_fpr(y_member, score, 0.05),
                **{f"meta_{k}": v for k, v in meta.items()},
            })

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[OK] {len(out)} attack rows -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
