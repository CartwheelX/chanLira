#!/usr/bin/env python3
"""Five-fold cross-fitted learned MIA for frozen noisy-output payloads."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reviewer_tools.reviewer_common import (
    atomic_write_csv,
    tpr_at_resolvable_fpr,
)
from reviewer_tools.qurift_noisy_eval import PAYLOAD_SCHEMA_VERSION


def load_payload(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    version = int(payload.get("payload_schema_version", 0))
    if version != PAYLOAD_SCHEMA_VERSION:
        raise ValueError(f"Unsupported noisy payload schema {version}: {path}")
    return payload


def attack_features(payload: dict, feature_mode: str) -> np.ndarray:
    pv = torch.as_tensor(payload["pv"]).float()
    if feature_mode == "pv":
        values = pv
    elif feature_mode == "pv+stats":
        values = torch.as_tensor(payload["X"]).float()
    elif feature_mode == "pv_mean_std":
        query = payload.get("query_pv")
        if query is None:
            std = torch.zeros_like(pv)
        else:
            query = torch.as_tensor(query).float()
            if query.ndim != 3 or query.shape[1:] != pv.shape:
                raise ValueError(
                    f"query_pv shape {tuple(query.shape)} is incompatible with pv {tuple(pv.shape)}"
                )
            std = query.std(dim=0, unbiased=False)
        values = torch.cat([pv, std], dim=1)
    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")
    output = values.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(output).all():
        raise ValueError("Learned-MIA features contain non-finite values")
    return output


def cross_fitted_scores(
    features: np.ndarray,
    membership: np.ndarray,
    *,
    folds: int,
    split_seed: int,
    attacker_seed: int,
) -> np.ndarray:
    membership = np.asarray(membership, dtype=int)
    if set(np.unique(membership)) != {0, 1}:
        raise ValueError("Membership labels must contain both 0 and 1")
    minimum = int(np.bincount(membership).min())
    folds = min(int(folds), minimum)
    if folds < 2:
        raise ValueError("At least two records per membership class are required")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(split_seed))
    scores = np.full(len(membership), np.nan, dtype=float)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(np.zeros(len(membership)), membership)
    ):
        scaler = StandardScaler().fit(features[train_index])
        train_x = scaler.transform(features[train_index])
        test_x = scaler.transform(features[test_index])
        classifier = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=min(32, len(train_index)),
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=int(attacker_seed) + fold,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            classifier.fit(train_x, membership[train_index])
        member_column = int(np.flatnonzero(classifier.classes_ == 1)[0])
        scores[test_index] = classifier.predict_proba(test_x)[:, member_column]
    if not np.isfinite(scores).all():
        raise RuntimeError("Cross-fitting did not produce one finite score per record")
    return scores


def stable_offset(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def evaluate_payload(
    path: Path,
    *,
    feature_mode: str,
    folds: int,
    split_seed: int,
    attacker_seed: int,
) -> dict:
    payload = load_payload(path)
    metadata = dict(payload.get("meta", {}) or {})
    membership = torch.as_tensor(payload["membership"]).long().numpy().reshape(-1)
    convention = str(metadata.get("membership_convention", "")).lower()
    if not convention.startswith("1=member"):
        raise ValueError(f"Noisy payload must use 1=member convention: {path}")
    features = attack_features(payload, feature_mode)
    if len(features) != len(membership):
        raise ValueError(f"Feature/label length mismatch in {path}")
    scores = cross_fitted_scores(
        features,
        membership,
        folds=folds,
        split_seed=split_seed,
        attacker_seed=attacker_seed + stable_offset(str(path)) % 100_000,
    )
    auc = float(roc_auc_score(membership, scores))
    prediction = (scores >= 0.5).astype(int)
    tpr5, attained5 = tpr_at_resolvable_fpr(membership, scores, 0.05)
    tpr10, attained10 = tpr_at_resolvable_fpr(membership, scores, 0.10)
    return {
        "target_id": metadata.get("target_id", path.parent.parent.name),
        "structural_cell_id": metadata.get("structural_cell_id", ""),
        "fm_kind": metadata.get("fm_kind", ""),
        "reps": metadata.get("reps", 0),
        "depth": metadata.get("depth", 0),
        "model_seed": metadata.get("model_seed", 0),
        "data_seed": metadata.get("data_seed", 0),
        "mode": metadata.get("mode", ""),
        "queries": metadata.get("queries", 0),
        "shots": metadata.get("shots", 0),
        "total_shots": metadata.get("total_shots", 0),
        "simulator_seed": metadata.get("simulator_seed", -1),
        "aggregation": metadata.get("aggregation", metadata.get("api_aggregation", "")),
        "calibration_timestamp": metadata.get("calibration_timestamp"),
        "snapshot_manifest_sha256": metadata.get("snapshot_manifest_sha256"),
        "attack": f"learned_mlp_{feature_mode.replace('+', '_plus_')}",
        "feature_mode": feature_mode,
        "auc": auc,
        "balanced_accuracy_at_0p5": float(
            balanced_accuracy_score(membership, prediction)
        ),
        "tpr_at_0_05_fpr": float(tpr5),
        "attained_fpr_for_0_05": float(attained5),
        "tpr_at_0_10_fpr": float(tpr10),
        "attained_fpr_for_0_10": float(attained10),
        "n_member": int((membership == 1).sum()),
        "n_nonmember": int((membership == 0).sum()),
        "folds": int(folds),
        "split_seed": int(split_seed),
        "attacker_seed": int(attacker_seed),
        "attack_training_access": (
            "labeled target-output auxiliary records; every reported score is "
            "out-of-fold for the learned attacker"
        ),
        "source_file": str(path.resolve()),
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "target_id", "structural_cell_id", "fm_kind", "reps", "depth",
        "model_seed", "data_seed", "mode", "queries", "shots", "total_shots",
        "aggregation", "attack", "feature_mode", "calibration_timestamp",
        "snapshot_manifest_sha256",
    ]
    return (
        raw.groupby(keys, dropna=False).auc.agg(["count", "mean", "std"])
        .reset_index()
        .rename(
            columns={
                "count": "n_simulator_replicates",
                "mean": "mean_auc",
                "std": "sd_across_simulator_seeds",
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature-modes", default="pv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--attacker-seed", type=int, default=41)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-payloads", type=int, default=None)
    args = parser.parse_args()
    paths = sorted(args.root.rglob("*.pt"))
    paths = [path for path in paths if path.parent.name == "payloads"]
    if args.max_payloads is not None:
        paths = paths[: args.max_payloads]
    if not paths:
        raise FileNotFoundError(f"No condition payloads below {args.root}")
    modes = [value.strip() for value in args.feature_modes.split(",") if value.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "learned_mia_raw.csv"
    rows: list[dict] = []
    completed: set[tuple[str, str]] = set()
    if args.resume and raw_path.is_file():
        existing = pd.read_csv(raw_path)
        rows = existing.to_dict("records")
        completed = set(zip(existing.source_file.astype(str), existing.feature_mode.astype(str)))
    total = len(paths) * len(modes)
    done = len(completed)
    for path in paths:
        for feature_mode in modes:
            key = (str(path.resolve()), feature_mode)
            if key in completed:
                continue
            row = evaluate_payload(
                path,
                feature_mode=feature_mode,
                folds=args.folds,
                split_seed=args.split_seed,
                attacker_seed=args.attacker_seed,
            )
            rows.append(row)
            completed.add(key)
            done += 1
            atomic_write_csv(pd.DataFrame(rows), raw_path)
            print(
                f"[{done}/{total}] {row['target_id']} {row['mode']} "
                f"q={row['queries']} shots={row['shots']} {row['attack']} "
                f"auc={row['auc']:.3f}",
                flush=True,
            )
    raw = pd.DataFrame(rows)
    atomic_write_csv(raw, raw_path)
    atomic_write_csv(summarize(raw), args.out_dir / "learned_mia_summary.csv")
    print(f"[OK] learned noisy MIA rows={len(raw)} -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
