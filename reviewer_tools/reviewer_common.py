#!/usr/bin/env python3
"""Shared, dependency-light helpers for the QuRiFT reviewer toolkit."""
from __future__ import annotations

import hashlib
import json
import random
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

CI_RECORD = "stratified percentile bootstrap over member/non-member records"
CI_MATCHED_PAIR = (
    "cluster percentile bootstrap over matched-pair IDs, preserving all target seeds and Z/ZZ pairing"
)
CI_STRUCTURAL = (
    "hierarchical percentile bootstrap over structural cells with target seeds nested"
)
CI_DATA_SEED = (
    "paired hierarchical percentile bootstrap over dataset/encoder blocks with data seeds nested"
)


def torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except Exception:
        pass


def stable_seed(*parts: Any, modulo: int = 2_147_483_647) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % modulo


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off", "", "nan", "none"}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        if hasattr(value, "numel") and value.numel() == 1:
            return float(value.detach().cpu().item())
        return float(value)
    except Exception:
        return default


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".csv.tmp",
        delete=False,
        dir=path.parent,
        encoding="utf-8",
        newline="",
    ) as handle:
        tmp = Path(handle.name)
        df.to_csv(handle, index=False)
    tmp.replace(path)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".json.tmp",
        delete=False,
        dir=path.parent,
        encoding="utf-8",
    ) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    tmp.replace(path)


def write_analysis_metadata(
    path: Path,
    *,
    script: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    ci_method: str,
    bootstrap_unit: str,
    bootstrap_replicates: int,
    error_bar_type: str = "",
    notes: str = "",
) -> None:
    atomic_write_json(
        {
            "script": script,
            "inputs": list(inputs),
            "outputs": list(outputs),
            "ci_method": ci_method,
            "bootstrap_unit": bootstrap_unit,
            "bootstrap_replicates": int(bootstrap_replicates),
            "error_bar_type": error_bar_type,
            "notes": notes,
        },
        path,
    )


def find_attack_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    found: set[Path] = set()
    for pattern in ("*attack_data*.pt", "target_attack_data.pt"):
        found.update(path for path in root.rglob(pattern) if path.is_file())
    return sorted(found)


def find_nested(mapping: Mapping[str, Any], paths: Iterable[Tuple[str, ...]]) -> Any:
    for path in paths:
        current: Any = mapping
        valid = True
        for key in path:
            if isinstance(current, Mapping) and key in current:
                current = current[key]
            else:
                valid = False
                break
        if valid:
            return current
    return None


def split_metric(payload: Mapping[str, Any], split: str, metric: str) -> float:
    aliases = {
        "acc": ("acc", "accuracy", "correct_acc"),
        "loss": ("loss", "nll", "ce", "cross_entropy"),
    }[metric]
    roots = ("target_metrics", "metrics", "target_eval", "eval_metrics")
    split_aliases = (split, "val" if split == "valid" else split)
    paths: list[Tuple[str, ...]] = []
    for root in roots:
        for split_name in split_aliases:
            for alias in aliases:
                paths.extend(
                    ((root, split_name, alias), (root, f"{split_name}_{alias}"))
                )
    for split_name in split_aliases:
        for alias in aliases:
            paths.extend(
                ((f"{split_name}_{alias}",), (f"target_{split_name}_{alias}",))
            )
    return safe_float(find_nested(payload, paths))


def flatten_scalar_meta(meta: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            output[f"{prefix}{key}"] = value
        elif hasattr(value, "numel") and value.numel() == 1:
            output[f"{prefix}{key}"] = value.item()
    return output


def membership_convention(payload: Mapping[str, Any]) -> str:
    meta = payload.get("meta", {}) or {}
    declared = str(meta.get("membership_convention", "")).strip().lower()
    if declared in {"1=member", "member_is_1", "one_is_member"}:
        return "1=member"
    if declared in {"0=member", "member_is_0", "zero_is_member"}:
        return "0=member"

    raw = payload.get("membership", payload.get("member", payload.get("is_member")))
    split = payload.get("split")
    if raw is not None and split is not None:
        import torch

        membership = torch.as_tensor(raw).detach().cpu().numpy().astype(int).reshape(-1)
        split_values = torch.as_tensor(split).detach().cpu().numpy().astype(int).reshape(-1)
        if len(membership) == len(split_values):
            train_values = membership[split_values == 0]
            test_values = membership[split_values == 1]
            if len(train_values) and len(test_values):
                if np.all(train_values == 0) and np.all(test_values == 1):
                    return "0=member"
                if np.all(train_values == 1) and np.all(test_values == 0):
                    return "1=member"
    # QuRiFT's current export writes train/member=0 and test/non-member=1.
    return "0=member"


def normalize_membership(payload: Mapping[str, Any]) -> np.ndarray:
    import torch

    raw = payload.get("membership", payload.get("member", payload.get("is_member")))
    if raw is None:
        raise KeyError("Attack payload has no membership/member/is_member field")
    values = torch.as_tensor(raw).detach().cpu().numpy().astype(int).reshape(-1)
    convention = membership_convention(payload)
    return (values == (1 if convention == "1=member" else 0)).astype(int)


def scalar_attack_scores(payload: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    import torch

    stats = payload.get("stats", {}) or {}
    output: Dict[str, np.ndarray] = {}
    aliases = {
        "loss": ("loss",),
        "entropy": ("entropy",),
        "confidence": ("conf", "confidence"),
        "margin": ("margin",),
    }
    signs = {"loss": -1.0, "entropy": -1.0, "confidence": 1.0, "margin": 1.0}
    for name, keys in aliases.items():
        value = next((stats[key] for key in keys if key in stats), None)
        if value is not None:
            output[name] = (
                signs[name]
                * torch.as_tensor(value).float().detach().cpu().numpy().reshape(-1)
            )
    correct = payload.get("correct", stats.get("correct", stats.get("correctness")))
    if correct is not None:
        output["correctness"] = (
            torch.as_tensor(correct).float().detach().cpu().numpy().reshape(-1)
        )
    probabilities = payload.get("pv", payload.get("probs", payload.get("probabilities")))
    if probabilities is not None:
        probs = torch.as_tensor(probabilities).float().detach().cpu()
        output["max_probability"] = probs.max(dim=1).values.numpy()
    return output


def stratified_bootstrap_metric(
    y: np.ndarray,
    score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int,
    seed: int,
) -> Tuple[float, float, int]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    members = np.flatnonzero(y == 1)
    nonmembers = np.flatnonzero(y == 0)
    if len(members) == 0 or len(nonmembers) == 0:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        index = np.concatenate(
            [
                rng.choice(members, len(members), replace=True),
                rng.choice(nonmembers, len(nonmembers), replace=True),
            ]
        )
        try:
            value = float(metric_fn(y[index], score[index]))
            if np.isfinite(value):
                values.append(value)
        except Exception:
            continue
    if not values:
        return float("nan"), float("nan"), 0
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        len(values),
    )


def stratified_bootstrap_auc(
    y: np.ndarray,
    score: np.ndarray,
    n_boot: int,
    seed: int,
    chunk_size: int = 2048,
) -> Tuple[float, float, int]:
    """Exact stratified record bootstrap for binary ROC-AUC.

    Sampling record indices with replacement is equivalent to drawing score-bin
    counts from a multinomial distribution.  Computing AUC from those counts
    avoids one Python/scikit-learn call per bootstrap replicate while preserving
    the bootstrap distribution, including the usual half credit for score ties.
    """
    y = np.asarray(y, dtype=int).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)
    if len(y) != len(score):
        raise ValueError(f"length mismatch: y={len(y)} score={len(score)}")
    if n_boot <= 0:
        return float("nan"), float("nan"), 0
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not np.all(np.isfinite(score)):
        raise ValueError("ROC-AUC scores contain non-finite values")

    member_scores = score[y == 1]
    nonmember_scores = score[y == 0]
    n_member = len(member_scores)
    n_nonmember = len(nonmember_scores)
    if n_member == 0 or n_nonmember == 0:
        return float("nan"), float("nan"), 0

    score_values = np.unique(np.concatenate((member_scores, nonmember_scores)))
    member_bins = np.searchsorted(score_values, member_scores)
    nonmember_bins = np.searchsorted(score_values, nonmember_scores)
    member_frequency = np.bincount(member_bins, minlength=len(score_values))
    nonmember_frequency = np.bincount(nonmember_bins, minlength=len(score_values))
    member_probability = member_frequency / float(n_member)
    nonmember_probability = nonmember_frequency / float(n_nonmember)

    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=float)
    denominator = float(n_member * n_nonmember)
    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        size = stop - start
        sampled_members = rng.multinomial(n_member, member_probability, size=size)
        sampled_nonmembers = rng.multinomial(
            n_nonmember, nonmember_probability, size=size
        )
        nonmembers_below = (
            np.cumsum(sampled_nonmembers, axis=1) - sampled_nonmembers
        )
        concordant = np.sum(
            sampled_members
            * (nonmembers_below + 0.5 * sampled_nonmembers),
            axis=1,
        )
        values[start:stop] = concordant / denominator

    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        n_boot,
    )


def stratified_bootstrap_tpr_at_fpr(
    y: np.ndarray,
    score: np.ndarray,
    requested_fpr: float,
    n_boot: int,
    seed: int,
    chunk_size: int = 1024,
) -> Tuple[float, float, int]:
    """Efficient exact stratified bootstrap for empirical TPR at bounded FPR.

    Multinomial score-bin counts are equivalent to resampling records with
    replacement. Tied scores remain in one bin, matching ROC threshold
    semantics rather than splitting ties to hit an artificial exact FPR.
    """
    y = np.asarray(y, dtype=int).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)
    if len(y) != len(score):
        raise ValueError("membership and score lengths differ")
    if not 0.0 <= requested_fpr <= 1.0:
        raise ValueError("requested_fpr must be in [0, 1]")
    members = score[y == 1]
    nonmembers = score[y == 0]
    if not len(members) or not len(nonmembers) or n_boot <= 0:
        return float("nan"), float("nan"), 0
    score_values = np.unique(np.concatenate((members, nonmembers)))
    member_frequency = np.bincount(
        np.searchsorted(score_values, members), minlength=len(score_values)
    )
    nonmember_frequency = np.bincount(
        np.searchsorted(score_values, nonmembers), minlength=len(score_values)
    )
    member_probability = member_frequency / float(len(members))
    nonmember_probability = nonmember_frequency / float(len(nonmembers))
    rng = np.random.default_rng(seed)
    output = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        size = stop - start
        sampled_members = rng.multinomial(len(members), member_probability, size=size)
        sampled_nonmembers = rng.multinomial(len(nonmembers), nonmember_probability, size=size)
        true_positive = np.cumsum(sampled_members[:, ::-1], axis=1)
        false_positive = np.cumsum(sampled_nonmembers[:, ::-1], axis=1)
        eligible = false_positive <= requested_fpr * len(nonmembers) + 1e-12
        eligible_tpr = np.where(eligible, true_positive / float(len(members)), 0.0)
        output[start:stop] = eligible_tpr.max(axis=1)
    return float(np.quantile(output, 0.025)), float(np.quantile(output, 0.975)), n_boot


def tpr_at_resolvable_fpr(
    y: np.ndarray, score: np.ndarray, requested_fpr: float
) -> Tuple[float, float]:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, score)
    eligible = np.flatnonzero(fpr <= requested_fpr + 1e-15)
    if not len(eligible):
        return 0.0, 0.0
    index = eligible[-1]
    return float(tpr[index]), float(fpr[index])


def cross_fitted_threshold_metrics(
    y: np.ndarray,
    score: np.ndarray,
    n_splits: int = 5,
    seed: int = 2026,
) -> Dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, roc_curve
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    minimum_class = min(int((y == 0).sum()), int((y == 1).sum()))
    if minimum_class < 2:
        return {
            "balanced_accuracy_crossfit": float("nan"),
            "membership_advantage_crossfit": float("nan"),
            "crossfit_tpr": float("nan"),
            "crossfit_fpr": float("nan"),
            "threshold_median": float("nan"),
            "threshold_folds": 0,
        }
    folds = max(2, min(n_splits, minimum_class))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    predictions = np.zeros_like(y)
    thresholds: list[float] = []
    for calibration, evaluation in splitter.split(score.reshape(-1, 1), y):
        fpr, tpr, threshold = roc_curve(y[calibration], score[calibration])
        best = int(np.nanargmax(tpr - fpr))
        selected = float(threshold[best])
        thresholds.append(selected)
        predictions[evaluation] = (score[evaluation] >= selected).astype(int)

    tp = int(((predictions == 1) & (y == 1)).sum())
    fn = int(((predictions == 0) & (y == 1)).sum())
    fp = int(((predictions == 1) & (y == 0)).sum())
    tn = int(((predictions == 0) & (y == 0)).sum())
    tpr_value = tp / max(tp + fn, 1)
    fpr_value = fp / max(fp + tn, 1)
    return {
        "balanced_accuracy_crossfit": float(balanced_accuracy_score(y, predictions)),
        "membership_advantage_crossfit": float(tpr_value - fpr_value),
        "crossfit_tpr": float(tpr_value),
        "crossfit_fpr": float(fpr_value),
        "threshold_median": float(np.median(thresholds)),
        "threshold_folds": int(folds),
    }


def percentile_interval(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
