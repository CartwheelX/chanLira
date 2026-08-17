
#!/usr/bin/env python3
"""
train_mia_attack_cvholdout_multigpu.py

Option 1 (small-N safe):
  - Hold out TEST once (untouched)
  - Tune hyperparams using Stratified K-fold CV on TRAINPOOL (train+val)
  - Retrain on full TRAINPOOL using epochs chosen from CV (median best_epoch across folds)
  - Evaluate once on TEST
  - Save per-target artifacts + per-run attack_results.json

NEW in this version:
  - Parse target-model train/test acc from sibling train.log (next to *_attack_data.pt)
  - Store it in attack_results.json as target_model_perf
  - Write global attack_summary.csv at --out root after launcher/single
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TextIO

# Required for deterministic CUDA matrix operations when deterministic
# algorithms are enabled in each learned-MIA subprocess.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# -------------------------
# Data structures
# -------------------------
@dataclass
class AttackDataset:
    target_id: str
    run_id: int
    dataset: str
    architecture: str
    role: str
    X: torch.Tensor
    y_true: torch.Tensor
    y_pred: torch.Tensor
    membership: torch.Tensor  # normalized: 1=member, 0=nonmember
    source_membership_convention: str
    split: torch.Tensor
    pv: torch.Tensor
    stats: Dict[str, torch.Tensor]
    meta: Dict[str, Any]
    source_path: Path


@dataclass
class AttackMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float
    tpr_at_low_fpr: Dict[str, float]
    tn: int
    fp: int
    fn: int
    tp: int


# -------------------------
# Discovery / loading
# -------------------------
def find_attack_data_files(root_dir: Path) -> List[Path]:
    return sorted(root_dir.rglob("*_attack_data.pt"))


def _require_keys(d: Dict[str, Any], keys: List[str], path: Path) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"Missing keys {missing} in {path}")


def normalize_membership_labels(
    data: Dict[str, Any],
) -> Tuple[torch.Tensor, str]:
    """Normalize attack labels to the reviewer convention 1=member, 0=nonmember."""
    raw = torch.as_tensor(data["membership"]).long().reshape(-1)
    split = torch.as_tensor(data["split"]).long().reshape(-1)
    meta = data.get("meta", {}) or {}
    declared = str(meta.get("membership_convention", "")).strip().lower()

    if declared in {"1=member", "member_is_1", "one_is_member"}:
        member_value = 1
        source_convention = "1=member"
    elif declared in {"0=member", "member_is_0", "zero_is_member"}:
        member_value = 0
        source_convention = "0=member"
    elif len(raw) == len(split):
        train_values = raw[split == 0]
        test_values = raw[split == 1]
        if (
            train_values.numel()
            and test_values.numel()
            and torch.all(train_values == 1)
            and torch.all(test_values == 0)
        ):
            member_value = 1
            source_convention = "1=member_inferred"
        else:
            # Current QuRiFT exports train/member=0 and test/nonmember=1.
            member_value = 0
            source_convention = "0=member_inferred"
    else:
        member_value = 0
        source_convention = "0=member_default"

    normalized = (raw == member_value).long()
    return normalized, source_convention


def load_attack_data(path: Path) -> AttackDataset:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch releases lacking weights_only.
        data = torch.load(path, map_location="cpu")
    _require_keys(data, ["X", "y_true", "y_pred", "membership", "split", "pv", "stats"], path)

    meta = data.get("meta", {})
    target_id = str(meta.get("target_id", path.parent.name)).strip() or path.parent.name
    run_id = int(meta.get("run_id", -1))
    dataset = str(meta.get("dataset", "unknown"))
    architecture = str(meta.get("model_type", "unknown")).upper()
    role = str(meta.get("role", "selected"))

    X = data["X"].float()
    y_true = data["y_true"].long()
    y_pred = data["y_pred"].long()
    membership, source_membership_convention = normalize_membership_labels(data)
    split = data["split"].long()
    pv = data["pv"].float()

    stats = data["stats"]
    if not isinstance(stats, dict):
        raise TypeError(f"Expected stats dict in {path}, got {type(stats)}")

    stats_clean: Dict[str, torch.Tensor] = {}
    for k, v in stats.items():
        stats_clean[k] = v.float() if torch.is_tensor(v) else torch.tensor(v).float()

    return AttackDataset(
        target_id=target_id,
        run_id=run_id,
        dataset=dataset,
        architecture=architecture,
        role=role,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        membership=membership,
        source_membership_convention=source_membership_convention,
        split=split,
        pv=pv,
        stats=stats_clean,
        meta=meta,
        source_path=path,
    )


def target_output_dir(out_dir: Path, dataset: AttackDataset) -> Path:
    """Return a collision-free directory based on the reviewer target ID."""
    safe_target_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset.target_id).strip("._")
    if not safe_target_id:
        safe_target_id = (
            f"{dataset.dataset}_{dataset.architecture}_run{dataset.run_id}"
        )
    return out_dir / safe_target_id


def set_attacker_seed(seed: int) -> None:
    """Seed every RNG used by learned-MIA training."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# -------------------------
# Target-model perf parsing (from sibling train.log)
# -------------------------

# -------------------------
# Standardization (fit on train only)
# -------------------------
def standardize_fit(X_train: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = X_train.mean(dim=0, keepdim=True)
    sd = X_train.std(dim=0, keepdim=True).clamp_min(eps)
    return mu, sd


def standardize_apply(X: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return (X - mu) / sd


# -------------------------
# Model (MLP attack)
# -------------------------
class MLPAttackModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        use_layernorm: bool = False,
        activation: str = "relu",
    ):
        super().__init__()
        act: nn.Module = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()

        layers: List[nn.Module] = []
        d = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act)
            layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def _loss_and_acc(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion, device: str) -> Tuple[float, float]:
    model.eval()
    logits = model(X.to(device))
    loss = float(criterion(logits, y.to(device).long()).item())
    pred = logits.argmax(dim=1).detach().cpu()
    acc = float((pred == y.detach().cpu().long()).float().mean().item())
    return loss, acc


# -------------------------
# Training with early stopping on fold-val (allowed in CV)
# -------------------------
def train_fold_early_stop(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_va: torch.Tensor,
    y_va: torch.Tensor,
    *,
    params: Dict[str, Any],
    device: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> Tuple[nn.Module, Dict[str, Any]]:
    input_dim = int(X_tr.shape[1])
    model = MLPAttackModel(
        input_dim=input_dim,
        hidden_dim=int(params["hidden_dim"]),
        n_layers=int(params["n_layers"]),
        dropout=float(params["dropout"]),
        use_layernorm=bool(params.get("layernorm", False)),
        activation=str(params.get("activation", "relu")),
    ).to(device)

    lr = float(params["lr"])
    wd = float(params["weight_decay"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    ds = torch.utils.data.TensorDataset(X_tr, y_tr.long())
    dl = torch.utils.data.DataLoader(ds, batch_size=min(int(batch_size), len(ds)), shuffle=True)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    bad = 0
    last_epoch = 0

    for epoch in range(1, max_epochs + 1):
        last_epoch = epoch
        model.train()
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = nn.CrossEntropyLoss()(logits, yb)
            loss.backward()
            optimizer.step()

        val_loss, _ = _loss_and_acc(model, X_va, y_va, criterion, device)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_epoch": int(best_epoch), "best_val_loss": float(best_val_loss), "stopped_epoch": int(last_epoch)}


# -------------------------
# Eval (once on final test)
# -------------------------
@torch.no_grad()
def evaluate_attack(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, device: str):
    model.eval()
    logits = model(X_test.to(device))
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()

    y_test_np = y_test.detach().cpu().numpy().astype(int)
    y_pred = probs.argmax(axis=1).astype(int)
    scores = probs[:, 1]  # P(non-member)

    acc = accuracy_score(y_test_np, y_pred)
    prec = precision_score(y_test_np, y_pred, zero_division=0)
    rec = recall_score(y_test_np, y_pred, zero_division=0)
    f1 = f1_score(y_test_np, y_pred, zero_division=0)

    try:
        auc_roc = roc_auc_score(y_test_np, scores)
    except Exception:
        auc_roc = 0.5
    try:
        auc_pr = average_precision_score(y_test_np, scores)
    except Exception:
        auc_pr = 0.0

    fpr, tpr, thr = roc_curve(y_test_np, scores)

    tpr_at_fpr: Dict[str, float] = {}
    for target_fpr in [0.001, 0.01, 0.05, 0.1]:
        idx = np.where(fpr <= target_fpr)[0]
        tpr_at_fpr[f"tpr@fpr={target_fpr}"] = float(tpr[idx[-1]]) if len(idx) > 0 else 0.0

    tn, fp, fn, tp = confusion_matrix(y_test_np, y_pred, labels=[0, 1]).ravel()

    metrics = AttackMetrics(
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        auc_roc=float(auc_roc),
        auc_pr=float(auc_pr),
        tpr_at_low_fpr=tpr_at_fpr,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )
    return metrics, fpr, tpr, thr


def plot_and_save_roc_loglog(
    fpr: np.ndarray,
    tpr: np.ndarray,
    *,
    title: str,
    save_prefix: Path,
    eps: float = 1e-6,
    font_family: str = "serif",
):
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = font_family

    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    fpr_c = np.clip(fpr, eps, 1.0)
    tpr_c = np.clip(tpr, eps, 1.0)

    fig, ax = plt.subplots(figsize=(6.5, 5.2), facecolor="white")
    ax.plot(fpr_c, tpr_c, linewidth=2)

    diag = np.linspace(eps, 1.0, 200)
    ax.plot(diag, diag, linestyle="--", linewidth=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(eps, 1.0)
    ax.set_ylim(eps, 1.0)

    ax.set_title(title, fontweight="normal")
    ax.set_xlabel("False Positive Rate (FPR)", fontweight="normal")
    ax.set_ylabel("True Positive Rate (TPR)", fontweight="normal")
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()

    fig.savefig(str(save_prefix) + "_roc_loglog.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(save_prefix) + "_roc_loglog.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# -------------------------
# Tuning helpers
# -------------------------
def _try_import_optuna():
    try:
        import optuna  # type: ignore
        return optuna
    except Exception:
        return None


def _sample_params_fallback(rng: np.random.Generator) -> Dict[str, Any]:
    hidden_dim = int(2 ** rng.integers(5, 10))  # 32..512
    n_layers = int(rng.integers(1, 5))          # 1..4
    dropout = float(rng.uniform(0.0, 0.6))
    lr = float(10 ** rng.uniform(-5, np.log10(3e-3)))
    weight_decay = float(10 ** rng.uniform(-6, -2))
    batch_size = int(rng.choice([8, 16, 32, 64, 128]))
    layernorm = bool(rng.choice([False, True]))
    activation = str(rng.choice(["relu", "gelu"]))
    return {
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "layernorm": layernorm,
        "activation": activation,
    }


def cv_score_params(
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    *,
    params: Dict[str, Any],
    device: str,
    max_epochs: int,
    patience: int,
    n_splits: int,
    seed: int,
) -> Tuple[float, List[int]]:
    y_np = y_pool.detach().cpu().numpy().astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs: List[float] = []
    best_epochs: List[int] = []

    for tr_idx, va_idx in skf.split(np.zeros_like(y_np), y_np):
        tr_idx_t = torch.from_numpy(tr_idx)
        va_idx_t = torch.from_numpy(va_idx)

        X_tr = X_pool[tr_idx_t]
        y_tr = y_pool[tr_idx_t]
        X_va = X_pool[va_idx_t]
        y_va = y_pool[va_idx_t]

        mu, sd = standardize_fit(X_tr)
        X_tr_s = standardize_apply(X_tr, mu, sd)
        X_va_s = standardize_apply(X_va, mu, sd)

        model, info = train_fold_early_stop(
            X_tr_s, y_tr, X_va_s, y_va,
            params=params,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=int(params["batch_size"]),
        )

        model.eval()
        with torch.no_grad():
            logits = model(X_va_s.to(device))
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        scores = probs[:, 1]
        y_va_np = y_va.detach().cpu().numpy().astype(int)
        try:
            auc = float(roc_auc_score(y_va_np, scores))
        except Exception:
            auc = 0.5

        aucs.append(auc)
        best_epochs.append(int(info["best_epoch"] if info["best_epoch"] > 0 else info["stopped_epoch"]))

    return float(np.mean(aucs)), best_epochs


def tune_params_cv(
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    *,
    device: str,
    n_trials: int,
    max_epochs: int,
    patience: int,
    n_splits: int,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # (your robust Optuna tuner from earlier; kept as-is)
    optuna = _try_import_optuna()
    lambda_std = 0.25

    storage = os.environ.get("OPTUNA_STORAGE", None)
    study_name = os.environ.get("OPTUNA_STUDY_NAME", None)

    if optuna is None:
        rng = np.random.default_rng(seed)
        best_params = None
        best_score = -1e9
        best_epochs = None

        for t in range(1, n_trials + 1):
            params = _sample_params_fallback(rng)
            mean_auc, epochs = cv_score_params(
                X_pool, y_pool,
                params=params,
                device=device,
                max_epochs=max_epochs,
                patience=patience,
                n_splits=n_splits,
                seed=seed + t,
            )
            score = float(mean_auc)
            if score > best_score:
                best_score = score
                best_params = params
                best_epochs = epochs

        assert best_params is not None and best_epochs is not None
        chosen_epochs = int(np.median(best_epochs))
        best_params["chosen_final_epochs"] = max(1, min(chosen_epochs, max_epochs))

        return best_params, {
            "method": "random_search",
            "best_cv_auc_mean": float(best_score),
            "fold_best_epochs": [int(e) for e in best_epochs],
            "chosen_final_epochs": int(best_params["chosen_final_epochs"]),
            "n_trials": int(n_trials),
            "n_splits": int(n_splits),
            "lambda_std": float(lambda_std),
            "storage": storage or "in_memory",
        }

    y_np = y_pool.detach().cpu().numpy().astype(int)

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=max(15, n_trials // 5),
        multivariate=True,
        group=True,
        consider_endpoints=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(10, n_trials // 6),
        n_warmup_steps=1,
        interval_steps=1
    )

    if study_name is None:
        study_name = f"mia_cv_{seed}_{np.random.default_rng(seed).integers(0, 10**9)}"

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage),
    )

    def objective(trial):
        params = {
            "hidden_dim": trial.suggest_int("hidden_dim", 32, 512, log=True),
            "n_layers": trial.suggest_int("n_layers", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.0, 0.6),
            "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128]),
            "layernorm": trial.suggest_categorical("layernorm", [False, True]),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + trial.number)
        aucs: List[float] = []
        best_epochs: List[int] = []

        for fold_i, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(y_np), y_np), start=1):
            tr_idx_t = torch.from_numpy(tr_idx)
            va_idx_t = torch.from_numpy(va_idx)

            X_tr = X_pool[tr_idx_t]
            y_tr = y_pool[tr_idx_t]
            X_va = X_pool[va_idx_t]
            y_va = y_pool[va_idx_t]

            mu, sd = standardize_fit(X_tr)
            X_tr_s = standardize_apply(X_tr, mu, sd)
            X_va_s = standardize_apply(X_va, mu, sd)

            model, info = train_fold_early_stop(
                X_tr_s, y_tr, X_va_s, y_va,
                params=params,
                device=device,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=int(params["batch_size"]),
            )

            with torch.no_grad():
                logits = model(X_va_s.to(device))
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            scores = probs[:, 1]
            y_va_np = y_va.detach().cpu().numpy().astype(int)

            try:
                auc = float(roc_auc_score(y_va_np, scores))
            except Exception:
                auc = 0.5

            aucs.append(auc)
            be = int(info["best_epoch"] if info["best_epoch"] > 0 else info["stopped_epoch"])
            best_epochs.append(be)

            trial.report(float(np.mean(aucs)), step=fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))
        robust_score = mean_auc - lambda_std * std_auc

        trial.set_user_attr("cv_auc_mean", mean_auc)
        trial.set_user_attr("cv_auc_std", std_auc)
        trial.set_user_attr("fold_best_epochs", [int(e) for e in best_epochs])
        trial.set_user_attr("chosen_final_epochs", int(np.median(best_epochs)))
        return robust_score

    study.optimize(objective, n_trials=int(n_trials), gc_after_trial=True)

    bt = study.best_trial
    best_params = dict(bt.params)

    fold_epochs = list(bt.user_attrs.get("fold_best_epochs", []))
    chosen_epochs = int(bt.user_attrs.get("chosen_final_epochs", max_epochs))
    chosen_epochs = max(1, min(chosen_epochs, max_epochs))
    best_params["chosen_final_epochs"] = chosen_epochs

    tuning_info = {
        "method": "optuna",
        "study_name": str(study.study_name),
        "storage": storage or "in_memory",
        "best_cv_score": float(bt.value),
        "best_cv_auc_mean": float(bt.user_attrs.get("cv_auc_mean", float("nan"))),
        "best_cv_auc_std": float(bt.user_attrs.get("cv_auc_std", float("nan"))),
        "fold_best_epochs": [int(e) for e in fold_epochs],
        "chosen_final_epochs": int(chosen_epochs),
        "n_trials": int(n_trials),
        "n_splits": int(n_splits),
        "lambda_std": float(lambda_std),
    }
    return best_params, tuning_info


# -------------------------
# Final training (no val, no test peeking)
# -------------------------
def train_final_on_pool(
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    *,
    params: Dict[str, Any],
    device: str,
    epochs: int,
    log_fh: Optional[TextIO] = None,
) -> nn.Module:
    def _log(s: str):
        if log_fh is not None:
            print(s, file=log_fh, flush=True)

    input_dim = int(X_pool.shape[1])
    model = MLPAttackModel(
        input_dim=input_dim,
        hidden_dim=int(params["hidden_dim"]),
        n_layers=int(params["n_layers"]),
        dropout=float(params["dropout"]),
        use_layernorm=bool(params.get("layernorm", False)),
        activation=str(params.get("activation", "relu")),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    bs = int(params["batch_size"])
    ds = torch.utils.data.TensorDataset(X_pool, y_pool.long())
    dl = torch.utils.data.DataLoader(ds, batch_size=min(bs, len(ds)), shuffle=True)

    _log("Epoch |    Loss (train) |   Acc (train)")
    _log("----- | -------------- | ------------")

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        tr_loss, tr_acc = _loss_and_acc(model, X_pool, y_pool, criterion, device)
        _log(f"{ep:5d} | {tr_loss:14.4f} | {tr_acc:12.3f}")

    return model


# -------------------------
# JSON helper
# -------------------------
def _json_default(o: Any):
    if isinstance(o, Path):
        return str(o)
    if torch.is_tensor(o):
        if o.numel() == 1:
            return o.item()
        return o.detach().cpu().tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    return str(o)


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


# -------------------------
# One target end-to-end (Option 1)
# -------------------------
def run_one_target(
    attack_data_path: Path,
    out_dir: Path,
    *,
    test_ratio: float,
    cv_folds: int,
    tune: bool,
    n_trials: int,
    max_epochs: int,
    patience: int,
    device_req: str,
    seed: int,
    cpu_threads: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    if cpu_threads is not None and int(cpu_threads) > 0:
        torch.set_num_threads(int(cpu_threads))
        torch.set_num_interop_threads(max(1, int(cpu_threads)))

    set_attacker_seed(seed)
    ds = load_attack_data(attack_data_path)

    # NEW: parse target model perf from sibling train.log
    train_log_path = find_train_log_near_attack_file(attack_data_path)
    # target_model_perf = parse_train_log_for_target_acc(train_log_path) if train_log_path else {
    #     "train_acc": float("nan"), "test_acc": float("nan"), "gap_acc": float("nan"),
    #     "source_train": "missing_log", "source_test": "missing_log",
    #     "train_log_path": None
    # }

    target_model_perf = parse_train_log_for_target_acc(train_log_path) if train_log_path else {
    "target_train_acc": float("nan"),
    "target_test_acc": float("nan"),
    "target_gap_acc": float("nan"),
    "train_log_path": None,
    }

    # if verbose and (not np.isfinite(target_model_perf["train_acc"]) or not np.isfinite(target_model_perf["test_acc"])):
    if verbose and (not np.isfinite(target_model_perf["target_train_acc"]) or not np.isfinite(target_model_perf["target_test_acc"])):

        print(f"[WARN] Could not parse target train/test acc from: {target_model_perf.get('train_log_path')}")

    # print(f'target_model_perf: {target_model_perf["train_acc"]}')
    # exit()
    run_out_dir = target_output_dir(out_dir, ds)
    run_out_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_out_dir / "attack_train.log"
    best_params_path = run_out_dir / "best_params.json"

    device = device_req
    if device == "cuda":
        if not torch.cuda.is_available():
            device = "cpu"
        else:
            device = "cuda:0"
            try:
                torch.cuda.set_device(0)
            except Exception:
                pass

    X = ds.X
    y = ds.membership  # normalized: 1=member, 0=nonmember
    y_np = y.detach().cpu().numpy().astype(int)

    n_nonmember = int((y == 0).sum().item())
    n_member = int((y == 1).sum().item())
    if n_nonmember < 2 or n_member < 2:
        raise ValueError(
            f"Too few samples per class: members={n_member}, "
            f"nonmembers={n_nonmember} in {attack_data_path}"
        )

    min_class = min(n_nonmember, n_member)
    feasible_folds = min(int(cv_folds), int(min_class))
    if feasible_folds < 2:
        feasible_folds = 1

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    (pool_idx, test_idx) = next(sss.split(np.zeros_like(y_np), y_np))

    pool_idx_t = torch.from_numpy(pool_idx)
    test_idx_t = torch.from_numpy(test_idx)

    X_pool = X[pool_idx_t]
    y_pool = y[pool_idx_t]
    X_test = X[test_idx_t]
    y_test = y[test_idx_t]

    mu_pool, sd_pool = standardize_fit(X_pool)
    X_pool_s = standardize_apply(X_pool, mu_pool, sd_pool)
    X_test_s = standardize_apply(X_test, mu_pool, sd_pool)

    if tune and feasible_folds >= 2:
        best_params, tuning_info = tune_params_cv(
            X_pool_s, y_pool,
            device=device,
            n_trials=n_trials,
            max_epochs=max_epochs,
            patience=patience,
            n_splits=feasible_folds,
            seed=seed,
        )
    else:
        best_params = {
            "hidden_dim": 128,
            "n_layers": 2,
            "dropout": 0.3,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 32,
            "layernorm": False,
            "activation": "relu",
            "chosen_final_epochs": max_epochs,
        }
        tuning_info = {"method": "fixed_or_no_cv", "note": f"feasible_folds={feasible_folds}", "best_cv_auc": float("nan")}

    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump({"best_params": best_params, "tuning_info": tuning_info}, f, indent=2, default=_json_default)

    final_epochs = int(best_params.get("chosen_final_epochs", max_epochs))
    final_epochs = max(1, min(final_epochs, max_epochs))

    with open(log_path, "w", encoding="utf-8") as log_fh:
        log_fh.write(f"Source: {attack_data_path}\n")
        log_fh.write(
            f"Target: {ds.target_id} ({ds.architecture} on {ds.dataset}, "
            f"source_run_id={ds.run_id})\n"
        )
        log_fh.write(f"Members={n_member}, NonMembers={n_nonmember}\n")
        log_fh.write(
            "Membership convention used for learned MIA: "
            "1=member,0=nonmember "
            f"(source={ds.source_membership_convention})\n"
        )
        log_fh.write(f"Holdout test_ratio={test_ratio} | TEST size={len(X_test)} | TRAINPOOL size={len(X_pool)}\n")
        log_fh.write(f"CV folds requested={cv_folds}, feasible={feasible_folds}\n")
        log_fh.write(f"Device used: {device}\n")
        log_fh.write(f"CPU threads: {cpu_threads}\n")

        # NEW: record target-model train/test/gap from train.log
        ta = target_model_perf.get("target_train_acc", float("nan"))
        te = target_model_perf.get("target_test_acc", float("nan"))
        ga = target_model_perf.get("target_gap_acc", float("nan"))
        log_fh.write(f"TargetModel(train.log): train_acc={ta} test_acc={te} gap_acc={ga}\n")

        log_fh.write(f"train.log path: {train_log_path}\n")

        log_fh.write(f"Tuning info: {tuning_info}\n")
        log_fh.write(f"Best params: {best_params}\n")
        log_fh.write(f"Final epochs (from CV median): {final_epochs}\n\n")
        log_fh.flush()

        model = train_final_on_pool(
            X_pool_s, y_pool,
            params=best_params,
            device=device,
            epochs=final_epochs,
            log_fh=log_fh,
        )

        metrics, fpr, tpr, thr = evaluate_attack(model, X_test_s, y_test, device=device)

        log_fh.write("\n" + "=" * 60 + "\n")
        log_fh.write("FINAL TEST RESULTS (untouched holdout)\n")
        log_fh.write("=" * 60 + "\n")
        log_fh.write(f"Accuracy:  {metrics.accuracy:.4f}\n")
        log_fh.write(f"Precision: {metrics.precision:.4f}\n")
        log_fh.write(f"Recall:    {metrics.recall:.4f}\n")
        log_fh.write(f"F1:        {metrics.f1:.4f}\n")
        log_fh.write(f"AUC-ROC:   {metrics.auc_roc:.4f}\n")
        log_fh.write(f"AUC-PR:    {metrics.auc_pr:.4f}\n")
        for k, v in metrics.tpr_at_low_fpr.items():
            log_fh.write(f"{k}: {v:.4f}\n")
        log_fh.write(f"Confusion: TN={metrics.tn} FP={metrics.fp} FN={metrics.fn} TP={metrics.tp}\n")
        log_fh.flush()

    torch.save(model.state_dict(), run_out_dir / "attack_model.pt")
    torch.save({"mu": mu_pool, "sd": sd_pool}, run_out_dir / "scaler.pt")

    roc_csv = run_out_dir / "roc_curve.csv"
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(roc_csv, index=False)

    title = f"ROC (log-log): {ds.target_id}"
    plot_and_save_roc_loglog(fpr, tpr, title=title, save_prefix=run_out_dir / "attack", eps=1e-6)

    results = {
        "target_meta": ds.meta,
        "target_id": ds.target_id,
        "attacker_seed": int(seed),
        "membership_convention": "1=member,0=nonmember",
        "source_membership_convention": ds.source_membership_convention,
        "target_model_perf": target_model_perf,  # NEW
        "split": {"test_ratio": float(test_ratio), "n_pool": int(len(X_pool)), "n_test": int(len(X_test))},
        "cv": {"requested_folds": int(cv_folds), "feasible_folds": int(feasible_folds)},
        "tuning_info": tuning_info,
        "best_params": best_params,
        "final_epochs": int(final_epochs),
        "metrics_test": asdict(metrics),
        "paths": {
            "run_out_dir": str(run_out_dir),
            "attack_train_log": str(log_path),
            "best_params_json": str(best_params_path),
            "roc_csv": str(roc_csv),
            "roc_pdf": str(run_out_dir / "attack_roc_loglog.pdf"),
            "target_train_log": str(train_log_path),
        },
    }
    with open(run_out_dir / "attack_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)

    if verbose:
        print(f"[Saved] {run_out_dir}")

    return results


# -------------------------
# Global summary writer (reads attack_results.json)
# -------------------------
def write_attack_summary(out_dir: Path) -> Path:
    rows: List[Dict[str, Any]] = []
    for p in sorted(out_dir.rglob("attack_results.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        meta = r.get("target_meta", {}) or {}
        m = r.get("metrics_test", {}) or {}
        tpr_map = m.get("tpr_at_low_fpr", {}) if isinstance(m, dict) else {}

        tp = r.get("target_model_perf", {}) or {}
        rows.append({
            "target_id": str(r.get("target_id", meta.get("target_id", p.parent.name))),
            "experiment": str(meta.get("experiment", "")),
            "dataset": str(meta.get("dataset", "unknown")),
            "architecture": str(meta.get("model_type", "unknown")).upper(),
            "run_id": int(meta.get("run_id", -1)),
            "model_seed": meta.get("model_seed", meta.get("seed", np.nan)),
            "data_seed": meta.get("data_seed", np.nan),
            "attacker_seed": r.get("attacker_seed", np.nan),
            "membership_convention": str(
                r.get("membership_convention", "1=member,0=nonmember")
            ),
            "n_wires": meta.get("n_wires", np.nan),
            "depth": meta.get("depth", np.nan),

            # target model perf from train.log
            "target_train_acc": _safe_float(tp.get("target_train_acc", np.nan)),
            "target_test_acc": _safe_float(tp.get("target_test_acc", np.nan)),
            "target_gap_acc": _safe_float(tp.get("target_gap_acc", np.nan)),

            # attack metrics
            "attack_acc": _safe_float(m.get("accuracy", np.nan)),
            "attack_auc": _safe_float(m.get("auc_roc", np.nan)),
            "attack_f1": _safe_float(m.get("f1", np.nan)),
            "attack_precision": _safe_float(m.get("precision", np.nan)),
            "attack_recall": _safe_float(m.get("recall", np.nan)),
            "tpr@fpr=0.001": _safe_float(tpr_map.get("tpr@fpr=0.001", np.nan)),
            "tpr@fpr=0.01": _safe_float(tpr_map.get("tpr@fpr=0.01", np.nan)),
            "tpr@fpr=0.05": _safe_float(tpr_map.get("tpr@fpr=0.05", np.nan)),
            "tpr@fpr=0.1": _safe_float(tpr_map.get("tpr@fpr=0.1", np.nan)),

            "tuning_method": str((r.get("tuning_info", {}) or {}).get("method", "unknown")),
            "cv_best_auc": _safe_float((r.get("tuning_info", {}) or {}).get("best_cv_auc_mean", np.nan)),
            "final_epochs": int(r.get("final_epochs", np.nan)) if str(r.get("final_epochs", "")).isdigit() else r.get("final_epochs", np.nan),
            "attack_results_json": str(p),
        })

    df = pd.DataFrame(rows)
    summary_path = out_dir / "attack_summary.csv"
    df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Results dir: {out_dir}")
    print(f"Summary CSV: {summary_path}")
    if len(df) > 0:
        mean_auc = float(pd.to_numeric(df["attack_auc"], errors="coerce").mean())
        mean_acc = float(pd.to_numeric(df["attack_acc"], errors="coerce").mean())
        mean_gap = float(pd.to_numeric(df["target_gap_acc"], errors="coerce").mean())
        print(f"Processed targets: {len(df)}")
        print(f"Mean attack AUC-ROC: {mean_auc:.4f}")
        print(f"Mean attack ACC:     {mean_acc:.4f}")
        print(f"Mean target gap_acc: {mean_gap:.4f}")
    return summary_path


# -------------------------
# GPU detection (nvidia-smi)
# -------------------------
def get_usable_gpus(
    *,
    min_free_mem_mb: int = 0,
    max_util: int = 15,
    max_mem_used_mb: int = 2000,
) -> List[str]:
    try:
        q = "nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv,noheader,nounits"
        out = subprocess.check_output(q.split()).decode("utf-8").strip()
        gpus: List[str] = []
        for line in out.splitlines():
            if not line.strip():
                continue
            idx, free_mem, used_mem, util = [x.strip() for x in line.split(",")]
            if int(free_mem) < int(min_free_mem_mb):
                continue
            if int(util) > int(max_util):
                continue
            if int(used_mem) > int(max_mem_used_mb):
                continue
            gpus.append(idx)
        return gpus
    except Exception:
        return []



def parse_train_log_for_target_acc(train_log_path: Optional[Path]) -> Dict[str, Any]:
    out = {
        "target_train_acc": np.nan,
        "target_test_acc": np.nan,
        "target_gap_acc": np.nan,
        "train_log_path": str(train_log_path) if train_log_path else None,
    }
    if train_log_path is None or not train_log_path.exists():
        return out

    txt = train_log_path.read_text(encoding="utf-8", errors="ignore")

    # Preferred (your exact line):
    # [PV Generation] Train acc=0.7400 Test acc=0.8800
    m = re.search(
        r"\[PV\s*Generation\].*?Train\s*acc\s*=\s*([0-9]*\.?[0-9]+).*?Test\s*acc\s*=\s*([0-9]*\.?[0-9]+)",
        txt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        tr = float(m.group(1))
        te = float(m.group(2))
        out["target_train_acc"] = tr
        out["target_test_acc"] = te
        out["target_gap_acc"] = tr - te
        return out

    # Fallback: parse the final "Test  | ... acc X"
    m2 = re.search(r"^Test\s*\|\s*loss\s*[0-9.]+\s*acc\s*([0-9.]+)\s*$", txt, flags=re.MULTILINE)
    if m2:
        out["target_test_acc"] = float(m2.group(1))

    # Fallback: last epoch row train/val acc
    # "  100 | ... | 0.740 / 0.780"
    epoch_row_re = re.compile(
        r"^\s*\d+\s*\|\s*[0-9.]+\s*/\s*[0-9.]+\s*\|\s*([0-9.]+)\s*/\s*([0-9.]+)\s*$",
        re.MULTILINE,
    )
    rows = epoch_row_re.findall(txt)
    if rows:
        out["target_train_acc"] = float(rows[-1][0])

    if np.isfinite(out["target_train_acc"]) and np.isfinite(out["target_test_acc"]):
        out["target_gap_acc"] = float(out["target_train_acc"] - out["target_test_acc"])

    return out


def rebuild_attack_summary(out_dir: Path, attack_files: List[Path]) -> Path:
    """
    Rebuilds attack_summary.csv by reading existing per-run attack_results.json
    + parsing target train/test acc from train.log next to each *_attack_data.pt.
    Works even if launcher used --resume and skipped training.
    """
    rows: List[Dict[str, Any]] = []

    for f in attack_files:
        # load meta to know dataset/arch/run_id
        try:
            d = load_attack_data(f)
        except Exception:
            continue

        run_out_dir = target_output_dir(out_dir, d)
        results_json = run_out_dir / "attack_results.json"

        attack_acc = np.nan
        attack_auc = np.nan
        attack_f1 = np.nan
        attack_precision = np.nan
        attack_recall = np.nan
        tpr001 = 0.0
        tpr01  = 0.0
        tpr05  = 0.0
        tpr1   = 0.0
        tuning_method = "unknown"
        cv_best_auc = np.nan
        final_epochs = np.nan
        attacker_seed = np.nan

        if results_json.exists():
            try:
                r = json.loads(results_json.read_text(encoding="utf-8", errors="ignore"))
                m = r.get("metrics_test", {}) or {}
                tpr_map = m.get("tpr_at_low_fpr", {}) if isinstance(m, dict) else {}

                attack_acc = m.get("accuracy", np.nan)
                attack_auc = m.get("auc_roc", np.nan)
                attack_f1 = m.get("f1", np.nan)
                attack_precision = m.get("precision", np.nan)
                attack_recall = m.get("recall", np.nan)

                tpr001 = tpr_map.get("tpr@fpr=0.001", 0.0)
                tpr01  = tpr_map.get("tpr@fpr=0.01", 0.0)
                tpr05  = tpr_map.get("tpr@fpr=0.05", 0.0)
                tpr1   = tpr_map.get("tpr@fpr=0.1", 0.0)

                tuning_info = r.get("tuning_info", {}) or {}
                tuning_method = tuning_info.get("method", "unknown")
                cv_best_auc = tuning_info.get("best_cv_auc", tuning_info.get("best_cv_auc_mean", np.nan))
                final_epochs = r.get("final_epochs", np.nan)
                attacker_seed = r.get("attacker_seed", np.nan)
            except Exception:
                pass

        # parse target train/test acc from train.log near attack file
        tl = find_train_log_near_attack_file(f)
        targ = parse_train_log_for_target_acc(tl)

        meta = d.meta or {}
        rows.append({
            "target_id": d.target_id,
            "experiment": meta.get("experiment", ""),
            "dataset": meta.get("dataset", d.dataset),
            "architecture": str(meta.get("model_type", d.architecture)).upper(),
            "run_id": meta.get("run_id", d.run_id),
            "model_seed": meta.get("model_seed", meta.get("seed", np.nan)),
            "data_seed": meta.get("data_seed", np.nan),
            "attacker_seed": attacker_seed,
            "membership_convention": "1=member,0=nonmember",
            "n_wires": meta.get("n_wires", -1),
            "depth": meta.get("depth", -1),

            # "target_train_acc": targ.get("train_acc", np.nan),
            # "target_test_acc": targ.get("test_acc", np.nan),
            # "target_gap_acc": targ.get("gap_acc", np.nan),
            "target_train_acc": targ.get("target_train_acc", np.nan),
            "target_test_acc": targ.get("target_test_acc", np.nan),
            "target_gap_acc": targ.get("target_gap_acc", np.nan),


            "attack_acc": attack_acc,
            "attack_auc": attack_auc,
            "attack_f1": attack_f1,
            "attack_precision": attack_precision,
            "attack_recall": attack_recall,
            "tpr@fpr=0.001": tpr001,
            "tpr@fpr=0.01": tpr01,
            "tpr@fpr=0.05": tpr05,
            "tpr@fpr=0.1": tpr1,

            "tuning_method": tuning_method,
            "cv_best_auc": cv_best_auc,
            "final_epochs": final_epochs,

           
            "target_trainlog": targ.get("train_log_path", None),
        })

    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / "attack_summary.csv"


    # --- Put synthetic rows first, MNIST rows last ---
    def _ds_rank(ds: str) -> int:
        ds_l = str(ds).strip().lower()
        return 1 if ds_l == "mnist" else 0   # 0 = synthetic/others first, 1 = MNIST last

    summary_df["_ds_rank"] = summary_df["dataset"].map(_ds_rank)

    # Optional: within each block, keep a nice deterministic order
    summary_df = summary_df.sort_values(
        by=["_ds_rank", "dataset", "architecture", "target_id"],
        ascending=[True, True, True, True],
        kind="mergesort",   # stable sort
    ).drop(columns=["_ds_rank"])

    summary_df.to_csv(summary_path, index=False)

    # print stats (skip NaNs safely)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Results dir: {out_dir}")
    print(f"Summary CSV: {summary_path}")
    print(f"Processed targets: {len(summary_df)}")

    if len(summary_df) > 0:
        if "attack_auc" in summary_df.columns:
            print(f"Mean attack AUC-ROC: {np.nanmean(summary_df['attack_auc'].values):.4f}")
        if "attack_acc" in summary_df.columns:
            print(f"Mean attack ACC:     {np.nanmean(summary_df['attack_acc'].values):.4f}")

        gap_vals = summary_df["target_gap_acc"].values if "target_gap_acc" in summary_df.columns else np.array([])
        if gap_vals.size == 0 or np.all(~np.isfinite(gap_vals)):
            print("Mean target gap_acc: N/A (could not parse train/test acc from train.log)")
        else:
            print(f"Mean target gap_acc: {np.nanmean(gap_vals):.4f}")

    return summary_path

# -------------------------
# Launcher (multi-GPU)
# -------------------------
def launcher_run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_attack_data_files(Path(args.attack_data_dir))
    if not files:
        print(f"No *_attack_data.pt files found in {args.attack_data_dir}")
        return

    if args.filter_dataset or args.filter_arch:
        kept: List[Path] = []
        for f in files:
            try:
                d = load_attack_data(f)
                if args.filter_dataset and d.dataset.lower() != args.filter_dataset.lower():
                    continue
                if args.filter_arch and d.architecture.upper() != args.filter_arch.upper():
                    continue
                kept.append(f)
            except Exception:
                continue
        files = kept

    if not files:
        print("No files after filtering.")
        return

    print(f"Found {len(files)} attack files (after filtering)")

    if args.gpus:
        gpus = [x.strip() for x in args.gpus.split(",") if x.strip() != ""]
        print(f"Using user-specified GPUs: {gpus}")
    else:
        gpus = get_usable_gpus(
            min_free_mem_mb=int(args.min_free_mem_mb),
            max_util=int(args.max_gpu_util),
            max_mem_used_mb=int(args.max_mem_used_mb),
        )
        print(f"Auto-detected usable GPUs: {gpus} (min_free_mem_mb={args.min_free_mem_mb}, max_util={args.max_gpu_util}, max_mem_used_mb={args.max_mem_used_mb})")

    if gpus and args.device == "cuda":
        tickets: List[str] = []
        for g in gpus:
            for _ in range(max(1, int(args.jobs_per_gpu))):
                tickets.append(g)
        print(f"Total GPU tickets: {len(tickets)} (jobs-per-gpu={args.jobs_per_gpu})")
    else:
        tickets = [""] * max(1, int(args.max_cpu_workers))
        print(f"No usable GPUs (or device!=cuda). Using CPU tickets: {len(tickets)}")

    def already_done(file_path: Path) -> bool:
        if not args.resume:
            return False
        try:
            d = load_attack_data(file_path)
            run_out_dir = target_output_dir(out_dir, d)
            return (run_out_dir / "attack_results.json").exists()
        except Exception:
            return False

    from queue import Queue
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ticket_q: "Queue[str]" = Queue()
    for t in tickets:
        ticket_q.put(t)

    status_rows: List[Dict[str, Any]] = []

    def run_one_file(file_path: Path) -> Dict[str, Any]:
        gpu_id = ticket_q.get()
        try:
            if already_done(file_path):
                return {"file": str(file_path), "status": "skipped_resume", "gpu": gpu_id, "returncode": 0, "error": ""}

            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--single",
                "--attack-data-path", str(file_path),
                "--out", str(out_dir),
                "--test-ratio", str(args.test_ratio),
                "--cv-folds", str(args.cv_folds),
                "--n-trials", str(args.n_trials),
                "--max-epochs", str(args.max_epochs),
                "--patience", str(args.patience),
                "--device", str(args.device),
                "--seed", str(args.seed),
                "--cpu-threads", str(args.cpu_threads),
            ]
            if args.tune:
                cmd.append("--tune")

            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            if gpu_id != "" and args.device == "cuda":
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""

            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
            if p.returncode != 0:
                return {
                    "file": str(file_path),
                    "status": "error",
                    "gpu": gpu_id,
                    "returncode": int(p.returncode),
                    "error": (p.stdout[-4000:] if p.stdout else "nonzero exit"),
                }

            return {"file": str(file_path), "status": "ok", "gpu": gpu_id, "returncode": 0, "error": ""}
        finally:
            ticket_q.put(gpu_id)

    max_workers = len(tickets)
    print(f"Parallel workers: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_one_file, f): f for f in files}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            status_rows.append(r)
            done += 1
            print(f"[{done}/{len(files)}] {Path(r['file']).name} -> {r['status']} (gpu={r['gpu']})")
            if r["status"] == "error":
                print("  tail:", r["error"].replace("\n", " ")[:300])

    status_csv = out_dir / "launcher_status.csv"
    with open(status_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "status", "gpu", "returncode", "error"])
        w.writeheader()
        for row in status_rows:
            w.writerow(row)

    print(f"\nDone. Launcher status CSV: {status_csv}")
    print(f"Results root: {out_dir}")
    rebuild_attack_summary(out_dir, files)

    # NEW: write global summary at end
    write_attack_summary(out_dir)


# -------------------------
# Single mode CLI
# -------------------------
def single_run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    attack_data_path = Path(args.attack_data_path)
    if not attack_data_path.exists():
        raise FileNotFoundError(f"--attack-data-path not found: {attack_data_path}")

    if args.resume:
        try:
            d = load_attack_data(attack_data_path)
            run_out_dir = target_output_dir(out_dir, d)
            if (run_out_dir / "attack_results.json").exists():
                print(f"[Resume] Skipping (already exists): {run_out_dir}")
                return
        except Exception:
            pass

    run_one_target(
        attack_data_path=attack_data_path,
        out_dir=out_dir,
        test_ratio=float(args.test_ratio),
        cv_folds=int(args.cv_folds),
        tune=bool(args.tune),
        n_trials=int(args.n_trials),
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        device_req=str(args.device),
        seed=int(args.seed),
        cpu_threads=int(args.cpu_threads),
        verbose=True,
    )

    # NEW: keep summary updated even for single runs
    write_attack_summary(out_dir)

def find_train_log_near_attack_file(attack_data_path: Path) -> Optional[Path]:
    # Most common: same folder
    p = attack_data_path.parent / "train.log"
    if p.exists():
        return p

    # Sometimes you may have multiple logs, pick the first one
    cand = sorted(attack_data_path.parent.glob("*.log"))
    if cand:
        return cand[0]

    return None

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MIA attack training (CV-holdout) with multi-GPU launcher")

    ap.add_argument("--attack-data-dir", type=str, default=None,
                    help="Root directory containing *_attack_data.pt files (launcher mode)")
    ap.add_argument("--attack-data-path", type=str, default=None,
                    help="Path to a single *_attack_data.pt file (single mode)")
    ap.add_argument("--out", type=str, required=True,
                    help="Output directory for attack results")

    ap.add_argument("--launcher", action="store_true",
                    help="Enable multi-GPU launcher over --attack-data-dir (spawns subprocesses).")
    ap.add_argument("--single", action="store_true",
                    help="Run one file from --attack-data-path (used by launcher subprocesses).")

    ap.add_argument("--test-ratio", type=float, default=0.2,
                    help="Holdout test fraction (stratified)")
    ap.add_argument("--cv-folds", type=int, default=5,
                    help="CV folds on trainpool for tuning (feasible folds auto-adjusted)")
    ap.add_argument("--tune", action="store_true",
                    help="Enable per-target tuning via CV (Optuna if available).")
    ap.add_argument("--n-trials", type=int, default=30,
                    help="Tuning trials per target")
    ap.add_argument("--max-epochs", type=int, default=200,
                    help="Max epochs used inside CV fold training")
    ap.add_argument("--patience", type=int, default=15,
                    help="Early stopping patience inside CV folds")

    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--filter-dataset", type=str, default=None)
    ap.add_argument("--filter-arch", type=str, default=None)

    ap.add_argument("--cpu-threads", type=int, default=1,
                    help="torch.set_num_threads() inside each subprocess (keep small when many jobs)")

    ap.add_argument("--resume", action="store_true",
                    help="Skip targets whose attack_results.json already exists.")

    ap.add_argument("--jobs-per-gpu", type=int, default=1,
                    help="Launcher: how many parallel subprocesses per GPU.")
    ap.add_argument("--max-cpu-workers", type=int, default=2,
                    help="Launcher: if no usable GPUs, use this many CPU subprocesses.")
    ap.add_argument("--gpus", type=str, default=None,
                    help="Launcher: comma-separated GPU indices to use (e.g., '2,3,4,5,7'). Overrides auto-detect.")
    ap.add_argument("--min-free-mem-mb", type=int, default=0,
                    help="Launcher auto-detect: require at least this much free memory (MB).")
    ap.add_argument("--max-gpu-util", type=int, default=15,
                    help="Launcher auto-detect: only use GPUs with util <= this percent.")
    ap.add_argument("--max-mem-used-mb", type=int, default=2000,
                    help="Launcher auto-detect: only use GPUs with memory.used <= this MB.")

    ap.add_argument("--summary-only", action="store_true",
                help="Rebuild attack_summary.csv from existing outputs + train.log parsing (no training).")


    args = ap.parse_args()

    if args.launcher:
        if not args.attack_data_dir:
            ap.error("--launcher requires --attack-data-dir")
    if args.single:
        if not args.attack_data_path:
            ap.error("--single requires --attack-data-path")

    if not args.launcher and not args.single:
        if args.attack_data_dir:
            args.launcher = True
        elif args.attack_data_path:
            args.single = True
        else:
            ap.error("Provide either --attack-data-dir (launcher) or --attack-data-path (single).")

    return args


def main():
    args = parse_args()

    if args.summary_only:
        if not args.attack_data_dir:
            raise ValueError("--summary-only requires --attack-data-dir")
        files = find_attack_data_files(Path(args.attack_data_dir))
        rebuild_attack_summary(Path(args.out), files)
        return

    if args.launcher:
        launcher_run(args)
        return
    if args.single:
        single_run(args)
        return
    raise RuntimeError("Unreachable")


if __name__ == "__main__":
    main()
