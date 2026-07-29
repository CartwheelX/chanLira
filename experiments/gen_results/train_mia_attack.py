#!/usr/bin/env python3
"""
train_mia_attack_cvholdout.py

Option 1 (small-N safe):
  - Hold out TEST once (untouched)
  - Tune hyperparams using Stratified K-fold CV on TRAINPOOL (train+val)
  - Retrain on full TRAINPOOL using epochs chosen from CV (median best_epoch across folds)
  - Evaluate once on TEST
  - Save: attack_train.log, best_params.json, attack_results.json, model.pt, scaler.pt, roc_curve.csv, ROC log-log plot

Usage:
  python train_mia_attack_cvholdout.py \
    --attack-data-dir experiments/gen_results/paper_arch_compare/saved_models_for_mia \
    --out experiments/gen_results/paper_arch_compare/mia_results \
    --test-ratio 0.2 --cv-folds 5 \
    --tune --n-trials 30 --max-epochs 200 --patience 15
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TextIO

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
    run_id: int
    dataset: str
    architecture: str
    role: str
    X: torch.Tensor
    y_true: torch.Tensor
    y_pred: torch.Tensor
    membership: torch.Tensor  # 0=member, 1=non-member
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


def load_attack_data(path: Path) -> AttackDataset:
    data = torch.load(path, map_location="cpu")
    _require_keys(data, ["X", "y_true", "y_pred", "membership", "split", "pv", "stats"], path)

    meta = data.get("meta", {})
    run_id = int(meta.get("run_id", -1))
    dataset = str(meta.get("dataset", "unknown"))
    architecture = str(meta.get("model_type", "unknown")).upper()
    role = str(meta.get("role", "selected"))

    X = data["X"].float()
    y_true = data["y_true"].long()
    y_pred = data["y_pred"].long()
    membership = data["membership"].long()
    split = data["split"].long()
    pv = data["pv"].float()

    stats = data["stats"]
    if not isinstance(stats, dict):
        raise TypeError(f"Expected stats dict in {path}, got {type(stats)}")

    stats_clean: Dict[str, torch.Tensor] = {}
    for k, v in stats.items():
        stats_clean[k] = v.float() if torch.is_tensor(v) else torch.tensor(v).float()

    return AttackDataset(
        run_id=run_id,
        dataset=dataset,
        architecture=architecture,
        role=role,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        membership=membership,
        split=split,
        pv=pv,
        stats=stats_clean,
        meta=meta,
        source_path=path,
    )


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
# Model
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
    pred = logits.argmax(dim=1).cpu()
    acc = float((pred == y.cpu().long()).float().mean().item())
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
    """
    Trains on fold-train, early-stops using fold-val loss.
    Returns best model (by val loss) and info including best_epoch.
    """
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
    dl = torch.utils.data.DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

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
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # fold-val monitor
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
    """
    Returns:
      mean_auc across folds (higher is better)
      list of best_epoch per fold
    """
    y_np = y_pool.detach().cpu().numpy().astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs: List[float] = []
    best_epochs: List[int] = []

    for tr_idx, va_idx in skf.split(np.zeros_like(y_np), y_np):
        X_tr = X_pool[torch.from_numpy(tr_idx)]
        y_tr = y_pool[torch.from_numpy(tr_idx)]
        X_va = X_pool[torch.from_numpy(va_idx)]
        y_va = y_pool[torch.from_numpy(va_idx)]

        # standardize per fold using fold-train only
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

        # fold val AUC
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
    """
    Tunes params to maximize mean CV AUC on TRAINPOOL only.
    Returns:
      best_params (includes chosen_final_epochs)
      tuning_info
    """
    optuna = _try_import_optuna()

    if optuna is None:
        rng = np.random.default_rng(seed)
        best_params = None
        best_score = -1.0
        best_epochs = None

        for t in range(1, n_trials + 1):
            params = _sample_params_fallback(rng)
            score, epochs = cv_score_params(
                X_pool, y_pool,
                params=params,
                device=device,
                max_epochs=max_epochs,
                patience=patience,
                n_splits=n_splits,
                seed=seed + t,
            )
            if score > best_score:
                best_score = score
                best_params = params
                best_epochs = epochs

        assert best_params is not None and best_epochs is not None
        chosen_epochs = int(np.median(best_epochs))
        best_params["chosen_final_epochs"] = max(1, min(chosen_epochs, max_epochs))

        return best_params, {
            "method": "random_search",
            "best_cv_auc": float(best_score),
            "fold_best_epochs": [int(e) for e in best_epochs],
            "chosen_final_epochs": int(best_params["chosen_final_epochs"]),
            "n_trials": int(n_trials),
            "n_splits": int(n_splits),
        }

    # Optuna path
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
        score, epochs = cv_score_params(
            X_pool, y_pool,
            params=params,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            n_splits=n_splits,
            seed=seed + trial.number,
        )
        trial.set_user_attr("fold_best_epochs", [int(e) for e in epochs])
        trial.set_user_attr("chosen_final_epochs", int(np.median(epochs)))
        return score  # maximize

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    fold_epochs = list(study.best_trial.user_attrs.get("fold_best_epochs", []))
    chosen_epochs = int(study.best_trial.user_attrs.get("chosen_final_epochs", max_epochs))
    best_params["chosen_final_epochs"] = max(1, min(chosen_epochs, max_epochs))

    tuning_info = {
        "method": "optuna",
        "best_cv_auc": float(study.best_value),
        "fold_best_epochs": [int(e) for e in fold_epochs],
        "chosen_final_epochs": int(best_params["chosen_final_epochs"]),
        "n_trials": int(n_trials),
        "n_splits": int(n_splits),
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
    """
    Train on entire TRAINPOOL for a fixed number of epochs (derived from CV),
    logging TRAIN loss/acc only. (No test during training.)
    """
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


# -------------------------
# One target end-to-end (Option 1)
# -------------------------
import random
import secrets

import numpy as np
import torch

seed = secrets.randbits(32)
print(f"Random seed for this run: {seed}")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
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
    device: str,
    seed: int,
    verbose: bool = True,
) -> Dict[str, Any]:

    ds = load_attack_data(attack_data_path)

    run_out_dir = out_dir / f"{ds.dataset}_{ds.architecture}_run{ds.run_id}"
    run_out_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_out_dir / "attack_train.log"
    best_params_path = run_out_dir / "best_params.json"

    X = ds.X
    y = ds.membership  # 0/1
    y_np = y.detach().cpu().numpy().astype(int)

    # safety: ensure both classes exist
    n0 = int((y == 0).sum().item())
    n1 = int((y == 1).sum().item())
    if n0 < 2 or n1 < 2:
        raise ValueError(f"Too few samples per class: members={n0}, nonmembers={n1} in {attack_data_path}")

    # choose feasible CV folds
    min_class = min(n0, n1)
    feasible_folds = min(cv_folds, min_class)
    if feasible_folds < 2:
        # cannot do CV; fall back to no tuning with fixed params
        feasible_folds = 1

    # Holdout TEST (stratified)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    (pool_idx, test_idx) = next(sss.split(np.zeros_like(y_np), y_np))

    pool_idx_t = torch.from_numpy(pool_idx)
    test_idx_t = torch.from_numpy(test_idx)

    X_pool = X[pool_idx_t]
    y_pool = y[pool_idx_t]
    X_test = X[test_idx_t]
    y_test = y[test_idx_t]

    # Standardize using TRAINPOOL only (fit once for final training/eval)
    mu_pool, sd_pool = standardize_fit(X_pool)
    X_pool_s = standardize_apply(X_pool, mu_pool, sd_pool)
    X_test_s = standardize_apply(X_test, mu_pool, sd_pool)

    # Tune on TRAINPOOL via CV
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
        tuning_info = {"method": "fixed_or_no_cv", "note": f"feasible_folds={feasible_folds}"}

    # Save params
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump({"best_params": best_params, "tuning_info": tuning_info}, f, indent=2, default=_json_default)

    # Final train on entire TRAINPOOL (no val, no test peeking)
    final_epochs = int(best_params.get("chosen_final_epochs", max_epochs))
    final_epochs = max(1, min(final_epochs, max_epochs))

    with open(log_path, "w", encoding="utf-8") as log_fh:
        log_fh.write(f"Source: {attack_data_path}\n")
        log_fh.write(f"Target: {ds.architecture} on {ds.dataset} (run_id={ds.run_id})\n")
        log_fh.write(f"Members={n0}, NonMembers={n1}\n")
        log_fh.write(f"Holdout test_ratio={test_ratio} | TEST size={len(X_test)} | TRAINPOOL size={len(X_pool)}\n")
        log_fh.write(f"CV folds requested={cv_folds}, feasible={feasible_folds}\n")
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

        # Evaluate once on TEST
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

    # Save model + scaler for reproducibility
    torch.save(model.state_dict(), run_out_dir / "attack_model.pt")
    torch.save({"mu": mu_pool, "sd": sd_pool}, run_out_dir / "scaler.pt")

    # Save ROC CSV + plot
    roc_csv = run_out_dir / "roc_curve.csv"
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(roc_csv, index=False)

    title = f"ROC (log-log): {ds.architecture} on {ds.dataset} (run_id={ds.run_id})"
    plot_and_save_roc_loglog(fpr, tpr, title=title, save_prefix=run_out_dir / "attack", eps=1e-6)

    # Save results JSON
    results = {
        "target_meta": ds.meta,
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
        },
    }
    with open(run_out_dir / "attack_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)

    if verbose:
        print(f"[Saved] {run_out_dir}")

    return results


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Option 1: holdout test, tune via CV on trainpool")

    ap.add_argument("--attack-data-dir", type=str, required=True,
                    help="Root directory containing *_attack_data.pt files")
    ap.add_argument("--out", type=str, required=True,
                    help="Output directory for attack results")

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

    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = "cpu"

    files = find_attack_data_files(Path(args.attack_data_dir))
    if not files:
        print(f"No *_attack_data.pt files found in {args.attack_data_dir}")
        return

    # optional filtering based on meta
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

    print(f"Found {len(files)} attack files (after filtering)")

    all_results: List[Dict[str, Any]] = []
    for f in files:
        try:
            r = run_one_target(
                attack_data_path=f,
                out_dir=out_dir,
                test_ratio=args.test_ratio,
                cv_folds=args.cv_folds,
                tune=args.tune,
                n_trials=args.n_trials,
                max_epochs=args.max_epochs,
                patience=args.patience,
                device=device,
                seed=args.seed,
                verbose=True,
            )
            all_results.append(r)
        except Exception as e:
            print(f"ERROR processing {f}: {e}")
            continue

    # summary CSV
    rows = []
    for r in all_results:
        meta = r.get("target_meta", {})
        m = r.get("metrics_test", {})
        tpr_map = m.get("tpr_at_low_fpr", {}) if isinstance(m, dict) else {}
        rows.append({
            "dataset": meta.get("dataset", "unknown"),
            "architecture": str(meta.get("model_type", "unknown")).upper(),
            "run_id": meta.get("run_id", -1),
            "n_wires": meta.get("n_wires", -1),
            "depth": meta.get("depth", -1),
            "attack_acc": m.get("accuracy", np.nan),
            "attack_auc": m.get("auc_roc", np.nan),
            "attack_f1": m.get("f1", np.nan),
            "attack_precision": m.get("precision", np.nan),
            "attack_recall": m.get("recall", np.nan),
            "tpr@fpr=0.001": tpr_map.get("tpr@fpr=0.001", 0.0),
            "tpr@fpr=0.01": tpr_map.get("tpr@fpr=0.01", 0.0),
            "tpr@fpr=0.05": tpr_map.get("tpr@fpr=0.05", 0.0),
            "tpr@fpr=0.1": tpr_map.get("tpr@fpr=0.1", 0.0),
            "tuning_method": r.get("tuning_info", {}).get("method", "unknown"),
            "cv_best_auc": r.get("tuning_info", {}).get("best_cv_auc", np.nan),
            "final_epochs": r.get("final_epochs", np.nan),
        })

    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / "attack_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Processed {len(all_results)} targets")
    print(f"Results dir: {out_dir}")
    print(f"Summary CSV: {summary_path}")
    if len(summary_df) > 0 and "attack_auc" in summary_df.columns:
        print(f"Mean AUC-ROC: {summary_df['attack_auc'].mean():.4f}")


if __name__ == "__main__":
    main()
