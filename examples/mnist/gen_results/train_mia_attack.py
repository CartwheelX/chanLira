# #!/usr/bin/env python3
# """
# train_mia_attack.py

# Trains MLP-based membership inference attacks on saved prediction vectors.
# Reads *_attack_data.pt files generated during target model training.

# Usage:
#   python train_mia_attack.py \
#     --attack-data-dir examples/mnist/gen_results/paper_arch_compare/saved_models_for_mia \
#     --out results/mia_attacks \
#     --attack-train-split 0.7
# """

# import argparse
# import json
# from dataclasses import dataclass, asdict
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.metrics import (
#     accuracy_score,
#     average_precision_score,
#     confusion_matrix,
#     f1_score,
#     precision_score,
#     recall_score,
#     roc_auc_score,
# )
# from sklearn.metrics import roc_curve


# # ==================== DATA STRUCTURES ====================
# @dataclass
# class AttackDataset:
#     """Container for loaded attack data from a single target model."""
#     run_id: int
#     dataset: str
#     architecture: str
#     role: str
#     X: torch.Tensor               # [N, features]
#     y_true: torch.Tensor          # [N]
#     y_pred: torch.Tensor          # [N]
#     membership: torch.Tensor      # [N] - 0=member(train), 1=non-member(test)
#     split: torch.Tensor           # [N] - 0=train, 1=test (from target model)
#     pv: torch.Tensor              # [N, num_classes]
#     stats: Dict[str, torch.Tensor]
#     meta: Dict[str, Any]
#     source_path: Path


# @dataclass
# class AttackMetrics:
#     """Standard MIA evaluation metrics."""
#     accuracy: float
#     precision: float
#     recall: float
#     f1: float
#     auc_roc: float
#     auc_pr: float
#     tpr_at_low_fpr: Dict[str, float]  # TPR @ FPR=0.01, 0.05, 0.1
#     tn: int
#     fp: int
#     fn: int
#     tp: int


# # ==================== MLP ATTACK MODEL ====================
# class MLPAttackModel(nn.Module):
#     """Simple MLP for binary membership classification."""
#     def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(hidden_dim, max(2, hidden_dim // 2))
#         self.fc3 = nn.Linear(max(2, hidden_dim // 2), 2)  # classes: 0/1

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)


# # ==================== DATA LOADING ====================
# def find_attack_data_files(root_dir: Path) -> List[Path]:
#     """Recursively find all *_attack_data.pt files."""
#     return sorted(root_dir.rglob("*_attack_data.pt"))


# def _require_keys(d: Dict[str, Any], keys: List[str], path: Path) -> None:
#     missing = [k for k in keys if k not in d]
#     if missing:
#         raise KeyError(f"Missing keys {missing} in {path}")


# def load_attack_data(path: Path) -> AttackDataset:
#     """Load a single *_attack_data.pt file."""
#     data = torch.load(path, map_location="cpu")
#     _require_keys(data, ["X", "y_true", "y_pred", "membership", "split", "pv", "stats"], path)

#     meta = data.get("meta", {})
#     run_id = int(meta.get("run_id", -1))
#     dataset = str(meta.get("dataset", "unknown"))
#     architecture = str(meta.get("model_type", "unknown")).upper()
#     role = "selected"  # optionally parse from path if you want

#     # Ensure sane dtypes
#     X = data["X"].float()
#     y_true = data["y_true"].long()
#     y_pred = data["y_pred"].long()
#     membership = data["membership"].long()
#     split = data["split"].long()
#     pv = data["pv"].float()

#     stats = data["stats"]
#     if not isinstance(stats, dict):
#         raise TypeError(f"Expected stats to be a dict in {path}, got {type(stats)}")

#     # Force any tensor stats to float for consistency
#     stats_clean: Dict[str, torch.Tensor] = {}
#     for k, v in stats.items():
#         stats_clean[k] = v.float() if torch.is_tensor(v) else torch.tensor(v).float()

#     return AttackDataset(
#         run_id=run_id,
#         dataset=dataset,
#         architecture=architecture,
#         role=role,
#         X=X,
#         y_true=y_true,
#         y_pred=y_pred,
#         membership=membership,
#         split=split,
#         pv=pv,
#         stats=stats_clean,
#         meta=meta,
#         source_path=path,
#     )


# def split_for_attack(
#     dataset: AttackDataset,
#     train_ratio: float = 0.7,
#     seed: int = 42
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Stratified split on membership labels:
#       - train_ratio of each class → attack_train
#       - remainder → attack_test
#     """
#     if not (0.0 < train_ratio < 1.0):
#         raise ValueError("train_ratio must be in (0, 1)")

#     X = dataset.X
#     y = dataset.membership  # 0=member, 1=non-member

#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     indices = torch.arange(len(y))
#     member_idx = indices[y == 0]
#     nonmember_idx = indices[y == 1]

#     if len(member_idx) < 2 or len(nonmember_idx) < 2:
#         raise ValueError(
#             f"Not enough samples per class to split (members={len(member_idx)}, nonmembers={len(nonmember_idx)}) "
#             f"in {dataset.source_path}"
#         )

#     member_idx = member_idx[torch.randperm(len(member_idx))]
#     nonmember_idx = nonmember_idx[torch.randperm(len(nonmember_idx))]

#     n_train_mem = max(1, int(len(member_idx) * train_ratio))
#     n_train_nonmem = max(1, int(len(nonmember_idx) * train_ratio))

#     # Keep at least 1 in test too
#     n_train_mem = min(n_train_mem, len(member_idx) - 1)
#     n_train_nonmem = min(n_train_nonmem, len(nonmember_idx) - 1)

#     train_idx = torch.cat([member_idx[:n_train_mem], nonmember_idx[:n_train_nonmem]])
#     test_idx = torch.cat([member_idx[n_train_mem:], nonmember_idx[n_train_nonmem:]])

#     train_idx = train_idx[torch.randperm(len(train_idx))]
#     test_idx = test_idx[torch.randperm(len(test_idx))]

#     return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# @torch.no_grad()
# def _loss_and_acc(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion, device: str):
#     model.eval()
#     Xd = X.to(device)
#     yd = y.to(device).long()
#     logits = model(Xd)
#     loss = float(criterion(logits, yd).item())
#     pred = logits.argmax(dim=1)
#     acc = float((pred == yd).float().mean().item())
#     return loss, acc

# from typing import TextIO, Optional
# # ==================== TRAINING ====================
# from typing import TextIO, Optional

# def train_attack_model(
#     X_train: torch.Tensor,
#     y_train: torch.Tensor,
#     X_test: torch.Tensor,
#     y_test: torch.Tensor,
#     *,
#     hidden_dim: int = 128,
#     dropout: float = 0.3,
#     lr: float = 1e-3,
#     epochs: int = 50,
#     batch_size: int = 64,
#     device: str = "cuda",
#     verbose: bool = True,
#     log_fh: Optional[TextIO] = None,   # <-- ADD THIS
# ) -> Tuple[MLPAttackModel, List[float]]:
#     """Train MLP attack model (prints + logs Loss/Acc for train & test)."""

#     def _log_line(s: str):
#         if verbose:
#             print(s)
#         if log_fh is not None:
#             print(s, file=log_fh, flush=True)

#     if batch_size <= 0:
#         raise ValueError("batch_size must be > 0")

#     input_dim = int(X_train.shape[1])
#     model = MLPAttackModel(input_dim, hidden_dim, dropout).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
#     criterion = nn.CrossEntropyLoss()

#     train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     test_losses: List[float] = []

#     _log_line("Epoch |    Loss (train / test) |   Acc (train / test)")
#     _log_line("----- | --------------------- | -------------------")

#     for epoch in range(1, epochs + 1):
#         model.train()
#         for X_batch, y_batch in train_loader:
#             X_batch = X_batch.to(device)
#             y_batch = y_batch.to(device).long()

#             optimizer.zero_grad(set_to_none=True)
#             logits = model(X_batch)
#             loss = criterion(logits, y_batch)
#             loss.backward()
#             optimizer.step()

#         # Evaluate on full splits
#         train_loss, train_acc = _loss_and_acc(model, X_train, y_train, criterion, device)
#         test_loss, test_acc = _loss_and_acc(model, X_test, y_test, criterion, device)
#         test_losses.append(test_loss)

#         _log_line(f"{epoch:5d} | {train_loss:12.4f} / {test_loss:7.4f} | {train_acc:7.3f} / {test_acc:7.3f}")

#     return model, test_losses

# # ==================== EVALUATION ====================
# @torch.no_grad()
# def evaluate_attack(
#     model: MLPAttackModel,
#     X_test: torch.Tensor,
#     y_test: torch.Tensor,
#     device: str = "cuda"
# ):
#     model.eval()

#     logits = model(X_test.to(device))
#     probs = F.softmax(logits, dim=1).cpu().numpy()

#     y_test_np = y_test.cpu().numpy().astype(int)
#     y_pred = probs.argmax(axis=1).astype(int)

#     # Score = P(non-member)
#     scores = probs[:, 1]

#     acc = accuracy_score(y_test_np, y_pred)
#     prec = precision_score(y_test_np, y_pred, zero_division=0)
#     rec = recall_score(y_test_np, y_pred, zero_division=0)
#     f1 = f1_score(y_test_np, y_pred, zero_division=0)

#     try:
#         auc_roc = roc_auc_score(y_test_np, scores)
#     except Exception:
#         auc_roc = 0.5
#     try:
#         auc_pr = average_precision_score(y_test_np, scores)
#     except Exception:
#         auc_pr = 0.0

#     fpr, tpr, thresholds = roc_curve(y_test_np, scores)

#     tpr_at_fpr: Dict[str, float] = {}
#     for target_fpr in [0.01, 0.05, 0.1]:
#         idx = np.where(fpr <= target_fpr)[0]
#         tpr_at_fpr[f"tpr@fpr={target_fpr}"] = float(tpr[idx[-1]]) if len(idx) > 0 else 0.0

#     tn, fp, fn, tp = confusion_matrix(y_test_np, y_pred, labels=[0, 1]).ravel()

#     metrics = AttackMetrics(
#         accuracy=float(acc),
#         precision=float(prec),
#         recall=float(rec),
#         f1=float(f1),
#         auc_roc=float(auc_roc),
#         auc_pr=float(auc_pr),
#         tpr_at_low_fpr=tpr_at_fpr,
#         tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
#     )

#     # return both
#     return metrics, fpr, tpr, thresholds

# def get_attack_model_config(n_samples: int, input_dim: int) -> dict:
#     """Adaptive MLP configuration based on dataset size."""
#     if n_samples < 150:
#         return {
#             "hidden_dim": min(64, input_dim * 8),
#             "dropout": 0.4,
#             "lr": 5e-4,
#             "epochs": 80,
#         }
#     return {
#         "hidden_dim": min(128, input_dim * 16),
#         "dropout": 0.3,
#         "lr": 1e-3,
#         "epochs": 50,
#     }

# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

# def plot_tsne_membership(dataset, max_points=5000, pca_dim=50, perplexity=30, seed=0, save_path=None):
#     # ---- extract ----
#     X = dataset.X
#     m = dataset.membership

#     # torch -> numpy if needed
#     try:
#         import torch
#         if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()
#         if isinstance(m, torch.Tensor): m = m.detach().cpu().numpy()
#     except Exception:
#         pass

#     X = np.asarray(X)
#     m = np.asarray(m).astype(int).reshape(-1)

#     n = X.shape[0]
#     if n != len(m):
#         raise ValueError(f"X has {n} rows but membership has {len(m)} entries")

#     # ---- optional subsample for speed ----
#     rng = np.random.default_rng(seed)
#     if n > max_points:
#         idx = rng.choice(n, size=max_points, replace=False)
#         Xs, ms = X[idx], m[idx]
#     else:
#         Xs, ms = X, m

#     # ---- PCA pre-reduction (recommended for t-SNE) ----
#     pca_out_dim = min(pca_dim, Xs.shape[1], Xs.shape[0] - 1)
#     if pca_out_dim >= 2:
#         Xp = PCA(n_components=pca_out_dim, random_state=seed).fit_transform(Xs)
#     else:
#         Xp = Xs

#     # ---- t-SNE ----
#     # sklearn changed arg names across versions (n_iter vs max_iter), so we try both
#     try:
#         tsne = TSNE(
#             n_components=2,
#             perplexity=min(perplexity, max(5, (len(Xp) - 1) // 3)),
#             learning_rate="auto",
#             init="pca",
#             max_iter=1500,
#             random_state=seed,
#         )
#     except TypeError:
#         tsne = TSNE(
#             n_components=2,
#             perplexity=min(perplexity, max(5, (len(Xp) - 1) // 3)),
#             learning_rate="auto",
#             init="pca",
#             n_iter=1500,
#             random_state=seed,
#         )

#     Z = tsne.fit_transform(Xp)

#     # ---- plot ----
#     fig, ax = plt.subplots(figsize=(7.5, 6), facecolor="white")
#     ax.scatter(Z[ms == 0, 0], Z[ms == 0, 1], s=10, alpha=0.75, label="Members (0)")
#     ax.scatter(Z[ms == 1, 0], Z[ms == 1, 1], s=10, alpha=0.75, label="Non-members (1)")
#     ax.set_title(f"t-SNE: {dataset.architecture} on {dataset.dataset} (run_id={dataset.run_id})", fontweight="normal")
#     ax.set_xlabel("t-SNE-1")
#     ax.set_ylabel("t-SNE-2")
#     ax.grid(True, alpha=0.25)
#     ax.legend(frameon=True)
#     plt.tight_layout()

#     if save_path:
#         fig.savefig(save_path, dpi=300, bbox_inches="tight")
#         print(f"[Saved] {save_path}")

#     plt.show()
#     return Z

# def _json_default(o: Any):
#     if isinstance(o, Path):
#         return str(o)
#     if torch.is_tensor(o):
#         if o.numel() == 1:
#             return o.item()
#         return o.detach().cpu().tolist()
#     if isinstance(o, (np.integer, np.floating)):
#         return o.item()
#     return str(o)


# def run_attack_on_single_target(
#     attack_data_path: Path,
#     out_dir: Path,
#     *,
#     train_ratio: float = 0.7,
#     device: str = "cuda",
#     seed: int = 42,
#     verbose: bool = True,
#     hidden_dim: Optional[int] = None,
#     dropout: Optional[float] = None,
#     lr: Optional[float] = None,
#     epochs: Optional[int] = None,
#     batch_size: Optional[int] = None,
#     force_config: Optional[Dict[str, Any]] = None,
# ) -> Dict[str, Any]:
#     """Run complete MIA attack pipeline on one target model."""
#     if verbose:
#         print(f"\n{'='*60}")
#         print(f"Processing: {attack_data_path.name}")
#         print(f"{'='*60}")

#     dataset = load_attack_data(attack_data_path)

#     if verbose:
#         print(f"Target: {dataset.architecture} on {dataset.dataset} (run_id={dataset.run_id})")
#         print(f"Total samples: {len(dataset.X)}")
#         print(f"  Members: {(dataset.membership == 0).sum().item()}")
#         print(f"  Non-members: {(dataset.membership == 1).sum().item()}")
#         print(f"Feature dim: {dataset.X.shape[1]}")
    
#     # print few  dataset.X
#     # print(f"dataset.X: {dataset.X}")
    
#     # plot dataset.X sne scatter
    
#     # plot_tsne_membership(dataset, max_points=5000, perplexity=30, save_path="tsne_membership.png")
    
#     # k = 5  # how many samples

#     # X = dataset.X
#     # m = dataset.membership

#     # print("First few samples:")
#     # for i in range(min(k, len(X))):
#     #     print(f"[{i}] membership={int(m[i])}  X[:10]={X[i, :10]}")
        
#     # exit()
#     X_train, y_train, X_test, y_test = split_for_attack(dataset, train_ratio=train_ratio, seed=seed)

#     # print(f"X_train shape: {X_train.shape}")
#     # print(f"y_train shape: {y_train.shape}")
#     # print(f"X_test shape: {X_test.shape}")
#     # print(f"y_test shape: {y_test.shape}")

#     # print(f"x_train: {X_train}, y_train: {y_train}")
#     # exit()


#     input_dim = int(X_train.shape[1])
#     n_samples = int(len(X_train))

#     if force_config is not None:
#         config = dict(force_config)
#     else:
#         config = get_attack_model_config(n_samples, input_dim)

#     # Override adaptive config with CLI values (if provided)
#     if hidden_dim is not None:
#         config["hidden_dim"] = int(hidden_dim)
#     if dropout is not None:
#         config["dropout"] = float(dropout)
#     if lr is not None:
#         config["lr"] = float(lr)
#     if epochs is not None:
#         config["epochs"] = int(epochs)

#     # Batch size: explicit > adaptive
#     if batch_size is None:
#         # reasonable adaptive default; guaranteed >= 1
#         batch_size_use = max(1, min(64, n_samples // 4))
#     else:
#         batch_size_use = int(batch_size)

#     # --- create run folder + log path BEFORE opening ---
#     run_out_dir = out_dir / f"{dataset.dataset}_{dataset.architecture}_run{dataset.run_id}"
#     run_out_dir.mkdir(parents=True, exist_ok=True)
#     attack_log_path = run_out_dir / "attack_train.log"

#     if verbose:
#         print("\nTraining MLP attack model...")

#     with open(attack_log_path, "w", encoding="utf-8") as log_fh:
#         log_fh.write(f"Target: {dataset.architecture} on {dataset.dataset} (run_id={dataset.run_id})\n")
#         log_fh.write(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}\n")
#         log_fh.write(f"Config: {config}\n")
#         log_fh.write(f"Batch size: {batch_size_use}\n\n")
#         log_fh.flush()

#         model, _ = train_attack_model(
#             X_train, y_train, X_test, y_test,
#             hidden_dim=config["hidden_dim"],
#             dropout=config["dropout"],
#             lr=config["lr"],
#             epochs=config["epochs"],
#             batch_size=batch_size_use,
#             device=device,
#             verbose=True,
#             log_fh=log_fh,
#         )

#     metrics, fpr, tpr, thr = evaluate_attack(model, X_test, y_test, device=device)

#     torch.save(model.state_dict(), run_out_dir / "attack_model.pt")


#     results: Dict[str, Any] = {
#         "target_meta": dataset.meta,
#         "attack_config": config,
#         "metrics": asdict(metrics),
#         "train_samples": int(len(X_train)),
#         "test_samples": int(len(X_test)),
#     }

#     with open(run_out_dir / "attack_results.json", "w") as f:
#         json.dump(results, f, indent=2, default=_json_default)

#     if verbose:
#         print(f"\nSaved to: {run_out_dir}")

#     roc_csv = run_out_dir / "roc_curve.csv"
#     pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(roc_csv, index=False)

#     # Plot title (paper-friendly, not bold)
#     title = f"ROC: {dataset.architecture} on {dataset.dataset} (run_id={dataset.run_id})"

#     # Save ROC plots
#     # save_prefix = run_out_dir / "attack"
#     # plot_and_save_roc(fpr, tpr, title=title, save_prefix=save_prefix, log_fpr=False)
#     # plot_and_save_roc(fpr, tpr, title=title, save_prefix=save_prefix, log_fpr=True)
#     title = f"ROC (log-log): {dataset.architecture} on {dataset.dataset} (run_id={dataset.run_id})"
#     save_prefix = run_out_dir / "attack"
#     plot_and_save_roc_loglog(fpr, tpr, title=title, save_prefix=save_prefix, eps=1e-6)


#     return results


# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# def plot_and_save_roc_loglog(
#     fpr: np.ndarray,
#     tpr: np.ndarray,
#     *,
#     title: str,
#     save_prefix: Path,
#     eps: float = 1e-6,
#     font_family: str = "serif",
# ):
#     """
#     Saves ROC with BOTH axes log-scaled:
#       - <save_prefix>_roc_loglog.png/.pdf

#     NOTE: fpr/tpr are clamped to >= eps to avoid log(0).
#     """
#     plt.rcParams["font.family"] = font_family

#     fpr = np.asarray(fpr, dtype=float)
#     tpr = np.asarray(tpr, dtype=float)

#     fpr_c = np.clip(fpr, eps, 1.0)
#     tpr_c = np.clip(tpr, eps, 1.0)

#     fig, ax = plt.subplots(figsize=(6.5, 5.2), facecolor="white")

#     ax.plot(fpr_c, tpr_c, linewidth=2)

#     # Diagonal baseline in log-log space (also clamped)
#     diag = np.linspace(eps, 1.0, 200)
#     ax.plot(diag, diag, linestyle="--", linewidth=1)

#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.set_xlim(eps, 1.0)
#     ax.set_ylim(eps, 1.0)

#     ax.set_title(title, fontweight="normal")
#     ax.set_xlabel("False Positive Rate (FPR)", fontweight="normal")
#     ax.set_ylabel("True Positive Rate (TPR)", fontweight="normal")
#     ax.grid(True, which="both", alpha=0.25)

#     plt.tight_layout()

#     # out_png = str(save_prefix) + "_roc_loglog.png"
#     out_pdf = str(save_prefix) + "_roc_loglog.pdf"
#     # fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
#     fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
#     plt.close(fig)

#     # return out_png, out_pdf
#     return out_pdf

# # ==================== MAIN PIPELINE ====================
# def main():
#     parser = argparse.ArgumentParser(description="Train MIA attacks on saved PVs")

#     parser.add_argument("--attack-data-dir", type=str, required=True,
#                         help="Root directory containing *_attack_data.pt files")
#     parser.add_argument("--out", type=str, default="results/mia_attacks",
#                         help="Output directory for attack results")

#     parser.add_argument("--attack-train-split", type=float, default=0.7,
#                         help="Fraction of attack data used for training (rest for testing)")
#     parser.add_argument("--hidden-dim", type=int, default=128,
#                         help="Override hidden layer size (set <=0 to use adaptive)")
#     parser.add_argument("--dropout", type=float, default=0.3,
#                         help="Override dropout (set <0 to use adaptive)")
#     parser.add_argument("--lr", type=float, default=1e-3,
#                         help="Override learning rate (set <=0 to use adaptive)")
#     parser.add_argument("--epochs", type=int, default=100,
#                         help="Override training epochs (set <=0 to use adaptive)")
#     parser.add_argument("--batch-size", type=int, default=64,
#                         help="Batch size (set <=0 to use adaptive)")

#     parser.add_argument("--seed", type=int, default=42, help="Random seed")
#     parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
#                         help="Device for training")

#     parser.add_argument("--filter-dataset", type=str, default=None,
#                         help="Only process this dataset (e.g., 'mnist')")
#     parser.add_argument("--filter-arch", type=str, default=None,
#                         help="Only process this architecture (e.g., 'QNN')")

#     args = parser.parse_args()

#     attack_data_dir = Path(args.attack_data_dir)
#     out_dir = Path(args.out)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     device = args.device
#     if device == "cuda" and not torch.cuda.is_available():
#         print("CUDA not available; falling back to CPU.")
#         device = "cpu"

#     attack_files = find_attack_data_files(attack_data_dir)
#     print(f"Found {len(attack_files)} attack data files")

#     # print(f"attack_files: {attack_files}")
#     # exit()
#     if not attack_files:
#         print(f"No *_attack_data.pt files found in {attack_data_dir}")
#         return

#     # Optional filtering (loads meta to decide)
#     if args.filter_dataset or args.filter_arch:
#         filtered: List[Path] = []
#         for f in attack_files:
#             try:
#                 ds = load_attack_data(f)
#                 if args.filter_dataset and ds.dataset.lower() != args.filter_dataset.lower():
#                     continue
#                 if args.filter_arch and ds.architecture.upper() != args.filter_arch.upper():
#                     continue
#                 filtered.append(f)
#             except Exception:
#                 continue
#         attack_files = filtered
#         print(f"After filtering: {len(attack_files)} files")

#     # Interpret "use adaptive" controls
#     hidden_dim = None if args.hidden_dim <= 0 else args.hidden_dim
#     dropout = None if args.dropout < 0 else args.dropout
#     lr = None if args.lr <= 0 else args.lr
#     epochs = None if args.epochs <= 0 else args.epochs
#     batch_size = None if args.batch_size <= 0 else args.batch_size

#     all_results: List[Dict[str, Any]] = []
#     for attack_file in attack_files:
#         try:
#             results = run_attack_on_single_target(
#                 attack_data_path=attack_file,
#                 out_dir=out_dir,
#                 train_ratio=args.attack_train_split,
#                 hidden_dim=hidden_dim,
#                 dropout=dropout,
#                 lr=lr,
#                 epochs=epochs,
#                 batch_size=batch_size,
#                 device=device,
#                 seed=args.seed,
#                 verbose=True,
#             )
#             all_results.append(results)
#         except Exception as e:
#             print(f"ERROR processing {attack_file}: {e}")
#             continue

#     # Summary CSV
#     summary_rows: List[Dict[str, Any]] = []
#     for r in all_results:
#         meta = r.get("target_meta", {})
#         metrics = r.get("metrics", {})
#         tpr_map = metrics.get("tpr_at_low_fpr", {}) if isinstance(metrics, dict) else {}
#         summary_rows.append({
#             "dataset": meta.get("dataset", "unknown"),
#             "architecture": str(meta.get("model_type", "unknown")).upper(),
#             "run_id": meta.get("run_id", -1),
#             "n_wires": meta.get("n_wires", -1),
#             "depth": meta.get("depth", -1),
#             "attack_acc": metrics.get("accuracy", np.nan),
#             "attack_auc": metrics.get("auc_roc", np.nan),
#             "attack_f1": metrics.get("f1", np.nan),
#             "attack_precision": metrics.get("precision", np.nan),
#             "attack_recall": metrics.get("recall", np.nan),
#             "tpr@fpr=0.01": tpr_map.get("tpr@fpr=0.01", 0.0),
#             "tpr@fpr=0.05": tpr_map.get("tpr@fpr=0.05", 0.0),
#             "tpr@fpr=0.1": tpr_map.get("tpr@fpr=0.1", 0.0),
#         })

#     summary_df = pd.DataFrame(summary_rows)
#     summary_path = out_dir / "attack_summary.csv"
#     summary_df.to_csv(summary_path, index=False)

#     print(f"\n{'='*60}")
#     print("SUMMARY")
#     print(f"{'='*60}")
#     print(f"Processed {len(all_results)} targets")
#     print(f"Results saved to: {out_dir}")
#     print(f"Summary CSV: {summary_path}")

#     if len(summary_df) > 0 and "attack_acc" in summary_df.columns:
#         print(f"\nMean Attack Accuracy: {summary_df['attack_acc'].mean():.4f}")
#     if len(summary_df) > 0 and "attack_auc" in summary_df.columns:
#         print(f"Mean Attack AUC-ROC:  {summary_df['attack_auc'].mean():.4f}")


# if __name__ == "__main__":
#     main()



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
    --attack-data-dir examples/mnist/gen_results/paper_arch_compare/saved_models_for_mia \
    --out examples/mnist/gen_results/paper_arch_compare/mia_results \
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
