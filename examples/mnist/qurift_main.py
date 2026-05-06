"""QuRiFT main experiment driver.

QuRiFT (Quantum Risk and Inference Fault-line Tracer) is the controlled audit
framework used for structural privacy analysis in QML experiments.
"""

import os
import subprocess


import sys

print("All arguments:", sys.argv)
if len(sys.argv) > 1:
    print(f"The first argument is: {sys.argv[1]}")

def get_best_gpu():
    """Returns GPU index with most free memory."""
    try:
        cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd.split()).decode("utf-8")
        lines = [ln.strip() for ln in output.strip().split("\n") if ln.strip()]
        lines.sort(key=lambda x: int(x.split(",")[1]), reverse=True)
        return lines[0].split(",")[0].strip()
    except Exception:
        return "0"


if "CUDA_VISIBLE_DEVICES" not in os.environ:
    best_gpu_id = get_best_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = best_gpu_id
    print(f"[GPU] Auto-selected GPU ID: {best_gpu_id}")
else:
    print(f"[GPU] Respecting CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")


# # 3. Normal PyTorch setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")



# License: MIT (TorchQuantum & this file)
import argparse
import random
from dataclasses import dataclass
from typing import Iterable, Sequence
import re
# import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# from examples.mnist.feature_encoder_factory import build_feature_map
# from examples.torchquantum_privacy.torchquantum.torchquantum.dataset.cifar10 import CIFAR10
import torchquantum as tq
from torchquantum.measurement import expval_joint_analytical
from torchquantum.dataset import MNIST, CIFAR10
from torch.optim.lr_scheduler import CosineAnnealingLR
# import encoder_oplist_factory as oplist_factory
from encoder_oplist_factory import *
from efficient_su2_generator import *
from z_feature_enc_generator import *
from zz_feature_enc_generator import *
from general_encoder_plus import *
from general_encoder_plus_new import *
from efficient_su2_from_qiskit import *
from sklearn.datasets import make_moons, make_circles, make_blobs, make_multilabel_classification
from pauli_op_generator import *
from efficient_su2_3w import * #dictionat file here only
from qiskit import QuantumCircuit as QkCircuit
from qiskit.circuit import ParameterVector
from torchquantum.plugin import (
    tq2qiskit_measurement,
    qiskit_assemble_circs,
    op_history2qiskit,
    op_history2qiskit_expand_params,
    tq2qiskit,
    op_history2qasm  
)
from torchquantum.layer.entanglement import EntangleCircular
from torchquantum.layer.entanglement import EntangleFull
from torchquantum.layer.entanglement.op2_layer import Op2QButterflyLayer
from torchquantum.layer.entanglement import EntanglePairwise
from torchquantum.layer.entanglement import EntangleLinear
from typing import Optional, Sequence, List, Tuple

import numpy as np


@dataclass
class QFCConfig:
    n_wires: int = 2
    depth: int = 5
    n_random_ops: int = 0
    batch_size: int = 32
    device: str = "cpu"
    
    # 1 --- Feature map selector (choose exactly one per run) ---
    encoder_oplist_name: str = "2x8_rxryrzrxryrzrxry" # GENERAL ENCODER IF USED
    
    fm_kind: str = "z"                  # 'z' | 'zz' | 'pauli' | 'eff_su2'
    # ZFeatureMap
    fm_z_reps: int = 1
    fm_z_alpha: float = 1.0
    fm_z_pad_mode: str = "zero"
    # ZZFeatureMap
    fm_zz_reps: int = 1
    fm_zz_alpha: float = 1.0
    fm_zz_entanglement: str = "ring"    # 'linear' | 'ring' | 'full'
    fm_zz_phi: str = "prod"             # 'prod' | 'pi_minus'
    fm_zz_pad_mode: str = "repeatlast"  # 'wrap' | 'repeatlast' | 'pad'
    
    
    # PauliFeatureMap
    fm_pauli_reps: int = 1
    fm_pauli_alpha: float = 1.0
    fm_pauli_entanglement : str = "linear"
    fm_pauli_terms: Sequence[str] = ("Z", "ZZ")  # e.g., ["Z0","Z1","ZZ01","XX01"]
    fm_pali_pad: str = "wrap"
    
    
    # Efficient SU2 **as feature map** (data-bound angles)
    # fm_eff_reps: int = 1
    fm_eff_single_ops: Sequence[str] = ("ry","rz")  # per-qubit stack
    fm_eff_alpha: float = np.pi
    fm_eff_ent_kind: str = "linear"                 # reuse your entangler factory
    fm_eff_twoq_op: str = "cx"
    fm_eff_pad_mod: str = "repeatlast"       # 'wrap' | 'repeatlast' | 'pad'
    # 2 ---  Entangler config
    qlayer_ent_kind: str = "circular"               # 'full' | 'circular' | 'linear' | 'pairwise' | 'butterfly'
    qlayer_twoq_op: str = "cx"
    qlayer_ent_trainable: bool = False
    qlayer_ent_wire_reverse: bool = False
    # 3 ---  Measurement configs
    num_classes: int = 4                     # logits out
    measure_ops: Optional[Sequence[str]] = None   # None => auto-generate like "("XX","YY","ZZ","XY"),  # K = 4"
    measure_pairs: Optional[List[Tuple[int,int]]] = None
    pool_pairs: bool = True                  # True => mean-pool across pairs (fixed feature size)
    pair_topology: str = "disjoint"          # 'disjoint' or 'ring'
    # 4 ---  Image -> encoder
    pool_hw: int = 4
    feature_dim: Optional[int] = None
def _pairs_disjoint(n): return [(i, i+1) for i in range(0, n-1, 2)] + ([(n-1,0)] if n%2 else [])
def _pairs_ring(n):     return [(i, (i+1) % n) for i in range(n)]
def _pairs_full(n):     return [(i, j) for i in range(n) for j in range(i+1, n)]



class VectorDataset(Dataset):
    """
    Lightweight vector dataset compatible with the TorchQuantum MNIST loader API.
    Each sample carries scaled features for quantum encoding plus original coordinates for plotting.
    """
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        scale: Optional[dict] = None,
        class_vectors: Optional[np.ndarray] = None,
    ):
        orig_feats = np.asarray(features, dtype=np.float32)
        feats = orig_feats.copy()
        labs = np.asarray(labels, dtype=np.int64)
        if feats.ndim != 2 or feats.shape[1] <= 0:
            raise ValueError("features must have shape (N, D>0)")
        if labs.shape[0] != feats.shape[0]:
            raise ValueError("features and labels must share the same length")
        if scale is not None:
            feats = _apply_feature_scale(feats, scale)
        self.features = torch.from_numpy(feats)
        self.features = self.features.clamp(-1.0, 1.0)
        self.targets = torch.from_numpy(labs)
        self.coords = torch.from_numpy(orig_feats)
        self.feature_dim = int(self.features.shape[1])
        self.scale = scale
        self.class_vectors = None
        if class_vectors is not None:
            self.class_vectors = torch.from_numpy(class_vectors.astype(np.int8))
    def __len__(self) -> int:
        return int(self.targets.shape[0])
    def __getitem__(self, idx: int):
        feat = self.features[idx]
        return {
            "image": feat,          # keep key name for compatibility with existing training loop
            "digit": self.targets[idx],
            "coords": self.coords[idx],
        }

def _generate_two_moons(n_samples: int, noise: float, separation: float, rng: np.random.RandomState, extra_feats: bool = False):
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer
    X, y = make_moons(
        n_samples=(n_outer, n_inner),
        noise=noise,
        random_state=rng,
        shuffle=True,
    )
    if separation != 0.5:
        # match previous behaviour where inner moon shifted by separation along y-axis
        X[y == 1, 1] += (separation - 0.5)
    
    if extra_feats:
        x1, x2 = X[:, 0:1], X[:, 1:2]
        X = np.concatenate([X, x1 * x2, x1 ** 2], axis=1)  # shape (N, 4)
    return X.astype(np.float32), y.astype(np.int64)

def _generate_circles(n_samples: int, noise: float, factor: float, rng: np.random.RandomState):
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        shuffle=True,
        random_state=rng,
    )
    return X.astype(np.float32), y.astype(np.int64)
def _generate_blobs(
    n_samples: int,
    cluster_std: float,
    center_distance: float,
    n_features: int,
    rng: np.random.RandomState,
):
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    n_features = max(2, int(n_features))
    cnt_a = n_samples // 2
    cnt_b = n_samples - cnt_a
    centers = np.zeros((2, n_features), dtype=np.float32)
    centers[0, 0] = -center_distance / 2.0
    centers[1, 0] = center_distance / 2.0
    X, y = make_blobs(
        n_samples=[cnt_a, cnt_b],
        centers=centers,
        cluster_std=cluster_std,
        random_state=rng,
    )
    return X.astype(np.float32), y.astype(np.int64)
def _generate_multiclass_from_multilabel(
    n_samples: int,
    *,
    n_features: int,
    n_classes: int,
    n_labels: int,
    length: int,
    allow_unlabeled: bool,
    rng: np.random.RandomState,
    base_vectors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")
    X, Y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_labels=n_labels,
        length=length,
        allow_unlabeled=allow_unlabeled,
        random_state=rng,
    )
    combos = Y.astype(np.int8)
    if base_vectors is not None and base_vectors.size > 0:
        combined = np.concatenate([base_vectors.astype(np.int8), combos], axis=0)
        uniques, inverse = np.unique(combined, axis=0, return_inverse=True)
        labels = inverse[base_vectors.shape[0]:]
    else:
        uniques, labels = np.unique(combos, axis=0, return_inverse=True)
    return X.astype(np.float32), labels.astype(np.int64), uniques.astype(np.int8)

def _apply_feature_scale(feats: np.ndarray, scale: dict) -> np.ndarray:
    if "min" not in scale or "max" not in scale or "target" not in scale:
        raise ValueError("scale dict must contain 'min', 'max', and 'target'.")
    feat_min = np.asarray(scale["min"], dtype=np.float32)
    feat_max = np.asarray(scale["max"], dtype=np.float32)
    tgt_min, tgt_max = scale["target"]
    denom = feat_max - feat_min
    denom = np.where(np.abs(denom) < 1e-8, 1.0, denom)
    norm = (feats - feat_min) / denom
    norm = np.clip(norm, 0.0, 1.0)
    return norm * (tgt_max - tgt_min) + tgt_min

def build_vector_dataset_dict(
    kind: str,
    train_samples: int,
    valid_samples: int,
    test_samples: int,
    *,
    noise: float = 0.0,
    seed: int = 0,
    separation: float = 0.5,
    factor: float = 0.5,
    cluster_std: float = 1.0,
    center_distance: float = 3.0,
    n_features: int = 2,
    scale_to_2pi: bool = False,
    multiclass_features: int = 8,
    multiclass_classes: int = 4,
    multiclass_labels: int = 2,
    multiclass_length: int = 50,
    multiclass_allow_unlabeled: bool = False,
    extra_feats: bool = False,
):
    kind = kind.lower()
    rng_train = np.random.RandomState(seed)
    if kind == "moons":
        train_feats, train_labs = _generate_two_moons(train_samples, noise=noise, separation=separation, rng=rng_train, extra_feats=extra_feats)
    elif kind == "circles":
        train_feats, train_labs = _generate_circles(train_samples, noise=noise, factor=factor, rng=rng_train)
    elif kind == "blobs":
        train_feats, train_labs = _generate_blobs(
            train_samples,
            cluster_std=cluster_std,
            center_distance=center_distance,
            n_features=n_features,
            rng=rng_train,
        )
    elif kind == "multiclass":
        train_feats, train_labs, class_vectors = _generate_multiclass_from_multilabel(
            train_samples,
            n_features=multiclass_features,
            n_classes=multiclass_classes,
            n_labels=multiclass_labels,
            length=multiclass_length,
            allow_unlabeled=multiclass_allow_unlabeled,
            rng=rng_train,
        )
    else:
        raise ValueError(f"Unknown vector dataset kind '{kind}'")
    if kind != "multiclass":
        class_vectors = None
    feat_min = train_feats.min(axis=0)
    feat_max = train_feats.max(axis=0)
    target_range = (0.0, 2.0 * math.pi) if scale_to_2pi else (-1.0, 1.0)
    scale_cfg = {
        "min": feat_min,
        "max": feat_max,
        "target": target_range,
    }
    datasets = {}
    datasets["train"] = VectorDataset(train_feats, train_labs, scale=scale_cfg, class_vectors=class_vectors)
    for split_name, n_samples, offset in [("valid", valid_samples, 1), ("test", test_samples, 2)]:
        if n_samples <= 0:
            feats = np.empty((0, train_feats.shape[1]), dtype=np.float32)
            labs = np.empty((0,), dtype=np.int64)
        else:
            split_rng = np.random.RandomState(seed + offset)
            if kind == "moons":
                feats, labs = _generate_two_moons(n_samples, noise=noise, separation=separation, rng=split_rng, extra_feats=extra_feats)
            elif kind == "circles":
                feats, labs = _generate_circles(n_samples, noise=noise, factor=factor, rng=split_rng)
            elif kind == "blobs":
                feats, labs = _generate_blobs(
                    n_samples,
                    cluster_std=cluster_std,
                    center_distance=center_distance,
                    n_features=n_features,
                    rng=split_rng,
                )
            else:
                feats, labs, class_vectors = _generate_multiclass_from_multilabel(
                    n_samples,
                    n_features=multiclass_features,
                    n_classes=multiclass_classes,
                    n_labels=multiclass_labels,
                    length=multiclass_length,
                    allow_unlabeled=multiclass_allow_unlabeled,
                    rng=split_rng,
                    base_vectors=class_vectors,
                )
        datasets[split_name] = VectorDataset(feats, labs, scale=scale_cfg, class_vectors=class_vectors)
    return datasets
def infer_num_classes(dataset_dict) -> int:
    """
    Determine the number of distinct class labels present in the training split.
    """
    train_split = dataset_dict["train"]
    if hasattr(train_split, "targets"):
        targets = train_split.targets
        if isinstance(targets, torch.Tensor):
            return int(torch.unique(targets).numel())
        try:
            return len(set(int(v) for v in targets))
        except TypeError:
            pass
    labels = set()
    for sample in train_split:
        digit = sample["digit"]
        if isinstance(digit, torch.Tensor):
            if digit.numel() == 1:
                labels.add(int(digit.item()))
            else:
                labels.update(int(x) for x in digit.view(-1).tolist())
        else:
            labels.add(int(digit))
    return len(labels)

def plot_vector_dataset(dataset_dict, split: str = "train", save_path: Optional[str] = None):
    ds = dataset_dict.get(split)
    if ds is None:
        raise ValueError(f"Unknown split '{split}' for vector dataset.")
    if not hasattr(ds, "coords"):
        raise AttributeError("VectorDataset missing 'coords' attribute required for plotting.")
    # coords = ds.coords.detach().cpu().numpy()
    coords = ds.features.detach().cpu().numpy()
    labels = ds.targets.detach().cpu().numpy()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 4))
    if coords.shape[1] > 2:
        try:
            from sklearn.decomposition import PCA
            proj = PCA(n_components=2, random_state=0).fit_transform(coords)
            xs, ys = proj[:, 0], proj[:, 1]
            plt.title(f"Vector dataset ({split}) - PCA projection")
        except Exception:
            xs = coords[:, 0]
            ys = coords[:, 1] if coords.shape[1] > 1 else np.zeros_like(xs)
            plt.title(f"Vector dataset ({split}) - first two dims")
    else:
        xs = coords[:, 0]
        ys = coords[:, 1] if coords.shape[1] > 1 else np.zeros_like(xs)
        plt.title(f"Vector dataset ({split})")
    scatter = plt.scatter(xs, ys, c=labels, cmap="coolwarm", edgecolor="k", s=20)
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()

def plot_vector_dataset_2d(dataset_dict, split: str = "train", save_path=None):
    ds = dataset_dict[split]
    # coords = ds.coords.detach().cpu().numpy()
    coords = ds.features.detach().cpu().numpy()
    labels = ds.targets.detach().cpu().numpy()

    if coords.shape[1] < 2:
        raise ValueError("Need at least 2D features to make this plot")

    xs, ys = coords[:, 0], coords[:, 1]

    # print(f"few xs: {xs[:5]}")
    # exit()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(xs, ys, c=labels, cmap="coolwarm", edgecolor="k", s=20)
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.title(f"Vector dataset ({split})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()
# -------- Z Feature Map --------
class ZFeatureEncoder(tq.QuantumModule):
    def __init__(self, n_wires: int, reps: int = 1, alpha: float = 1.0):
        super().__init__()
        self.n_wires, self.reps, self.alpha = n_wires, int(reps), float(alpha)
        self.h, self.rz = tq.Hadamard(), tq.RZ()
    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
        x = x[:, :self.n_wires]
        for _ in range(self.reps):
            for w in range(self.n_wires): self.h(qdev, wires=w)
            for w in range(self.n_wires): self.rz(qdev, wires=w, params=2.0*self.alpha*x[:, w])
# -------- ZZ Feature Map --------
class ZZFeatureEncoder(tq.QuantumModule):
    def __init__(self, n_wires: int, reps: int = 1, alpha: float = 1.0,
                 entanglement: str = "ring", phi_pair: str = "prod"):
        super().__init__()
        self.n_wires, self.reps, self.alpha = n_wires, int(reps), float(alpha)
        self.phi_pair = phi_pair
        self.h, self.rz, self.rzz = tq.Hadamard(), tq.RZ(), tq.RZZ()
        ent_tbl = {"linear": _pairs_disjoint, "ring": _pairs_ring, "full": _pairs_full}
        self.pairs = ent_tbl[entanglement](n_wires)
    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
        x = x[:, :self.n_wires]
        for _ in range(self.reps):
            for w in range(self.n_wires): self.h(qdev, wires=w)
            for w in range(self.n_wires): self.rz(qdev, wires=w, params=2.0*self.alpha*x[:, w])
            for a, b in self.pairs:
                phi = (np.pi - x[:, a]) * (np.pi - x[:, b]) if self.phi_pair == "pi_minus" else x[:, a]*x[:, b]
                self.rzz(qdev, wires=[a, b], params=2.0*self.alpha*phi)
# -------- Pauli Feature Map --------
class PauliFeatureEncoder(tq.QuantumModule):
    def __init__(self, terms: Sequence[str], reps: int = 1, alpha: float = 1.0):
        super().__init__()
        self.terms = tuple(terms)
        self.reps, self.alpha = int(reps), float(alpha)
        self.h, self.sdg, self.rz, self.rzz = tq.Hadamard(), tq.SDG(), tq.RZ(), tq.RZZ()
    def _basis_change(self, qdev, axis: str, wire: int, inv: bool = False):
        if axis == "X":
            self.h(qdev, wires=wire)
        elif axis == "Y":
            if not inv: self.sdg(qdev, wires=wire); self.h(qdev, wires=wire)
            else:       self.h(qdev, wires=wire); self.sdg(qdev, wires=wire)
    def _parse(self, term: str):
        axes = ''.join([c for c in term if c in "IXYZ"])
        idxs = list(map(int, re.findall(r"\d+", term)))
        return axes, idxs
    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
        for _ in range(self.reps):
            for term in self.terms:
                axes, idxs = self._parse(term)
                if len(idxs) == 1:
                    a = idxs[0]; ax = axes[-1]
                    if ax in "XY": self._basis_change(qdev, ax, a, inv=False)
                    self.rz(qdev, wires=a, params=2.0*self.alpha*x[:, a])
                    if ax in "XY": self._basis_change(qdev, ax, a, inv=True)
                elif len(idxs) == 2:
                    a, b = idxs; axa, axb = axes[0], axes[1]
                    if axa in "XY": self._basis_change(qdev, axa, a, inv=False)
                    if axb in "XY": self._basis_change(qdev, axb, b, inv=False)
                    self.rzz(qdev, wires=[a, b], params=2.0*self.alpha*(x[:, a]*x[:, b]))
                    if axa in "XY": self._basis_change(qdev, axa, a, inv=True)
                    if axb in "XY": self._basis_change(qdev, axb, b, inv=True)
                else:
                    raise NotImplementedError("Only 1- or 2-body Pauli terms supported.")
# -------- Efficient SU2 (as a feature map) --------
# class EfficientSU2FeatureEncoder(tq.QuantumModule):
#     def __init__(self, n_wires: int, reps: int = 1,
#                  single_ops: Sequence[str] = ("ry","rz"),
#                  alpha: float = np.pi,
#                  ent_kind: str = "linear", twoq_op: str = "cx"):
#         super().__init__()
#         self.n_wires, self.reps = n_wires, int(reps)
#         self.alpha = float(alpha)
#         self.single_ops = tuple(single_ops)
#         self.entangler = make_entangler(ent_kind, n_wires, two_qubit_op=twoq_op,
#                                         trainable=False, wire_reverse=False)
#         self._op_tbl = {"rx": tq.RX(), "ry": tq.RY(), "rz": tq.RZ()}
#     @tq.static_support
#     def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
#         xw = x[:, :self.n_wires]
#         for _ in range(self.reps):
#             for op in self.single_ops:
#                 gate = self._op_tbl[op.lower()]
#                 for w in range(self.n_wires):
#                     gate(qdev, wires=w, params=self.alpha * xw[:, w])
#             self.entangler(qdev)
from torchquantum.functional import func_name_dict
from typing import Iterable, Optional, Sequence
class EfficientSU2FeatureEncoder(tq.QuantumModule):
    """
    Efficient-SU2-like *feature map*:
      - per repetition: apply data-driven single-qubit ops (e.g., ('ry','rz')) to each wire
      - then apply a fixed entangler (e.g., 'linear' with CX)
    Repetitions are computed automatically to consume all features, with zero padding.
    Args:
      n_wires: number of qubits
      single_ops: sequence of 1q ops (each consumes 1 feature); default ('ry','rz')
      ent_kind: entangler topology for the *feature map* (e.g., 'linear','circular','full'); None => no entanglement
      twoq_op: two-qubit gate class name for the entangler (e.g., 'cx','cz','rxx', ...)
      alpha: global scaling for angles (features are multiplied by alpha)
      pad_mode: 'zero' | 'wrap' | 'repeatlast'
    """
    def __init__(
        self,
        n_wires: int,
        single_ops: Sequence[str] = ("ry", "rz"),
        ent_kind: Optional[str] = "linear",
        twoq_op: str = "cx",
        alpha: float = 1.0,
        pad_mode: str = "zero",
    ):
        super().__init__()
        assert n_wires >= 1
        self.n_wires = n_wires
        self.single_ops = tuple(single_ops)
        self.alpha = alpha
        assert pad_mode in ("zero", "wrap", "repeatlast")
        self.pad_mode = pad_mode
        # build entangler for the *feature map* (or disable)
        self.entangler = None
        if ent_kind is not None:
            self.entangler = make_entangler(
                kind=ent_kind,
                n_wires=n_wires,
                two_qubit_op=twoq_op,
                trainable=False,
                wire_reverse=False,
            )
        # validate ops exist in TQ functional table
        for op in self.single_ops:
            if op not in func_name_dict:
                raise ValueError(f"Unknown 1q op '{op}'")
            # we assume 1 param per op; if you later use 'u3', handle 3 params below
    def _next_theta(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """
        Returns a parameter tensor (B,) according to pad_mode if idx>=D.
        """
        B, D = x.shape
        if idx < D:
            return x[:, idx]
        # padding policy
        if self.pad_mode == "zero":
            return x.new_zeros(B)  # angle 0 -> identity
        elif self.pad_mode == "wrap":
            return self.alpha * x[:, idx % max(1, D)]
        else:  # 'repeatlast'
            src = D - 1 if D > 0 else 0
            return self.alpha * x[:, src]
    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
        """
        x: (B, D) arbitrary number of features.
        For ops that consume 1 feature each (ry/rz), reps = ceil(D / (n_wires * len(single_ops))).
        """
        B, D = x.shape
        # how many features we can encode per repetition:
        per_rep = self.n_wires * len(self.single_ops)
        reps = max(1, math.ceil(D / per_rep)) if per_rep > 0 else 1
        # print(f"EffSU2FeatureEncoder: n_wires={self.n_wires}, input feats D={D}, per_rep={per_rep}, reps={reps}")
        # exit()
        # consume features in a fixed order: for each rep, for each op, for each wire
        idx = 0
        for _ in range(reps):
            for op in self.single_ops:
                func = func_name_dict[op]
                print(func)
                
                # assume 1 parameter per op; extend here if you later use ops with >1 params
                for w in range(self.n_wires):
                    theta = self._next_theta(x, idx)  # (B,)
                    idx += 1
                    func(
                        qdev,
                        wires=w,
                        params=theta,
                        static=self.static_mode,
                        parent_graph=self.graph,
                    )
                    print(f"  applied {op} on wire {w} param {theta}")
            exit()
            if self.entangler is not None:
                self.entangler(qdev)
# __all__ = [
#     "ZFeatureEncoder", "ZZFeatureEncoder", "PauliFeatureEncoder", "EfficientSU2FeatureEncoder",
#     "build_feature_map"
# ]
def build_feature_map(cfg: QFCConfig, n_wires: int):
    kind = cfg.fm_kind.lower()
    if kind == "z":
        return ZFeatureEncoder(n_wires, reps=cfg.fm_z_reps, alpha=cfg.fm_z_alpha)
    if kind == "zz":
        return ZZFeatureEncoder(n_wires, reps=cfg.fm_zz_reps, alpha=cfg.fm_zz_alpha,
                                entanglement=cfg.fm_zz_entanglement, phi_pair=cfg.fm_zz_phi)
    if kind == "pauli":
        terms = cfg.fm_pauli_terms or [f"Z{i}" for i in range(n_wires)] + \
                                   [f"ZZ{i}{(i+1)%n_wires}" for i in range(n_wires)]
        return PauliFeatureEncoder(terms, reps=cfg.fm_pauli_reps, alpha=cfg.fm_pauli_alpha)
    if kind == "eff_su2":
        # n_wires: int,
        # single_ops: Sequence[str] = ("ry", "rz"),
        # ent_kind: Optional[str] = "linear",
        # twoq_op: str = "cx",
        # alpha: float = 1.0,
        # pad_mode: str = "zero"
        return EfficientSU2FeatureEncoder(n_wires,
                                          single_ops=cfg.fm_eff_single_ops,
                                          alpha=cfg.fm_eff_alpha,
                                          ent_kind=cfg.fm_eff_ent_kind,
                                          twoq_op=cfg.fm_eff_twoq_op
                                          )
    raise ValueError(f"Unknown fm_kind '{cfg.fm_kind}'")
def _default_measure_pairs(n_wires: int) -> List[Tuple[int,int]]:
    # even wires: (0,1),(2,3),...
    pairs = [(i, i+1) for i in range(0, n_wires - 1, 2)]
    # odd wires: also include the last wrapping to 0
    if n_wires % 2 == 1:
        pairs.append((n_wires - 1, 0))
    return pairs
def _measure_pairs(qdev, ops: Sequence[str], pairs: Sequence[Tuple[int,int]]) -> torch.Tensor:
    feats = []
    for (a, b) in pairs:
        for op in ops:
            # op like "XX","YY","ZZ","XY"
            feats.append(expval_joint_analytical(qdev, op, wires=[a, b]))  # <-- pass wires
    return torch.stack(feats, dim=1)  # (B, len(pairs)*len(ops))
def auto_pair_ops(k: int) -> tuple[str, ...]:
    """
    Return k 2-qubit Pauli ops over {X,Y,Z}.
    Order: axes first, then cross-terms.
    """
    canon = ["XX", "YY", "ZZ", "XY", "XZ", "YZ", "YX", "ZX", "ZY"]
    if k <= len(canon):
        return tuple(canon[:k])
    out = []
    while len(out) < k:
        out.extend(canon)
    return tuple(out[:k])
def _default_disjoint_pairs(n_wires: int) -> List[Tuple[int,int]]:
    # (0,1), (2,3), ...
    pairs = [(i, i+1) for i in range(0, n_wires - 1, 2)]
    if n_wires % 2 == 1:
        pairs.append((n_wires - 1, 0))
    return pairs
def _ring_pairs(n_wires: int) -> List[Tuple[int,int]]:
    # (0,1), (1,2), (2,3), ..., (N-1,0)
    return [(i, (i+1) % n_wires) for i in range(n_wires)]
def _embed_on_pair(op2: str, a: int, b: int, n_wires: int) -> str:
    # op2 like "XX","XY"
    s = ["I"] * n_wires
    s[a], s[b] = op2[0], op2[1]
    return "".join(s)

# @torch.no_grad()
def _measure_pooled_pairs(qdev, ops: Sequence[str], pairs: Sequence[Tuple[int,int]], n_wires: int) -> torch.Tensor:
    # print(f"in measured pooled pairs")
    cols = []
    for op in ops:
        vals = []
        for a, b in pairs:
            pauli = _embed_on_pair(op, a, b, n_wires)
            # print(f"Measuring pauli {pauli} on wires ({a},{b})")
            vals.append(expval_joint_analytical(qdev, pauli))   # (B,)
        cols.append(torch.stack(vals, 1).mean(1))               # mean over pairs
        # cols.append(torch.stack(vals, 1).max(1)[0])
    # exit()
    return torch.stack(cols, 1)  # (B, len(ops))


# @torch.no_grad()
def _measure_unpooled_pairs(qdev, ops: Sequence[str], pairs: Sequence[Tuple[int,int]], n_wires: int) -> torch.Tensor:
    # print(f"in measured un_pooled pairs")
    
    feats = []
    for a, b in pairs:
        for op in ops:
            pauli = _embed_on_pair(op, a, b, n_wires)
            # print(f"Measuring pauli {pauli} on wires ({a},{b})")
            feats.append(expval_joint_analytical(qdev, pauli))  # (B,)
        # exit()
    # exit()
    return torch.stack(feats, 1)  # (B, len(pairs)*len(ops))

def _op_from_name(name: str):
    name = name.lower()
    table = {
        "cx": tq.CNOT, "cnot": tq.CNOT, "cz": tq.CZ, "swap": tq.SWAP,
        "crx": tq.CRX, "cry": tq.CRY, "crz": tq.CRZ,
        "rxx": tq.RXX, "ryy": tq.RYY, "rzz": tq.RZZ,
    }
    if name not in table:
        raise ValueError(f"Unknown two-qubit op '{name}'")
    return table[name]
def make_entangler(kind: str, n_wires: int, two_qubit_op: str = "cx",
                   trainable: bool = False, wire_reverse: bool = False, *,
                   jump: int = 1, circular: bool = True):
    """
    kind: 'full' | 'circular' | 'butterfly'
    """
    op_cls = _op_from_name(two_qubit_op)
    # op_has_params = getattr(op_cls, "num_params", 0) > 0
    op_has_params = True
    if kind == "full":
        # all-to-all (a.k.a. dense / EntangleFull)
        return EntangleFull(op=op_cls, n_wires=n_wires,
                              has_params=op_has_params, trainable=trainable,
                              wire_reverse=wire_reverse)
    if kind == "linear":
        # nearest-neighbor chain
        return EntangleLinear(op=op_cls, n_wires=n_wires,
                              has_params=op_has_params, trainable=trainable,
                              wire_reverse=wire_reverse)
    if kind == "pairwise":
        # pairwise entanglement
        return EntanglePairwise(op=op_cls, n_wires=n_wires,
                                has_params=op_has_params, trainable=trainable,
                                wire_reverse=wire_reverse)
    
    if kind == "ring":
        # nearest-neighbor ring; tune jump/circular as needed
        return EntangleCircular(op=op_cls, n_wires=n_wires,
                            has_params=op_has_params, trainable=trainable,
                            wire_reverse=wire_reverse)
    if kind == "butterfly":
        return Op2QButterflyLayer(op=op_cls, n_wires=n_wires,
                                  has_params=op_has_params, trainable=trainable,
                                  wire_reverse=wire_reverse)
    raise ValueError(f"Unknown entanglement kind '{kind}'")
try:
    from qiskit.qasm2 import loads as qasm_loads
except Exception:
    qasm_loads = None
from qiskit import QuantumCircuit
from typing import Optional, Tuple, Any
# --- Minimal OpenQASM 2.0 gate defs to make Qiskit happy ---
_CRX_DEF = r"""
gate crx(theta) a,b {
    u1(pi/2) b;
    cx a,b;
    u3(theta, -pi/2, pi/2) b;
    cx a,b;
    u1(-pi/2) b;
}
"""
_RZZ_DEF = r"""
gate rzz(theta) a,b {
    cx a,b;
    rz(theta) b;
    cx a,b;
}
"""
_P_DEF = r"""
gate p(phi) q { u1(phi) q; }
"""
def make_qasm_qiskit_friendly(qasm: str, use_p: bool = False) -> str:
    # Always ensure header and include
    if "OPENQASM 2.0;" not in qasm:
        qasm = "OPENQASM 2.0;\n" + qasm
    if 'include "qelib1.inc";' not in qasm:
        qasm = qasm.replace("OPENQASM 2.0;",
                            'OPENQASM 2.0;\ninclude "qelib1.inc";', 1)
    # Strip existing header/include for clean reinsertion
    import re
    body = re.sub(r'^\s*OPENQASM 2\.0;\s*', '', qasm, count=1, flags=re.MULTILINE)
    body = re.sub(r'^\s*include\s+"qelib1\.inc";\s*', '', body, count=1, flags=re.MULTILINE)
    # Optional u1→p replacement (only in body, not definitions)
    if use_p:
        body = re.sub(r'\bu1\(', 'p(', body)
    # Build injection block. Only add defs that aren't already present.
    inject_parts = []
    if use_p and "gate p(" not in body:
        inject_parts.append(_P_DEF.strip())
    if "gate crx(" not in body:
        inject_parts.append(_CRX_DEF.strip())
    if "gate rzz(" not in body:
        inject_parts.append(_RZZ_DEF.strip())
    injection = ("\n\n".join(inject_parts) + "\n\n") if inject_parts else ""
    return 'OPENQASM 2.0;\ninclude "qelib1.inc";\n\n' + injection + body.lstrip()
# def draw_circuit_from_qasm(
#     qasm: str,
#     backend: str = "mpl",
#     save_path: Optional[str] = None
# ) -> Tuple[Any, QuantumCircuit]:
#     qasm = make_qasm_qiskit_friendly(qasm)
#     qc = (qasm_loads(qasm) if qasm_loads is not None
#           else QuantumCircuit.from_qasm_str(qasm))
#     out = qc.draw(output=backend)  # "mpl" for figure, "text" for ASCII
#     # print(out)
#     if save_path:
#         if backend == "mpl":
#             out.savefig(save_path, bbox_inches="tight", dpi=200)
#         else:
#             with open(save_path, "w") as f:
#                 f.write(str(out))
#     return out, qc
def draw_circuit_from_qasm(
    qasm: str,
    backend: str = "mpl",
    save_path: Optional[str] = None,
    prefer_p: bool = True,  # set True if you want P(...) instead of U1(...)
) -> Tuple[Any, QuantumCircuit]:
    """
    Render a QASM string with Qiskit. We run it through make_qasm_qiskit_friendly
    to inject missing gate defs (crx, rzz, optionally p) and the qelib1 include.
    prefer_p=True will rewrite u1(...) -> p(...) and inject a 'gate p(phi) q { u1(phi) q; }'
    definition. If the parser still errors on 'p', we automatically retry with prefer_p=False.
    """
    # 1) Fix and (optionally) convert to P gate form
    fixed = make_qasm_qiskit_friendly(qasm, use_p=prefer_p)
    # 2) Try to parse; if it fails specifically due to 'p' not defined, retry w/o P
    try:
        qc = (qasm_loads(fixed) if qasm_loads is not None
              else QuantumCircuit.from_qasm_str(fixed))
    except Exception as e:
        msg = str(e).lower()
        if prefer_p and ("'p' is not defined" in msg or " p " in msg and "not defined" in msg):
            # retry with u1 (no P conversion)
            fixed = make_qasm_qiskit_friendly(qasm, use_p=False)
            qc = (qasm_loads(fixed) if qasm_loads is not None
                  else QuantumCircuit.from_qasm_str(fixed))
        else:
            raise
    # 3) Draw + save
    out = qc.draw(output=backend)  # "mpl" | "text" | "latex" | ...
    if save_path:
        if backend == "mpl":
            out.savefig(save_path, bbox_inches="tight", dpi=200)
        else:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(str(out))
    return out, qc
# ----------------------------
# Quantum Layers (PyTorch-style)
# ----------------------------
class VariationalBlock(tq.QuantumModule):
    """
    Per-wire trainable RX/RY/RZ, then a topology-configurable entangler
    that works for any n_wires >= 2.
    """
    def __init__(self, n_wires: int, *, ent_kind: str, twoq_op: str,
                 ent_trainable: bool, ent_wire_reverse: bool):
        super().__init__()
        assert n_wires >= 2, "VariationalBlock requires at least 2 wires."
        self.n_wires = n_wires
        # Per-wire trainable 1q rotations
        self.rx = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.ry = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
        self.rz = nn.ModuleList([tq.RZ(has_params=True, trainable=True) for _ in range(n_wires)])
        # Entangler that scales with n_wires
        self.entangler = make_entangler(
            kind=ent_kind,
            n_wires=n_wires,
            two_qubit_op=twoq_op,
            trainable=ent_trainable,
            wire_reverse=ent_wire_reverse,
        )
    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        for w in range(self.n_wires):
            self.rx[w](qdev, wires=w)
            self.ry[w](qdev, wires=w)
            self.rz[w](qdev, wires=w)
        self.entangler(qdev)


class QuantumCircuit(tq.QuantumModule):
    def __init__(self, n_wires: int, depth: int, n_random_ops: int,
                 ent_kind: str, twoq_op: str, ent_trainable: bool, ent_wire_reverse: bool):
        super().__init__()
        self.n_wires = n_wires
        self.random = tq.RandomLayer(
            n_ops=n_random_ops, wires=list(range(n_wires))
        ) if n_random_ops > 0 else None
        self.blocks = nn.ModuleList([
            VariationalBlock(
                n_wires=n_wires,
                ent_kind=ent_kind,
                twoq_op=twoq_op,
                ent_trainable=ent_trainable,
                ent_wire_reverse=ent_wire_reverse,
            )
            for _ in range(depth)
        ])
    def forward(self, qdev: tq.QuantumDevice):
        if self.random is not None:
            self.random(qdev)
        for blk in self.blocks:
            blk(qdev)
# class QFCHead(nn.Module):
#     """
#     Simple head that maps measured observables -> logits.
#     If you keep len(measure_ops) == num_classes, the Linear is identity-like.
#     Otherwise, it lets you decouple #measurements from #classes.
#     """
#     def __init__(self, measure_ops: Sequence[str], num_classes: int):
#         super().__init__()
#         self.measure_ops = tuple(measure_ops)
#         self.linear = nn.Linear(len(self.measure_ops), num_classes, bias=True)
#     def forward(self, measured: torch.Tensor) -> torch.Tensor:
#         # measured: (B, M) where M == len(measure_ops)
#         return self.linear(measured)
class QFCHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes, bias=True)
    def forward(self, measured: torch.Tensor) -> torch.Tensor:
        return self.linear(measured)
    
# ----------------------------
# Full Model
# ----------------------------
class QFCModel(tq.QuantumModule):
    def __init__(self, cfg: QFCConfig):
        super().__init__()
        self.cfg = cfg
        self.D = cfg.feature_dim if cfg.feature_dim is not None else cfg.pool_hw * cfg.pool_hw
        
        if self.D <= 0:
            raise ValueError("Feature dimension must be positive.")
        self.func_list = List[Dict[str, Any]]
     
        ops = ("ry","rz")
      
        if cfg.fm_kind.lower() == "z": 
            # Build and save a ZZ op-list (example)
            # z_name, z_ops = make_z_oplist(n_wires=cfg.n_wires, D=self.D, alpha=cfg.fm_z_alpha, pad_mode=cfg.fm_z_pad_mode)
            # write_oplist_py("z_4x_auto.py", z_name, z_ops)
            # self.encoder = GeneralEncoderPlus(z_ops)
            # self.func_list = z_ops
            # thi has issues for now
            # z_name, z_ops =  build_tiled_pauli_oplist(
            #     n_wires=cfg.n_wires, D=self.D, paulis=["Z"], entanglement=None, pad_mode="wrap")
            
            # z_name, z_ops = build_tiled_pauli_oplist(
            #     n_wires=cfg.n_wires, D=self.D, entanglement="linear", pad_mode="wrap",
            #     pair_phi="prod"   # ignored here (no pair terms)
            # )
            
            # z_name,  z_ops  = build_z_oplist(n_wires=cfg.n_wires, D=self.D, pad_mode="wrap")
            # write_oplist_py("z_with_pauli.py", z_name, z_ops)
            # exit()
            # build Z only (no entanglement), 5 wires, D=16, wrap padding; keep H as H
            z_name, z_ops = build_tiled_pauli_oplist(
                n_wires=cfg.n_wires, D=self.D, paulis=["Z"], pad_mode=cfg.fm_z_pad_mode, repeats=cfg.fm_z_reps,
                expand_h_to_u3=True,
            )
            # self.encoder =  make_encoder_from_oplist(oplist, alpha=1.0, multi_index_rule="prod")
            write_oplist_py("z_with_pauli.py", z_name, z_ops)
            self.encoder =  GeneralEncoderPlus_new(z_ops, alpha=cfg.fm_z_alpha, multi_index_rule="sum")
            # self.encoder = GeneralEncoderPlus_new(z_ops, pad_mode="wrap", alpha=1.0)
            
            
            self.func_list = z_ops
            # exit()
        # .........
        if cfg.fm_kind.lower() == "zz":
            
            # print("in ZZ list")
            # # exit()
            # zz_name, zz_ops =  build_tiled_pauli_oplist(
            #     n_wires=cfg.n_wires, D=self.D, paulis=["Z", "ZZ"], entanglement=cfg.fm_zz_entanglement, pad_mode="wrap")
            
            # write_oplist_py("zz_with_pauli.py", zz_name, zz_ops)
            # # exit()
            # self.encoder = GeneralEncoderPlus_new(zz_ops, pad_mode="wrap", alpha=1.0) #oplist, pad_mode="wrap", alpha=1.0
            # self.func_list = zz_ops
           
            zz_name, zz_ops = build_tiled_pauli_oplist(
                n_wires=cfg.n_wires, D=self.D, 
                paulis=["Z", "ZZ"], 
                pad_mode=cfg.fm_zz_pad_mode, 
                entanglement=cfg.fm_zz_entanglement, 
                repeats=cfg.fm_zz_reps,
                expand_h_to_u3=True,
            )
            # self.encoder =  make_encoder_from_oplist(oplist, alpha=1.0, multi_index_rule="prod")
            write_oplist_py("zasdasd_with_pauli.py", zz_name, zz_ops)
            self.encoder =  GeneralEncoderPlus_new(zz_ops, alpha=cfg.fm_zz_alpha, multi_index_rule="sum")
            # self.encoder = GeneralEncoderPlus_new(z_ops, pad_mode="wrap", alpha=1.0)
            
            
            self.func_list = zz_ops
            
            # exit()
        # yyyyyyyy
        if cfg.fm_kind.lower() == "pauli":
            print("in pauli maping op list generator")
           
            # pauli_name, pauli_ops = build_pauli_map_qiskit_ops(
            # n_wires=cfg.n_wires, D=self.D,
            # paulis=cfg.fm_pauli_terms, #("Z","ZZ"),
            # entanglement="liear",
            # pad_mode="wrap"
            # )
            # # write_oplist_py("pauli_map_ops.py", pauli_name, pauli_ops)
            # save_encoder_oplist_py("encoder_ops_pauli.py", pauli_name, pauli_ops)
            # # exit()
            # self.encoder = GeneralEncoderPlus(pauli_ops)
            # self.func_list = pauli_ops
            # name, d = build_zz_oplist(n_wires=2, D=self.D, entanglement="linear", pad_mode="wrap")
            # save_encoder_oplist_py("encoder_ops_pauli.py", name, d)
            # updaing this area for pauli to pass here
            # pauli_name, pauli_ops =  build_tiled_pauli_oplist(
            #     n_wires=cfg.n_wires, D=self.D, paulis=cfg.fm_pauli_terms, entanglement="linear", pad_mode="wrap")
            # # exit()
            # write_oplist_py("with_pauli.py", pauli_name, pauli_ops)
            # self.encoder = GeneralEncoderPlus_new(pauli_ops, pad_mode="wrap", alpha=2.0)
            # self.func_list = pauli_ops
            pauli_name, pauli_ops = build_tiled_pauli_oplist(
                n_wires=cfg.n_wires, D=self.D, paulis=cfg.fm_pauli_terms, pad_mode=cfg.fm_pali_pad, entanglement=cfg.fm_pauli_entanglement, repeats=cfg.fm_pauli_reps,
                expand_h_to_u3=True,
            )
            # self.encoder =  make_encoder_from_oplist(oplist, alpha=1.0, multi_index_rule="prod")
            write_oplist_py("pauli_with_pauli.py", pauli_name, pauli_ops)
            self.encoder =  GeneralEncoderPlus_new(pauli_ops, alpha=cfg.fm_pauli_alpha, multi_index_rule="prod")
            # self.encoder = GeneralEncoderPlus_new(z_ops, pad_mode="wrap", alpha=1.0)
            
            
            self.func_list = pauli_ops
      
        if cfg.fm_kind.lower() == "eff_su2":
    
            print("In SU2 encoder part")
            
            su2_name, su2_op = build_efficient_su2_oplist_qisk_new(
                D=self.D, n_wires=cfg.n_wires,
                single_ops=("ry","rz"),
                entanglement=cfg.fm_eff_ent_kind, twoq=cfg.fm_eff_twoq_op,
                pad_mode=cfg.fm_eff_pad_mod, alpha=cfg.fm_eff_alpha
            )
            
            save_oplist_py("efficient_su2_3w.py", su2_name, su2_op)
            # exit()
            # Note: out of the Gen ecnoder for SU2, run for longer epochs to see which one works best
            # self.encoder = self.encoder = tq.GeneralEncoder(su2_op) # older one working as well
            self.encoder = GeneralEncoderPlus_new(su2_op, alpha=1.0, multi_index_rule="prod")
            self.func_list = su2_op
            # exit()
     
        # this part is to generate variation quantum circuit
        # Q-layer / VQC
        # 99999999999999999999999999

        
        self.vqc_circuit = QuantumCircuit(
            n_wires=cfg.n_wires,
            depth=cfg.depth,
            n_random_ops=cfg.n_random_ops,
            ent_kind=cfg.qlayer_ent_kind,
            twoq_op=cfg.qlayer_twoq_op,
            ent_trainable=cfg.qlayer_ent_trainable,
            ent_wire_reverse=cfg.qlayer_ent_wire_reverse,
        )
        # pick ops automatically if not provided (use #classes as K by default)

        self.measure = tq.MeasureAll(tq.PauliZ)

        if not self.cfg.measure_ops:
            self.cfg.measure_ops = auto_pair_ops(self.cfg.num_classes) # NOte: we may want to debug this
            # print(f"Measuring ops: {self.cfg.measure_ops}")
            # exit()
        
        # pick pairs/topology automatically if not provided
        if not self.cfg.measure_pairs:
            if self.cfg.pair_topology == "ring":
                self.pairs = _ring_pairs(self.cfg.n_wires)
            else:
                self.pairs = _default_disjoint_pairs(self.cfg.n_wires)
        else:
            self.pairs = self.cfg.measure_pairs
        
        # decide head input size based on pooling choice
        if self.cfg.pool_pairs:
            in_feats = len(self.cfg.measure_ops)              # (B, K)
        else:
            in_feats = len(self.cfg.measure_ops) * len(self.pairs)  # (B, P*K)

        # self.head = QFCHead(in_features=in_feats, num_classes=self.cfg.num_classes)
        # self.head = QFCHead(in_features=self.cfg.num_classes, num_classes=self.cfg.num_classes)
        self.linear = nn.Linear(cfg.n_wires, self.cfg.num_classes, bias=True) # for built-in measure all
        # if using measure all from tq
        # self.head = QFCHead(in_features=self.cfg.num_classes, num_classes=self.cfg.num_classes)

            
        # if not getattr(self.cfg, "measure_ops", None):
        #     self.cfg.measure_ops = auto_pair_ops(self.cfg.num_classes)
        # self.head = QFCHead(self.cfg.measure_ops, self.cfg.num_classes)
        print(f"M Ops: {self.cfg.measure_ops}, Pairs: {self.pairs}, Head in_features: {in_feats}, classes: {self.cfg.num_classes}")
    
    def qiskit_phi(self, vals: List[torch.Tensor]) -> torch.Tensor:
        # vals: [xi] or [xi, xj]; return (B,)
        pi = math.pi
        if len(vals) == 1:
            return 2.0 * vals[0]                     # Z term: P(2*x[i])
        elif len(vals) == 2:
            xi, xj = vals
            return 2.0 * (pi - xi) * (pi - xj)       # ZZ term: P(2*(π-xi)*(π-xj))
        else:
            raise ValueError("phi expects 1 or 2 inputs.")
        
    @staticmethod
    def _prep_features(x: torch.Tensor, pool_hw: int) -> torch.Tensor:

        # print(f"x.dim(): {x.dim()}")
        # exit()
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() == 4:
            bsz = x.shape[0]
            x = F.adaptive_avg_pool2d(x, output_size=(pool_hw, pool_hw)).view(bsz, -1)
            return x
        raise ValueError(f"Unsupported feature tensor shape {tuple(x.shape)}")
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        # 1) Build quantum device
        qdev = tq.QuantumDevice(
            n_wires=self.cfg.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=False
        )

        # 2) Encode classical features
        feats = self._prep_features(x, self.cfg.pool_hw)  # (B, 16)
        self.encoder(qdev, feats) # Quantum Feature mapping

        # 3) Variational circuit
        self.vqc_circuit(qdev)
       
        # 4) Measurements (joint Pauli expectations)
        # sssssssssss
        # if self.cfg.pool_pairs:

        #     measured = _measure_pooled_pairs(qdev, self.cfg.measure_ops, self.pairs, self.cfg.n_wires)  # (B, K)
        # else:
        #     measured = _measure_unpooled_pairs(qdev, self.cfg.measure_ops, self.pairs, self.cfg.n_wires)  # (B, P*K)
        
        measured = self.measure(qdev)  # use the built-in measure all
        # measured = self.measure(qdev) # use the built-in measure all
        # print(f"Measured shape: {measured.shape}")
        # print few of the measured values
        # print(f"Measured values (first 5 samples): {measured[:5]}")
        # exit()
        logits = self.linear(measured)
        # print(f"Logits shape: {logits.shape}")
        # exit()
        # logits = self.head(measured)
        
        return F.log_softmax(logits, dim=1)

def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _mlp_param_count(input_dim: int, hidden: int, num_classes: int) -> int:
    # Linear1 weights/bias + LayerNorm gamma/beta + Linear2 weights/bias
    return hidden * input_dim + hidden + 2 * hidden + hidden * num_classes + num_classes


def _hidden_for_budget(input_dim: int, num_classes: int, target_params: int, *, allow_overshoot: bool = False) -> int:
    if target_params <= num_classes:
        return 1
    best = 1
    for hidden in range(1, 10_000):
        total = _mlp_param_count(input_dim, hidden, num_classes)
        if total == target_params:
            return hidden
        if total < target_params:
            best = hidden
            continue
        # total > target_params
        if allow_overshoot:
            return hidden
        break
    return best

class ClassicalBenchmarkMLP(nn.Module):
    def __init__(self, cfg: QFCConfig, target_params: int, *, allow_overshoot: bool = False):
        super().__init__()
        self.cfg = cfg
        self.feature_dim = cfg.feature_dim or cfg.pool_hw * cfg.pool_hw
        hidden = _hidden_for_budget(
            self.feature_dim,
            cfg.num_classes,
            target_params,
            allow_overshoot=True,
        )
        self.backbone = nn.Sequential(
            nn.Linear(self.feature_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, cfg.num_classes),
        )

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() == 4:
            return F.adaptive_avg_pool2d(x, (self.cfg.pool_hw, self.cfg.pool_hw)).view(x.size(0), -1)
        raise ValueError(f"Unsupported input shape {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._prep(x)
        logits = self.backbone(feats)
        return F.log_softmax(logits, dim=1)


def build_classical_baseline(cfg: QFCConfig, reference: nn.Module) -> ClassicalBenchmarkMLP:
    budget = count_trainable_params(reference)
    # print(f"Building classical baseline MLP with target param budget: {budget}")
    # exit()
    return ClassicalBenchmarkMLP(cfg, budget)


# # 8888888888888888888888888888888888888888888888888888888888888888
# class ClassicalAngleEncoder(nn.Module):
#     def __init__(self, feature_dim: int, n_wires: int):
#         super().__init__()
#         self.feature_dim = feature_dim
#         self.n_wires = n_wires

#     def forward(self, feats: torch.Tensor) -> torch.Tensor:
#         pooled = F.adaptive_avg_pool1d(feats.unsqueeze(1), self.n_wires)
#         return torch.tanh(pooled.squeeze(1))


# class ClassicalVariationalBlock1D(nn.Module):
#     def __init__(self, n_wires: int, param_budget: int, ent_trainable: bool):
#         super().__init__()
#         self.n_wires = n_wires
#         self.ent_trainable = ent_trainable
#         budget = max(0, int(param_budget))
#         self.theta = nn.Parameter(torch.zeros(budget, dtype=torch.float32))

#     def _tile(self, vec: torch.Tensor, length: int) -> torch.Tensor:
#         if length <= 0:
#             return vec.new_zeros(0)
#         if vec.numel() == 0:
#             return vec.new_zeros(length)
#         reps = math.ceil(length / vec.numel())
#         return vec.repeat(reps)[:length]

#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         if self.theta.numel() == 0:
#             return z
#         needed_rot = 3 * self.n_wires
#         tiled = self._tile(self.theta.view(-1), needed_rot)
#         rx = tiled[:self.n_wires]
#         ry = tiled[self.n_wires:2 * self.n_wires]
#         rz = tiled[2 * self.n_wires:3 * self.n_wires]
#         z = z + torch.sin(rx)[None, :] + torch.cos(ry)[None, :] + torch.tanh(rz)[None, :]
#         shift = torch.roll(z, shifts=1, dims=1)
#         if self.ent_trainable:
#             ent_vals = self._tile(self.theta.view(-1)[needed_rot:], self.n_wires)
#             alpha = torch.sigmoid(ent_vals)[None, :]
#         else:
#             alpha = z.new_zeros(1, self.n_wires)
#         return (1.0 - alpha) * z + alpha * shift

# class ClassicalMeasurement(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, latent: torch.Tensor) -> torch.Tensor:
#         return torch.tanh(latent)

# class ClassicalQFCReplica(nn.Module):
#     def __init__(self, cfg: QFCConfig, block_param_counts: Sequence[int]):
#         super().__init__()
#         self.cfg = cfg
#         feat_dim = cfg.feature_dim or cfg.pool_hw * cfg.pool_hw
#         self.pool_hw = cfg.pool_hw
#         self.feature_dim = feat_dim
#         self.pool = nn.AdaptiveAvgPool2d((cfg.pool_hw, cfg.pool_hw))
#         self.encoder = ClassicalAngleEncoder(feat_dim, cfg.n_wires)
#         self.blocks = nn.ModuleList([
#             ClassicalVariationalBlock1D(cfg.n_wires, count, cfg.ent_trainable)
#             for count in block_param_counts
#         ])
#         self.measure = ClassicalMeasurement()
#         self.linear = nn.Linear(cfg.n_wires, cfg.num_classes, bias=True)
#         self.dummy_head = nn.Linear(cfg.num_classes, cfg.num_classes, bias=True)

#     def _prep(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() == 2:
#             return x
#         if x.dim() == 3:
#             x = x.unsqueeze(1)
#         if x.dim() == 4:
#             bsz = x.size(0)
#             return self.pool(x).view(bsz, -1)
#         raise ValueError(f"Unsupported input shape {tuple(x.shape)}")

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         feats = self._prep(x)
#         latent = self.encoder(feats)
#         for blk in self.blocks:
#             latent = blk(latent)
#         measured = self.measure(latent)
#         logits = self.linear(measured)
#         return F.log_softmax(logits, dim=1)


# def build_classical_replica(cfg: QFCConfig, reference: QFCModel) -> ClassicalQFCReplica:
#     block_counts = [
#         count_trainable_params(block)
#         for block in reference.vqc_circuit.blocks
#     ]
#     replica = ClassicalQFCReplica(cfg, block_counts)
#     ref_params = count_trainable_params(reference)
#     cls_params = count_trainable_params(replica)
#     if ref_params != cls_params:
#         raise ValueError(f"Param mismatch (quantum={ref_params}, classical={cls_params}).")
#     return replica
# 8888888888888888888888888888888888888888888888888888888888888888
class VanillaCNN(nn.Module):
    """
    Vanilla CNN (Conv->ReLU->MaxPool with 3x3 kernels).
    Produces a single-channel map that is adaptively pooled to (pool_hw, pool_hw),
    then flattened to (B, pool_hw*pool_hw) for the quantum encoder.
    """
    def __init__(self, in_ch: int = 1, pool_hw: int = 4, dropout: float = 0.5):
        super().__init__()
        self.pool_hw = int(pool_hw)
        self.dropout = float(dropout)
        # 3x3 kernels with padding=1 to keep size before pooling
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # compress to 1 channel before adaptive pooling
        self.to1 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.to1(x))
        # x = F.relu(x)


        x = F.adaptive_avg_pool2d(x, (self.pool_hw, self.pool_hw))
        return x.view(x.size(0), -1)  # (B, pool_hw*pool_hw)

    @staticmethod
    def create(in_ch=1, pool_hw=4, dropout=0.5):
        return VanillaCNN(in_ch=in_ch, pool_hw=pool_hw, dropout=dropout)  


# class CNNPreprocessor(nn.Module):
#     def __init__(self, out_dim):  # out_dim = number of quantum angles you need
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)       # 3x3
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)      # 3x3
#         self.pool  = nn.MaxPool2d(2)
#         self.gap   = nn.AdaptiveAvgPool2d(1)              # -> B x C x 1 x 1
#         self.head  = nn.Linear(64, out_dim)               # tiny head

#     def forward(self, x):
#         x = F.relu(self.conv1(x))          # B x 32 x H x W
#         x = self.pool(x)                   # /2
#         x = F.relu(self.conv2(x))          # B x 64 x H/2 x W/2
#         x = self.pool(x)                   # /4
#         x = self.gap(x).squeeze(-1).squeeze(-1)   # B x 64
#         angles = self.head(x)              # B x out_dim
#         # angles = torch.tanh(angles) * torch.pi    # scale to [-pi, pi]
#         return angles



class CNNPreprocessor(nn.Module):
    def __init__(self, out_dim=20):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2)              # 28->14->7
        self.gap   = nn.AdaptiveAvgPool2d(1)      # Bx64x1x1
        self.head  = nn.Linear(64, out_dim)       # 64 -> 20 angles

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))      # Bx32x14x14
        x = self.pool(F.relu(self.conv2(x)))      # Bx64x7x7
        x = self.gap(x).squeeze(-1).squeeze(-1)   # Bx64
        angles = self.head(x)                     # Bx20
        # angles = torch.tanh(angles) * torch.pi    # scale to [-π, π]
        return angles

# Hybrid Quantum Neural Network target model
class HybridQNN(tq.QuantumModule):
    def __init__(self, cfg: QFCConfig):
        super().__init__()
        self.cfg = cfg
        self.D = cfg.feature_dim if cfg.feature_dim is not None else cfg.pool_hw * cfg.pool_hw
        if self.D <= 0:
            raise ValueError("Feature dimension must be positive.")
        self.func_list = List[Dict[str, Any]]

        
        self.fe = CNNPreprocessor(out_dim=16)
        # self.fe = CNNPreprocessor(out_dim=cfg.n_wires * 2) #10*2
        # print the cnn model and exit
        # print(self.cnn)   
        # exit()
        ops = ("ry","rz")
      
        if cfg.fm_kind.lower() == "z": 
           
            z_name, z_ops = build_tiled_pauli_oplist(
                n_wires=cfg.n_wires, D=self.D, paulis=["Z"], pad_mode=cfg.fm_z_pad_mode, repeats=cfg.fm_z_reps,
                expand_h_to_u3=True,
            )
            # self.encoder =  make_encoder_from_oplist(oplist, alpha=1.0, multi_index_rule="prod")
            write_oplist_py("z_with_pauli.py", z_name, z_ops)
            self.encoder =  GeneralEncoderPlus_new(z_ops, alpha=cfg.fm_z_alpha, multi_index_rule="sum")
            # self.encoder = GeneralEncoderPlus_new(z_ops, pad_mode="wrap", alpha=1.0)
            
            
            self.func_list = z_ops
            # exit()
        
        if cfg.fm_kind.lower() == "zz":
            
           
            zz_name, zz_ops = build_tiled_pauli_oplist(
                n_wires=cfg.n_wires, D=self.D, paulis=["Z", "ZZ"], pad_mode=cfg.fm_zz_pad_mode, entanglement=cfg.fm_zz_entanglement, repeats=cfg.fm_zz_reps,
                expand_h_to_u3=True,)
            # self.encoder =  make_encoder_from_oplist(oplist, alpha=1.0, multi_index_rule="prod")
            write_oplist_py("zasdasd_with_pauli.py", zz_name, zz_ops)
            self.encoder =  GeneralEncoderPlus_new(zz_ops, alpha=cfg.fm_zz_alpha, multi_index_rule="sum")
            # self.encoder = GeneralEncoderPlus_new(z_ops, pad_mode="wrap", alpha=1.0)
            
            
            self.func_list = zz_ops
            
            # exit()
        # yyyyyyyy
        if cfg.fm_kind.lower() == "pauli":
            print("in pauli maping op list generator")
            pauli_name, pauli_ops = build_tiled_pauli_oplist(
                n_wires=cfg.n_wires, D=self.D, paulis=cfg.fm_pauli_terms, pad_mode=cfg.fm_pali_pad, entanglement=cfg.fm_pauli_entanglement, repeats=cfg.fm_pauli_reps,
                expand_h_to_u3=True,
            )
            # self.encoder =  make_encoder_from_oplist(oplist, alpha=1.0, multi_index_rule="prod")
            write_oplist_py("pauli_with_pauli.py", pauli_name, pauli_ops)
            self.encoder =  GeneralEncoderPlus_new(pauli_ops, alpha=cfg.fm_pauli_alpha, multi_index_rule="prod")
            # self.encoder = GeneralEncoderPlus_new(z_ops, pad_mode="wrap", alpha=1.0)
            
            
            self.func_list = pauli_ops
      
        if cfg.fm_kind.lower() == "eff_su2":
    
            print("In SU2 encoder part")
            
            su2_name, su2_op = build_efficient_su2_oplist_qisk_new(
                D=self.D, n_wires=cfg.n_wires,
                single_ops=("ry","rz"),
                entanglement=cfg.fm_eff_ent_kind, twoq=cfg.fm_eff_twoq_op,
                pad_mode=cfg.fm_eff_pad_mod, alpha=cfg.fm_eff_alpha
            )
            
            save_oplist_py("efficient_su2_3w_ecp2.py", su2_name, su2_op)
            # exit()
            # Note: out of the Gen ecnoder for SU2, run for longer epochs to see which one works best
            # self.encoder = self.encoder = tq.GeneralEncoder(su2_op) # older one working as well
            self.encoder = GeneralEncoderPlus_new(su2_op, alpha=1.0, multi_index_rule="prod")
            self.func_list = su2_op
            # exit()
     
    #  55555555555555555
        # this part is to generate variation quantum circuit
        
        self.vqc_circuit = QuantumCircuit(
            n_wires=cfg.n_wires,
            depth=cfg.depth,
            n_random_ops=cfg.n_random_ops,
            ent_kind=cfg.qlayer_ent_kind,
            twoq_op=cfg.qlayer_twoq_op,
            ent_trainable=cfg.qlayer_ent_trainable,
            ent_wire_reverse=cfg.qlayer_ent_wire_reverse,
        )
        
        # pick ops automatically if not provided (use #classes as K by default)

        self.measure = tq.MeasureAll(tq.PauliZ)

        if not self.cfg.measure_ops:
            self.cfg.measure_ops = auto_pair_ops(self.cfg.num_classes) # NOte: we may want to debug this
            # print(f"Measuring ops: {self.cfg.measure_ops}")
            # exit()
        
        # pick pairs/topology automatically if not provided
        if not self.cfg.measure_pairs:
            if self.cfg.pair_topology == "ring":
                self.pairs = _ring_pairs(self.cfg.n_wires)
            else:
                self.pairs = _default_disjoint_pairs(self.cfg.n_wires)
        else:
            self.pairs = self.cfg.measure_pairs
            
        # decide head input size based on pooling choice
        if self.cfg.pool_pairs:
            in_feats = len(self.cfg.measure_ops)              # (B, K)
        else:
            in_feats = len(self.cfg.measure_ops) * len(self.pairs)  # (B, P*K)

        # self.head = QFCHead(in_features=in_feats, num_classes=self.cfg.num_classes)
        # if using measure all from tq
        self.head = QFCHead(in_features=self.cfg.n_wires, num_classes=self.cfg.num_classes)

            
        # if not getattr(self.cfg, "measure_ops", None):
        #     self.cfg.measure_ops = auto_pair_ops(self.cfg.num_classes)
        # self.head = QFCHead(self.cfg.measure_ops, self.cfg.num_classes)
        print(f"M Ops: {self.cfg.measure_ops}, Pairs: {self.pairs}, Head in_features: {in_feats}, classes: {self.cfg.num_classes}")
    


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        # 1 Build quantum device
        qdev = tq.QuantumDevice(
            n_wires=self.cfg.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=False
        )

        # 2 Encode classical features
        # feats = self._prep_features(x, self.cfg.pool_hw)  # (B, 16)
        # feats = self.cnn(x) (drop out ->0.1, trAcc 98 and teAcc 87, bz 32,)
        feats = self.fe(x)
        # print(f"Features from CNN shape: {feats.shape}")
        # print(f"Features from CNN (first 5 samples): {feats[:1]}")
        # #
        # exit()
        self.encoder(qdev, feats) # Quantum Feature mapping

        # 3 Variational circuit
        self.vqc_circuit(qdev)
       
        # 4 Measurements (joint Pauli expectations)
        # sssssssssss
        # if self.cfg.pool_pairs:

        #     measured = _measure_pooled_pairs(qdev, self.cfg.measure_ops, self.pairs, self.cfg.n_wires)  # (B, K)
        # else:
        #     measured = _measure_unpooled_pairs(qdev, self.cfg.measure_ops, self.pairs, self.cfg.n_wires)  # (B, P*K)
        
        
        measured = self.measure(qdev) # use the built-in measure all
        # print(f"Measured shape: {measured.shape}")
        # print few of the measured values
        # print(f"Measured values (first 5 samples): {measured[:5]}")
        # exit()
        logits = self.head(measured)
        return F.log_softmax(logits, dim=1)


class QuanvolutionFilter(tq.QuantumModule):
    def __init__(self, n_wires, bsz, device):
        super().__init__()
        self.n_wires = 4
        self.device = device
        # self.q_device = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        # print("Initializing Quanv filter")
        # print(f"Number of wires: {self.n_wires}, Batch size: {bsz}, Device: {device}")  
        # # exit()
        self.encoder = tq.GeneralEncoder(
        [   {'input_idx': [0], 'func': 'ry', 'wires': [0]},
            {'input_idx': [1], 'func': 'ry', 'wires': [1]},
            {'input_idx': [2], 'func': 'ry', 'wires': [2]},
            {'input_idx': [3], 'func': 'ry', 'wires': [3]},])

        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        size = 28
        x = x.view(bsz, size, size)

        data_list = []
        self.q_device = tq.QuantumDevice(self.n_wires, bsz=bsz, device=self.device)
        # print("Quanv filter forward pass")
        # print(f"Input x shape: {x.shape}")
        # exit()
        for c in range(0, size, 2):
            for r in range(0, size, 2):
                data = torch.transpose(torch.cat((x[:, c, r], x[:, c, r+1], x[:, c+1, r], x[:, c+1, r+1])).view(4, bsz), 0, 1)
                if use_qiskit:
                    data = self.qiskit_processor.process_parameterized(
                        self.q_device, self.encoder, self.q_layer, self.measure, data)
                else:
                    # print("Processing patch through quantum circuit")
                    # print(f"Patch data shape: {data.shape}")
                    # # exit()
                    self.encoder(self.q_device, data)
                    # print("Encoding done")
                    # exit()
                    self.q_layer(self.q_device)
                    data = self.measure(self.q_device)

                data_list.append(data.view(bsz, 4))
        
        result = torch.cat(data_list, dim=1).float()
        # print(f"Quanv filter output shape: {result.shape}")
        # exit()
        return result


from torchquantum.layer import U3CU3Layer0, NLocal
class  TrainableQuanvFilter(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        self.arch = {"n_wires": self.n_wires, "n_blocks": 7, "n_layers_per_block": 2}
        # self.arch = {"n_wires": self.n_wires, "n_blocks": 7, "n_layers_per_block": 2}
        # self.arch = {"n_wires": self.n_wires, "n_blocks": 2, "n_layers_per_block": 2}

        self.q_layer = U3CU3Layer0(self.arch)
        # self.q_layer = NLocal()
        # self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        x = F.avg_pool2d(x, 6).view(bsz, 4, 4)
        size = 4
        stride = 2
        x = x.view(bsz, size, size)

        data_list = []
        
    # [   (0,0) (0,1) (0,2) (0,3)
    #     (1,0) (1,1) (1,2) (1,3)
    #     (2,0) (2,1) (2,2) (2,3)
    #     (3,0) (3,1) (3,2) (3,3)]
        for c in range(0, size, stride): # range(0, max, stepsize)
            for r in range(0, size, stride):
                data = torch.transpose(
                    torch.cat(
                        (x[:, c, r], x[:, c, r + 1], x[:, c + 1, r], x[:, c + 1, r + 1])
                    ).view(4, bsz),
                    0,
                    1,
                )
                # print(f"Patch data shape: {data.shape}")  # (B, 4)
                # print(f"Patch data (first sample): {data}")
                # # exit()
                if use_qiskit:
                    data = self.qiskit_processor.process_parameterized(
                        qdev, self.encoder, self.q_layer, self.measure, data
                    )
                else:
                    self.encoder(qdev, data)
                    self.q_layer(qdev)
                    data = self.measure(qdev)

                data_list.append(data.view(bsz, 4))

        # print(f"data: {data.shape}")
        # print(f"data_list length: {len(data_list)}")
        # exit()
        # transpose to (bsz, channel, 2x2)
        result = torch.transpose(
            torch.cat(data_list, dim=1).view(bsz, 4, 4), 1, 2
        ).float()

        # print(f"result: {result.shape}")
        # exit()
        return result

class TrainableQuanvFilter_2(tq.QuantumModule):
    def __init__(self, *, stride=1):
        super().__init__()
        self.n_wires = 4
        self.stride = stride  # <-- now configurable; default 1 (overlapping)

        # angle encoder: map 4 scalars -> 4 RY gates (1 per wire)
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # quantum block (same as your original)
        self.arch = {"n_wires": self.n_wires, "n_blocks": 7, "n_layers_per_block": 2}
        self.q_layer = U3CU3Layer0(self.arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Optional: if you sometimes use qiskit path, set this outside or inject later.
        self.qiskit_processor = None  # placeholder; set if you use qiskit

    def forward(self, x, use_qiskit: bool = False):
        """
        x: (bsz, C, 28, 28) or (bsz, 1, 28, 28). We pool to 4x4, drop channel, then
           slide a 2x2 window with stride=self.stride (default 1).
        Returns: (bsz, 4, G, G) where G = 1 + (size-2)//stride (size=4 -> G=3 if stride=1).
        """
        bsz = x.shape[0]

        # (1) 6x6 avg pooling turns 28x28 -> 4x4 (since floor((28-6)/6 + 1) = 4)
        # Default stride = kernel_size, padding=0 in F.avg_pool2d
        x = F.avg_pool2d(x, 6)              # (bsz, C, 4, 4)
        x = x.view(bsz, 4, 4)               # drop channel -> (bsz, 4, 4)

        # (2) slide a 2x2 window with given stride over the 4x4 grid
        size = 4
        stride = self.stride
        assert stride >= 1, "stride must be >= 1"

        # top-left indices must allow c+1, r+1 to be valid -> up to size-2 inclusive
        # using range(0, size-1, stride) ensures c+1, r+1 are within [1..3]
        data_list = []
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)

        for c in range(0, size - 1, stride):
            for r in range(0, size - 1, stride):
                # collect the 2x2 patch values in (tl, tr, bl, br) order
                # shapes each: (bsz,)
                tl = x[:, c,     r    ]
                tr = x[:, c,     r + 1]
                bl = x[:, c + 1, r    ]
                br = x[:, c + 1, r + 1]

                # stack into (bsz, 4) for 4 input angles per sample
                data = torch.stack((tl, tr, bl, br), dim=1)  # (bsz, 4)

                # run through the quantum circuit
                if use_qiskit:
                    if self.qiskit_processor is None:
                        raise RuntimeError("qiskit_processor not set but use_qiskit=True")
                    data = self.qiskit_processor.process_parameterized(
                        qdev, self.encoder, self.q_layer, self.measure, data
                    )  # expected (bsz, 4) after measurement
                else:
                    self.encoder(qdev, data)  # load angles
                    self.q_layer(qdev)        # apply trainable circuit
                    data = self.measure(qdev) # (bsz, 4) in Z-basis

                data_list.append(data.view(bsz, 4))

        # (3) Arrange patches into a feature map: (bsz, 4, G, G)
        # number of positions per axis (top-lefts)
        G = 1 + (size - 2) // stride  # for size=4: stride=1 -> 3, stride=2 -> 2
        num_patches = G * G
        out = torch.stack(data_list, dim=1)         # (bsz, num_patches, 4)
        out = out.view(bsz, G, G, 4).permute(0, 3, 1, 2).contiguous()  # (bsz, 4, G, G)


        out = torch.cat(out, dim=1).view(bsz, 4, 3, 3)
        out = out.mean(dim=[2])  # average across 3x3 patches
        # result shape: [bsz, 4, 3]
        # Optionally pad to [bsz,4,4] if required
        out = F.pad(out, (0,1))  # adds one more feature to make [bsz,4,4]

        print(f"out size: {out.shape}")
        exit()
        return out.float()
    

class TrainableQuanvFilter_3(tq.QuantumModule):
    def __init__(self, *, pool_kernel=6, patch=2, stride=1, n_blocks=5, n_layers_per_block=2):
        super().__init__()
        self.n_wires = 4
        self.pool_kernel = pool_kernel          # 6 for 28x28 -> 4x4
        self.patch = patch                      # 2x2 quantum "receptive field"
        self.stride = stride                    # 1 (dense) or 2 (non-overlapping)
        
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        arch = {"n_wires": self.n_wires, "n_blocks": n_blocks, "n_layers_per_block": n_layers_per_block}
        self.q_layer = U3CU3Layer0(arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        """
        x: (B, C?, H=28, W=28) or (B, 28, 28). If a channel dim is present, we assume single-channel and squeeze it.
        Returns: (B, 4, H_out, W_out) where H_out = W_out = (pooled_size - patch)//stride + 1
        """
        bsz = x.shape[0]

        # allow (B, 1, 28, 28) or (B, 28, 28)
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)  # -> (B, 28, 28)
        elif x.dim() == 4 and x.shape[1] != 1:
            raise ValueError("Expected single-channel input (B,1,H,W) or (B,H,W).")

        # avg pool to reduce to a small grid (28x28 -> 4x4 with kernel=6, stride=6)
        x = F.avg_pool2d(x, kernel_size=self.pool_kernel).view(bsz, -1)  # (B, 16)
        size = int((x.shape[1]) ** 0.5)
        x = x.view(bsz, size, size)  # (B, 4, 4) for 28->6 pool

        patch = self.patch  # 2
        stride = self.stride
        # sliding window over the pooled grid
        H_out = (size - patch) // stride + 1
        W_out = (size - patch) // stride + 1
        if H_out <= 0 or W_out <= 0:
            raise ValueError("Invalid (size, patch, stride) combination.")

        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)

        data_list = []  # will hold (B, 4) chunks, one per patch, then we stack

        # iterate valid top-left corners of 2x2 patches
        for c in range(0, size - patch + 1, stride):
            for r in range(0, size - patch + 1, stride):
                # collect the 2x2 = 4 values from this patch for every batch element
                # x has shape (B, size, size); we want (B, 4)
                patch_vals = torch.stack(
                    (
                        x[:, c,     r    ],
                        x[:, c,     r + 1],
                        x[:, c + 1, r    ],
                        x[:, c + 1, r + 1],
                    ),
                    dim=1,  # (B, 4)
                )

                if use_qiskit:
                    data = self.qiskit_processor.process_parameterized(
                        qdev, self.encoder, self.q_layer, self.measure, patch_vals
                    )  # expected (B, 4)
                else:
                    self.encoder(qdev, patch_vals)
                    self.q_layer(qdev)
                    data = self.measure(qdev)  # (B, 4) expectation values

                data_list.append(data)  # each is (B, 4)

        # stack all patches along a new "patch" axis
        # shape: (num_patches, B, 4) -> (B, num_patches, 4)
        patches = torch.stack(data_list, dim=1)  # (B, H_out*W_out, 4)

        # rearrange to (B, 4, H_out, W_out)
        result = patches.view(bsz, H_out, W_out, 4).permute(0, 3, 1, 2).contiguous().float()


        # result = torch.cat(data_list, dim=1).view(bsz, 4, 3, 3)
        result = result.mean(dim=[2])  # average across 3x3 patches
        # result shape: [bsz, 4, 3]
        # Optionally pad to [bsz,4,4] if required
        result = F.pad(result, (0,1))  # adds one more feature to make [bsz,4,4]
        # print(f"results shpape: {result.shape}")
        # exit()
        return result


class TrainableQuanvFilter_4(tq.QuantumModule):
    """
    Quanvolutional filter that:
      - Slides 2x2 non-overlapping patches over 28x28 inputs -> [B, 4, 14, 14]
      - Optionally compresses to exactly n_features (<= 20) via adaptive avg pooling -> [B, n_features]
    """
    def __init__(self, n_features: Optional[int] = None):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.arch = {"n_wires": self.n_wires, "n_blocks": 7, "n_layers_per_block": 2}
        self.q_layer = U3CU3Layer0(self.arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

        # If provided, we will output exactly n_features (<= 20) as [B, n_features]
        if n_features is not None and (n_features < 1 or n_features > 20):
            raise ValueError("n_features must be in [1, 20].")
        self.n_features = n_features

    @torch.no_grad()
    def _ensure_image_shape(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B, 1, 28, 28] or [B, 28, 28]; convert to [B, 28, 28]
        if x.dim() == 4 and x.shape[1] == 1:
            return x[:, 0, :, :]
        elif x.dim() == 3:
            return x
        else:
            raise ValueError("Input must be [B, 1, 28, 28] or [B, 28, 28].")

    def forward(self, x: torch.Tensor, use_qiskit: bool = False):
        """
        Returns:
          - If self.n_features is None: [B, 4, 14, 14] quanvolution map (PennyLane-style)
          - Else: [B, self.n_features] pooled features (<= 20) for 10-qubit encoder
        """
        B = x.shape[0]
        x_img = self._ensure_image_shape(x)  # [B, 28, 28]
        device = x.device

        # Quantum device once per batch
        qdev = tq.QuantumDevice(self.n_wires, bsz=B, device=device)

        # Collect per-patch quantum outputs (each is [B, 4])
        data_list = []
        # Slide 2x2 non-overlapping patches with stride=2: j,k in {0,2,...,26} -> 14x14=196 patches
        for j in range(0, 28, 2):
            for k in range(0, 28, 2):
                # Build the 4-dim input vector per sample: (j,k), (j,k+1), (j+1,k), (j+1,k+1)
                # Shape -> [B, 4]
                patch = torch.stack(
                    [
                        x_img[:, j,     k    ],
                        x_img[:, j,     k + 1],
                        x_img[:, j + 1, k    ],
                        x_img[:, j + 1, k + 1],
                    ],
                    dim=1,
                )

                if use_qiskit:
                    # If you have a qiskit processor integrated, call it here
                    # data = self.qiskit_processor.process_parameterized(
                    #     qdev, self.encoder, self.q_layer, self.measure, patch
                    # )
                    raise NotImplementedError("use_qiskit=True path not implemented here.")
                else:
                    self.encoder(qdev, patch)
                    self.q_layer(qdev)
                    data = self.measure(qdev)  # [B, 4] expectation values

                data_list.append(data)  # each [B, 4]

        # Stack into [B, 4, 14, 14] map (PennyLane-style)
        y = torch.cat(data_list, dim=1).view(B, 4, 14, 14).float()

        if self.n_features is None:
            # Return the full quanv map
            return y

        # === Compress to exactly n_features (<= 20) ===
        # Flatten spatial -> [B, 4, 196]
        y_flat = y.view(B, 4, -1)  # 196 positions per channel

        # Choose K bins so that 4*K >= n_features, then slice to n_features
        n_feats = int(self.n_features)
        K = (n_feats + 3) // 4  # ceil(n_feats / 4)

        # Adaptive average pool the 196 positions down to K bins per channel -> [B, 4, K]
        y_pooled = F.adaptive_avg_pool1d(y_flat, output_size=K)

        # Concatenate channels -> [B, 4*K], then slice first n_features
        features = y_pooled.reshape(B, 4 * K)[:, :n_feats]  # [B, n_features]
        return features


# QuanVolutional Neural Network (QCNN) target model
class QCNN(tq.QuantumModule):
    def __init__(self, cfg: QFCConfig):
        super().__init__()
        self.cfg = cfg
        self.D = cfg.feature_dim if cfg.feature_dim is not None else cfg.pool_hw * cfg.pool_hw
        if self.D <= 0:
            raise ValueError("Feature dimension must be positive.")
        self.func_list = List[Dict[str, Any]]

        self.qf = TrainableQuanvFilter()
        # self.qf = TrainableQuanvFilter_3()
        # self.qf = TrainableQuanvFilter_4(n_features=16)



        # llllllllllllllllll
        # self.qf_hybrid = QuanvolutionFilter(cfg.n_wires, cfg.batch_size, cfg.device)
        # self.linear_hybrid = torch.nn.Linear(4*14*14, 10)
        self.linear_layer = torch.nn.Linear(16, 10)
        # print the cnn model and exit
        # print(self.cnn)   
        # exit()
       
        self.measure = tq.MeasureAll(tq.PauliZ)

        if cfg.fm_kind.lower() == "z":
            z_name, z_ops = build_tiled_pauli_oplist(
                n_wires=cfg.n_wires, D=self.D, paulis=["Z"], pad_mode=cfg.fm_z_pad_mode, repeats=cfg.fm_z_reps,
                expand_h_to_u3=True,
            )
            # self.encoder =  make_encoder_from_oplist(oplist, alpha=1.0, multi_index_rule="prod")
            write_oplist_py("z_with_pauli.py", z_name, z_ops)
            self.encoder =  GeneralEncoderPlus_new(z_ops, alpha=cfg.fm_z_alpha, multi_index_rule="sum")
            # self.encoder = GeneralEncoderPlus_new(z_ops, pad_mode="wrap", alpha=1.0)
            
            
            self.func_list = z_ops
            # exit()
        
        if cfg.fm_kind.lower() == "zz":
            
           
            zz_name, zz_ops = build_tiled_pauli_oplist(
                n_wires=cfg.n_wires, D=self.D, paulis=["Z", "ZZ"], pad_mode=cfg.fm_zz_pad_mode, entanglement=cfg.fm_zz_entanglement, repeats=cfg.fm_zz_reps,
                expand_h_to_u3=True,
            )
            # self.encoder =  make_encoder_from_oplist(oplist, alpha=1.0, multi_index_rule="prod")
            write_oplist_py("zasdasd_with_pauli.py", zz_name, zz_ops)
            self.encoder =  GeneralEncoderPlus_new(zz_ops, alpha=cfg.fm_zz_alpha, multi_index_rule="sum")
            # self.encoder = GeneralEncoderPlus_new(z_ops, pad_mode="wrap", alpha=1.0)
            
            
            self.func_list = zz_ops
            
            # exit()
        # yyyyyyyy
        if cfg.fm_kind.lower() == "pauli":
            print("in pauli maping op list generator")
            pauli_name, pauli_ops = build_tiled_pauli_oplist(
                n_wires=cfg.n_wires, D=self.D, paulis=cfg.fm_pauli_terms, pad_mode=cfg.fm_pali_pad, entanglement=cfg.fm_pauli_entanglement, repeats=cfg.fm_pauli_reps,
                expand_h_to_u3=True,
            )
            # self.encoder =  make_encoder_from_oplist(oplist, alpha=1.0, multi_index_rule="prod")
            write_oplist_py("pauli_with_pauli.py", pauli_name, pauli_ops)
            self.encoder =  GeneralEncoderPlus_new(pauli_ops, alpha=cfg.fm_pauli_alpha, multi_index_rule="prod")
            # self.encoder = GeneralEncoderPlus_new(z_ops, pad_mode="wrap", alpha=1.0)
            
            
            self.func_list = pauli_ops
      
        if cfg.fm_kind.lower() == "eff_su2":
    
            print("In SU2 encoder part")
            
            su2_name, su2_op = build_efficient_su2_oplist_qisk_new(
                D=self.D, n_wires=cfg.n_wires,
                single_ops=("ry","rz"),
                entanglement=cfg.fm_eff_ent_kind, twoq=cfg.fm_eff_twoq_op,
                pad_mode=cfg.fm_eff_pad_mod, alpha=cfg.fm_eff_alpha
            )
            
            save_oplist_py("efficient_su2_3w.py", su2_name, su2_op)
            # exit()
            # Note: out of the Gen ecnoder for SU2, run for longer epochs to see which one works best
            # self.encoder = self.encoder = tq.GeneralEncoder(su2_op) # older one working as well
            self.encoder = GeneralEncoderPlus_new(su2_op, alpha=1.0, multi_index_rule="prod")
            self.func_list = su2_op
            # exit()
     
        # this part is to generate variation quantum circuit
        
        self.vqc_circuit = QuantumCircuit(
            n_wires=cfg.n_wires,
            depth=cfg.depth,
            n_random_ops=cfg.n_random_ops,
            ent_kind=cfg.qlayer_ent_kind,
            twoq_op=cfg.qlayer_twoq_op,
            ent_trainable=cfg.qlayer_ent_trainable,
            ent_wire_reverse=cfg.qlayer_ent_wire_reverse,
        )
        
        # pick ops automatically if not provided (use #classes as K by default)


        if not self.cfg.measure_ops:
            self.cfg.measure_ops = auto_pair_ops(self.cfg.num_classes) # NOte: we may want to debug this
            # print(f"Measuring ops: {self.cfg.measure_ops}")
            # exit()
        
        # pick pairs/topology automatically if not provided
        if not self.cfg.measure_pairs:
            if self.cfg.pair_topology == "ring":
                self.pairs = _ring_pairs(self.cfg.n_wires)
            else:
                self.pairs = _default_disjoint_pairs(self.cfg.n_wires)
        else:
            self.pairs = self.cfg.measure_pairs
        
        # decide head input size based on pooling choice
        if self.cfg.pool_pairs:
            in_feats = len(self.cfg.measure_ops)              # (B, K)
        else:
            in_feats = len(self.cfg.measure_ops) * len(self.pairs)  # (B, P*K)

        # self.head = QFCHead(in_features=self.cfg.num_classes, num_classes=self.cfg.num_classes)
        # self.head = QFCHead(in_features=in_feats, num_classes=self.cfg.num_classes)

        self.head = QFCHead(in_features=self.cfg.n_wires, num_classes=self.cfg.num_classes) # this is for built-in measure all
        


        self.measure = tq.MeasureAll(tq.PauliZ)
        
            
        # if not getattr(self.cfg, "measure_ops", None):
        #     self.cfg.measure_ops = auto_pair_ops(self.cfg.num_classes)
        # self.head = QFCHead(self.cfg.measure_ops, self.cfg.num_classes)
        print(f"M Ops: {self.cfg.measure_ops}, Pairs: {self.pairs}, Head in_features: {in_feats}, classes: {self.cfg.num_classes}")
    

        self.arch = {"n_wires": cfg.n_wires, "n_blocks": 5, "n_layers_per_block": 2}
        # self.arch = {"n_wires": self.n_wires, "n_blocks": 2, "n_layers_per_block": 2}
        #1111111111111111111
        self.q_layer = U3CU3Layer0(self.arch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # print(f"Input x shape: {x.shape}")
        # bsz = x.shape[0]
        # x = x.view(bsz, 28, 28)
        # print(f"after reshaping Input x shape: {x.shape}")
        # exit()

        # with torch.no_grad():
        #   x = self.qf_hybrid(x, use_qiskit=False)
        # # print(f"Quanv output shape: {x.shape}")
        
        # x = self.linear_hybrid(x)
        # # print(f"Linear output shape: {x.shape}")
        # # exit()
        # return F.log_softmax(x, -1)
        
        
        bsz = x.size(0)
        # 1) Build quantum device
        qdev = tq.QuantumDevice(
            n_wires=self.cfg.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=False
        )

        # 2) Encode classical features
        # feats = self._prep_features(x, self.cfg.pool_hw)  # (B, 16)
        # feats = self.cnn(x) (drop out ->0.1, trAcc 98 and teAcc 87, bz 32,)
        # print(f"Input x shape: {x.shape}")
        # exit()
        
        x = x.view(-1, 28, 28)
        # print(f"after reshaping Input x shape: {x.shape}")
        # exit()
        
        # torch no grad the following
        # with torch.no_grad():
        feats = self.qf(x)
        # print(f"Quanv output shape: {feats.shape}")
        # print(f"Features from Quanv (first 5 samples): {feats[:1]}")
        # exit()
                # feats = feats.reshape(-1, 16)
                # # print(f"Quanv output shape: {feats.shape}")
                # # exit()
                # logits = self.linear_layer(feats)
                
                # return F.log_softmax(logits, dim=1)
        # print(f"Features from Quanv reshaped to: {feats.shape}")
        # exit()
        # qqqqqqqqqqqqqqqq
        # feats = self.fe(x)
        feats = feats.reshape(-1, 16)
        # print(f"Features from CNN shape: {feats.shape}")
        # print(f"Features from CNN (first 5 samples): {feats[:1]}")
        # #
        # exit()
        self.encoder(qdev, feats) # Quantum Feature mapping

        # 3) Variational circuit
        self.vqc_circuit(qdev)
        # self.q_layer(qdev)
       
        # 4) Measurements (joint Pauli expectations)
        # sssssssssss
        # if self.cfg.pool_pairs:

        #     measured = _measure_pooled_pairs(qdev, self.cfg.measure_ops, self.pairs, self.cfg.n_wires)  # (B, K)
        # else:
        #     measured = _measure_unpooled_pairs(qdev, self.cfg.measure_ops, self.pairs, self.cfg.n_wires)  # (B, P*K)
        measured = self.measure(qdev)
        
        # measured = self.measure(qdev) # use the built-in measure all
        # print(f"Measured shape: {measured.shape}")
        # # print few of the measured values
        # # print(f"Measured values (first 5 samples): {measured[:5]}")
        # exit()
        logits = self.head(measured)
        return F.log_softmax(logits, dim=1)

# def save_model(model, path):
#     torch.save(model.state_dict(), path)

    

# def load_model(model, path):
#     # sssssssssssssssssssss
#     # state = torch.load(path, map_location="cuda")
#     # # model.load_state_dict(torch.load(path))
#     # model.load_state_dict(state)
#     model2 = model.load_state_dict(torch.load(path))
    
#     return model2


def save_model(model, path):
    state = {
        k: v
        for k, v in model.state_dict().items()
        if "q_device" not in k  # drop simulator state buffers
    }
    torch.save(state, path)

def load_model(model, path, map_location="cuda"):
    state = torch.load(path, map_location=map_location)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"Ignored unexpected keys: {unexpected}")
    if missing:
        print(f"Ignored missing keys: {missing}")
    return model


# ----------------------------
# Train / Eval
# ----------------------------
def train_one_epoch(dataflow, model, device, optimizer):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for feed in dataflow["train"]:
        inputs = feed["image"].to(device, non_blocking=True).float()
        # print(f"Training batch input shape: {inputs.shape}")
        # print(f"Training batch input (first sample): {inputs[0]}")
        # print(f"Training batch input (first sample) shape: {inputs[0].shape}")
        # exit()

        if inputs.dim() >= 3:
            inputs = torch.pi * torch.tanh(inputs)/2.0
        # else:
        #     inputs = inputs.clamp(-1.0, 1.0) # this for blobs/moons/circles with qnn model
        # inputs = inputs.clamp(-1.0, 1.0)* math.pi
        # inputs = inputs.clamp(-1.0, 1.0)
        
        # inputs = torch.pi * torch.tanh(inputs) # for qcnn it will stick with this
        
        # inputs = inputs.mul(2.0).sub(1.0)
        # inputs = torch.pi * torch.sigmoid(inputs)
        # inputs = to_phase_from_minus1_1(inputs)

        

        # print(f"input shape: {inputs.shape}")
        # print(f"input (first sample): {inputs[:50]}")
        # exit()
        # inputs = torch.pi * inputs

        targets = feed["digit"].to(device, non_blocking=True)
        outputs = model(inputs) # the model here returns log softmax outputs, not logits, so no softmax needed
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bsz = targets.size(0)
        total_loss += loss.item() * bsz
        total_samples += bsz
        total_correct += (outputs.argmax(1) == targets).sum().item()
    return (total_loss / max(1, total_samples),
            total_correct / max(1, total_samples))





def to_phase_from_minus1_1(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(-1.0, 1.0)      # ensure inside [-1, 1]
    x = (x + 1.0) / 2.0         # [-1,1] -> [0,1]
    x = x * (1 - eps) + eps     # [0,1] -> (0,1]
    return x * (2 * math.pi)    # (0,1] -> (0, 2π]
    

@torch.no_grad()
def evaluate(dataflow, split, model, device):
    model.eval()
    targets_all, outputs_all = [], []
    for feed in dataflow[split]:
        inputs = feed["image"].to(device, non_blocking=True).float()
        if inputs.dim() >= 3:
            inputs = torch.pi * torch.tanh(inputs)/2.0
        # else:
        #     inputs = inputs.clamp(-1.0, 1.0)
        # inputs = inputs.clamp(-1.0, 1.0)* math.pi
        # inputs = torch.pi * torch.tanh(inputs)
        # inputs = inputs.clamp(-1.0, 1.0)
        # inputs = to_phase_from_minus1_1(inputs)

        # inputs = inputs.mul(2.0).sub(1.0)
        # inputs = torch.pi * inputs
        targets = feed["digit"].to(device, non_blocking=True)
        outputs = model(inputs) # the model here returns log softmax outputs, not logits, so no softmax needed
        targets_all.append(targets)
        outputs_all.append(outputs)
    targets_all = torch.cat(targets_all, dim=0)
    outputs_all = torch.cat(outputs_all, dim=0)
    preds = outputs_all.argmax(dim=1)
    acc = (preds == targets_all).float().mean().item()
    loss = F.nll_loss(outputs_all, targets_all).item()
    return loss, acc



def _preprocess_like_train(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Matches the preprocessing used in train_one_epoch/evaluate:
      - move to device
      - float()
      - for image-like tensors (dim >= 3): x -> (pi/2) * tanh(x)
    """
    x = x.to(device, non_blocking=True).float()
    if x.dim() >= 3:
        x = torch.pi * torch.tanh(x) / 2.0
    # else:
    #     Keep as-is (or add your vector scaling here if you later enable it)
    return x


@torch.no_grad()
def collect_probs_and_labels(dataflow, split: str, model, device):
    """
    Collect prediction vectors (PVs) and labels from `dataflow[split]`.

    Assumptions:
      - model(x) returns LOG-probabilities (log-softmax already applied).
      - Therefore, PVs as probabilities are obtained via exp(log_probs).

    Returns:
      probs_all: Tensor [N, C] (probability vectors)
      y_all:     Tensor [N]
    """
    model.eval()
    probs_all = []
    y_all = []

    for batch in dataflow[split]:
        x = _preprocess_like_train(batch["image"], device)
        y = batch["digit"].to(device, non_blocking=True).long()

        log_probs = model(x)          # already log-softmax'd inside forward
        probs = log_probs.exp()       # convert log-probs -> probs (NO extra softmax)

        probs_all.append(probs.detach().cpu())
        y_all.append(y.detach().cpu())

    probs_all = torch.cat(probs_all, dim=0)
    y_all = torch.cat(y_all, dim=0)
    return probs_all, y_all


def build_attack_features(probs: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Returns:
      X_pv_stats: [N, C + 5] = [pv || loss || entropy || conf || margin || correct]
      stats dict
    """
    eps = 1e-12
    N, C = probs.shape

    y_true = y_true.long()
    pred = probs.argmax(dim=1)
    correct = (pred == y_true).float()

    p_true = probs[torch.arange(N), y_true].clamp_min(eps)
    loss = (-torch.log(p_true)).unsqueeze(1)  # [N,1]

    entropy = (-(probs.clamp_min(eps) * torch.log(probs.clamp_min(eps))).sum(dim=1)).unsqueeze(1)  # [N,1]

    top2 = probs.topk(k=min(2, C), dim=1).values
    conf = top2[:, 0:1]  # [N,1]
    if C >= 2:
        margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
    else:
        margin = conf.clone()

    correct_col = correct.unsqueeze(1)

    X = torch.cat([probs, loss, entropy, conf, margin, correct_col], dim=1)

    stats = dict(pred=pred, correct=correct, loss=loss.squeeze(1), entropy=entropy.squeeze(1),
                 conf=conf.squeeze(1), margin=margin.squeeze(1))
    return X, stats

import hashlib
import json

def _sha256_int_tensor(t: torch.Tensor) -> str:
    arr = t.detach().cpu().long().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()

@torch.no_grad()
def metrics_from_probs(probs: torch.Tensor, y_true: torch.Tensor) -> dict:
    eps = 1e-12
    y_true = y_true.long()
    pred = probs.argmax(dim=1)
    acc = (pred == y_true).float().mean().item()
    p_true = probs[torch.arange(len(y_true)), y_true].clamp_min(eps)
    loss = (-torch.log(p_true)).mean().item()
    return {"loss": float(loss), "acc": float(acc), "N": int(y_true.numel())}

def split_fingerprint(y_true: torch.Tensor, num_classes: int) -> dict:
    y = y_true.detach().cpu().long()
    hist = torch.bincount(y, minlength=num_classes).tolist()
    return {"sha256_y": _sha256_int_tensor(y), "hist": hist, "N": int(y.numel())}


# ----------------------------
# Script
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    # add arguments for n_wries
    parser.add_argument("--n-wires", type=int, default=10, help="Override number of qubits; auto-selected if omitted.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    #add runid argument
    parser.add_argument("--run-id", type=int, default=0, help="Run ID for experiment tracking.")
    # add target model path argument
    # parser.add_argument("--target-model-path", type=str, default="qcnn_model.pth", help="Path to save/load the trained model.")
    parser.add_argument("--target-model-path", type=str, default=None, help="Explicit path to save the trained model.")

    # Q-Layer arguments (QuantumCircuit)
    parser.add_argument("--depth", type=int, default=2)
    # add q layer entanglement kind
    parser.add_argument("--qlayer-ent-kind", choices=["linear", "ring","pairwise", "full"], default="full", help="Entanglement pattern for variational circuit.")
    # add these options below to --qlayer-twoq-op{
    #     "cx": tq.CNOT, "cnot": tq.CNOT, "cz": tq.CZ, "swap": tq.SWAP,
    #     "crx": tq.CRX, "cry": tq.CRY, "crz": tq.CRZ,
    #     "rxx": tq.RXX, "ryy": tq.RYY, "rzz": tq.RZZ,
    # }
    # add argurment fo qlayer_ent_trainable
    # parser.add_argument("--qlayer-ent-trainable", action="store_true", help="Make entangling gate parameters trainable.")
    parser.add_argument("--no-qlayer-ent-trainable", action="store_false",
        default=True,
        help="Disable trainable entangling gate parameters.",
    )
    # add argument for qlayer_ent_wire_reverse
    parser.add_argument("--qlayer-ent-wire-reverse", action="store_true", help="Reverse the order of wires for entangling gates.")
    
    # parser.add_argument("--qlayer-twoq-op", choices=["cx", "cz", "", "iswap"], default="cx", help="Two-qubit gate for variational circuit.")
    parser.add_argument("--qlayer-twoq-op",
                        choices=["cx", "cz", "swap", "crx", "cry", "crz", "rxx", "ryy", "rzz"],
                        default="cx",
                        help="Two-qubit gate for variational circuit.")
    
    parser.add_argument("--random-ops", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None, help="None => auto")
    parser.add_argument("--dataset", choices=["cifar10", "mnist", "moons", "circles", "blobs", "multiclass"], default="mnist")
    
    # add argument for feature map kind
    parser.add_argument("--fm-kind", choices=["z", "zz", "pauli", "eff_su2"], default="eff_su2", help="Type of quantum feature map to use.")
    
    parser.add_argument("--fm-z-reps", type=int, default=1, help="Repetitions for Z feature map.")
    parser.add_argument("--fm-z-alpha", type=float, default=1.0, help="Alpha scaling for Z feature map.")   
    parser.add_argument("--fm-z-pad-mode", choices=["wrap", "repeatlast", "zero"], default="wrap", help="Padding mode for Z feature map.")

        #     # fm_kind="z",
    #     # fm_z_reps=1,
    #     # fm_z_alpha=1.0,
    #     # fm_z_pad_mode="wrap", # pad_mode in ("wrap", "repeatlast", "zero")

    parser.add_argument("--fm-zz-pad-mode", choices=["wrap", "repeatlast", "zero"], default="wrap", help="Padding mode for ZZ feature map.")
    parser.add_argument("--fm-zz-entanglement", choices=["linear", "ring", "full"], default="linear", help="Entanglement pattern for ZZ feature map.")
    parser.add_argument("--fm-zz-reps", type=int, default=1, help="Repetitions for ZZ feature map.")
    # parser.add_argument("--fm-zz-alpha", type=float, default=1.0, help="Alpha scaling for ZZ feature map.")
    
      #     fm_kind = "zz",
    #     fm_zz_reps = 4,
    #     fm_zz_alpha = 1.0,
    #     fm_zz_entanglement = "linear",   # 'linear' | 'ring' | 'full'
    #     fm_zz_phi = "pi_minus",            # or prod  | 'pi_minus'
    #     fm_zz_pad_mode = "wrap",
        
        
    parser.add_argument("--fm-pauli-terms", nargs="+", default=["Z", "ZZ"], help="Pauli terms for Pauli feature map.")
    parser.add_argument("--fm-pauli-pad", choices=["wrap", "repeatlast", "zero"], default="wrap", help="Padding mode for Pauli feature map.")
    parser.add_argument("--fm-pauli-entanglement", choices=["linear", "ring", "full"], default="linear", help="Entanglement pattern for Pauli feature map.")
    parser.add_argument("--fm-pauli-reps", type=int, default=1, help="Repetitions for Pauli feature map.")
    parser.add_argument("--fm-pauli-alpha", type=float, default=1.0, help="Alpha scaling for Pauli feature map.")
    
      #     # #Note: do not use h, leads to theta errors for transpilation list in build_tiled_pauli_oplist
    #     # # for pauli feature map
    #     # fm_kind = "pauli",
    #     # fm_pauli_reps = 1,
    #     # fm_pauli_alpha = 1.0,
    #     # fm_pauli_entanglement = "linear",   # 'linear' | 'ring' | 'full'
    #     # fm_pali_pad = "wrap",
    #     # fm_pauli_terms = ["Z","ZZ"] #must be genralized for qubit numbers

    parser.add_argument("--fm-eff-ent-kind", choices=["linear", "ring", "full"], default="linear", help="Entanglement pattern for Efficient SU2 feature map.")
    parser.add_argument("--fm-eff-reps", type=int, default=1, help="Repetitions for Efficient SU2 feature map.")
    parser.add_argument("--fm-eff-twoq-op", choices=["cz", "cnot","cx", "iswap"], default="cx", help="Two-qubit gate for Efficient SU2 feature map.")
    parser.add_argument("--fm-eff-pad-mod", choices=["wrap", "repeatlast", "zero"], default="wrap", help="Padding mode for Efficient SU2 feature map.")
    parser.add_argument("--fm-eff-alpha", type=float, default=1.0, help ="Alpha scaling for Efficient SU2 feature map.")    
    
        # #     fm_kind = "eff_su2",
    # #     # fm_eff_reps = 2,
    # #     fm_eff_alpha = 1.0, # not sure what to sure, need to investigate
    # #     fm_eff_ent_kind = "linear",
    # #     fm_eff_pad_mod = "wrap", # 'wrap' | 'repeatlast' | 'pad'
    # #     fm_eff_twoq_op = "cx"

    # parser.add_argument("--ent-kind", choices=["linear", "full"], default="linear", help="Entanglement pattern for variational circuit.")
    # parser.add_argument("--twoq-op", choices=["cz", "cnot", "iswap"], default="cz", help="Two-qubit gate for variational circuit.")
    # parser.add_argument("--ent-trainable", action="store_true", help="Make entangling gates trainable.")


    
    # dddddddddddddd
    parser.add_argument("--pool-hw", type=int, default=4, help="Adaptive pooling HW for image datasets (ignored for vector data)")
    
    
    
    parser.add_argument("--vector-train", type=int, default=100, help="Samples for vector datasets (moons/circles/blobs) training split.")
    parser.add_argument("--vector-valid", type=int, default=2000, help="Samples for vector datasets validation split.")
    parser.add_argument("--vector-test", type=int, default=2000, help="Samples for vector datasets test split.")
    
    parser.add_argument("--moons-noise", type=float, default=0.1)
    parser.add_argument("--moons-separation", type=float, default=0.5)
    parser.add_argument("--vector-scale-to-2pi", action="store_true", help="Rescale vector dataset features into [0, 2π].")
    parser.add_argument("--extra-feats", action="store_true", help="Add extra features to vector datasets (moons/circles/blobs).")
   
    parser.add_argument("--circles-noise", type=float, default=0.1)
    parser.add_argument("--circles-factor", type=float, default=0.5)
    parser.add_argument("--blobs-cluster-std", type=float, default=0.8)
    parser.add_argument("--blobs-center-distance", type=float, default=3.0)
    parser.add_argument("--blobs-n-features", type=int, default=2)
    
    
    parser.add_argument("--multiclass-features", type=int, default=2)
    parser.add_argument("--multiclass-classes", type=int, default=2)
    parser.add_argument("--multiclass-labels", type=int, default=2)
    parser.add_argument("--multiclass-length", type=int, default=50)
    parser.add_argument("--multiclass-allow-unlabeled", action="store_true")
    # parser.add_argument("--n-wires", type=int, default=None, help="Override number of qubits; auto-selected if omitted.")
    parser.add_argument("--plot-vector", action="store_true", help="Plot the generated vector dataset split before training.")
    parser.add_argument("--plot-vector-split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--plot-vector-path", type=str, default=None, help="Optional path to save the vector scatter plot.")
    
    
    parser.add_argument("--train_target", action="store_true")

    # add target model type, like user has to enter and supported types are qnn, hqnn, qcnn
    parser.add_argument("--model-type", choices=["qnn", "hqnn", "qcnn", "mlp_qnn"], default="qnn", help="Type of quantum model to use.")


    # MIA ATTACK arguments
    parser.add_argument("--export-attack-data", action="store_true",
                    help="Export MIA dataset (PVs) for train+test splits using saved target model.")
    parser.add_argument("--attack-data-out", type=str, default=None,
                        help="Output .pt path for attack dataset. If None, saves alongside target model.")
    # parser.add_argument("--attack-feature-mode", choices=["pv", "pv+stats"], default="pv+stats",
                        # help="pv: only probability vector. pv+stats: append [loss, entropy, conf, margin, correct].")
    parser.add_argument("--attack-feature-mode", choices=["pv", "pv+stats"], default="pv+stats",
                        help="pv: only probability vector. pv+stats: append [loss, entropy, conf, margin, correct].")
    
    parser.add_argument("--attack-metrics-out", type=str, default=None,
                    help="Optional JSON path to write export metrics/fingerprints (besides the .pt).")

    
    args = parser.parse_args()
    # Repro
    

    
    if args.dataset in {"moons", "blobs", "circles"}:
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    elif args.dataset in {"mnist"}:
        # print("Setting seed for MNIST dataset...")
        # exit()
        seed = 43

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    

    # seed = 43

    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    

    if args.dataset == "mnist":
        n_train_samples = args.vector_train  # 600
        n_test_samples = args.vector_test  # 600
        n_valid_samples = args.vector_valid  # 600
        
        # dataset = MNIST(
        #     root="./mnist_data",
        #     train_valid_split_ratio=[0.5, 0.5],  # 600 / 600 split
        #     digits_of_interest=[0, 2, 4, 5, 6, 7, 8, 9, 1, 3],
        #     n_test_samples=n_test_samples,
        #     n_train_samples=n_train_samples,
        #     same_n_samples_each_class=True,
        # )
        dataset = MNIST(
            root="./mnist_data",
            train_valid_split_ratio=[0.9, 0.1],  # fine; we sub-sample later anyway
            digits_of_interest=[0, 1, 3, 8], #  0, 2, 4, 5, 6, 7, 8, 9, 1, 3
            n_test_samples=n_test_samples,
            n_valid_samples=n_valid_samples,     # <-- THIS WAS MISSING
            n_train_samples=n_train_samples,
            same_n_samples_each_class=True,
        )
        feature_dim = args.pool_hw * args.pool_hw
        # default_n_wires = args.n_wires if args.n_wires is not None else 10 # change as needed

    elif args.dataset == "cifar10":
        #     n_test_samples=100,
        # )
        # feature_dim = args.pool_hw * args.pool_hw * 3  # RGB
        # default_n_wires = 12 # change as needed
        dataset = CIFAR10(
        root="data/cifar10",
        train_valid_split_ratio=[0.9, 0.1],
        center_crop=32,
        resize=32,
        resize_mode="bilinear",
        binarize=False,
        grayscale=True,
        # digits_of_interest=tuple(range(10)),
        digits_of_interest=(0, 4),

        )

        train = dataset["train"]
        print("Train size:", len(train))
        sample = train[45]
        print("Keys in sample:", list(sample.keys()))
        print("Image shape:", sample["image"].shape)
        print("Label (mapped):", sample["digit"])

        feature_dim = args.pool_hw * args.pool_hw
        default_n_wires = 8 # change as needed
        # exit()
        # loader = DataLoader(train, batch_size=8, shuffle=True, num_workers=0)
        # batch = next(iter(loader))
        # print("Batch image tensor shape:", batch["image"].shape)
        # print("Batch labels:", batch["digit"])

    else:

        ## To DO: print the args for vector dataset corresponding the selected dataset
        print("Generating vector dataset...")
        print(f"Selected dataset: {args.dataset}")
        print(f"batch size: {args.batch_size}")
        print(f"epochs: {args.epochs}")

        if args.dataset in {"moons"}:
            print(f"'moons' dataset with noise={args.moons_noise}, separation={args.moons_separation}")
            print(f"dataset samples: train={args.vector_train}, valid={args.vector_valid}, test={args.vector_test}")
        elif args.dataset in {"circles"}:
            print(f"'circles' dataset with noise={args.circles_noise}, factor={args.circles_factor}")
            print(f"dataset samples: train={args.vector_train}, valid={args.vector_valid}, test={args.vector_test}")
        elif args.dataset in {"blobs"}:
            print(f"'blobs' dataset with cluster_std={args.blobs_cluster_std}, center_distance={args.blobs_center_distance}, n_features={args.blobs_n_features}")
            print(f"dataset samples: train={args.vector_train}, valid={args.vector_valid}, test={args.vector_test}")

        # exit()
        vector_kwargs = dict(
            kind=args.dataset,
            train_samples=args.vector_train,
            valid_samples=args.vector_valid,
            test_samples=args.vector_test,
            noise=args.moons_noise,
            separation=args.moons_separation,
            factor=args.circles_factor,
            cluster_std=args.blobs_cluster_std,
            center_distance=args.blobs_center_distance,
            n_features=args.blobs_n_features,
            seed=seed,
            # pppppppppppppppppp
            scale_to_2pi=False,
            multiclass_features=args.multiclass_features,
            multiclass_classes=args.multiclass_classes,
            multiclass_labels=args.multiclass_labels,
            multiclass_length=args.multiclass_length,
            multiclass_allow_unlabeled=args.multiclass_allow_unlabeled,
            extra_feats=args.extra_feats,
        )
        
        if args.dataset == "moons":
            vector_kwargs["noise"] = args.moons_noise
        elif args.dataset == "circles":
            vector_kwargs["noise"] = args.circles_noise
        
        dataset = build_vector_dataset_dict(**vector_kwargs)
        
        if args.plot_vector:
            plot_vector_dataset(dataset, split=args.plot_vector_split, save_path=args.plot_vector_path)
        
        feature_dim = dataset["train"].feature_dim
        default_n_wires = feature_dim

    dataset = {split: dataset[split] for split in dataset}
    num_classes = infer_num_classes(dataset)

    #print the total samoles in each split
    for split in dataset:
        print(f"Total samples in {split} split: {len(dataset[split])}")
    # exit()
    
    if args.dataset in {"moons", "circles", "blobs"}:
        num_classes = 2
    elif args.dataset == "multiclass":
        num_classes = args.multiclass_classes

    n_wires = args.n_wires if args.n_wires is not None else 10
    pool_hw_cfg = args.pool_hw if args.dataset in ("mnist", "cifar10") else 1

    # if args.dataset in {"moons", "circles", "blobs", "multiclass"} and n_wires != feature_dim:
    #     raise ValueError(
    #         f"For vector datasets, set --n-wires equal to feature dimension ({feature_dim}) to avoid redundant tiling."
    #     )
    
    print(f"Dataset kind: {args.dataset}")
    print(f"Detected num_classes={num_classes}")
    print(f"Using n_wires={n_wires}, feature_dim={feature_dim}, pool_hw={pool_hw_cfg}")
    print(f"args.model_type: {args.model_type}")
    # exit()
    train_sample = dataset["train"][0]
    sample_image = train_sample["image"]
    if torch.is_tensor(sample_image):
        sample_shape = tuple(sample_image.shape)
        sample_numel = int(sample_image.numel())
    else:
        sample_shape = type(sample_image)
        sample_numel = "n/a"
    print(f"Sample keys: {list(train_sample.keys())}")
    print(f"Sample 'image' shape: {sample_shape}, numel: {sample_numel}")
    print(f"Sample content: {train_sample}")
    print(f"Train split size: {len(dataset['train'])}")
    if getattr(dataset["train"], "scale", None):
        scale_cfg = dataset["train"].scale
        scale_min = scale_cfg['min'].tolist() if hasattr(scale_cfg['min'], 'tolist') else scale_cfg['min']
        scale_max = scale_cfg['max'].tolist() if hasattr(scale_cfg['max'], 'tolist') else scale_cfg['max']
        print(f"Vector scaling applied: target={scale_cfg['target']} min={scale_min} max={scale_max}")
    # here
    # exit()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Safer DataLoader defaults:
    # - workers=0 on CPU; a few workers on GPU
    # - pin_memory True only on CUDA
    # num_workers = (4 if use_cuda else 0) if args.workers is None else args.workers
    train_workers = 2 if use_cuda else 0
    eval_workers  = 0  # Windows-friendly, avoids worker crashes during eval
    dataflow = {}   
    for split in dataset:
        # print(f"Building DataLoader for split '{split}'...")
        # exit()
        is_train = (split == "train")
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=args.batch_size,
            shuffle=is_train,                 # train=True, val/test=False
            num_workers=train_workers if is_train else eval_workers,
            pin_memory=bool(use_cuda) if is_train else False,
            persistent_workers=False,         # safer on Windows
            drop_last=False,
        )
    

    mapper_cfg = {"fm_kind": args.fm_kind.lower()}

    if mapper_cfg["fm_kind"] == "z":
        mapper_cfg |= dict(
            fm_z_reps=args.fm_z_reps, 
            fm_z_alpha=1.0, 
            fm_z_pad_mode=args.fm_z_pad_mode
        )

        print("Using Z feature map with user-defined settings.")
        print(f"fm_z_reps: {args.fm_z_reps}")
        print(f"fm_z_alpha: {args.fm_z_alpha}")
        print(f"fm_z_pad_mode: {args.fm_z_pad_mode}")

    elif mapper_cfg["fm_kind"] == "zz":
        mapper_cfg |= dict(
            fm_zz_reps=args.fm_zz_reps,
            fm_zz_entanglement=args.fm_zz_entanglement,  # 'linear' | 'ring' | 'full'
            fm_zz_pad_mode=args.fm_zz_pad_mode,
            fm_zz_alpha=1.0,
            fm_zz_phi="pi_minus",  # or prod  | 'pi_minus'
        )

        print("Using ZZ feature map with user-defined settings.")
        print(f"fm_zz_reps: {args.fm_zz_reps}")
        print(f"fm_zz_entanglement: {args.fm_zz_entanglement}")
        print(f"fm_zz_pad_mode: {args.fm_zz_pad_mode}")
   
    elif mapper_cfg["fm_kind"] == "pauli":
        mapper_cfg |= dict(
            fm_pauli_reps=args.fm_pauli_reps,
            fm_pauli_alpha=1.0,
            fm_pauli_entanglement=args.fm_pauli_entanglement,  # 'linear' | 'ring' | 'full'
            fm_pali_pad=args.fm_pauli_pad,
            fm_pauli_terms=args.fm_pauli_terms, #must be genralized for qubit numbers
        )

        print("Using Pauli feature map with user-defined settings.")
        print(f"fm_pauli_reps: {args.fm_pauli_reps}")
        print(f"fm_pauli_entanglement: {args.fm_pauli_entanglement}")
        print(f"fm_pauli_pad: {args.fm_pauli_pad}")

    elif mapper_cfg["fm_kind"] == "eff_su2":
        mapper_cfg |= dict(
            fm_eff_alpha=1.0, # not sure what to sure, need to investigate
            fm_eff_ent_kind=args.fm_eff_ent_kind, 
            fm_eff_pad_mod=args.fm_eff_pad_mod, # 'wrap' | 'repeatlast' | 'pad'
            fm_eff_twoq_op=args.fm_eff_twoq_op, 
        )

        print("Using Efficient SU2 feature map with user-defined settings.")
        print(f"fm_eff_reps: {args.fm_eff_reps}")
        print(f"fm_eff_ent_kind: {args.fm_eff_ent_kind}")
        print(f"fm_eff_pad_mod: {args.fm_eff_pad_mod}")
        print(f"fm_eff_twoq_op: {args.fm_eff_twoq_op}")

    else:
        raise ValueError(f"Unknown mapper {mapper_cfg['fm_kind']}")

    cfg = QFCConfig(
        n_wires=args.n_wires,
        depth=args.depth,
        batch_size=args.batch_size,
        device=device,
        n_random_ops=args.random_ops,
        encoder_oplist_name="4x4_ryzxy",
        num_classes=num_classes,
        # Q-Layer params
        qlayer_ent_kind=args.qlayer_ent_kind, # default is "full"
        qlayer_ent_trainable=args.no_qlayer_ent_trainable, # default is True
        qlayer_ent_wire_reverse=args.qlayer_ent_wire_reverse, # default is False
        qlayer_twoq_op=args.qlayer_twoq_op, # default is "cx"
        
        # Measurement / Pooling params
        pool_pairs=False,
        pair_topology="ring",
        pool_hw=pool_hw_cfg,
        feature_dim=feature_dim,
        measure_ops=None,

        # Encoder / Feature Map params
        **mapper_cfg,
    )

    
    if cfg.pool_pairs:
        print(f"_measure_pooled_pairs selected")
    else:
        print(f"_measure_unpooled_pairs selected")
    
    
    

    # Target model
    if args.model_type == "qnn":
        model = QFCModel(cfg).to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    elif args.model_type == "mlp_qnn":
        model1 = QFCModel(cfg).to(device)
        model = build_classical_baseline(cfg, model1).to(device)

        # num_trainable_params1 = sum(
        #     p.numel() for p in model1.parameters() if p.requires_grad
        # )
        # print("Trainable parameters for QFC:", num_trainable_params1)

        # num_trainable_params = sum(
        #     p.numel() for p in model.parameters() if p.requires_grad
        # )
        # print("Trainable parameters:", num_trainable_params)
        # exit()
        optimizer = optim.Adam(model.parameters(), lr=5e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
            
    elif args.model_type == "hqnn":
        model = HybridQNN(cfg).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    
    elif args.model_type == "qcnn":
        model = QCNN(cfg).to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-2)

        # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        # optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
        
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        # print("QCNN model type is not yet implemented. Please choose 'qnn' or 'hqnn'.")
        # exit(0)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # if args.model_type != "mlp_qnn":
    #     export_full_with_symbolic_encoder(
    #         model,
    #         D=feature_dim,
    #         pad_mode="wrap",
    #         backend="mpl",
    #         save_path="full_symbolic.png",
    #     )
  
    # exit()
    # print header once
    # path = f"{args.model_type}_model_{args.dataset}_{args.batch_size}_{args.epochs}.pt"
    # suffix = [
    #     args.dataset,
    #     f"wrs{args.n_wires}",
    #     f"d{args.depth}",
    #     f"rand{args.random_ops}",
    #     f"tr{args.vector_train}",
    #     f"vv{args.vector_valid}",
    #     f"ts{args.vector_test}",
    #     f"bsz{args.batch_size}",
    #     f"ep{args.epochs}",
    #     f"noise{args.moons_noise}"
    # ]
    # path = f"{args.model_type}_{'_'.join(suffix)}.pt"
    
    mapper_suffix = [f"fm{args.fm_kind.lower()}"]
    if args.fm_kind.lower() == "z":
        mapper_suffix += [
            f"zPad-{args.fm_z_pad_mode}",
            f"zReps{args.fm_z_reps}",
        ]
    elif args.fm_kind.lower() == "zz":
        mapper_suffix += [
            f"zzPad-{args.fm_zz_pad_mode}",
            f"zzEnt-{args.fm_zz_entanglement}",
            f"zzReps{args.fm_zz_reps}",
        ]
    elif args.fm_kind.lower() == "pauli":
        mapper_suffix += [
            f"pPad-{args.fm_pauli_pad}",
            f"pEnt-{args.fm_pauli_entanglement}",
            f"pReps{args.fm_pauli_reps}",
        ]
    elif args.fm_kind.lower() == "eff_su2":
        mapper_suffix += [
            f"effPad-{args.fm_eff_pad_mod}",
            f"effEnt-{args.fm_eff_ent_kind}",
            f"eff2q-{args.fm_eff_twoq_op}",
            f"effReps{args.fm_eff_reps}",
        ]

    suffix = [
        args.dataset,
        f"nwr{args.n_wires}",
        f"d{args.depth}",
        f"qlEnt-{args.qlayer_ent_kind}",
        f"ql2q-{args.qlayer_twoq_op}",
        f"qlEntRev{int(args.qlayer_ent_wire_reverse)}",
        f"ranOp{args.random_ops}",
        f"tr{args.vector_train}",
        f"vv{args.vector_valid}",
        f"ts{args.vector_test}",
        f"bsz{args.batch_size}",
        f"ep{args.epochs}",
        f"noise{args.moons_noise}"
    ] + ["_"] + mapper_suffix
    path = f"id{args.run_id}_{args.model_type}_{'_'.join(suffix)}.pt"
    # print(f"Model save/load path: {path}")
    # exit()
    
    if args.target_model_path:
        # If the sweep script passed a path, use it directly
        save_target_path = args.target_model_path
    else:
        # Fallback to the old logic if needed, but not necessary for the sweep
        # (You can just raise an error if args.target_model_path is None)
        # For simplicity in testing, let's assume the sweep always passes the path.
        save_target_path = f"id{args.run_id}_fallback_model.pt"

    
    if args.model_type != "mlp_qnn":


        from pathlib import Path
        
        circuit_dir = Path("all_circ") 
        circuit_dir.mkdir(parents=True, exist_ok=True)
        path_circuit = circuit_dir / f"{args.model_type}_{'_'.join(suffix)}.png"

        export_full_with_symbolic_encoder(
            model,
            D=feature_dim,
            pad_mode="wrap",
            backend="mpl",
            save_path=path_circuit,
        )
  

 
    # ---------------------------
    # MIA: EXPORT ATTACK DATA MODE
    # ---------------------------

    if args.export_attack_data:
            # Use the freshly trained model instead of loading from disk
            model.eval()
            
            # Determine output path
            out_path = args.attack_data_out
            if out_path is None:
                # If no explicit path, derive from model path or run_id
                if args.target_model_path:
                    out_path = str(Path(args.target_model_path).with_suffix("")) + "_attack_data.pt"
                else:
                    out_path = f"attack_data_run{args.run_id}.pt"
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"args.export_attack_data: {args.export_attack_data}")

    if args.train_target:
        print(f"{'Epoch':>5} | {'Loss (train / val)':>21} | {'Acc (train / val)':>19}")
        print("-"*5 + " | " + "-"*21 + " | " + "-"*19)
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(dataflow, model, device, optimizer)
            va_loss, va_acc = evaluate(dataflow, "valid", model, device)
            loss_col = f"{tr_loss:.4f} / {va_loss:.4f}"
            acc_col  = f"{tr_acc:.3f} / {va_acc:.3f}"
            print(f"{epoch:5d} | {loss_col:>21} | {acc_col:>19}")
            scheduler.step()
        
        # Test evaluation
        te_loss, te_acc = evaluate(dataflow, "test", model, device)
        print(f"Test  | loss {te_loss:.4f} acc {te_acc:.3f}")
        
        # === GENERATE PVs IMMEDIATELY AFTER TRAINING ===
        if args.export_attack_data:
            print("\n[PV Generation] Starting...")
            model.eval()
            
            # Collect probabilities for train (members) and test (non-members)
            probs_tr, y_tr = collect_probs_and_labels(dataflow, "train", model, device)
            probs_te, y_te = collect_probs_and_labels(dataflow, "test", model, device)
            
            # Compute metrics
            tr_metrics = metrics_from_probs(probs_tr, y_tr)
            te_metrics = metrics_from_probs(probs_te, y_te)
            
            # Build attack features
            if args.attack_feature_mode == "pv":
                X_tr = probs_tr
                X_te = probs_te
                # Derive stats for logging
                pred_tr = probs_tr.argmax(dim=1)
                pred_te = probs_te.argmax(dim=1)
                correct_tr = (pred_tr == y_tr).long()
                correct_te = (pred_te == y_te).long()
                eps = 1e-12
                ptrue_tr = probs_tr[torch.arange(len(y_tr)), y_tr.long()].clamp_min(eps)
                ptrue_te = probs_te[torch.arange(len(y_te)), y_te.long()].clamp_min(eps)
                loss_tr = (-torch.log(ptrue_tr))
                loss_te = (-torch.log(ptrue_te))
                entropy_tr = (-(probs_tr.clamp_min(eps) * torch.log(probs_tr.clamp_min(eps))).sum(dim=1))
                entropy_te = (-(probs_te.clamp_min(eps) * torch.log(probs_te.clamp_min(eps))).sum(dim=1))
                conf_tr = probs_tr.max(dim=1).values
                conf_te = probs_te.max(dim=1).values
                if probs_tr.shape[1] >= 2:
                    top2_tr = probs_tr.topk(k=2, dim=1).values
                    top2_te = probs_te.topk(k=2, dim=1).values
                    margin_tr = (top2_tr[:, 0] - top2_tr[:, 1])
                    margin_te = (top2_te[:, 0] - top2_te[:, 1])
                else:
                    margin_tr = conf_tr.clone()
                    margin_te = conf_te.clone()
            else:
                X_tr, st_tr = build_attack_features(probs_tr, y_tr)
                X_te, st_te = build_attack_features(probs_te, y_te)
                pred_tr = st_tr["pred"]
                pred_te = st_te["pred"]
                correct_tr = st_tr["correct"].long()
                correct_te = st_te["correct"].long()
                loss_tr = st_tr["loss"]
                loss_te = st_te["loss"]
                entropy_tr = st_tr["entropy"]
                entropy_te = st_te["entropy"]
                conf_tr = st_tr["conf"]
                conf_te = st_te["conf"]
                margin_tr = st_tr["margin"]
                margin_te = st_te["margin"]
            
            # Membership labels
            mem_tr = torch.zeros(len(y_tr), dtype=torch.long)
            mem_te = torch.ones(len(y_te), dtype=torch.long)
            
            # Concatenate everything
            X = torch.cat([X_tr, X_te], dim=0).float()
            pv = torch.cat([probs_tr, probs_te], dim=0).float()
            y_true = torch.cat([y_tr, y_te], dim=0).long()
            y_pred = torch.cat([pred_tr, pred_te], dim=0).long()
            correct = torch.cat([correct_tr, correct_te], dim=0).long()
            membership = torch.cat([mem_tr, mem_te], dim=0).long()
            split = torch.cat([
                torch.zeros(len(y_tr), dtype=torch.long),
                torch.ones(len(y_te), dtype=torch.long),
            ], dim=0)
            
            # Build payload
            payload = {
                "X": X,
                "pv": pv,
                "pv_dim": int(pv.shape[1]),
                "y_true": y_true,
                "y_pred": y_pred,
                "correct": correct,
                "membership": membership,
                "split": split,
                "meta": {
                    "dataset": args.dataset,
                    "model_type": args.model_type,
                    "run_id": int(args.run_id),
                    "attack_feature_mode": args.attack_feature_mode,
                    "n_wires": int(args.n_wires),
                    "depth": int(args.depth),
                    "ql_ent": args.qlayer_ent_kind,
                    "ql_op": args.qlayer_twoq_op,
                    "vector_train": int(args.vector_train),
                    "vector_test": int(args.vector_test),
                },
                "stats": {
                    "loss": torch.cat([loss_tr, loss_te], dim=0).float(),
                    "entropy": torch.cat([entropy_tr, entropy_te], dim=0).float(),
                    "conf": torch.cat([conf_tr, conf_te], dim=0).float(),
                    "margin": torch.cat([margin_tr, margin_te], dim=0).float(),
                },
                "target_metrics": {
                    "train": tr_metrics,
                    "test": te_metrics,
                }
            }
            
            # Determine save path
            out_path = args.attack_data_out
            if out_path is None:
                if args.target_model_path:
                    out_path = str(Path(args.target_model_path).with_suffix("")) + "_attack_data.pt"
                else:
                    out_path = f"attack_data_run{args.run_id}.pt"
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(payload, out_path)
            print(f"[PV Generation] Saved: {out_path}")
            print(f"[PV Generation] Members={int((membership==0).sum())} NonMembers={int((membership==1).sum())}")
            print(f"[PV Generation] Train acc={tr_metrics['acc']:.4f} Test acc={te_metrics['acc']:.4f}")
        
        # Optionally save model
        if args.target_model_path:
            Path(save_target_path).parent.mkdir(parents=True, exist_ok=True)
            save_model(model, save_target_path)
            print(f"[Model] Saved: {save_target_path}")

    # else:
    #     model2 = load_model(model, save_target_path)
    #     te_loss, te_acc = evaluate(dataflow, "test", model2, device)
    #     print(f"Loaded: Test  | loss {te_loss:.4f} acc {te_acc:.3f}")
if __name__ == "__main__":
    main()
