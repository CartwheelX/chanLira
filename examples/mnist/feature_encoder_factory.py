# License: MIT (TorchQuantum & this file)

import argparse
import random
from dataclasses import dataclass
from typing import Iterable, Sequence
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from examples.mnist.mnist_2qubit_4class import QFCConfig, make_entangler
import torchquantum as tq
from torchquantum.measurement import expval_joint_analytical
from torchquantum.dataset import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR


def _pairs_disjoint(n): return [(i, i+1) for i in range(0, n-1, 2)] + ([(n-1,0)] if n%2 else [])
def _pairs_ring(n):     return [(i, (i+1) % n) for i in range(n)]
def _pairs_full(n):     return [(i, j) for i in range(n) for j in range(i+1, n)]


# -------- Z Feature Map --------
class ZFeatureEncoder(tq.QuantumModule):
    def __init__(self, n_wires: int, reps: int = 1, alpha: float = 1.0):
        super().__init__()
        self.n_wires, self.reps, self.alpha = n_wires, int(reps), float(alpha)
        self.h, self.rz = tq.H(), tq.RZ()

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
        self.h, self.rz, self.rzz = tq.H(), tq.RZ(), tq.RZZ()
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
        self.h, self.sdg, self.rz, self.rzz = tq.H(), tq.SDG(), tq.RZ(), tq.RZZ()

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
class EfficientSU2FeatureEncoder(tq.QuantumModule):
    def __init__(self, n_wires: int, reps: int = 1,
                 single_ops: Sequence[str] = ("ry","rz"),
                 alpha: float = np.pi,
                 ent_kind: str = "linear", twoq_op: str = "cx"):
        super().__init__()
        self.n_wires, self.reps = n_wires, int(reps)
        self.alpha = float(alpha)
        self.single_ops = tuple(single_ops)
        self.entangler = make_entangler(ent_kind, n_wires, two_qubit_op=twoq_op,
                                        trainable=False, wire_reverse=False)
        self._op_tbl = {"rx": tq.RX(), "ry": tq.RY(), "rz": tq.RZ()}

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
        xw = x[:, :self.n_wires]
        for _ in range(self.reps):
            for op in self.single_ops:
                gate = self._op_tbl[op.lower()]
                for w in range(self.n_wires):
                    gate(qdev, wires=w, params=self.alpha * xw[:, w])
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
        return EfficientSU2FeatureEncoder(n_wires, reps=cfg.fm_eff_reps,
                                          single_ops=cfg.fm_eff_single_ops,
                                          alpha=cfg.fm_eff_alpha,
                                          ent_kind=cfg.fm_eff_ent_kind,
                                          twoq_op=cfg.fm_eff_twoq_op)
    raise ValueError(f"Unknown fm_kind '{cfg.fm_kind}'")