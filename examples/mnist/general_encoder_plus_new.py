# # # general_encoder_plus.py
# # import math
# # from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

# # import torch
# # import torch.nn as nn
# # import torchquantum as tq


# # Number = Union[int, float]
# # IdxLike = Optional[Union[int, Sequence[int]]]

# # # Map friendly gate names in op-lists to TorchQuantum gate modules.
# # # Notes:
# # # - 'u1' and 'p' are both implemented with RZ (global phase ignored, as in Qiskit).
# # # - 'cnot' and 'cx' are aliases.
# # # - 'u3' supports fixed constants (basis-change blocks), or a 3-tuple constant.
# # FUNC_TABLE: Dict[str, Optional[type]] = {
# #     "h": tq.hadamard,
# #     "s": tq.S,
# #     "sdg": tq.SDG,
# #     "id": None,               # no-op
# #     "rx": tq.RX,
# #     "ry": tq.RY,
# #     "rz": tq.RZ,
# #     "u1": tq.RZ,              # U1(λ) == RZ(λ) up to global phase
# #     "p": tq.RZ,               # Qiskit P(λ) == U1(λ) == RZ(λ) (phase ignored)
# #     "u3": tq.U3,
# #     "cx": tq.CNOT,
# #     "cnot": tq.CNOT,
# #     "cz": getattr(tq, "CZ", None),  # some TQ versions have CZ, some not
# #     # add more two-qubit ops as needed (e.g., tq.RXX, tq.RZZ, …)
# # }


# # def _to_list(x: Union[int, Sequence[int]]) -> List[int]:
# #     return [int(x)] if isinstance(x, int) else [int(i) for i in x]


# # def _broadcast_const(B: int, val: Number, device, dtype) -> torch.Tensor:
# #     """Create a (B,) tensor filled with a constant."""
# #     return torch.full((B,), float(val), device=device, dtype=dtype)


# # def _combine_inputs(vals: List[torch.Tensor], how: str = "sum") -> torch.Tensor:
# #     """Combine a list of (B,) tensors into a single (B,) according to `how`."""
# #     if len(vals) == 0:
# #         raise ValueError("No values to combine.")
# #     if len(vals) == 1:
# #         return vals[0]
# #     if how == "sum":
# #         out = vals[0]
# #         for v in vals[1:]:
# #             out = out + v
# #         return out
# #     elif how == "prod":
# #         out = vals[0]
# #         for v in vals[1:]:
# #             out = out * v
# #         return out
# #     elif how == "mean":
# #         out = vals[0]
# #         for v in vals[1:]:
# #             out = out + v
# #         return out / len(vals)
# #     else:
# #         raise ValueError(f"Unknown combine rule '{how}'. Use 'sum'|'prod'|'mean'.")


# # class GeneralEncoderPlus_new(tq.QuantumModule):
# #     """
# #     Execute a list of gate-ops on a TorchQuantum device, driven by classical features.

# #     Each op is a dict with:
# #       - 'func': str                  # gate name, e.g. 'h','u1','rz','cx','u3', …
# #       - 'wires': List[int]           # target wire(s)
# #       - 'input_idx': None|int|[int,…]# which feature(s) to use (None => constant-only gate)
# #       - 'params': optional           # constant angle(s). For 'u3': [theta, phi, lam]
# #       - 'combine': optional[str]     # how to combine multiple features: 'sum'|'prod'|'mean' (default 'sum')
# #       - 'scale': optional[float]     # per-op scale multiplier (applied after combine)

# #     Class-wide options:
# #       - pad_mode: 'zero'|'wrap'|'repeatlast'  (for out-of-range feature indices)
# #       - alpha: global scale applied to all feature-derived angles (not to 'params' constants)
# #       - angle_map: optional callable(List[torch.Tensor]) -> torch.Tensor
# #                    final transform *after* combine/scale (e.g. lambda xs: 2*xs[0]
# #                    or for pair: lambda xs: 2*(xs[0]-math.pi)*(xs[1]-math.pi)).
# #                    If provided, it receives the list of raw per-input tensors (before combine).

# #     Example op:
# #       {"input_idx":[i,j], "func":"u1", "wires":[t], "combine":"prod", "scale":2.0}
# #       -> theta = 2.0 * alpha * (x[:,i] * x[:,j])  (then apply RZ(theta) on wire t)

# #     For pure constants (basis changes), use:
# #       {"input_idx":None, "func":"u3", "wires":[q], "params":[pi/2, 0, pi]}
# #     """

# #     def __init__(
# #         self,
# #         oplist: List[Dict],
# #         *,
# #         pad_mode: str = "wrap",
# #         alpha: float = 1.0,
# #         angle_map: Optional[Callable[[List[torch.Tensor]], torch.Tensor]] = None,
# #     ):
# #         super().__init__()
# #         assert pad_mode in ("zero", "wrap", "repeatlast")
# #         self.oplist = oplist
# #         self.pad_mode = pad_mode
# #         self.alpha = float(alpha)
# #         self.angle_map = angle_map

# #         # cache of gate modules to avoid re-instantiation overhead
# #         self._gate_cache: Dict[str, nn.Module] = {}

# #     # ---------- helpers for feature -> angle ----------

# #     def _pad_fetch(self, x: torch.Tensor, idx: int) -> torch.Tensor:
# #         """Return (B,) feature with padding policy."""
# #         B, D = x.shape
# #         if D == 0:
# #             # degenerate, just zeros
# #             return x.new_zeros(B)
# #         if 0 <= idx < D:
# #             return x[:, idx]
# #         if self.pad_mode == "zero":
# #             return x.new_zeros(B)
# #         elif self.pad_mode == "wrap":
# #             return x[:, idx % D]
# #         else:  # 'repeatlast'
# #             return x[:, D - 1]

# #     def _gather_inputs(
# #         self, x: torch.Tensor, idxs: Sequence[int]
# #     ) -> List[torch.Tensor]:
# #         """Collect list of (B,) tensors for the requested indices with padding."""
# #         return [self._pad_fetch(x, i) for i in idxs]

# #     def _get_gate(self, name: str) -> Optional[nn.Module]:
# #         name = name.lower()
# #         if name not in FUNC_TABLE:
# #             raise ValueError(f"Unknown gate '{name}' in op-list.")
# #         ctor = FUNC_TABLE[name]
# #         if ctor is None:
# #             return None  # e.g., 'id'
# #         if name not in self._gate_cache:
# #             self._gate_cache[name] = ctor()
# #         return self._gate_cache[name]

# #     # ---------- forward ----------

# #     @tq.static_support  # ok to keep; we don't pass 'static' to gates that don't accept it
# #     def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
# #         """
# #         x: (B, D) features. Angles derived per-op as described in the class docstring.
# #         """
# #         assert x.dim() == 2, "x must be (B, D)"
# #         B, D = x.shape
# #         device, dtype = x.device, x.dtype

# #         for op in self.oplist:
# #             gname: str = op["func"].lower()
# #             wires: List[int] = _to_list(op["wires"])
# #             input_idx: IdxLike = op.get("input_idx", None)

# #             gate = self._get_gate(gname)
# #             if gate is None:
# #                 # explicit no-op
# #                 continue

# #             # -------- constants-only gates (basis changes etc.) --------
# #             if input_idx is None:
# #                 const = op.get("params", None)
# #                 if const is None:
# #                     # no params required (e.g., H, S, SDG, CX, CZ)
# #                     gate(qdev, wires=wires)
# #                 else:
# #                     # with constants; support 'u3' 3-params, or 1-param gates
# #                     if gname == "u3":
# #                         if not (isinstance(const, (list, tuple)) and len(const) == 3):
# #                             raise ValueError("u3 needs params [theta, phi, lam].")
# #                         th = _broadcast_const(B, const[0], device, dtype)
# #                         ph = _broadcast_const(B, const[1], device, dtype)
# #                         la = _broadcast_const(B, const[2], device, dtype)
# #                         params = torch.stack([th, ph, la], dim=1)  # (B, 3)
# #                         gate(qdev, wires=wires, params=params)
# #                     else:
# #                         if isinstance(const, (list, tuple)):
# #                             if len(const) != 1:
# #                                 raise ValueError(
# #                                     f"{gname} expects a single constant or omit 'params'."
# #                                 )
# #                             const = const[0]
# #                         theta = _broadcast_const(B, const, device, dtype)
# #                         gate(qdev, wires=wires, params=theta)
# #                 continue

# #             # -------- feature-driven parameter(s) --------
# #             # normalize indices to list[int]
# #             idxs = _to_list(input_idx)

# #             # fetch raw per-index tensors (B,)
# #             raw_vals = self._gather_inputs(x, idxs)

# #             # optional custom angle_map (e.g., Qiskit-like φ)
# #             if self.angle_map is not None:
# #                 theta = self.angle_map(raw_vals)  # must return (B,)
# #                 if theta.dim() != 1 or theta.shape[0] != B:
# #                     raise ValueError("angle_map must return shape (B,).")
# #             else:
# #                 # default combine rule
# #                 how = op.get("combine", "sum")
# #                 theta = _combine_inputs(raw_vals, how=how)

# #             # per-op scale then global alpha
# #             local_scale = float(op.get("scale", 1.0))
# #             theta = local_scale * self.alpha * theta  # (B,)

# #             # route parameters by gate type
# #             if gname in ("rz", "u1", "p", "rx", "ry"):
# #                 gate(qdev, wires=wires, params=theta)
# #             elif gname == "u3":
# #                 # allow (rare) feature-driven U3 if 'params' absent and 3 inputs are given
# #                 if len(raw_vals) != 3:
# #                     raise ValueError("Feature-driven u3 needs 3 input indices.")
# #                 th = raw_vals[0] * local_scale * self.alpha
# #                 ph = raw_vals[1] * local_scale * self.alpha
# #                 la = raw_vals[2] * local_scale * self.alpha
# #                 params = torch.stack([th, ph, la], dim=1)  # (B, 3)
# #                 gate(qdev, wires=wires, params=params)
# #             else:
# #                 # two-qubit gates like CX/CZ have no params
# #                 gate(qdev, wires=wires)




# # oplist_encoder.py
# import torch
# import torchquantum as tq

# class OplistEncoder(tq.QuantumModule):
#     """
#     Runs the op-list produced by the builder.
#     Each op:
#       {
#         "input_idx": None | [i, ...],   # None => constant gate; list => data-driving indices
#         "func": "u1"|"u3"|"cx"|...,
#         "wires": [int,...],
#         # optional:
#         "params": [..],                 # numeric constants for constant gates
#         "scale": float,                 # extra multiplicative (e.g., 2.0 from Qiskit)
#       }
#     """
#     def __init__(self, oplist, *, alpha: float = 1.0, multi_index_rule: str = "prod"):
#         super().__init__()
#         assert multi_index_rule in ("prod", "sum")
#         self.oplist = oplist
#         self.alpha = float(alpha)
#         self.multi_index_rule = multi_index_rule

#     # ---- map textual gate names to TorchQuantum gate classes ----
#     @staticmethod
#     def _get_gate_class(func_name: str):
#         f = func_name.lower()
#         if f == "u3":
#             return tq.U3
#         if f in ("u1", "p", "phase", "rz"):
#             return getattr(tq, "U1", tq.RZ)  # if U1 missing, fall back to RZ
#         if f in ("cx", "cnot"):
#             return getattr(tq, "CNOT", getattr(tq, "CX", None))
#         if f == "s":
#             return tq.S
#         if f in ("sdg", "sdag"):
#             return getattr(tq, "SDG", getattr(tq, "Sdg", None))
#         if f == "h":
#             return tq.H
#         # try generic fallback (e.g., 'rx','ry','rz','rzz'…)
#         cand = getattr(tq, f.upper(), None)
#         if cand is None:
#             raise ValueError(f"Unsupported gate '{func_name}' for TorchQuantum mapping.")
#         return cand

#     # ---- tensor helpers ----
#     @staticmethod
#     def _broadcast_scalar_to_batch(value, B, dtype, device):
#         return torch.tensor(value, dtype=dtype, device=device).expand(B)

#     @staticmethod
#     def _broadcast_vec_to_batch(vec3, B, dtype, device):
#         t = torch.tensor(vec3, dtype=dtype, device=device).view(1, -1)
#         return t.expand(B, t.shape[1])

#     def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
#         """
#         x: (B, D)
#         For multi-index entries, combine features by product (default) or sum.
#         Effective angle = alpha * scale * combine(features).
#         """
#         B, D = x.shape
#         dtype, device = x.dtype, x.device

#         for op in self.oplist:
#             func = op["func"].lower()
#             wires = op["wires"]
#             Gate = self._get_gate_class(func)

#             if op.get("input_idx") is None:
#                 # constant gate
#                 if "params" in op and op["params"] is not None:
#                     p = op["params"]
#                     if func == "u3":
#                         params = self._broadcast_vec_to_batch(p, B, dtype, device)
#                     else:
#                         val = p[0] if isinstance(p, (list, tuple)) else p
#                         params = self._broadcast_scalar_to_batch(val, B, dtype, device)
#                     Gate()(qdev, wires=wires, params=params)
#                 else:
#                     Gate()(qdev, wires=wires)
#                 continue

#             # data-driven phase
#             idxs = op["input_idx"]
#             scale = float(op.get("scale", 1.0))
#             if len(idxs) == 1:
#                 base = x[:, idxs[0]]
#             else:
#                 feats = x[:, idxs]
#                 base = feats.sum(dim=1) if self.multi_index_rule == "sum" else feats.prod(dim=1)
#             theta = self.alpha * scale * base
#             Gate()(qdev, wires=wires, params=theta)

# def GeneralEncoderPlus_new(oplist, *, alpha: float = 1.0, multi_index_rule: str = "prod") -> OplistEncoder:
#     """Build the encoder directly from a raw op-list."""
#     return OplistEncoder(oplist, alpha=alpha, multi_index_rule=multi_index_rule)


# oplist_encoder.py
import torch
import torchquantum as tq

class OplistEncoder(tq.QuantumModule):
    def __init__(self, oplist, *, alpha: float = 1.0, multi_index_rule: str = "prod"):
        super().__init__()
        assert multi_index_rule in ("prod", "sum")
        self.oplist = oplist
        self.alpha = float(alpha)
        self.multi_index_rule = multi_index_rule

    @staticmethod
    def _get_gate_class(func_name: str):
        f = func_name.lower()
        if f == "u3":
            return tq.U3
        if f in ("u1", "p", "phase", "rz"):
            return getattr(tq, "U1", tq.RZ)  # fall back to RZ if U1 not in TorchQuantum
        if f in ("cx", "cnot"):
            return getattr(tq, "CNOT", getattr(tq, "CX", None))
        if f == "s":
            return tq.S
        if f in ("sdg", "sdag"):
            return getattr(tq, "SDG", getattr(tq, "Sdg", None))
        if f == "h":
            return tq.H
        cand = getattr(tq, f.upper(), None)
        if cand is None:
            raise ValueError(f"Unsupported gate '{func_name}' for TorchQuantum mapping.")
        return cand

    @staticmethod
    def _broadcast_scalar_to_batch(value, B, dtype, device):
        return torch.tensor(float(value), dtype=dtype, device=device).expand(B)

    @staticmethod
    def _broadcast_vec_to_batch(vec3, B, dtype, device):
        # vec3 may be list/tuple/torch tensor; coerce to list of 3 floats
        v = [float(vec3[0]), float(vec3[1]), float(vec3[2])]
        t = torch.tensor(v, dtype=dtype, device=device).view(1, 3)
        return t.expand(B, 3)

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
        """
        x: (B, D)
        For multi-index entries, combine features by product (default) or sum.
        Effective angle = alpha * scale * combine(features).
        """
        B, D = x.shape
        dtype, device = x.dtype, x.device

        for op in self.oplist:
            func = op["func"].lower()
            wires = op["wires"]
            Gate = self._get_gate_class(func)

            # ----- constant gate (no input_idx) -----
            if op.get("input_idx") is None:
                if "params" in op and op["params"] is not None:
                    p = op["params"]
                    if func == "u3":
                        params = self._broadcast_vec_to_batch(p, B, dtype, device)
                    else:
                        # p can be [val] or val
                        val = p[0] if isinstance(p, (list, tuple)) else p
                        params = self._broadcast_scalar_to_batch(val, B, dtype, device)
                    Gate()(qdev, wires=wires, params=params)
                else:
                    Gate()(qdev, wires=wires)
                continue

            # ----- data-driven phase -----
            idxs = op["input_idx"]
            # print( "idxs:", idxs)
            scale = float(op.get("scale", 1.0))
            # scale = 1.0

            # print(f"scale is {scale}")
            # print(f"comnine rule is {self.multi_index_rule} ")
            # exit()

            combine_rule = op.get("combine", self.multi_index_rule)

            if len(idxs) == 1:
                base = x[:, idxs[0]]
            else:
                feats = x[:, idxs]
                base = feats.sum(dim=1) if combine_rule == "sum" else feats.prod(dim=1)
                # print( "feats:", feats)

            theta = self.alpha * scale * base
            # print(f"theta is {theta}")
            Gate()(qdev, wires=wires, params=theta)

            # combine_rule = op.get("combine", self.multi_index_rule)
        # print(f"combine_rule final is {self.multi_index_rule} ")
        # exit()

def GeneralEncoderPlus_new(oplist, *, alpha: float = 1.0, multi_index_rule: str = "prod") -> OplistEncoder:
    """Build the encoder directly from a raw op-list."""
    print(f"multi_index_rule in general encoder plus new is {multi_index_rule} ")
    # exit()
    return OplistEncoder(oplist, alpha=alpha, multi_index_rule=multi_index_rule)
