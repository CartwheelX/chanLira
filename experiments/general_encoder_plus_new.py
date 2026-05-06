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
