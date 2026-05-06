import torchquantum as tq
import torch.nn as nn

from torchquantum.layer.entanglement import EntanglementLayer

# ---- entangler factory -------------------------------------------------------
def _op_from_name(name: str):
    name = name.lower()
    table = {
        "cx": tq.CNOT, "cnot": tq.CNOT, "cz": tq.CZ, "swap": tq.SWAP,
        "crx": tq.CRX, "cry": tq.CRY, "crz": tq.CRZ,
        "rxx": tq.RXX, "ryy": tq.RYY, "rzz": tq.RZZ, "rzx": tq.RZX,
    }
    if name not in table:
        raise ValueError(f"Unknown two-qubit op '{name}'")
    return table[name]

def make_entangler(kind: str, n_wires: int, trainable: bool=False, wire_reverse: bool = False, two_qubit_op: str = "cx"):
    """
    kind: 'full' | 'circular' | 'butterfly' | 'pairwise'
    two_qubit_op: one of {'cx','cz','swap','crx','cry','crz','rxx','ryy','rzz','rzx'}
    """
    op_cls = _op_from_name(two_qubit_op)
    op_has_params = getattr(op_cls, "num_params", 0) > 0
    

    


    # entanglement_to_class = {
    #         "full": EntangleFull,
    #         "linear": EntangleLinear,
    #         "pairwise": EntanglePairwise,
    #         "circular": EntangleCircular,
    #     }

    return EntanglementLayer(op=op_cls, n_wires=n_wires, kind=kind, has_params=op_has_params, trainable=trainable, wire_reverse=wire_reverse)
    # # Prefer the dedicated entanglement module if available; otherwise use stable aliases.
    # if kind == "full":
    #     from torchquantum.layer.entanglement import EntangleFull
    #     return EntangleFull(op=op_cls, n_wires=n_wires, has_params=op_has_params)
    
 
    # if kind == "circular":
    #     # nearest-neighbor ring
    #     from torchquantum.layer.entanglement import EntangleCircular
    #     return EntangleCircular(op=op_cls, n_wires=n_wires, has_params=op_has_params)

    # if kind == "butterfly":
    #     from torchquantum.layer.entanglement.op2_layer import Op2QButterflyLayer
    #     return Op2QButterflyLayer(op=op_cls, n_wires=n_wires, has_params=op_has_params)

    # if kind == "pairwise":
    #     try:
    #         from torchquantum.layer.entanglement import EntanglePairwise
    #         return EntanglePairwise(op=op_cls, n_wires=n_wires, has_params=op_has_params)
    #     except Exception:
    #         # Fallback: (0,1), (2,3), ...
    #         class PairwiseFallback(tq.QuantumModule):
    #             def __init__(self):
    #                 super().__init__()
    #                 self.ops = nn.ModuleList()
    #                 for a in range(0, n_wires - 1, 2):
    #                     self.ops.append(op_cls(has_params=op_has_params, trainable=op_has_params))
    #                     # store the pair alongside the op
    #                     self.ops[-1].__pair__ = [a, a + 1]
    #             @tq.static_support
    #             def forward(self, qdev: tq.QuantumDevice):
    #                 for op in self.ops:
    #                     op(qdev, wires=op.__pair__)
    #         return PairwiseFallback()

    # raise ValueError(f"Unknown entanglement kind '{kind}'")