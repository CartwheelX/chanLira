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
