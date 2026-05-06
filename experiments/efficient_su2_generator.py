import math
from typing import List, Tuple, Sequence, Optional, Dict, Any

# ---------- entangler pairs ----------
def _pairs_linear(n: int):   return [(i, i+1) for i in range(n-1)]
def _pairs_ring(n: int):     return [(i, (i+1) % n) for i in range(n)]
def _pairs_full(n: int):     return [(i, j) for i in range(n) for j in range(i+1, n)]
def _pairs_pairwise(n: int): return [(i, i+1) for i in range(0, n-1, 2)] + ([(n-1, 0)] if n % 2 else [])

_ENT_TBL = {
    "linear":   _pairs_linear,
    "ring":     _pairs_ring,
    "circular": _pairs_ring,   # NEW alias
    "full":     _pairs_full,
    "pairwise": _pairs_pairwise,
}

# Map to TorchQuantum func names used by GeneralEncoder
_TW0Q_ALIAS = {"cx": "cnot", "cnot": "cnot", "cz": "cz", "swap": "swap", "rxx": "rxx", "ryy": "ryy", "rzz": "rzz"}

def generate_efficient_su2_oplist(
    D: int,
    n_wires: int,
    single_ops: Sequence[str] = ("ry", "rz"),
    entanglement: str = "linear",          # 'linear'|'ring'|'circular'|'full'|'pairwise'
    twoq_op: str = "cx",
    pad_mode: str = "repeatlast",          # 'wrap' | 'repeatlast' | 'pad'
    name: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]], int]:
    """
    Emit a GeneralEncoder func_list for an Efficient-SU2-style feature map.
    If pad_mode is 'wrap' or 'repeatlast', all input_idx are in [0, D-1].
    If 'pad', sequential indices continue past D; caller should zero-pad features.
    """
    assert n_wires >= 1 and D >= 1
    single_ops = tuple(op.lower() for op in single_ops)
    entanglement = entanglement.lower()
    if entanglement not in _ENT_TBL:
        raise ValueError(f"Unknown entanglement '{entanglement}'")
    pairs = _ENT_TBL[entanglement](n_wires)
    twq = _TW0Q_ALIAS.get(twoq_op.lower(), twoq_op.lower())

    per_rep = n_wires * len(single_ops)
    reps = max(1, math.ceil(D / per_rep))
    D_needed = reps * per_rep

    def _map_idx(idx: int) -> int:
        if idx < D: return idx
        if pad_mode == "wrap":        return idx % D
        if pad_mode == "repeatlast":  return D - 1
        # 'pad': leave past-D indices as-is (caller pads features later)
        return idx

    func_list: List[Dict[str, Any]] = []
    idx = 0
    for _ in range(reps):
        # data-bound 1q layers
        for op in single_ops:
            for w in range(n_wires):
                src = _map_idx(idx)
                func_list.append({"input_idx": [src], "func": op, "wires": [w]})
                idx += 1
        # fixed entangler (data independent)
        for a, b in pairs:
            func_list.append({"input_idx": None, "func": twq, "wires": [a, b]})

    if name is None:
        name = f"{n_wires}x{reps}_{''.join(single_ops)}_{entanglement}_{twq}"
    return name, func_list, D_needed

# ---------- pretty writer ----------
def _fmt_item(item: dict) -> str:
    ii = "None" if item["input_idx"] is None else str(list(item["input_idx"]))
    return f'{{"input_idx": {ii}, "func": "{item["func"]}", "wires": {list(item["wires"])}}}'

def save_oplist_py(path: str, name: str, func_list: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("encoder_op_list_name_dict = {\n")
        f.write(f'  "{name}": [\n')
        for i, it in enumerate(func_list):
            comma = "," if i < len(func_list) - 1 else ""
            f.write(f"    {_fmt_item(it)}{comma}\n")
        f.write("  ]\n}\n")
