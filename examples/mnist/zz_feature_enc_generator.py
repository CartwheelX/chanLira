import math, json
from typing import List, Dict, Any, Optional, Tuple
import torch

# ---------- pair builders ----------
def _pairs_linear(n: int): return [(i, i+1) for i in range(n-1)]
def _pairs_ring(n: int):   return _pairs_linear(n) + ([(n-1, 0)] if n > 2 else [])
def _pairs_full(n: int):   return [(i, j) for i in range(n) for j in range(i+1, n)]

_PAIR_TBL = {"linear": _pairs_linear, "ring": _pairs_ring, "full": _pairs_full}

# ---------- ZZ map oplist generator ----------
def generate_zz_feature_oplist(
    D: int,
    n_wires: int,
    *,
    entanglement: str = "ring",          # 'linear' | 'ring' | 'full'
    phi_mode: str = "prod",              # 'prod' | 'pi_minus'
    reps: Optional[int] = None,          # None => ceil(D / n_wires)
    pad_mode: str = "repeatlast",        # 'wrap' | 'repeatlast' | 'pad'
    name: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]], int, List[Tuple[int,int]]]:
    """
    Returns (name, func_list, D_needed, pairs).
    func_list matches TorchQuantum GeneralEncoder format.
    D is the length of your *raw* feature vector per sample.
    """
    assert n_wires >= 1 and D >= 1
    pad_mode = pad_mode.lower()
    entanglement = entanglement.lower()
    phi_mode = phi_mode.lower()
    assert pad_mode in ("wrap","repeatlast","pad")
    assert entanglement in _PAIR_TBL
    assert phi_mode in ("prod","pi_minus")

    pairs = _PAIR_TBL[entanglement](n_wires)
    n_pairs = len(pairs)

    if reps is None:
        reps = max(1, math.ceil(D / n_wires))
    D_needed = reps * (n_wires + n_pairs)  # augmented length expected by this oplist

    # Layout per repetition:
    #   block start = rep*(n_wires + n_pairs)
    #   indices [block .. block+n_wires-1]      -> RZ angles for wires 0..N-1
    #   indices [block+n_wires .. block+n_wires+n_pairs-1] -> RZZ angles for pairs in 'pairs' order
    func_list: List[Dict[str, Any]] = []

    for r in range(reps):
        block = r * (n_wires + n_pairs)

        # H on all qubits (no parameters)
        for w in range(n_wires):
            func_list.append({"input_idx": None, "func": "h", "wires": [w]})

        # RZ layer: one angle per wire -> uses indices [block .. block+n_wires-1]
        for w in range(n_wires):
            func_list.append({"input_idx": [block + w], "func": "rz", "wires": [w]})

        # ZZ layer: one angle per pair -> uses indices right after the RZ block
        for k, (a, b) in enumerate(pairs):
            func_list.append({"input_idx": [block + n_wires + k], "func": "rzz", "wires": [a, b]})

    if name is None:
        name = f"{n_wires}x{reps}_hrzz_{entanglement}_{phi_mode}"

    return name, func_list, D_needed, pairs

# ---------- feature augmentation to match the oplist ----------
@torch.no_grad()
def zz_augmented_features(
    x_raw: torch.Tensor,  # (B, D)
    n_wires: int,
    *,
    entanglement: str = "ring",
    phi_mode: str = "prod",
    alpha: float = 1.0,
    reps: Optional[int] = None,
    pad_mode: str = "repeatlast"
) -> torch.Tensor:
    """
    Build x_aug to feed GeneralEncoder built by generate_zz_feature_oplist.
    Per rep we emit: [2α·x_0..2α·x_{N-1}, 2α·φ_01, 2α·φ_12, ...] in the same order/pairs.
    """
    B, D = x_raw.shape
    entanglement = entanglement.lower()
    phi_mode = phi_mode.lower()
    pad_mode = pad_mode.lower()
    pairs = _PAIR_TBL[entanglement](n_wires)
    n_pairs = len(pairs)

    if reps is None:
        reps = max(1, math.ceil(D / n_wires))

    D_needed = reps * (n_wires + n_pairs)
    x_aug = x_raw.new_empty(B, D_needed)

    def map_idx(idx: int) -> Optional[int]:
        if idx < D:
            return idx
        if pad_mode == "wrap":
            return idx % max(1, D)
        if pad_mode == "repeatlast":
            return D - 1 if D > 0 else 0
        # 'pad' -> represent zero by returning None
        return None

    for r in range(reps):
        # gather this rep's N wire features (raw)
        raw_idx_start = r * n_wires
        xs = []
        for w in range(n_wires):
            src = map_idx(raw_idx_start + w)
            if src is None:
                xs.append(torch.zeros(B, device=x_raw.device, dtype=x_raw.dtype))
            else:
                xs.append(x_raw[:, src])
        # unary angles: 2*alpha * x
        z_angles = [2.0 * alpha * t for t in xs]
        # pairwise φ:
        if phi_mode == "prod":
            phis = [xs[a] * xs[b] for (a, b) in pairs]
        else:  # 'pi_minus'
            import math as _m
            pis = [_m.pi - t for t in xs]  # elementwise; broadcasting works with tensors
            phis = [pis[a] * pis[b] for (a, b) in pairs]
        zz_angles = [2.0 * alpha * p for p in phis]

        block = r * (n_wires + n_pairs)
        # write into augmented vector in the same order as the generator
        x_aug[:, block : block + n_wires] = torch.stack(z_angles, dim=1)
        if n_pairs > 0:
            x_aug[:, block + n_wires : block + n_wires + n_pairs] = torch.stack(zz_angles, dim=1)

    return x_aug

# ---------- pretty writer (same output shape you want) ----------
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
