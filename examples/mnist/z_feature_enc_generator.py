import math
from typing import List, Dict, Any, Optional, Tuple

def generate_z_feature_oplist(
    D: int,
    n_wires: int,
    *,
    pad_mode: str = "repeatlast",         # 'wrap' | 'repeatlast' | 'pad'
    name: Optional[str] = None            # None => f"{n_wires}x{reps}_hrz"
) -> Tuple[str, List[Dict[str, Any]], int]:
    """
    Build a GeneralEncoder func_list for a Z feature map:
      per rep: H on all wires, then RZ on all wires (one feature per wire).

    Returns: (name, func_list, D_needed)
      - name: suggested dict key, e.g. "4x3_hrz"
      - func_list: list of {"input_idx": [...]/None, "func": "...", "wires": [...]}
      - D_needed: reps * n_wires  (useful if you choose pad_mode='pad')
    """
    assert n_wires >= 1 and D >= 1
    pad_mode = pad_mode.lower()
    assert pad_mode in ("wrap", "repeatlast", "pad")

    # reps default
    
    reps = max(1, math.ceil(D / n_wires))
    D_needed = reps * n_wires

    def _map_idx(idx: int) -> int:
        if idx < D:
            return idx
        if pad_mode == "wrap":
            return idx % D
        if pad_mode == "repeatlast":
            return D - 1
        # 'pad': leave as is; caller pads features to D_needed with zeros
        return idx

    func_list: List[Dict[str, Any]] = []
    idx = 0
    for _ in range(reps):
        # H layer (no parameters)
        for w in range(n_wires):
            func_list.append({"input_idx": None, "func": "h", "wires": [w]})
        # RZ layer (data-bound, one feature per wire)
        for w in range(n_wires):
            src = _map_idx(idx)
            func_list.append({"input_idx": [src], "func": "rz", "wires": [w]})
            idx += 1

    if name is None:
        name = f"{n_wires}x{reps}_hrz"
    return name, func_list, D_needed

# ---------- pretty writer (same format you asked for) ----------
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
