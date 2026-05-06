import math, os, json
from typing import List, Dict, Sequence, Optional

def _idx_map(k: int, D: int, pad: str) -> int:
    if k < D:
        return k
    if pad == "wrap":
        return k % D
    # 'repeatlast'
    return max(0, D - 1)

def generate_encoder_oplist(
    D: int,
    n_wires: int,
    single_ops: Sequence[str] = ("ry", "rz"),
    *,
    reps: Optional[int] = None,          # None => auto to cover all D
    order: str = "op-then-wire",         # or "wire-then-op"
    pad: str = "repeatlast",             # 'repeatlast' | 'wrap'
) -> List[Dict]:
    """
    Build a TorchQuantum GeneralEncoder func_list, e.g.:
      [{"input_idx":[i], "func":"ry", "wires":[q]}, ...]

    Each 1q op consumes 1 feature. If D isn't divisible by n_wires*len(single_ops),
    we keep mapping indices with 'wrap' or 'repeatlast' (no zero angles).
    """
    assert n_wires >= 1
    assert single_ops and all(isinstance(op, str) for op in single_ops)
    assert pad in ("repeatlast", "wrap")

    if D <= 0:
        return []

    per_rep = n_wires * len(single_ops)
    R = reps if isinstance(reps, int) and reps > 0 else max(1, math.ceil(D / per_rep))

    func_list: List[Dict] = []
    src = 0
    for _ in range(R):
        if order == "op-then-wire":
            for op in single_ops:
                for q in range(n_wires):
                    func_list.append({"input_idx": [_idx_map(src, D, pad)], "func": op, "wires": [q]})
                    src += 1
        elif order == "wire-then-op":
            for q in range(n_wires):
                for op in single_ops:
                    func_list.append({"input_idx": [_idx_map(src, D, pad)], "func": op, "wires": [q]})
                    src += 1
        else:
            raise ValueError(f"Unknown order '{order}'")
    return func_list

def make_name(n_wires: int, single_ops: Sequence[str], reps: int) -> str:
    """
    Your requested convention: 'wires x repeat_opsPerWire _ concatenatedOps'
    repeat_opsPerWire = reps * len(single_ops)
    Example: n_wires=4, single_ops=('ry','rz'), reps=3 -> '4x3_ryrz'
    """
    return f"{n_wires}x{reps * max(1, len(single_ops))}_{''.join(single_ops)}"
def _fmt_item(item: dict) -> str:
    # Keep key order: input_idx, func, wires
    ii = item["input_idx"]
    if not isinstance(ii, list):  # be robust if someone passed an int
        ii = [ii]
    w = item["wires"]
    if not isinstance(w, list):
        w = [w]
    return (
        '{'
        f'"input_idx": {ii}, '
        f'"func": "{item["func"]}", '
        f'"wires": {w}'
        '}'
    )

def _render_oplist_block(name: str, func_list: list[dict]) -> str:
    lines = []
    lines.append("encoder_op_list_name_dict = {")
    lines.append(f'  "{name}": [')
    for i, it in enumerate(func_list):
        comma = "," if i < len(func_list) - 1 else ""
        lines.append(f"    {_fmt_item(it)}{comma}")
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines) + "\n"

def save_oplist_py(path: str, name: str, func_list: list[dict], *, append: bool = False) -> None:
    """
    Write a .py file where each entry is one line:
      {"input_idx": [k], "func": "ry", "wires": [q]},
    If append=True, we read any existing dict and rewrite the whole file in the same style.
    """
    import importlib.util, types, os, tempfile

    if not append or not os.path.exists(path):
        txt = _render_oplist_block(name, func_list)
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)
        return

    # Append mode: load existing dict, add new key, rewrite nicely
    spec = importlib.util.spec_from_file_location("genops", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    d: dict = getattr(mod, "encoder_op_list_name_dict", {})
    if name in d:
        raise RuntimeError(f"Key '{name}' already exists in {path}.")
    d[name] = func_list

    # Rewrite whole file with our one-line style
    lines = ["encoder_op_list_name_dict = {"]

    keys_sorted = list(d.keys())
    for ki, k in enumerate(keys_sorted):
        fl = d[k]
        lines.append(f'  "{k}": [')
        for i, it in enumerate(fl):
            comma = "," if i < len(fl) - 1 else ""
            lines.append(f"    {_fmt_item(it)}{comma}")
        lines.append("  ]" + ("," if ki < len(keys_sorted) - 1 else ""))
    lines.append("}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")