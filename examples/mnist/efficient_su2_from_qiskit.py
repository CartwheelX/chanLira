# efficient_su2_oplist.py
from typing import List, Dict, Any, Optional, Sequence
import math

# ---------- entangler edges ----------
def _entangler_edges(n: int, kind: Optional[str]) -> List[List[int]]:
    if not kind or str(kind).lower() in ("none", "null"):
        return []
    k = str(kind).lower()
    if k in ("linear", "line"):
        return [[i, i+1] for i in range(n-1)]
    if k in ("circular", "ring"):
        return [[i, i+1] for i in range(n-1)] + [[n-1, 0]]
    if k in ("full", "alltoall", "all_to_all", "dense"):
        return [[i, j] for i in range(n) for j in range(i+1, n)]
    raise ValueError(f"Unknown entanglement kind '{kind}'")

# ---------- index padding ----------
def _map_idx(idx: int, D: int, pad_mode: str) -> Optional[int]:
    if D <= 0:
        return None
    if 0 <= idx < D:
        return idx
    p = str(pad_mode).lower()
    if p == "zero":
        return None
    if p == "wrap":
        return idx % D
    if p in ("repeatlast", "repeat_last", "last"):
        return D - 1
    raise ValueError(f"Bad pad_mode '{pad_mode}'")


_TW0Q_ALIAS = {"cx": "cnot", "cnot": "cnot", "cz": "cz", "swap": "swap", "rxx": "rxx", "ryy": "ryy", "rzz": "rzz"}

# ---------- main builders ----------
def build_efficient_su2_oplist_qisk_new(
    n_wires: int,
    D: int,
    *,
    single_ops: Sequence[str] = ("ry", "rz"),
    entanglement: Optional[str] = "linear",
    twoq: str = "cx",
    pad_mode: str = "wrap",
    reps: Optional[int] = None,
    alpha: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Efficient-SU2 feature-map style:
      per rep: for op in single_ops: apply op(feature) on each wire; then fixed entangler.
    Rotation gates carry {"input_idx":[i], "scale": alpha}. Entanglers have no input_idx.

    - n_wires: number of qubits
    - D: number of classical features
    - single_ops: any subset of ("rx","ry","rz")
    - entanglement: "linear" | "circular" | "full" | None
    - twoq: entangler gate name ("cx","cz",...)
    - pad_mode: "wrap" | "zero" | "repeatlast"
    - reps: if None, uses ceil(D / (n_wires * len(single_ops))) (min 1)
    - alpha: global angle scale
    """
    # validate single_ops
    ok = {"rx", "ry", "rz"}
    sops = [op.lower() for op in single_ops]
    if any(op not in ok for op in sops):
        bad = [op for op in sops if op not in ok]
        raise ValueError(f"Eff-SU2 only supports X/Y/Z rotations; got {bad}")

    per_rep = max(1, n_wires * max(1, len(sops)))
    reps = reps if reps is not None else (1 if per_rep == 0 else max(1, math.ceil(D / per_rep)))

    edges = _entangler_edges(n_wires, entanglement)
    oplist: List[Dict[str, Any]] = []
    
    twq = _TW0Q_ALIAS.get(twoq.lower(), twoq.lower())
    name: Optional[str] = None
    idx = 0
    for _ in range(reps):
        # data-encoding single-qubit rotations
        for op in sops:
            for w in range(n_wires):
                src = _map_idx(idx, D, pad_mode)
                idx += 1
                if src is None:
                    # zero-padding -> identity; skip emitting a rotation
                    continue
                oplist.append({
                    "input_idx": [int(src)],
                    "func": op,
                    "wires": [int(w)],
                    "scale": float(alpha),
                })
        # entangler layer (fixed, no parameters)
        for a, b in edges:
            oplist.append({
                "input_idx": None,
                "func": twoq.lower(),
                "wires": [int(a), int(b)],
            })


    if name is None:
        name = f"{n_wires}x{reps}_{''.join(single_ops)}_{entanglement}_{twq}"
    return name, oplist


def build_efficient_su2_from_paulis(
    n_wires: int,
    D: int,
    *,
    paulis: Sequence[str] = ("Y", "Z"),
    entanglement: Optional[str] = "linear",
    twoq: str = "cx",
    pad_mode: str = "wrap",
    reps: Optional[int] = None,
    alpha: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper: map Pauli letters to rotation gates:
      "X"->rx, "Y"->ry, "Z"->rz.
    Multi-qubit strings like "XX","ZZ" are not part of Efficient-SU2 and will raise.
    """
    singles: List[str] = []
    for p in paulis:
        p = p.upper()
        if len(p) != 1:
            raise ValueError(f"Eff-SU2 (feature layer) only takes 1-qubit Pauli letters; got '{p}'")
        if p == "X":
            singles.append("rx")
        elif p == "Y":
            singles.append("ry")
        elif p == "Z":
            singles.append("rz")
        else:
            raise ValueError(f"Unknown Pauli '{p}'")
    return build_efficient_su2_oplist_qisk_new(
        n_wires=n_wires,
        D=D,
        single_ops=singles,
        entanglement=entanglement,
        twoq=twoq,
        pad_mode=pad_mode,
        reps=reps,
        alpha=alpha,
    )
