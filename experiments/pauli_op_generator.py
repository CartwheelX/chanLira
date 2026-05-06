import math, re
from typing import List, Dict, Any, Sequence, Optional
from numbers import Number

from qiskit.circuit.library import PauliFeatureMap
from qiskit.circuit import QuantumCircuit

# ---------- small helpers ----------
def _is_num(x) -> bool:
    return isinstance(x, Number)

def _is_symbolic(x) -> bool:
    if isinstance(x, Number):
        return False
    return hasattr(x, "parameters") and len(getattr(x, "parameters", [])) > 0

def _ordered_params(params) -> List:
    def key(p):
        m = re.search(r"(\d+)$", p.name)
        if m:
            return (p.name[: m.start()], int(m.group(1)))
        return (p.name, -1)
    return sorted(list(params), key=key)

# ---------- translate one rep ----------
def _translate_single_rep(qc: QuantumCircuit) -> List[Dict[str, Any]]:
    qb2i = {qb: i for i, qb in enumerate(qc.qubits)}
    base_params = _ordered_params(qc.parameters)
    base_pos = {p: i for i, p in enumerate(base_params)}
    out: List[Dict[str, Any]] = []

    for inst, qargs, _ in qc.data:
        name = inst.name.lower()
        wires = [qb2i[q] for q in qargs]

        # data phases
        if name in ("p", "phase", "rz", "u1"):
            lam = inst.params[0]
            if _is_symbolic(lam):
                idxs = sorted(base_pos[p] for p in lam.parameters)
                # coefficient of the monomial in *its own* variables
                try:
                    coeff = float(lam.bind({p: 1.0 for p in lam.parameters}))
                except Exception:
                    coeff = 2.0
                # PauliFeatureMap should always give coeff==2.0; sanitize if not
                if not math.isfinite(coeff) or abs(coeff - 2.0) > 1e-6:
                    coeff = 2.0
                combine = "prod" if len(idxs) > 1 else "id"
                out.append({"input_idx": idxs, "func": "u1", "wires": wires,
                            "scale": coeff, "combine": combine})
            else:
                out.append({"input_idx": None, "func": "u1", "wires": wires,
                            "params": [float(lam)]})
            continue

        # some decompositions keep symbolic as U(0,0,λ)
        if name in ("u", "u3"):
            th, ph, lam = inst.params
            if (_is_num(th) and _is_num(ph)
                and abs(float(th)) < 1e-12 and abs(float(ph)) < 1e-12
                and _is_symbolic(lam)):
                idxs = sorted(base_pos[p] for p in lam.parameters)
                try:
                    coeff = float(lam.bind({p: 1.0 for p in lam.parameters}))
                except Exception:
                    coeff = 2.0
                if not math.isfinite(coeff) or abs(coeff - 2.0) > 1e-6:
                    coeff = 2.0
                combine = "prod" if len(idxs) > 1 else "id"
                out.append({"input_idx": idxs, "func": "u1", "wires": wires,
                            "scale": coeff, "combine": combine})
            else:
                try:
                    params = [float(th), float(ph), float(lam)]
                except Exception:
                    params = [None, None, None]
                out.append({"input_idx": None, "func": "u3", "wires": wires, "params": params})
            continue

        # all other fixed gates (h/s/sdg/cx/…)
        out.append({"input_idx": None, "func": name, "wires": wires})

    return out

# ---------- index mapping per rep ----------
def _map_indices_for_rep(idxs: Sequence[int], base: int, D: int, pad_mode: str):
    if D <= 0:
        return (None, True)
    gidxs = [base + i for i in idxs]
    if pad_mode == "zero":
        if any(g >= D for g in gidxs):
            return (None, True)
        return (gidxs, False)
    if pad_mode == "wrap":
        return ([g % D for g in gidxs], False)
    if pad_mode == "repeatlast":
        last = D - 1
        return ([g if g < D else last for g in gidxs], False)
    raise ValueError(f"Unknown pad_mode '{pad_mode}'")

# ---------- public builder ----------
def build_tiled_pauli_oplist(
    n_wires: int,
    D: int,
    paulis: Sequence[str],
    entanglement: str = "linear",
    pad_mode: str = "wrap",
    repeats = 1,
    *,
    expand_h_to_u3: bool = False,   # just a cosmetic post-pass
) -> tuple[str, List[Dict[str, Any]]]:
    """
    Returns (name, op_list) for a tiled PauliFeatureMap faithful to Qiskit’s structure.
    - multi-index phases carry {"combine":"prod", "scale":2.0}
    - single-index phases carry {"combine":"id",   "scale":2.0}
    """
    assert n_wires >= 1
    paulis = tuple(p.upper() for p in paulis)
    assert pad_mode in ("wrap", "repeatlast", "zero")
    ent_tag = "ring" if entanglement == "ring" else entanglement

    # Build *one repetition* with Qiskit, no transpile to avoid phase fusion
    prep = PauliFeatureMap(
        feature_dimension=n_wires,
        reps=repeats,
        paulis=list(paulis),
        entanglement=("circular" if entanglement == "ring" else entanglement),
    )
    qc = prep.decompose()

    base_ops = _translate_single_rep(qc)

    # Tile across reps to consume D features
    reps = max(1, math.ceil(D / n_wires)) if D > 0 else 1
    tiled: List[Dict[str, Any]] = []
    for r in range(reps):
        base = r * n_wires
        for op in base_ops:
            if op.get("input_idx") is None:
                tiled.append(dict(op))
                continue
            idxs = op["input_idx"]
            mapped, force_zero = _map_indices_for_rep(idxs, base, D, pad_mode)
            if force_zero:
                tiled.append({"input_idx": None, "func": "u1", "wires": op["wires"], "params": [0.0]})
            else:
                step = {k: op[k] for k in op if k not in ("input_idx",)}
                step.update({"input_idx": mapped})
                tiled.append(step)

    # Cosmetic: expand H to U3(pi/2, 0, pi) *after* extraction so it can't spoil scales
    if expand_h_to_u3:
        H_U3 = [math.pi/2, 0.0, math.pi]
        new_tiled = []
        for s in tiled:
            if s["func"] == "h":
                new_tiled.append({"input_idx": None, "func": "u3", "wires": s["wires"], "params": H_U3[:]})
            else:
                new_tiled.append(s)
        tiled = new_tiled

    name = f"{n_wires}x{reps}_pauli_{'_'.join(p.lower() for p in paulis)}_{ent_tag}_{pad_mode}"
    return name, tiled
