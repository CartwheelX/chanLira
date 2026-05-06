import math
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Optional, Union, List, Tuple, Dict, Any

from torchquantum.functional import func_name_dict


# ------------------------------------------------------------------------------------------
# The following stuff is for drawing the encoding part of the circuit only for visulaization
# ------------------------------------------------------------------------------------------




import math
from typing import Optional, List, Dict, Any

import torchquantum as tq
from qiskit import QuantumCircuit as QkCircuit
from qiskit.circuit import ParameterVector
from torchquantum.plugin import op_history2qiskit


def _resolve_idx(i: int, D: int, pad_mode: str) -> Optional[int]:
    """Map a requested feature index to a valid [0..D-1] or None (for zero-pad)."""
    if D <= 0:
        return None
    if 0 <= i < D:
        return i
    p = pad_mode.lower()
    if p == "zero":
        return None
    if p == "wrap":
        return i % D
    if p in ("repeatlast", "repeat_last", "last"):
        return D - 1
    raise ValueError(f"Bad pad_mode '{pad_mode}'")


def _apply_symbolic_gate(
    qc: QkCircuit,
    entry: Dict[str, Any],
    xvec: ParameterVector,
    D: int,
    pad_mode: str,
    alpha: float,
):
    """
    Understands ops shaped like your encoder oplist entries:
      - {"func":"h"/"s"/"sdg"/"id", "wires":[q]}
      - {"func":"cx"/"cnot"/"cz", "wires":[c,t]}
      - {"func":"rz"/"u1"/"p"/"rx"/"ry", "wires":[q], "input_idx":[...], "combine":..., "scale":...}
      - {"func":"u3", "wires":[q], "input_idx":None, "params":[theta,phi,lam]}  # constants
    Notes:
      * We draw u1/p/rz as P(θ) for clarity when possible.
      * combine: 'sum' (default) | 'prod' | 'mean'
      * θ := alpha * scale * <combine(x[idxs])>
    """
    func   = str(entry["func"]).lower()
    wires  = entry["wires"]
    idxs   = entry.get("input_idx", None)
    scale  = float(entry.get("scale", 1.0))
    combine = str(entry.get("combine", "sum")).lower()

    # ---- no-parameter ops ----
    if idxs in (None, [], ()):
        if func == "id":
            return
        if func == "h":
            qc.h(wires[0]); return
        if func == "s":
            qc.s(wires[0]); return
        if func == "sdg":
            qc.sdg(wires[0]); return
        if func in ("cx", "cnot"):
            qc.cx(wires[0], wires[1]); return
        if func == "cz":
            qc.cz(wires[0], wires[1]); return
        if func == "u3":
            # constants-only u3
            params = entry.get("params", None)
            if not (isinstance(params, (list, tuple)) and len(params) == 3):
                # nothing to draw if malformed
                return
            th, ph, la = params
            # Prefer qc.u if available; fall back to u3
            if hasattr(qc, "u"):
                qc.u(th, ph, la, wires[0])
            else:
                qc.u3(th, ph, la, wires[0])
            return
        # unknown no-parameter op -> ignore quietly
        return

    # ---- parameterized ops: build symbolic θ from xvec ----
    if not isinstance(idxs, (list, tuple)):
        idxs = [int(idxs)]
    # map indices with padding policy
    mapped = [_resolve_idx(int(i), D, pad_mode) for i in idxs]
    # zero-pad => skip (identity) if any source is None and combine != 'sum' case
    # For 'sum': missing sources contribute 0; for 'prod' they contribute 1; for 'mean' skip None
    terms = []
    for j, mj in zip(idxs, mapped):
        if mj is None:
            # zero padding: contribute neutral element depending on combine
            if combine == "sum":
                terms.append(0.0)
            elif combine == "prod":
                terms.append(1.0)
            elif combine == "mean":
                # we handle mean by counting actual terms later; skip None
                continue
        else:
            terms.append(xvec[mj])

    if len(terms) == 0:
        # everything was padded out to neutral -> identity
        return

    if combine == "sum":
        theta = terms[0]
        for t in terms[1:]:
            theta = theta + t
    elif combine == "prod":
        theta = terms[0]
        for t in terms[1:]:
            theta = theta * t
    elif combine == "mean":
        # mean over present (non-None) entries
        cnt = 0
        theta = 0.0
        for t in terms:
            if hasattr(t, "parameters") or isinstance(t, float) or isinstance(t, int):
                theta = theta + t
                cnt += 1
        theta = theta / max(1, cnt)
    else:
        # unknown -> default to sum
        theta = terms[0]
        for t in terms[1:]:
            theta = theta + t

    theta = alpha * scale * theta

    # Draw it
    if func in ("u1", "p", "rz"):
        qc.p(theta, wires[0]); return
    if func == "rx":
        qc.rx(theta, wires[0]); return
    if func == "ry":
        qc.ry(theta, wires[0]); return
    # two-qubit parameterized gates (optional if you ever include them)
    if func in ("rzz", "rxx", "ryy", "crx", "cry", "crz"):
        getattr(qc, func)(theta, wires[0], wires[1]); return

    # fallback: try a QC method of same name with (theta, *wires)
    if hasattr(qc, func):
        getattr(qc, func)(theta, *wires)


def export_full_with_symbolic_encoder(
    model,
    D: int,
    pad_mode: Optional[str] = None,   # if None, read from model.encoder.pad_mode or default 'wrap'
    alpha: Optional[float] = None,    # if None, read from model.encoder.alpha or default 1.0
    backend: str = "mpl",
    save_path: Optional[str] = None,
):
    """
    Build a Qiskit circuit:
      encoder (symbolic in terms of x[i]) ∘ variational (numeric via op_history)
    Works with the new raw-oplist encoder or older ones with model.func_list.
    """
    n = model.cfg.n_wires

    # 1) locate the encoder op list
    func_list = None
    if hasattr(model, "func_list"):
        func_list = model.func_list
    elif hasattr(model, "encoder") and hasattr(model.encoder, "oplist"):
        func_list = model.encoder.oplist
    elif hasattr(model, "encoder") and hasattr(model.encoder, "func_list"):
        func_list = model.encoder.func_list
    if func_list is None:
        raise TypeError("Could not find encoder op list (looked for model.func_list or model.encoder.oplist).")

    # 2) determine pad_mode & alpha
    if pad_mode is None:
        pad_mode = getattr(getattr(model, "encoder", None), "pad_mode", "wrap")
    if alpha is None:
        alpha = float(getattr(getattr(model, "encoder", None), "alpha", 1.0))

    # 3) symbolic encoder QC
    enc_qc = QkCircuit(n, name="encoder")
    x = ParameterVector("x", D)
    for entry in func_list:
        _apply_symbolic_gate(enc_qc, entry, x, D, pad_mode, alpha)

    # 4) numeric variational QC (skip encoder)
    qdev = tq.QuantumDevice(n_wires=n, bsz=1, device="cpu", record_op=True)
    # Prefer an explicit "variational only" method if you have one
    if hasattr(model, "circuit_only"):
        model.circuit_only(qdev)
    else:
        # otherwise, call the variational block directly if exposed;
        # or keep your current model.circuit(qdev) if that's already "VQ only"
        model.vqc_circuit(qdev)

    vq_qc = op_history2qiskit(n, qdev.op_history)
    vq_qc.name = "variational"

    # 5) compose
    full_qc = enc_qc.compose(vq_qc)
    full_qc.name = "encoder_plus_variational"

    # 6) draw/save
    fig = full_qc.draw(output=backend)
    if save_path:
        if backend == "mpl":
            fig.savefig(save_path, bbox_inches="tight", dpi=200)
        else:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(str(fig))
    return full_qc

# ------------------------------------------------------------------------------------------


def qiskit_phi(vals: List[torch.Tensor]) -> torch.Tensor:
    # vals: [xi] or [xi, xj]; return (B,)
    pi = math.pi
    if len(vals) == 1:
        return 2.0 * vals[0]                     # Z term: P(2*x[i])
    elif len(vals) == 2:
        xi, xj = vals
        return 2.0 * (pi - xi) * (pi - xj)       # ZZ term: P(2*(π-xi)*(π-xj))
    else:
        raise ValueError("phi expects 1 or 2 inputs.")
    

class GeneralEncoderPlus(tq.QuantumModule):
    """
    Backwards compatible with GeneralEncoder entries:
      - Standard ops: {"input_idx":[...], "func":"rz", "wires":[...], "scale": s?}
      - ZZ map ops:   {"func":"rzz_phi", "input_idx":[i,j], "wires":[a,b], "phi": "prod|pi_minus", "scale": s}
    Notes:
      * 'scale' (if present) multiplies the angle.
      * Missing 'input_idx' or empty => treated as no-parameter gate.
    """

    def __init__(self, func_list: list[dict]):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
        B = x.shape[0]
        for info in self.func_list:
            func = info["func"].lower()
            wires = info["wires"]
            scale = float(info.get("scale", 1.0))
            idxs  = info.get("input_idx", None)

            if func == "rzz_phi":
                ia, ib = idxs
                xa, xb = x[:, ia], x[:, ib]
                phi_kind = info.get("phi", "prod").lower()
                if phi_kind == "prod":
                    theta = scale * (xa * xb)
                elif phi_kind in ("pi_minus", "piminus"):
                    theta = scale * ((math.pi - xa) * (math.pi - xb))
                else:
                    raise ValueError(f"unknown phi '{phi_kind}'")

                # functional call (accepts static/parent_graph)
                func_name_dict["rzz"](
                    qdev, wires=wires, params=theta,
                    static=self.static_mode, parent_graph=self.graph
                )
                continue

            # standard ops (rz/rx/ry/..., possibly with scale)
            gate = func_name_dict[func]
            if idxs is None:
                params = None
            else:
                params = x[:, idxs]
                if params.ndim == 2 and params.size(1) == 1:
                    params = params.squeeze(1)
                params = params * scale

            gate(qdev, wires=wires, params=params,
                 static=self.static_mode, parent_graph=self.graph)


import math, json
from pathlib import Path

#-------------------------------------------------------OPLIST GENERATOR------------------------------------------------
# ---------- pairs ----------
def _pairs_linear(n):
    # chain: (0,1),(1,2),...,(n-2,n-1)
    return [(i, i+1) for i in range(n-1)]

def _pairs_ring(n):
    # ring: (0,1),(1,2),...,(n-1,0)
    return [(i, (i+1) % n) for i in range(n)]

def _pairs_full(n):
    return [(i, j) for i in range(n) for j in range(i+1, n)]

def _get_pairs(n_wires, entanglement: str):
    ent = entanglement.lower()
    if ent in ("ring", "circular"):
        return _pairs_ring(n_wires)
    if ent in ("linear", "chain"):
        return _pairs_linear(n_wires)
    if ent in ("full", "all"):
        return _pairs_full(n_wires)
    raise ValueError(f"Unknown entanglement '{entanglement}'")


# ---------- padding ----------
def _pad_index(idx: int, D: int, pad_mode: str):
    if idx < D:
        return idx
    if D <= 0:
        return None  # nothing to map to
    pm = pad_mode.lower()
    if pm == "zero":
        return None   # signal: skip this op (angle 0 => identity)
    if pm == "wrap":
        return idx % D
    if pm in ("repeatlast", "repeat_last", "last"):
        return D - 1
    raise ValueError(f"bad pad_mode '{pad_mode}'")


# ---------- Z Feature Map: H on all wires, then RZ( 2*alpha * x_i ) ----------
def make_z_oplist(n_wires: int, D: int, *, alpha: float = 1.0,
                  pad_mode: str = "zero", reps: Optional[int] = None) -> Tuple[str, List[Dict[str, Any]]]:
# def make_z_oplist(n_wires: int, D: int, *, alpha: float = 1.0,
#                   pad_mode: str = "zero", reps: int | None = None):
    if reps is None:
        reps = max(1, math.ceil(D / max(1, n_wires)))
    name = f"{n_wires}x{reps}_hrz"
    ops = []

    feat_idx = 0
    for r in range(reps):
        # H on all wires
        for w in range(n_wires):
            ops.append({"input_idx": None, "func": "h", "wires": [w]})

        # per-wire data RZ (2*alpha*x_i); store scale so loader can multiply
        for w in range(n_wires):
            idx = _pad_index(feat_idx, D, pad_mode)
            feat_idx += 1
            if idx is None:
                continue  # zero padding => skip (identity)
            ops.append({"input_idx": [idx], "func": "rz", "wires": [w], "scale": 2.0 * alpha})

    return name, ops


# ---------- ZZ Feature Map (Qiskit-style): H, RZ(2αx_i), then ZZ(2α φ(x_i,x_j)) ----------
# def make_zz_oplist(n_wires: int, D: int, *, entanglement: str = "ring",
#                    phi: str = "prod", alpha: float = 1.0,
#                    pad_mode: str = "zero", reps: int | None = None):
def make_zz_oplist(n_wires: int, D: int, *, entanglement: str = "ring",
                   phi: str = "prod", alpha: float = 1.0,
                   pad_mode: str = "zero", reps: Optional[int] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """
    φ choices:
      'prod'     => x_a * x_b
      'pi_minus' => (π - x_a) * (π - x_b)

    NOTE: We do NOT consume extra features for ZZ; φ uses the SAME per-wire features of that rep.
    So reps = ceil(D / n_wires).
    """
    if reps is None:
        reps = max(1, math.ceil(D / max(1, n_wires)))
    pairs = _get_pairs(n_wires, entanglement)
    name = f"{n_wires}x{reps}_hrzz_{entanglement.lower()}_{phi.lower()}"
    ops = []

    # For each rep we map a chunk of size n_wires to the per-wire RZ.
    # RZZ uses those same indices (recorded via 'src': [idx_a, idx_b]).
    for r in range(reps):
        base = r * n_wires

        # H layer
        for w in range(n_wires):
            ops.append({"input_idx": None, "func": "h", "wires": [w]})

        # per-wire RZ from data
        used_idx = []
        for w in range(n_wires):
            idx = _pad_index(base + w, D, pad_mode)
            used_idx.append(idx)
            if idx is None:
                continue
            ops.append({"input_idx": [idx], "func": "rz", "wires": [w], "scale": 2.0 * alpha})

        # RZZ with φ(x_a, x_b) using the *same* indices of this rep
        for a, b in pairs:
            ia, ib = used_idx[a], used_idx[b]
            if ia is None or ib is None:
                # zero padding on either -> skip this RZZ (angle 0)
                continue
            ops.append({
                "input_idx": [ia, ib],     # we store both sources
                "func": "rzz_phi",         # to be interpreted by GeneralEncoderPlus
                "wires": [a, b],
                "phi": phi,                # 'prod' or 'pi_minus'
                "scale": 2.0 * alpha
            })

    return name, ops


# ---------- writer ----------
# def write_oplist_py(out_path: str | Path, name: str, ops: list[dict]):
def write_oplist_py(out_path: Union[str, Path], name: str, ops: List[Dict[str, Any]]) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("encoder_op_list_name_dict = {\n")
        f.write(f'  "{name}": [\n')
        for op in ops:
            f.write(f"    {json.dumps(op)},\n")
        f.write("  ]\n}\n")
    return str(out_path)

import re

def _pad_index_pauli(idx: int, D: int, pad_mode: str) -> Optional[int]:
    """Return a valid index or None to signal identity (zero angle)."""
    if idx < D:
        return idx
    if pad_mode == "wrap":
        return idx % max(1, D)
    if pad_mode in ("repeatlast", "repeat_last", "last"):
        return D - 1 if D > 0 else 0
    return None  # pad_mode == "zero" => identity

def _parse_term(term: str) -> Tuple[str, List[int]]:
    # e.g. 'ZZ01' -> axes='ZZ', idxs=[0,1]
    axes = ''.join(ch for ch in term if ch in "IXYZ")
    idxs = list(map(int, re.findall(r"\d+", term)))
    return axes, idxs

def _basis_in_ops(axis: str, wire: int) -> List[Dict[str, Any]]:
    # transform axis to Z-basis
    if axis == "X":
        return [{"input_idx": None, "func": "h",   "wires": [wire]}]
    if axis == "Y":
        # Sdg then H
        return [{"input_idx": None, "func": "sdg", "wires": [wire]},
                {"input_idx": None, "func": "h",   "wires": [wire]}]
    return []

def _basis_out_ops(axis: str, wire: int) -> List[Dict[str, Any]]:
    # inverse of basis change
    if axis == "X":
        return [{"input_idx": None, "func": "h",   "wires": [wire]}]
    if axis == "Y":
        # H then S
        return [{"input_idx": None, "func": "h",   "wires": [wire]},
                {"input_idx": None, "func": "s",   "wires": [wire]}]
    return []

def build_pauli_feature_oplist(
    n_wires: int,
    D: int,
    *,
    terms: List[str],
    reps: Optional[int] = None,
    pad_mode: str = "wrap",
    name_prefix: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Create a GeneralEncoder-compatible op-list for a PauliFeatureMap.
    Each term uses one feature per repetition.  Basis changes are handled.
    """
    per_rep = len(terms)
    if per_rep == 0:
        raise ValueError("No Pauli terms provided.")
    if reps is None:
        reps = max(1, (D + per_rep - 1) // per_rep)

    ops: List[Dict[str, Any]] = []
    idx = 0

    for _ in range(reps):
        for term in terms:
            axes, idxs = _parse_term(term)
            # decide which feature to consume
            k = _pad_index_pauli(idx, D, pad_mode)
            idx += 1
            if k is None:
                # skip identity contribution
                continue

            if len(idxs) == 1:
                q = idxs[0]
                ax = axes[-1]
                ops.extend(_basis_in_ops(ax, q))
                # RZ: the feature value will be scaled later in GeneralEncoderPlus
                ops.append({"input_idx": [k], "func": "rz", "wires": [q], "scale": 2.0})
                ops.extend(_basis_out_ops(ax, q))
            elif len(idxs) == 2:
                a, b = idxs
                ax_a, ax_b = axes[0], axes[1]
                ops.extend(_basis_in_ops(ax_a, a))
                ops.extend(_basis_in_ops(ax_b, b))
                # Implement two‑qubit Pauli via CX–RZ–CX: angle = 2 * feature
                ops.append({"input_idx": None, "func": "cx", "wires": [a, b]})
                ops.append({"input_idx": [k], "func": "rz", "wires": [b], "scale": 2.0})
                ops.append({"input_idx": None, "func": "cx", "wires": [a, b]})
                ops.extend(_basis_out_ops(ax_a, a))
                ops.extend(_basis_out_ops(ax_b, b))
            else:
                raise NotImplementedError("Only 1- and 2-body Pauli strings are supported.")

    # create a name if none supplied
    name = name_prefix or f"{n_wires}x{reps}_pauli_{'_'.join(terms)}"
    return name, ops



# ---------- PauliFeatureMap op-list (Qiskit-style) ----------
from typing import Optional, Sequence, List, Dict, Any, Tuple
import re

def _pad_index(k: int, D: int, pad_mode: str) -> Optional[int]:
    if k < D:
        return k
    if pad_mode == "wrap":
        return k % max(1, D)
    if pad_mode == "repeatlast":
        return max(0, D - 1)
    return None  # 'zero' -> treat as identity (skip this rotation)

def _pairs_linear(n: int) -> List[Tuple[int,int]]:
    return [(i, i+1) for i in range(n-1)]

def _pairs_ring(n: int) -> List[Tuple[int,int]]:
    return [(i, (i+1) % n) for i in range(n)]

def _pairs_full(n: int) -> List[Tuple[int,int]]:
    return [(i, j) for i in range(n) for j in range(i+1, n)]

def _basis_change_ops(axis: str, wire: int, inverse: bool) -> List[Dict[str, Any]]:
    """
    Basis change to/from Z:
      X: H
      Y: S† then H (forward), H then S (inverse)
      Z/I: no change
    """
    ops: List[Dict[str, Any]] = []
    a = axis.upper()
    if a == "X":
        ops.append({"input_idx": None, "func": "h", "wires": [wire]})
    elif a == "Y":
        if not inverse:
            ops.append({"input_idx": None, "func": "sdg", "wires": [wire]})
            ops.append({"input_idx": None, "func": "h",   "wires": [wire]})
        else:
            ops.append({"input_idx": None, "func": "h", "wires": [wire]})
            ops.append({"input_idx": None, "func": "s", "wires": [wire]})
    # Z or I -> nothing
    return ops

def _parse_pauli_label(label: str) -> Tuple[str, int, Optional[int]]:
    """
    Accept 'Z', 'X', 'Y', 'ZZ', 'XX', 'XY', ... optionally with indices embedded
    like 'Z0', 'ZZ01', 'XY12'. If indices are not embedded, this function
    returns only axes; indices are supplied by the builder (all wires / entanglement pairs).
    """
    axes = ''.join(c for c in label.upper() if c in "IXYZ")
    idxs = list(map(int, re.findall(r"\d+", label)))
    if len(axes) == 1:
        return axes, (idxs[0] if idxs else -1), None
    elif len(axes) == 2:
        if len(idxs) >= 2:
            return axes, idxs[0], idxs[1]
        return axes, -1, -1  # to be filled by entanglement pairing
    else:
        raise ValueError(f"Unsupported Pauli label: {label}")

def build_pauli_map_qiskit_ops(
    n_wires: int,
    D: int,
    *,
    paulis: Sequence[str] = ("Z", "ZZ"),
    entanglement: str = "linear",   # 'linear' | 'ring' | 'full'
    reps: Optional[int] = None,
    pad_mode: str = "wrap",         # 'wrap' | 'repeatlast' | 'zero'
    name_prefix: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Qiskit-like PauliFeatureMap:
      - For each repetition and each pauli in `paulis`:
         * 1-body (X/Y/Z): apply basis change, U1(θ) on each wire, undo basis change.
         * 2-body (XX/XY/YY/XZ/YZ/ZZ): basis-change both, CX-U1(θ)-CX on the pair, undo basis changes.
      - θ is bound to a *single* classical feature index (GeneralEncoder style).
      - Feature indices advance sequentially; padding avoids out-of-range.

    Returns: (name, op_list)
    """
    paulis = tuple(p.upper() for p in paulis)
    # choose pairing set if we need it
    if entanglement == "linear":
        pairs = _pairs_linear(n_wires)
    elif entanglement == "ring":
        pairs = _pairs_ring(n_wires)
    elif entanglement == "full":
        pairs = _pairs_full(n_wires)
    else:
        raise ValueError(f"Unknown entanglement: {entanglement}")

    # Count features consumed per repetition
    n_1body = sum(1 for p in paulis if len(p) == 1)
    n_2body = sum(1 for p in paulis if len(p) == 2)
    per_rep = n_1body * n_wires + n_2body * len(pairs)
    if per_rep == 0:
        raise ValueError("Empty Pauli set; nothing to encode.")
    if reps is None:
        reps = (D + per_rep - 1) // per_rep  # ceil(D / per_rep)

    ops: List[Dict[str, Any]] = []
    idx = 0

    for _ in range(reps):
        for p in paulis:
            if len(p) == 1:
                axis, fixed_i, _ = _parse_pauli_label(p)
                # loop over all wires unless index embedded
                wire_list = [fixed_i] if fixed_i >= 0 else list(range(n_wires))
                for w in wire_list:
                    # basis to Z
                    ops.extend(_basis_change_ops(axis, w, inverse=False))
                    # angle from next feature index (or padded)
                    k = _pad_index(idx, D, pad_mode); idx += 1
                    if k is not None and axis != "I":
                        ops.append({"input_idx": [k], "func": "u1", "wires": [w]})
                    # undo basis change
                    ops.extend(_basis_change_ops(axis, w, inverse=True))

            elif len(p) == 2:
                ax0, ax1 = p[0], p[1]
                axes, fixed_i, fixed_j = _parse_pauli_label(p)
                # choose pairs: embedded indices -> only that pair; else all according to entanglement
                pair_list = ([(fixed_i, fixed_j)]
                             if (fixed_i is not None and fixed_i >= 0 and fixed_j is not None and fixed_j >= 0)
                             else pairs)
                for a, b in pair_list:
                    # basis to Z for both
                    ops.extend(_basis_change_ops(ax0, a, inverse=False))
                    ops.extend(_basis_change_ops(ax1, b, inverse=False))

                    # CX - U1(θ) - CX on (a -> control, b -> target)
                    k = _pad_index(idx, D, pad_mode); idx += 1
                    ops.append({"input_idx": None, "func": "cx", "wires": [a, b]})
                    if k is not None and not (ax0 == "I" and ax1 == "I"):
                        ops.append({"input_idx": [k], "func": "u1", "wires": [b]})
                    ops.append({"input_idx": None, "func": "cx", "wires": [a, b]})

                    # undo basis
                    ops.extend(_basis_change_ops(ax0, a, inverse=True))
                    ops.extend(_basis_change_ops(ax1, b, inverse=True))
            else:
                raise ValueError(f"Unsupported Pauli term: {p}")

    pauli_tag = "_".join(paulis).lower()
    name = name_prefix or f"{n_wires}x{reps}_pauli_{pauli_tag}_{entanglement}"
    return name, ops


def save_encoder_oplist_py(filepath: str, name: str, ops: List[Dict[str, Any]]) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("encoder_op_list_name_dict = {\n")
        f.write(f'  "{name}": [\n')
        for o in ops:
            f.write("    " + repr(o).replace("'", '"') + ",\n")
        f.write("  ]\n}\n")
