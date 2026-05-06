


# # # # --- Build tiled PauliFeatureMap oplist (faithful to Qiskit decomposition) ---
# # # # Requirements: qiskit-terra >= 0.24 (a.k.a. Qiskit 0.24+) or Qiskit 1.x
# # # # # Produces: {"<name>": [ {input_idx, func, wires, (params)} , ... ]}

# # # # import math, re
# # # # from typing import List, Dict, Any, Sequence, Optional
# # # # from numbers import Number

# # # # from qiskit.circuit.library import PauliFeatureMap
# # # # from qiskit import transpile
# # # # from qiskit.circuit import ParameterExpression, QuantumCircuit

# # # # # ---------- helpers: parameter ordering / detection ----------
# # # # def _is_num(x) -> bool:
# # # #     return isinstance(x, Number)

# # # # def _is_symbolic(x) -> bool:
# # # #     if isinstance(x, Number):
# # # #         return False
# # # #     # Parameter or ParameterExpression with at least one parameter
# # # #     return hasattr(x, "parameters") and len(getattr(x, "parameters", [])) > 0

# # # # def _ordered_params(params) -> List:
# # # #     """Return circuit parameters ordered by (name root, numeric suffix)."""
# # # #     def key(p):
# # # #         m = re.search(r"(\d+)$", p.name)
# # # #         if m:
# # # #             return (p.name[: m.start()], int(m.group(1)))
# # # #         return (p.name, -1)
# # # #     return sorted(list(params), key=key)

# # # # # ---------- translator from one Qiskit rep -> encoder op skeleton ----------
# # # # def _translate_single_rep(qc: QuantumCircuit) -> List[Dict[str, Any]]:
# # # #     """
# # # #     Translate a single-rep PauliFeatureMap circuit (already decomposed/transpiled)
# # # #     into a list of ops where data-carrying phases produce 'input_idx': [...]
# # # #     (indices refer to the single-rep parameter vector order 0..n_wires-1).
# # # #     """
# # # #     qb2i = {qb: i for i, qb in enumerate(qc.qubits)}
# # # #     base_params = _ordered_params(qc.parameters)
# # # #     base_pos = {p: i for i, p in enumerate(base_params)}
# # # #     out: List[Dict[str, Any]] = []

# # # #     for inst, qargs, _ in qc.data:
# # # #         name = inst.name.lower()
# # # #         wires = [qb2i[q] for q in qargs]

# # # #         # DATA PHASES (single-qubit): p/phase/rz/u1
# # # #         if name in ("p", "phase", "rz", "u1"):
# # # #             lam = inst.params[0]
# # # #             if _is_symbolic(lam):
# # # #                 idxs = sorted(base_pos[p] for p in lam.parameters)
# # # #                 out.append({"input_idx": idxs, "func": "u1", "wires": wires})
# # # #             else:
# # # #                 out.append({"input_idx": None, "func": "u1", "wires": wires, "params": [float(lam)]})
# # # #             continue

# # # #         # DATA PHASES occasionally encoded as U(0,0,lambda)
# # # #         if name in ("u", "u3"):
# # # #             th, ph, lam = inst.params
# # # #             if _is_num(th) and _is_num(ph) and abs(float(th)) < 1e-12 and abs(float(ph)) < 1e-12 and _is_symbolic(lam):
# # # #                 idxs = sorted(base_pos[p] for p in lam.parameters)
# # # #                 out.append({"input_idx": idxs, "func": "u1", "wires": wires})
# # # #             else:
# # # #                 # keep constant U3 (basis change) as-is
# # # #                 try:
# # # #                     params = [float(th), float(ph), float(lam)]
# # # #                 except Exception:
# # # #                     params = [None, None, None]
# # # #                 out.append({"input_idx": None, "func": "u3", "wires": wires, "params": params})
# # # #             continue

# # # #         # Everything else (H/S/SDG/CX, etc.) is constant structure
# # # #         out.append({"input_idx": None, "func": name, "wires": wires})

# # # #     return out

# # # # # ---------- per-index tiling with pad policy ----------
# # # # def _map_indices_for_rep(idxs: Sequence[int], base: int, D: int, pad_mode: str) -> (Optional[List[int]], bool):
# # # #     """
# # # #     Map single-rep parameter indices -> global indices for repetition with base offset.
# # # #     Returns (mapped_indices or None, force_zero_flag).
# # # #     If pad_mode == 'zero' and any index >= D (or D==0), returns (None, True).
# # # #     For 'wrap'/'repeatlast', always returns mapped indices (if D==0 => zero).
# # # #     """
# # # #     if D <= 0:
# # # #         return (None, True)  # no data -> zero angle

# # # #     gidxs = [base + i for i in idxs]
# # # #     if pad_mode == "zero":
# # # #         if any(g >= D for g in gidxs):
# # # #             return (None, True)
# # # #         return (gidxs, False)

# # # #     if pad_mode == "wrap":
# # # #         return ([g % D for g in gidxs], False)

# # # #     if pad_mode == "repeatlast":
# # # #         last = D - 1
# # # #         return ([g if g < D else last for g in gidxs], False)

# # # #     raise ValueError(f"Unknown pad_mode '{pad_mode}'")

# # # # # ---------- public builder ----------
# # # # def build_tiled_pauli_oplist(
# # # #     n_wires: int,
# # # #     D: int,
# # # #     paulis: Sequence[str],
# # # #     entanglement: str = "linear",
# # # #     pad_mode: str = "wrap",
# # # #     *,
# # # #     transpile_to: Optional[Sequence[str]] = ("u3", "cx", "u1", "sdg", "s"), #"h", "cx", "u1", "sdg", "s","u3"
# # # # ) -> Dict[str, List[Dict[str, Any]]]:
# # # #     """
# # # #     Build an encoder op list for Qiskit's PauliFeatureMap(paulis=..., entanglement=..., reps=1),
# # # #     then tile across reps so that ceil(D / n_wires) repetitions of that single-rep structure
# # # #     consume all D features with the desired pad_mode.

# # # #       pad_mode:
# # # #         - 'wrap'       : indices beyond D wrap mod D
# # # #         - 'repeatlast' : indices beyond D clamp to D-1
# # # #         - 'zero'       : data phases referencing out-of-range indices become angle 0

# # # #       transpile_to:
# # # #         Choose basis gates you want the decomposition in. ('u3','cx','p') keeps phase gates explicit.
# # # #         If None, skip transpile and use raw decompose() output.
# # # #     """
# # # #     assert n_wires >= 1
# # # #     paulis = tuple(p.upper() for p in paulis)
# # # #     assert pad_mode in ("wrap", "repeatlast", "zero")

# # # #     # 1) Build single-rep circuit from Qiskit
# # # #     prep = PauliFeatureMap(
# # # #         feature_dimension=n_wires,
# # # #         reps=1,
# # # #         paulis=list(paulis),
# # # #         entanglement=("circular" if entanglement == "ring" else entanglement),
# # # #     )
# # # #     qc = prep.decompose()
# # # #     if transpile_to:
# # # #         qc = transpile(qc, basis_gates=list(transpile_to), optimization_level=0)

# # # #     # 2) Translate one rep into skeleton ops (data gates carry local idx list)
# # # #     base_ops = _translate_single_rep(qc)

# # # #     # 3) Tile across reps to consume D
# # # #     reps = max(1, math.ceil(D / n_wires)) if D > 0 else 1
# # # #     tiled: List[Dict[str, Any]] = []
# # # #     for r in range(reps):
# # # #         base = r * n_wires
# # # #         for op in base_ops:
# # # #             if op.get("input_idx") is None:
# # # #                 # constant gate (basis-change or CX)
# # # #                 tiled.append(dict(op))
# # # #                 continue

# # # #             idxs = op["input_idx"]
# # # #             mapped, force_zero = _map_indices_for_rep(idxs, base, D, pad_mode)

# # # #             if force_zero:
# # # #                 # encode an explicit zero-angle phase (u1 0) to be explicit
# # # #                 tiled.append({"input_idx": None, "func": "u1", "wires": op["wires"], "params": [0.0]})
# # # #             else:
# # # #                 # keep as data gate with global indices
# # # #                 tiled.append({"input_idx": mapped, "func": "u1", "wires": op["wires"]})

# # # #     # 4) Name
# # # #     pauli_tag = "_".join(p.lower() for p in paulis)
# # # #     ent_tag = "ring" if entanglement == "ring" else entanglement
# # # #     name = f"{n_wires}x{reps}_pauli_{pauli_tag}_{ent_tag}_{pad_mode}"

# # # #     return name, tiled

# # # # # # ---------- quick smoke test ----------
# # # # # if __name__ == "__main__":
# # # # #     import json
# # # # #     d = build_tiled_pauli_oplist(
# # # # #         n_wires=4, D=6, paulis=["Y","XX"], entanglement="linear", pad_mode="wrap"
# # # # #     )
# # # # #     print(json.dumps(d, indent=2))



# # # # --- Build tiled PauliFeatureMap oplist (faithful to Qiskit’s structure) ---
# # # # Qiskit 0.24+ (Terra 0.24+) or Qiskit 1.x

# # # import math, re
# # # from math import pi
# # # from typing import List, Dict, Any, Sequence, Optional, Tuple
# # # from numbers import Number

# # # from qiskit.circuit.library import PauliFeatureMap
# # # from qiskit import transpile
# # # from qiskit.circuit import QuantumCircuit


# # # def replace_h_with_u3(oplist):
# # #     out = []
# # #     for op in oplist:
# # #         if op["func"].lower() == "h":
# # #             out.append({
# # #                 "input_idx": None,
# # #                 "func": "u3",
# # #                 "wires": op["wires"],
# # #                 "params": [pi/2, 0.0, pi],
# # #             })
# # #         else:
# # #             out.append(op)
# # #     return out

# # # # ---------- helpers ----------
# # # def _is_num(x) -> bool:
# # #     return isinstance(x, Number)

# # # def _is_symbolic(x) -> bool:
# # #     if isinstance(x, Number):
# # #         return False
# # #     return hasattr(x, "parameters") and len(getattr(x, "parameters", [])) > 0

# # # def _ordered_params(params) -> List:
# # #     """Stable sort for ParameterVector entries like x0, x1, ..."""
# # #     def key(p):
# # #         m = re.search(r"(\d+)$", p.name)
# # #         if m:
# # #             return (p.name[: m.start()], int(m.group(1)))
# # #         return (p.name, -1)
# # #     return sorted(list(params), key=key)

# # # # ---------- translate a single-rep circuit into a skeleton ----------
# # # def _translate_single_rep(qc: QuantumCircuit, *, pair_phi: str) -> List[Dict[str, Any]]:
# # #     """
# # #     Produce ops with local parameter indices (0..n_wires-1) for data phases.
# # #     Inject scale=2.0 to match Qiskit’s P(2·φ(...)).
# # #     For pairwise phases, set 'combine' based on pair_phi ('prod' or 'qiskit').
# # #     """
# # #     qb2i = {qb: i for i, qb in enumerate(qc.qubits)}
# # #     base_params = _ordered_params(qc.parameters)
# # #     base_pos = {p: i for i, p in enumerate(base_params)}
# # #     out: List[Dict[str, Any]] = []

# # #     for inst, qargs, _ in qc.data:
# # #         name = inst.name.lower()
# # #         wires = [qb2i[q] for q in qargs]

# # #         # Data phases (single- or two-parameter), keep as u1 in our IR
# # #         if name in ("p", "phase", "rz", "u1"):
# # #             lam = inst.params[0]
# # #             if _is_symbolic(lam):
# # #                 idxs = sorted(base_pos[p] for p in lam.parameters)  # 1 or 2 indices
# # #                 op = {"input_idx": idxs, "func": "u1", "wires": wires, "scale": 2.0}
# # #                 if len(idxs) == 2:
# # #                     op["combine"] = ("prod" if pair_phi == "prod" else "qiskit")
# # #                 out.append(op)
# # #             else:
# # #                 out.append({"input_idx": None, "func": "u1", "wires": wires,
# # #                             "params": [float(lam)]})
# # #             continue

# # #         # Sometimes phases show up as U(0,0,λ)
# # #         if name in ("u", "u3"):
# # #             th, ph, lam = inst.params
# # #             if _is_num(th) and _is_num(ph) and abs(float(th)) < 1e-12 and abs(float(ph)) < 1e-12 and _is_symbolic(lam):
# # #                 idxs = sorted(base_pos[p] for p in lam.parameters)
# # #                 op = {"input_idx": idxs, "func": "u1", "wires": wires, "scale": 2.0}
# # #                 if len(idxs) == 2:
# # #                     op["combine"] = ("prod" if pair_phi == "prod" else "qiskit")
# # #                 out.append(op)
# # #             else:
# # #                 # keep constant U3 (basis change) as-is (rare if 'h' in basis)
# # #                 try:
# # #                     params = [float(th), float(ph), float(lam)]
# # #                 except Exception:
# # #                     params = [None, None, None]
# # #                 out.append({"input_idx": None, "func": "u3", "wires": wires, "params": params})
# # #             continue

# # #         # Structure gates: h/s/sdg/cx/... (stay as they are)
# # #         out.append({"input_idx": None, "func": name, "wires": wires})

# # #     return out

# # # # ---------- pad / index mapping ----------
# # # def _map_indices_for_rep(
# # #     idxs: Sequence[int], base: int, D: int, pad_mode: str
# # # ) -> Tuple[Optional[List[int]], bool]:
# # #     """
# # #     Map local indices -> global indices for repetition base.
# # #     Returns (mapped_indices or None, force_zero_flag).
# # #     """
# # #     if D <= 0:
# # #         return (None, True)

# # #     gidxs = [base + i for i in idxs]
# # #     if pad_mode == "zero":
# # #         if any(g >= D for g in gidxs):
# # #             return (None, True)
# # #         return (gidxs, False)

# # #     if pad_mode == "wrap":
# # #         return ([g % D for g in gidxs], False)

# # #     if pad_mode == "repeatlast":
# # #         last = D - 1
# # #         return ([g if g < D else last for g in gidxs], False)

# # #     raise ValueError(f"Unknown pad_mode '{pad_mode}'")

# # # # ---------- public builder ----------
# # # def build_tiled_pauli_oplist(
# # #     n_wires: int,
# # #     D: int,
# # #     paulis: Sequence[str],
# # #     entanglement: Optional[str] = "linear",   # "linear" | "ring" | None
# # #     pad_mode: str = "wrap",
# # #     *,
# # #     pair_phi: str = "prod",                   # "prod" or "qiskit"
# # #     transpile_to: Optional[Sequence[str]] = ("h", "cx", "p", "u1", "sdg", "s"),
# # #     # add_name_wrapper: bool = True,
# # # ) -> Dict[str, List[Dict[str, Any]]] | List[Dict[str, Any]]:
# # #     """
# # #     Build a single-rep PauliFeatureMap via Qiskit, decompose/transpile to a
# # #     small gate set, translate to an op list, then tile over reps = ceil(D/n_wires).

# # #     - If entanglement is None, we skip pair terms by giving Qiskit 'linear' then
# # #       dropping CX/P pairs (Qiskit requires a topology). Easiest is to pass paulis
# # #       without two-body strings when you truly want no entanglement.
# # #     - pair_phi:
# # #         "prod"    -> we tag two-params with combine:'prod' (angle = 2*xi*xj)
# # #         "qiskit"  -> tag combine:'qiskit' and implement the exact φ in your runtime encoder.
# # #     """
# # #     assert n_wires >= 1
# # #     assert pad_mode in ("wrap", "repeatlast", "zero")
# # #     assert pair_phi in ("prod", "qiskit")

# # #     paulis = tuple(p.upper() for p in paulis)

# # #     # 1) Build Qiskit circuit for one repetition
# # #     q_ent = ("circular" if entanglement == "ring" else entanglement or "linear")
# # #     prep = PauliFeatureMap(
# # #         feature_dimension=n_wires,
# # #         reps=1,
# # #         paulis=list(paulis),
# # #         entanglement=q_ent,
# # #     )
# # #     qc = prep.decompose()
# # #     if transpile_to:
# # #         qc = transpile(qc, basis_gates=list(transpile_to), optimization_level=0)

# # #     # 2) Translate one rep
# # #     base_ops = _translate_single_rep(qc, pair_phi=pair_phi)

# # #     # 3) Tile across reps to consume D features
# # #     reps = max(1, math.ceil(D / n_wires)) if D > 0 else 1
# # #     tiled: List[Dict[str, Any]] = []
# # #     for r in range(reps):
# # #         base = r * n_wires
# # #         for op in base_ops:
# # #             if op.get("input_idx") is None:
# # #                 tiled.append(dict(op))
# # #                 continue
# # #             # map local -> global feature indices with pad policy
# # #             idxs = op["input_idx"]
# # #             mapped, force_zero = _map_indices_for_rep(idxs, base, D, pad_mode)
# # #             if force_zero:
# # #                 # explicit zero-angle phase (keeps structure)
# # #                 zero = {"input_idx": None, "func": "u1", "wires": op["wires"], "params": [0.0]}
# # #                 tiled.append(zero)
# # #             else:
# # #                 new_op = dict(op)
# # #                 new_op["input_idx"] = mapped
# # #                 tiled.append(new_op)

# # #     # 4) Name and return
# # #     pauli_tag = "_".join(p.lower() for p in paulis)
# # #     ent_tag = "ring" if entanglement == "ring" else (entanglement or "none")
# # #     name = f"{n_wires}x{reps}_pauli_{pauli_tag}_{ent_tag}_{pad_mode}"
# # #     return name, tiled


# # # ===============================
# # # 1) Qiskit -> Tiled Op-List (no 'h'; H is U3(pi/2,0,pi))
# # # ===============================
# # import math, re
# # from typing import List, Dict, Any, Sequence, Optional, Tuple
# # from numbers import Number
# # from math import pi

# # from qiskit.circuit.library import PauliFeatureMap
# # from qiskit import transpile
# # from qiskit.circuit import ParameterExpression, QuantumCircuit

# # # ---------- helpers ----------
# # def _is_num(x) -> bool:
# #     return isinstance(x, Number)

# # def _is_symbolic(x) -> bool:
# #     if isinstance(x, Number):
# #         return False
# #     return hasattr(x, "parameters") and len(getattr(x, "parameters", [])) > 0

# # def _ordered_params(params) -> List:
# #     """Return circuit parameters ordered by (name root, numeric suffix)."""
# #     def key(p):
# #         m = re.search(r"(\d+)$", p.name)
# #         if m:
# #             return (p.name[: m.start()], int(m.group(1)))
# #         return (p.name, -1)
# #     return sorted(list(params), key=key)

# # def _num_scale_from_expr(expr: ParameterExpression) -> Optional[float]:
# #     """If expr is symbolic, evaluate at all-ones to extract the numeric scale (e.g., '2*x0' -> 2.0)."""
# #     try:
# #         if not _is_symbolic(expr):
# #             return float(expr)
# #         bind_map = {p: 1.0 for p in expr.parameters}
# #         return float(expr.bind(bind_map))
# #     except Exception:
# #         return None

# # # ---------- translate 1 rep (already decomposed / transpiled) ----------
# # def _translate_single_rep(qc: QuantumCircuit, *, h_as_u3: bool = True) -> List[Dict[str, Any]]:
# #     """
# #     Convert a single-repetition PauliFeatureMap circuit into a gate op-list.
# #     - Data-phase gates become {"input_idx":[...], "func":"u1", "wires":[...], "scale": <numeric>}
# #     - Constant basis-change gates (u2/u3/s/sdg/cx, etc.) are emitted with numeric params.
# #     - If h_as_u3=True, any 'u2(φ,λ)' is rewritten as 'u3(pi/2, φ, λ)' (so no 'h' remains).
# #     """
# #     qb2i = {qb: i for i, qb in enumerate(qc.qubits)}
# #     base_params = _ordered_params(qc.parameters)
# #     base_pos = {p: i for i, p in enumerate(base_params)}
# #     out: List[Dict[str, Any]] = []

# #     for inst, qargs, _ in qc.data:
# #         name = inst.name.lower()
# #         wires = [qb2i[q] for q in qargs]

# #         # --- data phases (p/phase/rz/u1) ---
# #         if name in ("p", "phase", "rz", "u1"):
# #             lam = inst.params[0]
# #             if _is_symbolic(lam):
# #                 idxs = sorted(base_pos[p] for p in lam.parameters)
# #                 scale = _num_scale_from_expr(lam)  # typically 2.0
# #                 op = {"input_idx": idxs, "func": "u1", "wires": wires}
# #                 if scale is not None and abs(scale - 1.0) > 1e-12:
# #                     op["scale"] = scale
# #                 out.append(op)
# #             else:
# #                 out.append({"input_idx": None, "func": "u1", "wires": wires, "params": [float(lam)]})
# #             continue

# #         # --- U / U3: treat U(0,0,λ) as data phase; keep other U3 numeric ---
# #         if name in ("u", "u3"):
# #             th, ph, lam = inst.params
# #             if _is_num(th) and _is_num(ph) and abs(float(th)) < 1e-12 and abs(float(ph)) < 1e-12 and _is_symbolic(lam):
# #                 idxs = sorted(base_pos[p] for p in lam.parameters)
# #                 scale = _num_scale_from_expr(lam)  # e.g., 2.0
# #                 op = {"input_idx": idxs, "func": "u1", "wires": wires}
# #                 if scale is not None and abs(scale - 1.0) > 1e-12:
# #                     op["scale"] = scale
# #                 out.append(op)
# #             else:
# #                 # keep numeric U3s (basis changes)
# #                 try:
# #                     params = [float(th), float(ph), float(lam)]
# #                 except Exception:
# #                     params = [None, None, None]
# #                 out.append({"input_idx": None, "func": "u3", "wires": wires, "params": params})
# #             continue

# #         # --- U2 -> U3(pi/2, φ, λ) (so no 'u2'/'h' survive) ---
# #         if name == "u2":
# #             phi, lam = inst.params
# #             if h_as_u3:
# #                 out.append({
# #                     "input_idx": None,
# #                     "func": "u3",
# #                     "wires": wires,
# #                     "params": [pi/2, float(phi), float(lam)]
# #                 })
# #             else:
# #                 out.append({"input_idx": None, "func": "u2", "wires": wires, "params": [float(phi), float(lam)]})
# #             continue

# #         # Everything else is constant structure (s, sdg, cx, etc.)
# #         out.append({"input_idx": None, "func": name, "wires": wires})

# #     return out

# # # ---------- pad policy for tiling ----------
# # def _map_indices_for_rep(
# #     idxs: Sequence[int], base: int, D: int, pad_mode: str
# # ) -> Tuple[Optional[List[int]], bool]:
# #     """
# #     Map per-rep local parameter indices to global indices given repetition base offset.
# #     Returns (mapped_indices or None, force_zero_flag).
# #     """
# #     if D <= 0:
# #         return (None, True)

# #     gidxs = [base + i for i in idxs]
# #     if pad_mode == "zero":
# #         if any(g >= D for g in gidxs):
# #             return (None, True)
# #         return (gidxs, False)
# #     if pad_mode == "wrap":
# #         return ([g % D for g in gidxs], False)
# #     if pad_mode == "repeatlast":
# #         last = D - 1
# #         return ([g if g < D else last for g in gidxs], False)
# #     raise ValueError(f"Unknown pad_mode '{pad_mode}'")

# # # ---------- public builder ----------
# # def build_tiled_pauli_oplist(
# #     n_wires: int,
# #     D: int,
# #     paulis: Sequence[str],
# #     entanglement: Optional[str] = "linear",
# #     pad_mode: str = "wrap",
# #     *,
# #     # keep 'h' out of the basis; 'u2' becomes 'u3(pi/2, φ, λ)'
# #     transpile_to: Optional[Sequence[str]] = ("u3", "cx", "u1", "sdg", "s"),
# #     h_as_u3: bool = True,
# # ) -> Tuple[str, List[Dict[str, Any]]]:
# #     """
# #     Build an encoder op-list for a single-rep Qiskit PauliFeatureMap and tile it to consume D features.

# #     Returns: (name, oplist), where oplist is a flat list of {"input_idx", "func", "wires", [params], [scale]}.

# #     paulis examples:
# #       - ["Z"]              -> Z feature map
# #       - ["Z","ZZ"]         -> ZZ feature map
# #       - ["Y","XX"]         -> generic PauliFeatureMap like your screenshot

# #     entanglement: "linear" | "circular" | "full" | None
# #     pad_mode:     "wrap" | "repeatlast" | "zero"
# #     """
# #     assert n_wires >= 1
# #     assert pad_mode in ("wrap", "repeatlast", "zero")
# #     paulis = tuple(p.upper() for p in paulis)

# #     prep = PauliFeatureMap(
# #         feature_dimension=n_wires,
# #         reps=1,
# #         paulis=list(paulis),
# #         entanglement=("circular" if entanglement == "ring" else entanglement),
# #     )
# #     qc = prep.decompose()
# #     if transpile_to:
# #         qc = transpile(qc, basis_gates=list(transpile_to), optimization_level=0)

# #     base_ops = _translate_single_rep(qc, h_as_u3=h_as_u3)

# #     reps = max(1, math.ceil(D / n_wires)) if D > 0 else 1
# #     tiled: List[Dict[str, Any]] = []
# #     for r in range(reps):
# #         base = r * n_wires
# #         for op in base_ops:
# #             if op.get("input_idx") is None:
# #                 # constant structure
# #                 tiled.append(dict(op))
# #                 continue

# #             idxs = op["input_idx"]
# #             mapped, force_zero = _map_indices_for_rep(idxs, base, D, pad_mode)
# #             if force_zero:
# #                 # explicit zero-phase
# #                 tiled.append({"input_idx": None, "func": "u1", "wires": op["wires"], "params": [0.0]})
# #             else:
# #                 newop = {"input_idx": mapped, "func": "u1", "wires": op["wires"]}
# #                 if "scale" in op:
# #                     newop["scale"] = op["scale"]
# #                 tiled.append(newop)

# #     pauli_tag = "_".join(p.lower() for p in paulis)
# #     ent_tag = "ring" if entanglement == "ring" else (entanglement if entanglement else "none")
# #     name = f"{n_wires}x{reps}_pauli_{pauli_tag}_{ent_tag}_{pad_mode}"
# #     return name, tiled

# # # Convenience wrappers (optional)
# # def build_z_oplist(n_wires: int, D: int, pad_mode: str = "wrap") -> Tuple[str, List[Dict[str, Any]]]:
# #     return build_tiled_pauli_oplist(n_wires, D, paulis=["Z"], entanglement=None, pad_mode=pad_mode)

# # def build_zz_oplist(n_wires: int, D: int, entanglement: str = "linear", pad_mode: str = "wrap") -> Tuple[str, List[Dict[str, Any]]]:
# #     return build_tiled_pauli_oplist(n_wires, D, paulis=["Z","ZZ"], entanglement=entanglement, pad_mode=pad_mode)


# # # ===============================
# # # 2) TorchQuantum Op-List Encoder
# # # ===============================
# # import torch
# # import torchquantum as tq

# # def _get_gate_class(func_name: str):
# #     """Map lowercase func string -> TorchQuantum gate class; fallback for U1->RZ, CX/CNOT, SDG variants."""
# #     f = func_name.lower()
# #     # Try the most common names first; adjust here if your TQ build uses different names.
# #     if f == "u3":
# #         return tq.U3
# #     if f in ("u1", "p", "phase", "rz"):
# #         return getattr(tq, "U1", tq.RZ)
# #     if f in ("cx", "cnot"):
# #         return getattr(tq, "CNOT", getattr(tq, "CX", None))
# #     if f == "s":
# #         return tq.S
# #     if f in ("sdg", "sdag"):
# #         return getattr(tq, "SDG", getattr(tq, "Sdg", None))
# #     if f == "h":
# #         # You said you’ll use U3(π/2,0,π) instead; kept for completeness.
# #         return tq.H
# #     # add more if needed (cz, x, y, etc.)
# #     return getattr(tq, f.upper())  # last resort (e.g., "rz","ry","rx" etc.)

# # def run_oplist_encoder(
# #     qdev: tq.QuantumDevice,
# #     x: torch.Tensor,                  # shape: (B, D)
# #     oplist: List[Dict[str, Any]],
# #     *,
# #     alpha: float = 1.0,
# #     multi_index_rule: str = "prod",   # "prod" (Qiskit default) or "sum"
# # ) -> None:
# #     """
# #     Execute a raw op-list on qdev. Each op is:
# #       {"input_idx": None or [..], "func": <str>, "wires": [..], optional "params": [...], optional "scale": c}

# #     - If input_idx is a list:
# #         * len==1 -> angle = alpha * c * x[:,i]
# #         * len>1  -> angle = alpha * c * product(x[:,i] for i in idxs)  (or sum if multi_index_rule="sum")
# #     - If 'params' is present (numeric), we broadcast to batch and pass directly.
# #     - For U3 numeric params, we broadcast a (B,3) tensor.
# #     """
# #     B, D = x.shape
# #     device = x.device
# #     for op in oplist:
# #         func = op["func"].lower()
# #         wires = op["wires"]
# #         Gate = _get_gate_class(func)
# #         if Gate is None:
# #             raise ValueError(f"Unknown/unsupported gate '{func}' for TorchQuantum mapping.")

# #         # Constant-parameter gate?
# #         if op.get("input_idx") is None:
# #             if "params" in op and op["params"] is not None:
# #                 p = op["params"]
# #                 if func == "u3":
# #                     # Make (B,3)
# #                     params = torch.tensor(p, dtype=x.dtype, device=device).view(1, 3).expand(B, 3)
# #                 else:
# #                     # Make (B,)
# #                     params = torch.tensor(p[0], dtype=x.dtype, device=device).expand(B)
# #                 Gate()(qdev, wires=wires, params=params)
# #             else:
# #                 Gate()(qdev, wires=wires)
# #             continue

# #         # Data-parameterized gate (always "u1" from the builder)
# #         idxs = op["input_idx"]
# #         s = float(op.get("scale", 1.0))
# #         if len(idxs) == 1:
# #             base = x[:, idxs[0]]
# #         else:
# #             feats = x[:, idxs]
# #             if multi_index_rule == "sum":
# #                 base = feats.sum(dim=1)
# #             else:
# #                 base = feats.prod(dim=1)
# #         theta = alpha * s * base  # include Qiskit's 2.0 factor via 'scale'
# #         Gate()(qdev, wires=wires, params=theta)


# # # ===============================
# # # 3) Quick usage examples
# # # ===============================
# # if __name__ == "__main__":
# #     import json

# #     # Build Z, ZZ, and a generic Pauli map (e.g., ["Y","XX"])
# #     name_z,  oplist_z  = build_z_oplist(n_wires=3, D=7, pad_mode="wrap")
# #     name_zz, oplist_zz = build_zz_oplist(n_wires=3, D=7, entanglement="linear", pad_mode="wrap")
# #     name_p,  oplist_p  = build_tiled_pauli_oplist(n_wires=4, D=6, paulis=["Y","XX"], entanglement="linear", pad_mode="wrap")

# #     print(name_z);  print(json.dumps(oplist_z[:12], indent=2))   # preview first few ops
# #     print(name_zz); print(json.dumps(oplist_zz[:16], indent=2))
# #     print(name_p);  print(json.dumps(oplist_p[:16], indent=2))

# #     # If you want to save exactly like your previous files:
# #     encoder_op_list_name_dict = {name_z: oplist_z}
# #     with open("encoder_op_list_name_dict.json", "w") as f:
# #         json.dump(encoder_op_list_name_dict, f, indent=2)

# #     # Example run on TorchQuantum device (uncomment when running in your env):
# #     # qdev = tq.QuantumDevice(n_wires=3, bsz=2, device='cpu')
# #     # x = torch.randn(2, 7)  # batch 2, D=7
# #     # run_oplist_encoder(qdev, x, oplist_zz, alpha=1.0, multi_index_rule="prod")

# # pauli_oplist_builder.py
# import math, re
# from typing import List, Dict, Any, Sequence, Optional
# from numbers import Number

# from qiskit.circuit.library import PauliFeatureMap
# from qiskit import transpile
# from qiskit.circuit import QuantumCircuit, ParameterExpression

# # ---------- small helpers ----------
# def _is_num(x) -> bool:
#     return isinstance(x, Number)

# def _is_symbolic(x) -> bool:
#     if isinstance(x, Number):
#         return False
#     # Parameter or ParameterExpression with at least one parameter
#     return hasattr(x, "parameters") and len(getattr(x, "parameters", [])) > 0

# def _ordered_params(params) -> List:
#     """Order parameters by (root, numeric suffix)."""
#     def key(p):
#         m = re.search(r"(\d+)$", p.name)
#         if m:
#             return (p.name[: m.start()], int(m.group(1)))
#         return (p.name, -1)
#     return sorted(list(params), key=key)

# def _extract_scale_from_param_expr(expr: ParameterExpression) -> float:
#     """Evaluate at all-ones to get the numeric prefactor (Qiskit default gives 2.0)."""
#     try:
#         bind_map = {p: 1.0 for p in expr.parameters}
#         val = expr.bind(bind_map)  # newer Terra
#         return float(val)
#     except Exception:
#         # Fallback: try assign_parameters
#         try:
#             val = expr.assign_parameters({p: 1.0 for p in expr.parameters})
#             return float(val)
#         except Exception:
#             # As a safe default for PauliFeatureMap (Qiskit uses 2.0 * φ), assume 2.0
#             return 2.0

# # ---------- translate ONE PauliFeatureMap(rep=1) circuit to a single-rep op skeleton ----------
# def _translate_single_rep(qc: QuantumCircuit, *, expand_h_to_u3: bool) -> List[Dict[str, Any]]:
#     """
#     Convert one decomposed/transpiled Qiskit circuit into an op skeleton:
#       - data-carrying phase (P/u1/rz) => {"input_idx":[...], "func":"u1", "wires":[i], "scale":2.0}
#       - constants (H/S/SDG/CX/U3 constants) => {"input_idx":None, "func":..., "wires":[...], ["params":[...]]}
#     Indices are LOCAL (0..n_wires-1) and will be tiled by the caller.
#     """
#     qb2i = {qb: i for i, qb in enumerate(qc.qubits)}
#     base_params = _ordered_params(qc.parameters)
#     base_pos = {p: i for i, p in enumerate(base_params)}
#     out: List[Dict[str, Any]] = []

#     for inst, qargs, _ in qc.data:
#         name = inst.name.lower()
#         wires = [qb2i[q] for q in qargs]

#         # data phases (p/phase/rz/u1 or U3(0,0,λ))
#         if name in ("p", "phase", "rz", "u1"):
#             lam = inst.params[0]
#             if _is_symbolic(lam):
#                 idxs = sorted(base_pos[p] for p in lam.parameters)
#                 scale = _extract_scale_from_param_expr(lam)
#                 out.append({"input_idx": idxs, "func": "u1", "wires": wires, "scale": scale})
#             else:
#                 out.append({"input_idx": None, "func": "u1", "wires": wires, "params": [float(lam)]})
#             continue

#         if name in ("u", "u3"):  # sometimes decomp hides a pure Z-phase as U3(0,0,λ)
#             th, ph, lam = inst.params
#             if _is_num(th) and _is_num(ph) and abs(float(th)) < 1e-12 and abs(float(ph)) < 1e-12 and _is_symbolic(lam):
#                 idxs = sorted(base_pos[p] for p in lam.parameters)
#                 scale = _extract_scale_from_param_expr(lam)
#                 out.append({"input_idx": idxs, "func": "u1", "wires": wires, "scale": scale})
#             else:
#                 # constant U3 (basis change, etc.)
#                 try:
#                     params = [float(th), float(ph), float(lam)]
#                 except Exception:
#                     params = [None, None, None]
#                 out.append({"input_idx": None, "func": "u3", "wires": wires, "params": params})
#             continue

#         # constants
#         if name == "h" and expand_h_to_u3:
#             # replace H with U3(pi/2, 0, pi)
#             out.append({"input_idx": None, "func": "u3", "wires": wires,
#                         "params": [math.pi/2, 0.0, math.pi]})
#         else:
#             out.append({"input_idx": None, "func": name, "wires": wires})

#     return out

# # ---------- pad/tiling helpers ----------
# def _map_indices_for_rep(idxs: Sequence[int], base: int, D: int, pad_mode: str):
#     """
#     Map single-rep local indices -> global indices for a given rep (offset=base).
#     Return (mapped_indices or None, force_zero_flag).
#     """
#     if D <= 0:
#         return (None, True)

#     gidxs = [base + i for i in idxs]
#     if pad_mode == "zero":
#         if any(g >= D for g in gidxs):
#             return (None, True)
#         return (gidxs, False)
#     if pad_mode == "wrap":
#         return ([g % D for g in gidxs], False)
#     if pad_mode == "repeatlast":
#         last = D - 1
#         return ([g if g < D else last for g in gidxs], False)
#     raise ValueError(f"Unknown pad_mode '{pad_mode}'")

# # ---------- public builder ----------
# def build_tiled_pauli_oplist(
#     n_wires: int,
#     D: int,
#     paulis: Sequence[str],
#     entanglement: Optional[str] = "linear",   # "linear" | "circular" | "full" | None
#     pad_mode: str = "wrap",                   # "wrap" | "repeatlast" | "zero"
#     *,
#     transpile_to: Optional[Sequence[str]] = ("u3", "cx", "u1", "sdg", "s", "h"),
#     expand_h_to_u3: bool = False,             # set True if you want H emitted as U3(pi/2,0,pi)
# ) -> (str, List[Dict[str, Any]]):
#     """
#     1) Build Qiskit PauliFeatureMap (reps=1) on n_wires, with given paulis/entanglement.
#     2) Decompose (and optionally transpile to a small gate set).
#     3) Translate one-rep circuit to an op skeleton (local indices).
#     4) Tile across reps so ceil(D/n_wires) repetitions consume all D features using pad_mode.
#     5) Return (name, oplist).

#     Notes:
#       • Z / ZZ / general Pauli are all handled (phase gates carry 'input_idx':[...], and 'scale' ≈ 2.0).
#       • If entanglement=None, you’ll only get single-qubit terms.
#     """
#     assert n_wires >= 1
#     paulis = tuple(p.upper() for p in paulis)
#     assert pad_mode in ("wrap", "repeatlast", "zero")

#     # Qiskit: "circular" == ring
#     ent_for_qiskit = None if entanglement is None else ("circular" if entanglement == "ring" else entanglement)

#     prep = PauliFeatureMap(
#         feature_dimension=n_wires,
#         reps=1,
#         paulis=list(paulis),
#         entanglement=ent_for_qiskit,
#     )
#     qc = prep.decompose()
#     if transpile_to:
#         qc = transpile(qc, basis_gates=list(transpile_to), optimization_level=0)

#     base_ops = _translate_single_rep(qc, expand_h_to_u3=expand_h_to_u3)

#     reps = max(1, math.ceil(D / n_wires)) if D > 0 else 1
#     tiled: List[Dict[str, Any]] = []
#     for r in range(reps):
#         base = r * n_wires
#         for op in base_ops:
#             if op.get("input_idx") is None:
#                 tiled.append(dict(op))
#                 continue
#             # data gate: adjust indices with pad policy
#             idxs = op["input_idx"]
#             mapped, force_zero = _map_indices_for_rep(idxs, base, D, pad_mode)
#             if force_zero:
#                 tiled.append({"input_idx": None, "func": "u1", "wires": op["wires"], "params": [0.0]})
#             else:
#                 out = {"input_idx": mapped, "func": "u1", "wires": op["wires"]}
#                 if "scale" in op:
#                     out["scale"] = float(op["scale"])
#                 tiled.append(out)

#     pauli_tag = "_".join(p.lower() for p in paulis)
#     ent_tag = "none" if entanglement is None else ("ring" if entanglement == "ring" else entanglement)
#     name = f"{n_wires}x{reps}_pauli_{pauli_tag}_{ent_tag}_{pad_mode}"
#     return name, tiled


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
