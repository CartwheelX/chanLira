#!/usr/bin/env python3
"""Apply the minimum deterministic/reviewer-export patch to qurift_main.py.

The patch is idempotent and defaults to a dry run. It is designed for the
CartwheelX/QuRiFT qurift_main.py revision whose GitHub blob SHA is
`e63df98e687335d8573b2c923b8bcb8b38ea6c0a`, but uses guarded source patterns
rather than line numbers.

Changes
-------
1. Separate model and data seeds (`--model-seed`, `--data-seed`).
2. Retain `--seed` as a legacy model-seed alias.
3. Pass Efficient-SU2 repetitions into every encoder constructor.
4. Disable fixed-name debug/circuit exports during concurrent reviewer runs.
5. Export complete target metadata, validation metrics, membership convention,
   and resource counts with every attack payload.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import re
from pathlib import Path

EXPECTED_BLOB_SHA = "9c80d5076b2e6c34c15ddbfbcf06cc94b09ba127"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one occurrence, found {count}")
    return text.replace(old, new, 1)


def replace_regex_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S | re.M)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return updated


def add_parser_arguments(text: str) -> str:
    arguments = [
        ("--seed", 'type=int, default=None, help="Legacy alias for --model-seed."'),
        ("--model-seed", 'type=int, default=43, help="Model initialization/training seed."'),
        ("--data-seed", 'type=int, default=43, help="Dataset generation/subsampling seed."'),
        ("--target-id", 'type=str, default=None, help="Unique reviewer target identifier stored in exports."'),
        ("--experiment-id", 'type=str, default=None, help="Reviewer experiment identifier stored in exports."'),
        ("--learning-rate", 'type=float, default=None, help="Optional common learning rate for controlled architecture comparisons."'),
    ]
    missing = [(option, config) for option, config in arguments if f'"{option}"' not in text]
    if not missing:
        return text
    pattern = r'^(\s*)parser\.add_argument\("--run-id"[^\n]*\)\s*$'
    match = re.search(pattern, text, flags=re.M)
    if not match:
        raise RuntimeError("Could not find --run-id parser argument")
    indent = match.group(1)
    addition = match.group(0) + "\n" + "\n".join(
        f'{indent}parser.add_argument("{option}", {config})'
        for option, config in missing
    )
    return text[: match.start()] + addition + text[match.end() :]


def replace_seed_block(text: str) -> str:
    if "def _set_qurift_reviewer_seed" in text:
        return text
    patterns = [
        (
            r'''\n    # Use the supplied seed, or generate a random one when omitted\s*
    if args\.seed is None:\s*
        seed = random\.SystemRandom\(\)\.randint\(0, 2\*\*32 - 1\)\s*
    else:\s*
        seed = args\.seed\s*
    print\(f"Seed used for this run: \{seed\}"\)\s*
    random\.seed\(seed\)\s*
    np\.random\.seed\(seed\)\s*
    torch\.manual_seed\(seed\)\s*
    if torch\.cuda\.is_available\(\):\s*
        torch\.cuda\.manual_seed_all\(seed\)\s*
    torch\.backends\.cudnn\.deterministic = True\s*
    torch\.backends\.cudnn\.benchmark = False\s*\n''',
            "replace dynamic seed block",
        ),
        (
            r'''\n    if args\.dataset in \{"moons", "blobs", "circles"\}:.*?\n        torch\.backends\.cudnn\.benchmark = False\s*\n''',
            "replace hardcoded seed block",
        ),
    ]
    replacement = r'''
    model_seed = int(args.seed if args.seed is not None else args.model_seed)
    data_seed = int(args.data_seed)

    def _set_qurift_reviewer_seed(seed_value: int) -> None:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    # Dataset construction/subsampling must not vary with target-model seed.
    _set_qurift_reviewer_seed(data_seed)
'''
    for pattern, label in patterns:
        if re.search(pattern, text, flags=re.S | re.M):
            return replace_regex_once(text, pattern, replacement, label)
    raise RuntimeError(
        "replace seed block: could not find either the supported dynamic or "
        "dataset-specific seed block"
    )


def add_model_seed_reset(text: str) -> str:
    if "reviewer_train_generator" in text:
        return text
    anchor = "    use_cuda = torch.cuda.is_available()"
    insertion = '''    # Reset RNGs after dataset construction so model initialization/training
    # varies independently from the fixed data split.
    _set_qurift_reviewer_seed(model_seed)
    reviewer_train_generator = torch.Generator()
    reviewer_train_generator.manual_seed(model_seed)

'''
    return replace_once(text, anchor, insertion + anchor, "insert model seed reset")


def add_dataloader_generator(text: str) -> str:
    if "generator=reviewer_train_generator" in text:
        return text
    anchor = "            drop_last=False,"
    replacement = anchor + "\n            generator=reviewer_train_generator if is_train else None,"
    return replace_once(text, anchor, replacement, "add deterministic DataLoader generator")


def add_eff_su2_config_field(text: str) -> str:
    """Ensure QFCConfig exposes fm_eff_reps as a real dataclass field."""
    if re.search(r"^\s*fm_eff_reps:\s*int\s*=", text, flags=re.M):
        return text
    commented = re.compile(r"^(\s*)#\s*fm_eff_reps:\s*int\s*=\s*1\s*$", re.M)
    updated, count = commented.subn(r"\1fm_eff_reps: int = 1", text, count=1)
    if count == 1:
        return updated
    anchor_text = '    # Efficient SU2 **as feature map** (data-bound angles)\n'
    if anchor_text not in text:
        raise RuntimeError("Could not locate Efficient-SU2 QFCConfig section")
    return text.replace(anchor_text, anchor_text + '    fm_eff_reps: int = 1\n', 1)


def patch_eff_su2(text: str) -> str:
    if "fm_eff_reps=args.fm_eff_reps" not in text:
        anchor = '''        mapper_cfg |= dict(
            fm_eff_alpha=1.0, # not sure what to sure, need to investigate'''
        replacement = '''        mapper_cfg |= dict(
            fm_eff_reps=args.fm_eff_reps,
            fm_eff_alpha=1.0,'''
        text = replace_once(text, anchor, replacement, "map Efficient-SU2 repetitions")

    # Patch every QNN/HQNN/QCNN builder call, including multiline calls where
    # ``reps`` may already occur several arguments after ``n_wires``.
    pattern = re.compile(
        r'build_efficient_su2_oplist_qisk_new\(\n(?P<body>.*?)(?P<close>^\s*\))',
        flags=re.S | re.M,
    )
    calls = list(pattern.finditer(text))
    if not calls:
        raise RuntimeError("No Efficient-SU2 constructor calls were found")

    def add_reps(match: re.Match[str]) -> str:
        body = match.group("body")
        if re.search(r"\breps\s*=\s*cfg\.fm_eff_reps\b", body):
            return match.group(0)
        body, count = re.subn(
            r"(D=self\.D,\s*n_wires=cfg\.n_wires,)",
            r"\1 reps=cfg.fm_eff_reps,",
            body,
            count=1,
        )
        if count != 1:
            raise RuntimeError(
                "Efficient-SU2 constructor call did not contain the expected "
                "D/n_wires arguments"
            )
        return "build_efficient_su2_oplist_qisk_new(\n" + body + match.group("close")

    text = pattern.sub(add_reps, text)
    return text


def guard_debug_exports(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    export_re = re.compile(r'^(\s*)(write_oplist_py|save_oplist_py)\((.*)\)\s*$')
    for line in lines:
        match = export_re.match(line)
        if not match:
            output.append(line)
            continue
        # Already nested under the guard in an idempotent rerun.
        if output and 'QURIFT_DISABLE_DEBUG_EXPORTS' in output[-1]:
            output.append(line)
            continue
        indent = match.group(1)
        output.append(f'{indent}if not os.environ.get("QURIFT_DISABLE_DEBUG_EXPORTS"):')
        output.append(f"{indent}    {line.strip()}")
    return "\n".join(output) + ("\n" if text.endswith("\n") else "")


def guard_circuit_export(text: str) -> str:
    if "from pathlib import Path" not in text:
        text = replace_once(
            text,
            "import argparse\n",
            "import argparse\nfrom pathlib import Path\n",
            "import Path at module scope",
        )
    text = re.sub(r"^\s{8}from pathlib import Path\s*\n", "", text, count=1, flags=re.M)

    old = '    if args.model_type != "mlp_qnn":'
    new = (
        '    if args.model_type != "mlp_qnn" and not '
        'os.environ.get("QURIFT_DISABLE_CIRCUIT_EXPORTS"):'
    )
    if new not in text:
        text = replace_once(text, old, new, "guard circuit figure export")
    old_path = '        path_circuit = circuit_dir / f"{args.model_type}_{\'_\'.join(suffix)}.png"'
    if old_path in text:
        new_path = '''        safe_target_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.target_id or f"run{args.run_id}")
        path_circuit = circuit_dir / f"{safe_target_id}_ms{model_seed}_ds{data_seed}_{args.model_type}_{'_'.join(suffix)}.png"'''
        text = text.replace(old_path, new_path, 1)
    return text


def patch_learning_rate(text: str) -> str:
    if "effective_learning_rate" not in text:
        anchor = "    # Target model\n"
        insertion = '''    default_learning_rates = {
        "qnn": 5e-2,
        "mlp_qnn": 5e-2,
        "hqnn": 1e-2,
        "qcnn": 5e-2,
    }
    effective_learning_rate = float(
        args.learning_rate
        if args.learning_rate is not None
        else default_learning_rates[args.model_type]
    )

'''
        text = replace_once(text, anchor, insertion + anchor, "add controlled learning rate")
    pattern = r"(optimizer\s*=\s*optim\.Adam\(model\.parameters\(\),\s*lr=)(?:5e-2|1e-2)(\))"
    text, count = re.subn(pattern, r"\1effective_learning_rate\2", text)
    if count == 0 and "lr=effective_learning_rate" not in text:
        raise RuntimeError("Could not patch architecture-specific Adam learning rates")
    return text


RESOURCE_HELPER = r'''
def reviewer_resource_summary(model: nn.Module, cfg: QFCConfig, model_type: str) -> dict:
    """Return exact trainable-parameter counts and main-stack gate counts.

    Gate counts cover every distinct exported encoder operation list plus the
    downstream variational circuit. QCNN quanvolutional front-end operations are
    intentionally not folded into this number because the wrapper uses a
    separate patch-wise circuit whose execution count depends on preprocessing.
    """
    from collections import Counter

    parameter_counts = Counter()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lname = name.lower()
        if any(token in lname for token in ("vqc", "q_layer", "qlayer", "quantum", "theta", "pqc", "qf.", "q_filter", "quanv")):
            component = "quantum"
        elif any(token in lname for token in ("cnn", "mlp", "head", "classifier", "linear", "fc", "backbone", "fe")):
            component = "classical"
        else:
            component = "unclassified"
        parameter_counts[component] += int(parameter.numel())

    gate_counts = Counter()
    seen_lists = set()
    for module in model.modules():
        operations = getattr(module, "func_list", None)
        if not isinstance(operations, list) or id(operations) in seen_lists:
            continue
        seen_lists.add(id(operations))
        for operation in operations:
            if isinstance(operation, dict):
                gate_counts[str(operation.get("func", "unknown")).lower()] += 1

    n_wires = int(cfg.n_wires)
    depth = int(cfg.depth)
    topology = str(cfg.qlayer_ent_kind).lower()
    if topology in {"ring", "circular"}:
        entanglers_per_layer = n_wires
    elif topology == "full":
        entanglers_per_layer = n_wires * (n_wires - 1) // 2
    elif topology == "pairwise":
        entanglers_per_layer = n_wires // 2
    else:
        entanglers_per_layer = max(n_wires - 1, 0)

    gate_counts["rx"] += depth * n_wires
    gate_counts["ry"] += depth * n_wires
    gate_counts["rz"] += depth * n_wires
    gate_counts[str(cfg.qlayer_twoq_op).lower()] += depth * entanglers_per_layer

    # The parameter-matched MLP receives the same structural configuration as
    # its QNN reference, but it has no quantum execution stack.
    if model_type == "mlp_qnn":
        gate_counts.clear()

    one_qubit_names = {"rx", "ry", "rz", "h", "x", "y", "z", "s", "sdg", "t", "u", "u1", "u2", "u3", "p"}
    two_qubit_names = {"cx", "cnot", "cz", "swap", "crx", "cry", "crz", "rxx", "ryy", "rzz", "iswap"}
    return {
        "trainable_parameters_total": int(sum(parameter_counts.values())),
        "trainable_parameters_quantum": int(parameter_counts["quantum"]),
        "trainable_parameters_classical": int(parameter_counts["classical"]),
        "trainable_parameters_unclassified": int(parameter_counts["unclassified"]),
        "gate_counts": dict(sorted(gate_counts.items())),
        "quantum_gate_count_total": int(sum(gate_counts.values())),
        "quantum_one_qubit_gates": int(sum(value for gate, value in gate_counts.items() if gate in one_qubit_names)),
        "quantum_two_qubit_gates": int(sum(value for gate, value in gate_counts.items() if gate in two_qubit_names)),
        "gate_count_scope": (
            "classical parameter-matched MLP; no quantum main stack"
            if model_type == "mlp_qnn"
            else "fixed encoder plus downstream variational circuit"
        ),
        "qcnn_frontend_included": False if model_type == "qcnn" else None,
    }
'''


def add_resource_helper(text: str) -> str:
    if "def reviewer_resource_summary" in text:
        return text
    anchor = "# ----------------------------\n# Script\n# ----------------------------"
    if anchor not in text:
        raise RuntimeError("Could not find Script section anchor")
    return text.replace(anchor, RESOURCE_HELPER + "\n\n" + anchor, 1)


def patch_validation_collection(text: str) -> str:
    if "probs_va, y_va" not in text:
        anchor = '            probs_tr, y_tr = collect_probs_and_labels(dataflow, "train", model, device)\n            probs_te, y_te = collect_probs_and_labels(dataflow, "test", model, device)'
        replacement = anchor + '\n            probs_va, y_va = collect_probs_and_labels(dataflow, "valid", model, device)'
        text = replace_once(text, anchor, replacement, "collect validation probabilities")
    if "va_metrics = metrics_from_probs" not in text:
        anchor = "            tr_metrics = metrics_from_probs(probs_tr, y_tr)\n            te_metrics = metrics_from_probs(probs_te, y_te)"
        replacement = anchor + "\n            va_metrics = metrics_from_probs(probs_va, y_va)"
        text = replace_once(text, anchor, replacement, "calculate validation metrics")
    return text


def patch_payload(text: str) -> str:
    if '"membership_convention": "0=member"' in text:
        return text

    meta_pattern = r'''                "meta": \{.*?                \},\n                "stats": \{'''
    meta_replacement = r'''                "meta": {
                    "target_id": args.target_id or f"run{args.run_id}",
                    "experiment": args.experiment_id or "",
                    "dataset": args.dataset,
                    "architecture": args.model_type,
                    "model_type": args.model_type,
                    "run_id": int(args.run_id),
                    "model_seed": int(model_seed),
                    "data_seed": int(data_seed),
                    "seed": int(model_seed),
                    "membership_convention": "0=member",
                    "attack_feature_mode": args.attack_feature_mode,
                    "learning_rate": float(effective_learning_rate),
                    "n_wires": int(args.n_wires),
                    "depth": int(args.depth),
                    "ql_ent": args.qlayer_ent_kind,
                    "ql_op": args.qlayer_twoq_op,
                    "fm_kind": args.fm_kind,
                    "reps": int({"z": args.fm_z_reps, "zz": args.fm_zz_reps,
                                 "pauli": args.fm_pauli_reps, "eff_su2": args.fm_eff_reps}[args.fm_kind]),
                    "fm_ent": ({"zz": args.fm_zz_entanglement,
                                "pauli": args.fm_pauli_entanglement,
                                "eff_su2": args.fm_eff_ent_kind}.get(args.fm_kind, "none")),
                    "fm_op": (args.fm_eff_twoq_op if args.fm_kind == "eff_su2" else "none"),
                    "pad_mode": ({"z": args.fm_z_pad_mode, "zz": args.fm_zz_pad_mode,
                                  "pauli": args.fm_pauli_pad,
                                  "eff_su2": args.fm_eff_pad_mod}[args.fm_kind]),
                    "vector_train": int(args.vector_train),
                    "vector_valid": int(args.vector_valid),
                    "vector_test": int(args.vector_test),
                },
                "stats": {'''
    text = replace_regex_once(text, meta_pattern, meta_replacement, "expand attack payload metadata")

    old_target = '''                "target_metrics": {
                    "train": tr_metrics,
                    "test": te_metrics,
                }
'''
    new_target = '''                "target_metrics": {
                    "train": tr_metrics,
                    "valid": va_metrics,
                    "test": te_metrics,
                },
                "resource_counts": reviewer_resource_summary(model, cfg, args.model_type),
'''
    text = replace_once(text, old_target, new_target, "add validation/resource payload fields")

    if "args.attack_metrics_out" in text and "reviewer_export_summary" not in text:
        anchor = '            torch.save(payload, out_path)\n            print(f"[PV Generation] Saved: {out_path}")'
        replacement = '''            torch.save(payload, out_path)
            if args.attack_metrics_out:
                reviewer_export_summary = {
                    "meta": payload["meta"],
                    "target_metrics": payload["target_metrics"],
                    "resource_counts": payload["resource_counts"],
                }
                metrics_path = Path(args.attack_metrics_out)
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_path.write_text(json.dumps(reviewer_export_summary, indent=2), encoding="utf-8")
            print(f"[PV Generation] Saved: {out_path}")'''
        text = replace_once(text, anchor, replacement, "write attack metrics JSON")
    return text


def add_attack_metrics_export(text: str) -> str:
    """Add the JSON summary even when the payload metadata was patched earlier."""
    if not (
        "args.attack_metrics_out" in text
        and "reviewer_export_summary" not in text
    ):
        return text
    anchor = '            torch.save(payload, out_path)\n            print(f"[PV Generation] Saved: {out_path}")'
    replacement = '''            torch.save(payload, out_path)
            if args.attack_metrics_out:
                reviewer_export_summary = {
                    "meta": payload["meta"],
                    "target_metrics": payload["target_metrics"],
                    "resource_counts": payload["resource_counts"],
                }
                metrics_path = Path(args.attack_metrics_out)
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_path.write_text(
                    json.dumps(reviewer_export_summary, indent=2),
                    encoding="utf-8",
                )
            print(f"[PV Generation] Saved: {out_path}")'''
    return replace_once(text, anchor, replacement, "write attack metrics JSON")


def patch_source(original: str) -> str:
    patched = original
    patched = add_parser_arguments(patched)
    patched = replace_seed_block(patched)
    if "seed=seed," in patched:
        patched = patched.replace("seed=seed,", "seed=data_seed,", 1)
    elif "seed=data_seed," not in patched:
        raise RuntimeError("Could not find vector dataset seed assignment")
    patched = add_model_seed_reset(patched)
    patched = add_dataloader_generator(patched)
    patched = add_eff_su2_config_field(patched)
    patched = patch_eff_su2(patched)
    patched = guard_debug_exports(patched)
    patched = guard_circuit_export(patched)
    patched = patch_learning_rate(patched)
    patched = add_resource_helper(patched)
    patched = patch_validation_collection(patched)
    patched = patch_payload(patched)
    patched = add_attack_metrics_export(patched)
    return patched


def git_blob_sha(text: str) -> str:
    raw = text.encode("utf-8")
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def validate_patched_source(text: str) -> None:
    required_patterns = {
        "Efficient-SU2 config field": r"^\s*fm_eff_reps:\s*int\s*=",
        "model seed argument": r"--model-seed",
        "data seed argument": r"--data-seed",
        "controlled learning rate": r"effective_learning_rate",
        "membership convention": r"membership_convention",
        "validation metrics": r'"valid": va_metrics',
        "resource counts": r"reviewer_resource_summary",
    }
    missing = [label for label, pattern in required_patterns.items() if not re.search(pattern, text, flags=re.M)]
    if missing:
        raise RuntimeError(f"Patched source failed validation: {missing}")
    builder_count = text.count("reps=cfg.fm_eff_reps")
    if builder_count < 3:
        raise RuntimeError(
            f"Expected at least three Efficient-SU2 builder calls to receive repetitions; found {builder_count}"
        )
    compile(text, "qurift_main.py.reviewed", "exec")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=Path("experiments/qurift_main.py"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--diff-out",
        type=Path,
        default=Path("reviewer_results/qurift_main_reviewer.patch"),
    )
    args = parser.parse_args()

    original = args.file.read_text(encoding="utf-8")
    source_blob_sha = git_blob_sha(original)
    if source_blob_sha != EXPECTED_BLOB_SHA:
        print(
            f"[WARN] Source git-blob SHA is {source_blob_sha}, expected {EXPECTED_BLOB_SHA}. "
            "The guarded source patterns will still be checked; inspect the dry-run diff carefully."
        )
    patched = patch_source(original)
    validate_patched_source(patched)
    if patched == original:
        print("[OK] File already appears patched; no changes required.")
        return

    diff = "".join(
        difflib.unified_diff(
            original.splitlines(True),
            patched.splitlines(True),
            fromfile=str(args.file),
            tofile=str(args.file) + ".reviewer",
        )
    )
    args.diff_out.parent.mkdir(parents=True, exist_ok=True)
    args.diff_out.write_text(diff, encoding="utf-8")
    print(diff)

    if args.apply:
        backup = args.file.with_suffix(args.file.suffix + ".pre_reviewer_patch.bak")
        backup.write_text(original, encoding="utf-8")
        args.file.write_text(patched, encoding="utf-8")
        print(f"[OK] Patched {args.file}; backup={backup}")
    else:
        print("[DRY RUN] Review the diff, then rerun with --apply.")


if __name__ == "__main__":
    main()
