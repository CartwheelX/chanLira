#!/usr/bin/env python3
"""Reconstruct saved QuRiFT reviewer targets and convert their quantum stack.

Primary supported scope is the main downstream QNN stack used by QFCModel.
HQNN is supported when it exposes a classical `fe` preprocessor followed by
`encoder`, `vqc_circuit`, `measure`, and a linear/head classifier.  QCNN can be
run only in `downstream_only` scope: its quanvolutional front-end is evaluated
exactly in TorchQuantum and noise is applied to the downstream encoder/PQC.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


SYNTHETIC_DEFAULTS = {
    "moons_noise": 0.3,
    "moons_separation": 0.5,
    "circles_noise": 0.3,
    "circles_factor": 0.5,
    "blobs_cluster_std": 2.1,
    "blobs_center_distance": 3.5,
    "blobs_n_features": 4,
}


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def clean_text(value: Any, default: str = "") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none", "na"} else default


def numeric_value(row: Mapping[str, Any], key: str, default: float) -> float:
    value = row.get(key, default)
    try:
        if pd.isna(value):
            return float(default)
    except Exception:
        pass
    return float(value)


def int_value(row: Mapping[str, Any], key: str, default: int) -> int:
    return int(round(numeric_value(row, key, default)))


def set_all_seeds(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def import_qurift_main(repo_root: Path):
    repo_root = repo_root.resolve()
    script = repo_root / "experiments" / "qurift_main.py"
    if not script.exists():
        raise FileNotFoundError(f"QuRiFT driver not found: {script}")
    experiments_dir = str(script.parent)
    root_text = str(repo_root)
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module_name = "qurift_main_checkpoint4"
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_target_row(targets_csv: Path, target_id: str) -> Dict[str, Any]:
    frame = pd.read_csv(targets_csv)
    if "target_id" not in frame.columns:
        raise KeyError(f"{targets_csv} has no target_id column")
    match = frame[frame["target_id"].astype(str) == str(target_id)]
    if len(match) != 1:
        raise ValueError(f"Expected one row for target_id={target_id!r}; found {len(match)}")
    row = match.iloc[0].to_dict()
    row["target_id"] = str(target_id)
    return row


def resolve_target_paths(row: Mapping[str, Any], run_root: Path) -> Tuple[Path, Path]:
    experiment = clean_text(row.get("experiment"), "reviewer")
    target_id = clean_text(row.get("target_id"))
    directory = run_root / experiment / target_id
    return directory / "target_model.pt", directory / "target_attack_data.pt"


def build_dataset(qmain: Any, row: Mapping[str, Any], repo_root: Path):
    dataset_name = clean_text(row.get("dataset")).lower()
    data_seed = int_value(row, "data_seed", int_value(row, "seed", 43))
    set_all_seeds(data_seed)
    n_train = int_value(row, "vector_train", 200)
    n_valid = int_value(row, "vector_valid", 200)
    n_test = int_value(row, "vector_test", 200)

    if dataset_name == "mnist":
        mnist_kwargs = {
            "root": str(repo_root / "data"),
            "train_valid_split_ratio": [0.9, 0.1],
            "digits_of_interest": [0, 1, 3, 8],
            "n_test_samples": n_test,
            "n_valid_samples": n_valid,
            "n_train_samples": n_train,
        }
        signature = inspect.signature(qmain.MNIST)
        if "same_n_samples_each_class" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            mnist_kwargs["same_n_samples_each_class"] = True
        dataset = qmain.MNIST(**mnist_kwargs)
        feature_dim = int_value(row, "pool_hw", 4) ** 2
    elif dataset_name == "fashion_mnist":
        from qurift.satml_fashion import build_fashion_mnist

        dataset, _ = build_fashion_mnist(
            qmain.MNIST,
            root=repo_root / "data",
            n_train=n_train,
            n_valid=n_valid,
            n_test=n_test,
            data_seed=data_seed,
        )
        feature_dim = int_value(row, "pool_hw", 4) ** 2
    elif dataset_name == "credit_default":
        from qurift.satml_data import prepare_credit_default

        data_path = Path(clean_text(row.get("credit_data_path"), "data/credit_default/credit_default.csv.gz"))
        if not data_path.is_absolute():
            data_path = repo_root / data_path
        prepared = prepare_credit_default(
            data_path,
            n_train=n_train,
            n_valid=n_valid,
            n_test=n_test,
            data_seed=data_seed,
            n_components=int_value(row, "credit_pca_components", int_value(row, "n_wires", 6)),
        )
        dataset = {
            split: qmain.VectorDataset(prepared.features[split], prepared.labels[split])
            for split in ("train", "valid", "test")
        }
        feature_dim = int_value(row, "credit_pca_components", int_value(row, "n_wires", 6))
    elif dataset_name == "breast_cancer_wdbc":
        from qurift.satml_wdbc import prepare_wdbc

        data_path = Path(clean_text(row.get("wdbc_data_path"), "data/wdbc/wdbc.csv.gz"))
        if not data_path.is_absolute():
            data_path = repo_root / data_path
        prepared = prepare_wdbc(
            data_path,
            n_train=n_train,
            n_valid=n_valid,
            n_test=n_test,
            data_seed=data_seed,
            n_components=int_value(row, "wdbc_pca_components", int_value(row, "n_wires", 6)),
        )
        dataset = {
            split: qmain.VectorDataset(prepared.features[split], prepared.labels[split])
            for split in ("train", "valid", "test")
        }
        feature_dim = int_value(row, "wdbc_pca_components", int_value(row, "n_wires", 6))
    elif dataset_name in {"moons", "circles", "blobs"}:
        kwargs = dict(
            kind=dataset_name,
            train_samples=n_train,
            valid_samples=n_valid,
            test_samples=n_test,
            seed=data_seed,
            scale_to_2pi=bool_value(row.get("vector_scale_to_2pi", False)),
            extra_feats=bool_value(row.get("extra_feats", False)),
            noise=SYNTHETIC_DEFAULTS["moons_noise"],
            separation=SYNTHETIC_DEFAULTS["moons_separation"],
            factor=SYNTHETIC_DEFAULTS["circles_factor"],
            cluster_std=SYNTHETIC_DEFAULTS["blobs_cluster_std"],
            center_distance=SYNTHETIC_DEFAULTS["blobs_center_distance"],
            n_features=SYNTHETIC_DEFAULTS["blobs_n_features"],
        )
        if dataset_name == "moons":
            kwargs["noise"] = numeric_value(row, "moons_noise", SYNTHETIC_DEFAULTS["moons_noise"])
            kwargs["separation"] = numeric_value(row, "moons_separation", SYNTHETIC_DEFAULTS["moons_separation"])
        elif dataset_name == "circles":
            kwargs["noise"] = numeric_value(row, "circles_noise", SYNTHETIC_DEFAULTS["circles_noise"])
            kwargs["factor"] = numeric_value(row, "circles_factor", SYNTHETIC_DEFAULTS["circles_factor"])
        else:
            kwargs["cluster_std"] = numeric_value(row, "blobs_cluster_std", SYNTHETIC_DEFAULTS["blobs_cluster_std"])
            kwargs["center_distance"] = numeric_value(row, "blobs_center_distance", SYNTHETIC_DEFAULTS["blobs_center_distance"])
            kwargs["n_features"] = int_value(row, "blobs_n_features", SYNTHETIC_DEFAULTS["blobs_n_features"])
        dataset = qmain.build_vector_dataset_dict(**kwargs)
        feature_dim = int(dataset["train"].feature_dim)
    else:
        raise NotImplementedError(
            "Target loader supports MNIST, Fashion-MNIST, Credit-default, WDBC, "
            f"Moons, Circles and Blobs; got {dataset_name!r}"
        )
    return {split: dataset[split] for split in dataset}, feature_dim


def build_config(qmain: Any, row: Mapping[str, Any], feature_dim: int, device: torch.device):
    fm = clean_text(row.get("fm_kind"), "z").lower()
    reps = int_value(row, "reps", 1)
    pad = clean_text(row.get("pad_mode"), "wrap").lower()
    fm_ent = clean_text(row.get("fm_ent"), "linear").lower()
    fm_op = clean_text(row.get("fm_op"), "cx").lower()
    angle_scale = numeric_value(row, "feature_angle_scale", 1.0)
    mapper: Dict[str, Any] = {"fm_kind": fm}
    if fm == "z":
        mapper.update(fm_z_reps=reps, fm_z_alpha=angle_scale, fm_z_pad_mode=pad)
    elif fm == "zz":
        mapper.update(
            fm_zz_reps=reps,
            fm_zz_alpha=angle_scale,
            fm_zz_entanglement=fm_ent or "linear",
            fm_zz_phi="pi_minus",
            fm_zz_pad_mode=pad,
        )
    elif fm == "eff_su2":
        if not hasattr(qmain.QFCConfig, "__dataclass_fields__") or "fm_eff_reps" not in qmain.QFCConfig.__dataclass_fields__:
            raise RuntimeError(
                "QFCConfig does not expose fm_eff_reps. Apply the Checkpoint 3 QuRiFT patch first."
            )
        mapper.update(
            fm_eff_reps=reps,
            fm_eff_alpha=angle_scale,
            fm_eff_ent_kind=fm_ent or "linear",
            fm_eff_twoq_op=fm_op or "cx",
            fm_eff_pad_mod=pad,
        )
    elif fm == "pauli":
        mapper.update(
            fm_pauli_reps=reps,
            fm_pauli_alpha=angle_scale,
            fm_pauli_entanglement=fm_ent or "linear",
            fm_pauli_terms=("Z", "ZZ"),
            fm_pali_pad=pad,
        )
    else:
        raise NotImplementedError(f"Unsupported feature map: {fm}")

    dataset_name = clean_text(row.get("dataset")).lower()
    num_classes = 4 if dataset_name in {"mnist", "fashion_mnist"} else 2
    cfg = qmain.QFCConfig(
        n_wires=int_value(row, "n_wires", 4),
        depth=int_value(row, "depth", 2),
        n_random_ops=0,
        batch_size=int_value(row, "batch_size", 16),
        device=str(device),
        num_classes=num_classes,
        qlayer_ent_kind=clean_text(row.get("ql_ent"), "linear").lower(),
        qlayer_twoq_op=clean_text(row.get("ql_op"), "cx").lower(),
        qlayer_ent_trainable=True,
        qlayer_ent_wire_reverse=bool_value(row.get("ql_rev", False)),
        pool_pairs=False,
        pair_topology="ring",
        pool_hw=4 if dataset_name in {"mnist", "fashion_mnist"} else 1,
        feature_dim=int(feature_dim),
        measure_ops=None,
        **mapper,
    )
    return cfg


def instantiate_model(qmain: Any, row: Mapping[str, Any], cfg: Any, device: torch.device):
    architecture = clean_text(row.get("architecture", row.get("model_type", "qnn")), "qnn").lower()
    model_seed = int_value(row, "model_seed", int_value(row, "seed", 43))
    set_all_seeds(model_seed)
    if architecture == "qnn":
        model = qmain.QFCModel(cfg)
    elif architecture == "hqnn":
        model = qmain.HybridQNN(cfg)
    elif architecture == "qcnn":
        model = qmain.QCNN(cfg)
    else:
        raise NotImplementedError(
            f"Noisy quantum evaluation does not apply to architecture {architecture!r}."
        )
    return model.to(device), architecture


def load_saved_model(model: torch.nn.Module, model_path: Path, device: torch.device) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Saved target model not found: {model_path}")
    state = torch.load(model_path, map_location=device)
    if isinstance(state, Mapping) and "state_dict" in state and isinstance(state["state_dict"], Mapping):
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    return {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "model_path": str(model_path.resolve()),
    }


def preprocess_like_train(inputs: torch.Tensor, device: torch.device) -> torch.Tensor:
    inputs = inputs.to(device).float()
    if inputs.dim() >= 3:
        inputs = torch.pi * torch.tanh(inputs) / 2.0
    return inputs


def sample_dataset_split(dataset: Any, indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    images: List[torch.Tensor] = []
    labels: List[int] = []
    for index in indices:
        item = dataset[int(index)]
        image = item["image"]
        if not torch.is_tensor(image):
            image = torch.as_tensor(image)
        images.append(image)
        label = item["digit"]
        labels.append(int(label.item()) if torch.is_tensor(label) else int(label))
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def choose_indices(length: int, count: Optional[int], rng: np.random.Generator) -> List[int]:
    if count is None or int(count) <= 0 or int(count) >= int(length):
        return list(range(int(length)))
    return sorted(int(value) for value in rng.choice(int(length), size=int(count), replace=False))


@dataclass
class SelectedSamples:
    inputs: torch.Tensor
    labels: torch.Tensor
    membership: torch.Tensor
    split_codes: torch.Tensor
    source_indices: List[int]
    sample_ids: List[str]
    split_names: List[str]


def select_member_nonmember_samples(
    dataset: Mapping[str, Any],
    *,
    n_member: Optional[int],
    n_nonmember: Optional[int],
    selection_seed: int,
) -> SelectedSamples:
    rng = np.random.default_rng(int(selection_seed))
    train_indices = choose_indices(len(dataset["train"]), n_member, rng)
    test_indices = choose_indices(len(dataset["test"]), n_nonmember, rng)
    x_train, y_train = sample_dataset_split(dataset["train"], train_indices)
    x_test, y_test = sample_dataset_split(dataset["test"], test_indices)
    inputs = torch.cat([x_train, x_test], dim=0)
    labels = torch.cat([y_train, y_test], dim=0)
    membership = torch.cat(
        [torch.ones(len(y_train), dtype=torch.long), torch.zeros(len(y_test), dtype=torch.long)]
    )
    split_codes = torch.cat(
        [torch.zeros(len(y_train), dtype=torch.long), torch.ones(len(y_test), dtype=torch.long)]
    )
    source_indices = list(train_indices) + list(test_indices)
    split_names = ["train"] * len(train_indices) + ["test"] * len(test_indices)
    sample_ids = []
    for split, index, label in zip(split_names, source_indices, labels.tolist()):
        raw = f"{split}|{index}|{label}".encode("utf-8")
        sample_ids.append(hashlib.sha256(raw).hexdigest()[:16])
    return SelectedSamples(
        inputs=inputs,
        labels=labels,
        membership=membership,
        split_codes=split_codes,
        source_indices=source_indices,
        sample_ids=sample_ids,
        split_names=split_names,
    )


@torch.no_grad()
def exact_probabilities(
    model: torch.nn.Module,
    samples: SelectedSamples,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    output: List[torch.Tensor] = []
    for start in range(0, len(samples.labels), int(batch_size)):
        x = preprocess_like_train(samples.inputs[start : start + batch_size], device)
        log_probs = model(x)
        output.append(log_probs.exp().detach().cpu())
    return torch.cat(output, dim=0)


def _qnn_features(model: torch.nn.Module, inputs: torch.Tensor, cfg: Any) -> torch.Tensor:
    if hasattr(model, "_prep_features"):
        return model._prep_features(inputs, cfg.pool_hw)
    if inputs.dim() == 2:
        return inputs
    if inputs.dim() == 3:
        inputs = inputs.unsqueeze(1)
    if inputs.dim() == 4:
        return F.adaptive_avg_pool2d(inputs, (cfg.pool_hw, cfg.pool_hw)).view(inputs.size(0), -1)
    raise ValueError(f"Unsupported QNN input shape: {tuple(inputs.shape)}")


def quantum_features_and_scope(
    model: torch.nn.Module,
    architecture: str,
    inputs: torch.Tensor,
    cfg: Any,
) -> Tuple[torch.Tensor, str]:
    architecture = architecture.lower()
    if architecture == "qnn":
        return _qnn_features(model, inputs, cfg), "full_main_quantum_stack"
    if architecture == "hqnn":
        if not hasattr(model, "fe"):
            raise NotImplementedError("HQNN model does not expose expected `fe` preprocessor")
        return model.fe(inputs), "full_main_quantum_stack_after_classical_preprocessor"
    if architecture == "qcnn":
        if not hasattr(model, "qf"):
            raise NotImplementedError("QCNN model does not expose expected `qf` front-end")
        reshaped = inputs.view(-1, 28, 28)
        features = model.qf(reshaped).reshape(-1, 16)
        return features, "downstream_stack_only_qcnn_frontend_exact"
    raise NotImplementedError(f"Unsupported architecture: {architecture}")


def apply_classical_head(model: torch.nn.Module, measured: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "linear") and callable(model.linear):
        logits = model.linear(measured)
    elif hasattr(model, "head") and callable(model.head):
        logits = model.head(measured)
    else:
        raise NotImplementedError(
            "Could not locate QuRiFT post-measurement classifier (`linear` or `head`)."
        )
    return torch.softmax(logits, dim=1)


def _as_circuit_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


@torch.no_grad()
def build_qiskit_circuits(
    qmain: Any,
    model: torch.nn.Module,
    architecture: str,
    cfg: Any,
    samples: SelectedSamples,
    *,
    device: torch.device,
    batch_size: int,
) -> Tuple[List[Any], str]:
    """Build one measured Qiskit circuit per selected sample.

    The conversion follows the same TorchQuantum operation-history route used by
    the reference QuantumNAT script.  It requires the QuRiFT model to expose
    `encoder`, `vqc_circuit`, and `measure`.
    """
    import torchquantum as tq
    try:
        from torchquantum.plugin import (
            tq2qiskit_measurement,
            qiskit_assemble_circs,
            op_history2qiskit,
            op_history2qiskit_expand_params,
        )
    except Exception as exc:
        raise ImportError("TorchQuantum Qiskit conversion plugin is unavailable") from exc

    for attribute in ("encoder", "vqc_circuit", "measure"):
        if not hasattr(model, attribute):
            raise NotImplementedError(f"Model lacks required quantum-stack attribute {attribute!r}")

    model.eval()
    circuits: List[Any] = []
    scope = ""
    for start in range(0, len(samples.labels), int(batch_size)):
        x = preprocess_like_train(samples.inputs[start : start + batch_size], device)
        features, current_scope = quantum_features_and_scope(model, architecture, x, cfg)
        if scope and current_scope != scope:
            raise RuntimeError("Quantum execution scope changed between batches")
        scope = current_scope
        bsz = int(features.shape[0])
        qdev = tq.QuantumDevice(
            n_wires=int(cfg.n_wires),
            bsz=bsz,
            device=features.device,
            record_op=True,
        )

        model.encoder(qdev, features)
        encoder_history = list(qdev.op_history)
        qdev.reset_op_history()
        encoder_circuits = _as_circuit_list(
            op_history2qiskit_expand_params(int(cfg.n_wires), encoder_history, bsz=bsz)
        )

        model.vqc_circuit(qdev)
        qlayer_history = list(qdev.op_history)
        qdev.reset_op_history()
        qlayer_circuit = op_history2qiskit(int(cfg.n_wires), qlayer_history)
        measurement_circuit = tq2qiskit_measurement(qdev, model.measure)
        assembled = qiskit_assemble_circs(
            encoder_circuits,
            qlayer_circuit,
            measurement_circuit,
        )
        circuits.extend(_as_circuit_list(assembled))

    if len(circuits) != len(samples.labels):
        raise RuntimeError(
            f"Circuit conversion produced {len(circuits)} circuits for {len(samples.labels)} samples"
        )
    return circuits, scope


def verify_attack_payload(
    attack_path: Path,
    *,
    exact_probs: torch.Tensor,
    samples: SelectedSamples,
) -> Dict[str, Any]:
    """Return non-fatal consistency information for an existing attack payload."""
    if not attack_path.exists():
        return {"available": False, "path": str(attack_path)}
    payload = torch.load(attack_path, map_location="cpu")
    meta = payload.get("meta", {}) or {}
    result: Dict[str, Any] = {
        "available": True,
        "path": str(attack_path.resolve()),
        "payload_target_id": meta.get("target_id"),
        "payload_model_seed": meta.get("model_seed", meta.get("seed")),
        "payload_data_seed": meta.get("data_seed"),
        "payload_membership_convention": meta.get("membership_convention", "legacy_or_unspecified"),
    }
    metrics = payload.get("target_metrics", {}) or {}
    result["payload_target_metrics"] = metrics
    # Selected samples are not necessarily in the payload's shuffled order, so
    # only aggregate exact metrics can be compared safely.
    labels = samples.labels
    split = samples.split_codes
    for code, name in ((0, "train_selected"), (1, "test_selected")):
        mask = split == code
        if mask.any():
            probs = exact_probs[mask]
            labs = labels[mask]
            acc = float((probs.argmax(1) == labs).float().mean().item())
            loss = float(F.nll_loss(torch.log(probs.clamp_min(1e-12)), labs).item())
            result[name] = {"acc": acc, "loss": loss, "N": int(mask.sum().item())}
    return result


def target_row_to_jsonable(row: Mapping[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and np.isnan(value):
            value = None
        output[str(key)] = value
    return output
