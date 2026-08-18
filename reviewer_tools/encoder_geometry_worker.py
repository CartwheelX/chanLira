#!/usr/bin/env python3
"""Compute post-encoder Hilbert-space geometry for one target-row JSON.

The worker imports QuRiFT from ``--repo-root`` and evaluates the fixed encoder
before the variational circuit. It writes exactly one CSV row, allowing the
multi-GPU launcher to use a unique output directory per data seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from reviewer_common import atomic_write_csv, seed_everything


def off_diagonal_mean(matrix: torch.Tensor) -> float:
    n = int(matrix.shape[0])
    if n < 2:
        return float("nan")
    numerator = matrix.sum() - matrix.diagonal().sum()
    return float((numerator / (n * (n - 1))).item())


def centered_kernel_label_alignment(
    kernel: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    labels = labels.to(kernel.device)
    label_kernel = (labels[:, None] == labels[None, :]).to(kernel.dtype)
    n = int(len(labels))
    center = torch.eye(n, device=kernel.device, dtype=kernel.dtype)
    center -= torch.ones((n, n), device=kernel.device, dtype=kernel.dtype) / n
    kernel_centered = center @ kernel @ center
    label_centered = center @ label_kernel @ center
    denominator = torch.linalg.norm(kernel_centered) * torch.linalg.norm(label_centered)
    if float(denominator.item()) <= 1e-12:
        return float("nan")
    return float(((kernel_centered * label_centered).sum() / denominator).item())


def effective_rank(kernel: torch.Tensor) -> float:
    symmetric = (kernel + kernel.T) / 2
    eigenvalues = torch.linalg.eigvalsh(symmetric).clamp_min(0)
    total = eigenvalues.sum()
    if float(total.item()) <= 1e-12:
        return 0.0
    probabilities = eigenvalues / total
    probabilities = probabilities[probabilities > 0]
    entropy = -(probabilities * torch.log(probabilities)).sum()
    return float(torch.exp(entropy).item())


def collect_split(dataset: Any, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    inputs: list[torch.Tensor] = []
    labels: list[int] = []
    for index in range(len(dataset[split])):
        sample = dataset[split][index]
        inputs.append(torch.as_tensor(sample["image"]))
        labels.append(int(torch.as_tensor(sample["digit"]).item()))
    return torch.stack(inputs), torch.tensor(labels, dtype=torch.long)


def build_dataset(
    qmain: Any,
    row: dict[str, Any],
    n_train: int,
    n_test: int,
    data_seed: int,
    repo_root: Path = Path("."),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    dataset_name = str(row["dataset"]).strip().lower()
    seed_everything(data_seed)

    if dataset_name == "mnist":
        import inspect

        mnist_kwargs = {
            "root": "./data",
            "train_valid_split_ratio": [0.9, 0.1],
            "digits_of_interest": [0, 1, 3, 8],
            "n_train_samples": n_train,
            "n_valid_samples": max(40, n_test // 2),
            "n_test_samples": n_test,
        }
        signature = inspect.signature(qmain.MNIST)
        if "same_n_samples_each_class" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            mnist_kwargs["same_n_samples_each_class"] = True
        dataset = qmain.MNIST(**mnist_kwargs)
        feature_dim = 16
        pool_hw = 4
    elif dataset_name == "fashion_mnist":
        from qurift.satml_fashion import build_fashion_mnist

        dataset, _ = build_fashion_mnist(
            qmain.MNIST,
            root=repo_root / "data",
            n_train=n_train,
            n_valid=max(40, n_test // 2),
            n_test=n_test,
            data_seed=data_seed,
        )
        feature_dim = 16
        pool_hw = 4
    elif dataset_name == "credit_default":
        from qurift.satml_data import prepare_credit_default

        prepared = prepare_credit_default(
            Path(str(row.get("credit_data_path", "data/credit_default/credit_default.csv.gz"))),
            n_train=int(row.get("vector_train", 200)),
            n_valid=int(row.get("vector_valid", 200)),
            n_test=int(row.get("vector_test", 2000)),
            data_seed=data_seed,
            n_components=int(row.get("credit_pca_components", row.get("n_wires", 6))),
        )
        dataset = {
            split: qmain.VectorDataset(prepared.features[split], prepared.labels[split])
            for split in ("train", "valid", "test")
        }
        feature_dim = int(row.get("credit_pca_components", row.get("n_wires", 6)))
        pool_hw = 1
    elif dataset_name == "breast_cancer_wdbc":
        from qurift.satml_wdbc import prepare_wdbc

        data_path = Path(str(row.get("wdbc_data_path", "data/wdbc/wdbc.csv.gz")))
        if not data_path.is_absolute():
            data_path = repo_root / data_path
        prepared = prepare_wdbc(
            data_path,
            n_train=int(row.get("vector_train", 160)),
            n_valid=int(row.get("vector_valid", 80)),
            n_test=int(row.get("vector_test", 329)),
            data_seed=data_seed,
            n_components=int(row.get("wdbc_pca_components", row.get("n_wires", 6))),
        )
        dataset = {
            split: qmain.VectorDataset(prepared.features[split], prepared.labels[split])
            for split in ("train", "valid", "test")
        }
        feature_dim = int(row.get("wdbc_pca_components", row.get("n_wires", 6)))
        pool_hw = 1
    else:
        kwargs: dict[str, Any] = {
            "kind": dataset_name,
            "train_samples": n_train,
            "valid_samples": max(40, n_test // 2),
            "test_samples": n_test,
            "seed": data_seed,
            "scale_to_2pi": False,
            "extra_feats": bool(row.get("extra_feats", dataset_name == "moons")),
        }
        if dataset_name == "moons":
            kwargs.update(noise=0.3, separation=0.5)
        elif dataset_name == "circles":
            kwargs.update(noise=0.3, factor=0.5)
        elif dataset_name == "blobs":
            kwargs.update(cluster_std=2.1, center_distance=3.5, n_features=4)
        else:
            raise ValueError(f"Unsupported geometry dataset: {dataset_name}")
        dataset = qmain.build_vector_dataset_dict(**kwargs)
        feature_dim = int(dataset["train"].feature_dim)
        pool_hw = 1

    train_x, train_y = collect_split(dataset, "train")
    test_x, test_y = collect_split(dataset, "test")
    train_x, train_y = train_x[:n_train], train_y[:n_train]
    test_x, test_y = test_x[:n_test], test_y[:n_test]
    return train_x, train_y, test_x, test_y, feature_dim, pool_hw


def clean_text(value: Any, default: str) -> str:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    return default if text.lower() in {"", "nan", "none", "na"} else text


def build_config(
    qmain: Any,
    row: dict[str, Any],
    feature_dim: int,
    pool_hw: int,
    device: torch.device,
    batch_size: int,
    num_classes: int,
):
    feature_map = clean_text(row["fm_kind"], "z").lower()
    repetitions = int(row["reps"])
    mapper: dict[str, Any] = {"fm_kind": feature_map}
    pad_mode = clean_text(row.get("pad_mode"), "wrap")
    fm_entanglement = clean_text(row.get("fm_ent"), "linear")
    fm_operation = clean_text(row.get("fm_op"), "cx")
    angle_scale = float(row.get("feature_angle_scale", 1.0))

    if feature_map == "z":
        mapper.update(fm_z_reps=repetitions, fm_z_pad_mode=pad_mode, fm_z_alpha=angle_scale)
    elif feature_map == "zz":
        mapper.update(
            fm_zz_reps=repetitions,
            fm_zz_pad_mode=pad_mode,
            fm_zz_entanglement=fm_entanglement,
            fm_zz_alpha=angle_scale,
        )
    elif feature_map == "eff_su2":
        mapper.update(
            fm_eff_reps=repetitions,
            fm_eff_pad_mod=pad_mode,
            fm_eff_ent_kind=fm_entanglement,
            fm_eff_twoq_op=fm_operation,
            fm_eff_alpha=angle_scale,
        )
    else:
        raise ValueError(f"Unsupported feature map for geometry: {feature_map}")

    return qmain.QFCConfig(
        n_wires=int(row["n_wires"]),
        depth=int(row.get("depth", 2)),
        n_random_ops=0,
        batch_size=batch_size,
        device=str(device),
        feature_dim=feature_dim,
        pool_hw=pool_hw,
        num_classes=num_classes,
        qlayer_ent_kind=clean_text(row.get("ql_ent"), "linear"),
        qlayer_twoq_op=clean_text(row.get("ql_op"), "crz"),
        qlayer_ent_trainable=False,
        qlayer_ent_wire_reverse=False,
        **mapper,
    )


def encode_states(
    qmain: Any,
    config: Any,
    inputs: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, str]:
    model = qmain.QFCModel(config).to(device).eval()
    operation_list = getattr(model, "func_list", None)
    operation_count = len(operation_list) if isinstance(operation_list, list) else 0
    operation_signature = hashlib.sha256(
        json.dumps(operation_list, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    states: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size].to(device).float()
            if batch.dim() >= 3:
                batch = torch.pi * torch.tanh(batch) / 2.0
            features = model._prep_features(batch, config.pool_hw)
            quantum_device = qmain.tq.QuantumDevice(
                n_wires=config.n_wires,
                bsz=len(batch),
                device=device,
                record_op=True,
            )
            model.encoder(quantum_device, features)
            if operation_count == 0:
                operation_count = len(getattr(quantum_device, "op_history", []))
            states.append(quantum_device.get_states_1d().detach().cpu())

    return torch.cat(states, dim=0), int(operation_count), operation_signature


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-json", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    row = json.loads(args.row_json.read_text(encoding="utf-8"))
    data_seed = int(row.get("data_seed", row.get("seed", 43)))
    seed_everything(data_seed)

    os.environ.setdefault("QURIFT_DISABLE_DEBUG_EXPORTS", "1")
    os.environ.setdefault("QURIFT_DISABLE_CIRCUIT_EXPORTS", "1")
    sys.path[:0] = [
        str(args.repo_root.resolve()),
        str((args.repo_root / "experiments").resolve()),
    ]
    import qurift_main as qmain

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    train_x, train_y, test_x, test_y, feature_dim, pool_hw = build_dataset(
        qmain,
        row,
        args.n_train,
        args.n_test,
        data_seed,
        args.repo_root.resolve(),
    )
    num_classes = int(torch.cat([train_y, test_y]).max().item()) + 1
    config = build_config(
        qmain,
        row,
        feature_dim,
        pool_hw,
        device,
        args.batch_size,
        num_classes,
    )

    train_states, operation_count, operation_signature = encode_states(
        qmain, config, train_x, args.batch_size, device
    )
    test_states, test_operation_count, test_operation_signature = encode_states(
        qmain, config, test_x, args.batch_size, device
    )
    if operation_count != test_operation_count or operation_signature != test_operation_signature:
        raise RuntimeError("Encoder operation list changed between train and test evaluation")

    all_states = torch.cat([train_states, test_states], dim=0)
    all_labels = torch.cat([train_y, test_y], dim=0)
    fidelity_kernel = torch.abs(all_states @ all_states.conj().T).pow(2).float()
    n_train = int(len(train_states))
    train_train = fidelity_kernel[:n_train, :n_train]
    train_test = fidelity_kernel[:n_train, n_train:]
    test_test = fidelity_kernel[n_train:, n_train:]

    same_class = all_labels[:, None] == all_labels[None, :]
    diagonal = torch.eye(len(all_labels), dtype=torch.bool)
    within_values = fidelity_kernel[same_class & ~diagonal]
    between_values = fidelity_kernel[~same_class]
    within_mean = float(within_values.mean().item())
    between_mean = float(between_values.mean().item())
    mmd2 = (
        off_diagonal_mean(train_train)
        + off_diagonal_mean(test_test)
        - 2.0 * float(train_test.mean().item())
    )
    state_signature = hashlib.sha256(
        all_states[: min(8, len(all_states))].numpy().tobytes()
    ).hexdigest()

    record = {
        **row,
        "data_seed": data_seed,
        "n_train": n_train,
        "n_test": int(len(test_states)),
        "encoder_operation_count": operation_count,
        "encoder_operation_signature": operation_signature,
        "state_signature": state_signature,
        "train_train_similarity": off_diagonal_mean(train_train),
        "train_test_similarity": float(train_test.mean().item()),
        "test_test_similarity": off_diagonal_mean(test_test),
        "within_class_similarity": within_mean,
        "between_class_similarity": between_mean,
        "class_similarity_gap": within_mean - between_mean,
        "mmd2_train_test": mmd2,
        "kernel_label_alignment": centered_kernel_label_alignment(
            fidelity_kernel, all_labels
        ),
        "effective_rank": effective_rank(fidelity_kernel),
        "geometry_kernel": "pure-state Hilbert-Schmidt/fidelity kernel",
        "geometry_stage": "immediately after fixed encoder; before variational circuit",
    }
    atomic_write_csv(pd.DataFrame([record]), args.out)
    print(f"[OK] {args.out.resolve()}")


if __name__ == "__main__":
    main()
