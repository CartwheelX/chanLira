#!/usr/bin/env python3
"""
Direct encoder-geometry audit for QuRiFT.

Computes the Hilbert-Schmidt kernel K_ij = |<psi_i|psi_j>|^2 immediately
after the fixed encoder, before the variational circuit.

Metrics:
  train-train / train-test / test-test similarity
  within-class / between-class similarity
  unbiased MMD^2(train,test)
  centered kernel-label alignment
  kernel effective rank

Requires the Efficient-SU2 repetition bug to be fixed before comparing reps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch


def offdiag_mean(x: torch.Tensor) -> float:
    n = x.shape[0]
    if n < 2:
        return float("nan")
    return float((x.sum() - x.diag().sum()).item() / (n * (n - 1)))


def centered_alignment(K: torch.Tensor, labels: torch.Tensor) -> float:
    Y = (labels[:, None] == labels[None, :]).float()
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - torch.ones((n, n), device=K.device) / n
    Kc = H @ K @ H
    Yc = H @ Y @ H
    denom = torch.linalg.norm(Kc) * torch.linalg.norm(Yc)
    return float((Kc * Yc).sum().item() / max(float(denom.item()), 1e-12))


def effective_rank(K: torch.Tensor) -> float:
    vals = torch.linalg.eigvalsh((K + K.T) / 2).clamp_min(0)
    total = vals.sum()
    if total <= 0:
        return 0.0
    p = vals / total
    p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum()).item())


def encode_states(qm, cfg, x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    model = qm.QFCModel(cfg).to(device)
    # Guard against the known bug: repetitions must be reflected in cfg.
    if cfg.fm_kind == "eff_su2" and not hasattr(cfg, "fm_eff_reps"):
        raise RuntimeError("QFCConfig lacks fm_eff_reps; apply the required fix first.")
    all_states = []
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size].to(device).float()
        if xb.dim() >= 3:
            xb = torch.pi * torch.tanh(xb) / 2.0
        feats = model._prep_features(xb, cfg.pool_hw)
        qdev = qm.tq.QuantumDevice(
            n_wires=cfg.n_wires, bsz=len(xb), device=device, record_op=False
        )
        model.encoder(qdev, feats)
        states = qdev.get_states_1d().detach().cpu()
        all_states.append(states)
    return torch.cat(all_states, dim=0)


def load_data(qm, dataset: str, n_train: int, n_test: int, seed: int):
    ds = dataset.lower()
    if ds == "mnist":
        obj = qm.MNIST(
            root="./data",
            train_valid_split_ratio=[0.9, 0.1],
            digits_of_interest=[0, 1, 3, 8],
            n_train_samples=n_train,
            n_valid_samples=max(40, n_test // 2),
            n_test_samples=n_test,
            same_n_samples_each_class=True,
        )
        pool_hw = 4
        feature_dim = 16
    else:
        kwargs = dict(
            kind=ds,
            train_samples=n_train,
            valid_samples=max(40, n_test // 2),
            test_samples=n_test,
            seed=seed,
            scale_to_2pi=False,
            extra_feats=True,
        )
        if ds == "moons":
            kwargs.update(noise=0.3, separation=0.5)
        elif ds == "circles":
            kwargs.update(noise=0.3, factor=0.5)
        elif ds == "blobs":
            kwargs.update(cluster_std=2.1, center_distance=3.5, n_features=4)
        obj = qm.build_vector_dataset_dict(**kwargs)
        pool_hw = 1
        feature_dim = obj["train"].feature_dim

    def collect(split: str):
        xs, ys = [], []
        for i in range(len(obj[split])):
            sample = obj[split][i]
            xs.append(torch.as_tensor(sample["image"]))
            ys.append(int(sample["digit"]))
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    xtr, ytr = collect("train")
    xte, yte = collect("test")
    return xtr, ytr, xte, yte, feature_dim, pool_hw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("geometry_results.csv"))
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "experiments"))
    import qurift_main as qm  # noqa: E402

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    targets = pd.read_csv(args.targets)
    rows = []

    for _, row in targets.iterrows():
        seed = int(row["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        xtr, ytr, xte, yte, feature_dim, pool_hw = load_data(
            qm, str(row["dataset"]), args.n_train, args.n_test, seed
        )

        mapper = {"fm_kind": str(row["fm_kind"]).lower()}
        reps = int(row["reps"])
        if mapper["fm_kind"] == "z":
            mapper.update(fm_z_reps=reps, fm_z_pad_mode=str(row["pad_mode"]))
        elif mapper["fm_kind"] == "zz":
            mapper.update(
                fm_zz_reps=reps,
                fm_zz_pad_mode=str(row["pad_mode"]),
                fm_zz_entanglement=str(row["fm_ent"]),
            )
        elif mapper["fm_kind"] == "eff_su2":
            mapper.update(
                fm_eff_reps=reps,
                fm_eff_pad_mod=str(row["pad_mode"]),
                fm_eff_ent_kind=str(row["fm_ent"]),
                fm_eff_twoq_op=str(row["fm_op"]),
            )

        cfg = qm.QFCConfig(
            n_wires=int(row["n_wires"]),
            depth=int(row["depth"]),
            batch_size=args.batch_size,
            device=str(device),
            feature_dim=feature_dim,
            pool_hw=pool_hw,
            num_classes=int(max(ytr.max(), yte.max()).item()) + 1,
            qlayer_ent_kind=str(row["ql_ent"]),
            qlayer_twoq_op=str(row["ql_op"]),
            qlayer_ent_trainable=False,
            qlayer_ent_wire_reverse=False,
            **mapper,
        )

        str_train = encode_states(qm, cfg, xtr, args.batch_size, device)
        str_test = encode_states(qm, cfg, xte, args.batch_size, device)
        states = torch.cat([str_train, str_test], dim=0)
        labels = torch.cat([ytr, yte], dim=0)
        K = torch.abs(states @ states.conj().T).pow(2).float()
        ntr = len(str_train)
        Ktt = K[:ntr, :ntr]
        Kte = K[:ntr, ntr:]
        Kee = K[ntr:, ntr:]

        same = labels[:, None] == labels[None, :]
        eye = torch.eye(len(labels), dtype=torch.bool)
        within = K[same & ~eye]
        between = K[~same]
        mmd2 = offdiag_mean(Ktt) + offdiag_mean(Kee) - 2 * float(Kte.mean())

        rows.append({
            "target_id": row["target_id"],
            "dataset": row["dataset"],
            "fm_kind": row["fm_kind"],
            "reps": reps,
            "n_wires": int(row["n_wires"]),
            "train_train_similarity": offdiag_mean(Ktt),
            "train_test_similarity": float(Kte.mean()),
            "test_test_similarity": offdiag_mean(Kee),
            "within_class_similarity": float(within.mean()),
            "between_class_similarity": float(between.mean()),
            "mmd2_train_test": mmd2,
            "kernel_label_alignment": centered_alignment(K, labels),
            "effective_rank": effective_rank(K),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[OK] Geometry results -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
