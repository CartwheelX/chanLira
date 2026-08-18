#!/usr/bin/env python3
"""Optional stochastic label-only boundary pilot with explicit shot accounting."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reviewer_tools.qurift_label_only_boundary import choose_evaluation_indices, vector_norm
from reviewer_tools.qurift_qiskit_bridge import (
    counts_to_z_expectations,
    load_backend_noise_snapshot,
    run_aer_counts,
    transpile_for_backend,
)
from reviewer_tools.qurift_target_loader import (
    apply_classical_head,
    build_config,
    build_dataset,
    build_qiskit_circuits,
    import_qurift_main,
    instantiate_model,
    load_saved_model,
    read_target_row,
    resolve_target_paths,
    sample_dataset_split,
    select_member_nonmember_samples,
)
from reviewer_tools.reviewer_common import (
    atomic_write_csv,
    atomic_write_json,
    stable_seed,
    stratified_bootstrap_auc,
    tpr_at_resolvable_fpr,
)


class NoisyLabelOracle:
    """Probability-returning simulator internally, class-label-only externally."""

    def __init__(
        self,
        *,
        qmain,
        model,
        config,
        device,
        snapshot,
        shots: int,
        simulator_seed: int,
        transpiler_seed: int,
        optimization_level: int,
        qiskit_batch_size: int,
    ):
        self.qmain = qmain
        self.model = model
        self.config = config
        self.device = device
        self.snapshot = snapshot
        self.shots = int(shots)
        self.simulator_seed = int(simulator_seed)
        self.transpiler_seed = int(transpiler_seed)
        self.optimization_level = int(optimization_level)
        self.qiskit_batch_size = int(qiskit_batch_size)
        self.query_points = 0
        self.circuit_shots = 0
        self.call_batches = 0

    def query(self, raw_inputs: torch.Tensor) -> torch.Tensor:
        if len(raw_inputs) == 0:
            return torch.empty(0, dtype=torch.long)
        samples = SimpleNamespace(
            inputs=raw_inputs,
            labels=torch.zeros(len(raw_inputs), dtype=torch.long),
        )
        circuits, _ = build_qiskit_circuits(
            self.qmain,
            self.model,
            "qnn",
            self.config,
            samples,
            device=self.device,
            batch_size=self.qiskit_batch_size,
        )
        transpiled = transpile_for_backend(
            circuits,
            backend=None,
            basis_gates=(
                self.snapshot.metadata.basis_gates
                or self.snapshot.metadata.noise_basis_gates
            ),
            coupling_map=self.snapshot.metadata.coupling_map,
            optimization_level=self.optimization_level,
            seed_transpiler=self.transpiler_seed,
        )
        counts = run_aer_counts(
            transpiled,
            shots=self.shots,
            seed_simulator=(
                self.simulator_seed * 1_000_003 + self.call_batches
            ),
            noise_model=self.snapshot.noise_model,
        )
        expectations = np.stack(
            [
                counts_to_z_expectations(item, int(self.config.n_wires))
                for item in counts
            ]
        )
        measured = torch.tensor(expectations, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            labels = apply_classical_head(self.model, measured).argmax(dim=1).cpu()
        self.call_batches += 1
        self.query_points += len(raw_inputs)
        self.circuit_shots += len(raw_inputs) * self.shots
        return labels


def boundary_score(
    *,
    oracle: NoisyLabelOracle,
    origin: torch.Tensor,
    original_prediction: int,
    true_label: int,
    anchor_inputs: torch.Tensor,
    anchor_predictions: torch.Tensor,
    anchors: int,
    binary_steps: int,
    norm: str,
) -> dict:
    if original_prediction != true_label:
        return {
            "boundary_distance": 0.0,
            "initially_correct": False,
            "anchors_used": 0,
            "boundary_query_points": 0,
            "censored": False,
        }
    candidates = torch.nonzero(anchor_predictions != original_prediction).flatten()
    if not len(candidates):
        return {
            "boundary_distance": np.nan,
            "initially_correct": True,
            "anchors_used": 0,
            "boundary_query_points": 0,
            "censored": True,
        }
    candidate_inputs = anchor_inputs[candidates]
    distances = vector_norm(candidate_inputs - origin, norm)
    count = min(int(anchors), len(candidates))
    selected = torch.topk(distances, k=count, largest=False).indices
    directions = candidate_inputs[selected] - origin
    low = torch.zeros(count)
    high = torch.ones(count)
    before = oracle.query_points
    view_shape = (count,) + (1,) * (origin.ndim - 1)
    for _ in range(binary_steps):
        middle = (low + high) / 2
        query_inputs = origin + middle.view(view_shape) * directions
        changed = oracle.query(query_inputs) != original_prediction
        high = torch.where(changed, middle, high)
        low = torch.where(changed, low, middle)
    distances = vector_norm(high.view(view_shape) * directions, norm)
    return {
        "boundary_distance": float(distances.min()),
        "initially_correct": True,
        "anchors_used": count,
        "boundary_query_points": oracle.query_points - before,
        "censored": False,
    }


def score(args: argparse.Namespace) -> None:
    repo = args.repo_root.resolve()
    qmain = import_qurift_main(repo)
    row = read_target_row(args.targets, args.target_id)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset, feature_dim = build_dataset(qmain, row, repo)
    config = build_config(qmain, row, feature_dim, device)
    model, architecture = instantiate_model(qmain, row, config, device)
    if architecture != "qnn":
        raise NotImplementedError("Noisy label-only pilot is restricted to QNN targets")
    model_path, _ = resolve_target_paths(row, args.run_root)
    load_saved_model(model, model_path, device)
    samples = select_member_nonmember_samples(
        dataset, n_member=None, n_nonmember=None,
        selection_seed=int(float(row.get("data_seed", 43))),
    )
    anchor_inputs, _ = sample_dataset_split(
        dataset["valid"], list(range(len(dataset["valid"])))
    )
    snapshot = load_backend_noise_snapshot(args.snapshot, require_noise=True)
    snapshot_manifest_sha256 = hashlib.sha256(
        (args.snapshot / "snapshot_manifest.json").read_bytes()
    ).hexdigest()
    membership = samples.membership.numpy().astype(int)
    labels = samples.labels.numpy().astype(int)
    selected = choose_evaluation_indices(
        membership, args.n_member, args.n_nonmember,
        stable_seed(args.seed, args.target_id, "noisy_label_selection"),
    )
    all_rows = []
    metric_rows = []
    for simulator_seed in args.simulator_seeds:
        oracle = NoisyLabelOracle(
            qmain=qmain, model=model, config=config, device=device, snapshot=snapshot,
            shots=args.shots, simulator_seed=simulator_seed,
            transpiler_seed=args.transpiler_seed,
            optimization_level=args.optimization_level,
            qiskit_batch_size=args.qiskit_batch_size,
        )
        origin_predictions = oracle.query(samples.inputs[selected])
        anchor_predictions = oracle.query(anchor_inputs)
        rows = []
        for position, sample_index in enumerate(selected):
            result = boundary_score(
                oracle=oracle,
                origin=samples.inputs[sample_index : sample_index + 1],
                original_prediction=int(origin_predictions[position]),
                true_label=int(labels[sample_index]),
                anchor_inputs=anchor_inputs,
                anchor_predictions=anchor_predictions,
                anchors=args.anchors,
                binary_steps=args.binary_steps,
                norm=args.norm,
            )
            rows.append(
                {
                    "target_id": args.target_id,
                    "simulator_seed": simulator_seed,
                    "sample_id": samples.sample_ids[sample_index],
                    "membership": int(membership[sample_index]),
                    "true_label": int(labels[sample_index]),
                    "predicted_label": int(origin_predictions[position]),
                    **result,
                }
            )
            if (position + 1) % max(len(selected) // 10, 1) == 0:
                print(
                    f"[{args.target_id} sim={simulator_seed}] {position + 1}/{len(selected)} "
                    f"query_points={oracle.query_points}",
                    flush=True,
                )
        frame = pd.DataFrame(rows)
        if frame.boundary_distance.isna().any():
            raise RuntimeError(
                f"No changed-label anchor for at least one sample: {args.target_id} sim={simulator_seed}"
            )
        y = frame.membership.to_numpy(int)
        values = frame.boundary_distance.to_numpy(float)
        auc = float(roc_auc_score(y, values))
        low, high, valid = stratified_bootstrap_auc(
            y, values, args.bootstrap,
            stable_seed(args.seed, args.target_id, simulator_seed, "noisy_label_auc"),
        )
        tpr5, fpr5 = tpr_at_resolvable_fpr(y, values, 0.05)
        tpr10, fpr10 = tpr_at_resolvable_fpr(y, values, 0.10)
        metric_rows.append(
            {
                "target_id": args.target_id,
                "structural_cell_id": row.get("structural_cell_id", ""),
                "fm_kind": row.get("fm_kind", ""),
                "reps": int(float(row.get("reps", 0))),
                "depth": int(float(row.get("depth", 0))),
                "model_seed": int(float(row.get("model_seed", 0))),
                "simulator_seed": simulator_seed,
                "mode": "noisy_shot",
                "shots_per_oracle_query": args.shots,
                "attack": f"noisy_label_only_chord_boundary_{args.norm}",
                "auc": auc,
                "auc_record_ci95_low": low,
                "auc_record_ci95_high": high,
                "valid_record_bootstrap_replicates": valid,
                "tpr_at_0_05_fpr": tpr5,
                "attained_fpr_for_0_05": fpr5,
                "tpr_at_0_10_fpr": tpr10,
                "attained_fpr_for_0_10": fpr10,
                "n_member": int((y == 1).sum()),
                "n_nonmember": int((y == 0).sum()),
                "oracle_query_points": int(oracle.query_points),
                "oracle_call_batches": int(oracle.call_batches),
                "total_circuit_shots": int(oracle.circuit_shots),
                "anchors": args.anchors,
                "binary_steps": args.binary_steps,
                "stochastic_query_policy": (
                    "one returned class label per prescribed-shot query; no majority vote"
                ),
                "scope_note": (
                    "stochastic chord-boundary proxy; not a certified minimum distance"
                ),
                "backend_name": snapshot.metadata.resolved_backend_name,
                "calibration_timestamp": snapshot.metadata.calibration_timestamp,
                "snapshot_manifest_sha256": snapshot_manifest_sha256,
            }
        )
        all_rows.extend(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(
        pd.DataFrame(all_rows),
        args.out_dir / "sample_scores" / f"{args.target_id}.csv",
    )
    atomic_write_csv(
        pd.DataFrame(metric_rows),
        args.out_dir / "target_scores" / f"{args.target_id}.csv",
    )
    atomic_write_json(
        {
            "target_id": args.target_id,
            "snapshot": str(args.snapshot.resolve()),
            "snapshot_manifest_sha256": snapshot_manifest_sha256,
            "shots_per_oracle_query": int(args.shots),
            "simulator_seeds": list(args.simulator_seeds),
            "anchors": int(args.anchors),
            "binary_steps": int(args.binary_steps),
            "attacker_observation": "one class label per independent API query",
            "real_hardware_execution": False,
        },
        args.out_dir / "metadata" / f"{args.target_id}.json",
    )
    print(f"[OK] optional noisy label-only -> {args.target_id}")


def aggregate(args: argparse.Namespace) -> None:
    paths = sorted((args.out_dir / "target_scores").glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No optional noisy label-only scores under {args.out_dir}")
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    atomic_write_csv(raw, args.out_dir / "noisy_label_only_raw.csv")
    print(f"[OK] optional noisy label-only targets={len(paths)} -> {args.out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("reviewer_runs"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--target-id", default=None)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--simulator-seeds", default="0")
    parser.add_argument("--n-member", type=int, default=50)
    parser.add_argument("--n-nonmember", type=int, default=50)
    parser.add_argument("--anchors", type=int, default=8)
    parser.add_argument("--binary-steps", type=int, default=8)
    parser.add_argument("--norm", choices=["l1", "l2", "linf"], default="l2")
    parser.add_argument("--transpiler-seed", type=int, default=2026)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--qiskit-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.targets = args.targets.resolve()
    args.run_root = args.run_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.snapshot = args.snapshot.resolve()
    args.simulator_seeds = [
        int(value.strip()) for value in args.simulator_seeds.split(",") if value.strip()
    ]
    if args.aggregate:
        aggregate(args)
    elif args.target_id:
        score(args)
    else:
        raise SystemExit("Specify --target-id or --aggregate")


if __name__ == "__main__":
    main()
