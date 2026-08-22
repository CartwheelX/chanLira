#!/usr/bin/env python3
"""Evaluate retained LiRA reference checkpoints under one frozen noise oracle."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
REVIEWER = ROOT / "reviewer_tools"
for path in (ROOT, REVIEWER):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reviewer_tools.qurift_lira_attack import (
    attack_scores,
    cell_id,
    load_context,
    reference_checkpoint_path,
    reference_distribution,
    reference_path,
    tensor_fingerprint,
    true_class_log_odds,
)
from reviewer_tools.qurift_qiskit_bridge import (
    counts_to_z_expectations,
    load_backend_noise_snapshot,
    run_aer_counts,
    transpile_for_backend,
)
from reviewer_tools.qurift_target_loader import (
    apply_classical_head,
    build_qiskit_circuits,
    instantiate_model,
    load_saved_model,
    resolve_target_paths,
)
from reviewer_tools.reviewer_common import (
    atomic_write_csv,
    atomic_write_json,
    cross_fitted_threshold_metrics,
    stable_seed,
    stratified_bootstrap_auc,
    tpr_at_resolvable_fpr,
)


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ints(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def parse_modes(text: str) -> list[str]:
    modes = [value.strip() for value in text.split(",") if value.strip()]
    if not modes or not set(modes).issubset({"ideal_shot", "noisy_shot"}):
        raise ValueError("--modes must contain ideal_shot and/or noisy_shot")
    return list(dict.fromkeys(modes))


def model_probabilities(
    model: torch.nn.Module,
    *,
    qmain: Any,
    row: Any,
    config: Any,
    samples: Any,
    device: torch.device,
    snapshot: Any,
    modes: list[str],
    simulator_seeds: list[int],
    shots: int,
    transpiler_seed: int,
    optimization_level: int,
    batch_size: int,
    seed_namespace: int,
    aer_max_parallel_threads: int | None,
) -> dict[str, np.ndarray]:
    circuits, _ = build_qiskit_circuits(
        qmain,
        model,
        "qnn",
        config,
        samples,
        device=device,
        batch_size=batch_size,
    )
    transpiled = transpile_for_backend(
        circuits,
        backend=None,
        basis_gates=(snapshot.metadata.basis_gates or snapshot.metadata.noise_basis_gates),
        coupling_map=snapshot.metadata.coupling_map,
        optimization_level=optimization_level,
        seed_transpiler=transpiler_seed,
    )
    output: dict[str, np.ndarray] = {}
    for mode in modes:
        values = []
        noise_model = snapshot.noise_model if mode == "noisy_shot" else None
        for simulator_seed in simulator_seeds:
            counts = run_aer_counts(
                transpiled,
                shots=shots,
                seed_simulator=(
                    int(simulator_seed) * 1_000_003 + int(seed_namespace) * 1_009
                ),
                noise_model=noise_model,
                max_parallel_threads=aer_max_parallel_threads,
            )
            expectations = np.stack(
                [counts_to_z_expectations(item, int(config.n_wires)) for item in counts]
            )
            measured = torch.tensor(expectations, dtype=torch.float32, device=device)
            with torch.no_grad():
                probabilities = apply_classical_head(model, measured).detach().cpu().numpy()
            values.append(probabilities.astype(np.float32))
        output[mode] = np.stack(values)
    return output


def load_reference_metadata(
    reference_dir: Path, structural_cell: str, expected: int, fingerprint: str
) -> tuple[np.ndarray, list[Path]]:
    inclusion_rows = []
    checkpoints = []
    for reference_id in range(expected):
        score_path = reference_path(reference_dir, structural_cell, reference_id)
        checkpoint = reference_checkpoint_path(reference_dir, structural_cell, reference_id)
        if not score_path.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing score/checkpoint for reference {reference_id}: "
                f"{score_path}, {checkpoint}"
            )
        with np.load(score_path, allow_pickle=False) as saved:
            observed_fingerprint = str(saved["candidate_fingerprint"])
            if observed_fingerprint != fingerprint:
                raise ValueError(
                    f"Reference candidate fingerprint mismatch in {score_path}"
                )
            inclusion_rows.append(saved["inclusion"].astype(bool))
        checkpoint_payload = torch_load(checkpoint)
        if str(checkpoint_payload.get("candidate_fingerprint")) != fingerprint:
            raise ValueError(f"Reference checkpoint fingerprint mismatch: {checkpoint}")
        checkpoints.append(checkpoint)
    inclusion = np.stack(inclusion_rows)
    if not np.all(inclusion.sum(axis=0) == expected // 2):
        raise ValueError("Reference checkpoint bank is not record-balanced")
    return inclusion, checkpoints


def reference_cache_key(protocol: dict[str, Any]) -> str:
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def load_reference_cache(
    cache_path: Path,
    metadata_path: Path,
    *,
    expected: dict[str, Any],
    modes: list[str],
    simulator_seeds: list[int],
    num_references: int,
    candidate_count: int,
) -> dict[str, np.ndarray] | None:
    if not cache_path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Reference-oracle cache protocol mismatch: {mismatches}")
    with np.load(cache_path, allow_pickle=False) as saved:
        output = {}
        expected_shape = (len(simulator_seeds), num_references, candidate_count)
        for mode in modes:
            key = f"scores_{mode}"
            if key not in saved.files or saved[key].shape != expected_shape:
                raise ValueError(
                    f"Invalid reference-oracle cache array {key}: "
                    f"expected {expected_shape}"
                )
            output[mode] = saved[key].astype(np.float32)
    observed_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    if observed_hash != metadata.get("cache_sha256"):
        raise ValueError(
            f"Reference-oracle cache hash mismatch: {cache_path.resolve()}"
        )
    return output


def save_reference_cache(
    cache_path: Path,
    metadata_path: Path,
    *,
    scores: dict[str, np.ndarray],
    sample_ids: list[str],
    metadata: dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_ids=np.asarray(sample_ids),
            **{f"scores_{mode}": values for mode, values in scores.items()},
        )
    temporary.replace(cache_path)
    cache_metadata = {
        **metadata,
        "cache": str(cache_path.resolve()),
        "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "real_hardware_execution": False,
    }
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(
        json.dumps(cache_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)


def score_target(args: argparse.Namespace) -> None:
    output = args.out_dir / "target_scores" / f"{args.target_id}.csv"
    metadata_path = args.out_dir / "metadata" / f"{args.target_id}.json"
    snapshot_manifest_sha256 = hashlib.sha256(
        (args.snapshot / "snapshot_manifest.json").read_bytes()
    ).hexdigest()
    modes = parse_modes(args.modes)
    simulator_seeds = parse_ints(args.simulator_seeds)
    expected_resume = {
        "snapshot": str(args.snapshot.resolve()),
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "modes": modes,
        "simulator_seeds": simulator_seeds,
        "shots": int(args.shots),
        "num_reference_models": int(args.num_references),
        "transpiler_seed": int(args.transpiler_seed),
        "optimization_level": int(args.optimization_level),
        "aer_max_parallel_threads": int(args.aer_max_parallel_threads),
    }
    if args.resume and output.is_file() and output.stat().st_size > 0:
        if not metadata_path.is_file():
            raise RuntimeError(f"Cannot validate resumed noisy LiRA output: {metadata_path}")
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (previous.get(key), expected)
            for key, expected in expected_resume.items()
            if previous.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                f"Noisy LiRA resume protocol mismatch for {args.target_id}: {mismatches}"
            )
        sample_payloads_ready = True
        for mode in modes:
            for simulator_seed in simulator_seeds:
                sample_path = args.out_dir / "sample_scores" / (
                    f"{args.target_id}_{mode}_sim{simulator_seed}.npz"
                )
                if not sample_path.is_file() or sample_path.stat().st_size == 0:
                    sample_payloads_ready = False
                    break
                try:
                    with np.load(sample_path, allow_pickle=False) as saved:
                        if not {
                            "sample_ids", "membership", "labels", "probabilities",
                            "observed_log_odds",
                        }.issubset(saved.files):
                            sample_payloads_ready = False
                            break
                except (OSError, ValueError):
                    sample_payloads_ready = False
                    break
            if not sample_payloads_ready:
                break
        if sample_payloads_ready:
            print(f"[SKIP] noisy LiRA target exists: {output.resolve()}")
            return
        print(
            f"[REBUILD] noisy LiRA payloads use an older/incomplete schema: "
            f"{args.target_id}"
        )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    qmain, row, _, config, samples = load_context(
        args.repo_root.resolve(), args.targets, args.target_id, device
    )
    snapshot = load_backend_noise_snapshot(args.snapshot, require_noise=True)
    structural_cell = cell_id(row)
    fingerprint = tensor_fingerprint(samples.inputs, samples.labels)
    inclusion, reference_checkpoints = load_reference_metadata(
        args.reference_dir, structural_cell, args.num_references, fingerprint
    )

    target_model, _ = instantiate_model(qmain, row, config, device)
    target_model_path, _ = resolve_target_paths(row, args.run_root)
    load_saved_model(target_model, target_model_path, device)
    target_probabilities = model_probabilities(
        target_model,
        qmain=qmain,
        row=row,
        config=config,
        samples=samples,
        device=device,
        snapshot=snapshot,
        modes=modes,
        simulator_seeds=simulator_seeds,
        shots=args.shots,
        transpiler_seed=args.transpiler_seed,
        optimization_level=args.optimization_level,
        batch_size=args.qiskit_batch_size,
        seed_namespace=900_000,
        aer_max_parallel_threads=args.aer_max_parallel_threads,
    )
    labels = samples.labels.numpy().astype(int)
    membership = samples.membership.numpy().astype(int)
    cache_protocol = {
        "schema_version": 1,
        "structural_cell": structural_cell,
        "candidate_fingerprint": fingerprint,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "modes": modes,
        "simulator_seeds": simulator_seeds,
        "shots": int(args.shots),
        "num_reference_models": int(args.num_references),
        "transpiler_seed": int(args.transpiler_seed),
        "optimization_level": int(args.optimization_level),
        "aer_max_parallel_threads": int(args.aer_max_parallel_threads),
    }
    reference_checkpoint_sha256 = [
        file_sha256(path) for path in reference_checkpoints
    ]
    cache_validation = {
        **cache_protocol,
        "reference_checkpoint_sha256": reference_checkpoint_sha256,
    }
    cache_token = reference_cache_key(cache_protocol)
    cache_path = args.out_dir / "reference_cache" / (
        f"{structural_cell}_{cache_token}.npz"
    )
    cache_metadata_path = cache_path.with_suffix(".json")
    reference_scores = load_reference_cache(
        cache_path,
        cache_metadata_path,
        expected=cache_validation,
        modes=modes,
        simulator_seeds=simulator_seeds,
        num_references=args.num_references,
        candidate_count=len(labels),
    )
    reference_cache_reused = reference_scores is not None
    if reference_scores is not None:
        print(f"[CACHE] noisy reference oracle -> {cache_path.resolve()}", flush=True)
    else:
        reference_scores = {
            mode: np.empty(
                (len(simulator_seeds), args.num_references, len(labels)),
                dtype=np.float32,
            )
            for mode in modes
        }
        for reference_id, checkpoint_path in enumerate(reference_checkpoints):
            model, _ = instantiate_model(qmain, row, config, device)
            saved = torch_load(checkpoint_path)
            model.load_state_dict(saved["state_dict"], strict=True)
            model.to(device).eval()
            probabilities = model_probabilities(
                model,
                qmain=qmain,
                row=row,
                config=config,
                samples=samples,
                device=device,
                snapshot=snapshot,
                modes=modes,
                simulator_seeds=simulator_seeds,
                shots=args.shots,
                transpiler_seed=args.transpiler_seed,
                optimization_level=args.optimization_level,
                batch_size=args.qiskit_batch_size,
                seed_namespace=reference_id,
                aer_max_parallel_threads=args.aer_max_parallel_threads,
            )
            for mode in modes:
                for seed_index in range(len(simulator_seeds)):
                    reference_scores[mode][seed_index, reference_id] = true_class_log_odds(
                        probabilities[mode][seed_index], labels
                    )
            print(
                f"[{args.target_id}] noisy reference "
                f"{reference_id + 1}/{args.num_references}",
                flush=True,
            )
        save_reference_cache(
            cache_path,
            cache_metadata_path,
            scores=reference_scores,
            sample_ids=samples.sample_ids,
            metadata=cache_validation,
        )
        print(f"[CACHE] saved noisy reference oracle -> {cache_path.resolve()}")

    rows = []
    sample_dir = args.out_dir / "sample_scores"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        for seed_index, simulator_seed in enumerate(simulator_seeds):
            observed = true_class_log_odds(
                target_probabilities[mode][seed_index], labels
            )
            distribution = reference_distribution(
                reference_scores[mode][seed_index], inclusion
            )
            attacks = attack_scores(observed, distribution)
            sample_path = sample_dir / (
                f"{args.target_id}_{mode}_sim{simulator_seed}.npz"
            )
            with sample_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    membership=membership.astype(np.uint8),
                    labels=labels,
                    sample_ids=np.asarray(samples.sample_ids),
                    probabilities=target_probabilities[mode][seed_index].astype(
                        np.float32
                    ),
                    observed_log_odds=observed.astype(np.float32),
                    **{name: score.astype(np.float32) for name, score in attacks.items()},
                )
            for attack, score in attacks.items():
                auc = float(roc_auc_score(membership, score))
                low, high, valid = stratified_bootstrap_auc(
                    membership,
                    score,
                    args.bootstrap,
                    stable_seed(args.seed, args.target_id, mode, simulator_seed, attack),
                )
                tpr5, attained5 = tpr_at_resolvable_fpr(membership, score, 0.05)
                tpr10, attained10 = tpr_at_resolvable_fpr(membership, score, 0.10)
                record = {
                    "target_id": args.target_id,
                    "structural_cell_id": row.get("structural_cell_id", structural_cell),
                    "fm_kind": row.get("fm_kind", ""),
                    "reps": int(float(row.get("reps", 0))),
                    "depth": int(float(row.get("depth", 0))),
                    "model_seed": int(float(row.get("model_seed", 0))),
                    "data_seed": int(float(row.get("data_seed", 0))),
                    "mode": mode,
                    "queries": 1,
                    "shots": int(args.shots),
                    "total_shots_per_target_record": int(args.shots),
                    "simulator_seed": int(simulator_seed),
                    "attack": attack,
                    "auc": auc,
                    "auc_record_ci95_low": low,
                    "auc_record_ci95_high": high,
                    "valid_record_bootstrap_replicates": valid,
                    "tpr_at_0_05_fpr": tpr5,
                    "attained_fpr_for_0_05": attained5,
                    "tpr_at_0_10_fpr": tpr10,
                    "attained_fpr_for_0_10": attained10,
                    "n_member": int((membership == 1).sum()),
                    "n_nonmember": int((membership == 0).sum()),
                    "num_reference_models": int(args.num_references),
                    "reference_calibration_circuit_shots": int(
                        args.num_references * len(labels) * args.shots
                    ),
                    "reference_execution_reused": bool(reference_cache_reused),
                    "reference_cache": str(cache_path.resolve()),
                    "target_attack_circuit_shots": int(len(labels) * args.shots),
                    "calibration_timestamp": snapshot.metadata.calibration_timestamp,
                    "backend_name": snapshot.metadata.resolved_backend_name,
                    "snapshot_manifest_sha256": snapshot_manifest_sha256,
                    "sample_score_file": str(sample_path.resolve()),
                }
                record.update(
                    cross_fitted_threshold_metrics(
                        membership,
                        score,
                        5,
                        stable_seed(args.seed, args.target_id, mode, simulator_seed, attack, "threshold"),
                    )
                )
                rows.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(pd.DataFrame(rows), output)
    atomic_write_json(
        {
            "target_id": args.target_id,
            **expected_resume,
            "backend": asdict(snapshot.metadata),
            "reference_protocol": "same frozen finite-shot/noise oracle as target",
            "reference_cache": str(cache_path.resolve()),
            "reference_cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            "reference_checkpoint_sha256": reference_checkpoint_sha256,
            "reference_execution_reused": bool(reference_cache_reused),
            "real_hardware_execution": False,
        },
        metadata_path,
    )
    print(f"[OK] noisy LiRA -> {output.resolve()}")


def aggregate(args: argparse.Namespace) -> None:
    paths = sorted((args.out_dir / "target_scores").glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No noisy LiRA target scores under {args.out_dir}")
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    atomic_write_csv(raw, args.out_dir / "noisy_lira_raw.csv")
    summary = (
        raw.groupby(
            ["target_id", "structural_cell_id", "fm_kind", "reps", "depth", "model_seed", "mode", "shots", "attack"],
            dropna=False,
        ).auc.agg(["count", "mean", "std"]).reset_index()
        .rename(columns={"count": "n_simulator_seeds", "mean": "mean_auc", "std": "sd_simulator"})
    )
    atomic_write_csv(summary, args.out_dir / "noisy_lira_summary.csv")
    targets = pd.read_csv(args.targets)
    for metadata_path in sorted((args.out_dir / "metadata").glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        target_id = str(metadata.get("target_id", metadata_path.stem))
        matched_rows = targets[targets["target_id"].astype(str).eq(target_id)]
        if len(matched_rows) != 1:
            raise ValueError(f"Cannot resolve target metadata row for {target_id}")
        structural_cell = cell_id(matched_rows.iloc[0].to_dict())
        checkpoints = [
            reference_checkpoint_path(args.reference_dir, structural_cell, reference_id)
            for reference_id in range(int(metadata["num_reference_models"]))
        ]
        if any(not path.is_file() for path in checkpoints):
            raise FileNotFoundError(
                f"Cannot bind cache provenance; reference checkpoint missing for {target_id}"
            )
        checkpoint_hashes = [file_sha256(path) for path in checkpoints]
        cache_path = Path(str(metadata["reference_cache"]))
        cache_metadata_path = cache_path.with_suffix(".json")
        if not cache_path.is_file() or not cache_metadata_path.is_file():
            raise FileNotFoundError(f"Missing reference cache provenance for {target_id}")
        cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        if file_sha256(cache_path) != cache_metadata.get("cache_sha256"):
            raise ValueError(f"Reference cache hash mismatch for {target_id}")
        existing_hashes = cache_metadata.get("reference_checkpoint_sha256")
        if existing_hashes not in (None, checkpoint_hashes):
            raise ValueError(f"Reference checkpoint hash mismatch for {target_id}")
        cache_metadata["reference_checkpoint_sha256"] = checkpoint_hashes
        atomic_write_json(cache_metadata, cache_metadata_path)
        metadata["reference_checkpoint_sha256"] = checkpoint_hashes
        atomic_write_json(metadata, metadata_path)
    print(f"[OK] noisy LiRA targets={len(paths)} rows={len(raw)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("reviewer_runs"))
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--target-id", default=None)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--num-references", type=int, default=16)
    parser.add_argument("--modes", default="ideal_shot,noisy_shot")
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--simulator-seeds", default="0,1,2,3,4")
    parser.add_argument("--transpiler-seed", type=int, default=2026)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--qiskit-batch-size", type=int, default=16)
    parser.add_argument("--aer-max-parallel-threads", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.targets = args.targets.resolve()
    args.run_root = args.run_root.resolve()
    args.reference_dir = args.reference_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.snapshot = args.snapshot.resolve()
    if args.aer_max_parallel_threads < 1:
        raise SystemExit("--aer-max-parallel-threads must be positive")
    if args.aggregate:
        aggregate(args)
    elif args.target_id:
        score_target(args)
    else:
        raise SystemExit("Specify --target-id or --aggregate")


if __name__ == "__main__":
    main()
