#!/usr/bin/env python3
"""Class-label-only decision-boundary MIA for QuRiFT targets.

The attacker never observes probabilities, logits, losses, gradients, quantum
states, or model parameters.  It queries only the predicted class.  For each
correctly classified candidate, it selects nearby public anchor inputs that
receive a different predicted class and performs class-label-only binary search
along each input chord.  The smallest observed boundary distance is the
membership score: larger distance indicates greater decision robustness and
therefore stronger membership evidence.

This is a query-based boundary-distance baseline, but not HopSkipJump/QEBA: the
search directions are candidate-to-anchor chords and are not gradient-refined.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from qurift_target_loader import (
    build_config,
    build_dataset,
    import_qurift_main,
    instantiate_model,
    load_saved_model,
    preprocess_like_train,
    read_target_row,
    resolve_target_paths,
    sample_dataset_split,
    select_member_nonmember_samples,
)
from reviewer_common import (
    CI_RECORD,
    atomic_write_csv,
    atomic_write_json,
    cross_fitted_threshold_metrics,
    stable_seed,
    stratified_bootstrap_auc,
    tpr_at_resolvable_fpr,
    write_analysis_metadata,
)


REFERENCE_COMMIT = "af3e8146279d595389aecf8eb6e47245129d6021"


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()) or "unnamed"


@torch.no_grad()
def query_class_labels(
    model: torch.nn.Module,
    raw_inputs: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Return only argmax labels; score-valued outputs are immediately discarded."""
    model.eval()
    predictions = []
    for start in range(0, len(raw_inputs), int(batch_size)):
        inputs = preprocess_like_train(
            raw_inputs[start : start + batch_size], device
        )
        predictions.append(model(inputs).argmax(dim=1).detach().cpu())
    return torch.cat(predictions)


def vector_norm(values: torch.Tensor, norm: str) -> torch.Tensor:
    flat = values.reshape(values.shape[0], -1)
    if norm == "l1":
        return flat.abs().sum(dim=1)
    if norm == "linf":
        return flat.abs().max(dim=1).values
    return torch.linalg.vector_norm(flat, ord=2, dim=1)


def choose_evaluation_indices(
    membership: np.ndarray,
    n_member: int | None,
    n_nonmember: int | None,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for value, requested in ((1, n_member), (0, n_nonmember)):
        available = np.flatnonzero(membership == value)
        if requested is None or requested <= 0 or requested >= len(available):
            chosen = available
        else:
            chosen = np.sort(rng.choice(available, size=requested, replace=False))
        selected.append(chosen)
    return np.sort(np.concatenate(selected))


def boundary_distance_for_sample(
    *,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    pool_predictions: torch.Tensor,
    anchor_inputs: torch.Tensor,
    anchor_predictions: torch.Tensor,
    sample_index: int,
    true_label: int,
    n_anchors: int,
    binary_steps: int,
    norm: str,
    device: torch.device,
    query_batch_size: int,
) -> dict[str, Any]:
    original_prediction = int(pool_predictions[sample_index].item())
    initially_correct = original_prediction == int(true_label)
    if not initially_correct:
        return {
            "boundary_distance": 0.0,
            "initially_correct": False,
            "anchors_used": 0,
            "boundary_queries": 0,
            "censored": False,
        }

    candidate_indices = torch.nonzero(
        anchor_predictions != original_prediction, as_tuple=False
    ).flatten().cpu()
    if not len(candidate_indices):
        return {
            "boundary_distance": float("nan"),
            "initially_correct": True,
            "anchors_used": 0,
            "boundary_queries": 0,
            "censored": True,
        }
    origin = inputs[sample_index : sample_index + 1]
    candidate_inputs = anchor_inputs[candidate_indices]
    distances = vector_norm(candidate_inputs - origin, norm)
    count = min(int(n_anchors), len(candidate_indices))
    nearest_positions = torch.topk(
        distances, k=count, largest=False, sorted=True
    ).indices
    anchors = candidate_inputs[nearest_positions]
    directions = anchors - origin
    low = torch.zeros(count, dtype=torch.float32)
    high = torch.ones(count, dtype=torch.float32)
    queries = 0
    view_shape = (count,) + (1,) * (inputs.ndim - 1)
    for _ in range(int(binary_steps)):
        middle = (low + high) / 2.0
        query_inputs = origin + middle.view(view_shape) * directions
        predictions = query_class_labels(
            model,
            query_inputs,
            device=device,
            batch_size=query_batch_size,
        )
        changed = predictions != original_prediction
        high = torch.where(changed, middle, high)
        low = torch.where(changed, low, middle)
        queries += count
    boundary_vectors = high.view(view_shape) * directions
    boundary_distances = vector_norm(boundary_vectors, norm)
    return {
        "boundary_distance": float(boundary_distances.min().item()),
        "initially_correct": True,
        "anchors_used": count,
        "boundary_queries": queries,
        "censored": False,
    }


def load_target(args: argparse.Namespace):
    repo_root = args.repo_root.resolve()
    qmain = import_qurift_main(repo_root)
    row = read_target_row(args.targets, args.target_id)
    architecture = str(row.get("architecture", "qnn")).strip().lower()
    if architecture not in {"qnn", "hqnn", "qcnn"}:
        raise NotImplementedError(
            f"Label-only boundary evaluation does not support {architecture!r}"
        )
    device = torch.device(args.device)
    dataset, feature_dim = build_dataset(qmain, row, repo_root)
    config = build_config(qmain, row, feature_dim, device)
    samples = select_member_nonmember_samples(
        dataset,
        n_member=None,
        n_nonmember=None,
        selection_seed=int(float(row.get("data_seed", 43))),
    )
    model, _ = instantiate_model(qmain, row, config, device)
    model_path, _ = resolve_target_paths(row, args.run_root)
    load_saved_model(model, model_path, device)
    anchor_inputs, _ = sample_dataset_split(
        dataset["valid"], list(range(len(dataset["valid"])))
    )
    return row, model, samples, anchor_inputs, device


def score_target(args: argparse.Namespace) -> None:
    metrics_path = (
        args.out_dir / "target_scores" / f"{safe_name(args.target_id)}.csv"
    )
    if args.resume and metrics_path.exists():
        print(f"[SKIP] target score exists: {metrics_path.resolve()}")
        return
    row, model, samples, anchor_inputs, device = load_target(args)
    all_predictions = query_class_labels(
        model,
        samples.inputs,
        device=device,
        batch_size=args.query_batch_size,
    )
    anchor_predictions = query_class_labels(
        model,
        anchor_inputs,
        device=device,
        batch_size=args.query_batch_size,
    )
    membership = samples.membership.numpy().astype(int)
    labels = samples.labels.numpy().astype(int)
    evaluation_indices = choose_evaluation_indices(
        membership,
        args.n_member,
        args.n_nonmember,
        stable_seed(args.seed, args.target_id, "label_only_selection"),
    )
    rows: list[dict[str, Any]] = []
    for position, sample_index in enumerate(evaluation_indices):
        result = boundary_distance_for_sample(
            model=model,
            inputs=samples.inputs,
            pool_predictions=all_predictions,
            anchor_inputs=anchor_inputs,
            anchor_predictions=anchor_predictions,
            sample_index=int(sample_index),
            true_label=int(labels[sample_index]),
            n_anchors=args.anchors,
            binary_steps=args.binary_steps,
            norm=args.norm,
            device=device,
            query_batch_size=args.query_batch_size,
        )
        rows.append(
            {
                "target_id": args.target_id,
                "sample_id": samples.sample_ids[int(sample_index)],
                "sample_index": int(sample_index),
                "source_split": samples.split_names[int(sample_index)],
                "source_index": samples.source_indices[int(sample_index)],
                "membership": int(membership[sample_index]),
                "true_label": int(labels[sample_index]),
                "predicted_label": int(all_predictions[sample_index]),
                **result,
                "initial_label_query": 1,
                "total_label_queries": 1 + int(result["boundary_queries"]),
                "norm": args.norm,
            }
        )
        if (
            position == 0
            or position + 1 == len(evaluation_indices)
            or (position + 1) % max(len(evaluation_indices) // 20, 1) == 0
        ):
            print(
                f"[{args.target_id}] {position + 1}/{len(evaluation_indices)}",
                flush=True,
            )
    sample_frame = pd.DataFrame(rows)
    if sample_frame["boundary_distance"].isna().any():
        raise RuntimeError(
            "Boundary score contains NaN because no changed-label anchors were available"
        )
    score = sample_frame["boundary_distance"].to_numpy(float)
    y = sample_frame["membership"].to_numpy(int)
    auc = float(roc_auc_score(y, score))
    low, high, valid = stratified_bootstrap_auc(
        y,
        score,
        args.bootstrap,
        stable_seed(args.seed, args.target_id, "label_only_boundary_bootstrap"),
    )
    tpr5, attained5 = tpr_at_resolvable_fpr(y, score, 0.05)
    tpr10, attained10 = tpr_at_resolvable_fpr(y, score, 0.10)
    record = {
        "target_id": args.target_id,
        "experiment": row.get("experiment", ""),
        "dataset": row.get("dataset", ""),
        "architecture": row.get("architecture", ""),
        "role": row.get("role", ""),
        "seed": int(float(row.get("seed", row.get("model_seed", 0)))),
        "model_seed": int(float(row.get("model_seed", row.get("seed", 0)))),
        "data_seed": int(float(row.get("data_seed", 0))),
        "structural_cell_id": row.get("structural_cell_id", ""),
        "fm_kind": row.get("fm_kind", ""),
        "reps": int(float(row.get("reps", 0))),
        "depth": int(float(row.get("depth", 0))),
        "attack": f"label_only_chord_boundary_{args.norm}",
        "access": (
            "predicted class labels only; candidate labels and held-out validation anchors"
        ),
        "auc": auc,
        "auc_record_ci95_low": low,
        "auc_record_ci95_high": high,
        "valid_record_bootstrap_replicates": valid,
        "tpr_at_0_05_fpr": tpr5,
        "attained_fpr_for_0_05": attained5,
        "tpr_at_0_10_fpr": tpr10,
        "attained_fpr_for_0_10": attained10,
        "n_member": int((y == 1).sum()),
        "n_nonmember": int((y == 0).sum()),
        "anchors": args.anchors,
        "binary_steps": args.binary_steps,
        "mean_label_queries": float(sample_frame["total_label_queries"].mean()),
        "initial_accuracy": float(sample_frame["initially_correct"].mean()),
        "score_definition": (
            "minimum observed input-space distance to a changed predicted label "
            "along nearest differently-classified validation-anchor chords; initially "
            "misclassified samples receive distance zero"
        ),
        "scope_note": (
            "genuine class-label-only decision-boundary proxy; not gradient-refined "
            "HopSkipJump/QEBA and not a certified minimum boundary distance"
        ),
        "ci_method": CI_RECORD,
    }
    record.update(
        cross_fitted_threshold_metrics(
            y,
            score,
            5,
            stable_seed(args.seed, args.target_id, "label_only_boundary_crossfit"),
        )
    )
    sample_path = (
        args.out_dir / "sample_scores" / f"{safe_name(args.target_id)}.csv"
    )
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(sample_frame, sample_path)
    record["sample_score_file"] = str(sample_path.resolve())
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(pd.DataFrame([record]), metrics_path)
    print(f"[OK] label-only boundary -> {metrics_path.resolve()}")


def aggregate(args: argparse.Namespace) -> None:
    paths = sorted((args.out_dir / "target_scores").glob("*.csv"))
    if not paths:
        raise SystemExit(f"No target scores found under {args.out_dir / 'target_scores'}")
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    raw_path = args.out_dir / "label_only_boundary_raw.csv"
    atomic_write_csv(raw, raw_path)
    groups = [
        column
        for column in (
            "attack",
            "dataset",
            "architecture",
            "fm_kind",
            "reps",
            "depth",
        )
        if column in raw.columns
    ]
    metrics = [
        "auc",
        "tpr_at_0_05_fpr",
        "tpr_at_0_10_fpr",
        "balanced_accuracy_crossfit",
        "membership_advantage_crossfit",
        "mean_label_queries",
    ]
    summary = (
        raw.groupby(groups, dropna=False)[metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary["error_bar_type"] = "mean ± sample SD across target-model seeds"
    summary_path = args.out_dir / "label_only_boundary_summary.csv"
    atomic_write_csv(summary, summary_path)
    write_analysis_metadata(
        args.out_dir / "analysis_metadata.json",
        script=Path(__file__).name,
        inputs=[str(args.targets), str(args.run_root)],
        outputs=[str(raw_path), str(summary_path)],
        ci_method=CI_RECORD,
        bootstrap_unit="member/non-member records within target; target seed for summary SD",
        bootstrap_replicates=args.bootstrap,
        error_bar_type="mean ± sample SD across target-model seeds",
        notes=(
            "Only predicted class labels are consumed by the attack. Boundary "
            "distance is estimated by binary searches toward held-out validation "
            "anchors and is not claimed to be the certified or globally minimal radius."
        ),
    )
    atomic_write_json(
        {
            "reference_repository_studied": "zhenglisec/Label-Only-MIA",
            "reference_commit": REFERENCE_COMMIT,
            "reference_license_note": (
                "No license file was present at the studied commit; no source code "
                "was copied. This is an independent QuRiFT implementation."
            ),
            "target_files": len(paths),
        },
        args.out_dir / "reference_provenance.json",
    )
    print(f"[OK] raw={len(raw)} summary={len(summary)} -> {args.out_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--repo-root", type=Path, default=Path("."))
        subparser.add_argument("--targets", type=Path, required=True)
        subparser.add_argument("--run-root", type=Path, default=Path("reviewer_runs"))
        subparser.add_argument(
            "--out-dir",
            type=Path,
            default=Path("reviewer_results/label_only_boundary"),
        )
        subparser.add_argument("--bootstrap", type=int, default=5000)
        subparser.add_argument("--seed", type=int, default=2026)
        subparser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
        subparser.add_argument("--resume", action="store_true")

    score = subparsers.add_parser("score-target")
    common(score)
    score.add_argument("--target-id", required=True)
    score.add_argument("--n-member", type=int, default=None)
    score.add_argument("--n-nonmember", type=int, default=None)
    score.add_argument("--anchors", type=int, default=16)
    score.add_argument("--binary-steps", type=int, default=10)
    score.add_argument("--norm", choices=["l1", "l2", "linf"], default="l2")
    score.add_argument("--query-batch-size", type=int, default=64)
    score.set_defaults(function=score_target)

    collect = subparsers.add_parser("aggregate")
    common(collect)
    collect.set_defaults(function=aggregate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "anchors") and args.anchors < 1:
        raise SystemExit("--anchors must be positive")
    if hasattr(args, "binary_steps") and args.binary_steps < 1:
        raise SystemExit("--binary-steps must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
