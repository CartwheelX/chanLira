#!/usr/bin/env python3
"""Validate the frozen Phase-7 design without launching training or circuit jobs.

The audit distinguishes three states that must not be conflated:

* design readiness: the confirmatory protocol is internally coherent and hash-bound;
* execution readiness: every target/reference checkpoint and snapshot is present; and
* publication-artifact readiness: an immutable checkpoint archive is identified.

This command is read-only with respect to experiment inputs.  It writes only a
machine-readable readiness payload and its Markdown rendering.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "reviewer_targets/channel_lira_phase7_protocol.json"
DEFAULT_PROTOCOL_LOCK = ROOT / "reviewer_targets/channel_lira_phase7_protocol.sha256"
DEFAULT_OUT = ROOT / "channel_lira_results/phase7_readiness"
DEFAULT_CANDIDATE_PROBE = DEFAULT_OUT / "CANDIDATE_PROBE.json"
DEFAULT_REFERENCE_DIR = ROOT / "channel_lira_results/phase7/references"
DEFAULT_RUN_ROOT = ROOT / "reviewer_runs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_path(value: str, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_targets(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Target manifest is empty: {path}")
    return rows


def validate_protocol_lock(protocol_path: Path, lock_path: Path) -> list[str]:
    if not lock_path.is_file():
        return ["the external Phase-7 protocol SHA-256 lock is missing"]
    fields = lock_path.read_text(encoding="utf-8").strip().split()
    if not fields:
        return ["the external Phase-7 protocol SHA-256 lock is empty"]
    if fields[0] != sha256(protocol_path):
        return ["the Phase-7 protocol differs from its external SHA-256 lock"]
    if len(fields) > 1 and Path(fields[1]).name != protocol_path.name:
        return ["the protocol SHA-256 lock names a different file"]
    return []


def canonical_cell(cell: str) -> str:
    return f"{cell}_wd0"


def validate_design(
    protocol: dict[str, Any], rows: list[dict[str, str]], root: Path = ROOT
) -> list[str]:
    """Return all frozen-design violations instead of failing on the first one."""
    errors: list[str] = []
    population = protocol.get("study_population", {})
    references = protocol.get("reference_protocol", {})
    serving = protocol.get("serving_protocol", {})
    endpoints = protocol.get("endpoint_hierarchy", {})
    attacks = protocol.get("attacks", {})
    success = protocol.get("success_criteria", {})
    provenance = protocol.get("provenance", {})

    if int(protocol.get("schema_version", -1)) != 1:
        errors.append("protocol schema_version must be 1")
    if protocol.get("protocol_id") != "channel_lira_phase7_confirmatory_v1":
        errors.append("unexpected protocol_id")
    if bool(protocol.get("automatic_execution", True)):
        errors.append("the readiness protocol must never authorize automatic execution")
    if bool(provenance.get("hardware_execution", True)):
        errors.append("Phase 7 must not claim or authorize hardware execution")
    if provenance.get("backend_name") != "ibm_kingston":
        errors.append("the frozen backend name must remain ibm_kingston")

    manifest = resolve_repo_path(str(provenance.get("phase7_target_manifest", "")), root)
    if not manifest.is_file():
        errors.append("frozen Phase-7 target manifest is missing")
    elif sha256(manifest) != provenance.get("phase7_target_manifest_sha256"):
        errors.append("Phase-7 target manifest hash does not match the frozen protocol")

    phase6 = resolve_repo_path(str(provenance.get("phase6_status", "")), root)
    if not phase6.is_file():
        errors.append("Phase-6 pilot status is missing")
    else:
        if sha256(phase6) != provenance.get("phase6_status_sha256"):
            errors.append("Phase-6 pilot status hash does not match the frozen protocol")
        try:
            if not bool(json.loads(phase6.read_text(encoding="utf-8"))["complete"]):
                errors.append("Phase-6 pilot status is not complete")
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"Phase-6 status is invalid: {exc}")

    pilot_cell = str(population.get("pilot_cell", ""))
    confirmatory_cells = [str(value) for value in population.get("confirmatory_cells", [])]
    expected_cells = [pilot_cell, *confirmatory_cells]
    if len(confirmatory_cells) != 4 or len(set(confirmatory_cells)) != 4:
        errors.append("exactly four unique confirmatory cells are required")
    if pilot_cell in confirmatory_cells:
        errors.append("the Phase-6 pilot cell must be excluded from the confirmatory cells")
    if len(rows) != int(population.get("total_targets", -1)):
        errors.append("target-manifest row count does not match total_targets")
    if len({row.get("target_id", "") for row in rows}) != len(rows):
        errors.append("target IDs must be unique")

    row_cells = sorted({row.get("structural_cell_id", "") for row in rows})
    if row_cells != sorted(expected_cells):
        errors.append("target-manifest structural cells do not match the frozen cells")
    for cell in expected_cells:
        cell_rows = [row for row in rows if row.get("structural_cell_id") == cell]
        expected_role = "pilot_replication" if cell == pilot_cell else "confirmatory"
        expected_count = (
            int(population.get("pilot_replication_targets", -1))
            if cell == pilot_cell
            else int(population.get("confirmatory_targets_per_cell", -1))
        )
        if len(cell_rows) != expected_count:
            errors.append(f"{cell} must contain exactly {expected_count} targets")
        if {row.get("phase7_analysis_role") for row in cell_rows} != {expected_role}:
            errors.append(f"{cell} has an invalid Phase-7 analysis role")

    expected_model_seeds = sorted(int(value) for value in population.get("model_seeds", []))
    shared_data_seed = int(population.get("shared_data_seed", -1))
    candidate = population.get("candidate_protocol", {})
    members = int(candidate.get("members_per_target", -1))
    nonmembers = int(candidate.get("nonmembers_per_target", -1))
    candidates = int(candidate.get("candidates_per_target", -1))
    for cell in expected_cells:
        cell_rows = [row for row in rows if row.get("structural_cell_id") == cell]
        if sorted(int(row["model_seed"]) for row in cell_rows) != expected_model_seeds:
            errors.append(f"{cell} model seeds differ from the frozen model seeds")
    for row in rows:
        if int(row.get("data_seed", -1)) != shared_data_seed:
            errors.append(f"{row.get('target_id')} does not use the shared data seed")
        if int(row.get("vector_train", -1)) != members:
            errors.append(f"{row.get('target_id')} member count is not frozen")
        if int(row.get("vector_test", -1)) != nonmembers:
            errors.append(f"{row.get('target_id')} nonmember count is not frozen")
        if row.get("experiment") != "channel_lira_phase7":
            errors.append(f"{row.get('target_id')} has the wrong experiment namespace")

    primary_fpr = float(candidate.get("primary_fpr", -1.0))
    expected_fp = int(round(nonmembers * primary_fpr))
    if candidates != members + nonmembers:
        errors.append("candidate count must equal members plus nonmembers")
    if members < int(candidate.get("minimum_members_for_primary_endpoint", 0)):
        errors.append("member population is below the primary-endpoint minimum")
    if nonmembers < int(candidate.get("minimum_nonmembers_for_primary_endpoint", 0)):
        errors.append("nonmember population is below the primary-endpoint minimum")
    if expected_fp < int(candidate.get("minimum_false_positives_at_primary_fpr", 0)):
        errors.append("too few false positives resolve the primary FPR")
    if not math.isclose(
        1.0 / nonmembers, float(candidate.get("fpr_resolution", -1.0))
    ):
        errors.append("recorded FPR resolution is inconsistent with nonmember count")
    if not math.isclose(
        1.0 / members, float(candidate.get("tpr_resolution", -1.0))
    ):
        errors.append("recorded TPR resolution is inconsistent with member count")
    if bool(candidate.get("ultra_low_fpr_0_1pct_supported", True)):
        errors.append("the protocol must not claim stable 0.1% FPR with 1,000 nonmembers")
    for field in ("candidate_fingerprint", "sample_ids_sha256"):
        value = str(candidate.get(field, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            errors.append(f"candidate protocol {field} must be a lowercase SHA-256 digest")

    reference_count = int(references.get("references_per_cell", -1))
    in_count = int(references.get("in_references_per_candidate", -1))
    out_count = int(references.get("out_references_per_candidate", -1))
    if reference_count != 16 or in_count != 8 or out_count != 8:
        errors.append("the reference design must remain exactly 8-IN/8-OUT over 16 models")
    if in_count + out_count != reference_count:
        errors.append("IN and OUT reference counts do not sum to the bank size")
    if int(references.get("reference_train_records", -1)) != candidates // 2:
        errors.append("balanced reference train size must be half the candidate pool")
    if int(references.get("total_reference_models", -1)) != len(expected_cells) * reference_count:
        errors.append("total reference-model count is inconsistent")

    primary = serving.get("primary", {})
    endpoint = endpoints.get("primary_endpoint", {})
    if primary.get("mode") != "noisy_shot" or int(primary.get("shots", -1)) != 128:
        errors.append("primary serving condition must be noisy_shot at 128 shots")
    if len(primary.get("simulator_seeds", [])) != 10:
        errors.append("primary serving condition must contain ten simulator seeds")
    if endpoint.get("metric") != "tpr_at_1pct_fpr":
        errors.append("primary endpoint must be TPR@1% FPR")
    if endpoint.get("primary_comparison") != "affine_channel_lira_minus_loss_mia":
        errors.append("the practical primary comparison must be ChannelLiRA minus loss MIA")
    if "confirmatory only" not in str(endpoints.get("confirmatory_subset", "")):
        errors.append("primary inference must explicitly exclude the pilot cell")

    learned = attacks.get("learned_mia", {})
    if learned.get("implementation") != "target_crossfit_learned_mia":
        errors.append("the frozen learned-MIA comparator was changed")
    if not bool(learned.get("frozen_without_redesign", False)):
        errors.append("the learned-MIA comparator must remain frozen without redesign")
    if "privileged victim-crossfit" not in str(learned.get("paper_label", "")):
        errors.append("the learned-MIA comparator must retain its access-qualified label")

    efficient = success.get("B_efficient_recovery", {})
    tpr_margin = float(efficient.get("primary_tpr_absolute_noninferiority_margin", -1.0))
    auc_margin = float(efficient.get("secondary_auc_absolute_noninferiority_margin", -1.0))
    if not (0.0 < tpr_margin <= 0.01):
        errors.append("primary TPR non-inferiority margin must be in (0, 0.01]")
    if not (0.0 < auc_margin <= 0.02):
        errors.append("secondary AUC non-inferiority margin must be in (0, 0.02]")

    return errors


def inspect_snapshot(protocol: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    provenance = protocol["provenance"]
    directory = resolve_repo_path(provenance["backend_snapshot"], root)
    manifest_path = directory / "snapshot_manifest.json"
    result: dict[str, Any] = {
        "directory": str(directory.resolve()),
        "manifest": str(manifest_path.resolve()),
        "ready": False,
        "errors": [],
    }
    if not manifest_path.is_file():
        result["errors"].append("snapshot manifest is missing")
        return result
    observed_manifest_hash = sha256(manifest_path)
    result["manifest_sha256"] = observed_manifest_hash
    if observed_manifest_hash != provenance["snapshot_manifest_sha256"]:
        result["errors"].append("snapshot manifest hash differs from the frozen protocol")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("snapshot_schema_version", -1)) != 2:
            result["errors"].append("snapshot must use reconstructable schema version 2")
        for filename, expected in manifest.get("files_sha256", {}).items():
            path = directory / filename
            if not path.is_file():
                result["errors"].append(f"snapshot file is missing: {filename}")
            elif sha256(path) != expected:
                result["errors"].append(f"snapshot file hash mismatch: {filename}")
        metadata_path = directory / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            result["backend_name"] = metadata.get("resolved_backend_name")
            result["calibration_timestamp"] = metadata.get("calibration_timestamp")
            if metadata.get("resolved_backend_name") != provenance.get("backend_name"):
                result["errors"].append("snapshot backend differs from the frozen backend")
            if not bool(metadata.get("noise_model_loaded", False)):
                result["errors"].append("snapshot metadata does not record a loaded noise model")
            if metadata.get("noise_load_error") not in {None, ""}:
                result["errors"].append("snapshot metadata records a noise-load error")
        else:
            result["errors"].append("snapshot metadata.json is missing")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result["errors"].append(f"snapshot metadata is invalid: {exc}")
    result["ready"] = not result["errors"]
    return result


def inspect_candidate_probe(
    protocol: dict[str, Any],
    protocol_path: Path,
    target_path: Path,
    probe_path: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(probe_path.resolve()),
        "ready": False,
        "errors": [],
    }
    if not probe_path.is_file():
        result["errors"].append(
            "candidate probe is missing; run experiments/probe_channel_lira_phase7_candidates.py"
        )
        return result
    try:
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        candidate = protocol["study_population"]["candidate_protocol"]
        references = protocol["reference_protocol"]
        expected_membership = {
            "0": int(candidate["nonmembers_per_target"]),
            "1": int(candidate["members_per_target"]),
        }
        checks = {
            "probe ready flag": bool(probe.get("ready", False)),
            "protocol hash": probe.get("protocol_sha256") == sha256(protocol_path),
            "target-manifest hash": probe.get("target_manifest_sha256") == sha256(target_path),
            "candidate count": int(probe.get("candidate_count", -1))
            == int(candidate["candidates_per_target"]),
            "membership counts": probe.get("membership_counts") == expected_membership,
            "candidate fingerprint": probe.get("candidate_fingerprint")
            == candidate.get("candidate_fingerprint"),
            "ordered sample-ID hash": probe.get("sample_ids_sha256")
            == candidate.get("sample_ids_sha256"),
            "unique sample IDs": bool(probe.get("sample_ids_unique", False)),
            "reference-design seed": int(probe.get("reference_design_seed", -1))
            == int(references["reference_design_seed"]),
            "reference-design shape": probe.get("reference_design_shape")
            == [
                int(references["references_per_cell"]),
                int(candidate["candidates_per_target"]),
            ],
            "per-reference train count": probe.get("per_reference_train_counts")
            == [int(references["reference_train_records"])],
            "per-candidate IN count": probe.get("per_candidate_in_counts")
            == [int(references["in_references_per_candidate"])],
            "no training": not bool(probe.get("training_performed", True)),
            "no circuit execution": not bool(probe.get("circuit_execution_performed", True)),
        }
        for label, passed in checks.items():
            if not passed:
                result["errors"].append(f"candidate probe failed {label} validation")
        result.update({
            "candidate_fingerprint": probe.get("candidate_fingerprint"),
            "sample_ids_sha256": probe.get("sample_ids_sha256"),
            "candidate_count": probe.get("candidate_count"),
            "membership_counts": probe.get("membership_counts"),
        })
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result["errors"].append(f"candidate probe is invalid: {exc}")
    result["ready"] = not result["errors"]
    return result


def inspect_execution_artifacts(
    protocol: dict[str, Any],
    rows: list[dict[str, str]],
    run_root: Path,
    reference_dir: Path,
) -> dict[str, Any]:
    expected_references = int(protocol["reference_protocol"]["references_per_cell"])
    candidate_count = int(
        protocol["study_population"]["candidate_protocol"]["candidates_per_target"]
    )
    targets = []
    expected_members = int(
        protocol["study_population"]["candidate_protocol"]["members_per_target"]
    )
    expected_nonmembers = int(
        protocol["study_population"]["candidate_protocol"]["nonmembers_per_target"]
    )
    for row in rows:
        directory = run_root / row["experiment"] / row["target_id"]
        required = {
            "model": directory / "target_model.pt",
            "attack_data": directory / "target_attack_data.pt",
            "summary": directory / "target_export_summary.json",
        }
        present = {
            name: path.is_file() and path.stat().st_size > 0
            for name, path in required.items()
        }
        payload_valid = False
        payload_error = ""
        if present["attack_data"]:
            try:
                import torch

                try:
                    payload = torch.load(
                        required["attack_data"], map_location="cpu", weights_only=False
                    )
                except TypeError:
                    payload = torch.load(required["attack_data"], map_location="cpu")
                membership = torch.as_tensor(payload["membership"]).reshape(-1)
                values, counts = membership.unique(return_counts=True)
                observed = {
                    int(value): int(count)
                    for value, count in zip(values.tolist(), counts.tolist())
                }
                if sorted(observed.values()) != sorted([expected_members, expected_nonmembers]):
                    raise ValueError(f"membership counts differ from the frozen population: {observed}")
                metadata = payload.get("meta", {})
                if str(metadata.get("target_id")) != row["target_id"]:
                    raise ValueError("target ID mismatch")
                if int(metadata.get("model_seed", -1)) != int(row["model_seed"]):
                    raise ValueError("model seed mismatch")
                if int(metadata.get("data_seed", -1)) != int(row["data_seed"]):
                    raise ValueError("data seed mismatch")
                payload_valid = True
            except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
                payload_error = f"{type(exc).__name__}: {exc}"
        targets.append({
            "target_id": row["target_id"],
            "ready": all(present.values()) and payload_valid,
            "directory": str(directory.resolve()),
            "present": present,
            "payload_valid": payload_valid,
            "payload_error": payload_error,
        })

    references = []
    for cell in sorted({row["structural_cell_id"] for row in rows}):
        directory = reference_dir / "reference_models" / canonical_cell(cell)
        records = []
        inclusion_rows = []
        fingerprints = set()
        for reference_id in range(expected_references):
            score = directory / f"reference_{reference_id:03d}.npz"
            checkpoint = score.with_suffix(".pt")
            record: dict[str, Any] = {
                "reference_id": reference_id,
                "score_ready": score.is_file() and score.stat().st_size > 0,
                "checkpoint_ready": checkpoint.is_file() and checkpoint.stat().st_size > 0,
                "metadata_valid": False,
                "checkpoint_metadata_valid": False,
            }
            score_fingerprint = None
            if record["score_ready"]:
                try:
                    import numpy as np

                    with np.load(score, allow_pickle=False) as saved:
                        inclusion = np.asarray(saved["inclusion"], dtype=bool)
                        if int(saved["reference_id"]) != reference_id:
                            raise ValueError("reference ID mismatch")
                        if int(saved["num_references"]) != expected_references:
                            raise ValueError("reference-count mismatch")
                        if len(inclusion) != candidate_count:
                            raise ValueError("candidate-count mismatch")
                        if int(inclusion.sum()) != candidate_count // 2:
                            raise ValueError("reference train-size mismatch")
                        inclusion_rows.append(inclusion)
                        score_fingerprint = str(saved["candidate_fingerprint"])
                        fingerprints.add(score_fingerprint)
                        record["metadata_valid"] = True
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
            if record["checkpoint_ready"]:
                try:
                    import torch

                    try:
                        payload = torch.load(
                            checkpoint, map_location="cpu", weights_only=False
                        )
                    except TypeError:
                        payload = torch.load(checkpoint, map_location="cpu")
                    if int(payload["reference_id"]) != reference_id:
                        raise ValueError("checkpoint reference ID mismatch")
                    if int(payload["num_references"]) != expected_references:
                        raise ValueError("checkpoint reference-count mismatch")
                    if str(payload["structural_cell"]) != canonical_cell(cell):
                        raise ValueError("checkpoint structural-cell mismatch")
                    if score_fingerprint is None or str(
                        payload["candidate_fingerprint"]
                    ) != score_fingerprint:
                        raise ValueError("checkpoint/score candidate fingerprint mismatch")
                    if not isinstance(payload.get("state_dict"), dict) or not payload["state_dict"]:
                        raise ValueError("checkpoint has no model state dictionary")
                    record["checkpoint_metadata_valid"] = True
                except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
                    record["checkpoint_error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
        balanced = False
        if len(inclusion_rows) == expected_references:
            inclusion = np.stack(inclusion_rows)
            balanced = bool(np.all(inclusion.sum(axis=0) == expected_references // 2))
        cell_ready = bool(
            all(row["score_ready"] for row in records)
            and all(row["checkpoint_ready"] for row in records)
            and all(row["metadata_valid"] for row in records)
            and all(row["checkpoint_metadata_valid"] for row in records)
            and balanced
            and len(fingerprints) == 1
        )
        references.append({
            "cell": cell,
            "directory": str(directory.resolve()),
            "ready": cell_ready,
            "scores_ready": sum(bool(row["score_ready"]) for row in records),
            "checkpoints_ready": sum(bool(row["checkpoint_ready"]) for row in records),
            "metadata_valid": sum(bool(row["metadata_valid"]) for row in records),
            "checkpoint_metadata_valid": sum(
                bool(row["checkpoint_metadata_valid"]) for row in records
            ),
            "balanced_inclusion": balanced,
            "candidate_fingerprints": sorted(fingerprints),
            "references": records,
        })

    return {
        "targets": targets,
        "target_bundles_ready": sum(bool(row["ready"]) for row in targets),
        "targets_expected": len(targets),
        "reference_cells": references,
        "reference_banks_ready": sum(bool(row["ready"]) for row in references),
        "reference_banks_expected": len(references),
        "reference_scores_ready": sum(row["scores_ready"] for row in references),
        "reference_checkpoints_ready": sum(row["checkpoints_ready"] for row in references),
        "reference_checkpoint_metadata_valid": sum(
            row["checkpoint_metadata_valid"] for row in references
        ),
        "references_expected": len(references) * expected_references,
    }


def project_costs(protocol: dict[str, Any]) -> dict[str, Any]:
    population = protocol["study_population"]
    references = protocol["reference_protocol"]
    serving = protocol["serving_protocol"]
    candidate_count = int(population["candidate_protocol"]["candidates_per_target"])
    total_cells = int(population["total_cells"])
    total_targets = int(population["total_targets"])
    confirmatory_cells = len(population["confirmatory_cells"])
    confirmatory_targets = int(population["confirmatory_targets"])
    refs_per_cell = int(references["references_per_cell"])

    def shots_for(models: int, shots: list[int], seeds: int) -> int:
        return models * candidate_count * sum(shots) * seeds

    primary_shots = int(serving["primary"]["shots"])
    primary_seeds = len(serving["primary"]["simulator_seeds"])
    primary_confirmatory_reference = shots_for(
        confirmatory_cells * refs_per_cell, [primary_shots], primary_seeds
    )
    primary_confirmatory_target = shots_for(
        confirmatory_targets, [primary_shots], primary_seeds
    )

    noisy_shots = [primary_shots, *[int(v) for v in serving["secondary_noisy"]["shots"]]]
    noisy_seeds = len(serving["primary"]["simulator_seeds"])
    ideal_shots = [int(v) for v in serving["ideal_diagnostic"]["shots"]]
    ideal_seeds = len(serving["ideal_diagnostic"]["simulator_seeds"])
    full_noisy_reference = shots_for(total_cells * refs_per_cell, noisy_shots, noisy_seeds)
    full_ideal_reference = shots_for(total_cells * refs_per_cell, ideal_shots, ideal_seeds)
    full_noisy_target = shots_for(total_targets, noisy_shots, noisy_seeds)
    full_ideal_target = shots_for(total_targets, ideal_shots, ideal_seeds)

    per_attack_matched = shots_for(refs_per_cell, [primary_shots], primary_seeds)
    calibration_models = int(
        protocol["cost_accounting"]["channel_calibration_models_per_attack"]
    )
    per_attack_channel = shots_for(calibration_models, [primary_shots], primary_seeds)
    amortized_models = int(
        protocol["cost_accounting"]["channel_calibration_models_per_cell_amortized"]
    )
    amortized_channel = shots_for(amortized_models, [primary_shots], primary_seeds)

    return {
        "units": "simulated circuit shots; circuit transpilation/runtime are reported separately after execution",
        "primary_confirmatory": {
            "noisy_reference_shots": primary_confirmatory_reference,
            "target_query_shots": primary_confirmatory_target,
            "combined_shots": primary_confirmatory_reference + primary_confirmatory_target,
        },
        "primary_per_attacked_target": {
            "matched_noisy_lira_reference_shots": per_attack_matched,
            "channel_lira_auxiliary_calibration_shots": per_attack_channel,
            "channel_to_matched_ratio": per_attack_channel / per_attack_matched,
        },
        "primary_per_cell_amortized": {
            "matched_noisy_lira_reference_shots": per_attack_matched,
            "channel_lira_unique_calibration_shots": amortized_channel,
            "channel_to_matched_ratio": amortized_channel / per_attack_matched,
        },
        "full_frozen_matrix": {
            "noisy_reference_shots": full_noisy_reference,
            "ideal_reference_shots": full_ideal_reference,
            "all_reference_shots": full_noisy_reference + full_ideal_reference,
            "noisy_target_shots": full_noisy_target,
            "ideal_target_shots": full_ideal_target,
            "all_target_shots": full_noisy_target + full_ideal_target,
            "all_simulated_shots": (
                full_noisy_reference + full_ideal_reference
                + full_noisy_target + full_ideal_target
            ),
        },
    }


def publication_status(protocol: dict[str, Any]) -> dict[str, Any]:
    policy = protocol["publication_artifacts"]
    uri = policy.get("archive_uri")
    digest = policy.get("archive_sha256")
    ready = bool(uri and digest)
    return {
        "ready": ready,
        "immutable_checkpoint_archive_required": bool(
            policy["immutable_checkpoint_archive_required"]
        ),
        "archive_uri": uri,
        "archive_sha256": digest,
        "blocks_experiment_execution": False,
        "blocks_artifact_submission": not ready,
    }


def build_readiness(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    target_path = resolve_repo_path(protocol["provenance"]["phase7_target_manifest"])
    rows = read_targets(target_path)
    design_errors = validate_protocol_lock(protocol_path, args.protocol_lock.resolve())
    design_errors.extend(validate_design(protocol, rows))
    candidate_probe = inspect_candidate_probe(
        protocol, protocol_path, target_path, args.candidate_probe.resolve()
    )
    design_errors.extend(candidate_probe["errors"])
    snapshot = inspect_snapshot(protocol)
    artifacts = inspect_execution_artifacts(
        protocol, rows, args.run_root.resolve(), args.reference_dir.resolve()
    )
    costs = project_costs(protocol)
    publication = publication_status(protocol)
    design_ready = not design_errors
    training_ready = bool(design_ready and snapshot["ready"])
    scoring_ready = bool(
        training_ready
        and artifacts["target_bundles_ready"] == artifacts["targets_expected"]
        and artifacts["reference_banks_ready"] == artifacts["reference_banks_expected"]
    )
    training_blockers = []
    if not design_ready:
        training_blockers.append("the frozen scientific design is invalid")
    if not snapshot["ready"]:
        training_blockers.append("the frozen backend snapshot is not hash-valid")
    scoring_blockers = list(training_blockers)
    missing_targets = artifacts["targets_expected"] - artifacts["target_bundles_ready"]
    if missing_targets:
        scoring_blockers.append(
            f"{missing_targets}/{artifacts['targets_expected']} Phase-7 target bundles are missing"
        )
    missing_refs = artifacts["references_expected"] - artifacts["reference_checkpoints_ready"]
    if missing_refs:
        scoring_blockers.append(
            f"{missing_refs}/{artifacts['references_expected']} Phase-7 reference checkpoints are missing"
        )
    invalid_banks = artifacts["reference_banks_expected"] - artifacts["reference_banks_ready"]
    if invalid_banks and not missing_refs:
        scoring_blockers.append(
            f"{invalid_banks}/{artifacts['reference_banks_expected']} reference banks fail metadata/balance checks"
        )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "protocol_lock": str(args.protocol_lock.resolve()),
        "target_manifest": str(target_path.resolve()),
        "target_manifest_sha256": sha256(target_path),
        "design_ready": design_ready,
        "design_errors": design_errors,
        "training_ready": training_ready,
        "training_blockers": training_blockers,
        "noisy_scoring_ready": scoring_ready,
        "scoring_blockers": scoring_blockers,
        "automatic_execution": False,
        "snapshot": snapshot,
        "candidate_probe": candidate_probe,
        "artifacts": artifacts,
        "low_fpr_gate": protocol["study_population"]["candidate_protocol"],
        "cost_projection": costs,
        "publication_artifacts": publication,
    }


def write_outputs(payload: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "READINESS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts = payload["artifacts"]
    low_fpr = payload["low_fpr_gate"]
    costs = payload["cost_projection"]
    primary = costs["primary_confirmatory"]
    per_attack = costs["primary_per_attacked_target"]
    full = costs["full_frozen_matrix"]
    design_status = "READY" if payload["design_ready"] else "INVALID"
    training_status = "READY" if payload["training_ready"] else "BLOCKED"
    scoring_status = "READY" if payload["noisy_scoring_ready"] else "BLOCKED"
    publication_status_text = (
        "READY" if payload["publication_artifacts"]["ready"] else "PENDING"
    )
    design_errors = "\n".join(
        f"- {value}" for value in payload["design_errors"]
    ) or "- none"
    training_blockers = "\n".join(
        f"- {value}" for value in payload["training_blockers"]
    ) or "- none"
    scoring_blockers = "\n".join(
        f"- {value}" for value in payload["scoring_blockers"]
    ) or "- none"
    report = f"""# ChannelLiRA Phase 7 readiness

## Design status: {design_status}

## Training status: {training_status}

## Noisy-scoring status: {scoring_status}

## Publication-artifact status: {publication_status_text}

This audit does not train models or execute circuits. Phase 6 is explicitly pilot
evidence. The Phase-7 primary analysis is restricted to four confirmatory cells
and uses noisy 128-shot TPR@1% FPR. The existing learned attack is retained without
redesign and is labeled the privileged victim-crossfit learned comparator.

| Design gate | Frozen value |
|---|---:|
| Pilot replication targets | 3 |
| Confirmatory targets | 12 |
| References per cell | 16 |
| Total reference models | 80 |
| Members per target | {low_fpr['members_per_target']:,} |
| Nonmembers per target | {low_fpr['nonmembers_per_target']:,} |
| False positives at 1% FPR | {low_fpr['expected_false_positives_at_primary_fpr']} |
| FPR/TPR empirical resolution | {100 * low_fpr['fpr_resolution']:.1f} percentage points |
| Stable 0.1% FPR supported | {str(low_fpr['ultra_low_fpr_0_1pct_supported']).lower()} |
| Materialized candidate probe | {int(payload['candidate_probe']['ready'])} |

## Artifact availability

| Item | Available | Required |
|---|---:|---:|
| Target checkpoint bundles | {artifacts['target_bundles_ready']} | {artifacts['targets_expected']} |
| Reference score files | {artifacts['reference_scores_ready']} | {artifacts['references_expected']} |
| Reference checkpoints | {artifacts['reference_checkpoints_ready']} | {artifacts['references_expected']} |
| Hash-bound checkpoint metadata | {artifacts['reference_checkpoint_metadata_valid']} | {artifacts['references_expected']} |
| Complete balanced reference banks | {artifacts['reference_banks_ready']} | {artifacts['reference_banks_expected']} |
| Frozen snapshot | {int(payload['snapshot']['ready'])} | 1 |

## Projected serving cost

| Scope | Simulated circuit shots |
|---|---:|
| Confirmatory primary matched-reference execution | {primary['noisy_reference_shots']:,} |
| Confirmatory primary target execution | {primary['target_query_shots']:,} |
| Full frozen noisy reference matrix | {full['noisy_reference_shots']:,} |
| Full frozen ideal reference matrix | {full['ideal_reference_shots']:,} |
| Full frozen target matrix | {full['all_target_shots']:,} |
| Full frozen total | {full['all_simulated_shots']:,} |

At the primary condition for one attacked target, matched noisy LiRA requires
{per_attack['matched_noisy_lira_reference_shots']:,} reference shots. ChannelLiRA
requires {per_attack['channel_lira_auxiliary_calibration_shots']:,} auxiliary
calibration shots under the frozen two-model leave-target-out threat model, a ratio
of {per_attack['channel_to_matched_ratio']:.3f}. This is not zero-cost calibration.

## Design errors

{design_errors}

## Training blockers

{training_blockers}

## Noisy-scoring blockers

{scoring_blockers}

The absent target/reference artifacts are expected at this stage. The passed design
and snapshot gates permit a later explicitly requested training stage; noisy scoring
remains blocked until those artifacts validate. An immutable checkpoint archive and
hashes are required before artifact submission, but do not block training.
"""
    (out_dir / "READINESS.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-lock", type=Path, default=DEFAULT_PROTOCOL_LOCK)
    parser.add_argument("--candidate-probe", type=Path, default=DEFAULT_CANDIDATE_PROBE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require-design-ready", action="store_true")
    parser.add_argument("--require-training-ready", action="store_true")
    parser.add_argument("--require-scoring-ready", action="store_true")
    args = parser.parse_args()
    payload = build_readiness(args)
    write_outputs(payload, args.out_dir.resolve())
    print(
        f"design_ready={payload['design_ready']} "
        f"training_ready={payload['training_ready']} "
        f"noisy_scoring_ready={payload['noisy_scoring_ready']} "
        f"targets={payload['artifacts']['target_bundles_ready']}/"
        f"{payload['artifacts']['targets_expected']} "
        f"references={payload['artifacts']['reference_checkpoints_ready']}/"
        f"{payload['artifacts']['references_expected']}",
        flush=True,
    )
    if args.require_design_ready and not payload["design_ready"]:
        raise SystemExit(1)
    if args.require_training_ready and not payload["training_ready"]:
        raise SystemExit(1)
    if args.require_scoring_ready and not payload["noisy_scoring_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
