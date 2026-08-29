from __future__ import annotations

import copy
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.check_channel_lira_phase7_readiness import (
    DEFAULT_PROTOCOL,
    DEFAULT_PROTOCOL_LOCK,
    DEFAULT_CANDIDATE_PROBE,
    inspect_execution_artifacts,
    inspect_candidate_probe,
    inspect_snapshot,
    project_costs,
    read_targets,
    resolve_repo_path,
    sha256,
    validate_design,
    validate_protocol_lock,
)
from experiments.channel_lira_phase7_stage1 import (
    DEFAULT_TARGETS as DEFAULT_STAGE1_TARGETS,
    EXPECTED_TARGETS as STAGE1_TARGETS,
    validate_stage1_manifest,
)
from experiments.run_channel_lira_phase7_stage1 import (
    DEFAULT_OUT as DEFAULT_STAGE1_OUT,
    DEFAULT_REFERENCE_DIR as DEFAULT_STAGE1_REFERENCES,
    DEFAULT_RUN_ROOT as DEFAULT_STAGE1_RUN_ROOT,
    DEFAULT_SNAPSHOT as DEFAULT_STAGE1_SNAPSHOT,
    validate_locked_stage,
)
from experiments.channel_lira_phase7_stage2 import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPECTED_CELLS as STAGE2_CELLS,
    EXPECTED_TARGETS as STAGE2_TARGETS,
    evaluate_gates,
    hierarchical_intervals,
    validate_raw_output_seal,
    validate_stage2_manifest,
)
from experiments.run_channel_lira_phase7_stage2 import (
    DEFAULT_OUT as DEFAULT_STAGE2_OUT,
    DEFAULT_REFERENCE_DIR as DEFAULT_STAGE2_REFERENCES,
    DEFAULT_RUN_ROOT as DEFAULT_STAGE2_RUN_ROOT,
    DEFAULT_SNAPSHOT as DEFAULT_STAGE2_SNAPSHOT,
    DEFAULT_TARGETS as DEFAULT_STAGE2_TARGETS,
    PRIMARY_TOTAL_SHOTS,
    _process_tree_pids,
    group_targets_by_cell,
    noisy_target_command,
    run_cell_queues,
    validate_locked_stage as validate_locked_stage2,
)


class ChannelLiRAPhase7ReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        cls.targets_path = resolve_repo_path(
            cls.protocol["provenance"]["phase7_target_manifest"]
        )
        cls.rows = read_targets(cls.targets_path)

    def test_frozen_design_is_hash_bound_and_valid(self) -> None:
        self.assertEqual(validate_protocol_lock(DEFAULT_PROTOCOL, DEFAULT_PROTOCOL_LOCK), [])
        self.assertEqual(validate_design(self.protocol, self.rows), [])
        self.assertEqual(
            sha256(self.targets_path),
            self.protocol["provenance"]["phase7_target_manifest_sha256"],
        )

    def test_pilot_is_excluded_from_twelve_target_confirmatory_subset(self) -> None:
        pilot = [row for row in self.rows if row["phase7_analysis_role"] == "pilot_replication"]
        confirmatory = [row for row in self.rows if row["phase7_analysis_role"] == "confirmatory"]
        self.assertEqual(len(pilot), 3)
        self.assertEqual(len(confirmatory), 12)
        self.assertEqual({row["structural_cell_id"] for row in pilot}, {"eff_su2_r1_d2"})
        self.assertNotIn(
            "eff_su2_r1_d2",
            {row["structural_cell_id"] for row in confirmatory},
        )

    def test_primary_low_fpr_gate_rejects_the_old_200_record_population(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        candidate = protocol["study_population"]["candidate_protocol"]
        candidate.update({
            "members_per_target": 200,
            "nonmembers_per_target": 200,
            "candidates_per_target": 400,
            "expected_false_positives_at_primary_fpr": 2,
            "fpr_resolution": 0.005,
            "tpr_resolution": 0.005,
        })
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row["vector_train"] = "200"
            row["vector_test"] = "200"
        errors = validate_design(protocol, rows)
        self.assertIn("member population is below the primary-endpoint minimum", errors)
        self.assertIn("nonmember population is below the primary-endpoint minimum", errors)
        self.assertIn("too few false positives resolve the primary FPR", errors)

    def test_existing_learned_comparator_is_frozen_and_access_qualified(self) -> None:
        learned = self.protocol["attacks"]["learned_mia"]
        self.assertEqual(learned["implementation"], "target_crossfit_learned_mia")
        self.assertTrue(learned["frozen_without_redesign"])
        self.assertIn("privileged victim-crossfit", learned["paper_label"])
        self.assertIn("Does not support claims", learned["claim_limit"])

    def test_cost_projection_counts_calibration_and_reference_serving_separately(self) -> None:
        costs = project_costs(self.protocol)
        self.assertEqual(
            costs["primary_confirmatory"]["noisy_reference_shots"], 163_840_000
        )
        self.assertEqual(
            costs["primary_per_attacked_target"]["matched_noisy_lira_reference_shots"],
            40_960_000,
        )
        self.assertEqual(
            costs["primary_per_attacked_target"]["channel_lira_auxiliary_calibration_shots"],
            5_120_000,
        )
        self.assertEqual(
            costs["primary_per_attacked_target"]["channel_to_matched_ratio"], 0.125
        )
        self.assertEqual(
            costs["full_frozen_matrix"]["all_simulated_shots"], 3_793_920_000
        )

    def test_frozen_snapshot_is_content_hash_valid(self) -> None:
        snapshot = inspect_snapshot(self.protocol)
        self.assertTrue(snapshot["ready"], snapshot["errors"])
        self.assertEqual(
            snapshot["manifest_sha256"],
            self.protocol["provenance"]["snapshot_manifest_sha256"],
        )

    def test_materialized_candidate_probe_is_bound_to_the_frozen_design(self) -> None:
        probe = inspect_candidate_probe(
            self.protocol, DEFAULT_PROTOCOL, self.targets_path, DEFAULT_CANDIDATE_PROBE
        )
        self.assertTrue(probe["ready"], probe["errors"])
        self.assertEqual(probe["candidate_count"], 2000)
        self.assertEqual(probe["membership_counts"], {"0": 1000, "1": 1000})

    def test_empty_artifact_roots_are_not_execution_ready(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = inspect_execution_artifacts(
                self.protocol, self.rows, root / "runs", root / "references"
            )
        self.assertEqual(artifacts["target_bundles_ready"], 0)
        self.assertEqual(artifacts["reference_checkpoints_ready"], 0)
        self.assertEqual(artifacts["references_expected"], 80)

    def test_stage1_manifest_is_an_exact_pilot_subset_of_locked_targets(self) -> None:
        rows = validate_stage1_manifest(self.protocol, DEFAULT_STAGE1_TARGETS)
        self.assertEqual(tuple(row["target_id"] for row in rows), STAGE1_TARGETS)
        self.assertEqual({row["phase7_analysis_role"] for row in rows}, {"pilot_replication"})

    def test_stage1_compute_requires_exact_protocol_hash_acknowledgement(self) -> None:
        arguments = argparse.Namespace(
            protocol=DEFAULT_PROTOCOL.resolve(),
            protocol_lock=DEFAULT_PROTOCOL_LOCK.resolve(),
            candidate_probe=DEFAULT_CANDIDATE_PROBE.resolve(),
            targets=DEFAULT_STAGE1_TARGETS.resolve(),
            snapshot=DEFAULT_STAGE1_SNAPSHOT.resolve(),
            acknowledge_protocol_hash="",
            out_dir=DEFAULT_STAGE1_OUT.resolve(),
            run_root=DEFAULT_STAGE1_RUN_ROOT.resolve(),
            reference_dir=DEFAULT_STAGE1_REFERENCES.resolve(),
        )
        validate_locked_stage(arguments, require_ack=False)
        with self.assertRaisesRegex(ValueError, "acknowledge-protocol-hash"):
            validate_locked_stage(arguments, require_ack=True)
        arguments.acknowledge_protocol_hash = sha256(DEFAULT_PROTOCOL)
        validate_locked_stage(arguments, require_ack=True)

    def test_stage2_manifest_is_exact_confirmatory_projection(self) -> None:
        rows = validate_stage2_manifest(self.protocol, DEFAULT_STAGE2_TARGETS)
        self.assertEqual(tuple(row["target_id"] for row in rows), STAGE2_TARGETS)
        self.assertEqual(
            tuple(dict.fromkeys(row["structural_cell_id"] for row in rows)),
            STAGE2_CELLS,
        )
        self.assertEqual({row["phase7_analysis_role"] for row in rows}, {"confirmatory"})

    def test_stage2_parallel_plan_has_one_sequential_queue_per_cell(self) -> None:
        rows = validate_stage2_manifest(self.protocol, DEFAULT_STAGE2_TARGETS)
        grouped = group_targets_by_cell(rows)
        self.assertEqual(tuple(grouped), STAGE2_CELLS)
        self.assertEqual(tuple(value for targets in grouped.values() for value in targets), STAGE2_TARGETS)
        self.assertTrue(all(len(targets) == 3 for targets in grouped.values()))

        arguments = argparse.Namespace(
            python="python",
            targets=DEFAULT_STAGE2_TARGETS,
            run_root=DEFAULT_STAGE2_RUN_ROOT,
            reference_dir=DEFAULT_STAGE2_REFERENCES,
            noisy_dir=DEFAULT_STAGE2_OUT / "noisy_lira",
            snapshot=DEFAULT_STAGE2_SNAPSHOT,
            qiskit_batch_size=16,
            aer_max_parallel_threads=64,
            record_bootstrap=200,
            score_device="cuda",
        )
        command = noisy_target_command(arguments, STAGE2_TARGETS[0])
        self.assertIn("--resume", command)
        self.assertEqual(command[command.index("--aer-max-parallel-threads") + 1], "64")
        self.assertEqual(command[command.index("--simulator-seeds") + 1], "0,1,2,3,4,5,6,7,8,9")

    def test_stage2_parallel_scheduler_records_components_and_wall_clock(self) -> None:
        with TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            run_cell_queues(
                {
                    "cell_a": [("a1", ["/bin/true"]), ("a2", ["/bin/true"])],
                    "cell_b": [("b1", ["/bin/true"]), ("b2", ["/bin/true"])],
                },
                max_workers=2,
                out_dir=out_dir,
                poll_interval=0.01,
            )
            records = json.loads(
                (out_dir / "STAGE_TIMINGS.json").read_text(encoding="utf-8")
            )["records"]
            components = [row for row in records if row.get("parallel_group")]
            wall_clock = [row for row in records if not row.get("parallel_group")]
            self.assertEqual({row["label"] for row in components}, {"a1", "a2", "b1", "b2"})
            self.assertEqual(
                [row["label"] for row in wall_clock],
                ["primary_noisy_scoring_parallel_wall_clock"],
            )

            with self.assertRaisesRegex(ValueError, "max_workers"):
                run_cell_queues({}, max_workers=0, out_dir=out_dir)

    def test_stage2_process_monitor_tolerates_disappearing_proc_entries(self) -> None:
        vanished = Path("/proc/999999999/status").parent
        with patch(
            "experiments.run_channel_lira_phase7_stage2.Path.iterdir",
            return_value=[vanished],
        ):
            self.assertEqual(_process_tree_pids(42), {42})

    def test_stage2_scoring_requires_hash_and_exact_shot_budget_acknowledgement(self) -> None:
        arguments = argparse.Namespace(
            stage="score",
            protocol=DEFAULT_PROTOCOL.resolve(),
            protocol_lock=DEFAULT_PROTOCOL_LOCK.resolve(),
            candidate_probe=DEFAULT_CANDIDATE_PROBE.resolve(),
            targets=DEFAULT_STAGE2_TARGETS.resolve(),
            snapshot=DEFAULT_STAGE2_SNAPSHOT.resolve(),
            acknowledge_protocol_hash=sha256(DEFAULT_PROTOCOL),
            acknowledge_shot_budget=0,
            out_dir=DEFAULT_STAGE2_OUT.resolve(),
            run_root=DEFAULT_STAGE2_RUN_ROOT.resolve(),
            reference_dir=DEFAULT_STAGE2_REFERENCES.resolve(),
        )
        with self.assertRaisesRegex(ValueError, "acknowledge-shot-budget"):
            validate_locked_stage2(arguments, require_ack=True)
        arguments.acknowledge_shot_budget = PRIMARY_TOTAL_SHOTS
        validate_locked_stage2(arguments, require_ack=True)

    def test_stage2_hierarchical_bootstrap_uses_cells_and_target_means(self) -> None:
        rows = []
        for cell in STAGE2_CELLS:
            for target in range(3):
                rows.append({
                    "cell": cell,
                    "target_id": f"{cell}_{target}",
                    "contrast": "constant_contrast",
                    "tpr_at_1pct_fpr_difference": 0.012,
                    "auc_difference": 0.02,
                    "tpr_at_5pct_fpr_difference": 0.03,
                })
        first = hierarchical_intervals(
            rows, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED
        )
        second = hierarchical_intervals(
            rows, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED
        )
        self.assertEqual(first, second)
        primary = next(row for row in first if row["metric"] == "tpr_at_1pct_fpr")
        self.assertAlmostEqual(float(primary["point_estimate"]), 0.012)
        self.assertAlmostEqual(float(primary["ci95_lower"]), 0.012)
        self.assertAlmostEqual(float(primary["ci95_upper"]), 0.012)

    def test_stage2_gate_logic_allows_a_plus_b_auditing_paper(self) -> None:
        def interval(contrast: str, metric: str, lower: float) -> dict[str, object]:
            return {
                "contrast": contrast,
                "metric": metric,
                "point_estimate": lower + 0.01,
                "ci95_lower": lower,
                "ci95_upper": lower + 0.02,
            }

        intervals = [
            interval("matched_reference_minus_mismatched", "tpr_at_1pct_fpr", 0.001),
            interval("affine_minus_matched_reference", "tpr_at_1pct_fpr", -0.004),
            interval("affine_minus_matched_reference", "auc", -0.009),
            interval("affine_minus_loss", "tpr_at_1pct_fpr", -0.001),
        ]
        decision = evaluate_gates(self.protocol, intervals)
        self.assertTrue(decision["A_channel_mismatch"])
        self.assertTrue(decision["B_efficient_recovery"])
        self.assertFalse(decision["C_practical_attack"])
        self.assertAlmostEqual(
            float(decision["amortized_channel_to_matched_cost_ratio"]), 3 / 16
        )
        self.assertTrue(decision["secondary_execution_warranted"])
        self.assertIn("A+B only", decision["paper_decision"])

    def test_stage2_raw_output_seal_detects_post_seal_mutation(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "protocol.json"
            targets = root / "targets.csv"
            artifact = root / "raw.npz"
            seal = root / "seal.json"
            protocol.write_text("{}\n", encoding="utf-8")
            targets.write_text("target_id\nexample\n", encoding="utf-8")
            artifact.write_bytes(b"frozen")
            payload = {
                "schema_version": 1,
                "scope": "phase7_stage2_primary",
                "protocol_sha256": sha256(protocol),
                "target_manifest_sha256": sha256(targets),
                "artifact_count": 1,
                "files": {str(artifact): sha256(artifact)},
            }
            seal.write_text(json.dumps(payload), encoding="utf-8")
            validate_raw_output_seal(
                seal, protocol_path=protocol, target_manifest=targets
            )
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_raw_output_seal(
                    seal, protocol_path=protocol, target_manifest=targets
                )


if __name__ == "__main__":
    unittest.main()
