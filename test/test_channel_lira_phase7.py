from __future__ import annotations

import copy
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

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


if __name__ == "__main__":
    unittest.main()
