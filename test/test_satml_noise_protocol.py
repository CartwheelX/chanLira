from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from reviewer_tools.qurift_noisy_eval import API_AGGREGATION
from satml_tools.analyze_noise_studies import n1_analysis, n2_analysis, validate_frozen_snapshot
from satml_tools.build_noise_study_targets import build_manifests, normalized_source
from satml_tools.noisy_learned_mia import attack_features
from satml_tools.analyze_noisy_lira import main as analyze_noisy_lira_main


class SaTMLNoiseProtocolTests(unittest.TestCase):
    def test_learned_attack_features_keep_query_variability_explicit(self) -> None:
        pv = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
        query_pv = torch.stack([pv, torch.tensor([[0.6, 0.4], [0.5, 0.5]])])
        payload = {
            "pv": pv,
            "X": torch.cat([pv, torch.ones(2, 5)], dim=1),
            "query_pv": query_pv,
        }
        self.assertEqual(attack_features(payload, "pv").shape, (2, 2))
        self.assertEqual(attack_features(payload, "pv+stats").shape, (2, 7))
        mean_std = attack_features(payload, "pv_mean_std")
        self.assertEqual(mean_std.shape, (2, 4))
        self.assertGreater(float(mean_std[0, 2]), 0.0)

    def test_analysis_rejects_mixed_frozen_snapshots(self) -> None:
        attacks = pd.DataFrame({
            "mode": ["noisy_shot", "noisy_shot"],
            "snapshot_manifest_sha256": ["snapshot-a", "snapshot-b"],
        })
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_frozen_snapshot(attacks)

    def test_frozen_noise_manifests_have_declared_scopes(self) -> None:
        source = normalized_source(
            Path("reviewer_targets/multiseed_factorial_targets.csv")
        )
        manifests = build_manifests(source)
        self.assertEqual({name: len(frame) for name, frame in manifests.items()}, {
            "n1": 36, "n2": 6, "n3": 6, "n3_label": 2,
        })
        self.assertEqual(
            len(manifests["n1"][["fm_kind", "reps", "depth", "model_seed"]].drop_duplicates()),
            36,
        )
        self.assertEqual(set(manifests["n2"].depth), {6})
        self.assertEqual(set(manifests["n2"].model_seed), {43})
        endpoints = set(zip(manifests["n3"].fm_kind, manifests["n3"].reps, manifests["n3"].depth))
        self.assertEqual(endpoints, {("eff_su2", 1, 6), ("zz", 5, 6)})

    def test_n1_interaction_uses_paired_model_seed_units(self) -> None:
        rows = []
        exact = {(1, 2): 0.50, (5, 2): 0.55, (1, 6): 0.52, (5, 6): 0.62}
        noisy = {(1, 2): 0.51, (5, 2): 0.55, (1, 6): 0.53, (5, 6): 0.61}
        for mode, values, queries, shots, aggregation in (
            ("exact", exact, 0, 0, "exact"),
            ("noisy_shot", noisy, 1, 512, API_AGGREGATION),
        ):
            for model_seed in (43, 44, 45):
                for simulator_seed in ((-1,) if mode == "exact" else (0, 1)):
                    for fm_kind, fm_offset in (("eff_su2", 0.0), ("z", 0.03), ("zz", 0.05)):
                        for (reps, depth), auc in values.items():
                            rows.append({
                                "target_id": f"t_{fm_kind}_{reps}_{depth}_{model_seed}",
                                "structural_cell_id": f"{fm_kind}_r{reps}_d{depth}",
                                "fm_kind": fm_kind, "reps": reps, "depth": depth,
                                "model_seed": model_seed, "data_seed": model_seed,
                                "mode": mode, "queries": queries, "shots": shots,
                                "total_shots": queries * shots,
                                "simulator_seed": simulator_seed,
                            "aggregation": aggregation, "attack": "loss",
                            "snapshot_manifest_sha256": "snapshot-a",
                                "attack_family": "scalar_threshold", "auc": auc + fm_offset,
                            })
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            n1_analysis(pd.DataFrame(rows), out_dir=out, bootstrap=250, seed=7)
            raw = pd.read_csv(out / "n1_factorial_effects_raw.csv")
            exact_interaction = raw[
                raw["mode"].eq("exact") & raw.fm_kind.eq("z")
                & raw.effect.eq("repetition_by_depth_interaction")
            ]
            noisy_interaction = raw[
                raw["mode"].eq("noisy_shot") & raw.fm_kind.eq("z")
                & raw.effect.eq("repetition_by_depth_interaction")
            ]
            self.assertEqual(len(exact_interaction), 3)
            self.assertEqual(len(noisy_interaction), 3)
            self.assertTrue((exact_interaction.effect_auc.round(8) == 0.05).all())
            self.assertTrue((noisy_interaction.effect_auc.round(8) == 0.04).all())
            summary = pd.read_csv(out / "n1_factorial_effects_summary.csv")
            row = summary[
                summary["mode"].eq("exact")
                & summary.effect.eq("repetition_by_depth_interaction")
                & summary.scope.eq("encoder_specific")
            ].iloc[0]
            self.assertEqual(int(row.n_model_seed_blocks), 3)
            feature = raw[
                raw["mode"].eq("exact") & raw.effect.eq("feature_zz_minus_eff_su2")
            ]
            self.assertEqual(len(feature), 3)
            self.assertTrue((feature.effect_auc.round(8) == 0.05).all())
            self.assertTrue(feature.effect_scope.eq("paired_feature_map").all())

    def test_n2_equal_budget_contrast_is_explicit(self) -> None:
        policies = {
            (1, 128): 0.55,
            (1, 512): 0.57,
            (1, 2560): 0.60,
            (5, 128): 0.59,
            (5, 512): 0.63,
            (20, 128): 0.62,
        }
        rows = []
        for target_index in range(6):
            for simulator_seed in (0, 1):
                for (queries, shots), auc in policies.items():
                    rows.append({
                        "target_id": f"target_{target_index}",
                        "structural_cell_id": f"cell_{target_index}",
                        "fm_kind": "z", "reps": 1, "depth": 6,
                        "model_seed": 43, "data_seed": 43,
                        "mode": "noisy_shot", "queries": queries, "shots": shots,
                        "total_shots": queries * shots,
                        "simulator_seed": simulator_seed,
                        "aggregation": API_AGGREGATION, "attack": "loss",
                        "snapshot_manifest_sha256": "snapshot-a",
                        "attack_family": "scalar_threshold",
                        "auc": auc + target_index * 0.001,
                    })
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            n2_analysis(pd.DataFrame(rows), out_dir=out, bootstrap=250, seed=11)
            summary = pd.read_csv(out / "n2_query_contrasts_summary.csv")
            five = summary[summary.contrast.eq("equal_total_5x512_minus_1x2560")].iloc[0]
            twenty = summary[summary.contrast.eq("equal_total_20x128_minus_1x2560")].iloc[0]
            self.assertAlmostEqual(float(five.mean_auc_difference), 0.03)
            self.assertAlmostEqual(float(twenty.mean_auc_difference), 0.02)
            self.assertEqual(int(five.n_target_checkpoints), 6)
            self.assertIn("targeted query-policy", five.scope)

    def test_n3_pairs_structural_endpoints_by_model_seed(self) -> None:
        targets = []
        exact = []
        noisy = []
        for model_seed in (43, 44, 45):
            for cell, auc in (("eff_su2_r1_d6", 0.55), ("zz_r5_d6", 0.62)):
                target_id = f"{cell}_s{model_seed}"
                targets.append({"target_id": target_id})
                exact.append({
                    "target_id": target_id, "structural_cell_id": f"{cell}_wd0",
                    "model_seed": model_seed, "attack": "lira_online_fixed_variance",
                    "auc": auc,
                })
                for mode, offset in (("ideal_shot", -0.01), ("noisy_shot", -0.03)):
                    for simulator_seed in (0, 1):
                        noisy.append({
                            "target_id": target_id, "structural_cell_id": cell,
                            "model_seed": model_seed, "mode": mode, "shots": 512,
                            "attack": "lira_online_fixed_variance",
                            "auc": auc + offset,
                            "snapshot_manifest_sha256": "snapshot-a",
                            "simulator_seed": simulator_seed,
                        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_path, exact_path, noisy_path = (
                root / "targets.csv", root / "exact.csv", root / "noisy.csv"
            )
            pd.DataFrame(targets).to_csv(target_path, index=False)
            pd.DataFrame(exact).to_csv(exact_path, index=False)
            pd.DataFrame(noisy).to_csv(noisy_path, index=False)
            out = root / "analysis"
            argv = [
                "analyze_noisy_lira.py", "--targets", str(target_path),
                "--exact", str(exact_path), "--noisy", str(noisy_path),
                "--out-dir", str(out), "--bootstrap", "250",
            ]
            with patch("sys.argv", argv):
                analyze_noisy_lira_main()
            summary = pd.read_csv(out / "n3_endpoint_contrasts.csv")
            self.assertEqual(set(summary["mode"]), {"exact", "ideal_shot", "noisy_shot"})
            self.assertTrue((summary.mean_auc_difference.round(8) == 0.07).all())
            self.assertTrue((summary.n_paired_model_seeds == 3).all())


if __name__ == "__main__":
    unittest.main()
