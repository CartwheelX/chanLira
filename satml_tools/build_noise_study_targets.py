#!/usr/bin/env python3
"""Freeze the matched MNIST target manifests for SaTML noise studies N1--N3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reviewer_tools.reviewer_common import atomic_write_csv, atomic_write_json


FEATURE_MAPS = ("eff_su2", "z", "zz")
REPETITIONS = (1, 5)
DEPTHS = (2, 6)
MODEL_SEEDS = (43, 44, 45)


def normalized_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "target_id", "dataset", "architecture", "fm_kind", "reps", "depth",
        "model_seed", "data_seed", "structural_cell_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Source target table is missing columns: {sorted(missing)}")
    frame = frame[
        frame.dataset.astype(str).str.lower().eq("mnist")
        & frame.architecture.astype(str).str.lower().eq("qnn")
    ].copy()
    for column in ("reps", "depth", "model_seed", "data_seed"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    frame["fm_kind"] = frame.fm_kind.astype(str).str.lower()
    return frame


def select_grid(
    source: pd.DataFrame,
    *,
    feature_maps: Iterable[str],
    repetitions: Iterable[int],
    depths: Iterable[int],
    model_seeds: Iterable[int],
    study: str,
    purpose: str,
) -> pd.DataFrame:
    feature_maps = tuple(feature_maps)
    repetitions = tuple(int(value) for value in repetitions)
    depths = tuple(int(value) for value in depths)
    model_seeds = tuple(int(value) for value in model_seeds)
    selected = source[
        source.fm_kind.isin(feature_maps)
        & source.reps.isin(repetitions)
        & source.depth.isin(depths)
        & source.model_seed.isin(model_seeds)
    ].copy()
    expected = len(feature_maps) * len(repetitions) * len(depths) * len(model_seeds)
    key = ["fm_kind", "reps", "depth", "model_seed"]
    if len(selected) != expected or selected.duplicated(key).any():
        observed = selected[key].sort_values(key).to_dict("records")
        raise ValueError(
            f"{study} expected exactly {expected} unique target rows, found "
            f"{len(selected)}; observed={observed}"
        )
    selected["noise_study"] = study
    selected["noise_study_purpose"] = purpose
    selected["selection_frozen_before_noisy_outcomes"] = True
    return selected.sort_values(key).reset_index(drop=True)


def build_manifests(source: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n1 = select_grid(
        source,
        feature_maps=FEATURE_MAPS,
        repetitions=REPETITIONS,
        depths=DEPTHS,
        model_seeds=MODEL_SEEDS,
        study="N1_structural_factorial",
        purpose=(
            "Primary 3x2x2 matched structural factorial under exact, ideal-shot, "
            "and one frozen backend-derived noise model; three trained target seeds."
        ),
    )
    n2 = select_grid(
        source,
        feature_maps=FEATURE_MAPS,
        repetitions=REPETITIONS,
        depths=(6,),
        model_seeds=(43,),
        study="N2_query_policy",
        purpose=(
            "Targeted API-query study at fixed depth and one checkpoint seed; not "
            "used as independent target-model replication."
        ),
    )
    endpoint_source = source[
        (
            source.fm_kind.eq("eff_su2")
            & source.reps.eq(1)
            & source.depth.eq(6)
        )
        | (
            source.fm_kind.eq("zz")
            & source.reps.eq(5)
            & source.depth.eq(6)
        )
    ]
    # N3 deliberately selects two diagonal endpoints rather than a Cartesian
    # feature-map/repetition grid.
    n3 = endpoint_source[endpoint_source.model_seed.isin(MODEL_SEEDS)].copy()
    if len(n3) != 6 or n3.duplicated(["fm_kind", "reps", "depth", "model_seed"]).any():
        raise ValueError(f"N3 expected six endpoint checkpoints, found {len(n3)}")
    n3["noise_study"] = "N3_attack_breadth"
    n3["noise_study_purpose"] = (
        "Predeclared low/high structural endpoints for noisy LiRA across three "
        "trained target seeds."
    )
    n3["selection_frozen_before_noisy_outcomes"] = True
    n3 = n3.sort_values(["fm_kind", "model_seed"]).reset_index(drop=True)
    n3_label = n3[n3.model_seed.eq(43)].copy().reset_index(drop=True)
    n3_label["noise_study"] = "N3_label_only_optional"
    n3_label["noise_study_purpose"] = (
        "Optional stochastic label-only pilot on the two predeclared endpoints; "
        "query and circuit-shot costs must be reported."
    )
    return {"n1": n1, "n2": n2, "n3": n3, "n3_label": n3_label}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("reviewer_targets/multiseed_factorial_targets.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("satml_targets/noise"))
    args = parser.parse_args()
    manifests = build_manifests(normalized_source(args.source))
    filenames = {
        "n1": "mnist_noise_n1_structural_targets.csv",
        "n2": "mnist_noise_n2_query_targets.csv",
        "n3": "mnist_noise_n3_lira_targets.csv",
        "n3_label": "mnist_noise_n3_label_targets.csv",
    }
    outputs = {}
    for name, frame in manifests.items():
        path = args.out_dir / filenames[name]
        atomic_write_csv(frame, path)
        outputs[name] = {"path": str(path), "targets": int(len(frame))}
    atomic_write_json(
        {
            "source": str(args.source),
            "outputs": outputs,
            "factors": {
                "feature_maps": list(FEATURE_MAPS),
                "repetitions": list(REPETITIONS),
                "depths": list(DEPTHS),
                "model_seeds": list(MODEL_SEEDS),
            },
            "note": (
                "N1 is the primary 36-checkpoint structural factorial. N2 and N3 "
                "are targeted robustness studies and are not additional model-seed replication."
            ),
        },
        args.out_dir / "noise_target_manifest.json",
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
