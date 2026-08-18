#!/usr/bin/env python3
"""Print visible progress for a target manifest without hiding worker logs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("satml_runs"))
    parser.add_argument("--tail-errors", type=int, default=3)
    args = parser.parse_args()
    targets = pd.read_csv(args.targets)
    counts = {"complete": 0, "running_or_partial": 0, "not_started": 0, "error": 0}
    errors = []
    for _, row in targets.iterrows():
        directory = args.run_root / str(row.get("experiment", "reviewer")) / str(row.target_id)
        model = directory / "target_model.pt"
        attack = directory / "target_attack_data.pt"
        log = directory / "train.log"
        if model.exists() and model.stat().st_size and attack.exists() and attack.stat().st_size:
            counts["complete"] += 1
        elif log.exists():
            text = log.read_text(encoding="utf-8", errors="replace")
            if "Traceback (most recent call last)" in text:
                counts["error"] += 1
                errors.append((str(row.target_id), "\n".join(text.splitlines()[-args.tail_errors :])))
            else:
                counts["running_or_partial"] += 1
        else:
            counts["not_started"] += 1
    total = len(targets)
    print(f"targets={total} complete={counts['complete']} partial={counts['running_or_partial']} errors={counts['error']} not_started={counts['not_started']}")
    print(f"progress={100.0 * counts['complete'] / max(total, 1):.1f}%")
    for target_id, tail in errors[-5:]:
        print(f"\n[ERROR] {target_id}\n{tail}")


if __name__ == "__main__":
    main()
