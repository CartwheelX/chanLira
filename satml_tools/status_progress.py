#!/usr/bin/env python3
"""Summarize an incrementally written launcher status CSV."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def summarize(path: Path, expected: Optional[int] = None) -> str:
    if not path.exists():
        return f"status_file={path} not_created expected={expected if expected is not None else 'unknown'}"
    frame = pd.read_csv(path)
    if "status" not in frame:
        raise ValueError(f"{path} has no status column")
    counts = frame.status.astype(str).value_counts().to_dict()
    observed = len(frame)
    total = expected if expected is not None else observed
    remaining = max(total - observed, 0)
    fields = [f"status_file={path}", f"observed={observed}", f"expected={total}", f"remaining={remaining}"]
    fields.extend(f"{status}={count}" for status, count in sorted(counts.items()))
    lines = [" ".join(fields)]
    failures = frame[frame.status.astype(str).isin(["error", "failed"])]
    for _, row in failures.tail(3).iterrows():
        name = row.get("name", row.get("target_id", "unknown"))
        lines.append(f"[ERROR] {name} log={row.get('log', '')}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=None)
    args = parser.parse_args()
    if args.expected is not None and args.expected <= 0:
        parser.error("--expected must be positive")
    print(summarize(args.csv, args.expected))


if __name__ == "__main__":
    main()
