#!/usr/bin/env python3
"""Probe IBM backend access and verify that a backend-derived Aer model loads.

No circuit is submitted to hardware.  This script only queries backend metadata
and constructs a local Qiskit Aer NoiseModel.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from qurift_qiskit_bridge import load_backend_noise_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-name", required=True)
    parser.add_argument("--noise-backend-name", default=None)
    parser.add_argument("--ibm-account-name", default=None)
    parser.add_argument("--require-noise", action="store_true")
    parser.add_argument("--allow-backend-mismatch", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reviewer_results/backend_noise_probe.json"),
    )
    args = parser.parse_args()

    context = load_backend_noise_context(
        args.backend_name,
        args.noise_backend_name,
        account_name=args.ibm_account_name,
        require_noise=args.require_noise,
        allow_backend_mismatch=args.allow_backend_mismatch,
    )
    output = asdict(context.metadata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2))
    print(f"[OK] Wrote: {args.out.resolve()}")
    if args.require_noise and not output["noise_model_loaded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
