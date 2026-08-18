#!/usr/bin/env python3
"""Import only prespecified retained MNIST artifacts into the SaTML clone."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ARTIFACT_NAMES = ("target_model.pt", "target_attack_data.pt", "target_export_summary.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_targets(
    source_root: Path,
    destination_root: Path,
    target_ids: list[str],
    *,
    force: bool = False,
) -> list[dict[str, object]]:
    records = []
    for target_id in target_ids:
        source_dir = source_root / "reviewer_runs" / "multiseed_factorial" / target_id
        destination_dir = destination_root / "reviewer_runs" / "multiseed_factorial" / target_id
        for name in ARTIFACT_NAMES:
            source = source_dir / name
            destination = destination_dir / name
            if not source.exists() or source.stat().st_size == 0:
                raise FileNotFoundError(f"Missing retained artifact: {source}")
            source_hash = sha256(source)
            status = "copied"
            if destination.exists():
                destination_hash = sha256(destination)
                if destination_hash == source_hash:
                    status = "already_identical"
                elif not force:
                    raise FileExistsError(
                        f"Destination differs: {destination}; inspect it or rerun with --force"
                    )
                else:
                    status = "replaced"
            if status in {"copied", "replaced"}:
                destination_dir.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                shutil.copy2(source, temporary)
                temporary.replace(destination)
            records.append(
                {"target_id": target_id, "artifact": name, "sha256": source_hash,
                 "bytes": source.stat().st_size, "status": status,
                 "source": str(source.resolve()), "destination": str(destination.resolve())}
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--destination-repo", type=Path, default=Path("."))
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("satml_targets/noise/mnist_noise_n1_structural_targets.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("satml_results/imported_mnist_manifest.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    targets = pd.read_csv(args.targets)
    if "target_id" not in targets:
        raise ValueError(f"{args.targets} has no target_id column")
    target_ids = targets.target_id.astype(str).tolist()
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("Target list contains duplicates")
    records = import_targets(
        args.source_repo.resolve(), args.destination_repo.resolve(), target_ids, force=args.force
    )
    payload = {
        "source_repo": str(args.source_repo.resolve()),
        "destination_repo": str(args.destination_repo.resolve()),
        "target_ids": target_ids,
        "artifacts": records,
        "note": "Scientific artifacts copied byte-for-byte; SHA-256 recorded; no credentials included.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] targets={len(target_ids)} artifacts={len(records)} manifest={args.out.resolve()}")


if __name__ == "__main__":
    main()
