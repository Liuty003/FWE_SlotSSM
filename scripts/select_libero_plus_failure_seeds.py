#!/usr/bin/env python3
"""Select validated failure seeds for one frozen LIBERO-Plus task family."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-manifest", type=Path, required=True)
    parser.add_argument("--failure-classification", type=Path, required=True)
    parser.add_argument("--failure-type", action="append", required=True)
    parser.add_argument("--confidence", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    family = json.loads(args.family_manifest.read_text())
    classification = json.loads(args.failure_classification.read_text())
    family_ids = {row["sample_id"] for row in family["records"]}
    requested_types = set(args.failure_type)
    requested_confidence = set(args.confidence)
    records = []
    for row in classification["records"]:
        if row["sample_id"] not in family_ids or row["failure_type"] not in requested_types:
            continue
        if requested_confidence and row["confidence"] not in requested_confidence:
            continue
        metadata = Path(row["metadata"])
        steps = Path(row["steps"])
        video = metadata.parent / "rollout.mp4"
        missing = [str(path) for path in (metadata, steps, video) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete seed {row['sample_id']}: {missing}")
        records.append({**row, "video": str(video.resolve())})
    records.sort(key=lambda row: (row["suite"], row["sample_id"]))
    if not records:
        raise RuntimeError("no failure seeds matched the requested family and labels")
    payload = {
        "kind": "libero_plus_failure_seed_manifest_v1",
        "schema_version": 1,
        "family_manifest": str(args.family_manifest.resolve()),
        "failure_classification": str(args.failure_classification.resolve()),
        "selector": {
            "failure_types": sorted(requested_types),
            "confidence": sorted(requested_confidence),
        },
        "num_seeds": len(records),
        "failure_type_counts": dict(sorted(Counter(
            row["failure_type"] for row in records
        ).items())),
        "perturbation_counts": dict(sorted(Counter(
            row["perturbation_category"] for row in records
        ).items())),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "num_seeds": len(records),
        "sample_ids": [row["sample_id"] for row in records],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
