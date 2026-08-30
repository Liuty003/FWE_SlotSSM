#!/usr/bin/env python3
"""Select deterministic teacher-pilot forks across LIBERO-10 family/type clusters."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-cluster", type=int, default=1)
    parser.add_argument("--onset-filename", default="failure-onsets-v2.json")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.per_cluster < 1:
        raise ValueError("per-cluster must be positive")
    index = json.loads(args.family_index.read_text())
    records = []
    cluster_counts = Counter()
    for family in index["families"]:
        family_name = family["family_name"]
        onset_path = Path(family["family_manifest"]).with_name(args.onset_filename)
        onsets = json.loads(onset_path.read_text())["records"]
        failure_types = sorted({
            record["failure_type"] for record in onsets if record["onset_status"] == "located"
        })
        for failure_type in failure_types:
            candidates = [
                record for record in onsets
                if record["onset_status"] == "located" and record["failure_type"] == failure_type
            ]
            candidates.sort(key=lambda record: (
                {"high": 0, "medium": 1, "low": 2, "none": 3}.get(record["confidence"], 4),
                record.get("difficulty_level", 99),
                record["sample_id"],
            ))
            for record in candidates[:args.per_cluster]:
                records.append({**record, "family_name": family_name})
                cluster_counts[f"{family_name}:{failure_type}"] += 1
    payload = {
        "kind": "libero_10_teacher_pilot_onset_manifest_v1",
        "schema_version": 1,
        "family_index": str(args.family_index.resolve()),
        "selection": "confidence then difficulty then sample_id, per family/failure-type",
        "per_cluster": args.per_cluster,
        "onset_filename": args.onset_filename,
        "num_records": len(records),
        "cluster_counts": dict(sorted(cluster_counts.items())),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output), "num_records": len(records),
        "cluster_counts": payload["cluster_counts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
