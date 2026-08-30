#!/usr/bin/env python3
"""Select family- and perturbation-balanced successful OFT gate negatives."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=30)
    return parser.parse_args()


def balanced(records: list[dict], count: int) -> list[dict]:
    groups: dict[tuple[str, int], deque[dict]] = defaultdict(deque)
    for record in sorted(records, key=lambda item: int(item["task_id"])):
        groups[(record["perturbation_category"], int(record["difficulty_level"]))].append(record)
    selected = []
    keys = sorted(groups)
    while len(selected) < count and keys:
        remaining = []
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].popleft())
            if groups[key]:
                remaining.append(key)
        keys = remaining
    return selected


def main() -> None:
    args = arguments()
    if args.per_family < 1:
        raise ValueError("--per-family must be positive")
    records = []
    for summary_path in sorted(args.family_root.glob("*/baseline-summary.json")):
        summary = json.loads(summary_path.read_text())
        candidates = [artifact for artifact in summary["artifacts"] if artifact["success"]]
        for record in balanced(candidates, args.per_family):
            records.append({
                "family_name": summary["family_name"],
                "sample_id": record["sample_id"],
                "task_id": int(record["task_id"]),
                "perturbation_category": record["perturbation_category"],
                "difficulty_level": int(record["difficulty_level"]),
                "metadata": record["metadata"],
                "steps": record["steps"],
            })
    payload = {
        "kind": "libero10_balanced_normal_gate_sources_v1",
        "per_family": args.per_family,
        "num_records": len(records),
        "task_ids_csv": ",".join(str(record["task_id"]) for record in records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "num_records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
