#!/usr/bin/env python3
"""Freeze a reproducible task-family subset of the official LIBERO-Plus manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-manifest", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument(
        "--official-name-regex", required=True,
        help="Python regular expression matched with re.search against official_name.",
    )
    parser.add_argument("--family-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def counts(records: list[dict], key: str) -> dict[str, int]:
    return dict(sorted(Counter(
        "unassigned" if row[key] is None else str(row[key]) for row in records
    ).items()))


def main() -> None:
    args = parse_args()
    if args.expected_count is not None and args.expected_count < 1:
        raise ValueError("expected-count must be positive")
    manifest = json.loads(args.global_manifest.read_text())
    if manifest.get("kind") != "libero_plus_official_global_manifest_v1":
        raise ValueError("global-manifest has an unsupported kind")
    expression = re.compile(args.official_name_regex)
    records = [
        row for row in manifest["records"]
        if row["suite"] == args.suite and expression.search(row["official_name"])
    ]
    if not records:
        raise RuntimeError("task-family selector matched no official records")
    if args.expected_count is not None and len(records) != args.expected_count:
        raise RuntimeError(f"expected {args.expected_count} records, matched {len(records)}")
    sample_ids = [row["sample_id"] for row in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("task-family manifest contains duplicate sample ids")

    payload = {
        "kind": "libero_plus_task_family_manifest_v1",
        "schema_version": 1,
        "family_name": args.family_name,
        "global_manifest": str(args.global_manifest.resolve()),
        "global_manifest_sha256": sha256(args.global_manifest),
        "selector": {
            "suite": args.suite,
            "official_name_regex": args.official_name_regex,
        },
        "num_instances": len(records),
        "perturbation_counts": counts(records, "perturbation_category"),
        "difficulty_counts": counts(records, "difficulty_level"),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "family_name": args.family_name,
        "num_instances": len(records),
        "perturbation_counts": payload["perturbation_counts"],
        "difficulty_counts": payload["difficulty_counts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
