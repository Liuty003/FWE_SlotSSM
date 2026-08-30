#!/usr/bin/env python3
"""Split one LIBERO-10 Gate feature tensor into semantic-family shards."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.data.v4_relation_features import load, save


TASK = re.compile(r"task(\d+)(?:-|:)episode(\d+)")


def identity(value: str) -> tuple[int, int]:
    match = TASK.search(value)
    if match is None:
        raise ValueError(f"cannot parse source identity from {value}")
    return int(match.group(1)), int(match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--failure-routes", type=Path, required=True)
    parser.add_argument("--normal-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    report_path = args.features.with_suffix(".json")
    report = json.loads(report_path.read_text())
    routes = json.loads(args.failure_routes.read_text())
    normal = json.loads(args.normal_manifest.read_text())
    family_by_identity = {
        identity(row["sample_id"]): row["family_name"]
        for row in routes["records"] if row.get("family_name")
    }
    family_by_identity.update({
        (int(row["task_id"]), int(row.get("episode_id", 0))): row["family_name"]
        for row in normal["records"]
    })
    records_by_family: dict[str, list[dict]] = {}
    for row in report["source_records"]:
        key = identity(row["source"])
        family = family_by_identity.get(key)
        if family is None:
            raise KeyError(f"no task family for {key}")
        records_by_family.setdefault(family, []).append(row)

    features = load(args.features)
    summary = {}
    for family, records in sorted(records_by_family.items()):
        source_ids = torch.tensor([int(row["source_id"]) for row in records], dtype=torch.int64)
        mask = features.source_ids[:, None].eq(source_ids[None]).any(dim=1)
        selected = features.index(torch.where(mask)[0])
        output = args.output_root / family / f"{args.output_name}.safetensors"
        save(selected, output)
        output.with_suffix(".json").write_text(json.dumps({
            **report,
            "kind": "libero10_gate_family_feature_shard_v1",
            "parent_features": str(args.features.resolve()),
            "family_name": family,
            "output": str(output.resolve()),
            "rows": selected.num_rows,
            "sources": len(records),
            "source_records": records,
        }, indent=2) + "\n")
        summary[family] = {"rows": selected.num_rows, "sources": len(records), "output": str(output)}
        print(json.dumps({"family": family, **summary[family]}), flush=True)
    (args.output_root / f"{args.output_name}-summary.json").write_text(json.dumps({
        "kind": "libero10_gate_family_feature_split_v1",
        "parent_features": str(args.features.resolve()),
        "families": summary,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
