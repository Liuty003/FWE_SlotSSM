#!/usr/bin/env python3
"""Split V4.3 recovery features according to the audited handoff manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.data.v4_relation_features import load, save


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--handoff-manifest", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    args = parser.parse_args()
    features = load(args.features)
    report = json.loads(args.features.with_suffix(".json").read_text())
    manifest = json.loads(args.handoff_manifest.read_text())
    split_by_source = {
        str(Path(record["source"]).resolve()): record["recommended_split"]
        for record in manifest["records"] if record["status"] == "certified_local_recovery"
    }
    ids = {"train": [], "val": []}
    records = {"train": [], "val": []}
    for source in report.get("sources", []):
        split = split_by_source.get(str(Path(source["source"]).resolve()))
        if split not in ids:
            raise ValueError(f"feature source has no train/val assignment: {source['source']}")
        ids[split].append(int(source["source_id"]))
        records[split].append(source)
    for split, output in (("train", args.train_output), ("val", args.val_output)):
        if not ids[split]:
            raise ValueError(f"manifest has no {split} feature sources")
        mask = (features.source_ids[:, None] == torch.tensor(ids[split])[None]).any(dim=1)
        value = features.index(torch.where(mask)[0])
        save(value, output)
        output.with_suffix(".json").write_text(json.dumps({
            **report, "kind": "v43_handoff_feature_split", "parent_features": str(args.features),
            "handoff_manifest": str(args.handoff_manifest), "split": split, "output": str(output),
            "rows": value.num_rows, "sources": records[split],
        }, indent=2) + "\n")
    print(json.dumps({
        "train_rows": int(sum((features.source_ids == source).sum() for source in ids["train"])),
        "val_rows": int(sum((features.source_ids == source).sum() for source in ids["val"])),
        "train_sources": len(ids["train"]), "val_sources": len(ids["val"]),
    }))


if __name__ == "__main__":
    main()
