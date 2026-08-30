#!/usr/bin/env python3
"""Crop existing fork-aligned V4 features to certified V4.3 local windows.

This is primarily a fast stride-8 smoke path while the denser stride-4
features are being frozen.  It never guesses a handoff: all boundaries come
from the audited manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.data.v4_relation_features import load, save


def source_id(task_id: int, episode: int) -> int:
    identity = f"benchmark:libero_10:{task_id}:{episode}"
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") & ((1 << 63) - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--handoff-manifest", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--sampling-stride", type=int, default=8)
    args = parser.parse_args()
    features = load(args.features)
    parent_report = json.loads(args.features.with_suffix(".json").read_text())
    manifest = json.loads(args.handoff_manifest.read_text())
    selected = {"train": [], "val": []}
    summary = {"train": [], "val": []}
    for record in manifest["records"]:
        split = record.get("recommended_split")
        handoff = record.get("certified_local_handoff")
        if split not in selected or handoff is None:
            continue
        identity = source_id(int(record["task_id"]), int(record["episode"]))
        candidates = torch.where(features.source_ids.eq(identity) & features.recovery_mask)[0]
        count = len(range(int(record["fork_timestep"]), int(handoff) - 7, args.sampling_stride))
        if len(candidates) < count:
            raise ValueError(f"task{record['task_id']} has {len(candidates)} recovery rows, needs {count}")
        chosen = candidates[:count]
        selected[split].append(chosen)
        summary[split].append({"task_id": record["task_id"], "source_id": identity, "rows": count})
    for split, output in (("train", args.train_output), ("val", args.val_output)):
        indices = torch.cat(selected[split])
        value = features.index(indices)
        save(value, output)
        output.with_suffix(".json").write_text(json.dumps({
            **parent_report, "kind": "v43_certified_recovery_crop", "output": str(output),
            "parent_features": str(args.features), "handoff_manifest": str(args.handoff_manifest),
            "split": split, "rows": value.num_rows, "sources": summary[split],
            "sampling_contract": f"V4.3 certified local windows cropped on legacy stride {args.sampling_stride}",
        }, indent=2) + "\n")
    print(json.dumps({split: {"rows": sum(row["rows"] for row in rows), "sources": len(rows)}
                      for split, rows in summary.items()}))


if __name__ == "__main__":
    main()
