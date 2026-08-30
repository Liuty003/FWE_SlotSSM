#!/usr/bin/env python3
"""Replace a v3 feature file's source id with its physical-episode identity.

Useful for frozen features created before v3 switched from path identities to
``(task_suite, task_id, episode_id)`` identities.  Feature tensors are copied
unchanged; only split provenance is corrected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.data.v3_sequence_features import from_tensors, load, save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--trajectory-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def physical_source_id(metadata: dict) -> int:
    keys = ("task_suite", "task_id", "episode_id")
    if not all(key in metadata for key in keys):
        raise ValueError(f"metadata missing physical source fields: {keys}")
    identity = "benchmark:" + ":".join(str(metadata[key]) for key in keys)
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") & ((1 << 63) - 1)


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.trajectory_metadata.read_text())
    features = load(args.features)
    import torch
    tensors = features.payload()
    source = physical_source_id(metadata)
    tensors["source_ids"] = torch.full((features.num_sequences,), source, dtype=torch.int64)
    save(from_tensors(tensors), args.output)
    report = {
        "kind": "v3_feature_source_reindex", "input": str(args.features),
        "metadata": str(args.trajectory_metadata), "output": str(args.output),
        "physical_source_id": source, "sequences": features.num_sequences,
    }
    args.output.with_suffix(".source.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
