#!/usr/bin/env python3
"""Select a provenance-addressable subset of frozen recovery features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-features", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--branch-metadata-contains", nargs="*", default=[],
        help="Optional provenance.branch_metadata substring filter.",
    )
    parser.add_argument(
        "--task-ids", type=int, nargs="+",
        help="Optional task-id filter. It composes with --branch-metadata-contains.",
    )
    return parser.parse_args()


def main() -> None:
    config = args()
    payload = load_file(str(config.features))
    records = [json.loads(line) for line in config.manifest.read_text().splitlines() if line.strip()]
    samples = int(next(iter(payload.values())).shape[0])
    if len(records) != samples:
        raise ValueError(f"feature/manifest row mismatch: {samples} vs {len(records)}")
    indices = [
        index for index, record in enumerate(records)
        if (not config.branch_metadata_contains or any(
            token in str(record.get("provenance", {}).get("branch_metadata", ""))
            for token in config.branch_metadata_contains
        ))
        and (config.task_ids is None or int(record.get("task_id", -1)) in set(config.task_ids))
    ]
    if not indices:
        raise RuntimeError("selection is empty")
    import torch
    selected = torch.tensor(indices, dtype=torch.long)
    filtered = {key: value.index_select(0, selected) for key, value in payload.items()}
    config.output_features.parent.mkdir(parents=True, exist_ok=True)
    config.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    save_file(filtered, str(config.output_features))
    config.output_manifest.write_text("".join(json.dumps(records[index]) + "\n" for index in indices))
    print(json.dumps({"samples": len(indices), "indices": indices, "output": str(config.output_features)}))


if __name__ == "__main__":
    main()
