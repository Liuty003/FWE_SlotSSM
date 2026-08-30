#!/usr/bin/env python3
"""Merge compatible v3 feature files while preserving whole source sequences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.data.v3_sequence_features import V3SequenceFeatures, load, save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-same-source",
        action="store_true",
        help=(
            "Permit normal and recovery windows from the same physical source "
            "in this one merged split.  Never use this to merge train and "
            "validation data together."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    sets = [load(path) for path in args.inputs]
    reference = sets[0]
    for item in sets[1:]:
        for key in ("visual_tokens", "action_hidden", "proprio", "actions"):
            if getattr(item, key).shape[1:] != getattr(reference, key).shape[1:]:
                raise ValueError(f"incompatible v3 {key} shape in merge")
        if (item.gate_labels is None) != (reference.gate_labels is None) or \
           (item.recovery_targets is None) != (reference.recovery_targets is None) or \
           (item.source_ids is None) != (reference.source_ids is None):
            raise ValueError("all merged v3 files must share optional-label/source fields")
    if reference.source_ids is not None:
        seen = set()
        for item in sets:
            current = set(item.source_ids.tolist())
            overlap = seen & current
            if overlap and not args.allow_same_source:
                raise ValueError(
                    "the same source trajectory occurs in multiple merge inputs; "
                    "deduplicate it before merging, or use --allow-same-source "
                    "only when intentionally retaining normal/recovery windows "
                    "of that physical source in one split"
                )
            seen |= current
    merged = V3SequenceFeatures(
        visual_tokens=torch.cat([item.visual_tokens for item in sets]),
        action_hidden=torch.cat([item.action_hidden for item in sets]),
        proprio=torch.cat([item.proprio for item in sets]), actions=torch.cat([item.actions for item in sets]),
        gate_labels=None if reference.gate_labels is None else torch.cat([item.gate_labels for item in sets]),
        recovery_targets=None if reference.recovery_targets is None else torch.cat([item.recovery_targets for item in sets]),
        recovery_mask=None if reference.recovery_mask is None else torch.cat([item.recovery_mask for item in sets]),
        source_ids=None if reference.source_ids is None else torch.cat([item.source_ids for item in sets]),
    )
    save(merged, args.output)
    print({"inputs": [str(path) for path in args.inputs], "output": str(args.output), "sequences": merged.num_sequences})


if __name__ == "__main__":
    main()
