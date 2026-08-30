#!/usr/bin/env python3
"""Select complete v3 feature rows by physical source id.

This is deliberately source-level: it lets a replacement mixed trajectory
supersede an obsolete teacher without ever splitting overlapping windows from
the same physical rollout across train and validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.data.v3_sequence_features import load, save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--include-source-id", type=int, nargs="+")
    group.add_argument("--exclude-source-id", type=int, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = load(args.input)
    if features.source_ids is None:
        raise ValueError("source_ids are mandatory for source-level filtering")
    requested = set(args.include_source_id or args.exclude_source_id)
    matches = torch.zeros(features.num_sequences, dtype=torch.bool)
    for source_id in requested:
        matches |= features.source_ids.cpu() == source_id
    keep = matches if args.include_source_id is not None else ~matches
    if not bool(keep.any()):
        raise ValueError("filter would produce an empty feature set")
    selected = features.index(torch.where(keep)[0])
    save(selected, args.output)
    print({
        "input": str(args.input), "output": str(args.output),
        "input_sequences": features.num_sequences, "output_sequences": selected.num_sequences,
        "sources": selected.source_ids.unique(sorted=True).tolist(),
    })


if __name__ == "__main__":
    main()
