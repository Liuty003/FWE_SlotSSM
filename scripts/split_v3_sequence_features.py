#!/usr/bin/env python3
"""Split v3 frozen features by source trajectory, never by overlapping windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.data.v3_sequence_features import load, save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--val-source-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_source_fraction < 1.0:
        raise ValueError("val-source-fraction must lie in (0,1)")
    import torch

    features = load(args.features)
    if features.source_ids is None:
        raise ValueError("v3 source_ids are required; refusing window-level random split")
    source_ids = torch.unique(features.source_ids.cpu(), sorted=True)
    if source_ids.numel() < 2:
        raise ValueError("at least two independent sources are required for a source-disjoint split")
    generator = torch.Generator().manual_seed(args.seed)
    permuted = source_ids[torch.randperm(source_ids.numel(), generator=generator)]
    val_count = max(1, round(source_ids.numel() * args.val_source_fraction))
    val_sources = permuted[:val_count]
    val_mask = (features.source_ids[:, None] == val_sources[None]).any(dim=1)
    train_indices, val_indices = (~val_mask).nonzero().squeeze(1), val_mask.nonzero().squeeze(1)
    if not train_indices.numel() or not val_indices.numel():
        raise RuntimeError("empty train or validation feature split")
    train, val = features.index(train_indices), features.index(val_indices)
    save(train, args.train_output)
    save(val, args.val_output)
    report = {
        "kind": "v3_source_disjoint_feature_split", "input": str(args.features),
        "train_output": str(args.train_output), "val_output": str(args.val_output),
        "seed": args.seed, "train_sources": train.source_ids.unique().tolist(),
        "val_sources": val.source_ids.unique().tolist(),
        "train_sequences": train.num_sequences, "val_sequences": val.num_sequences,
    }
    args.train_output.with_suffix(".split.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
