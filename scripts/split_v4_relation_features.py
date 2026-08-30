#!/usr/bin/env python3
"""Create a strict physical-source split for V4 single-decision features."""

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
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--stratify-gate", action="store_true",
                        help="Split recovery-bearing and normal-only physical sources separately.")
    args = parser.parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must lie in (0, 1)")
    features = load(args.features)
    provenance_path = args.features.with_suffix(".json")
    if not provenance_path.is_file():
        raise FileNotFoundError(f"missing feature provenance: {provenance_path}")
    provenance = json.loads(provenance_path.read_text())
    sources = torch.unique(features.source_ids, sorted=True)
    if sources.numel() < 2:
        raise ValueError("at least two physical sources are required for a source-disjoint split")
    generator = torch.Generator().manual_seed(args.seed)
    if args.stratify_gate:
        recovery_sources = torch.unique(features.source_ids[features.gate_labels.eq(1)], sorted=True)
        recovery_set = set(recovery_sources.tolist())
        normal_sources = torch.tensor([source for source in sources.tolist() if source not in recovery_set], dtype=sources.dtype)
        if recovery_sources.numel() < 2 or normal_sources.numel() < 2:
            raise ValueError("stratified split needs at least two recovery and two normal-only sources")
        selected = []
        for group in (recovery_sources, normal_sources):
            shuffled = group[torch.randperm(group.numel(), generator=generator)]
            count = max(1, min(group.numel() - 1, round(group.numel() * args.val_fraction)))
            selected.append(shuffled[:count])
        val_sources = torch.cat(selected)
    else:
        shuffled = sources[torch.randperm(sources.numel(), generator=generator)]
        val_count = max(1, min(sources.numel() - 1, round(sources.numel() * args.val_fraction)))
        val_sources = shuffled[:val_count]
    val_mask = (features.source_ids[:, None] == val_sources[None]).any(dim=1)
    train_mask = ~val_mask
    train, val = features.index(torch.where(train_mask)[0]), features.index(torch.where(val_mask)[0])
    save(train, args.train_output); save(val, args.val_output)
    for output, split_name, value in ((args.train_output, "train", train), (args.val_output, "val", val)):
        output.with_suffix(".json").write_text(json.dumps({
            **provenance, "kind": "v4_relation_feature_split", "parent_features": str(args.features),
            "split": split_name, "output": str(output), "rows": value.num_rows,
            "sources": int(value.source_ids.unique().numel()),
        }, indent=2) + "\n")
    report = {
        "kind": "v4_relation_source_split", "features": str(args.features), "seed": args.seed,
        "stratify_gate": args.stratify_gate,
        "train_rows": train.num_rows, "val_rows": val.num_rows,
        "train_sources": sorted(train.source_ids.unique().tolist()), "val_sources": sorted(val.source_ids.unique().tolist()),
        "labels": {"train": {str(value): int((train.gate_labels == value).sum()) for value in (-1, 0, 1)},
                   "val": {str(value): int((val.gate_labels == value).sum()) for value in (-1, 0, 1)}},
    }
    args.train_output.with_suffix(".split.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
