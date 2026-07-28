#!/usr/bin/env python
"""Construct real schema-v2 tensor windows and report trainable coverage."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovery.data.dataset import read_jsonl
from recovery.data.tensor_dataset import (
    RecoverySequenceDataset,
    collate_recovery_sequences,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--event-only", action="store_true")
    return parser.parse_args()


def expand(value: Path) -> list[Path]:
    if value.is_file():
        return [value]
    return sorted(value.glob("task*-episode*/transitions-v2.jsonl"))


def main() -> None:
    args = arguments()
    rows = []
    paths = [path for value in args.inputs for path in expand(value)]
    for path in paths:
        rows.extend(read_jsonl(path))
    by_split = {}
    for row in rows:
        by_split.setdefault(row.split, []).append(row)

    reports = {}
    for split, split_rows in sorted(by_split.items()):
        dataset = RecoverySequenceDataset(
            split_rows,
            window_length=args.window_length,
            stride=args.stride,
            event_only=args.event_only,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_recovery_sequences,
        )
        counts = Counter()
        shape_contract = None
        for batch in loader:
            sequence = batch["sequence"]
            if shape_contract is None:
                shape_contract = {
                    key: list(value.shape)
                    for key, value in sequence.items()
                }
            counts["timesteps"] += int(sequence["phase_valid"].numel())
            counts["visual_xyz_valid"] += int(
                sequence["visual_target_xyz_valid"].sum()
            )
            counts["expert_action_valid"] += int(
                sequence["expert_action_valid"].sum()
            )
            counts["phase_valid"] += int(sequence["phase_valid"].sum())
            counts["failure_injection"] += int(
                sequence["failure_injection"].sum()
            )
            counts["failure_detection_event"] += int(
                sequence["failure_detection_event"].sum()
            )
            for value in sequence.values():
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    raise ValueError(f"non-finite tensor in split {split}")
        reports[split] = {
            "num_rows": len(split_rows),
            "num_windows": len(dataset),
            "num_episodes": len({row.episode_id for row in split_rows}),
            "counts_in_overlapping_windows": dict(counts),
            "first_batch_shape_contract": shape_contract,
        }
    result = {
        "schema_version": 1,
        "kind": "recovery_tensor_dataset_audit",
        "inputs": [str(path.resolve()) for path in paths],
        "window_length": args.window_length,
        "stride": args.stride,
        "event_only": args.event_only,
        "splits": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
