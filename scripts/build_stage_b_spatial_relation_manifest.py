#!/usr/bin/env python
"""Validate relation-mask samples and define leakage-free Stage-B v2 splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


REQUIRED_ROLES = {"target", "distractor", "goal", "robot"}
COMPOSITIONAL_TASK_SPLIT = {
    "train": [0, 1, 2, 4, 7, 9],
    "validation": [5, 6],
    "test": [3, 8],
}
REFERENCE_SEEN_FOLDS = {
    "compositional_reference_seen_fold_a": {
        "train": [0, 1, 2, 4, 6, 7, 9], "validation": [5, 8], "test": [3],
    },
    "compositional_reference_seen_fold_b": {
        "train": [0, 1, 2, 3, 4, 7, 9], "validation": [5, 8], "test": [6],
    },
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-manifest", type=Path,
        default=Path(
            "/data2/lty/naive-e-slotssm-runs/"
            "stage-b-spatial-relation-data/manifest.json"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--train-states", default="0-34")
    parser.add_argument("--validation-states", default="35-41")
    parser.add_argument("--test-states", default="42-49")
    parser.add_argument("--min-visible-pixels", type=int, default=16)
    return parser.parse_args()


def parse_range(value: str) -> list[int]:
    result = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = (int(item) for item in part.split("-", 1))
            if stop < start:
                raise ValueError(f"Invalid state range: {part}")
            result.extend(range(start, stop + 1))
        else:
            result.append(int(part))
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate state IDs in {value}")
    return result


def validate_sample(sample: dict, min_visible_pixels: int) -> None:
    rgb = Path(sample["rgb"])
    masks_path = Path(sample["masks"])
    if not rgb.is_file() or not masks_path.is_file():
        raise FileNotFoundError(f"Missing sample files: {rgb} / {masks_path}")
    expected = REQUIRED_ROLES | {
        f"reference_{index}" for index in range(len(sample["references"]))
    }
    with np.load(masks_path) as masks:
        if set(masks.files) != expected:
            raise RuntimeError(
                f"task {sample['task_id']} state {sample['state_id']}: "
                f"mask keys {set(masks.files)} != {expected}"
            )
        for role in expected:
            mask = masks[role]
            if mask.shape != (256, 256) or mask.dtype != np.uint8:
                raise RuntimeError(f"Invalid {role} mask: {mask.shape} {mask.dtype}")
            values = np.unique(mask)
            if not set(values.tolist()) <= {0, 1}:
                raise RuntimeError(f"Non-binary {role} mask values: {values}")
            if int(mask.sum()) < min_visible_pixels:
                raise RuntimeError(f"Insufficient {role} visibility: {int(mask.sum())}")


def records_for(samples, *, tasks=None, states=None):
    task_set = set(tasks) if tasks is not None else None
    state_set = set(states) if states is not None else None
    return [
        {"task_id": int(item["task_id"]), "state_id": int(item["state_id"])}
        for item in samples
        if (task_set is None or int(item["task_id"]) in task_set)
        and (state_set is None or int(item["state_id"]) in state_set)
    ]


def main() -> None:
    args = arguments()
    if args.min_visible_pixels < 1:
        raise ValueError("--min-visible-pixels must be positive")
    source = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    samples = source["samples"]
    keys = [(int(item["task_id"]), int(item["state_id"])) for item in samples]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate (task_id, state_id) samples")
    for sample in samples:
        validate_sample(sample, args.min_visible_pixels)

    state_splits = {
        "train": parse_range(args.train_states),
        "validation": parse_range(args.validation_states),
        "test": parse_range(args.test_states),
    }
    flat_states = [state for values in state_splits.values() for state in values]
    if len(flat_states) != len(set(flat_states)):
        raise RuntimeError(f"State leakage across splits: {state_splits}")
    task_flat = [task for values in COMPOSITIONAL_TASK_SPLIT.values() for task in values]
    if len(task_flat) != len(set(task_flat)) or set(task_flat) != set(range(10)):
        raise RuntimeError("Compositional task split must partition tasks 0-9")

    protocols = {
        "state_generalization": {
            split: records_for(samples, states=states)
            for split, states in state_splits.items()
        },
        "compositional_task_generalization": {
            split: records_for(samples, tasks=tasks)
            for split, tasks in COMPOSITIONAL_TASK_SPLIT.items()
        },
    }
    for protocol, task_splits in REFERENCE_SEEN_FOLDS.items():
        fold_tasks = [task for values in task_splits.values() for task in values]
        if len(fold_tasks) != len(set(fold_tasks)) or set(fold_tasks) != set(range(10)):
            raise RuntimeError(f"{protocol} must partition tasks 0-9")
        protocols[protocol] = {
            split: records_for(samples, tasks=tasks)
            for split, tasks in task_splits.items()
        }
    output = {
        "schema_version": 2,
        "source_manifest": str(args.input_manifest),
        "num_samples": len(samples),
        "relations": dict(sorted(Counter(item["relation"] for item in samples).items())),
        "protocols": protocols,
        "protocol_rationale": {
            "state_generalization": "All tasks occur in every split; state IDs never overlap.",
            "compositional_task_generalization": (
                "Open-vocabulary stress test: tasks never overlap and cookie references are "
                "absent from train, conflating new-category grounding with new compositions."
            ),
            "compositional_reference_seen_fold_a": (
                "Every held-out reference category occurs in train; cookie is trained under "
                "next_to (task 6) and tested under on (task 3)."
            ),
            "compositional_reference_seen_fold_b": (
                "Every held-out reference category occurs in train; cookie is trained under "
                "on (task 3) and tested under next_to (task 6)."
            ),
        },
    }
    output_path = args.output or args.input_manifest.with_name("split_manifest.json")
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output_path),
        "num_samples": len(samples),
        "split_sizes": {
            protocol: {split: len(rows) for split, rows in splits.items()}
            for protocol, splits in protocols.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
