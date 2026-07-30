#!/usr/bin/env python
"""Evaluate a task-6 execution Mamba checkpoint with its RGB motion gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.train_task6_execution_mamba import (
    ExecutionMamba,
    evaluate,
    load_episodes,
)
from scripts.train_task6_visual_state_baseline import parse_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-episodes", default="1,6,8")
    parser.add_argument("--test-episodes", default="7,9")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExecutionMamba().to(device).eval()
    model.load_state_dict(load_file(args.checkpoint))
    split_ids = {
        "validation": parse_ids(args.validation_episodes),
        "test": parse_ids(args.test_episodes),
    }
    metrics = {
        name: evaluate(
            model, load_episodes(args.sequence_dir, ids), device
        )
        for name, ids in split_ids.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "episode_split": split_ids,
        "metrics": metrics,
    }, indent=2), encoding="utf-8")
    for name, report in metrics.items():
        print(json.dumps({
            "split": name,
            "effective_progress_balanced_accuracy":
                report["effective_progress"]["balanced_accuracy"],
            "stall_balanced_accuracy":
                report["stall"]["balanced_accuracy"],
            "controlled_balanced_accuracy":
                report["controlled"]["balanced_accuracy"],
            "failed_acquisition": report["failed_acquisition"],
        }), flush=True)


if __name__ == "__main__":
    main()
