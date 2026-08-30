#!/usr/bin/env python3
"""Emit causal early-recovery windows whose residual target is explicitly zero.

The actions are retained only because the frozen-feature schema requires a
recovery chunk.  ``train_neural_recovery_residual.py`` treats every row in
this manifest's feature file as a preservation (zero residual) label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata"))
    parser.add_argument("--end-offset", type=int, default=23,
                        help="exclusive offset inside the learned recovery segment")
    parser.add_argument("--chunk-length", type=int, default=8)
    return parser.parse_args()


def normalize(action: list[float], stats: dict) -> list[float]:
    value = np.asarray(action, dtype=np.float32)
    low, high = np.asarray(stats["q01"], dtype=np.float32), np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", [True] * len(value)), dtype=bool)
    output = value.copy()
    output[mask] = 2.0 * (value[mask] - low[mask]) / (high[mask] - low[mask] + 1e-8) - 1.0
    return output.tolist()


def main() -> None:
    args = parse_args()
    if args.chunk_length < 1 or args.end_offset < 1:
        raise ValueError("chunk-length and end-offset must be positive")
    episode = json.loads(args.episode_json.read_text())
    recovery = [row for row in episode["decisions"] if row["mode"] == "RECOVERY"]
    if len(recovery) < args.chunk_length:
        raise RuntimeError("episode has too few recovery rows")
    end = min(args.end_offset, len(recovery) - args.chunk_length + 1)
    if end < 1:
        raise RuntimeError("selected recovery prefix has no complete action chunk")
    rollout = args.episode_json.with_name("online_trajectory.jsonl")
    if not rollout.is_file():
        raise FileNotFoundError(rollout)
    task_id = int(args.episode_json.stem.split("-episode", 1)[0].removeprefix("task"))
    stats = json.loads((args.checkpoint / "dataset_statistics.json").read_text())["libero_10"]["action"]
    candidate = recovery[0]["selected_candidate_name"]
    if candidate is None or any(row["selected_candidate_name"] != candidate for row in recovery):
        raise ValueError("recovery mode must be selected and constant")
    records = []
    for offset in range(end):
        row = recovery[offset]
        records.append({
            "schema_version": 3,
            "kind": "book_regrasp_action_window",
            "split": "train",
            "source_key": f"libero_10-task{task_id}-early-neural-recovery-preserve",
            "task_id": task_id,
            "candidate_name": candidate,
            "candidate_program_descriptor": [0.025, 0.0, 0.105, 0.004, 12.0, 0.0, 0.0],
            "branch_timestep": int(row["step"]),
            "recovery_phase": 0,
            "language": "pick up the book and place it in the back compartment of the caddy",
            "online_observation": row["online_observation"],
            "recovery_target_action_chunk": [
                normalize(item["action"], stats) for item in recovery[offset:offset + args.chunk_length]
            ],
            "teacher_target_relative_position": row["predicted_target_relative_position"],
            "provenance": {
                "online_rollout": str(rollout),
                "zero_residual_preservation": True,
                "recovery_offset": offset,
            },
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record) + "\n" for record in records))
    print(json.dumps({"output": str(args.output), "samples": len(records), "task_id": task_id,
                      "recovery_offsets": [0, end - 1]}))


if __name__ == "__main__":
    main()
