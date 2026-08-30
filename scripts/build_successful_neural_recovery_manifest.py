#!/usr/bin/env python3
"""Distil successful pure-neural recovery actions into causal train windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata"))
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--chunk-length", type=int, default=8)
    return parser.parse_args()


def normalized_action(raw: list[float], stats: dict) -> list[float]:
    value = np.asarray(raw, dtype=np.float32)
    low, high = np.asarray(stats["q01"], dtype=np.float32), np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", [True] * len(value)), dtype=bool)
    output = value.copy()
    output[mask] = 2.0 * (value[mask] - low[mask]) / (high[mask] - low[mask] + 1e-8) - 1.0
    return output.tolist()


def main() -> None:
    config = args()
    if config.stride < 1 or config.chunk_length < 1:
        raise ValueError("stride and chunk-length must be positive")
    stats = json.loads((config.checkpoint / "dataset_statistics.json").read_text())["libero_10"]["action"]
    records = []
    for episode_path in config.episode_json:
        episode = json.loads(episode_path.read_text())
        if not episode.get("success"):
            raise ValueError(f"only successful rollouts may be distilled: {episode_path}")
        rollout = episode_path.with_name("online_trajectory.jsonl")
        if not rollout.is_file():
            raise FileNotFoundError(rollout)
        decisions = [row for row in episode["decisions"] if row["mode"] == "RECOVERY"]
        if len(decisions) < config.chunk_length:
            raise ValueError(f"too few recovery decisions in {episode_path}")
        task_id = int(episode_path.stem.split("-episode")[0].removeprefix("task"))
        language = "pick up the book and place it in the back compartment of the caddy"
        candidate = decisions[0]["selected_candidate_name"]
        if candidate is None:
            raise ValueError(f"recovery selector missing in {episode_path}")
        descriptor = [0.025, 0.0, 0.105, 0.004, 12.0, 0.0, 0.0]
        for offset, decision in enumerate(decisions):
            if offset % config.stride or offset + config.chunk_length > len(decisions):
                continue
            policy_step = int(decision["step"])
            records.append({
                "schema_version": 3,
                "kind": "book_regrasp_action_window",
                "split": "train",
                "source_key": f"libero_10-task{task_id}-episode0-successful-neural",
                "task_id": task_id,
                "candidate_name": candidate,
                "candidate_program_descriptor": descriptor,
                "branch_timestep": policy_step,
                "recovery_phase": 0,
                "language": language,
                "online_observation": decision["online_observation"],
                "recovery_target_action_chunk": [normalized_action(row["action"], stats) for row in decisions[offset:offset + config.chunk_length]],
                # Preserve the existing visual geometry action-conditioning
                # head by self-distilling its causal prediction; this is not
                # a privileged pose label and is never used online as input.
                "teacher_target_relative_position": decision["predicted_target_relative_position"],
                "provenance": {"online_rollout": str(rollout), "successful_pure_neural_self_distillation": True},
            })
    if not records:
        raise RuntimeError("no windows emitted")
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text("".join(json.dumps(row) + "\n" for row in records))
    print(json.dumps({"output": str(config.output), "samples": len(records), "tasks": sorted({r['task_id'] for r in records})}))


if __name__ == "__main__":
    main()
