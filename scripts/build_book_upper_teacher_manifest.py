#!/usr/bin/env python3
"""Build causal action windows from recorded successful anchor-FSM teachers.

The anchor-FSM is used *only* as an offline demonstrator.  The emitted
examples expose RGB, wrist RGB, language, proprio and executed-action history
to the later model.  Target/placement anchors are converted into a
teacher-only relative-position loss and are never copied into a deployment
observation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CANDIDATE_OFFSETS = {
    "fsm_xyp0_p0": np.asarray([0.0, 0.0], dtype=np.float32),
    "fsm_xym25_p0": np.asarray([0.0, -0.025], dtype=np.float32),
    "fsm_xyp0_m25": np.asarray([0.0, 0.025], dtype=np.float32),
    "fsm_xyp0_p25": np.asarray([-0.025, 0.0], dtype=np.float32),
    "fsm_xyp25_p0": np.asarray([0.025, 0.0], dtype=np.float32),
}
LANGUAGE = "pick up the book and place it in the back compartment of the caddy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata"))
    parser.add_argument("--chunk-length", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def normalized_action(raw: list[float], stats: dict) -> list[float]:
    value = np.asarray(raw, dtype=np.float32)
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", [True] * len(value)), dtype=bool)
    output = value.copy()
    output[mask] = 2.0 * (value[mask] - low[mask]) / (high[mask] - low[mask] + 1e-8) - 1.0
    return output.clip(-1.0, 1.0).tolist()


def candidate_for_offset(value: list[float]) -> str:
    offset = np.asarray(value, dtype=np.float32)
    if offset.shape != (2,):
        raise ValueError(f"invalid geometry offset {value}")
    name, distance = min(
        ((name, float(np.linalg.norm(offset - expected))) for name, expected in CANDIDATE_OFFSETS.items()),
        key=lambda item: item[1],
    )
    if distance > 1e-4:
        raise ValueError(f"unrecognised candidate offset {value}")
    return name


def recovery_segments(decisions: list[dict]) -> list[list[dict]]:
    """Split records so no imitation target crosses recovery->normal."""
    segments: list[list[dict]] = []
    current: list[dict] = []
    previous_step: int | None = None
    for row in decisions:
        if row.get("mode") != "RECOVERY":
            if current:
                segments.append(current)
                current = []
            previous_step = None
            continue
        step = int(row["step"])
        if previous_step is not None and step != previous_step + 1:
            segments.append(current)
            current = []
        current.append(row)
        previous_step = step
    if current:
        segments.append(current)
    return segments


def main() -> None:
    args = parse_args()
    if args.chunk_length < 1 or args.stride < 1:
        raise ValueError("chunk length and stride must be positive")
    stats = json.loads((args.checkpoint / "dataset_statistics.json").read_text())["libero_10"]["action"]
    records: list[dict] = []
    found: list[int] = []
    for task_id in args.task_ids:
        episode_path = args.teacher_root / f"task{task_id}" / f"task{task_id}-episode0.json"
        rollout = episode_path.with_name("online_trajectory.jsonl")
        if not episode_path.is_file() or not rollout.is_file():
            raise FileNotFoundError(f"missing collected teacher task {task_id}: {episode_path}")
        episode = json.loads(episode_path.read_text())
        if not episode.get("success"):
            raise RuntimeError(f"teacher replay did not succeed: task {task_id}")
        found.append(task_id)
        for segment in recovery_segments(episode["decisions"]):
            if len(segment) < args.chunk_length:
                continue
            candidate = candidate_for_offset(segment[0]["geometry_xy_offset"])
            if any(candidate_for_offset(row["geometry_xy_offset"]) != candidate for row in segment):
                raise ValueError(f"candidate mode changed in task {task_id} recovery segment")
            anchor = np.asarray(segment[0]["task_specific_target_anchor"], dtype=np.float32)
            if anchor.shape != (3,):
                raise ValueError(f"missing target anchor in task {task_id}")
            for offset in range(0, len(segment) - args.chunk_length + 1, args.stride):
                row = segment[offset]
                observation = row.get("online_observation")
                if observation is None:
                    raise ValueError(f"teacher task {task_id} did not save online images")
                eef = np.asarray(observation["proprio"]["robot0_eef_pos"], dtype=np.float32)
                records.append({
                    "schema_version": 3,
                    "kind": "book_regrasp_action_window",
                    "split": "train",
                    "source_key": f"libero_10-task{task_id}-anchor-fsm-teacher",
                    "task_id": task_id,
                    "candidate_name": candidate,
                    "candidate_program_descriptor": [float(CANDIDATE_OFFSETS[candidate][0]),
                                                     float(CANDIDATE_OFFSETS[candidate][1]),
                                                     0.105, 0.004, 12.0, 0.0, 0.0],
                    "branch_timestep": int(row["step"]),
                    "recovery_phase": 0,
                    "language": LANGUAGE,
                    "online_observation": observation,
                    "recovery_target_action_chunk": [
                        normalized_action(item["action"], stats)
                        for item in segment[offset:offset + args.chunk_length]
                    ],
                    "teacher_target_relative_position": (anchor - eef).tolist(),
                    "provenance": {
                        "online_rollout": str(rollout.resolve()),
                        "offline_teacher": "anchor_fsm_complete_macro",
                        "teacher_only_target_anchor": anchor.tolist(),
                    },
                })
    if not records:
        raise RuntimeError("no complete recovery windows were emitted")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "samples": len(records), "tasks": found,
                      "chunk_length": args.chunk_length, "stride": args.stride}))


if __name__ == "__main__":
    main()
