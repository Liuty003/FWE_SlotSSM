#!/usr/bin/env python3
"""Build causal action windows from successful verified-lift placement searches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--chunk-length", type=int, default=8)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    return parser.parse_args()


def main() -> None:
    config = args()
    if config.stride < 1 or config.chunk_length < 1:
        raise ValueError("stride and chunk-length must be positive")
    records = []
    for path in sorted(config.branches_root.rglob("metadata.json")):
        metadata = json.loads(path.read_text())
        if metadata.get("kind") != "offline_book_placement_search_branch_v1" or not metadata.get("task_success"):
            continue
        lift_snapshot = Path(metadata["source_verified_lift_snapshot"])
        lift_manifest = json.loads((lift_snapshot / "manifest.json").read_text())
        prelift_metadata = lift_snapshot.parent.parent / "branches" / lift_manifest["source_candidate_name"] / "metadata.json"
        if not prelift_metadata.is_file():
            raise FileNotFoundError(f"missing pre-lift branch metadata: {prelift_metadata}")
        rows = [json.loads(line) for line in path.with_name("steps.jsonl").read_text().splitlines() if line]
        if len(rows) != int(metadata["num_actions"]):
            raise ValueError(f"transition count mismatch: {path}")
        previous = None
        for index, row in enumerate(rows):
            observation = dict(row["online_observation"])
            observation["previous_executed_action"] = previous if previous is not None else [0.0] * 7
            previous = row["executed_action"]
            if index % config.stride or index + config.chunk_length > len(rows):
                continue
            teacher = row["teacher_only"]
            relative = teacher.get("target_relative_position")
            if relative is None or len(relative) != 3:
                raise ValueError(f"missing teacher-only geometry at {path}:{index}")
            records.append({
                "schema_version": 3,
                "kind": "book_regrasp_action_window",
                "split": config.split,
                "source_key": f"libero_10-task{metadata['task_id']}-episode{metadata['episode_id']}",
                "task_id": int(metadata["task_id"]),
                # These trajectories begin after the verified fsm_xyp25_p0
                # grasp.  Retain that deployable mode label so the existing
                # five-way recovery adapter can consume the windows.
                "candidate_name": "fsm_xyp25_p0",
                "candidate_program_descriptor": [0.025, 0.0, 0.105, 0.004, 12.0, 0.0, 0.0],
                "language": "pick up the book and place it in the back compartment of the caddy",
                "provenance": {
                    "branch_metadata": str(path),
                    "prelift_branch_metadata": str(prelift_metadata),
                    "verified_completion": True,
                },
                "branch_timestep": index,
                "recovery_phase": int(teacher["recovery_phase"]),
                "online_observation": observation,
                "recovery_target_action_chunk": [item["recovery_action_head_space"] for item in rows[index:index + config.chunk_length]],
                "teacher_target_relative_position": relative,
                "teacher_target_quaternion_wxyz": teacher.get("target_quaternion_wxyz"),
            })
    if not records:
        raise RuntimeError("no successful placement trajectories found")
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text("".join(json.dumps(row) + "\n" for row in records))
    print(json.dumps({"output": str(config.output), "action_windows": len(records), "tasks": sorted({row["task_id"] for row in records})}))


if __name__ == "__main__":
    main()
