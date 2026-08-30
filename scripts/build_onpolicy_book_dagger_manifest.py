#!/usr/bin/env python3
"""Build causal recovery-action windows from teacher branches off neural rollouts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--chunk-length", type=int, default=8)
    return parser.parse_args()


def descriptor(name: str) -> list[float]:
    match = re.fullmatch(r"fsm_xy([pm])(\d+)_([pm])(\d+)", name)
    if match is None:
        raise ValueError(f"unrecognised recovery candidate: {name}")
    x = (-1.0 if match.group(1) == "m" else 1.0) * int(match.group(2)) / 1000.0
    y = (-1.0 if match.group(3) == "m" else 1.0) * int(match.group(4)) / 1000.0
    return [x, y, 0.105, 0.004, 12.0, 0.0, 0.0]


def main() -> None:
    config = args()
    if config.stride < 1 or config.chunk_length < 1:
        raise ValueError("stride and chunk-length must be positive")
    records = []
    for metadata_path in sorted(config.branches_root.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("kind") != "offline_book_regrasp_branch_v2" or not metadata.get("recovery_verified"):
            continue
        rows = [json.loads(line) for line in metadata_path.with_name("steps.jsonl").read_text().splitlines() if line]
        if len(rows) != int(metadata["num_actions"]):
            raise ValueError(f"branch length mismatch: {metadata_path}")
        for index, row in enumerate(rows):
            if index % config.stride or index + config.chunk_length > len(rows):
                continue
            observation = dict(row["online_observation"])
            observation["previous_executed_action"] = (
                rows[index - 1]["executed_action"] if index else [0.0] * 7
            )
            teacher = row["teacher_only"]
            relative = teacher.get("target_relative_position")
            if relative is None or len(relative) != 3:
                raise ValueError(f"missing teacher geometry: {metadata_path}:{index}")
            records.append({
                "schema_version": 3,
                "kind": "book_regrasp_action_window",
                "split": config.split,
                "source_key": f"libero_10-task{metadata['task_id']}-episode{metadata['episode_id']}",
                "task_id": int(metadata["task_id"]),
                "candidate_name": metadata["candidate_name"],
                "candidate_program_descriptor": descriptor(metadata["candidate_name"]),
                "branch_timestep": index,
                "recovery_phase": int(teacher.get("recovery_phase", 0)),
                "language": config.language,
                "online_observation": observation,
                "recovery_target_action_chunk": [
                    item["recovery_action_head_space"] for item in rows[index:index + config.chunk_length]
                ],
                "teacher_target_relative_position": relative,
                "teacher_target_quaternion_wxyz": teacher.get("target_quaternion_wxyz"),
                "provenance": {"branch_metadata": str(metadata_path), "source_fork": metadata["source_fork"], "onpolicy_dagger": True},
            })
    if not records:
        raise RuntimeError("no verified on-policy teacher windows")
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text("".join(json.dumps(record) + "\n" for record in records))
    print(json.dumps({"output": str(config.output), "samples": len(records)}))


if __name__ == "__main__":
    main()
