#!/usr/bin/env python3
"""Build causal shared-recovery manifests from all Robot-34 feedback branches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--chunk-length", type=int, default=8)
    parser.add_argument(
        "--max-windows-per-branch", type=int,
        help="Optional cap retaining the earliest causal action windows from each verified branch.",
    )
    parser.add_argument(
        "--force-split", choices=("train", "val", "test"),
        help="Assign every emitted branch to one explicit split (used for DAgger augmentation only).",
    )
    return parser.parse_args()


def split(source_key: str) -> str:
    value = int(hashlib.sha256(f"robot34-book-recovery::{source_key}".encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 70 else "val" if value < 85 else "test"


def observation(row: dict, previous_action: list[float] | None) -> dict:
    value = dict(row["online_observation"])
    value["previous_executed_action"] = previous_action if previous_action is not None else [0.0] * 7
    for key in ("rgb", "wrist_rgb"):
        if not Path(value[key]).is_file():
            raise FileNotFoundError(value[key])
    return value


def main() -> None:
    args = arguments()
    if args.stride < 1 or args.chunk_length < 1 or (
        args.max_windows_per_branch is not None and args.max_windows_per_branch < 1
    ):
        raise ValueError("stride and chunk-length must be positive")
    actions, outcomes = [], []
    for metadata_path in sorted(args.branches_root.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("kind") != "offline_book_regrasp_branch_v2":
            continue
        if metadata.get("observation_timing") != "pre_action":
            raise ValueError(f"non-causal branch: {metadata_path}")
        rows = [json.loads(line) for line in metadata_path.with_name("steps.jsonl").read_text().splitlines() if line]
        if len(rows) != int(metadata["num_actions"]):
            raise ValueError(f"transition count mismatch: {metadata_path}")
        fork = json.loads((Path(metadata["source_fork"]) / "manifest.json").read_text())
        source_key = f"{metadata['task_suite']}-task{metadata['task_id']}-episode{metadata['episode_id']}"
        common = {
            "schema_version": 3,
            "split": args.force_split or split(source_key),
            "source_key": source_key,
            "task_id": int(metadata["task_id"]),
            "candidate_name": metadata["candidate_name"],
            "candidate_program_descriptor": [
                *metadata["candidate_program"]["initial_xy_offset_m"],
                float(metadata["candidate_program"]["grasp_eef_height_m"]),
                float(metadata["candidate_program"]["xy_tolerance_m"]),
                float(metadata["candidate_program"]["close_steps"]),
                0.0,
                0.0,
            ],
            "language": fork.get("task_language", "pick up the book and place it in the back compartment of the caddy"),
            "provenance": {"branch_metadata": str(metadata_path), "source_fork": metadata["source_fork"]},
        }
        previous = None
        first = observation(rows[0], previous)
        outcomes.append({
            **common,
            "kind": "book_regrasp_branch_outcome",
            "initial_online_observation": first,
            "recovery_verified_label": bool(metadata["recovery_verified"]),
        })
        if not metadata["recovery_verified"]:
            continue
        emitted = 0
        for index, row in enumerate(rows):
            current = observation(row, previous)
            previous = row["executed_action"]
            if index % args.stride or index + args.chunk_length > len(rows):
                continue
            teacher = row.get("teacher_only", {})
            target = [step["recovery_action_head_space"] for step in rows[index : index + args.chunk_length]]
            actions.append({
                **common,
                "kind": "book_regrasp_action_window",
                "branch_timestep": index,
                "recovery_phase": int(teacher.get("recovery_phase", 0)),
                "online_observation": current,
                "recovery_target_action_chunk": target,
                "teacher_target_relative_position": teacher.get("target_relative_position"),
                "teacher_target_quaternion_wxyz": teacher.get("target_quaternion_wxyz"),
            })
            emitted += 1
            if args.max_windows_per_branch is not None and emitted >= args.max_windows_per_branch:
                break
    if not outcomes or not actions:
        raise RuntimeError("expected both branch outcomes and verified action windows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records = outcomes + actions
    for filename, records in (("branch_outcomes.jsonl", outcomes), ("action_windows.jsonl", actions), ("all_records.jsonl", all_records)):
        (args.output_dir / filename).write_text("".join(json.dumps(record) + "\n" for record in records))
    report = {
        "branches": len(outcomes),
        "verified_branches": sum(record["recovery_verified_label"] for record in outcomes),
        "action_windows": len(actions),
        "sources": len({record["source_key"] for record in outcomes}),
        "splits": {name: sum(record["split"] == name for record in outcomes) for name in ("train", "val", "test")},
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
