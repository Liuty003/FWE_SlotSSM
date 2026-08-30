#!/usr/bin/env python3
"""Build leakage-audited training manifests for the book local-regrasp pilot.

The collector stores simulator pose/contact only in branch metadata.  This
builder deliberately never copies those fields into model inputs: it emits
RGB, wrist RGB, language, proprio, executed-action history and labels only.
All windows sharing an OFT source state are assigned to one split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches-root", type=Path, action="append", required=True)
    parser.add_argument("--natural-root", type=Path, required=True)
    parser.add_argument(
        "--stability-root", type=Path, action="append", required=True,
        help="One or more roots containing the three-repeat source audit.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-task-ids", default="",
                        help="Comma-separated stable task ids retained only for audit, never emitted into this manifest.")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--chunk-length", type=int, default=8)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def source_key(task_id: int, episode_id: int = 0) -> str:
    return f"libero_10-task{task_id}-episode{episode_id}"


def hashed_split(key: str) -> str:
    bucket = int(hashlib.sha256(f"book-regrasp-v2::{key}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def teacher_regrasp_phase(timestep: int) -> int:
    """Phase of the fixed local teacher program at its causal action index.

    0=backoff/open, 1=lateral align, 2=descend, 3=close, 4=lift/verify.
    This label is created offline from the teacher program timing; it is never
    an online simulator-state input.
    """
    if timestep < 6:
        return 0
    if timestep < 16:
        return 1
    if timestep < 28:
        return 2
    if timestep < 40:
        return 3
    return 4


def online_observation(row: dict[str, Any], previous_action: list[float] | None) -> dict[str, Any]:
    observation = row["online_observation"]
    for field in ("rgb", "wrist_rgb", "proprio"):
        if not observation.get(field):
            raise ValueError(f"missing online {field} at {row}")
    for field in ("rgb", "wrist_rgb"):
        if not Path(observation[field]).is_file():
            raise FileNotFoundError(observation[field])
    return {
        "rgb": observation["rgb"],
        "wrist_rgb": observation["wrist_rgb"],
        "proprio": observation["proprio"],
        "previous_executed_action": previous_action,
    }


def candidate_program_descriptor(metadata: dict[str, Any]) -> list[float]:
    """Return the deployable macro parameters for one matched FSM branch.

    Old branches predate ``candidate_program``.  Their names losslessly encode
    the two offset values, so retain them rather than silently dropping early
    data.  No teacher pose/contact field is used here.
    """
    program = metadata.get("candidate_program")
    if isinstance(program, dict):
        offset = program.get("initial_xy_offset_m")
        if isinstance(offset, list) and len(offset) == 2:
            return [float(offset[0]), float(offset[1]),
                    float(program.get("grasp_eef_height_m", 0.105)),
                    float(program.get("xy_tolerance_m", 0.004)),
                    float(program.get("close_steps", 12)), 0.0, 0.0]
    match = re.fullmatch(r"fsm_xy([pm])(\d+)_([pm])(\d+)", str(metadata["candidate_name"]))
    if match is None:
        raise ValueError(f"cannot recover deployable candidate parameters: {metadata['candidate_name']}")
    x = (-1.0 if match.group(1) == "m" else 1.0) * int(match.group(2)) / 1000.0
    y = (-1.0 if match.group(3) == "m" else 1.0) * int(match.group(4)) / 1000.0
    return [x, y, 0.105, 0.004, 12.0, 0.0, 0.0]


def main() -> None:
    args = parse_args()
    if args.stride < 1 or args.chunk_length < 1:
        raise ValueError("stride and chunk length must be positive")
    excluded_task_ids = {int(value) for value in args.exclude_task_ids.split(",") if value.strip()}

    # First reject any source whose outcome changes across independent replays.
    audit: dict[int, list[bool]] = defaultdict(list)
    for root in args.stability_root:
        for path in root.glob("**/metadata.json"):
            item = read_json(path)
            if item.get("kind") != "oft_nominal_rollout":
                continue
            audit[int(item["task_id"])].append(bool(item["success"]))
    stable: dict[int, bool] = {
        task_id: outcomes[0]
        for task_id, outcomes in audit.items()
        # Several independent audit batches may exist for one source.  They
        # add evidence; requiring exactly three would incorrectly discard a
        # source with six consistent replays.  Any disagreement still
        # excludes the source entirely.
        if len(outcomes) >= 3 and len(set(outcomes)) == 1
    }

    natural: dict[str, dict[str, Any]] = {}
    for metadata_path in sorted(args.natural_root.glob("**/metadata.json")):
        item = read_json(metadata_path)
        task_id, episode_id = int(item["task_id"]), int(item["episode_id"])
        if task_id in excluded_task_ids:
            continue
        if task_id not in stable:
            continue
        if bool(item["success"]) != stable[task_id]:
            # This fresh RGB recording disagrees with the three-repeat audit.
            # It remains an audit artifact but is never a training source.
            continue
        steps_path = metadata_path.parent / "steps.jsonl"
        rows = read_jsonl(steps_path)
        if len(rows) != int(item["transitions"]):
            raise ValueError(f"transition count mismatch: {metadata_path}")
        natural[source_key(task_id, episode_id)] = {
            "metadata": item, "metadata_path": metadata_path, "rows": rows,
        }

    branches: list[dict[str, Any]] = []
    for branch_root in args.branches_root:
        for metadata_path in sorted(branch_root.glob("**/metadata.json")):
            item = read_json(metadata_path)
            if item.get("kind") != "offline_book_regrasp_branch_v2":
                raise ValueError(f"not v2 branch metadata: {metadata_path}")
            if item.get("observation_timing") != "pre_action":
                raise ValueError(f"non-causal branch data: {metadata_path}")
            task_id, episode_id = int(item["task_id"]), int(item["episode_id"])
            key = source_key(task_id, episode_id)
            if stable.get(task_id) is not False:
                raise ValueError(f"branch source is not a stable natural failure: {metadata_path}")
            if key not in natural:
                raise ValueError(f"missing matching natural RGB rollout for {metadata_path}")
            rows = read_jsonl(metadata_path.parent / "steps.jsonl")
            if len(rows) != int(item["num_actions"]):
                raise ValueError(f"branch transition count mismatch: {metadata_path}")
            if [row["timestep"] for row in rows] != list(range(len(rows))):
                raise ValueError(f"non-contiguous branch timesteps: {metadata_path}")
            branches.append({"metadata": item, "metadata_path": metadata_path, "rows": rows, "source_key": key})

    if not branches:
        raise RuntimeError("no v2 branches found")

    # Ensure the sparse successful branches provide a true held-out recovery
    # source, rather than leaking candidate windows into every split.
    positive_sources = sorted({item["source_key"] for item in branches if item["metadata"]["recovery_verified"]})
    if len(positive_sources) < 3:
        raise RuntimeError(f"need at least three successful recovery sources, got {positive_sources}")
    splits = {key: hashed_split(key) for key in natural}
    for key, split in zip(positive_sources[:3], ("train", "val", "test")):
        splits[key] = split

    action_windows: list[dict[str, Any]] = []
    branch_outcomes: list[dict[str, Any]] = []
    for branch in branches:
        item, rows, key = branch["metadata"], branch["rows"], branch["source_key"]
        language = natural[key]["metadata"]["task_language"]
        previous = None
        for index, row in enumerate(rows):
            observation = online_observation(row, previous)
            previous = row["executed_action"]
            if index == 0:
                branch_outcomes.append({
                    "schema_version": 2,
                    "kind": "book_regrasp_branch_outcome",
                    "split": splits[key],
                    "source_key": key,
                    "task_id": item["task_id"],
                    "candidate_name": item["candidate_name"],
                    "candidate_program_descriptor": candidate_program_descriptor(item),
                    "language": language,
                    "initial_online_observation": observation,
                    "recovery_verified_label": bool(item["recovery_verified"]),
                    "provenance": {"branch_metadata": str(branch["metadata_path"]), "source_fork": item["source_fork"]},
                })
            if not item["recovery_verified"] or index + args.chunk_length > len(rows):
                continue
            target = [step["recovery_action_head_space"] for step in rows[index : index + args.chunk_length]]
            phase = int(row.get("teacher_only", {}).get("recovery_phase", teacher_regrasp_phase(index)))
            if not 0 <= phase < 5:
                raise ValueError(f"invalid recovery phase at {branch['metadata_path']}:{index}")
            action_windows.append({
                "schema_version": 2,
                "kind": "book_regrasp_action_window",
                "split": splits[key],
                "source_key": key,
                "task_id": item["task_id"],
                "candidate_name": item["candidate_name"],
                "branch_timestep": index,
                # Fixed-duration branches omit this field and use the legacy
                # timing map.  Feedback teachers record their actual phase,
                # which may include retries and therefore cannot be inferred
                # from a global timestep.
                "recovery_phase": phase,
                "language": language,
                "online_observation": observation,
                "recovery_target_action_chunk": target,
                "provenance": {"branch_metadata": str(branch["metadata_path"]), "source_fork": item["source_fork"]},
            })
            geometry = row.get("teacher_only", {}).get("target_relative_position")
            if geometry is not None:
                if len(geometry) != 3:
                    raise ValueError(f"invalid teacher relative target position at {branch['metadata_path']}:{index}")
                # Offline target only: never copied into online_observation.
                action_windows[-1]["teacher_target_relative_position"] = [float(value) for value in geometry]
            target_quaternion = row.get("teacher_only", {}).get("target_quaternion_wxyz")
            if target_quaternion is not None:
                if len(target_quaternion) != 4:
                    raise ValueError(f"invalid teacher target quaternion at {branch['metadata_path']}:{index}")
                # This remains a teacher-only label.  Its purpose is to learn
                # grasp/placement pose compatibility, not to expose object
                # orientation to the online policy directly.
                action_windows[-1]["teacher_target_quaternion_wxyz"] = [float(value) for value in target_quaternion]

    gate_windows: list[dict[str, Any]] = []
    fork_timestep_by_source: dict[str, int] = {}
    for branch in branches:
        fork = read_json(Path(branch["metadata"]["source_fork"]) / "manifest.json")
        key = branch["source_key"]
        # A learned-policy correction fork is a state reached *after* the
        # normal rollout has already failed and recovery actions have been
        # executed.  It is valid action-teacher data, but its timestamp cannot
        # label a natural-trajectory gate observation.
        if fork.get("kind") != "learned_oracle_recovery_state_v1":
            fork_timestep_by_source[key] = int(fork["fork_timestep"])
    for key, payload in natural.items():
        rows, item = payload["rows"], payload["metadata"]
        if item["success"]:
            for index in range(0, len(rows), args.stride):
                row = rows[index]
                previous = rows[index - 1]["executed_action"] if index else None
                gate_windows.append({
                    "schema_version": 2, "kind": "book_execution_gate_window", "split": splits[key],
                    "source_key": key, "task_id": item["task_id"], "timestep": index,
                    "language": item["task_language"], "online_observation": online_observation(row, previous),
                    "gate_label": "NORMAL",
                })
        elif key in fork_timestep_by_source:
            index = fork_timestep_by_source[key]
            row = rows[index]
            previous = rows[index - 1]["executed_action"] if index else None
            gate_windows.append({
                "schema_version": 2, "kind": "book_execution_gate_window", "split": splits[key],
                "source_key": key, "task_id": item["task_id"], "timestep": index,
                "language": item["task_language"], "online_observation": online_observation(row, previous),
                "gate_label": "RECOVERY",
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "action_windows.jsonl", action_windows)
    write_jsonl(args.output_dir / "branch_outcomes.jsonl", branch_outcomes)
    write_jsonl(args.output_dir / "gate_windows.jsonl", gate_windows)
    source_rows = [
        {"source_key": key, "split": splits[key], "stable_success": bool(payload["metadata"]["success"]),
         "metadata": str(payload["metadata_path"])}
        for key, payload in sorted(natural.items())
    ]
    (args.output_dir / "source_splits.json").write_text(json.dumps(source_rows, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": 2,
        "online_input_contract": ["rgb", "wrist_rgb", "language", "proprio", "previous_executed_action"],
        "stable_sources": len(natural),
        "stable_normal_sources": sum(item["metadata"]["success"] for item in natural.values()),
        "stable_failure_sources": sum(not item["metadata"]["success"] for item in natural.values()),
        "branches": len(branches),
        "successful_branches": sum(item["metadata"]["recovery_verified"] for item in branches),
        "action_windows": len(action_windows),
        "branch_outcomes": len(branch_outcomes),
        "gate_windows": len(gate_windows),
        "positive_recovery_sources": positive_sources,
        "split_counts": {split: sum(value == split for value in splits.values()) for split in ("train", "val", "test")},
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
