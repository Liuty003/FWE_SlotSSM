#!/usr/bin/env python3
"""Build source-disjoint Robot-34 NORMAL/RECOVERY gate supervision.

Recovery trigger times are derived offline from teacher-verified local-regrasp
forks.  Emitted examples retain only causal camera, proprio, and action-history
observations; simulator contact and object pose never enter a gate record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--branches-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positive-before", type=int, default=8)
    parser.add_argument("--positive-after", type=int, default=12)
    parser.add_argument("--normal-stride", type=int, default=4)
    return parser.parse_args()


def split(source_key: str) -> str:
    value = int(hashlib.sha256(f"robot34-book-recovery::{source_key}".encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 70 else "val" if value < 85 else "test"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def gate_record(row: dict, *, source_key: str, task_id: int, language: str, label: str, trigger: int | None) -> dict:
    observation = row.get("online_observation")
    if observation is None:
        raise ValueError(f"missing rendered causal observation at {source_key}:{row['policy_step']}")
    for camera in ("rgb", "wrist_rgb"):
        if not Path(observation[camera]).is_file():
            raise FileNotFoundError(observation[camera])
    return {
        "schema_version": 3,
        "kind": "book_execution_gate_window",
        "split": split(source_key),
        "source_key": source_key,
        "task_id": task_id,
        "timestep": int(row["policy_step"]),
        "language": language,
        "online_observation": observation,
        "gate_label": label,
        "gate_label_provenance": (
            {"type": "teacher_verified_recovery_fork_v1", "trigger_timestep": trigger}
            if trigger is not None else {"type": "nominal_success_v1"}
        ),
    }


def main() -> None:
    args = arguments()
    if args.positive_before < 0 or args.positive_after < 0 or args.normal_stride < 1:
        raise ValueError("invalid gate window parameters")
    outputs = args.replay_root / "outputs"
    forks = args.replay_root / "forks"
    if not outputs.is_dir() or not forks.is_dir() or not args.branches_root.is_dir():
        raise FileNotFoundError("expected replay outputs/forks and feedback branches")

    verified_by_task: dict[int, list[int]] = defaultdict(list)
    for metadata_path in args.branches_root.rglob("metadata.json"):
        metadata = read_json(metadata_path)
        if metadata.get("kind") == "offline_book_regrasp_branch_v2" and metadata.get("recovery_verified"):
            verified_by_task[int(metadata["task_id"])].append(int(Path(metadata["source_fork"]).name.removeprefix("step")))

    windows, source_rows = [], []
    for metadata_path in sorted(outputs.glob("libero_10-task*-episode0.json")):
        metadata = read_json(metadata_path)
        task_id = int(metadata["task_id"])
        source_key = f"libero_10-task{task_id}-episode0"
        trajectory = forks / f"task{task_id}-episode0" / "trajectory.jsonl"
        rows = read_jsonl(trajectory)
        if not rows:
            raise ValueError(f"empty replay trajectory: {trajectory}")
        # Precompute expects ``metadata.parent / steps.jsonl``.  Create a
        # lightweight source view instead of copying potentially large image data.
        source_dir = args.output_dir / "sources" / source_key
        source_dir.mkdir(parents=True, exist_ok=True)
        steps_link = source_dir / "steps.jsonl"
        if not steps_link.exists():
            os.symlink(trajectory.resolve(), steps_link)
        source_metadata = source_dir / "metadata.json"
        source_metadata.write_text(json.dumps({"task_id": task_id, "success": bool(metadata["success"]),
                                               "task_language": metadata["task_description"]}, indent=2) + "\n")
        source_rows.append({"source_key": source_key, "split": split(source_key), "stable_success": bool(metadata["success"]),
                            "metadata": str(source_metadata)})
        if metadata["success"]:
            for row in rows[::args.normal_stride]:
                windows.append(gate_record(row, source_key=source_key, task_id=task_id,
                                           language=metadata["task_description"], label="NORMAL", trigger=None))
            continue
        candidates = verified_by_task.get(task_id, [])
        if not candidates:
            # This failure is not a valid local-repair source; do not teach a
            # false positive trigger merely because the nominal task failed.
            continue
        trigger = min(candidates)
        for row in rows:
            timestep = int(row["policy_step"])
            positive = trigger - args.positive_before <= timestep <= trigger + args.positive_after
            normal = timestep < trigger - args.positive_before - 4 and timestep % args.normal_stride == 0
            if positive or normal:
                windows.append(gate_record(row, source_key=source_key, task_id=task_id,
                                           language=metadata["task_description"],
                                           label="RECOVERY" if positive else "NORMAL", trigger=trigger))
    if not windows:
        raise RuntimeError("no gate windows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gate_windows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in windows))
    (args.output_dir / "source_splits.json").write_text(json.dumps(source_rows, indent=2) + "\n")
    counts = defaultdict(int)
    for row in windows:
        counts[f"{row['split']}:{row['gate_label']}"] += 1
    report = {"kind": "robot34_book_gate_manifest_v1", "sources": len(source_rows), "windows": len(windows),
              "label_counts": dict(counts), "verified_trigger_tasks": sorted(verified_by_task),
              "online_inputs_permitted": ["main_rgb", "wrist_rgb", "language", "proprio", "executed_action_history"]}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
