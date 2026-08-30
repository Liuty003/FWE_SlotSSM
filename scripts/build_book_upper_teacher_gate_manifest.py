#!/usr/bin/env python3
"""Emit direct-online gate positives at verified teacher intervention states.

The intervention timestep is a teacher-only annotation from an archived
successful FSM rollout.  At inference the resulting gate reads only frozen
RGB/wrist/language/proprio/action-history features; it has no step counter or
anchor input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LANGUAGE = "pick up the book and place it in the back compartment of the caddy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-frames", type=int, default=4,
                        help="consecutive teacher RECOVERY observations from its verified entry")
    parser.add_argument("--repeat", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.positive_frames < 1 or args.repeat < 1:
        raise ValueError("positive-frames and repeat must be positive")
    records: list[dict] = []
    summary: dict[str, object] = {"tasks": [], "base_rows": 0}
    for task_id in args.task_ids:
        episode_path = args.teacher_root / f"task{task_id}" / f"task{task_id}-episode0.json"
        rollout = episode_path.with_name("online_trajectory.jsonl")
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        if not episode.get("success") or not rollout.is_file():
            raise RuntimeError(f"teacher task {task_id} is not a successful saved rollout")
        recovery = [row for row in episode["decisions"] if row.get("mode") == "RECOVERY"]
        if len(recovery) < args.positive_frames:
            raise ValueError(f"task {task_id} has only {len(recovery)} recovery records")
        first = recovery[0]
        chosen = recovery[:args.positive_frames]
        if any(int(row["step"]) != int(first["step"]) + index for index, row in enumerate(chosen)):
            raise ValueError(f"task {task_id} recovery prefix is not consecutive")
        for repeat_id in range(args.repeat):
            for row in chosen:
                observation = row.get("online_observation")
                if observation is None:
                    raise ValueError(f"task {task_id} lacks saved online observation")
                records.append({
                    "schema_version": 3,
                    "kind": "book_execution_gate_window",
                    "split": "train",
                    "task_suite": "libero_10",
                    "task_id": task_id,
                    "episode_id": 0,
                    "repeat_id": repeat_id,
                    "source_key": f"forced-teacher-recovery-task{task_id}",
                    "timestep": int(row["step"]),
                    "online_observation": observation,
                    "gate_label": "RECOVERY",
                    "language": LANGUAGE,
                    "provenance": {
                        "online_rollout": str(rollout.resolve()),
                        "offline_teacher": "anchor_fsm_verified_intervention",
                        "teacher_only_intervention_step": int(first["step"]),
                    },
                })
        summary["tasks"].append(task_id)  # type: ignore[index]
        summary["base_rows"] = int(summary["base_rows"]) + len(chosen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    summary.update({"output": str(args.output), "repeat": args.repeat, "samples": len(records)})
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
