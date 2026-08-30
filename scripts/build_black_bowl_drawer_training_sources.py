#!/usr/bin/env python3
"""Promote only simulator-verified black-bowl teachers to V4.3 sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-multiple-per-task", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    accepted, rejected = [], []
    seen = set()
    for root in args.teacher_roots:
        for metadata_path in sorted(root.rglob("metadata.json")):
            source = metadata_path.parent
            metadata = json.loads(metadata_path.read_text())
            task_id = int(metadata["task_id"])
            identity = str(source.resolve()) if args.allow_multiple_per_task else task_id
            if identity in seen:
                continue
            if metadata.get("success") is not True:
                rejected.append({"task_id": task_id, "source": str(source),
                                 "reason": "simulator_predicates_failed"})
                continue
            raw_path = source / "raw-transitions.jsonl"
            raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
            if len(raw) < 8:
                rejected.append({"task_id": task_id, "source": str(source),
                                 "reason": "shorter_than_action_horizon"})
                continue
            suffix = f"-{source.name}" if args.allow_multiple_per_task else ""
            destination = args.output_root / f"task{task_id}-episode0{suffix}"
            destination.mkdir(parents=True, exist_ok=True)
            steps = []
            for index, row in enumerate(raw):
                action = row["online_observation"]["recovery_action"]
                steps.append({
                    "kind": "black_bowl_drawer_v43_recovery_transition",
                    "timestep": index,
                    "phase": "TEACHER_RECOVERY",
                    "gate_label": "RECOVERY",
                    "online_observation": row["online_observation"],
                    "training_only": row.get("training_only", {}),
                    "executed_action": action,
                })
            (destination / "steps.jsonl").write_text("".join(
                json.dumps(row, separators=(",", ":")) + "\n" for row in steps))
            promoted = {
                **metadata,
                "kind": "black_bowl_drawer_v43_successful_recovery",
                "schema": "v4_rgbd_rollout_v2",
                "fork_timestep": 0,
                "wait_steps": 0,
                "open_loop_steps": 8,
                "transitions": len(steps),
                "source_teacher": str(source.resolve()),
                "rollout_id": f"black-bowl-teacher:{task_id}:{source.name}:{source.resolve()}",
                "gate_semantics": "RECOVERY for the full teacher; task terminates at completion",
            }
            (destination / "metadata.json").write_text(json.dumps(promoted, indent=2) + "\n")
            accepted.append({"task_id": task_id, "source": str(source.resolve()),
                             "destination": str(destination.resolve()), "transitions": len(steps)})
            seen.add(identity)
    report = {"kind": "black_bowl_drawer_v43_training_sources_v1",
              "accepted": accepted, "rejected": rejected,
              "accepted_count": len(accepted), "rejected_count": len(rejected)}
    (args.output_root / "source-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
