#!/usr/bin/env python3
"""Consolidate verified V4.3 DAgger feedback teachers into feature sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dagger-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=8)
    args = parser.parse_args()
    if args.action_horizon < 1:
        raise ValueError("--action-horizon must be positive")
    reports = sorted({path.resolve() for root in args.dagger_roots
                      for path in root.rglob("dagger-report.json")})
    if not reports:
        raise ValueError("no dagger-report.json found")
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    seen_teachers = set()
    for report_path in reports:
        report = json.loads(report_path.read_text())
        for teacher_record in report.get("teachers", []):
            if teacher_record.get("success") is not True:
                continue
            teacher = Path(teacher_record["teacher"]).resolve()
            if teacher in seen_teachers:
                continue
            seen_teachers.add(teacher)
            metadata = json.loads((teacher / "metadata.json").read_text())
            if metadata.get("success") is not True:
                raise ValueError(f"report accepted unsuccessful teacher: {teacher}")
            raw = rows(teacher / "raw-transitions.jsonl")
            task_id = int(metadata["task_id"])
            fork_name = teacher.name
            source = args.output_root / f"task{task_id}-{fork_name}"
            source.mkdir(parents=True, exist_ok=True)
            converted = []
            for index, row in enumerate(raw):
                action = row.get("online_observation", {}).get("recovery_action")
                if not isinstance(action, list) or len(action) != 7:
                    raise ValueError(f"{teacher}:{index} has no seven-dimensional teacher action")
                converted.append({
                    "kind": "v43_dagger_feedback_transition",
                    "timestep": index,
                    "phase": "TEACHER_RECOVERY",
                    "gate_label": "RECOVERY",
                    "online_observation": row["online_observation"],
                    "training_only": row.get("training_only", {}),
                    "executed_action": action,
                })
            with (source / "steps.jsonl").open("w") as stream:
                for row in converted:
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            source_metadata = {
                **metadata,
                "kind": "v43_dagger_verified_feedback_source",
                "schema": "v4_rgbd_rollout_v2",
                "fork_timestep": 0,
                "transitions": len(converted),
                "rollout_id": f"dagger:{metadata['task_suite']}:{task_id}:{metadata['episode_id']}:"
                              f"{metadata['failure_timestep']}:{fork_name}",
                "dagger_teacher": str(teacher),
                "action_horizon": args.action_horizon,
            }
            (source / "metadata.json").write_text(json.dumps(source_metadata, indent=2) + "\n")
            complete_targets = max(0, len(converted) - args.action_horizon + 1)
            records.append({
                "task_id": task_id, "source": str(source.resolve()),
                "teacher": str(teacher), "transitions": len(converted),
                "complete_action_targets": complete_targets,
                "target_role": metadata.get("target_role"),
            })
    payload = {
        "kind": "v43_dagger_verified_dataset_v1",
        "sources": len(records),
        "transitions": sum(record["transitions"] for record in records),
        "complete_action_targets": sum(record["complete_action_targets"] for record in records),
        "records": sorted(records, key=lambda row: (row["task_id"], row["source"])),
    }
    (args.output_root / "dataset-report.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
