#!/usr/bin/env python
"""Build a canonical, cross-task manifest for current recovery branches.

The input encoders remain task-specific.  This manifest aligns their semantic
role order to ``(recovery_target, goal_or_context, distractor)`` and keeps all
simulator outcomes as labels only.  It is the explicit data boundary before a
shared recovery head is trained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CANONICAL_ROLE_ORDER = ("recovery_target", "goal_or_context", "distractor")
ROLE_PERMUTATION_BY_TASK = {
    # Source role order: porcelain_mug, plate, pudding.  The current recovery
    # target is pudding, while mug is collateral/distractor context.
    6: (2, 1, 0),
    # Source role order is already target mug, microwave, distractor mug.
    9: (0, 1, 2),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot-metadata", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-lift-m", type=float, default=.01)
    return parser.parse_args()


def first_row(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"empty branch trajectory: {path}")


def candidate_outcome(source: Path, candidate_name: str) -> dict:
    # source/raw-trajectories/<candidate>/raw-transitions.jsonl
    branch_root = source.parents[2]
    path = branch_root / "candidate_branches.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [item for item in payload["candidates"] if item["candidate_name"] == candidate_name]
    if len(matches) != 1:
        raise ValueError(f"expected one {candidate_name} outcome in {path}, got {len(matches)}")
    return matches[0]


def main() -> None:
    args = arguments()
    records = []
    seen = set()
    for metadata_path in args.slot_metadata:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for branch in metadata["branches"]:
            source = Path(branch["source"])
            identity = first_row(source)
            task_id = int(identity["task_id"])
            key = (task_id, int(identity["source_episode_id"]), identity["candidate_name"], str(source))
            if key in seen:
                raise ValueError(f"duplicate branch source: {key}")
            seen.add(key)
            if task_id not in ROLE_PERMUTATION_BY_TASK:
                raise ValueError(f"no canonical role mapping for task {task_id}")
            outcome = candidate_outcome(source, identity["candidate_name"])
            attached = bool(outcome["attached_after"])
            lift = float(outcome["target_lift_m"])
            verified = attached and lift >= args.minimum_lift_m
            records.append({
                "schema_version": 1,
                "kind": "canonical_recovery_candidate_branch",
                "task_suite": identity["task_suite"],
                "task_id": task_id,
                "source_episode_id": int(identity["source_episode_id"]),
                "failure_timestep": int(identity["failure_timestep"]),
                "source_group_id": (
                    f"task{task_id}-ep{identity['source_episode_id']}"
                    f"-t{identity['failure_timestep']}"
                ),
                "canonical_role_order": list(CANONICAL_ROLE_ORDER),
                "source_role_permutation": list(ROLE_PERMUTATION_BY_TASK[task_id]),
                "slot_file": branch["output"],
                "raw_trajectory": str(source),
                "candidate_name": identity["candidate_name"],
                "candidate_index": int(identity["candidate_index"]),
                "action_steps": int(outcome["steps_executed"]),
                "label_teacher_only": {
                    "attached_after": attached,
                    "target_lift_m": lift,
                    "regrasp_verified": verified,
                    "teacher_value": float(outcome["teacher_value"]),
                    "task_success_after": bool(outcome["task_success_after"]),
                },
                "provenance": {
                    "slot_metadata": str(metadata_path.resolve()),
                    "candidate_branches": str(source.parents[2] / "candidate_branches.json"),
                    "online_input_excludes_teacher_outcomes": True,
                },
            })
    records.sort(key=lambda row: (row["task_id"], row["source_episode_id"], row["candidate_index"]))
    if not records:
        raise ValueError("no branches found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    groups = {}
    for row in records:
        group = groups.setdefault(row["source_group_id"], {"task_id": row["task_id"], "candidates": 0, "verified": 0})
        group["candidates"] += 1
        group["verified"] += int(row["label_teacher_only"]["regrasp_verified"])
    groups_per_task = {
        task_id: sum(group["task_id"] == task_id for group in groups.values())
        for task_id in sorted({row["task_id"] for row in records})
    }
    insufficient_tasks = [
        str(task_id) for task_id, count in groups_per_task.items() if count < 6
    ]
    report = {
        "schema_version": 1,
        "kind": "canonical_recovery_candidate_branch_manifest",
        "canonical_role_order": list(CANONICAL_ROLE_ORDER),
        "minimum_lift_m": args.minimum_lift_m,
        "branches": len(records),
        "verified_regrasp_branches": sum(row["label_teacher_only"]["regrasp_verified"] for row in records),
        "groups": groups,
        "task_counts": {
            str(task_id): sum(row["task_id"] == task_id for row in records)
            for task_id in sorted({row["task_id"] for row in records})
        },
        "source_groups_per_task": groups_per_task,
        "training_readiness": (
            "shared-head-smoke-only: task-specific RGB encoders and fewer than six "
            "independent source groups for task(s) " + ", ".join(insufficient_tasks)
            if insufficient_tasks else
            "ready for a group-held-out shared-head experiment; shared visual encoder and "
            "full p_fail/future-slot loop still require separate audits"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
