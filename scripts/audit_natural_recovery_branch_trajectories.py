#!/usr/bin/env python
"""Audit raw natural-failure recovery branch trajectories before enrichment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def check_action(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 7
        and all(isinstance(item, (int, float)) for item in value)
    )


def audit_branch(path: Path) -> dict:
    rows = load_rows(path)
    errors: list[str] = []
    if not rows:
        errors.append("empty trajectory")
        return {"path": str(path), "valid": False, "errors": errors}
    first = rows[0]
    identity = {
        key: first.get(key)
        for key in (
            "task_suite", "task_id", "branch_episode_id", "source_episode_id",
            "failure_timestep", "candidate_name", "candidate_index", "target_name",
        )
    }
    attached_before = attached_after = masks_visible = 0
    for index, row in enumerate(rows):
        if row.get("branch_timestep") != index:
            errors.append(
                f"row {index}: non-contiguous branch_timestep={row.get('branch_timestep')}"
            )
        for key, expected in identity.items():
            if row.get(key) != expected:
                errors.append(f"row {index}: identity drift in {key}")
        online = row.get("online_observation")
        teacher = row.get("teacher_only")
        if not isinstance(online, dict) or not isinstance(teacher, dict):
            errors.append(f"row {index}: missing online_observation/teacher_only")
            continue
        if any("teacher" in key or "target_position" in key for key in online):
            errors.append(f"row {index}: privileged key leaked into online_observation")
        if not check_action(online.get("recovery_action")):
            errors.append(f"row {index}: invalid recovery_action")
        for key in ("rgb", "depth", "target_mask"):
            value = online.get(key)
            if not isinstance(value, str) or not Path(value).is_file():
                errors.append(f"row {index}: missing {key} media")
        measurement = online.get("target_mask_measurement")
        if not isinstance(measurement, dict):
            errors.append(f"row {index}: missing target mask measurement")
        elif int(measurement.get("visible_pixels", 0)) > 0:
            masks_visible += 1
        if not isinstance(teacher.get("attached_before"), bool):
            errors.append(f"row {index}: attached_before is not bool")
        if not isinstance(teacher.get("attached_after"), bool):
            errors.append(f"row {index}: attached_after is not bool")
        attached_before += int(teacher.get("attached_before") is True)
        attached_after += int(teacher.get("attached_after") is True)
    return {
        "path": str(path),
        "valid": not errors,
        "identity": identity,
        "rows": len(rows),
        "visible_mask_rows": masks_visible,
        "attached_before_rows": attached_before,
        "attached_after_rows": attached_after,
        "errors": errors[:100],
    }


def main() -> None:
    args = arguments()
    paths = sorted(args.root.rglob("raw-transitions.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no raw trajectories below {args.root}")
    branches = [audit_branch(path) for path in paths]
    by_task: dict[str, dict[str, int]] = {}
    for branch in branches:
        task = str(branch.get("identity", {}).get("task_id"))
        summary = by_task.setdefault(task, {"branches": 0, "rows": 0})
        summary["branches"] += 1
        summary["rows"] += int(branch.get("rows", 0))
    report = {
        "schema_version": 1,
        "kind": "natural_recovery_raw_trajectory_audit",
        "root": str(args.root.resolve()),
        "valid": all(branch["valid"] for branch in branches),
        "branches": branches,
        "totals": {
            "branches": len(branches),
            "rows": sum(int(branch.get("rows", 0)) for branch in branches),
            "visible_mask_rows": sum(
                int(branch.get("visible_mask_rows", 0)) for branch in branches
            ),
            "attached_after_rows": sum(
                int(branch.get("attached_after_rows", 0)) for branch in branches
            ),
            "by_task": by_task,
        },
    }
    output = args.output or args.root / "raw-trajectory-audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["totals"], indent=2, ensure_ascii=False))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
