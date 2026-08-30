#!/usr/bin/env python3
"""Locate auditable takeover boundaries for automatically classified failures."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-classification", type=Path, required=True)
    parser.add_argument("--failure-type", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-close-context-steps", type=int, default=8)
    parser.add_argument("--post-close-audit-steps", type=int, default=24)
    parser.add_argument("--close-confirmation-steps", type=int, default=2)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def relational_target(rows: list[dict]) -> tuple[str, str, list[str]]:
    goals = rows[0]["privileged_audit"]["goal_predicates"]
    entities = rows[0]["privileged_audit"]["entities"]
    for goal in goals:
        predicate = list(goal["predicate"])
        if len(predicate) >= 3 and predicate[0].lower() in {"in", "on"}:
            target, destination = predicate[1], predicate[2]
            if entities.get(target, {}).get("kind") == "object":
                return target, destination, predicate
    raise ValueError("no supported relational object target in rollout goals")


def first_confirmed_close(rows: list[dict], confirmation: int) -> int | None:
    close = [float(row["executed_action"][6]) > 0 for row in rows]
    for index in range(0, len(close) - confirmation + 1):
        if all(close[index:index + confirmation]):
            return index
    return None


def target_gripper_contact(row: dict, target: str) -> bool:
    for contact in row["privileged_audit"].get("contacts", []):
        names = (str(contact.get("geom1") or ""), str(contact.get("geom2") or ""))
        target_side = [name for name in names if target in name]
        other_side = [name for name in names if target not in name]
        if target_side and any(
            token in name.lower()
            for name in other_side
            for token in ("finger", "gripper", "hand", "pad")
        ):
            return True
    return False


def locate(row: dict, args: argparse.Namespace) -> dict:
    steps_path = Path(row["steps"])
    rows = load_jsonl(steps_path)
    target, destination, goal = relational_target(rows)
    close = first_confirmed_close(rows, args.close_confirmation_steps)
    if close is None:
        return {**row, "onset_status": "no_confirmed_close", "fork_timestep": None}
    fork = max(0, close - args.pre_close_context_steps)
    end = min(len(rows), close + args.post_close_audit_steps + 1)
    target_positions = [
        item["privileged_audit"]["entities"][target]["position"] for item in rows
    ]
    eef_positions = [
        item["online_observation"]["proprio"]["robot0_eef_pos"] for item in rows
    ]
    start_position = target_positions[close]
    post_positions = target_positions[close:end]
    max_displacement = max(distance(start_position, value) for value in post_positions)
    max_lift = max(float(value[2]) - float(start_position[2]) for value in post_positions)
    aperture = [sum(abs(float(value)) for value in item["online_observation"]["proprio"]["robot0_gripper_qpos"])
                for item in rows]
    first_response = next((
        index for index in range(max(1, close), min(len(rows), close + 16))
        if aperture[index] < aperture[index - 1] - 0.002
    ), None)
    dense_bundle = rows[0]["online_observation"].get("dense_bundle")
    video = steps_path.with_name("rollout.mp4")
    return {
        **row,
        "onset_status": "located",
        "goal_predicate": goal,
        "target_entity": target,
        "destination_entity": destination,
        "first_close_command_timestep": close,
        "first_gripper_response_timestep": first_response,
        "fork_timestep": fork,
        "pre_close_context_steps": close - fork,
        "eef_target_distance_at_fork_m": distance(eef_positions[fork], target_positions[fork]),
        "eef_target_distance_at_close_m": distance(eef_positions[close], target_positions[close]),
        "post_close_max_target_displacement_m": max_displacement,
        "post_close_max_target_lift_m": max_lift,
        "target_gripper_contact_in_audit_window": any(
            target_gripper_contact(item, target) for item in rows[close:end]
        ),
        "dense_bundle": dense_bundle,
        "dense_bundle_exists": bool(dense_bundle and Path(dense_bundle).is_file()),
        "video": str(video.resolve()),
        "video_exists": video.is_file(),
    }


def main() -> None:
    args = parse_args()
    if min(args.pre_close_context_steps, args.post_close_audit_steps) < 0:
        raise ValueError("audit window sizes must be non-negative")
    if args.close_confirmation_steps < 1:
        raise ValueError("close-confirmation-steps must be positive")
    classification = json.loads(args.failure_classification.read_text())
    requested = set(args.failure_type)
    source = [row for row in classification["records"] if row["failure_type"] in requested]
    records = [locate(row, args) for row in source]
    payload = {
        "kind": "libero_plus_failure_onset_manifest_v1",
        "schema_version": 1,
        "failure_classification": str(args.failure_classification.resolve()),
        "selector": {"failure_types": sorted(requested)},
        "configuration": {
            "pre_close_context_steps": args.pre_close_context_steps,
            "post_close_audit_steps": args.post_close_audit_steps,
            "close_confirmation_steps": args.close_confirmation_steps,
        },
        "num_records": len(records),
        "status_counts": dict(sorted(Counter(row["onset_status"] for row in records).items())),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "num_records": len(records),
        "status_counts": payload["status_counts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
