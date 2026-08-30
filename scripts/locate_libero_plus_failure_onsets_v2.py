#!/usr/bin/env python3
"""Locate failure-type-aware recovery forks in LIBERO-Plus archival rollouts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SUPPORTED = {
    "empty_grasp_or_failed_contact",
    "transfer_or_placement_failure",
    "partial_goal_completion",
    "goal_regression",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-classification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-event-context-steps", type=int, default=8)
    parser.add_argument("--confirmation-steps", type=int, default=4)
    parser.add_argument("--partial-recovery-budget", type=int, default=120)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def goal_values(rows: list[dict]) -> list[list[bool | None]]:
    return [
        [goal.get("value") for goal in row["privileged_audit"]["goal_predicates"]]
        for row in rows
    ]


def first_confirmed_close(rows: list[dict], confirmation: int = 2) -> int | None:
    close = [float(row["executed_action"][6]) > 0 for row in rows]
    for index in range(len(close) - confirmation + 1):
        if all(close[index:index + confirmation]):
            return index
    return None


def first_stable_partial(values: list[list[bool | None]], confirmation: int) -> int | None:
    for index, state in enumerate(values):
        satisfied = [position for position, value in enumerate(state) if value is True]
        if not 0 < len(satisfied) < len(state):
            continue
        window = values[index:min(len(values), index + confirmation)]
        if len(window) == confirmation and all(
            all(item[position] is True for position in satisfied) for item in window
        ):
            return index
    return None


def first_regression(values: list[list[bool | None]]) -> int | None:
    for index in range(1, len(values)):
        if any(before is True and after is False for before, after in zip(values[index - 1], values[index])):
            return index
    return None


def locate(record: dict, args: argparse.Namespace) -> dict:
    steps_path = Path(record["steps"])
    rows = load_jsonl(steps_path)
    video = steps_path.with_name("rollout.mp4")
    dense_bundle = rows[0].get("online_observation", {}).get("dense_bundle") if rows else None
    common = {
        **record,
        "num_transitions": len(rows),
        "video": str(video.resolve()),
        "video_exists": video.is_file() and video.stat().st_size > 0,
        "dense_bundle": dense_bundle,
        "dense_bundle_exists": bool(dense_bundle and Path(dense_bundle).is_file()),
    }
    if not rows or record["failure_type"] not in SUPPORTED:
        return {
            **common, "onset_status": "manual_review", "onset_strategy": None,
            "event_timestep": None, "fork_timestep": None,
        }

    failure_type = record["failure_type"]
    values = goal_values(rows)
    if failure_type == "partial_goal_completion":
        event = first_stable_partial(values, args.confirmation_steps)
        # Start late enough to expose the nominal failure, but preserve a full
        # bounded recovery budget. Never start before the completed subgoal is stable.
        fork = None if event is None else max(event, len(rows) - args.partial_recovery_budget)
        strategy = "stable_partial_goal_then_budgeted_tail"
    elif failure_type == "goal_regression":
        event = first_regression(values)
        # A recovery-action teacher must start after the predicate has actually
        # regressed.  The preceding context is still useful for gate training,
        # but using it as the action fork creates a contradictory positive in
        # which the teacher is asked to redo an already-satisfied goal.
        fork = event
        strategy = "post_goal_regression"
    else:
        event = first_confirmed_close(rows)
        fork = None if event is None else max(0, event - args.pre_event_context_steps)
        strategy = "pre_first_confirmed_close"

    if event is None or fork is None:
        return {
            **common, "onset_status": "event_not_found", "onset_strategy": strategy,
            "event_timestep": None, "fork_timestep": None,
        }
    predicates = rows[event]["privileged_audit"]["goal_predicates"]
    return {
        **common,
        "onset_status": "located",
        "onset_strategy": strategy,
        "event_timestep": event,
        "fork_timestep": fork,
        "teacher_fork_timestep": fork,
        "gate_context_timestep": max(0, event - args.pre_event_context_steps),
        "available_recorded_steps": len(rows) - fork,
        "goal_values_at_event": [goal.get("value") for goal in predicates],
        "goal_predicates_at_event": [goal.get("predicate") for goal in predicates],
    }


def main() -> None:
    args = arguments()
    if min(args.pre_event_context_steps, args.confirmation_steps, args.partial_recovery_budget) < 1:
        raise ValueError("onset parameters must be positive")
    classification = json.loads(args.failure_classification.read_text())
    records = [
        locate(record, args) for record in classification["records"]
        if record["failure_type"] != "success"
    ]
    payload = {
        "kind": "libero_plus_failure_onset_manifest_v3",
        "schema_version": 3,
        "failure_classification": str(args.failure_classification.resolve()),
        "configuration": {
            "pre_event_context_steps": args.pre_event_context_steps,
            "confirmation_steps": args.confirmation_steps,
            "partial_recovery_budget": args.partial_recovery_budget,
        },
        "num_records": len(records),
        "status_counts": dict(sorted(Counter(record["onset_status"] for record in records).items())),
        "strategy_counts": dict(sorted(Counter(str(record["onset_strategy"]) for record in records).items())),
        "failure_type_status_counts": {
            failure_type: dict(sorted(Counter(
                record["onset_status"] for record in records if record["failure_type"] == failure_type
            ).items()))
            for failure_type in sorted({record["failure_type"] for record in records})
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output), "num_records": len(records),
        "status_counts": payload["status_counts"],
        "failure_type_status_counts": payload["failure_type_status_counts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
