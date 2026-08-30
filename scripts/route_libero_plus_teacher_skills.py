#!/usr/bin/env python3
"""Route located LIBERO-Plus failures to task-agnostic privileged teacher skills."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-context-steps", type=int, default=32)
    parser.add_argument("--family-index", type=Path)
    return parser.parse_args()


def family_by_sample_id(path: Path) -> dict[str, str]:
    index = json.loads(path.read_text())
    result: dict[str, str] = {}
    for family in index["families"]:
        family_name = family["family_name"]
        manifest = json.loads(Path(family["family_manifest"]).read_text())
        for record in manifest["records"]:
            sample_id = record["sample_id"]
            if sample_id in result:
                raise ValueError(f"sample belongs to multiple families: {sample_id}")
            result[sample_id] = family_name
    return result


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def distance(first, second) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(first, second)))


def predicate_values(row: dict) -> list[tuple[list[str], bool | None]]:
    return [
        (list(goal["predicate"]), goal.get("value"))
        for goal in row["privileged_audit"]["goal_predicates"]
    ]


def predicate_value(row: dict, predicate: list[str]) -> bool | None:
    for candidate, value in predicate_values(row):
        if candidate == predicate:
            return value
    return None


def quaternion_rotation_matrix_wxyz(quaternion: list[float]) -> list[list[float]]:
    w, x, y, z = (float(value) for value in quaternion)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def pre_irreversible_tilt_fork(
    rows: list[dict],
    predicate: list[str],
    *,
    pre_context: int,
    confirmation: int = 4,
    minimum_initial_alignment: float = 0.9,
    alignment_drop: float = 0.1,
) -> int | None:
    """Locate the onset of an upright target being knocked over.

    LIBERO object meshes do not share a canonical local-up axis.  We therefore
    choose whichever local body axis is most vertical in the initial state and
    track that same axis.  Initially slanted objects such as books are excluded
    instead of being incorrectly canonicalized upright.
    """
    if len(predicate) < 2 or predicate[0].lower() not in {"in", "on"}:
        return None
    target = predicate[1]
    try:
        initial_quaternion = rows[0]["privileged_audit"]["entities"][target][
            "quaternion_wxyz"
        ]
    except (KeyError, IndexError, TypeError):
        return None
    initial_rotation = quaternion_rotation_matrix_wxyz(initial_quaternion)
    vertical_components = [abs(value) for value in initial_rotation[2]]
    local_up_axis = max(range(3), key=vertical_components.__getitem__)
    initial_alignment = vertical_components[local_up_axis]
    if initial_alignment < minimum_initial_alignment:
        return None
    threshold = initial_alignment - alignment_drop
    for index in range(len(rows) - confirmation + 1):
        alignments = []
        for row in rows[index:index + confirmation]:
            try:
                quaternion = row["privileged_audit"]["entities"][target]["quaternion_wxyz"]
            except (KeyError, TypeError):
                break
            rotation = quaternion_rotation_matrix_wxyz(quaternion)
            alignments.append(abs(rotation[2][local_up_axis]))
        if len(alignments) == confirmation and all(value < threshold for value in alignments):
            return max(0, index - pre_context)
    return None


def failed_release_fork(
    rows: list[dict],
    predicate: list[str],
    *,
    pre_context: int,
    confirmation: int = 4,
    lift_height: float = 0.02,
    held_distance: float = 0.14,
) -> int | None:
    """Return a fork just before a failed placement release.

    This intentionally uses privileged geometry only in the offline teacher
    builder.  A candidate release must follow evidence that the selected target
    was lifted near the end effector, remain open for several steps, and still
    leave the selected goal false.  The online recovery model does not receive
    these fields.
    """
    if len(predicate) < 2 or predicate[0].lower() not in {"in", "on"}:
        return None
    target = predicate[1]
    try:
        initial_z = float(rows[0]["privileged_audit"]["entities"][target]["position"][2])
    except (KeyError, IndexError, TypeError):
        return None

    lifted_before = False
    for index, row in enumerate(rows):
        try:
            target_position = row["privileged_audit"]["entities"][target]["position"]
            eef_position = row["online_observation"]["proprio"]["robot0_eef_pos"]
            gripper_closed = float(row["executed_action"][6]) > 0
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        lifted_before = lifted_before or (
            float(target_position[2]) >= initial_z + lift_height
            and distance(target_position, eef_position) <= held_distance
        )
        if not lifted_before or index == 0 or gripper_closed:
            continue
        try:
            was_closed = float(rows[index - 1]["executed_action"][6]) > 0
            stable_open = all(
                float(future["executed_action"][6]) <= 0
                for future in rows[index:index + confirmation]
            )
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if (
            was_closed
            and len(rows[index:index + confirmation]) == confirmation
            and stable_open
            and predicate_value(row, predicate) is not True
        ):
            return max(0, index - pre_context)
    return None


def relational_score(rows: list[dict], event: int, predicate: list[str], failure_type: str,
                     context: int) -> tuple[float, float]:
    target = predicate[1]
    start = rows[0]["privileged_audit"]["entities"][target]["position"]
    prefix = rows[:event + 1]
    maximum_motion = max(
        distance(start, row["privileged_audit"]["entities"][target]["position"])
        for row in prefix
    )
    local = rows[max(0, event - context):event + 1]
    closest_eef = min(
        distance(
            row["online_observation"]["proprio"]["robot0_eef_pos"],
            row["privileged_audit"]["entities"][target]["position"],
        )
        for row in local
    )
    # Placement failures should follow the object that actually moved; empty
    # grasps should follow the object the gripper approached most closely.
    return ((-maximum_motion, closest_eef) if failure_type == "transfer_or_placement_failure"
            else (closest_eef, -maximum_motion))


def choose_predicate(record: dict, rows: list[dict], event: int, context: int) -> tuple[list[str] | None, str]:
    failure_type = record["failure_type"]
    # For partial tasks the first stable partial state may occur at t=0 (for
    # example, a stove is initially on).  Select among goals that are still
    # unsatisfied at the proposed recovery fork, not among stale t=0 goals that
    # the nominal policy subsequently completed.
    selection_event = (
        int(record["fork_timestep"])
        if failure_type == "partial_goal_completion"
        else event
    )
    current = predicate_values(rows[selection_event])
    if failure_type == "goal_regression" and event > 0:
        previous = predicate_values(rows[event - 1])
        regressed = [
            predicate for (predicate, before), (_, after) in zip(previous, current)
            if before is True and after is False
        ]
        if regressed:
            return regressed[0], "regressed_goal"
    remaining = [predicate for predicate, value in current if value is not True]
    if not remaining:
        return None, "no_remaining_goal"
    if failure_type == "partial_goal_completion":
        relational = [predicate for predicate in remaining if predicate[0].lower() in {"in", "on"}]
        if relational:
            # A partially completed task can still contain both an object goal
            # and a later articulation goal.  Recover the remaining relational
            # manipulation first; successful demonstrations can later teach the
            # subgoal order without hard-coding task ids.
            return min(
                relational,
                key=lambda predicate: relational_score(
                    rows, len(rows) - 1, predicate, failure_type, context
                ),
            ), "attempted_remaining_relational_goal"
        return remaining[0], "remaining_goal_after_partial_completion"
    relational = [predicate for predicate in remaining if predicate[0].lower() in {"in", "on"}]
    if relational:
        return min(
            relational,
            key=lambda predicate: relational_score(rows, event, predicate, failure_type, context),
        ), "attempted_relational_goal"
    return remaining[0], "remaining_nonrelational_goal"


def route(record: dict, args: argparse.Namespace) -> dict:
    common = {**record}
    if record.get("onset_status") != "located":
        return {**common, "route_status": "manual_review", "teacher_skill": None}
    rows = load_jsonl(Path(record["steps"]))
    event = int(record["event_timestep"])
    predicate, reason = choose_predicate(record, rows, event, args.contact_context_steps)
    if predicate is None:
        return {
            **common, "route_status": "unsupported", "route_reason": reason,
            "teacher_skill": None, "selected_predicate": None,
        }
    name = predicate[0].lower()
    original_fork = int(record["fork_timestep"])
    fork = original_fork
    fork_strategy = record.get("onset_strategy")
    if record["failure_type"] in {"partial_goal_completion", "transfer_or_placement_failure"}:
        release_fork = failed_release_fork(
            rows,
            predicate,
            pre_context=args.contact_context_steps // 4,
        )
        if release_fork is not None:
            fork = release_fork
            fork_strategy = "pre_failed_target_release"
        else:
            tilt_fork = pre_irreversible_tilt_fork(
                rows,
                predicate,
                pre_context=3 * args.contact_context_steps // 4,
            )
            # This detector is a safeguard against an already-irreversible
            # nominal fork.  It may only move recovery earlier, never discard
            # a cleaner pre-contact fork in favor of a later tilt event.
            if tilt_fork is not None and tilt_fork < fork:
                fork = tilt_fork
                fork_strategy = "pre_irreversible_target_tilt"

    value_at_fork = predicate_value(rows[fork], predicate)
    common.update({
        "original_fork_timestep": original_fork,
        "fork_timestep": fork,
        "teacher_fork_timestep": fork,
        "teacher_fork_strategy": fork_strategy,
        "selected_predicate_value_at_fork": value_at_fork,
    })
    if value_at_fork is True:
        return {
            **common,
            "route_status": "invalid_fork_already_satisfied",
            "route_reason": "selected_predicate_true_at_teacher_fork",
            "teacher_skill": None,
            "selected_predicate": predicate,
        }
    if name in {"in", "on"} and len(predicate) >= 3:
        routed = {
            "teacher_skill": "relational_pick_place",
            "target_name": predicate[1],
            "destination_name": predicate[2],
        }
    elif name == "close" and len(predicate) >= 2:
        routed = {"teacher_skill": "articulation_close", "fixture_or_region": predicate[1]}
    elif name == "turnon" and len(predicate) >= 2:
        routed = {"teacher_skill": "articulation_turnon", "fixture_or_region": predicate[1]}
    else:
        return {
            **common, "route_status": "unsupported", "route_reason": reason,
            "teacher_skill": None, "selected_predicate": predicate,
        }
    return {
        **common, "route_status": "routed", "route_reason": reason,
        "selected_predicate": predicate, **routed,
    }


def main() -> None:
    args = arguments()
    if args.contact_context_steps < 1:
        raise ValueError("contact-context-steps must be positive")
    onset = json.loads(args.onset_manifest.read_text())
    if args.family_index is not None:
        families = family_by_sample_id(args.family_index)
        missing = []
        for record in onset["records"]:
            family_name = families.get(record["sample_id"])
            if family_name is None:
                missing.append(record["sample_id"])
            else:
                record["family_name"] = family_name
        if missing:
            raise ValueError(f"family mapping missing for {len(missing)} records")
    records = [route(record, args) for record in onset["records"]]
    payload = {
        "kind": "libero_plus_privileged_teacher_route_manifest_v2",
        "schema_version": 2,
        "onset_manifest": str(args.onset_manifest.resolve()),
        "routing_contract": "offline privileged route; task id is not a routing feature",
        "num_records": len(records),
        "route_status_counts": dict(sorted(Counter(record["route_status"] for record in records).items())),
        "teacher_skill_counts": dict(sorted(Counter(
            str(record.get("teacher_skill")) for record in records
        ).items())),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output), "num_records": len(records),
        "route_status_counts": payload["route_status_counts"],
        "teacher_skill_counts": payload["teacher_skill_counts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
