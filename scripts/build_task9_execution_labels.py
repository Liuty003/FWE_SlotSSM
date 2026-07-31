#!/usr/bin/env python
"""Build privileged teacher labels for natural task-9 recovery events.

Privileged object positions are used only to create offline supervision and
audit visual triggers.  They are never part of the runtime model input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MUG = "white_yellow_mug_1_pos"
TABLE_Z = 0.900


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-ids", type=int, nargs="+", required=True)
    parser.add_argument("--assessment-delay", type=int, default=15)
    return parser.parse_args()


def load_episode(root: Path, episode: int):
    path = root / "task9" / f"task9-episode{episode}" / "steps.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines()]


def vector(row, when, key):
    return np.asarray(row[f"privileged_object_state_{when}"][key], np.float32)


def eef(row, when):
    return np.asarray(
        row[f"robot_state_{when}"]["robot0_eef_pos"], np.float32
    )


def gripper_aperture(row, when="before"):
    qpos = np.asarray(
        row[f"robot_state_{when}"]["robot0_gripper_qpos"], np.float32
    )
    return float(np.abs(qpos).mean())


def find_grasp_attempts(rows, near_threshold=0.13):
    """Return close-action onsets near the target mug."""
    attempts = []
    previous_close = False
    for index, row in enumerate(rows):
        action = np.asarray(row["executed_action"], np.float32)
        closing = float(action[6]) > 0.5
        mug = vector(row, "before", MUG)
        relative = mug - eef(row, "before")
        if (
            closing
            and not previous_close
            and gripper_aperture(row) > 0.03
            and float(np.linalg.norm(relative)) < near_threshold
        ):
            attempts.append(index)
        previous_close = closing
    return attempts


def classify_attempt(rows, attempt, assessment_delay=15):
    assessment = min(attempt + assessment_delay, len(rows) - 1)
    start_mug = vector(rows[attempt], "before", MUG)
    start_eef = eef(rows[attempt], "before")
    current_mug = vector(rows[assessment], "after", MUG)
    current_eef = eef(rows[assessment], "after")
    lift = float(current_mug[2] - start_mug[2])
    relative_drift = float(np.linalg.norm(
        (current_mug - current_eef) - (start_mug - start_eef)
    ))
    acquired = lift >= 0.015 and relative_drift < 0.045
    return {
        "attempt_timestep": int(attempt),
        "assessment_timestep": int(assessment),
        "lift_m": lift,
        "relative_drift_m": relative_drift,
        "acquired": bool(acquired),
        "failed_acquisition": bool(not acquired),
    }


def first_transport_drop(rows, attempt, assessment, acquired):
    """Find a causal loss after a verified acquisition.

    A drop requires a previously lifted mug, a large fall from the running
    peak, and loss of the gripper-relative pose.  This excludes ordinary
    vertical transport and the pre-grasp collision in episode 6.
    """
    if not acquired:
        return None
    start_mug = vector(rows[attempt], "before", MUG)
    peak_z = float(start_mug[2])
    reference_relative = None
    for index in range(attempt, len(rows)):
        mug = vector(rows[index], "after", MUG)
        current_eef = eef(rows[index], "after")
        peak_z = max(peak_z, float(mug[2]))
        lift = float(mug[2] - start_mug[2])
        if reference_relative is None and lift >= 0.025:
            reference_relative = mug - current_eef
        if index <= assessment or reference_relative is None:
            continue
        drift = float(np.linalg.norm((mug - current_eef) - reference_relative))
        fall = float(peak_z - mug[2])
        on_table = float(mug[2]) < TABLE_Z + 0.012
        if fall > 0.03 and drift > 0.04 and on_table:
            return {
                "drop_timestep": int(index),
                "peak_z_m": peak_z,
                "current_z_m": float(mug[2]),
                "fall_from_peak_m": fall,
                "relative_drift_m": drift,
            }
    return None


def label_episode(rows, assessment_delay=15):
    attempts = find_grasp_attempts(rows)
    attempt_events = []
    for attempt in attempts:
        event = classify_attempt(rows, attempt, assessment_delay)
        event["transport_drop"] = first_transport_drop(
            rows,
            attempt=attempt,
            assessment=event["assessment_timestep"],
            acquired=event["acquired"],
        )
        attempt_events.append(event)

    output = []
    for index, row in enumerate(rows):
        mug = vector(row, "after", MUG)
        current_eef = eef(row, "after")
        relative = mug - current_eef
        active_attempt = next(
            (
                event for event in reversed(attempt_events)
                if event["attempt_timestep"] <= index
            ),
            None,
        )
        failed_acquisition = bool(
            active_attempt
            and index >= active_attempt["assessment_timestep"]
            and active_attempt["failed_acquisition"]
        )
        drop_event = next(
            (
                event["transport_drop"] for event in attempt_events
                if event["transport_drop"]
                and index >= event["transport_drop"]["drop_timestep"]
            ),
            None,
        )
        action = np.asarray(row["executed_action"], np.float32)
        output.append({
            "episode_id": int(row["episode_id"]),
            "timestep": int(row["timestep"]),
            "mug_relative_to_eef_xyz_m": relative.tolist(),
            "mug_height_m": float(mug[2]),
            "mug_lift_from_table_m": float(mug[2] - TABLE_Z),
            "gripper_aperture": gripper_aperture(row, "after"),
            "executed_action": action.tolist(),
            "acquisition_assessment_active": bool(
                active_attempt
                and active_attempt["assessment_timestep"]
                <= index
                <= active_attempt["assessment_timestep"] + 15
            ),
            "failed_acquisition": failed_acquisition,
            "transport_drop": bool(drop_event),
            "recovery_failure_type": (
                "transport_drop" if drop_event
                else "empty_acquisition" if failed_acquisition
                else "none"
            ),
        })
    return output, attempt_events


def main():
    args = arguments()
    all_rows, summaries = [], []
    for episode in args.episode_ids:
        rows = load_episode(args.source_root, episode)
        labels, events = label_episode(rows, args.assessment_delay)
        all_rows.extend(labels)
        summaries.append({
            "episode_id": episode,
            "frames": len(labels),
            "attempts": events,
        })
        print(json.dumps(summaries[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row) for row in all_rows) + "\n",
        encoding="utf-8",
    )
    metadata = args.output.with_name("metadata.json")
    metadata.write_text(json.dumps({
        "schema_version": 1,
        "input_contract": (
            "privileged object state is offline teacher supervision only; "
            "runtime inputs must be RGB, robot state, and executed actions"
        ),
        "episodes": summaries,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
