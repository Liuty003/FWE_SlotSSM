#!/usr/bin/env python
"""Build privileged teacher labels for natural task-6 execution failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PUDDING = "chocolate_pudding_1_pos"
MUG = "porcelain_mug_1_pos"
RIGHT_REGION = (0.10, 0.20, 0.05, 0.15)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-ids", type=int, nargs="+", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    return parser.parse_args()


def distance_to_region(position, region=RIGHT_REGION):
    x0, x1, y0, y1 = region
    x, y = float(position[0]), float(position[1])
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return float(np.hypot(dx, dy))


def find_steps(roots, episode):
    name = Path(f"task6-episode{episode}") / "steps.jsonl"
    matches = [root / name for root in roots if (root / name).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one trace for episode {episode}, found {matches}"
        )
    return matches[0]


def stable_stage2(rows, confirm=5):
    result, streak, active = [], 0, False
    for row in rows:
        value = bool(row.get("mug_on_plate_oracle", False))
        streak = streak + 1 if value else 0
        active = active or streak >= confirm
        result.append(active)
    return result


def grasp_attempt_indices(rows, stage2):
    attempts = []
    for index, row in enumerate(rows):
        before = row["privileged_object_state_before"]
        pudding = np.asarray(before[PUDDING], np.float32)
        eef = np.asarray(
            row["robot_state_before"]["robot0_eef_pos"], np.float32
        )
        gripper = np.asarray(
            row["robot_state_before"]["robot0_gripper_qpos"], np.float32
        )
        action = np.asarray(row["executed_action"], np.float32)
        if (
            stage2[index]
            and float(action[6]) > 0.5
            and float(np.abs(gripper).mean()) > 0.03
            and float(np.linalg.norm(pudding - eef)) < 0.11
        ):
            attempts.append(index)
    return attempts


def label_episode(rows, horizon):
    stage2 = stable_stage2(rows)
    attempts = grasp_attempt_indices(rows, stage2)
    stage2_start = next(
        (index for index, value in enumerate(stage2) if value), None
    )
    stage2_mug = (
        np.asarray(
            rows[stage2_start]["privileged_object_state_before"][MUG],
            np.float32,
        )
        if stage2_start is not None else None
    )
    output = []
    for index, row in enumerate(rows):
        future_index = min(index + horizon, len(rows) - 1)
        acquisition_index = min(index + 20, len(rows) - 1)
        future = rows[future_index]
        acquisition_future = rows[acquisition_index]
        before = row["privileged_object_state_before"]
        future_state = future["privileged_object_state_after"]
        pudding = np.asarray(before[PUDDING], np.float32)
        pudding_future = np.asarray(future_state[PUDDING], np.float32)
        mug = np.asarray(before[MUG], np.float32)
        mug_future = np.asarray(future_state[MUG], np.float32)
        eef = np.asarray(row["robot_state_before"]["robot0_eef_pos"], np.float32)
        eef_future = np.asarray(
            future["robot_state_after"]["robot0_eef_pos"], np.float32
        )
        gripper = np.asarray(
            row["robot_state_before"]["robot0_gripper_qpos"], np.float32
        )
        action = np.asarray(row["executed_action"], np.float32)
        distance = distance_to_region(pudding)
        future_distance = distance_to_region(pudding_future)
        progress = distance - future_distance
        pudding_displacement = float(np.linalg.norm(pudding_future - pudding))
        mug_displacement = float(np.linalg.norm(mug_future - mug))
        mug_stage_displacement = (
            float(np.linalg.norm(mug - stage2_mug))
            if stage2_mug is not None and stage2[index] else 0.0
        )
        relative = pudding - eef
        future_relative = pudding_future - eef_future
        relative_drift = float(np.linalg.norm(future_relative - relative))
        gripper_closed = float(np.abs(gripper).mean()) < 0.025
        gripper_open = float(np.abs(gripper).mean()) > 0.03
        near_pudding = float(np.linalg.norm(relative)) < 0.11
        action_effort = float(np.linalg.norm(action[:6]))
        reaches_goal = future_distance == 0.0
        effective_progress = reaches_goal or progress > 0.005
        stall = (
            stage2[index]
            and action_effort > 0.15
            and progress < 0.002
            and pudding_displacement < 0.004
        )
        controlled = (
            stage2[index]
            and gripper_closed
            and near_pudding
            and pudding_displacement > 0.004
            and relative_drift < 0.025
        )
        acquisition_state = acquisition_future[
            "privileged_object_state_after"
        ]
        acquisition_lift = float(
            acquisition_state[PUDDING][2] - pudding[2]
        )
        grasp_attempt = (
            stage2[index]
            and float(action[6]) > 0.5
            and gripper_open
            and near_pudding
        )
        future_failed_acquisition = grasp_attempt and acquisition_lift < 0.01
        # A recovery trigger must be causal.  Only call the grasp currently
        # failed after enough post-close observations exist and the pudding
        # still has not lifted.  The prospective label above remains useful
        # as a distinct future-risk target.
        eligible_attempts = [
            attempt for attempt in attempts
            if 15 <= index - attempt <= 30
        ]
        acquisition_assessment_active = bool(eligible_attempts)
        failed_acquisition = False
        if eligible_attempts:
            attempt = eligible_attempts[-1]
            attempt_pudding = np.asarray(
                rows[attempt]["privileged_object_state_before"][PUDDING],
                np.float32,
            )
            attempt_eef = np.asarray(
                rows[attempt]["robot_state_before"]["robot0_eef_pos"],
                np.float32,
            )
            current_pudding = np.asarray(
                row["privileged_object_state_after"][PUDDING], np.float32
            )
            current_eef = np.asarray(
                row["robot_state_after"]["robot0_eef_pos"], np.float32
            )
            acquisition_relative_drift = float(np.linalg.norm(
                (current_pudding - current_eef)
                - (attempt_pudding - attempt_eef)
            ))
            failed_acquisition = (
                stage2[index]
                and float(current_pudding[2] - attempt_pudding[2]) < 0.01
                and acquisition_relative_drift > 0.02
            )
        causal_slip_drift = 0.0
        slip_lookback = 15
        if index >= slip_lookback:
            past = rows[index - slip_lookback]
            past_pudding = np.asarray(
                past["privileged_object_state_before"][PUDDING],
                np.float32,
            )
            past_eef = np.asarray(
                past["robot_state_before"]["robot0_eef_pos"], np.float32
            )
            causal_slip_drift = float(np.linalg.norm(
                relative - (past_pudding - past_eef)
            ))
        slip_risk = (
            stage2[index]
            and gripper_closed
            and near_pudding
            # Causal evidence: the object is no longer maintaining its
            # gripper-relative pose over the preceding observation window.
            # This detects episode3 before the arm pushes toward the mug,
            # whereas the old future-horizon threshold produced zero labels.
            and causal_slip_drift > 0.015
            and not reaches_goal
        )
        mug_on_plate_future = bool(
            future.get("mug_on_plate_oracle", False)
        )
        collateral_risk = (
            stage2[index]
            and (
                # Successful stage-1 placements settle by up to 7.6 mm in
                # this development set.  Eight millimetres separates that
                # settling from the visible task-6 collateral collisions.
                mug_stage_displacement > 0.008
                or mug_displacement > 0.012
                or not mug_on_plate_future
            )
        )
        candidate_value = float(
            progress
            - 2.0 * max(mug_displacement - 0.004, 0.0)
            - 0.02 * float(slip_risk)
        )
        output.append({
            "episode_id": int(row["episode_id"]),
            "timestep": int(row["timestep"]),
            "future_timestep": int(future["timestep"]),
            "stage2_active": bool(stage2[index]),
            "pudding_goal_distance_m": distance,
            "future_pudding_goal_distance_m": future_distance,
            "pudding_goal_progress_m": progress,
            "pudding_displacement_m": pudding_displacement,
            "pudding_eef_relative_drift_m": relative_drift,
            "causal_slip_drift_m": causal_slip_drift,
            "pudding_relative_to_eef_xyz_m": relative.tolist(),
            "mug_displacement_m": mug_displacement,
            "mug_stage_displacement_m": mug_stage_displacement,
            "gripper_closed": bool(gripper_closed),
            "near_pudding": bool(near_pudding),
            "effective_progress": bool(effective_progress),
            "stall": bool(stall),
            "controlled": bool(controlled),
            "grasp_attempt": bool(grasp_attempt),
            "acquisition_assessment_active": acquisition_assessment_active,
            "future_failed_acquisition": bool(future_failed_acquisition),
            "failed_acquisition": bool(failed_acquisition),
            "slip_risk": bool(slip_risk),
            "collateral_risk": bool(collateral_risk),
            "candidate_value": candidate_value,
            "executed_action": row["executed_action"],
            "robot_features": [
                *eef.tolist(),
                float(np.abs(gripper).mean()),
            ],
        })
    return output


def main():
    args = arguments()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = []
    for episode in args.episode_ids:
        source = find_steps(args.source_root, episode)
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        labels = label_episode(rows, args.horizon)
        all_rows.extend(labels)
        stage2 = [row for row in labels if row["stage2_active"]]
        summary = {
            "episode_id": episode,
            "frames": len(labels),
            "stage2_frames": len(stage2),
            **{
                key: sum(row[key] for row in stage2)
                for key in (
                    "effective_progress", "stall", "controlled",
                    "grasp_attempt", "failed_acquisition",
                    "slip_risk", "collateral_risk",
                )
            },
            "source": str(source),
        }
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")
    metadata = {
        "schema_version": 1,
        "kind": "task6_privileged_execution_teacher_labels",
        "horizon": args.horizon,
        "model_input_excludes_privileged_state": True,
        "episodes": summaries,
        "rows": len(all_rows),
        "output": str(args.output),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
