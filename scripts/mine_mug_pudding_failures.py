#!/usr/bin/env python3
"""Mine causal mug+pudding failures from recorded LIBERO-plus OFT actions.

The source traces contain deployable observations and an exact MuJoCo snapshot
every eight steps, but intentionally contain no privileged object state.  This
tool restores the step-0 snapshot, replays the already-executed actions (no VLA
inference), and uses simulator state only to create offline teacher labels.

The output is a manifest, not a training dataset.  Ambiguous failed episodes
are routed to ``needs_review`` instead of being assigned a fabricated recovery
fork.  Exact fork reconstruction and teacher search consume the accepted rows
in a later stage of the data factory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


MUG = "porcelain_mug_1_pos"
PUDDING = "chocolate_pudding_1_pos"
PLATE = "plate_1_pos"
RED_MUG = "red_coffee_mug_1_pos"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout-root",
        type=Path,
        default=Path(
            "/data2/lty/naive-e-slotssm-runs/oft-plus-robot-global-v1/forks"
        ),
    )
    parser.add_argument("--task-start", type=int, default=525)
    parser.add_argument("--task-end", type=int, default=566)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-report-root",
        type=Path,
        action="append",
        default=[],
        help="Root searched recursively for libero_10-taskN-episode0.json reports.",
    )
    parser.add_argument(
        "--libero-root", type=Path, default=Path("/data/lty/code/LIBERO-plus")
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for expected, row in enumerate(rows):
        if int(row["timestep"]) != expected:
            raise ValueError(f"non-contiguous trajectory at {path}:{expected}")
    return rows


def vector(obs: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(obs[key], dtype=np.float32)


def robot_state(obs: dict[str, Any]) -> dict[str, list[float]]:
    return {
        key: np.asarray(obs[key]).tolist()
        for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    }


def object_state(obs: dict[str, Any]) -> dict[str, list[float]]:
    keys = [MUG, PUDDING, PLATE]
    if RED_MUG in obs:
        keys.append(RED_MUG)
    return {key: vector(obs, key).tolist() for key in keys}


def stable_true(values: list[bool], confirm: int = 5) -> list[bool]:
    output: list[bool] = []
    active = False
    streak = 0
    for value in values:
        streak = streak + 1 if value else 0
        active = active or streak >= confirm
        output.append(active)
    return output


def event_candidates(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return causal recovery candidates and an episode-level stage summary."""
    mug_predicate = [bool(row["mug_on_plate_before"]) for row in rows]
    stage2 = stable_true(mug_predicate)
    stage2_start = next((i for i, active in enumerate(stage2) if active), None)
    attempts: dict[str, list[int]] = {"mug": [], "pudding": []}
    candidates: list[dict[str, Any]] = []

    def state(index: int, key: str) -> np.ndarray:
        return np.asarray(rows[index]["privileged_object_state_before"][key], np.float32)

    def eef(index: int) -> np.ndarray:
        return np.asarray(rows[index]["robot_state_before"]["robot0_eef_pos"], np.float32)

    for index, row in enumerate(rows):
        action = np.asarray(row["executed_action"], np.float32)
        gripper = np.asarray(row["robot_state_before"]["robot0_gripper_qpos"], np.float32)
        gripper_open = float(np.abs(gripper).mean()) > 0.03
        # A common Plus failure is semantic target confusion: during stage 1
        # OFT descends/closes at the red distractor while remaining far from
        # the instructed white (porcelain) mug.  It is causal at close time
        # and should not be mislabeled as a late generic timeout.
        objects = row["privileged_object_state_before"]
        closing_edge = bool(
            float(action[6]) > 0.5
            and (index == 0 or float(rows[index - 1]["executed_action"][6]) < 0.0)
        )
        active_role = "pudding" if stage2[index] else "mug"
        active_key = PUDDING if stage2[index] else MUG
        active_distance = float(np.linalg.norm(state(index, active_key) - eef(index)))
        if closing_edge and active_distance > 0.18:
            candidates.append({
                "timestep": index,
                "failure_type": "premature_empty_close",
                "target_role": active_role,
                "evidence": {
                    "intended_object": (
                        "chocolate_pudding_1" if stage2[index] else "porcelain_mug_1"
                    ),
                    "intended_eef_distance_m": active_distance,
                    "previous_gripper_action": float(rows[index - 1]["executed_action"][6])
                    if index else None,
                    "current_gripper_action": float(action[6]),
                },
                "confidence": "high",
            })
        if not stage2[index] and RED_MUG in objects and float(action[6]) > 0.5:
            target_distance = float(np.linalg.norm(state(index, MUG) - eef(index)))
            distractor_distance = float(np.linalg.norm(state(index, RED_MUG) - eef(index)))
            # ``robot0_eef_pos`` is the controller frame above the fingertips,
            # so absolute grasp distances are larger than object/finger contact
            # distances.  Require the distractor to be both locally close and
            # at least 5 cm closer than the instructed target.
            if distractor_distance < 0.20 and target_distance - distractor_distance > 0.05:
                candidates.append({
                    "timestep": index,
                    "failure_type": "wrong_object_acquisition",
                    "target_role": "mug",
                    "evidence": {
                        "intended_object": "porcelain_mug_1",
                        "distractor_object": "red_coffee_mug_1",
                        "intended_eef_distance_m": target_distance,
                        "distractor_eef_distance_m": distractor_distance,
                    },
                    "confidence": "high",
                })
        for role, key, enabled in (
            ("mug", MUG, not stage2[index]),
            ("pudding", PUDDING, stage2[index]),
        ):
            near = float(np.linalg.norm(state(index, key) - eef(index))) < 0.11
            if enabled and float(action[6]) > 0.5 and gripper_open and near:
                attempts[role].append(index)

        role = "pudding" if stage2[index] else "mug"
        key = PUDDING if role == "pudding" else MUG
        eligible = [attempt for attempt in attempts[role] if 15 <= index - attempt <= 30]
        if eligible:
            attempt = eligible[-1]
            object_lift = float(state(index, key)[2] - state(attempt, key)[2])
            relative_drift = float(np.linalg.norm(
                (state(index, key) - eef(index)) - (state(attempt, key) - eef(attempt))
            ))
            if object_lift < 0.01 and relative_drift > 0.02:
                candidates.append({
                    "timestep": index,
                    "failure_type": "failed_acquisition",
                    "target_role": role,
                    "evidence": {
                        "attempt_timestep": attempt,
                        "object_lift_m": object_lift,
                        "object_eef_relative_drift_m": relative_drift,
                    },
                    "confidence": "high",
                })
                # One causal label per close attempt is enough.
                attempts[role] = [value for value in attempts[role] if value != attempt]

        lookback = 15
        if index >= lookback:
            past = index - lookback
            closed = float(np.abs(gripper).mean()) < 0.025
            near = float(np.linalg.norm(state(index, key) - eef(index))) < 0.11
            drift = float(np.linalg.norm(
                (state(index, key) - eef(index)) - (state(past, key) - eef(past))
            ))
            if closed and near and drift > 0.015:
                candidates.append({
                    "timestep": index,
                    "failure_type": "slip",
                    "target_role": role,
                    "evidence": {"lookback": lookback, "relative_drift_m": drift},
                    "confidence": "medium",
                })

        if stage2[index] and not row["mug_on_plate_after"]:
            candidates.append({
                "timestep": index,
                "failure_type": "collateral_mug_displacement",
                "target_role": "mug",
                "evidence": {"mug_on_plate_before": True, "mug_on_plate_after": False},
                "confidence": "high",
            })

    # Merge repeated frame-level detections into event onsets.  Events of the
    # same kind within 12 steps describe the same causal failure episode.
    candidates.sort(key=lambda item: (item["timestep"], item["failure_type"]))
    merged: list[dict[str, Any]] = []
    for item in candidates:
        previous = next((
            old for old in reversed(merged)
            if old["failure_type"] == item["failure_type"]
            and old["target_role"] == item["target_role"]
        ), None)
        if previous is None or item["timestep"] - previous["timestep"] > 12:
            merged.append(item)
    return merged, {
        "stage2_reached": stage2_start is not None,
        "stage2_start_timestep": stage2_start,
        "final_mug_on_plate": bool(rows[-1]["mug_on_plate_after"]),
        "final_pudding_in_right_region": bool(rows[-1]["pudding_in_right_region_after"]),
        "task_success_from_predicates": bool(
            rows[-1]["mug_on_plate_after"] and rows[-1]["pudding_in_right_region_after"]
        ),
    }


def choose_failure(
    candidates: list[dict[str, Any]], summary: dict[str, Any], baseline_success: bool
) -> tuple[dict[str, Any] | None, str]:
    if baseline_success:
        return None, "normal_success"
    high = [item for item in candidates if item["confidence"] == "high"]
    if high:
        return high[0], "auto_high_confidence"
    if candidates:
        return candidates[0], "needs_review_medium_confidence"
    if not summary["stage2_reached"]:
        return None, "needs_review_mug_stage_incomplete_without_causal_event"
    return None, "needs_review_pudding_stage_incomplete_without_causal_event"


def locate_report(roots: list[Path], task_id: int, episode: int) -> Path | None:
    name = f"libero_10-task{task_id}-episode{episode}.json"
    matches = [match for root in roots for match in root.rglob(name)]
    if len(matches) > 1:
        raise RuntimeError(f"multiple baseline reports for task {task_id}: {matches}")
    return matches[0] if matches else None


def replay_task(args: argparse.Namespace, task_id: int, env: Any) -> tuple[list[dict[str, Any]], Path]:
    source = args.rollout_root / f"task{task_id}-episode{args.episode}"
    trajectory = source / "trajectory.jsonl"
    snapshot = source / "step0000" / "mujoco_state.npz"
    if not trajectory.is_file() or not snapshot.is_file():
        raise FileNotFoundError(f"incomplete source rollout: {source}")
    actions = read_jsonl(trajectory)
    flat_state = np.load(snapshot)["mujoco_state"]
    obs = env.regenerate_obs_from_state(flat_state)
    controller = env.env.robots[0].controller
    controller.update(force=True)
    controller.reset_goal()
    rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(actions):
        before = obs
        mug_before = bool(env.env._eval_predicate(["on", "porcelain_mug_1", "plate_1"]))
        pudding_before = bool(env.env._eval_predicate(
            ["on", "chocolate_pudding_1", "living_room_table_plate_right_region"]
        ))
        obs, _, done, _ = env.step(source_row["executed_action"])
        mug_after = bool(env.env._eval_predicate(["on", "porcelain_mug_1", "plate_1"]))
        pudding_after = bool(env.env._eval_predicate(
            ["on", "chocolate_pudding_1", "living_room_table_plate_right_region"]
        ))
        rows.append({
            "task_id": task_id,
            "episode_id": args.episode,
            "timestep": index,
            "executed_action": source_row["executed_action"],
            "robot_state_before": robot_state(before),
            "robot_state_after": robot_state(obs),
            "privileged_object_state_before": object_state(before),
            "privileged_object_state_after": object_state(obs),
            "mug_on_plate_before": mug_before,
            "mug_on_plate_after": mug_after,
            "pudding_in_right_region_before": pudding_before,
            "pudding_in_right_region_after": pudding_after,
            "done_after": bool(done),
        })
    return rows, trajectory


def main() -> None:
    args = arguments()
    if args.task_end < args.task_start:
        raise ValueError("task-end must not precede task-start")
    default_reports = [
        Path("/data2/lty/naive-e-slotssm-runs/libero-plus/oft-plus-robot-levels1-4-v1"),
        Path("/data2/lty/naive-e-slotssm-runs/libero-plus/oft-plus-robot-level5-v1"),
    ]
    report_roots = args.baseline_report_root or default_reports
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    sys.path[:0] = [str(args.libero_root), str(Path(__file__).resolve().parents[1])]

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for task_id in range(args.task_start, args.task_end + 1):
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
        )
        env.seed(0)
        env.reset()
        env.set_init_state(suite.get_task_init_states(task_id)[args.episode])
        try:
            rows, trajectory = replay_task(args, task_id, env)
        finally:
            env.close()
        report_path = locate_report(report_roots, task_id, args.episode)
        if report_path is None:
            raise FileNotFoundError(f"missing baseline report for task {task_id}")
        baseline = json.loads(report_path.read_text())
        candidates, summary = event_candidates(rows)
        selected, status = choose_failure(candidates, summary, bool(baseline["success"]))
        enriched = args.output_dir / f"task{task_id}-episode{args.episode}-teacher-state.jsonl"
        with enriched.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        record = {
            "task_id": task_id,
            "episode": args.episode,
            "language": task.language,
            "baseline_success": bool(baseline["success"]),
            "status": status,
            "selected_failure": selected,
            "failure_candidates": candidates,
            "stage_summary": summary,
            "source_trajectory": str(trajectory.resolve()),
            "teacher_state_trace": str(enriched.resolve()),
            "baseline_report": str(report_path.resolve()),
            "baseline_video": baseline.get("video"),
            "nearest_snapshot_stride": 8,
        }
        records.append(record)
        print(json.dumps({
            "task_id": task_id, "success": baseline["success"], "status": status,
            "selected": selected, "candidate_count": len(candidates),
        }), flush=True)

    payload = {
        "kind": "mug_pudding_counterfactual_failure_manifest_v1",
        "schema_version": 1,
        "task_suite": "libero_10",
        "task_range": [args.task_start, args.task_end],
        "episode": args.episode,
        "label_semantics": {
            "privileged_state": "offline teacher/audit only; forbidden as online model input",
            "selected_failure": "earliest causal event selected for recovery fork",
        },
        "records": records,
        "summary": {
            "tasks": len(records),
            "baseline_successes": sum(row["baseline_success"] for row in records),
            "auto_high_confidence": sum(row["status"] == "auto_high_confidence" for row in records),
            "needs_review": sum(row["status"].startswith("needs_review") for row in records),
        },
    }
    output = args.output_dir / "failure-manifest.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output.resolve()), **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
