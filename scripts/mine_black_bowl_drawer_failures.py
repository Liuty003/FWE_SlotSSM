#!/usr/bin/env python3
"""Mine causal failures for black-bowl -> bottom-drawer -> close tasks.

The policy trace is replayed from its exact MuJoCo snapshot.  Privileged
predicates and object state are written only as offline teacher/audit labels;
they are never intended as online recovery inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


BOWL = "akita_black_bowl_1_pos"
DISTRACTOR = "wine_bottle_1_pos"
BOWL_GOAL = ["in", "akita_black_bowl_1", "white_cabinet_1_bottom_region"]
DRAWER_GOAL = ["close", "white_cabinet_1_bottom_region"]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, default=Path(
        "/data2/lty/naive-e-slotssm-runs/oft-plus-robot-global-v1/forks"))
    parser.add_argument("--task-start", type=int, default=409)
    parser.add_argument("--task-end", type=int, default=452)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-report-root", type=Path, action="append", default=[])
    parser.add_argument("--libero-root", type=Path, default=Path("/data/lty/code/LIBERO-plus"))
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(int(row["timestep"]) != index for index, row in enumerate(rows)):
        raise ValueError(f"non-contiguous trajectory: {path}")
    return rows


def locate_report(roots: list[Path], task_id: int, episode: int) -> Path:
    name = f"libero_10-task{task_id}-episode{episode}.json"
    matches = [path for root in roots for path in root.rglob(name)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one baseline report for task{task_id}, found {matches}")
    return matches[0]


def replay(args: argparse.Namespace, task_id: int, env: Any) -> tuple[list[dict[str, Any]], Path]:
    source = args.rollout_root / f"task{task_id}-episode{args.episode}"
    trajectory = source / "trajectory.jsonl"
    snapshot = source / "step0000" / "mujoco_state.npz"
    actions = read_jsonl(trajectory)
    obs = env.regenerate_obs_from_state(np.load(snapshot)["mujoco_state"])
    controller = env.env.robots[0].controller
    controller.update(force=True)
    controller.reset_goal()
    output = []
    for index, source_row in enumerate(actions):
        before = obs
        bowl_in_before = bool(env.env._eval_predicate(BOWL_GOAL))
        drawer_closed_before = bool(env.env._eval_predicate(DRAWER_GOAL))
        obs, _, done, _ = env.step(source_row["executed_action"])
        output.append({
            "task_id": task_id, "episode_id": args.episode, "timestep": index,
            "executed_action": source_row["executed_action"],
            "robot_state_before": {
                key: np.asarray(before[key]).tolist()
                for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
            },
            "privileged_object_state_before": {
                key: np.asarray(before[key]).tolist() for key in (BOWL, DISTRACTOR)
            },
            "bowl_in_bottom_drawer_before": bowl_in_before,
            "bowl_in_bottom_drawer_after": bool(env.env._eval_predicate(BOWL_GOAL)),
            "bottom_drawer_closed_before": drawer_closed_before,
            "bottom_drawer_closed_after": bool(env.env._eval_predicate(DRAWER_GOAL)),
            "done_after": bool(done),
        })
    return output, trajectory


def stable_onset(values: list[bool], confirmation: int = 5) -> int | None:
    streak = 0
    for index, value in enumerate(values):
        streak = streak + 1 if value else 0
        if streak >= confirmation:
            return index - confirmation + 1
    return None


def failure_candidates(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bowl_onset = stable_onset([bool(row["bowl_in_bottom_drawer_after"]) for row in rows])
    candidates: list[dict[str, Any]] = []
    grasp_attempts: list[int] = []

    def xyz(index: int, key: str) -> np.ndarray:
        return np.asarray(rows[index]["privileged_object_state_before"][key], np.float32)

    def eef(index: int) -> np.ndarray:
        return np.asarray(rows[index]["robot_state_before"]["robot0_eef_pos"], np.float32)

    for index, row in enumerate(rows):
        action = np.asarray(row["executed_action"], np.float32)
        previous_grip = -1.0 if index == 0 else float(rows[index - 1]["executed_action"][6])
        closing_edge = float(action[6]) > 0.5 and previous_grip < 0.0
        bowl_distance = float(np.linalg.norm(xyz(index, BOWL) - eef(index)))
        bottle_distance = float(np.linalg.norm(xyz(index, DISTRACTOR) - eef(index)))
        stage = "close_drawer" if bowl_onset is not None and index >= bowl_onset else "place_bowl"
        if stage == "place_bowl" and closing_edge:
            if bowl_distance > 0.20:
                candidates.append({"timestep": index, "failure_type": "premature_empty_close",
                                   "target_role": "black_bowl", "confidence": "high",
                                   "evidence": {"bowl_eef_distance_m": bowl_distance}})
            if bottle_distance < 0.20 and bowl_distance - bottle_distance > 0.05:
                candidates.append({"timestep": index, "failure_type": "wrong_object_acquisition",
                                   "target_role": "black_bowl", "confidence": "high",
                                   "evidence": {"bowl_eef_distance_m": bowl_distance,
                                                "wine_bottle_eef_distance_m": bottle_distance}})
            if bowl_distance < 0.14:
                grasp_attempts.append(index)
        for attempt in list(grasp_attempts):
            if 18 <= index - attempt <= 32:
                lift = float(xyz(index, BOWL)[2] - xyz(attempt, BOWL)[2])
                relative_drift = float(np.linalg.norm(
                    (xyz(index, BOWL) - eef(index)) - (xyz(attempt, BOWL) - eef(attempt))))
                if lift < 0.012 and relative_drift > 0.025:
                    candidates.append({"timestep": index, "failure_type": "failed_bowl_acquisition",
                                       "target_role": "black_bowl", "confidence": "high",
                                       "evidence": {"attempt_timestep": attempt, "bowl_lift_m": lift,
                                                    "object_eef_relative_drift_m": relative_drift}})
                grasp_attempts.remove(attempt)
        if row["bowl_in_bottom_drawer_before"] and not row["bowl_in_bottom_drawer_after"]:
            candidates.append({"timestep": index, "failure_type": "bowl_displaced_from_drawer",
                               "target_role": "black_bowl", "confidence": "high", "evidence": {}})

    if bowl_onset is not None and not rows[-1]["bottom_drawer_closed_after"]:
        # The first stable placement is the causal handoff point for the
        # second subtask: release the bowl, acquire the handle, close drawer.
        candidates.append({"timestep": bowl_onset, "failure_type": "drawer_close_stage_incomplete",
                           "target_role": "bottom_drawer", "confidence": "high",
                           "evidence": {"stable_bowl_placement_timestep": bowl_onset}})

    candidates.sort(key=lambda item: (item["timestep"], item["failure_type"]))
    merged: list[dict[str, Any]] = []
    for item in candidates:
        old = next((candidate for candidate in reversed(merged)
                    if candidate["failure_type"] == item["failure_type"]), None)
        if old is None or item["timestep"] - old["timestep"] > 12:
            merged.append(item)
    summary = {
        "bowl_stage_reached": bowl_onset is not None,
        "bowl_stage_start_timestep": bowl_onset,
        "final_bowl_in_bottom_drawer": bool(rows[-1]["bowl_in_bottom_drawer_after"]),
        "final_bottom_drawer_closed": bool(rows[-1]["bottom_drawer_closed_after"]),
        "task_success_from_predicates": bool(
            rows[-1]["bowl_in_bottom_drawer_after"] and rows[-1]["bottom_drawer_closed_after"]),
    }
    return merged, summary


def choose(candidates: list[dict[str, Any]], summary: dict[str, Any], success: bool):
    if success:
        return None, "normal_success"
    high = [item for item in candidates if item["confidence"] == "high"]
    if high:
        # Prefer the actual failed manipulation over an earlier incidental
        # empty close; for drawer-stage failures the explicit stage event wins.
        priority = {"wrong_object_acquisition": 0, "failed_bowl_acquisition": 1,
                    "bowl_displaced_from_drawer": 2, "drawer_close_stage_incomplete": 3,
                    "premature_empty_close": 4}
        return min(high, key=lambda item: (priority[item["failure_type"]], item["timestep"])), "auto_high_confidence"
    stage = "bowl" if not summary["bowl_stage_reached"] else "drawer_close"
    return None, f"needs_review_{stage}_stage_without_causal_event"


def main() -> None:
    args = arguments()
    roots = args.baseline_report_root or [
        Path("/data2/lty/naive-e-slotssm-runs/libero-plus/oft-plus-robot-levels1-4-v1"),
        Path("/data2/lty/naive-e-slotssm-runs/libero-plus/oft-plus-robot-level5-v1"),
    ]
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
        trace = args.output_dir / f"task{task_id}-episode{args.episode}-teacher-state.jsonl"
        trajectory = args.rollout_root / f"task{task_id}-episode{args.episode}" / "trajectory.jsonl"
        if trace.is_file():
            rows = read_jsonl(trace)
        else:
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
            env.seed(0); env.reset(); env.set_init_state(suite.get_task_init_states(task_id)[args.episode])
            try:
                rows, trajectory = replay(args, task_id, env)
            finally:
                env.close()
        baseline_path = locate_report(roots, task_id, args.episode)
        baseline = json.loads(baseline_path.read_text())
        candidates, summary = failure_candidates(rows)
        selected, status = choose(candidates, summary, bool(baseline["success"]))
        if not trace.is_file():
            with trace.open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
        records.append({"task_id": task_id, "episode": args.episode, "language": task.language,
                        "baseline_success": bool(baseline["success"]), "status": status,
                        "selected_failure": selected, "failure_candidates": candidates,
                        "stage_summary": summary, "source_trajectory": str(trajectory.resolve()),
                        "teacher_state_trace": str(trace.resolve()),
                        "baseline_report": str(baseline_path.resolve()),
                        "baseline_video": baseline.get("video"), "nearest_snapshot_stride": 8})
        print(json.dumps({"task_id": task_id, "success": bool(baseline["success"]),
                          "status": status, "selected": selected}), flush=True)
    payload = {"kind": "black_bowl_bottom_drawer_failure_manifest_v1", "schema_version": 1,
               "task_suite": "libero_10", "task_range": [args.task_start, args.task_end],
               "episode": args.episode,
               "label_semantics": {"privileged_state": "offline teacher/audit only",
                                   "selected_failure": "causal recovery handoff candidate"},
               "records": records,
               "summary": {"tasks": len(records),
                           "baseline_successes": sum(row["baseline_success"] for row in records),
                           "auto_high_confidence": sum(row["status"] == "auto_high_confidence" for row in records),
                           "needs_review": sum(row["status"].startswith("needs_review") for row in records)}}
    output = args.output_dir / "failure-manifest.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output.resolve()), **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
