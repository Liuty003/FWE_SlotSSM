#!/usr/bin/env python3
"""Run a checkpointed privileged Cartesian controller on mug+pudding tasks.

This is deliberately an overfit benchmark ceiling.  It reads simulator object
positions but changes the world only through the standard 7-D robot action.
Every rollout emits a report and MP4, so it cannot be confused with a learned
RGB policy result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", type=Path, default=Path("/data/lty/code/LIBERO-plus"))
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    sys.path.insert(0, str(args.libero_root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "mug_pudding_privileged_cartesian_v1":
        raise ValueError(f"unsupported checkpoint: {checkpoint.get('format')}")
    params = checkpoint["controller"]

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.transform_utils as transform_utils
    from robosuite.utils.control_utils import orientation_error

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(args.task_id)
    states = suite.get_task_init_states(args.task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    env.seed(0)
    description = task.language
    target_orientation = transform_utils.quat2mat(
        np.asarray(params["target_eef_quaternion_xyzw"], np.float32)
    )
    frames: list[np.ndarray] = []
    actions: list[list[float]] = []
    phase_steps: dict[str, int] = {}
    diagnostics: dict[str, dict] = {}
    obs = None

    def step(action: np.ndarray, phase: str) -> bool:
        nonlocal obs
        frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        actions.append(action.tolist())
        phase_steps[phase] = phase_steps.get(phase, 0) + 1
        obs, _, done, _ = env.step(action.tolist())
        return bool(done)

    def hold(n: int, gripper: float, phase: str) -> bool:
        for _ in range(n):
            if step(np.asarray([0, 0, 0, 0, 0, 0, gripper], np.float32), phase):
                return True
        return False

    def move_to(target: np.ndarray, gripper: float, phase: str) -> bool:
        gain = float(params["position_gain"])
        max_action = float(params["max_translation_action"])
        tolerance = float(params["position_tolerance_m"])
        orientation_tolerance = float(params["orientation_tolerance_rad"])
        for _ in range(int(params["max_move_steps"])):
            error = target - np.asarray(obs["robot0_eef_pos"], np.float32)
            current_orientation = transform_utils.quat2mat(
                np.asarray(obs["robot0_eef_quat"], np.float32)
            )
            rotation_error = orientation_error(target_orientation, current_orientation)
            if (
                float(np.linalg.norm(error)) <= tolerance
                and float(np.linalg.norm(rotation_error)) <= orientation_tolerance
            ):
                return False
            xyz = np.clip(error * gain, -max_action, max_action)
            rotation = np.clip(
                rotation_error * float(params["orientation_gain"]),
                -float(params["max_rotation_action"]),
                float(params["max_rotation_action"]),
            )
            action = np.asarray([*xyz, *rotation, gripper], np.float32)
            if step(action, phase):
                return True
        return False

    def pick_and_place(
        object_key: str,
        grasp_offset: np.ndarray,
        destination: np.ndarray,
        phase_prefix: str,
    ) -> bool:
        object_position = np.asarray(obs[object_key], np.float32)
        diagnostics[phase_prefix] = {"object_initial": object_position.tolist()}
        grasp = object_position + grasp_offset
        approach = grasp.copy()
        approach[2] += float(params["approach_height_m"])
        if move_to(approach, -1.0, phase_prefix + "_approach"): return True
        if move_to(grasp, -1.0, phase_prefix + "_descend"): return True
        if hold(int(params["close_steps"]), 1.0, phase_prefix + "_close"): return True
        diagnostics[phase_prefix]["object_after_close"] = np.asarray(
            obs[object_key], np.float32
        ).tolist()
        lift = np.asarray(obs["robot0_eef_pos"], np.float32)
        lift[2] += float(params["lift_height_m"])
        if move_to(lift, 1.0, phase_prefix + "_lift"): return True
        diagnostics[phase_prefix]["object_after_lift"] = np.asarray(
            obs[object_key], np.float32
        ).tolist()
        destination_grasp = destination + grasp_offset
        destination_approach = destination_grasp.copy()
        destination_approach[2] += float(params["approach_height_m"])
        if move_to(destination_approach, 1.0, phase_prefix + "_transport"): return True
        if move_to(destination_grasp, 1.0, phase_prefix + "_place"): return True
        if hold(int(params["open_steps"]), -1.0, phase_prefix + "_open"): return True
        retreat = np.asarray(obs["robot0_eef_pos"], np.float32)
        retreat[2] += float(params["retreat_height_m"])
        return move_to(retreat, -1.0, phase_prefix + "_retreat")

    success = False
    try:
        env.reset()
        obs = env.set_init_state(states[args.episode_id])
        hold(int(params["wait_steps"]), -1.0, "wait")
        mug_destination = np.asarray(obs["plate_1_pos"], np.float32).copy()
        mug_destination[2] += float(params["mug_center_above_plate_m"])
        pick_and_place(
            "porcelain_mug_1_pos",
            np.asarray(params["mug_grasp_offset_m"], np.float32),
            mug_destination,
            "mug",
        )
        mug_on_plate = bool(env.env._eval_predicate(["on", "porcelain_mug_1", "plate_1"]))
        plate = np.asarray(obs["plate_1_pos"], np.float32)
        pudding_destination = np.asarray(obs["chocolate_pudding_1_pos"], np.float32).copy()
        pudding_destination[:2] = plate[:2] + np.asarray(params["pudding_right_offset_xy_m"], np.float32)
        pick_and_place(
            "chocolate_pudding_1_pos",
            np.asarray(params["pudding_grasp_offset_m"], np.float32),
            pudding_destination,
            "pudding",
        )
        hold(int(params["settle_steps"]), -1.0, "settle")
        success = bool(env.check_success())
        pudding_in_region = bool(env.env._eval_predicate(
            ["on", "chocolate_pudding_1", "living_room_table_plate_right_region"]
        ))
    finally:
        env.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"libero_10-task{args.task_id}-episode{args.episode_id}"
    video = args.output_dir / f"{stem}.mp4"
    imageio.mimsave(video, frames, fps=args.fps)
    report = {
        "policy": "mug-pudding-privileged-cartesian-checkpoint",
        "scope_warning": "privileged task-specific overfit controller; not learned RGB generalization",
        "checkpoint": str(args.checkpoint.resolve()),
        "task_suite": "libero_10",
        "task_id": args.task_id,
        "episode_id": args.episode_id,
        "task_description": description,
        "success": success,
        "mug_on_plate": mug_on_plate,
        "pudding_in_right_region": pudding_in_region,
        "num_frames": len(frames),
        "phase_steps": phase_steps,
        "diagnostics": diagnostics,
        "actions": actions,
        "video": str(video.resolve()),
    }
    (args.output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "actions"}, indent=2))


if __name__ == "__main__":
    main()
