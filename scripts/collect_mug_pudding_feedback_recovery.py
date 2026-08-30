#!/usr/bin/env python3
"""Collect a local feedback recovery teacher from an exact failure fork.

This is an offline privileged teacher only.  It stops after verified re-grasp
and lift; it deliberately does not complete the remaining task.  Each
pre-action MuJoCo state is saved for later counterfactual OFT handoff audits.
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
    parser.add_argument("--fork-dir", type=Path, required=True)
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, default=Path("/data/lty/code/LIBERO-plus"))
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    fork = json.loads((args.fork_dir / "manifest.json").read_text())
    if fork.get("kind") != "mug_pudding_exact_failure_fork_v1":
        raise ValueError("expected an exact mug+pudding failure fork")
    checkpoint = torch.load(args.controller_checkpoint, map_location="cpu", weights_only=False)
    params = checkpoint["controller"]
    target_role = fork["target_role"]
    target_key = "porcelain_mug_1_pos" if target_role == "mug" else "chocolate_pudding_1_pos"
    target_name = "porcelain_mug_1" if target_role == "mug" else "chocolate_pudding_1"
    grasp_offset = np.asarray(
        params["mug_grasp_offset_m"] if target_role == "mug" else params["pudding_grasp_offset_m"],
        np.float32,
    )

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(args.libero_root), str(root)]
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv
    import robosuite.utils.transform_utils as transform_utils
    from robosuite.utils.control_utils import orientation_error
    from recovery.data.branch_trajectory import BranchTrajectoryWriter
    from recovery.teachers.sim_contact import extract_target_gripper_contact
    from scripts.collect_v4_oft_rgbd_rollouts import calibration

    suite = benchmark.get_benchmark_dict()[fork["task_suite"]]()
    task = suite.get_task(int(fork["task_id"]))
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
        camera_depths=True, camera_segmentations="instance",
    )
    env.seed(0)
    env.reset()
    env.set_init_state(suite.get_task_init_states(int(fork["task_id"]))[int(fork["episode_id"])])
    state = np.load(args.fork_dir / fork["state_file"])
    obs = env.regenerate_obs_from_state(state)
    agent_calibration = calibration(env, "agentview", 256, 256, np)
    controller = env.env.robots[0].controller
    controller.update(force=True)
    controller.reset_goal()
    env.env.timestep = 0
    env.env.done = False

    args.output_dir.mkdir(parents=True, exist_ok=True)
    states_dir = args.output_dir / "states"
    states_dir.mkdir(exist_ok=True)
    writer = BranchTrajectoryWriter(
        args.output_dir.parent,
        task_suite=fork["task_suite"], task_id=int(fork["task_id"]),
        source_episode_id=int(fork["episode_id"]), failure_timestep=int(fork["failure_timestep"]),
        candidate_name=args.output_dir.name, candidate_index=0,
        source_rollout=str(args.fork_dir.resolve()), failure_type=fork["failure_type"],
        target_name=target_name, target_instance_id=env.instance_to_id.get(target_name),
        source_nominal_action=None,
    )
    # BranchTrajectoryWriter owns this same named directory below its parent.
    if writer.directory.resolve() != args.output_dir.resolve():
        raise ValueError("output-dir basename must identify the teacher candidate")

    target_orientation = transform_utils.quat2mat(
        np.asarray(params.get("target_eef_quaternion_xyzw", [0.9996, 0.0, -0.0283, 0.0]), np.float32)
    )
    frames: list[np.ndarray] = []
    phases: list[str] = []
    previous_action = None
    initial_target = np.asarray(obs[target_key], np.float32).copy()
    step_index = 0

    def execute(action: np.ndarray, phase: str) -> bool:
        nonlocal obs, previous_action, step_index
        pre = obs
        np.save(states_dir / f"step-{step_index:04d}.npy", env.sim.get_state().flatten())
        frames.append(np.ascontiguousarray(pre["agentview_image"][::-1, ::-1]))
        contact_before, attached_before = extract_target_gripper_contact(
            env.env.sim, target_name=target_name
        )
        before_position = np.asarray(pre[target_key], np.float32)
        obs, _, done, _ = env.step(action.tolist())
        contact_after, attached_after = extract_target_gripper_contact(
            env.env.sim, target_name=target_name
        )
        writer.record(
            step=step_index, observation=pre, recovery_action=action,
            previous_recovery_action=previous_action,
            teacher_contact_before=contact_before, teacher_attached_before=attached_before,
            teacher_target_position_before=before_position,
            teacher_contact_after=contact_after, teacher_attached_after=attached_after,
            teacher_target_position_after=np.asarray(obs[target_key], np.float32),
            task_success_after=bool(env.check_success()), done_after=bool(done),
        )
        phases.append(phase)
        previous_action = action.copy()
        step_index += 1
        return bool(done)

    def move_to(target: np.ndarray, gripper: float, phase: str) -> bool:
        for _ in range(int(params["max_move_steps"])):
            position_delta = target - np.asarray(obs["robot0_eef_pos"], np.float32)
            current = transform_utils.quat2mat(np.asarray(obs["robot0_eef_quat"], np.float32))
            rotation_delta = orientation_error(target_orientation, current)
            if (float(np.linalg.norm(position_delta)) <= float(params["position_tolerance_m"])
                    and float(np.linalg.norm(rotation_delta)) <= float(params.get("orientation_tolerance_rad", 0.02))):
                return False
            action = np.asarray([
                *np.clip(position_delta * float(params["position_gain"]),
                         -float(params["max_translation_action"]), float(params["max_translation_action"])),
                *np.clip(rotation_delta * float(params.get("orientation_gain", 3.0)),
                         -float(params.get("max_rotation_action", 0.5)),
                         float(params.get("max_rotation_action", 0.5))),
                gripper,
            ], np.float32)
            if execute(action, phase):
                return True
        raise RuntimeError(f"feedback teacher failed to converge during {phase}")

    try:
        grasp = np.asarray(obs[target_key], np.float32) + grasp_offset
        approach = grasp.copy()
        approach[2] += float(params["approach_height_m"])
        move_to(approach, -1.0, "approach")
        move_to(grasp, -1.0, "descend")
        for _ in range(int(params["close_steps"])):
            execute(np.asarray([0, 0, 0, 0, 0, 0, 1], np.float32), "close")
        lift = np.asarray(obs["robot0_eef_pos"], np.float32).copy()
        lift[2] += float(params["lift_height_m"])
        move_to(lift, 1.0, "verify_lift")
        final_contact, final_attached = extract_target_gripper_contact(
            env.env.sim, target_name=target_name
        )
        final_target = np.asarray(obs[target_key], np.float32)
        target_lift = float(final_target[2] - initial_target[2])
        local_success = bool(final_attached and target_lift >= 0.03)
        np.save(states_dir / f"step-{step_index:04d}.npy", env.sim.get_state().flatten())
        frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        video = args.output_dir / "rollout.mp4"
        imageio.mimsave(video, frames, fps=args.fps)
        metadata = {
            "kind": "mug_pudding_local_feedback_recovery_teacher_v1",
            "schema": "v4_rgbd_rollout_v2",
            "success": local_success,
            "local_recovery_predicate": "target attached after >=3cm verified lift",
            "task_suite": fork["task_suite"], "task_id": fork["task_id"],
            "episode_id": fork["episode_id"], "failure_timestep": fork["failure_timestep"],
            "fork_timestep": 0, "task_language": task.language,
            "transitions": step_index, "geometry_view": "agentview",
            "agentview_calibration": agent_calibration,
            "wait_steps": 10, "open_loop_steps": 8,
            "rollout_id": f"counterfactual:{fork['task_suite']}:{fork['task_id']}:{fork['episode_id']}:failure{fork['failure_timestep']}",
            "failure_type": fork["failure_type"], "target_role": target_role,
            "steps": step_index, "target_lift_m": target_lift,
            "attached_after": bool(final_attached), "contact_after": final_contact,
            "phase_by_step": phases, "source_fork": str(args.fork_dir.resolve()),
            "controller_checkpoint": str(args.controller_checkpoint.resolve()),
            "states_dir": str(states_dir.resolve()), "rollout": str(video.resolve()),
            "teacher_only_generation": True,
            "online_inputs_permitted": [
                "main_rgb", "wrist_rgb", "main_depth", "wrist_depth",
                "camera_calibration", "language", "proprio",
            ],
            "training_only_inputs": ["instance_segmentation", "mujoco_state"],
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2))
        if not local_success:
            raise RuntimeError("feedback teacher did not satisfy local recovery predicate")
    finally:
        env.close()


if __name__ == "__main__":
    main()
