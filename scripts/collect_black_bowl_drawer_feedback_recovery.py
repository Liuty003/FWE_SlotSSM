#!/usr/bin/env python3
"""Privileged feedback teacher for black bowl -> bottom drawer -> close."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


BOWL_NAME = "akita_black_bowl_1"
BOWL_KEY = f"{BOWL_NAME}_pos"
BOWL_GOAL = ["in", BOWL_NAME, "white_cabinet_1_bottom_region"]
DRAWER_GOAL = ["close", "white_cabinet_1_bottom_region"]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fork-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, default=Path("/data/lty/code/LIBERO-plus"))
    parser.add_argument("--drawer-demonstration", type=Path, default=Path(
        "/data2/lty/naive-e-slotssm-runs/oft-plus-robot-global-v1/forks/"
        "task409-episode0/trajectory.jsonl"))
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--move-linear-limit", type=float, default=0.5,
                        help="Per-axis limit for ordinary grasp/transfer motion; lower values create smoother teachers.")
    parser.add_argument("--move-rotation-limit", type=float, default=0.5)
    parser.add_argument("--move-position-gain", type=float, default=5.0)
    parser.add_argument("--move-rotation-gain", type=float, default=2.5)
    parser.add_argument("--move-profile", choices=("clip", "tanh"), default="clip")
    parser.add_argument("--max-linear-slew", type=float,
                        help="Maximum per-step change of each translation command axis.")
    parser.add_argument("--max-rotation-slew", type=float,
                        help="Maximum per-step change of each rotation command axis.")
    parser.add_argument("--stable-confirmation-steps", type=int, default=1,
                        help="Consecutive settled controller steps required before changing phase.")
    parser.add_argument("--minimum-jerk-reference-steps", type=int, default=0,
                        help="Ramp each translation waypoint with a quintic minimum-jerk reference.")
    return parser.parse_args()


def main():
    args = arguments()
    if min(args.move_linear_limit, args.move_rotation_limit,
           args.move_position_gain, args.move_rotation_gain) <= 0:
        raise ValueError("motion-profile limits and gains must be positive")
    if args.stable_confirmation_steps < 1 or args.minimum_jerk_reference_steps < 0 or any(
            value is not None and value <= 0
            for value in (args.max_linear_slew, args.max_rotation_slew)):
        raise ValueError("stable confirmation must be positive and slew limits, when set, positive")
    fork = json.loads((args.fork_dir / "manifest.json").read_text())
    if fork.get("kind") != "black_bowl_bottom_drawer_exact_failure_fork_v1":
        raise ValueError("expected black-bowl exact failure fork")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(args.libero_root), str(root)]
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv
    from robosuite.utils.control_utils import orientation_error
    import robosuite.utils.transform_utils as T
    from recovery.data.branch_trajectory import BranchTrajectoryWriter
    from recovery.teachers.sim_contact import extract_target_gripper_contact
    from scripts.collect_v4_oft_rgbd_rollouts import calibration

    suite = benchmark.get_benchmark_dict()[fork["task_suite"]]()
    task = suite.get_task(int(fork["task_id"]))
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = SegmentationRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                camera_depths=True, camera_segmentations="instance")
    env.seed(0); env.reset()
    env.set_init_state(suite.get_task_init_states(int(fork["task_id"]))[int(fork["episode_id"])])
    obs = env.regenerate_obs_from_state(np.load(args.fork_dir / fork["state_file"]))
    controller = env.env.robots[0].controller
    controller.update(force=True); controller.reset_goal()
    env.env.timestep = 0; env.env.done = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    states = args.output_dir / "states"; states.mkdir(exist_ok=True)
    writer = BranchTrajectoryWriter(
        args.output_dir.parent, task_suite=fork["task_suite"], task_id=int(fork["task_id"]),
        source_episode_id=int(fork["episode_id"]), failure_timestep=int(fork["failure_timestep"]),
        candidate_name=args.output_dir.name, candidate_index=0,
        source_rollout=str(args.fork_dir.resolve()), failure_type=fork["failure_type"],
        target_name=BOWL_NAME, target_instance_id=env.instance_to_id.get(BOWL_NAME),
        source_nominal_action=None)
    if writer.directory.resolve() != args.output_dir.resolve():
        raise ValueError("output-dir basename mismatch")
    calibration_data = calibration(env, "agentview", 256, 256, np)
    frames, phases = [], []
    drawer_joint_history = []
    previous = None
    step = 0

    def execute(action, phase):
        nonlocal obs, previous, step
        action = np.asarray(action, np.float32)
        pre = obs
        np.save(states / f"step-{step:04d}.npy", env.sim.get_state().flatten())
        frames.append(np.ascontiguousarray(pre["agentview_image"][::-1, ::-1]))
        cb, ab = extract_target_gripper_contact(env.env.sim, target_name=BOWL_NAME)
        pb = np.asarray(pre[BOWL_KEY], np.float32)
        obs, _, done, _ = env.step(action.tolist())
        ca, aa = extract_target_gripper_contact(env.env.sim, target_name=BOWL_NAME)
        writer.record(step=step, observation=pre, recovery_action=action,
                      previous_recovery_action=previous, teacher_contact_before=cb,
                      teacher_attached_before=ab, teacher_target_position_before=pb,
                      teacher_contact_after=ca, teacher_attached_after=aa,
                      teacher_target_position_after=np.asarray(obs[BOWL_KEY], np.float32),
                      task_success_after=bool(env.check_success()), done_after=bool(done))
        phases.append(phase); previous = action; step += 1
        return bool(done)

    def move_to(position, quaternion, gripper, phase, max_steps=180, allow_contact_stall=False):
        desired = T.quat2mat(np.asarray(quaternion, np.float32))
        target_position = np.asarray(position, np.float32)
        start_position = np.asarray(obs["robot0_eef_pos"], np.float32).copy()
        stable = 0
        for move_step in range(max_steps):
            current_position = np.asarray(obs["robot0_eef_pos"], np.float32)
            delta = target_position - current_position
            rotation = orientation_error(desired, T.quat2mat(np.asarray(obs["robot0_eef_quat"], np.float32)))
            reference_complete = (args.minimum_jerk_reference_steps == 0 or
                                  move_step >= args.minimum_jerk_reference_steps)
            settled = (reference_complete and np.linalg.norm(delta) < 0.012 and
                       np.linalg.norm(rotation) < 0.06)
            stable = stable + 1 if settled else 0
            if settled:
                translation_command = np.zeros(3, np.float32)
                rotation_command = np.zeros(3, np.float32)
            elif args.move_profile == "tanh":
                command_delta = delta
                if not reference_complete:
                    u = float(move_step + 1) / args.minimum_jerk_reference_steps
                    blend = 10 * u**3 - 15 * u**4 + 6 * u**5
                    reference = start_position + blend * (target_position - start_position)
                    command_delta = reference - current_position
                translation_command = args.move_linear_limit * np.tanh(
                    command_delta * args.move_position_gain / args.move_linear_limit)
                rotation_command = args.move_rotation_limit * np.tanh(
                    rotation * args.move_rotation_gain / args.move_rotation_limit)
            else:
                translation_command = np.clip(delta * args.move_position_gain,
                                              -args.move_linear_limit, args.move_linear_limit)
                rotation_command = np.clip(rotation * args.move_rotation_gain,
                                           -args.move_rotation_limit, args.move_rotation_limit)
            prior = np.zeros(6, np.float32) if previous is None else np.asarray(previous[:6], np.float32)
            if args.max_linear_slew is not None:
                translation_command = prior[:3] + np.clip(
                    translation_command - prior[:3], -args.max_linear_slew, args.max_linear_slew)
            if args.max_rotation_slew is not None:
                rotation_command = prior[3:] + np.clip(
                    rotation_command - prior[3:], -args.max_rotation_slew, args.max_rotation_slew)
            action = np.r_[translation_command, rotation_command, gripper]
            if execute(action, phase):
                return
            if (stable >= args.stable_confirmation_steps and
                    np.linalg.norm(translation_command) < 0.02 and
                    np.linalg.norm(rotation_command) < 0.03):
                return
        if allow_contact_stall:
            return
        raise RuntimeError(f"teacher failed to converge in {phase}")

    grasp_quat = [0.7680383, -0.6363008, 0.0477351, -0.0544048]
    # Task409 is a verified successful rollout from the same fixed-view task
    # family.  At step 200 the gripper first reaches the moving drawer face;
    # its pose relative to g37 is invariant to the randomized cabinet offset.
    demonstration = [json.loads(line) for line in args.drawer_demonstration.read_text().splitlines()]
    demo_by_step = {int(row["policy_step"]): row for row in demonstration}
    demo_anchor = demo_by_step[200]
    drawer_quat = demo_anchor["proprio"]["robot0_eef_quat"]
    try:
        bowl_in = bool(env.env._eval_predicate(BOWL_GOAL))
        if not bowl_in:
            # Always open first: wrong-object forks may still hold the bottle.
            for _ in range(10): execute([0, 0, 0.25, 0, 0, 0, -1], "release_distractor")
            bowl = np.asarray(obs[BOWL_KEY], np.float32)
            grasp = bowl + np.asarray([-0.040, -0.012, 0.040], np.float32)
            approach = grasp + np.asarray([0, 0, 0.11], np.float32)
            move_to(approach, grasp_quat, -1, "approach_bowl")
            move_to(grasp, grasp_quat, -1, "descend_bowl")
            for _ in range(14): execute([0, 0, 0, 0, 0, 0, 1], "close_bowl")
            lift = np.asarray(obs["robot0_eef_pos"], np.float32) + [0, 0, 0.14]
            move_to(lift, grasp_quat, 1, "lift_bowl")
            region_id = env.env.sim.model.site_name2id("white_cabinet_1_bottom_region")
            region = np.asarray(env.env.sim.data.site_xpos[region_id], np.float32)
            place_eef = region + np.asarray([-0.035, -0.045, 0.040], np.float32)
            place_high = place_eef.copy(); place_high[2] += 0.11
            move_to(place_high, grasp_quat, 1, "transfer_bowl")
            move_to(place_eef, grasp_quat, 1, "lower_bowl")
            for _ in range(14): execute([0, 0, 0, 0, 0, 0, -1], "release_bowl")
            retreat = place_eef + np.asarray([0, -0.10, 0.10], np.float32)
            move_to(retreat, grasp_quat, -1, "retreat_from_bowl")
        else:
            for _ in range(14): execute([0, 0, 0.15, 0, 0, 0, -1], "release_and_clear_bowl")

        # Align to the demonstrated contact pose relative to the moving drawer
        # face.  The previous center-line heuristic contacted fixed cabinetry.
        handle_id = env.env.sim.model.geom_name2id("white_cabinet_1_g37")
        handle = np.asarray(env.env.sim.data.geom_xpos[handle_id], np.float32)
        contact_offset = np.asarray([0.107, -0.146, 0.018], np.float32)
        push_start = handle + contact_offset
        approach = push_start + np.asarray([0.0, -0.06, 0.07], np.float32)
        move_to(approach, drawer_quat, -1, "approach_drawer")
        move_to(push_start, drawer_quat, -1, "contact_drawer", max_steps=100)
        # The demonstrated stable-contact pose is eef-handle ~=
        # (+10.5cm, -11.9cm, -2.0cm). Keep its X/Z/orientation closed-loop,
        # while applying a persistent +Y command to generate contact force.
        stable_offset = np.asarray([0.105, -0.119, -0.020], np.float32)
        stable_quat = demo_by_step[215]["proprio"]["robot0_eef_quat"]
        stable_orientation = T.quat2mat(np.asarray(stable_quat, np.float32))
        drawer_joint = "white_cabinet_1_bottom_level"
        last_progress_q = float(env.env.sim.data.get_joint_qpos(drawer_joint))
        stalled = 0
        for push_step in range(260):
            q = float(env.env.sim.data.get_joint_qpos(drawer_joint))
            drawer_joint_history.append({"transition": step, "qpos": q})
            if bool(env.env._eval_predicate(DRAWER_GOAL)):
                break
            # q grows from roughly -0.146 (open) toward 0 (closed).
            if q > last_progress_q + 0.0015:
                last_progress_q = q
                stalled = 0
            else:
                stalled += 1
            handle = np.asarray(env.env.sim.data.geom_xpos[handle_id], np.float32)
            current = np.asarray(obs["robot0_eef_pos"], np.float32)
            desired = handle + stable_offset
            rotation = orientation_error(
                stable_orientation,
                T.quat2mat(np.asarray(obs["robot0_eef_quat"], np.float32)))
            action = np.r_[
                np.clip((desired[0] - current[0]) * 6.0, -0.45, 0.45),
                0.72,
                np.clip((desired[2] - current[2]) * 6.0, -0.45, 0.45),
                np.clip(rotation * 3.0, -0.5, 0.5), -1]
            execute(action, "feedback_push_drawer_closed")
            # Back off and reacquire the moving face when contact has stopped
            # producing joint motion, then resume force control.
            if stalled >= 28:
                handle = np.asarray(env.env.sim.data.geom_xpos[handle_id], np.float32)
                reacquire = handle + stable_offset + np.asarray([0, -0.055, 0.045], np.float32)
                move_to(reacquire, stable_quat, -1, "reacquire_drawer")
                move_to(handle + stable_offset, stable_quat, -1, "recontact_drawer",
                        max_steps=70, allow_contact_stall=True)
                last_progress_q = float(env.env.sim.data.get_joint_qpos(drawer_joint))
                stalled = 0
        success = bool(env.env._eval_predicate(BOWL_GOAL) and env.env._eval_predicate(DRAWER_GOAL))
        np.save(states / f"step-{step:04d}.npy", env.sim.get_state().flatten())
        frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        video = args.output_dir / "rollout.mp4"
        imageio.mimsave(video, frames, fps=args.fps)
        metadata = {"kind": "black_bowl_bottom_drawer_feedback_teacher_v1",
                    "schema": "v4_rgbd_rollout_v2", "success": success,
                    "task_suite": fork["task_suite"], "task_id": fork["task_id"],
                    "episode_id": fork["episode_id"], "failure_timestep": fork["failure_timestep"],
                    "failure_type": fork["failure_type"], "target_role": fork["target_role"],
                    "task_language": task.language, "transitions": step,
                    "phase_by_step": phases, "geometry_view": "agentview",
                    "agentview_calibration": calibration_data,
                    "source_fork": str(args.fork_dir.resolve()), "rollout": str(video.resolve()),
                    "drawer_demonstration": str(args.drawer_demonstration.resolve()),
                    "drawer_demonstration_steps": [200, 229],
                    "drawer_joint_history": drawer_joint_history,
                    "motion_profile": {
                        "move_linear_limit": args.move_linear_limit,
                        "move_rotation_limit": args.move_rotation_limit,
                        "move_position_gain": args.move_position_gain,
                        "move_rotation_gain": args.move_rotation_gain,
                        "move_profile": args.move_profile,
                        "max_linear_slew": args.max_linear_slew,
                        "max_rotation_slew": args.max_rotation_slew,
                        "stable_confirmation_steps": args.stable_confirmation_steps,
                        "minimum_jerk_reference_steps": args.minimum_jerk_reference_steps,
                    },
                    "teacher_only_generation": True,
                    "final_bowl_in_bottom_drawer": bool(env.env._eval_predicate(BOWL_GOAL)),
                    "final_bottom_drawer_closed": bool(env.env._eval_predicate(DRAWER_GOAL))}
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2))
        if not success: raise RuntimeError("teacher did not complete both task predicates")
    finally:
        env.close()


if __name__ == "__main__":
    main()
