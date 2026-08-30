#!/usr/bin/env python3
"""Privileged offline teacher for a relational object-to-container recovery."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fork-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--destination-site", required=True)
    parser.add_argument("--predicate-name", choices=("in", "on"), default="in")
    parser.add_argument("--failure-type", default="empty_grasp_or_failed_contact")
    parser.add_argument("--grasp-offset", type=float, nargs=3, default=(0.0, 0.0, 0.025))
    parser.add_argument("--place-offset", type=float, nargs=3, default=(-0.015, 0.005, 0.185))
    parser.add_argument("--grasp-quaternion", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--grasp-profile-file", type=Path)
    parser.add_argument("--grasp-profile-key")
    parser.add_argument("--libero-root", type=Path, default=Path("/data/lty/code/LIBERO-plus"))
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--linear-limit", type=float, default=0.35)
    parser.add_argument("--rotation-limit", type=float, default=0.35)
    parser.add_argument("--position-gain", type=float, default=4.0)
    parser.add_argument("--rotation-gain", type=float, default=2.0)
    parser.add_argument("--linear-slew", type=float, default=0.06)
    parser.add_argument("--rotation-slew", type=float, default=0.06)
    parser.add_argument("--stable-confirmation-steps", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.linear_limit, args.rotation_limit, args.position_gain, args.rotation_gain,
           args.linear_slew, args.rotation_slew, args.stable_confirmation_steps) <= 0:
        raise ValueError("controller limits, gains, slew, and stable confirmation must be positive")
    fork = json.loads((args.fork_dir / "manifest.json").read_text())
    if fork.get("kind") != "offline_recovery_fork_snapshot_v1":
        raise ValueError("expected an offline_recovery_fork_snapshot_v1")
    if fork.get("exact_archival_prefix_validated") is not True:
        raise ValueError("teacher requires an exact archival fork")
    state_path = args.fork_dir / fork["state_file"]
    with np.load(state_path, allow_pickle=False) as bundle:
        state = np.asarray(bundle["mujoco_state"], dtype=np.float64)

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
    task_id = int(fork["task_id"])
    episode_id = int(fork["episode_id"])
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
        camera_depths=True, camera_segmentations="instance",
    )
    env.seed(0)
    env.reset()
    env.set_init_state(suite.get_task_init_states(task_id)[episode_id])
    obs = env.regenerate_obs_from_state(state)
    controller = env.env.robots[0].controller
    controller.update(force=True)
    controller.reset_goal()
    env.env.timestep = 0
    env.env.done = False

    target_key = f"{args.target_name}_pos"
    if target_key not in obs:
        raise KeyError(f"target observation key missing: {target_key}")
    resolved_destination_site = args.destination_site
    try:
        destination_site_id = env.env.sim.model.site_name2id(resolved_destination_site)
    except Exception as direct_error:
        # Relational predicates may name a movable object (for example
        # ``On(mug, plate_1)``), while motion planning needs that object's
        # concrete MuJoCo site.  LIBERO exposes a canonical default site for
        # these entities; keep the predicate name unchanged for verification.
        default_site = f"{args.destination_site}_default_site"
        try:
            destination_site_id = env.env.sim.model.site_name2id(default_site)
        except Exception as default_error:
            raise KeyError(
                f"destination site missing: {args.destination_site} "
                f"(also tried {default_site})"
            ) from default_error
        resolved_destination_site = default_site

    args.output_dir.mkdir(parents=True, exist_ok=True)
    states = args.output_dir / "states"
    states.mkdir(exist_ok=True)
    source_steps = Path(fork["source_steps"])
    source_rows = [json.loads(line) for line in source_steps.read_text().splitlines() if line.strip()]
    fork_timestep = int(fork["fork_timestep"])
    source_action = source_rows[fork_timestep]["executed_action"]
    source_initial_target_position = np.asarray(
        source_rows[0]["privileged_audit"]["entities"][args.target_name]["position"],
        dtype=np.float32,
    )
    source_initial_target_z = float(source_initial_target_position[2])
    writer = BranchTrajectoryWriter(
        args.output_dir.parent,
        task_suite=fork["task_suite"], task_id=task_id, source_episode_id=episode_id,
        failure_timestep=fork_timestep, candidate_name=args.output_dir.name, candidate_index=0,
        source_rollout=str(source_steps.resolve()), failure_type=args.failure_type,
        target_name=args.target_name, target_instance_id=env.instance_to_id.get(args.target_name),
        source_nominal_action=source_action,
    )
    if writer.directory.resolve() != args.output_dir.resolve():
        raise ValueError("output-dir basename mismatch")

    frames: list[np.ndarray] = []
    phases: list[str] = []
    previous: np.ndarray | None = None
    step = 0
    finished = False

    def target_position() -> np.ndarray:
        return np.asarray(obs[target_key], dtype=np.float32)

    def execute(action, phase: str) -> bool:
        nonlocal obs, previous, step, finished
        if finished:
            return True
        action = np.asarray(action, dtype=np.float32)
        pre = obs
        before_position = np.asarray(pre[target_key], dtype=np.float32)
        np.save(states / f"step-{step:04d}.npy", env.sim.get_state().flatten())
        frames.append(np.ascontiguousarray(pre["agentview_image"][::-1, ::-1]))
        contact_before, attached_before = extract_target_gripper_contact(
            env.env.sim, target_name=args.target_name,
        )
        obs, _, done, _ = env.step(action.tolist())
        contact_after, attached_after = extract_target_gripper_contact(
            env.env.sim, target_name=args.target_name,
        )
        finished = bool(done)
        writer.record(
            step=step, observation=pre, recovery_action=action,
            previous_recovery_action=previous,
            teacher_contact_before=contact_before, teacher_attached_before=attached_before,
            teacher_target_position_before=before_position,
            teacher_contact_after=contact_after, teacher_attached_after=attached_after,
            teacher_target_position_after=np.asarray(obs[target_key], dtype=np.float32),
            task_success_after=bool(env.check_success()), done_after=finished,
        )
        phases.append(phase)
        previous = action
        step += 1
        return finished

    grasp_profile = None
    if args.grasp_profile_file is not None:
        if not args.grasp_profile_key:
            raise ValueError("--grasp-profile-key is required with --grasp-profile-file")
        grasp_profiles = json.loads(args.grasp_profile_file.read_text())
        grasp_profile = grasp_profiles["profiles"].get(args.grasp_profile_key)
        if grasp_profile is None:
            raise KeyError(f"grasp profile missing: {args.grasp_profile_key}")

    target_entity = env.env.objects_dict[args.target_name]
    target_body_id = env.env.sim.model.body_name2id(target_entity.root_body)
    target_rotation = np.asarray(
        env.env.sim.data.body_xmat[target_body_id], dtype=np.float32
    ).reshape(3, 3)
    desired_quaternion = np.asarray(args.grasp_quaternion, dtype=np.float32)
    if grasp_profile is not None:
        desired_quaternion = np.asarray(
            T.mat2quat(
                target_rotation @ T.quat2mat(np.asarray(
                    grasp_profile["relative_grasp_quaternion_xyzw"], dtype=np.float32
                ))
            ),
            dtype=np.float32,
        )
    enclosed_destination = args.destination_site.endswith("heating_region")
    handled_mug_grasp = enclosed_destination and "mug" in args.target_name.lower()
    handle_direction: np.ndarray | None = None
    if handled_mug_grasp and grasp_profile is None:
        # Successful OFT rollouts task646/649/650 all use the same diagonal
        # side grasp near the handle: EEF-minus-mug is approximately
        # [0.042, 0.000, 0.070] in mug-local coordinates.  Preserve their
        # object-relative EEF rotation instead of forcing a horizontal wrist,
        # which drives some robot initial states into a joint limit.
        handle_direction = target_rotation[:, 0]
        demonstration_relative_quaternion = np.asarray(
            [-0.908415, -0.068020, 0.411669, 0.026152], dtype=np.float32
        )
        desired_quaternion = np.asarray(
            T.mat2quat(
                target_rotation @ T.quat2mat(demonstration_relative_quaternion)
            ),
            dtype=np.float32,
        )

    def move_to(position, gripper: float, phase: str, max_steps: int = 180,
                stop_on_bilateral_contact: bool = False,
                stop_on_selected_predicate: bool = False) -> bool:
        desired_rotation = T.quat2mat(desired_quaternion)
        target = np.asarray(position, dtype=np.float32)
        stable = 0
        for _ in range(max_steps):
            if stop_on_selected_predicate and env.env._eval_predicate(
                [args.predicate_name, args.target_name, args.destination_site]
            ):
                return False
            if stop_on_bilateral_contact:
                contact, _ = extract_target_gripper_contact(
                    env.env.sim, target_name=args.target_name,
                )
                # Object centers are not grasp affordances.  For differently
                # shaped mugs, moka pots and boxes, bilateral finger contact is
                # a stronger, geometry-adaptive descent endpoint than forcing
                # the EEF to an often collision-infeasible center offset.
                if contact.get("bilateral_finger_contact"):
                    return False
            current = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
            delta = target - current
            rotation = orientation_error(
                desired_rotation, T.quat2mat(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
            )
            settled = np.linalg.norm(delta) < 0.010 and np.linalg.norm(rotation) < 0.05
            stable = stable + 1 if settled else 0
            translation = args.linear_limit * np.tanh(
                delta * args.position_gain / args.linear_limit
            )
            rotation_command = args.rotation_limit * np.tanh(
                rotation * args.rotation_gain / args.rotation_limit
            )
            prior = np.zeros(6, dtype=np.float32) if previous is None else previous[:6]
            translation = prior[:3] + np.clip(
                translation - prior[:3], -args.linear_slew, args.linear_slew
            )
            rotation_command = prior[3:] + np.clip(
                rotation_command - prior[3:], -args.rotation_slew, args.rotation_slew
            )
            if settled:
                translation[:] = 0
                rotation_command[:] = 0
            if execute(np.r_[translation, rotation_command, gripper], phase):
                return True
            if stable >= args.stable_confirmation_steps:
                return False
        raise RuntimeError(f"teacher failed to converge in {phase}")

    calibration_data = calibration(env, "agentview", 256, 256, np)
    initial_target_z = float(target_position()[2])
    initial_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    initial_contact, initial_attached = extract_target_gripper_contact(
        env.env.sim, target_name=args.target_name,
    )
    target_elevated_at_fork = initial_target_z >= source_initial_target_z + 0.01
    target_displaced_at_fork = (
        np.linalg.norm(target_position() - source_initial_target_position) >= 0.03
    )
    target_near_eef_at_fork = np.linalg.norm(target_position() - initial_eef) <= 0.11
    target_likely_held_at_fork = bool(
        float(source_action[6]) > 0
        and (target_elevated_at_fork or target_displaced_at_fork)
        and target_near_eef_at_fork
        and (
            initial_attached
            or initial_contact.get("bilateral_finger_contact", False)
        )
    )
    profiled_desired_quaternion = desired_quaternion.copy()
    if target_likely_held_at_fork or initial_contact.get("target_gripper_contact"):
        # When the nominal rollout reaches the fork with the object held (or
        # already in useful contact), preserve its wrist pose.  Rotating toward
        # a fresh demonstration grasp while carrying the object can knock it
        # loose before recovery has even started.
        desired_quaternion = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
    target_secured_from_fork = False
    try:
        if target_likely_held_at_fork:
            # Placement-failure forks deliberately start before the bad
            # release, while the nominal policy still holds the object.  Do
            # not destroy that useful state by opening and re-grasping.
            target_secured_from_fork = True
        elif initial_contact.get("target_gripper_contact"):
            # A failed-contact fork can already have one finger touching the
            # object.  First exploit that contact with a close-and-lift probe;
            # only fall back to a full re-grasp if the object does not follow.
            for _ in range(16):
                if execute([0, 0, 0, 0, 0, 0, 1], "close_from_fork_contact"):
                    break
            probe = np.asarray(obs["robot0_eef_pos"], dtype=np.float32) + [0.0, 0.0, 0.10]
            move_to(probe, 1, "probe_lift_from_fork_contact")
            target_secured_from_fork = float(target_position()[2]) >= initial_target_z + 0.03

        if not target_secured_from_fork:
            desired_quaternion = profiled_desired_quaternion
            # Clear a stale empty close and move above the current object pose.
            for _ in range(8):
                if execute([0, 0, 0.12, 0, 0, 0, -1], "open_and_clear"):
                    break
            target = target_position()
            if grasp_profile is not None:
                grasp = target + target_rotation @ np.asarray(
                    grasp_profile["grasp_offset"], dtype=np.float32
                )
                approach = grasp + np.asarray([0.0, 0.0, 0.13], dtype=np.float32)
                move_to(approach, -1, "approach_profiled_target")
            elif handled_mug_grasp:
                assert handle_direction is not None
                handle_offset = target_rotation @ np.asarray(
                    [0.042, 0.0, 0.070], dtype=np.float32
                )
                grasp = target + handle_offset
                approach = target + target_rotation @ np.asarray(
                    [0.100, 0.0, 0.120], dtype=np.float32
                )
                move_to(approach, -1, "approach_handle_diagonal")
            else:
                grasp = target + np.asarray(args.grasp_offset, dtype=np.float32)
                approach = grasp + np.asarray([0.0, 0.0, 0.13], dtype=np.float32)
                move_to(approach, -1, "approach_target")
            move_to(grasp, -1, "descend_target", stop_on_bilateral_contact=True)
            for _ in range(16):
                if execute([0, 0, 0, 0, 0, 0, 1], "close_target"):
                    break

        lift = np.asarray(obs["robot0_eef_pos"], dtype=np.float32) + [0.0, 0.0, 0.16]
        move_to(lift, 1, "lift_held_target" if target_secured_from_fork else "lift_target")
        lift_height = float(target_position()[2]) - initial_target_z
        if lift_height < 0.04 and not target_likely_held_at_fork:
            raise RuntimeError(f"teacher failed to lift target: {lift_height:.4f} m")

        destination = np.asarray(
            env.env.sim.data.site_xpos[destination_site_id], dtype=np.float32
        )
        if enclosed_destination:
            # A microwave is not an open-top receptacle.  Its region center is
            # inside the cavity and local -Y points through the open door.  A
            # vertical lower would collide with the roof, so enter and leave
            # horizontally in the destination site's own frame.  Keeping this
            # geometry relative to site_xmat makes it invariant to fixture yaw.
            destination_rotation = np.asarray(
                env.env.sim.data.site_xmat[destination_site_id], dtype=np.float32
            ).reshape(3, 3)
            if handled_mug_grasp:
                # Successful rollouts keep the Franka hand outside the cavity:
                # entry is about 18 cm in front of the region center and the
                # release EEF remains about 10 cm in front.  Sending the EEF to
                # the site center itself collides with the microwave roof.
                entry = destination + destination_rotation @ np.asarray(
                    [-0.015, -0.180, 0.030], dtype=np.float32
                )
                place = destination + destination_rotation @ np.asarray(
                    [-0.015, -0.115, 0.025], dtype=np.float32
                )
            else:
                place = destination + destination_rotation @ np.asarray(
                    [0.0, 0.0, 0.01], dtype=np.float32
                )
                entry = place + destination_rotation @ np.asarray(
                    [0.0, -0.18, 0.0], dtype=np.float32
                )
            move_to(entry, 1, "transfer_to_entry")
            move_to(
                place, 1, "insert_target",
                stop_on_selected_predicate=handled_mug_grasp,
            )
        else:
            place = destination + np.asarray(args.place_offset, dtype=np.float32)
            occupied_positions = []
            fork_goals = source_rows[fork_timestep]["privileged_audit"]["goal_predicates"]
            for goal in fork_goals:
                predicate = goal.get("predicate", [])
                if (
                    goal.get("value") is True
                    and len(predicate) >= 3
                    and predicate[0].lower() in {"in", "on"}
                    and predicate[2] == args.destination_site
                    and predicate[1] != args.target_name
                    and f"{predicate[1]}_pos" in obs
                ):
                    occupied_positions.append(np.asarray(obs[f"{predicate[1]}_pos"], dtype=np.float32))
            occupancy_offset = np.zeros(3, dtype=np.float32)
            if occupied_positions:
                site_rotation = np.asarray(
                    env.env.sim.data.site_xmat[destination_site_id], dtype=np.float32
                ).reshape(3, 3)
                site_size = np.asarray(
                    env.env.sim.model.site_size[destination_site_id], dtype=np.float32
                )
                span_x = float(np.clip(0.45 * site_size[0], 0.04, 0.08))
                span_y = float(np.clip(0.45 * site_size[min(1, len(site_size) - 1)], 0.04, 0.08))
                candidates = [
                    site_rotation @ np.asarray([x, y, 0.0], dtype=np.float32)
                    for x in (-span_x, 0.0, span_x)
                    for y in (-span_y, 0.0, span_y)
                ]
                occupancy_offset = max(
                    candidates,
                    key=lambda offset: min(
                        np.linalg.norm((destination + offset)[:2] - occupied[:2])
                        for occupied in occupied_positions
                    ) - 0.15 * np.linalg.norm(offset[:2]),
                )
                place = place + occupancy_offset
            transfer = place + np.asarray([0.0, 0.0, 0.13], dtype=np.float32)
            move_to(transfer, 1, "transfer_target")
            move_to(place, 1, "lower_target")
        for _ in range(16):
            if execute([0, 0, 0, 0, 0, 0, -1], "release_target"):
                break
        if not finished:
            if enclosed_destination:
                move_to(entry, -1, "retreat_from_enclosure")
            else:
                retreat = np.asarray(obs["robot0_eef_pos"], dtype=np.float32) + [0.0, 0.0, 0.12]
                move_to(retreat, -1, "retreat")
        for _ in range(12):
            if execute([0, 0, 0, 0, 0, 0, -1], "settle_and_verify"):
                break

        verification_predicate = [args.predicate_name, args.target_name, args.destination_site]
        selected_predicate_success = bool(env.env._eval_predicate(verification_predicate))
        task_success = bool(env.check_success())
        # Multi-subgoal LIBERO-10 teachers are local recoveries: completing the
        # selected predicate is sufficient for later OFT handoff certification.
        success = selected_predicate_success
        np.save(states / f"step-{step:04d}.npy", env.sim.get_state().flatten())
        frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        video = args.output_dir / "rollout.mp4"
        imageio.mimsave(video, frames, fps=args.fps)
        metadata = {
            "kind": "relational_pick_place_feedback_teacher_v1",
            "schema": "v4_rgbd_rollout_v2",
            "success": success,
            "task_success": task_success,
            "verification_predicate": verification_predicate,
            "selected_predicate_success": selected_predicate_success,
            "failure_type": args.failure_type,
            "task_suite": fork["task_suite"], "task_id": task_id, "episode_id": episode_id,
            "fork_timestep": fork_timestep, "task_language": task.language,
            "target_name": args.target_name, "destination_site": args.destination_site,
            "resolved_destination_site": resolved_destination_site,
            "transitions": step, "phase_by_step": phases,
            "source_fork": str(args.fork_dir.resolve()), "rollout": str(video.resolve()),
            "agentview_calibration": calibration_data,
            "teacher_only_generation": True,
            "online_policy_inputs": [
                "main_rgb", "wrist_rgb", "language", "proprio", "executed_action_history"
            ],
            "training_only_auxiliary": ["depth", "instance_segmentation", "privileged_contact"],
            "target_lift_m": float(target_position()[2]) - initial_target_z,
            "controller": {
                "linear_limit": args.linear_limit, "rotation_limit": args.rotation_limit,
                "position_gain": args.position_gain, "rotation_gain": args.rotation_gain,
                "linear_slew": args.linear_slew, "rotation_slew": args.rotation_slew,
                "stable_confirmation_steps": args.stable_confirmation_steps,
                "grasp_offset": list(args.grasp_offset), "place_offset": list(args.place_offset),
                "grasp_quaternion": list(args.grasp_quaternion),
                "enclosed_destination": enclosed_destination,
                "handled_mug_grasp": handled_mug_grasp,
                "resolved_grasp_quaternion": desired_quaternion.tolist(),
                "grasp_profile_file": None if args.grasp_profile_file is None else str(args.grasp_profile_file.resolve()),
                "grasp_profile_key": args.grasp_profile_key,
                "grasp_profile": grasp_profile,
                "target_contact_at_fork": initial_contact,
                "target_attached_at_fork": initial_attached,
                "target_likely_held_at_fork": target_likely_held_at_fork,
                "target_secured_from_fork": target_secured_from_fork,
                "occupancy_offset": occupancy_offset.tolist() if not enclosed_destination else [0.0, 0.0, 0.0],
            },
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2), flush=True)
        if not success:
            raise RuntimeError("teacher did not complete the selected relational predicate")
    except Exception as error:
        # Failed teacher branches are valuable verifier negatives and must be
        # auditable by video.  Never leave a silent partial directory.
        np.save(states / f"step-{step:04d}.npy", env.sim.get_state().flatten())
        frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        video = args.output_dir / "rollout.mp4"
        if frames:
            imageio.mimsave(video, frames, fps=args.fps)
        verification_predicate = [args.predicate_name, args.target_name, args.destination_site]
        diagnostic = {
            "kind": "relational_pick_place_feedback_teacher_v1",
            "schema": "v4_rgbd_rollout_v2",
            "success": False,
            "task_success": bool(env.check_success()),
            "verification_predicate": verification_predicate,
            "selected_predicate_success": bool(env.env._eval_predicate(verification_predicate)),
            "failure_type": args.failure_type,
            "failure_reason": f"{type(error).__name__}: {error}",
            "task_suite": fork["task_suite"], "task_id": task_id, "episode_id": episode_id,
            "fork_timestep": fork_timestep, "task_language": task.language,
            "target_name": args.target_name, "destination_site": args.destination_site,
            "resolved_destination_site": resolved_destination_site,
            "transitions": step, "phase_by_step": phases,
            "source_fork": str(args.fork_dir.resolve()), "rollout": str(video.resolve()),
            "agentview_calibration": calibration_data,
            "teacher_only_generation": True,
            "online_policy_inputs": [
                "main_rgb", "wrist_rgb", "language", "proprio", "executed_action_history"
            ],
            "training_only_auxiliary": ["depth", "instance_segmentation", "privileged_contact"],
            "target_lift_m": float(target_position()[2]) - initial_target_z,
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        print(json.dumps(diagnostic, indent=2), flush=True)
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
