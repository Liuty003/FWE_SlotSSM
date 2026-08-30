#!/usr/bin/env python3
"""Collect feedback-teacher book regrasp trajectories from an OFT failure fork.

Unlike the fixed-duration branch collector, this offline-only teacher reads
simulator target pose/contact after every action and transitions through a
retract/align/descend/close/lift state machine.  It may retry an unsuccessful
grasp with a changed lateral approach.  Pose/contact are stored only under
``teacher_only``; deployment observations remain causal RGB, wrist RGB,
proprio and executed-action history.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PHASE_NAMES = (
    "BACKOFF", "ALIGN", "DESCEND", "CLOSE", "LIFT_VERIFY",
    "TRANSPORT", "PLACE", "OPEN", "RETREAT",
)


def env_action_to_oft(action: np.ndarray) -> np.ndarray:
    """Convert LIBERO's env gripper convention to OFT action-head space."""
    target = np.asarray(action, dtype=np.float32).copy()
    target[-1] = (1.0 - target[-1]) / 2.0
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fork-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, default=Path("/data/lty/code/LIBERO-plus"))
    parser.add_argument("--oft-root", type=Path, default=Path("/data/lty/code/openvla-oft-official"))
    parser.add_argument("--target-key", default="black_book_1_pos")
    parser.add_argument("--target-quat-key", default="black_book_1_quat",
                        help="Teacher-only book orientation observation key, in w,x,y,z order.")
    parser.add_argument("--target-name", default="black_book_1")
    parser.add_argument("--xy-offsets", default="0,0;0,0.025;0,-0.025;0.025,0;-0.025,0",
                        help="Semicolon-separated oracle approach offsets in metres.")
    parser.add_argument("--max-steps", type=int, default=110)
    parser.add_argument("--grasp-eef-height-m", type=float, default=0.105,
                        help="EEF-reference height above book center at which to start closing.")
    parser.add_argument("--xy-tolerance-m", type=float, default=0.004)
    parser.add_argument("--close-steps", type=int, default=12)
    parser.add_argument("--lift-threshold-m", type=float, default=0.015)
    parser.add_argument(
        "--placement-anchor",
        help=("Optional teacher-only EEF x,y,z placement anchor. With this option the "
              "teacher records regrasp, transport, place, release, and retreat."),
    )
    parser.add_argument("--placement-xy-tolerance-m", type=float, default=0.006)
    parser.add_argument("--placement-open-steps", type=int, default=7)
    parser.add_argument("--placement-retreat-steps", type=int, default=6)
    parser.add_argument(
        "--save-lift-snapshots-dir", type=Path,
        help=("Optional teacher-only output root. Each verified lift is saved as a "
              "MuJoCo fork before transport, so placement residual candidates can "
              "be searched without repeating regrasp."),
    )
    parser.add_argument(
        "--container-body-name", default="desk_caddy_1",
        help="Teacher-only MuJoCo body whose pose is recorded with a verified lift.",
    )
    return parser.parse_args()


def parse_offsets(value: str) -> list[np.ndarray]:
    offsets = []
    for item in value.split(";"):
        x_y = [float(part) for part in item.split(",") if part.strip()]
        if len(x_y) != 2:
            raise ValueError(f"each xy offset must contain two values: {item!r}")
        offsets.append(np.asarray(x_y, dtype=np.float32))
    if not offsets:
        raise ValueError("at least one xy offset is required")
    return offsets


def clip_action(xyz: np.ndarray, gripper: float) -> np.ndarray:
    return np.asarray([*np.clip(xyz, -0.65, 0.65), 0.0, 0.0, 0.0, gripper], dtype=np.float32)


def main() -> None:
    args = parse_args()
    manifest_path, state_path = args.fork_dir / "manifest.json", args.fork_dir / "mujoco_state.npz"
    if not manifest_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("fork directory must contain manifest.json and mujoco_state.npz")
    if args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("teacher_only"):
        raise ValueError("FSM collection accepts teacher-only fork snapshots only")
    state = np.load(state_path)["mujoco_state"]
    offsets = parse_offsets(args.xy_offsets)
    placement_anchor = None
    if args.placement_anchor is not None:
        try:
            placement_anchor = np.asarray(
                [float(value) for value in args.placement_anchor.split(",")], dtype=np.float32,
            )
        except ValueError as error:
            raise ValueError("--placement-anchor must be x,y,z") from error
        if placement_anchor.shape != (3,):
            raise ValueError("--placement-anchor must be x,y,z")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    # LIBERO-Plus must shadow OFT's bundled legacy LIBERO implementation.
    # OFT still follows it on the path to provide the evaluator utilities.
    sys.path[:0] = [str(args.libero_root), str(args.oft_root), str(Path(__file__).resolve().parents[1])]
    os.chdir(args.oft_root)
    import imageio.v2 as imageio
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image, get_libero_wrist_image
    from experiments.robot.libero.run_libero_eval import GenerateConfig
    from recovery.teachers.sim_contact import extract_target_gripper_contact

    task_suite = str(manifest["task_suite"])
    task_id, episode_id = int(manifest["task_id"]), int(manifest["episode_id"])
    cfg = GenerateConfig(task_suite_name=task_suite, num_trials_per_task=1)
    task = benchmark.get_benchmark_dict()[task_suite]().get_task(task_id)
    initial_state = benchmark.get_benchmark_dict()[task_suite]().get_task_init_states(task_id)[episode_id]

    for candidate_index, initial_offset in enumerate(offsets):
        name = f"fsm_xy{initial_offset[0] * 1000:+.0f}_{initial_offset[1] * 1000:+.0f}".replace("+", "p").replace("-", "m")
        branch_dir = args.output_root / name
        rgb_dir, wrist_dir = branch_dir / "rgb", branch_dir / "wrist_rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        wrist_dir.mkdir(parents=True, exist_ok=True)
        env, language = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        try:
            env.reset(); env.set_init_state(initial_state)
            obs = env.regenerate_obs_from_state(state)
            controller = env.env.robots[0].controller
            controller.update(force=True); controller.reset_goal()
            env.env.timestep = 0; env.env.done = False
            target_initial = np.asarray(obs[args.target_key], dtype=np.float32)
            _, attached_before = extract_target_gripper_contact(env.sim, target_name=args.target_name)
            phase, phase_steps, retry = 0, 0, 0
            records, frames, task_success, saved_lift_snapshot = [], [], False, False
            for step in range(args.max_steps):
                target = np.asarray(obs[args.target_key], dtype=np.float32)
                target_quat = np.asarray(obs[args.target_quat_key], dtype=np.float32)
                if target_quat.shape != (4,):
                    raise ValueError(f"expected quaternion at {args.target_quat_key}, got {target_quat.shape}")
                eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
                _, attached = extract_target_gripper_contact(env.sim, target_name=args.target_name)
                phase_before = phase
                desired_xy = target[:2] + initial_offset
                # On retry, move the approach laterally away from the first
                # failed contact. This yields genuine state-conditioned paths.
                if retry:
                    desired_xy = desired_xy + np.asarray([0.0, 0.030 if initial_offset[1] <= 0 else -0.030], np.float32)
                if phase == 0:
                    action = clip_action(np.asarray([0.0, 0.0, 0.35], np.float32), -1.0)
                    transition = phase_steps >= 4 and eef[2] >= target[2] + 0.10
                    if transition:
                        phase, phase_steps = 1, 0
                elif phase == 1:
                    action = clip_action(np.asarray([(desired_xy[0] - eef[0]) / 0.0105,
                                                     (desired_xy[1] - eef[1]) / 0.0105, 0.0], np.float32), -1.0)
                    transition = phase_steps >= 2 and float(np.linalg.norm(desired_xy - eef[:2])) <= args.xy_tolerance_m
                    if transition:
                        phase, phase_steps = 2, 0
                elif phase == 2:
                    # The EEF reference frame is roughly 10 cm above the
                    # book center at a viable pinch contact in this LIBERO
                    # task.  It is not the object's surface height.
                    desired_z = target[2] + args.grasp_eef_height_m
                    action = clip_action(np.asarray([0.0, 0.0, (desired_z - eef[2]) / 0.0105], np.float32), -1.0)
                    transition = phase_steps >= 2 and eef[2] <= desired_z + 0.006
                    if transition:
                        phase, phase_steps = 3, 0
                elif phase == 3:
                    action = clip_action(np.zeros(3, np.float32), 1.0)
                    if attached:
                        phase, phase_steps = 4, 0
                    elif phase_steps >= args.close_steps - 1:
                        if retry >= 1:
                            break
                        retry += 1; phase, phase_steps = 0, 0
                elif phase == 4:
                    action = clip_action(np.asarray([0.0, 0.0, 0.28], np.float32), 1.0)
                    if attached and target[2] - target_initial[2] >= args.lift_threshold_m:
                        if args.save_lift_snapshots_dir is not None and not saved_lift_snapshot:
                            snapshot_dir = args.save_lift_snapshots_dir / name
                            snapshot_dir.mkdir(parents=True, exist_ok=True)
                            # LIBERO's BDDL domain already resolves fixture
                            # names to their root bodies. Prefer that mapping
                            # because MuJoCo's raw body name can carry a
                            # fixture-specific suffix in some scenes.
                            container_id = env.env.obj_body_id.get(args.container_body_name)
                            if container_id is None:
                                container_id = env.sim.model.body_name2id(args.container_body_name)
                            container_position = np.asarray(env.sim.data.body_xpos[container_id], dtype=np.float32)
                            container_quaternion = np.asarray(env.sim.data.body_xquat[container_id], dtype=np.float32)
                            np.savez_compressed(
                                snapshot_dir / "mujoco_state.npz",
                                mujoco_state=env.sim.get_state().flatten(),
                            )
                            snapshot_manifest = {
                                "kind": "offline_book_verified_lift_snapshot_v1",
                                "teacher_only": True,
                                "task_suite": task_suite,
                                "task_id": task_id,
                                "episode_id": episode_id,
                                "task_language": language,
                                "source_recovery_fork": str(args.fork_dir.resolve()),
                                "source_candidate_name": name,
                                "state_file": "mujoco_state.npz",
                                "teacher_only_lift": {
                                    "target_position": target.tolist(),
                                    "target_quaternion_wxyz": target_quat.tolist(),
                                    "eef_position": eef.tolist(),
                                    "container_body_name": args.container_body_name,
                                    "container_position": container_position.tolist(),
                                    "container_quaternion_wxyz": container_quaternion.tolist(),
                                },
                                "online_inputs_permitted": [
                                    "main_rgb", "wrist_rgb", "language", "proprio", "executed_action_history",
                                ],
                            }
                            (snapshot_dir / "manifest.json").write_text(
                                json.dumps(snapshot_manifest, indent=2) + "\n"
                            )
                            saved_lift_snapshot = True
                        # Continue through placement when requested; otherwise
                        # preserve the original lift-only regrasp collector.
                        if phase_steps >= 3:
                            if placement_anchor is None:
                                break
                            phase, phase_steps = 5, 0
                    elif not attached and phase_steps >= 3:
                        if retry >= 1:
                            break
                        retry += 1; phase, phase_steps = 0, 0
                elif phase == 5:
                    action = clip_action(np.asarray([
                        (placement_anchor[0] - eef[0]) / 0.0105,
                        (placement_anchor[1] - eef[1]) / 0.0105,
                        0.0,
                    ], np.float32), 1.0)
                    if (phase_steps >= 2 and float(np.linalg.norm(
                            placement_anchor[:2] - eef[:2])) <= args.placement_xy_tolerance_m):
                        phase, phase_steps = 6, 0
                elif phase == 6:
                    action = clip_action(np.asarray([
                        0.0, 0.0, (placement_anchor[2] - eef[2]) / 0.0105,
                    ], np.float32), 1.0)
                    if phase_steps >= 2 and eef[2] <= placement_anchor[2] + 0.006:
                        phase, phase_steps = 7, 0
                elif phase == 7:
                    action = clip_action(np.zeros(3, np.float32), -1.0)
                    if phase_steps >= args.placement_open_steps - 1:
                        phase, phase_steps = 8, 0
                else:
                    action = clip_action(np.asarray([0.0, 0.0, 0.28], np.float32), -1.0)
                    if phase_steps >= args.placement_retreat_steps - 1:
                        break

                main, wrist = get_libero_image(obs), get_libero_wrist_image(obs)
                imageio.imwrite(rgb_dir / f"step-{step:04d}.png", main)
                imageio.imwrite(wrist_dir / f"step-{step:04d}.png", wrist)
                frames.append(main)
                proprio = {key: np.asarray(obs[key]).tolist() for key in
                            ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")}
                obs, _, done, _ = env.step(action.tolist())
                task_success = task_success or bool(done)
                records.append({
                    "kind": "offline_book_regrasp_transition", "timestep": step,
                    "online_observation": {"rgb": str((rgb_dir / f"step-{step:04d}.png").resolve()),
                                           "wrist_rgb": str((wrist_dir / f"step-{step:04d}.png").resolve()),
                                           "proprio": proprio},
                    "recovery_action_head_space": env_action_to_oft(action).tolist(),
                    "executed_action": action.tolist(),
                    "teacher_only": {"target_position": target.tolist(),
                                     "target_quaternion_wxyz": target_quat.tolist(),
                                     "target_relative_position": (target - eef).tolist(),
                                     # Dense privileged labels for a future
                                     # RGB/wrist grasp-quality head.  These
                                     # never enter online observations.
                                     "target_attached": bool(attached),
                                     "target_lift_from_fork_m": float(target[2] - target_initial[2]),
                                     "recovery_phase": phase_before,
                                     "recovery_phase_name": PHASE_NAMES[phase_before],
                                     "fsm_retry": retry},
                })
                phase_steps = phase_steps + 1 if phase == phase_before else 0
                if task_success:
                    break
            target_after = np.asarray(obs[args.target_key], dtype=np.float32)
            contact, attached_after = extract_target_gripper_contact(env.sim, target_name=args.target_name)
            lift_m = float(target_after[2] - target_initial[2])
            verified = (bool(task_success) if placement_anchor is not None
                        else bool(attached_after and lift_m >= args.lift_threshold_m))
        finally:
            env.close()
        with (branch_dir / "steps.jsonl").open("w") as stream:
            stream.write("".join(json.dumps(record) + "\n" for record in records))
        imageio.mimsave(branch_dir / "branch.mp4", frames, fps=30)
        metadata = {"kind": "offline_book_regrasp_branch_v2", "observation_timing": "pre_action",
                    "teacher_only": {"target_pose": True, "contact": True, "feedback_fsm": True},
                    "source_fork": str(args.fork_dir.resolve()), "task_suite": task_suite,
                    "task_id": task_id, "episode_id": episode_id, "candidate_name": name,
                    "candidate_index": candidate_index, "num_actions": len(records),
                    # This is the candidate *program* exposed to the future
                    # visual chooser.  It is not a simulator measurement:
                    # selecting an XY approach offset is the controller's
                    # own decision.  Keep the physical units so a learned
                    # scorer can distinguish otherwise identical BACKOFF
                    # first actions across matched branches.
                    "candidate_program": {
                        "initial_xy_offset_m": initial_offset.tolist(),
                        "grasp_eef_height_m": args.grasp_eef_height_m,
                        "xy_tolerance_m": args.xy_tolerance_m,
                        "close_steps": args.close_steps,
                        "placement_anchor": None if placement_anchor is None else placement_anchor.tolist(),
                    },
                    "attached_before": bool(attached_before), "attached_after": bool(attached_after),
                    "target_lift_m": lift_m, "task_success": bool(task_success),
                    "recovery_verified": verified, "teacher_contact": contact,
                    "teacher_program": ("feedback_regrasp_place_fsm_v1" if placement_anchor is not None
                                        else "feedback_regrasp_fsm_v1"),
                    "online_inputs_permitted": ["main_rgb", "wrist_rgb", "language", "proprio", "executed_action_history"]}
        (branch_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
