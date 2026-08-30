#!/usr/bin/env python3
"""Offline teacher search for book placement residuals after a verified lift.

The input snapshot must be produced by ``collect_book_regrasp_fsm_branches``
after contact and lift have been verified.  Thus every candidate starts with
the same grasp quality and searches only TRANSPORT -> PLACE -> OPEN -> RETREAT.
Object/caddy poses and task completion are teacher-only; saved trajectories
retain only causal RGB, wrist RGB, proprio and executed actions as inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lift-snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-placement-anchor", required=True,
                        help="Teacher-only EEF x,y,z anchor in metres.")
    parser.add_argument("--xy-residuals-m", default="-0.015,0,0.015")
    parser.add_argument("--z-residuals-m", default="-0.010,0,0.010")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--transport-lift-m", type=float, default=0.060,
                        help="Teacher-only EEF clearance added before horizontal transport.")
    parser.add_argument("--target-key", default="black_book_1_pos")
    parser.add_argument("--target-quat-key", default="black_book_1_quat")
    parser.add_argument("--placement-xy-tolerance-m", type=float, default=0.006)
    parser.add_argument("--open-steps", type=int, default=7)
    parser.add_argument("--retreat-steps", type=int, default=6)
    parser.add_argument("--libero-root", type=Path, default=Path("/data/lty/code/LIBERO-plus"))
    parser.add_argument("--oft-root", type=Path, default=Path("/data/lty/code/openvla-oft-official"))
    return parser.parse_args()


def values(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def anchor(text: str) -> np.ndarray:
    parsed = np.asarray(values(text), dtype=np.float32)
    if parsed.shape != (3,):
        raise ValueError("--base-placement-anchor must be x,y,z")
    return parsed


def env_action_to_oft(action: np.ndarray) -> np.ndarray:
    result = np.asarray(action, dtype=np.float32).copy()
    result[-1] = (1.0 - result[-1]) / 2.0
    return result


def action(xyz: np.ndarray, gripper: float) -> np.ndarray:
    return np.asarray([*np.clip(xyz, -0.65, 0.65), 0.0, 0.0, 0.0, gripper], np.float32)


def main() -> None:
    args = parse_args()
    manifest_path = args.lift_snapshot_dir / "manifest.json"
    state_path = args.lift_snapshot_dir / "mujoco_state.npz"
    if not manifest_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("lift snapshot needs manifest.json and mujoco_state.npz")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "offline_book_verified_lift_snapshot_v1" or not manifest.get("teacher_only"):
        raise ValueError("expected a verified-lift teacher-only snapshot")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    state = np.load(state_path)["mujoco_state"]
    base = anchor(args.base_placement_anchor)
    residuals = list(product(values(args.xy_residuals_m), values(args.xy_residuals_m), values(args.z_residuals_m)))

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    # LIBERO-Plus must shadow OFT's bundled legacy LIBERO implementation.
    sys.path[:0] = [str(args.libero_root), str(args.oft_root), str(Path(__file__).resolve().parents[1])]
    os.chdir(args.oft_root)
    import imageio.v2 as imageio
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image, get_libero_wrist_image
    from experiments.robot.libero.run_libero_eval import GenerateConfig

    task_suite, task_id, episode_id = manifest["task_suite"], int(manifest["task_id"]), int(manifest["episode_id"])
    task = benchmark.get_benchmark_dict()[task_suite]().get_task(task_id)
    initial_state = benchmark.get_benchmark_dict()[task_suite]().get_task_init_states(task_id)[episode_id]
    cfg = GenerateConfig(task_suite_name=task_suite, num_trials_per_task=1)
    for index, residual in enumerate(residuals):
        residual_array = np.asarray(residual, np.float32)
        target = base + residual_array
        name = f"placement_{index:02d}_dx{residual[0] * 1000:+.0f}_dy{residual[1] * 1000:+.0f}_dz{residual[2] * 1000:+.0f}".replace("+", "p").replace("-", "m")
        branch_dir = args.output_root / name
        rgb_dir, wrist_dir = branch_dir / "rgb", branch_dir / "wrist_rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True); wrist_dir.mkdir(parents=True, exist_ok=True)
        env, _ = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        records, frames, success = [], [], False
        try:
            env.reset(); env.set_init_state(initial_state)
            obs = env.regenerate_obs_from_state(state)
            controller = env.env.robots[0].controller
            controller.update(force=True); controller.reset_goal()
            env.env.timestep = 0; env.env.done = False
            initial_eef_z = float(obs["robot0_eef_pos"][2])
            phase, phase_steps = "TRANSPORT_LIFT", 0
            for step in range(args.max_steps):
                eef = np.asarray(obs["robot0_eef_pos"], np.float32)
                phase_before = phase
                if phase == "TRANSPORT_LIFT":
                    command = action(np.asarray([0.0, 0.0, (initial_eef_z + args.transport_lift_m - eef[2]) / .0105]), 1.0)
                    if phase_steps >= 2 and eef[2] >= initial_eef_z + args.transport_lift_m - .006:
                        phase, phase_steps = "TRANSPORT", 0
                elif phase == "TRANSPORT":
                    command = action(np.asarray([(target[0] - eef[0]) / .0105, (target[1] - eef[1]) / .0105, 0.0]), 1.0)
                    if phase_steps >= 2 and np.linalg.norm(target[:2] - eef[:2]) <= args.placement_xy_tolerance_m:
                        phase, phase_steps = "PLACE", 0
                elif phase == "PLACE":
                    command = action(np.asarray([0.0, 0.0, (target[2] - eef[2]) / .0105]), 1.0)
                    if phase_steps >= 2 and eef[2] <= target[2] + .006:
                        phase, phase_steps = "OPEN", 0
                elif phase == "OPEN":
                    command = action(np.zeros(3), -1.0)
                    if phase_steps >= args.open_steps - 1:
                        phase, phase_steps = "RETREAT", 0
                else:
                    command = action(np.asarray([0.0, 0.0, .28]), -1.0)
                    if phase_steps >= args.retreat_steps - 1:
                        break
                main, wrist = get_libero_image(obs), get_libero_wrist_image(obs)
                main_path, wrist_path = rgb_dir / f"step-{step:04d}.png", wrist_dir / f"step-{step:04d}.png"
                imageio.imwrite(main_path, main); imageio.imwrite(wrist_path, wrist); frames.append(main)
                proprio = {key: np.asarray(obs[key]).tolist() for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")}
                book_position = np.asarray(obs[args.target_key], np.float32)
                book_quaternion = np.asarray(obs[args.target_quat_key], np.float32)
                obs, _, done, _ = env.step(command.tolist()); success = success or bool(done)
                records.append({"kind": "offline_book_placement_transition", "timestep": step,
                                "online_observation": {"rgb": str(main_path.resolve()), "wrist_rgb": str(wrist_path.resolve()), "proprio": proprio},
                                "recovery_action_head_space": env_action_to_oft(command).tolist(), "executed_action": command.tolist(),
                                "teacher_only": {"placement_anchor": target.tolist(), "placement_residual": residual_array.tolist(),
                                                 "recovery_phase": {"TRANSPORT_LIFT": 5, "TRANSPORT": 5, "PLACE": 6, "OPEN": 7, "RETREAT": 8}[phase_before],
                                                 "target_position": book_position.tolist(),
                                                 "target_quaternion_wxyz": book_quaternion.tolist(),
                                                 "target_relative_position": (book_position - eef).tolist()}})
                phase_steps = phase_steps + 1 if phase == phase_before else 0
                if success:
                    break
        finally:
            env.close()
        (branch_dir / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        imageio.mimsave(branch_dir / "branch.mp4", frames, fps=30)
        metadata = {"kind": "offline_book_placement_search_branch_v1", "teacher_only": True,
                    "source_verified_lift_snapshot": str(args.lift_snapshot_dir.resolve()), "task_suite": task_suite,
                    "task_id": task_id, "episode_id": episode_id, "placement_anchor": target.tolist(),
                    "placement_residual": residual_array.tolist(), "task_success": bool(success), "num_actions": len(records),
                    "transport_lift_m": args.transport_lift_m,
                    "online_inputs_permitted": ["main_rgb", "wrist_rgb", "language", "proprio", "executed_action_history"]}
        (branch_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
