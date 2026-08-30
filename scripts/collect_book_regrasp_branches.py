#!/usr/bin/env python3
"""Generate teacher-only local regrasp branches from a saved book-task fork.

This is deliberately a narrow data-collection tool for the book→caddy pilot.
It reads the book position and gripper contact only to generate and label
offline branches.  Those signals are never written into online model inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fork-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path,
                        default=Path("/data/lty/code/LIBERO-plus"))
    parser.add_argument("--oft-root", type=Path,
                        default=Path("/data/lty/code/openvla-oft-official"))
    parser.add_argument("--target-key", default="black_book_1_pos")
    parser.add_argument("--target-name", default="black_book_1")
    parser.add_argument("--lift-threshold-m", type=float, default=0.015)
    parser.add_argument("--xy-scales", default="0.80,1.00,1.20")
    parser.add_argument(
        "--lateral-offsets", default="0.0",
        help="Comma-separated action-space offsets perpendicular to the target direction.",
    )
    parser.add_argument("--z-scales", default="0.85,1.10")
    return parser.parse_args()


def env_action_to_oft(action: np.ndarray) -> np.ndarray:
    """Convert LIBERO environment gripper convention back to OFT head space."""
    target = np.asarray(action, dtype=np.float32).copy()
    target[-1] = (1.0 - target[-1]) / 2.0  # env +1 close / -1 open -> OFT 0 / 1
    return target


def local_regrasp_candidates(
    relative_xyz: np.ndarray,
    xy_scales: tuple[float, ...],
    lateral_offsets: tuple[float, ...],
    z_scales: tuple[float, ...],
) -> dict[str, np.ndarray]:
    """Small action-space uncertainty set around an oracle local displacement."""
    gain_m_per_unit_step = 0.0105
    raise_steps, move_steps, descend_steps = 6, 10, 12
    raise_action = 0.30
    raised_m = gain_m_per_unit_step * raise_steps * raise_action
    candidates: dict[str, np.ndarray] = {}
    for xy_scale in xy_scales:
        for lateral_offset in lateral_offsets:
            for z_scale in z_scales:
                horizontal = np.clip(
                    relative_xyz[:2] * xy_scale / (gain_m_per_unit_step * move_steps),
                    -0.85,
                    0.85,
                )
                norm = float(np.linalg.norm(horizontal))
                if norm > 1e-6:
                    horizontal = horizontal + lateral_offset * np.array(
                        [-horizontal[1], horizontal[0]], dtype=np.float32
                    ) / norm
                    horizontal = np.clip(horizontal, -0.85, 0.85)
                descend = np.clip(
                    (relative_xyz[2] - raised_m) * z_scale
                    / (gain_m_per_unit_step * descend_steps),
                    -0.70,
                    -0.10,
                )
                lateral_tag = f"p{int(lateral_offset * 100):03d}" if lateral_offset >= 0 else f"m{int(-lateral_offset * 100):03d}"
                name = f"xy{int(xy_scale * 100):03d}_lat{lateral_tag}_z{int(z_scale * 100):03d}"
                open_action = np.array([0, 0, raise_action, 0, 0, 0, -1], np.float32)
                move_action = np.array([horizontal[0], horizontal[1], 0, 0, 0, 0, -1], np.float32)
                down_action = np.array([0, 0, descend, 0, 0, 0, -1], np.float32)
                close_action = np.array([0, 0, 0, 0, 0, 0, 1], np.float32)
                lift_action = np.array([0, 0, 0.25, 0, 0, 0, 1], np.float32)
                candidates[name] = np.stack(
                    [open_action] * raise_steps
                    + [move_action] * move_steps
                    + [down_action] * descend_steps
                    + [close_action] * 12
                    + [lift_action] * 10
                )
    return candidates


def main() -> None:
    args = parse_args()
    manifest_path = args.fork_dir / "manifest.json"
    state_path = args.fork_dir / "mujoco_state.npz"
    if not manifest_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("fork directory must contain manifest.json and mujoco_state.npz")
    if not (args.libero_root / "libero" / "libero").is_dir():
        raise FileNotFoundError(args.libero_root)
    if not args.oft_root.is_dir():
        raise FileNotFoundError(args.oft_root)
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("teacher_only"):
        raise ValueError("fork manifest must be teacher-only")
    state = np.load(state_path)["mujoco_state"]

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    # The local repository has legacy ``experiments.robot`` modules with the
    # same package name.  OFT's utilities must win this import resolution.
    # LIBERO-Plus must shadow OFT's bundled legacy LIBERO implementation.
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
    suite = benchmark.get_benchmark_dict()[task_suite]()
    task = suite.get_task(task_id)
    initial_state = suite.get_task_init_states(task_id)[episode_id]

    # Read the oracle-relative geometry once to make a matched candidate set.
    env, _ = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    try:
        env.reset()
        env.set_init_state(initial_state)
        obs = env.regenerate_obs_from_state(state)
        target_before = np.asarray(obs[args.target_key], dtype=np.float32)
        eef_before = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        parse_values = lambda value: tuple(float(part) for part in value.split(",") if part.strip())
        candidates = local_regrasp_candidates(
            target_before - eef_before,
            parse_values(args.xy_scales),
            parse_values(args.lateral_offsets),
            parse_values(args.z_scales),
        )
    finally:
        env.close()

    for name, actions in candidates.items():
        branch_dir = args.output_root / name
        main_dir, wrist_dir = branch_dir / "rgb", branch_dir / "wrist_rgb"
        main_dir.mkdir(parents=True, exist_ok=True)
        wrist_dir.mkdir(parents=True, exist_ok=True)
        env, _ = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        try:
            env.reset()
            env.set_init_state(initial_state)
            obs = env.regenerate_obs_from_state(state)
            controller = env.env.robots[0].controller
            controller.update(force=True)
            controller.reset_goal()
            env.env.timestep = 0
            env.env.done = False
            target_before = np.asarray(obs[args.target_key], dtype=np.float32)
            _, attached_before = extract_target_gripper_contact(env.sim, target_name=args.target_name)
            records, frames = [], []
            for index, action in enumerate(actions):
                main = get_libero_image(obs)
                wrist = get_libero_wrist_image(obs)
                # The observation is the policy input *before* ``action`` is
                # executed.  Keep proprio at that same causal timestep; using
                # the post-step state would leak the effect of the label.
                proprio = {
                    "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"]).tolist(),
                    "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"]).tolist(),
                    "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"]).tolist(),
                }
                # Teacher-only visual-geometry supervision.  These values are
                # deliberately outside ``online_observation`` and are never
                # read by the online runner; they train an estimator from
                # RGB/wrist/robot history to the grasp-relative displacement.
                target_position = np.asarray(obs[args.target_key], dtype=np.float32)
                eef_position = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
                imageio.imwrite(main_dir / f"step-{index:04d}.png", main)
                imageio.imwrite(wrist_dir / f"step-{index:04d}.png", wrist)
                frames.append(main)
                obs, _, _, _ = env.step(action.tolist())
                records.append({
                    "kind": "offline_book_regrasp_transition",
                    "timestep": index,
                    "online_observation": {
                        "rgb": str((main_dir / f"step-{index:04d}.png").resolve()),
                        "wrist_rgb": str((wrist_dir / f"step-{index:04d}.png").resolve()),
                        "proprio": proprio,
                    },
                    "recovery_action_head_space": env_action_to_oft(action).tolist(),
                    "executed_action": action.tolist(),
                    "teacher_only": {
                        "target_position": target_position.tolist(),
                        "target_relative_position": (target_position - eef_position).tolist(),
                    },
                })
            target_after = np.asarray(obs[args.target_key], dtype=np.float32)
            contact, attached_after = extract_target_gripper_contact(env.sim, target_name=args.target_name)
            lift_m = float(target_after[2] - target_before[2])
            verified = bool(attached_after and lift_m >= args.lift_threshold_m)
        finally:
            env.close()
        with (branch_dir / "steps.jsonl").open("w") as output:
            for record in records:
                output.write(json.dumps(record) + "\n")
        imageio.mimsave(branch_dir / "branch.mp4", frames, fps=30)
        branch_manifest = {
            "kind": "offline_book_regrasp_branch_v2",
            "observation_timing": "pre_action",
            "teacher_only": {"target_pose": True, "contact": True},
            "source_fork": str(args.fork_dir.resolve()),
            "task_suite": task_suite,
            "task_id": task_id,
            "episode_id": episode_id,
            "candidate_name": name,
            "num_actions": int(len(actions)),
            "attached_before": bool(attached_before),
            "attached_after": bool(attached_after),
            "target_lift_m": lift_m,
            "recovery_verified": verified,
            "teacher_contact": contact,
            "online_inputs_permitted": [
                "main_rgb", "wrist_rgb", "language", "proprio", "executed_action_history"
            ],
        }
        (branch_dir / "metadata.json").write_text(json.dumps(branch_manifest, indent=2) + "\n")
        print(json.dumps(branch_manifest), flush=True)


if __name__ == "__main__":
    main()
