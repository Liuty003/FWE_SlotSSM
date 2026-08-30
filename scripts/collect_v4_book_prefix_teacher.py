#!/usr/bin/env python3
"""Re-record a verified book recovery teacher as a complete V4 RGB-D trace.

The input branches may have been selected with privileged simulator state, but
this replay writes it only as provenance.  Output rows expose RGB-D, fixed
calibration, language, and proprio; simulator pose/contact/FSM are never
written as online inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_v3_book_prefix_teacher import action_rows, online_proprio, read_json, read_jsonl
from scripts.collect_v4_oft_rgbd_rollouts import calibration, depth_and_segmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rollout", type=Path, required=True)
    parser.add_argument("--fork-timestep", type=int, required=True)
    parser.add_argument("--recovery-branch", type=Path, required=True)
    parser.add_argument("--completion-branch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, default=Path("/data/lty/code/LIBERO-plus"))
    parser.add_argument("--oft-root", type=Path, default=Path("/data/lty/code/openvla-oft-official"))
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--open-loop-steps", type=int, default=8,
                        help="Open-loop action chunk size used by the source OFT rollout.")
    parser.add_argument("--max-fork-eef-error-m", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.wait_steps < 0 or args.open_loop_steps < 1:
        raise ValueError("wait-steps must be nonnegative and open-loop-steps must be positive")
    source_steps = args.source_rollout / "steps.jsonl"
    for path in (source_steps, args.recovery_branch, args.completion_branch, args.libero_root / "libero" / "libero"):
        if not path.exists():
            raise FileNotFoundError(path)
    source_meta = read_json(args.source_rollout / "metadata.json")
    recorded_open_loop = source_meta.get("open_loop_steps", source_meta.get("num_open_loop_steps"))
    if recorded_open_loop is not None and int(recorded_open_loop) != args.open_loop_steps:
        raise ValueError(f"source rollout used open_loop_steps={recorded_open_loop}, not {args.open_loop_steps}")
    identity = ("task_suite", "task_id", "episode_id")
    if any(key not in source_meta for key in identity):
        raise ValueError(f"source metadata needs {identity}")
    source, recovery, completion = action_rows(source_steps), action_rows(args.recovery_branch), action_rows(args.completion_branch)
    if not 0 < args.fork_timestep <= len(source):
        raise ValueError("fork-timestep must select a nonempty source prefix")
    expected = read_jsonl(args.recovery_branch)[0].get("online_observation", {}).get("proprio", {}).get("robot0_eef_pos")
    if not isinstance(expected, list) or len(expected) != 3:
        raise ValueError("recovery branch first row lacks pre-action end-effector position")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    sys.path[:0] = [str(args.libero_root), str(args.oft_root), str(ROOT)]
    os.chdir(args.oft_root)
    import imageio.v2 as imageio
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv
    from experiments.robot.libero.libero_utils import get_libero_dummy_action
    from experiments.robot.libero.run_libero_eval import GenerateConfig

    suite = benchmark.get_benchmark_dict()[source_meta["task_suite"]]()
    task_id, episode_id = int(source_meta["task_id"]), int(source_meta["episode_id"])
    task, states = suite.get_task(task_id), suite.get_task_init_states(task_id)
    cfg = GenerateConfig(task_suite_name=source_meta["task_suite"], num_trials_per_task=1)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = SegmentationRenderEnv(bddl_file_name=str(bddl), camera_heights=cfg.env_img_res, camera_widths=cfg.env_img_res,
                               camera_depths=True, camera_segmentations="instance")
    env.seed(0)
    directories = {name: args.output_dir / name for name in ("rgb", "wrist_rgb", "depth", "wrist_depth", "instance")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    phases, video, success, fork_error = [], [], False, None

    def record(obs, action: list[float], phase: str) -> None:
        index, frame = len(records), f"step-{len(records):04d}"
        main, wrist = obs["agentview_image"][::-1, ::-1], obs["robot0_eye_in_hand_image"][::-1, ::-1]
        main_depth, segmentation = depth_and_segmentation(obs, "agentview", np)
        wrist_depth, _ = depth_and_segmentation(obs, "robot0_eye_in_hand", np)
        rgb_path, wrist_path = directories["rgb"] / f"{frame}.png", directories["wrist_rgb"] / f"{frame}.png"
        depth_path, wrist_depth_path = directories["depth"] / f"{frame}.npz", directories["wrist_depth"] / f"{frame}.npz"
        imageio.imwrite(rgb_path, main); imageio.imwrite(wrist_path, wrist)
        np.savez_compressed(depth_path, depth=main_depth); np.savez_compressed(wrist_depth_path, depth=wrist_depth)
        instance_path = None
        if segmentation is not None:
            instance_path = directories["instance"] / f"{frame}.npz"
            np.savez_compressed(instance_path, instance=segmentation.astype(np.int32))
        records.append({
            "kind": "v4_book_prefix_teacher_transition", "timestep": index, "phase": phase,
            "gate_label": "NORMAL" if phase == "NORMAL_PREFIX" else "RECOVERY",
            "online_observation": {"rgb": str(rgb_path.resolve()), "wrist_rgb": str(wrist_path.resolve()),
                                   "depth": str(depth_path.resolve()), "wrist_depth": str(wrist_depth_path.resolve()),
                                   "proprio": online_proprio(obs, np)},
            "training_only": {"instance_segmentation": None if instance_path is None else str(instance_path.resolve())},
            "executed_action": [float(value) for value in action],
        })
        phases.append(phase); video.append(main)

    try:
        obs = env.reset(); obs = env.set_init_state(states[episode_id])
        agent_calibration = calibration(env, "agentview", cfg.env_img_res, cfg.env_img_res, np)
        for _ in range(args.wait_steps):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        for action in source[:args.fork_timestep]:
            record(obs, action, "NORMAL_PREFIX"); obs, _, _, _ = env.step(action)
        fork_error = float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"], np.float32) - np.asarray(expected, np.float32)))
        if fork_error > args.max_fork_eef_error_m:
            raise RuntimeError(f"fork replay mismatch {fork_error:.4f}m > {args.max_fork_eef_error_m:.4f}m")
        done = False
        for action in recovery:
            record(obs, action, "TEACHER_RECOVERY"); obs, _, done, _ = env.step(action)
            if done:
                break
        if not done and not bool(env.check_success()):
            for action in completion:
                record(obs, action, "TEACHER_COMPLETION"); obs, _, done, _ = env.step(action)
                if done:
                    break
        success = bool(env.check_success())
    finally:
        env.close()
    for start in range(0, len(records) - 7, 8):
        if all(phase != "NORMAL_PREFIX" for phase in phases[start:start + 8]):
            records[start]["recovery_target_action_chunk"] = [records[offset]["executed_action"] for offset in range(start, start + 8)]
    metadata = {
        "kind": "v4_book_prefix_canonical_teacher", "schema": "v4_rgbd_rollout_v2", **{key: source_meta[key] for key in identity},
        "task_language": task.language, "success": success, "transitions": len(records), "fork_timestep": args.fork_timestep,
        "fork_eef_error_m": fork_error, "geometry_view": "agentview", "agentview_calibration": agent_calibration,
        "source_rollout": str(args.source_rollout.resolve()), "recovery_branch": str(args.recovery_branch.resolve()),
        "completion_branch": str(args.completion_branch.resolve()), "teacher_only_branch_selection": True,
        "wait_steps": args.wait_steps, "open_loop_steps": args.open_loop_steps,
        "rollout_id": f"book:{source_meta['task_suite']}:{source_meta['task_id']}:{source_meta['episode_id']}:fork{args.fork_timestep}",
        "libero_root": str(args.libero_root.resolve()), "oft_root": str(args.oft_root.resolve()), "env_seed": 0,
        "online_inputs_permitted": ["main_rgb", "wrist_rgb", "main_depth", "wrist_depth", "camera_calibration", "language", "proprio"],
        "training_only_inputs": ["instance_segmentation"],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    if video:
        imageio.mimsave(args.output_dir / "rollout.mp4", video, fps=30)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
