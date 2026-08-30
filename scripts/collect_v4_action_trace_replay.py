#!/usr/bin/env python3
"""Replay a verified successful action trace into the strict V4 RGB-D schema."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_v3_book_prefix_teacher import online_proprio
from scripts.collect_v4_oft_rgbd_rollouts import calibration, depth_and_segmentation


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--decision-stride", type=int, default=8)
    parser.add_argument("--max-eef-drift-m", type=float, default=0.01)
    parser.add_argument("--libero-root", type=Path, default=Path("/data/lty/code/LIBERO-plus"))
    parser.add_argument("--oft-root", type=Path, default=Path("/data/lty/code/openvla-oft-official"))
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if min(args.decision_stride, args.max_eef_drift_m) <= 0 or args.wait_steps < 0:
        raise ValueError("invalid replay configuration")
    for path in (args.input_report, args.libero_root / "libero/libero", args.oft_root):
        if not path.exists():
            raise FileNotFoundError(path)
    source = json.loads(args.input_report.read_text())
    decisions = source.get("decisions")
    if source.get("success") is not True or not isinstance(decisions, list) or not decisions:
        raise ValueError("input must be a nonempty, verified-success action trace")
    if any(row.get("mode") not in ("NORMAL", "RECOVERY", "RECOVERY_COMPLETE") for row in decisions):
        raise ValueError("action trace contains an unsupported controller mode")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    sys.path[:0] = [str(args.libero_root), str(args.oft_root), str(ROOT)]
    import imageio.v2 as imageio
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task, states = suite.get_task(args.task_id), suite.get_task_init_states(args.task_id)
    if not 0 <= args.episode < len(states):
        raise ValueError("episode is outside benchmark init states")
    if source.get("task") != task.language:
        raise ValueError("input report language does not match requested benchmark task")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
        camera_depths=True, camera_segmentations="instance",
    )
    env.seed(0)
    directories = {name: args.output_dir / name for name in ("rgb", "wrist_rgb", "depth", "wrist_depth", "instance")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    records, phases, video, drift = [], [], [], []
    try:
        observation = env.reset()
        observation = env.set_init_state(states[args.episode])
        agent_calibration = calibration(env, "agentview", 256, 256, np)
        for _ in range(args.wait_steps):
            observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        for index, decision in enumerate(decisions):
            source_observation = decision.get("online_observation")
            expected = None
            if source_observation is not None:
                if not isinstance(source_observation, dict):
                    raise ValueError(f"decision {index} online_observation must be an object or null")
                source_proprio = source_observation.get("proprio")
                if source_proprio is not None:
                    if not isinstance(source_proprio, dict):
                        raise ValueError(f"decision {index} proprio must be an object or null")
                    expected = source_proprio.get("robot0_eef_pos")
            if expected is not None:
                if not isinstance(expected, list) or len(expected) != 3:
                    raise ValueError(f"decision {index} has a malformed pre-action EEF position")
                error = float(np.linalg.norm(np.asarray(observation["robot0_eef_pos"]) - np.asarray(expected)))
                drift.append(error)
                if error > args.max_eef_drift_m:
                    raise RuntimeError(f"replay drift at decision {index}: {error:.6f}m")
            # The source runner deliberately preserves upstream OFT precision
            # on NORMAL actions, while its geometry recovery controller emits
            # float32.  Recreate that distinction before MuJoCo execution.
            raw = np.asarray(decision["action"], dtype=None if decision["mode"] == "NORMAL" else np.float32)
            if raw.shape != (7,) or not np.isfinite(raw).all():
                raise ValueError(f"decision {index} action must be finite [7]")
            executed = raw.copy()
            executed[-1] = -np.sign(2.0 * executed[-1] - 1.0)
            frame = f"step-{index:04d}"
            main = observation["agentview_image"][::-1, ::-1]
            wrist = observation["robot0_eye_in_hand_image"][::-1, ::-1]
            main_depth, instance = depth_and_segmentation(observation, "agentview", np)
            wrist_depth, _ = depth_and_segmentation(observation, "robot0_eye_in_hand", np)
            rgb_path = directories["rgb"] / f"{frame}.png"
            wrist_path = directories["wrist_rgb"] / f"{frame}.png"
            depth_path = directories["depth"] / f"{frame}.npz"
            wrist_depth_path = directories["wrist_depth"] / f"{frame}.npz"
            instance_path = directories["instance"] / f"{frame}.npz"
            imageio.imwrite(rgb_path, main)
            imageio.imwrite(wrist_path, wrist)
            np.savez_compressed(depth_path, depth=main_depth)
            np.savez_compressed(wrist_depth_path, depth=wrist_depth)
            if instance is None:
                raise RuntimeError("strict V4 replay requires instance segmentation for Slot supervision")
            np.savez_compressed(instance_path, instance=instance.astype(np.int32))
            phase = decision["mode"]
            records.append({
                "kind": "v4_verified_action_trace_transition", "timestep": index,
                "phase": phase, "gate_label": "NORMAL" if phase == "NORMAL" else "RECOVERY",
                "online_observation": {
                    "rgb": str(rgb_path.resolve()), "wrist_rgb": str(wrist_path.resolve()),
                    "depth": str(depth_path.resolve()), "wrist_depth": str(wrist_depth_path.resolve()),
                    "proprio": online_proprio(observation, np),
                },
                "training_only": {"instance_segmentation": str(instance_path.resolve())},
                "executed_action": executed.tolist(),
            })
            phases.append(phase)
            video.append(main)
            observation, _, done, _ = env.step(executed.tolist())
            if done and index + 1 != len(decisions):
                raise RuntimeError(f"trace terminated early at decision {index}")
        success = bool(env.check_success())
        if not success:
            raise RuntimeError("action trace replay did not reproduce source success")
    finally:
        env.close()

    for start in range(0, len(records) - args.decision_stride + 1, args.decision_stride):
        if all(phase != "NORMAL" for phase in phases[start:start + args.decision_stride]):
            records[start]["recovery_target_action_chunk"] = [
                records[offset]["executed_action"] for offset in range(start, start + args.decision_stride)
            ]
    metadata = {
        "kind": "v4_verified_action_trace_replay", "schema": "v4_rgbd_rollout_v2",
        "task_suite": args.task_suite, "task_id": args.task_id, "episode_id": args.episode,
        "task_language": task.language, "success": True, "transitions": len(records),
        "wait_steps": args.wait_steps, "open_loop_steps": args.decision_stride,
        "decision_stride": args.decision_stride, "env_seed": 0,
        "fork_timestep": next(index for index, phase in enumerate(phases) if phase != "NORMAL"),
        "rollout_id": f"verified-trace:{args.task_suite}:{args.task_id}:{args.episode}:{args.input_report.resolve()}",
        "libero_root": str(args.libero_root.resolve()), "oft_root": str(args.oft_root.resolve()),
        "eef_drift_reference_rows": len(drift),
        "max_replay_eef_drift_m": None if not drift else max(drift),
        "replay_verification": "final benchmark success reproduced; per-row EEF drift checked where source proprio exists",
        "source_action_trace": str(args.input_report.resolve()),
        "geometry_view": "agentview", "agentview_calibration": agent_calibration,
        "online_inputs_permitted": ["main_rgb", "wrist_rgb", "main_depth", "wrist_depth", "camera_calibration", "language", "proprio"],
        "training_only_inputs": ["instance_segmentation"],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    imageio.mimsave(args.output_dir / "rollout.mp4", video, fps=30)
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
