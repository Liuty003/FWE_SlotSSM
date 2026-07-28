#!/usr/bin/env python
"""Screen several LIBERO initial states while loading OpenVLA only once."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from run_openvla_baseline import (
    SUITE_MAX_STEPS,
    extract_privileged_object_state,
    extract_robot_state,
    jsonable,
    load_openvla,
    predict_action,
    write_json,
)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial")
    p.add_argument("--task-suite", default="libero_spatial", choices=tuple(SUITE_MAX_STEPS))
    p.add_argument("--task-id", type=int, default=9)
    p.add_argument("--episode-ids", type=int, nargs="+", default=list(range(1, 10)))
    p.add_argument("--wait-steps", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device-map", default="balanced")
    p.add_argument("--gpu-memory-gib", type=int, default=7)
    p.add_argument("--output-root", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = arguments()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    invalid = [e for e in args.episode_ids if not 0 <= e < len(initial_states)]
    if invalid:
        raise ValueError(f"invalid episode ids {invalid}; available range is 0..{len(initial_states)-1}")

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(0)
    # EGL must own a live context before the large multi-GPU model is loaded.
    env.reset()
    model, processor, input_device = load_openvla(
        args.checkpoint, args.device_map, args.gpu_memory_gib
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    max_steps = args.max_steps or SUITE_MAX_STEPS[args.task_suite]
    noop = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    summary = []

    try:
        for episode_id in args.episode_ids:
            env.reset()
            obs = env.set_init_state(initial_states[episode_id])
            episode_dir = args.output_root / f"task{args.task_id}-episode{episode_id}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            steps_path = episode_dir / "steps.jsonl"
            steps_path.write_text("", encoding="utf-8")
            frames = []
            for _ in range(args.wait_steps):
                obs, _, done, _ = env.step(noop)
                if done:
                    break

            success = False
            for step in range(max_steps):
                image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                frames.append(image)
                action = predict_action(
                    model, processor, image, task.language, args.task_suite,
                    args.center_crop, input_device,
                )
                before = obs
                obs, reward, done, info = env.step(action.tolist())
                success = bool(env.check_success())
                record = {
                    "episode_id": episode_id, "timestep": step,
                    "planned_action": action.tolist(), "executed_action": action.tolist(),
                    "robot_state_before": extract_robot_state(before),
                    "robot_state_after": extract_robot_state(obs),
                    "privileged_object_state_before": extract_privileged_object_state(before),
                    "privileged_object_state_after": extract_privileged_object_state(obs),
                    "reward": float(reward), "done": bool(done), "success": success,
                    "info": {str(k): jsonable(v) for k, v in dict(info).items()},
                }
                with steps_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if success:
                    break

            video = episode_dir / "rollout.mp4"
            imageio.mimsave(video, frames, fps=20)
            metadata = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "policy": "openvla", "task_suite": args.task_suite,
                "task_id": args.task_id, "episode_id": episode_id,
                "task_language": task.language, "completed": True,
                "success": success, "num_policy_steps": len(frames),
                "video": str(video.resolve()), "steps_jsonl": str(steps_path.resolve()),
            }
            write_json(episode_dir / "metadata.json", metadata)
            summary.append(metadata)
            write_json(args.output_root / "summary.json", {"episodes": summary})
            print(f"episode={episode_id} steps={len(frames)} success={success}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
