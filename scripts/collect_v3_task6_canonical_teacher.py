#!/usr/bin/env python3
"""Turn an audited task-6 recovery branch into a canonical v3 teacher.

The branch selector may use teacher-only signals offline, but this collector
writes only RGB, wrist RGB, proprio and exact executed actions.  It replays a
recorded nominal prefix to the branch fork, executes the selected recovery
macro, then lets frozen OFT finish the original task.  A result is admitted as
a canonical teacher only if the final environment task succeeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rollout", type=Path, required=True,
                        help="Nominal source directory containing steps.jsonl.")
    parser.add_argument("--recovery-branch", type=Path, required=True,
                        help="Selected BranchTrajectoryWriter raw-transitions.jsonl.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--open-loop-steps", type=int, default=8)
    parser.add_argument("--wait-steps", type=int, default=20,
                        help="Must match the branch collector's stabilization steps.")
    parser.add_argument("--max-completion-steps", type=int, default=280)
    parser.add_argument("--max-fork-eef-error-m", type=float, default=0.03)
    parser.add_argument("--oft-root", type=Path, default=Path("/data/lty/code/openvla-oft-official"))
    parser.add_argument("--libero-root", type=Path, default=ROOT / "experiments" / "robot" / "libero")
    parser.add_argument("--checkpoint", type=Path, default=Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-10"))
    parser.add_argument("--skip-checkpoint-config-sync", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def proprio(obs, np) -> dict[str, list[float]]:
    return {
        "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32).tolist(),
        "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float32).tolist(),
        "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).tolist(),
    }


def main() -> None:
    args = parse_args()
    for path in (args.source_rollout / "steps.jsonl", args.recovery_branch, args.oft_root,
                 args.libero_root / "libero" / "libero", args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.open_loop_steps < 1 or args.wait_steps < 0 or args.max_completion_steps < 1:
        raise ValueError("invalid collection step count")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    sys.path[:0] = [str(args.oft_root), str(args.libero_root), str(ROOT)]
    os.chdir(args.oft_root)

    import imageio.v2 as imageio
    import numpy as np
    from libero.libero import benchmark
    import experiments.robot.openvla_utils as openvla_utils
    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action, get_libero_env, get_libero_image, get_libero_wrist_image,
    )
    from experiments.robot.libero.run_libero_eval import (
        GenerateConfig, get_image_resize_size, initialize_model, prepare_observation,
        process_action, set_seed_everywhere, validate_config,
    )
    from experiments.robot.robot_utils import get_action

    source, branch = read_jsonl(args.source_rollout / "steps.jsonl"), read_jsonl(args.recovery_branch)
    if not source or not branch:
        raise ValueError("source rollout and recovery branch must both be non-empty")
    first = branch[0]
    if first.get("task_suite") != "libero_10" or int(first.get("task_id", -1)) != 6:
        raise ValueError("this collector is intentionally limited to audited libero_10 task 6 branches")
    failure_timestep = int(first["failure_timestep"])
    episode_id = int(first["source_episode_id"])
    if failure_timestep > len(source):
        raise ValueError("branch failure timestep exceeds source rollout length")
    if any("executed_action" not in row for row in source[:failure_timestep]):
        raise ValueError("source prefix lacks exact executed_action")
    for row in branch:
        recovery = row.get("online_observation", {}).get("recovery_action")
        if not isinstance(recovery, list) or len(recovery) != 7:
            raise ValueError("branch lacks a seven-dimensional executed recovery_action")

    if args.skip_checkpoint_config_sync:
        expected = {"AutoConfig": "configuration_prismatic.OpenVLAConfig",
                    "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction"}
        if json.loads((args.checkpoint / "config.json").read_text()).get("auto_map") != expected:
            raise ValueError("checkpoint auto_map is not initialized; omit --skip-checkpoint-config-sync once")
        openvla_utils.update_auto_map = lambda _checkpoint: None

    cfg = GenerateConfig(pretrained_checkpoint=str(args.checkpoint), task_suite_name="libero_10",
                         num_trials_per_task=1, num_open_loop_steps=args.open_loop_steps, use_wandb=False)
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(6)
    init_states = suite.get_task_init_states(6)
    if not 0 <= episode_id < len(init_states):
        raise ValueError(f"invalid source episode {episode_id}")
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize = get_image_resize_size(cfg)
    env, language = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir, wrist_dir = args.output_dir / "rgb", args.output_dir / "wrist_rgb"
    rgb_dir.mkdir(exist_ok=True); wrist_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    video: list[Any] = []

    def record(obs, action, *, phase: str, planned_action=None) -> None:
        frame = f"step-{len(records):04d}"
        rgb, wrist = get_libero_image(obs), get_libero_wrist_image(obs)
        imageio.imwrite(rgb_dir / f"{frame}.png", rgb)
        imageio.imwrite(wrist_dir / f"{frame}.png", wrist)
        video.append(rgb)
        records.append({
            "kind": "v3_canonical_recovery_transition", "task_suite": "libero_10", "task_id": 6,
            "episode_id": episode_id, "failure_timestep": failure_timestep, "timestep": len(records),
            "phase": phase, "gate_label": 1,
            "online_observation": {"rgb": str((rgb_dir / f"{frame}.png").resolve()),
                                   "wrist_rgb": str((wrist_dir / f"{frame}.png").resolve()),
                                   "proprio": proprio(obs, np)},
            "planned_action": (np.asarray(planned_action, np.float32).tolist()
                               if planned_action is not None else np.asarray(action, np.float32).tolist()),
            "executed_action": np.asarray(action, np.float32).tolist(),
        })

    try:
        env.reset(); obs = env.set_init_state(init_states[episode_id])
        for _ in range(args.wait_steps):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        for row in source[:failure_timestep]:
            obs, _, _, _ = env.step(row["executed_action"])
        expected_eef = np.asarray(first["online_observation"]["proprio"]["robot0_eef_pos"], np.float32)
        fork_error = float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"], np.float32) - expected_eef))
        if fork_error > args.max_fork_eef_error_m:
            raise RuntimeError(f"fork replay mismatch {fork_error:.4f}m > {args.max_fork_eef_error_m:.4f}m")
        for row in branch:
            action = np.asarray(row["online_observation"]["recovery_action"], np.float32)
            record(obs, action, phase="TEACHER_RECOVERY")
            obs, _, done, _ = env.step(action.tolist())
            if done:
                break
        queue: deque = deque()
        success = bool(env.check_success())
        for _ in range(args.max_completion_steps):
            if success or done:
                break
            if not queue:
                policy_obs, _ = prepare_observation(obs, resize)
                queue.extend(get_action(cfg, model, policy_obs, language, processor, action_head,
                                        proprio_projector, noisy_action_projector, cfg.use_film))
            planned = np.asarray(queue.popleft(), np.float32)
            action = process_action(planned, cfg.model_family)
            record(obs, action, phase="OFT_COMPLETION", planned_action=planned)
            obs, _, done, _ = env.step(action.tolist())
            success = bool(env.check_success())
        metadata = {
            "kind": "v3_canonical_recovery_teacher", "task_suite": "libero_10", "task_id": 6,
            "episode_id": episode_id, "task_language": language, "success": bool(success),
            "transitions": len(records), "failure_timestep": failure_timestep,
            "recovery_branch": str(args.recovery_branch), "source_rollout": str(args.source_rollout),
            "fork_eef_error_m": fork_error, "schema": "v3_canonical_teacher_v1",
            "teacher_only_branch_selection": True, "saved_frame_images": True,
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with (args.output_dir / "steps.jsonl").open("w") as handle:
            for row in records:
                handle.write(json.dumps(row) + "\n")
        if success:
            with (args.output_dir / "transitions.jsonl").open("w") as handle:
                for row in records:
                    handle.write(json.dumps(row) + "\n")
        imageio.mimsave(args.output_dir / "rollout.mp4", video, fps=30)
        print(json.dumps(metadata), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
