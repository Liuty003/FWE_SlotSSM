#!/usr/bin/env python3
"""Replay a book failure prefix followed by an offline-selected teacher.

The offline branch selector may have used object state, but this collector
recreates the full trajectory in LIBERO and writes only deploy-time inputs.
Unlike a placement-only historical trace, its NORMAL prefix gives the V3 gate
the actual handoff boundary it must recognise online.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--max-fork-eef-error-m", type=float, default=0.01)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def action_rows(path: Path) -> list[list[float]]:
    rows = read_jsonl(path)
    actions = [row.get("executed_action") for row in rows]
    if not actions or any(not isinstance(value, list) or len(value) != 7 for value in actions):
        raise ValueError(f"{path} lacks seven-dimensional executed_action rows")
    return actions


def online_proprio(obs, np) -> dict[str, list[float]]:
    return {
        "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32).tolist(),
        "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float32).tolist(),
        "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).tolist(),
    }


def main() -> None:
    args = parse_args()
    source_steps = args.source_rollout / "steps.jsonl"
    paths = (source_steps, args.recovery_branch, args.completion_branch, args.oft_root,
             args.libero_root / "libero" / "libero")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    source_meta = read_json(args.source_rollout / "metadata.json")
    identity = ("task_suite", "task_id", "episode_id")
    if any(key not in source_meta for key in identity):
        raise ValueError(f"source metadata needs {identity}")
    if source_meta["task_suite"] != "libero_10":
        raise ValueError("book prefix collector currently supports libero_10 only")
    source = action_rows(source_steps)
    if not 0 < args.fork_timestep <= len(source):
        raise ValueError("fork-timestep must select a nonempty source prefix")
    recovery = action_rows(args.recovery_branch)
    completion = action_rows(args.completion_branch)
    recovery_first = read_jsonl(args.recovery_branch)[0]
    expected = recovery_first.get("online_observation", {}).get("proprio", {}).get("robot0_eef_pos")
    if not isinstance(expected, list) or len(expected) != 3:
        raise ValueError("recovery branch first row lacks pre-action end-effector position")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    # ``get_libero_env`` lives in OFT's evaluator utilities.  Put Plus first
    # so its benchmark package wins over OFT's vendored 10-task LIBERO copy.
    sys.path[:0] = [str(args.libero_root), str(args.oft_root), str(ROOT)]
    os.chdir(args.oft_root)
    import imageio.v2 as imageio
    import numpy as np
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
    from experiments.robot.libero.run_libero_eval import GenerateConfig

    suite = benchmark.get_benchmark_dict()[source_meta["task_suite"]]()
    task_id, episode_id = int(source_meta["task_id"]), int(source_meta["episode_id"])
    task = suite.get_task(task_id)
    states = suite.get_task_init_states(task_id)
    if not 0 <= episode_id < len(states):
        raise ValueError("source episode is outside this benchmark task")
    # Match ``save_oft_recovery_fork_snapshot.py`` exactly.  A hard-coded
    # gripper value here caused a small state discrepancy before the branch;
    # that discrepancy can turn a standalone-successful placement into a
    # failed full prefix -> recovery rollout.
    cfg = GenerateConfig(task_suite_name=source_meta["task_suite"], num_trials_per_task=1)
    env, language = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir, wrist_dir = args.output_dir / "rgb", args.output_dir / "wrist_rgb"
    rgb_dir.mkdir(exist_ok=True); wrist_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    phases: list[str] = []
    video: list[Any] = []
    success = False
    fork_error: float | None = None

    def record(obs, action: list[float], phase: str) -> None:
        index = len(records)
        rgb_path, wrist_path = rgb_dir / f"step-{index:04d}.png", wrist_dir / f"step-{index:04d}.png"
        rgb, wrist = obs["agentview_image"][::-1, ::-1], obs["robot0_eye_in_hand_image"][::-1, ::-1]
        imageio.imwrite(rgb_path, rgb); imageio.imwrite(wrist_path, wrist)
        video.append(rgb)
        records.append({
            "kind": "v3_book_prefix_teacher_transition", "timestep": index, "phase": phase,
            "gate_label": "NORMAL" if phase == "NORMAL_PREFIX" else "RECOVERY",
            "online_observation": {"rgb": str(rgb_path.resolve()), "wrist_rgb": str(wrist_path.resolve()),
                                   "proprio": online_proprio(obs, np)},
            "executed_action": [float(value) for value in action],
        })
        phases.append(phase)

    try:
        obs = env.reset(); obs = env.set_init_state(states[episode_id])
        for _ in range(args.wait_steps):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        for action in source[:args.fork_timestep]:
            record(obs, action, "NORMAL_PREFIX")
            obs, _, _, _ = env.step(action)
        fork_error = float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"], np.float32) - np.asarray(expected, np.float32)))
        if fork_error > args.max_fork_eef_error_m:
            raise RuntimeError(f"fork replay mismatch {fork_error:.4f}m > {args.max_fork_eef_error_m:.4f}m")
        for action in recovery:
            record(obs, action, "TEACHER_RECOVERY")
            obs, _, done, _ = env.step(action)
            if done:
                break
        if not bool(env.check_success()) and not done:
            for action in completion:
                record(obs, action, "TEACHER_COMPLETION")
                obs, _, done, _ = env.step(action)
                if done:
                    break
        success = bool(env.check_success())
        # Attach unambiguous action chunks only at the extractor's decision
        # boundaries and only when all eight actions are teacher recovery.
        for start in range(0, len(records) - 7, 8):
            if all(phase != "NORMAL_PREFIX" for phase in phases[start:start + 8]):
                records[start]["recovery_target_action_chunk"] = [
                    records[index]["executed_action"] for index in range(start, start + 8)
                ]
    finally:
        env.close()

    metadata = {
        "kind": "v3_book_prefix_canonical_teacher", "schema": "v3_canonical_teacher_v2",
        **{key: source_meta[key] for key in identity}, "task_language": language,
        "success": success, "transitions": len(records), "fork_timestep": args.fork_timestep,
        "fork_eef_error_m": fork_error, "source_rollout": str(args.source_rollout.resolve()),
        "recovery_branch": str(args.recovery_branch.resolve()),
        "completion_branch": str(args.completion_branch.resolve()),
        "teacher_only_branch_selection": True,
        "online_inputs_permitted": ["main_rgb", "wrist_rgb", "language", "proprio", "executed_action_history"],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "steps.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    if video:
        imageio.mimsave(args.output_dir / "rollout.mp4", video, fps=30)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
