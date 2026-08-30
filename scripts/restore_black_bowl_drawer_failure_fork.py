#!/usr/bin/env python3
"""Restore an exact black-bowl/drawer failure state selected by the miner."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-manifest", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, default=Path("/data/lty/code/LIBERO-plus"))
    return parser.parse_args()


def main() -> None:
    args = arguments()
    payload = json.loads(args.failure_manifest.read_text())
    matches = [row for row in payload["records"] if int(row["task_id"]) == args.task_id]
    if len(matches) != 1:
        raise ValueError(f"expected one task{args.task_id} record, got {len(matches)}")
    record = matches[0]
    failure = record.get("selected_failure")
    if failure is None:
        raise ValueError(f"task{args.task_id} has no accepted selected_failure")
    failure_timestep = int(failure["timestep"])
    source_steps = Path(record["source_trajectory"])
    source_root = source_steps.parent
    rows = [json.loads(line) for line in source_steps.read_text().splitlines() if line.strip()]
    snapshot_steps = sorted(
        int(path.parent.name.removeprefix("step"))
        for path in source_root.glob("step*/mujoco_state.npz")
        if int(path.parent.name.removeprefix("step")) <= failure_timestep)
    if not snapshot_steps:
        raise FileNotFoundError(f"no snapshot at or before step {failure_timestep}")
    snapshot_timestep = snapshot_steps[-1]
    snapshot_path = source_root / f"step{snapshot_timestep:04d}" / "mujoco_state.npz"

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    sys.path[:0] = [str(args.libero_root), str(Path(__file__).resolve().parents[1])]
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[payload["task_suite"]]()
    task = suite.get_task(args.task_id)
    episode = int(record["episode"])
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(0); env.reset(); env.set_init_state(suite.get_task_init_states(args.task_id)[episode])
    try:
        obs = env.regenerate_obs_from_state(np.load(snapshot_path)["mujoco_state"])
        controller = env.env.robots[0].controller
        controller.update(force=True); controller.reset_goal()
        for timestep in range(snapshot_timestep, failure_timestep):
            if int(rows[timestep]["timestep"]) != timestep:
                raise ValueError("source trajectory is not timestep-indexed")
            obs, _, done, _ = env.step(rows[timestep]["executed_action"])
            if done:
                raise RuntimeError("source completed before selected failure")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.output_dir / "mujoco_state.npy", env.sim.get_state().flatten())
        imageio.imwrite(args.output_dir / "main_rgb.png",
                       np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        imageio.imwrite(args.output_dir / "wrist_rgb.png",
                       np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
        manifest = {
            "kind": "black_bowl_bottom_drawer_exact_failure_fork_v1",
            "teacher_only": True, "task_suite": payload["task_suite"],
            "task_id": args.task_id, "episode_id": episode, "language": record["language"],
            "failure_timestep": failure_timestep, "failure_type": failure["failure_type"],
            "target_role": failure["target_role"], "failure_evidence": failure["evidence"],
            "source_snapshot_timestep": snapshot_timestep,
            "source_snapshot": str(snapshot_path.resolve()),
            "source_trajectory": str(source_steps.resolve()),
            "actions_replayed_after_snapshot": failure_timestep - snapshot_timestep,
            "state_file": "mujoco_state.npy", "observation_timing": "pre_action_at_failure_timestep",
            "online_inputs_permitted": ["main_rgb", "wrist_rgb", "language", "proprio",
                                        "executed_action_history"],
            "teacher_only_fields": ["mujoco_state", "failure_evidence"],
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest))
    finally:
        env.close()


if __name__ == "__main__":
    main()
