#!/usr/bin/env python3
"""Replay an auditable OFT rollout prefix and save an offline branch snapshot.

The saved MuJoCo state is a *teacher-only* artifact.  It is used to fork
counterfactual recovery branches offline; online policy inputs remain RGB,
wrist RGB, language, proprio, and executed-action history.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-steps", type=Path, required=True)
    parser.add_argument(
        "--dense-bundle", type=Path,
        help="Archival v4 dense-audit.npz; defaults to the source-steps sibling when present.",
    )
    parser.add_argument("--fork-timestep", type=int, required=True,
                        help="Snapshot immediately before this recorded action.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--libero-root", type=Path,
                        default=Path("/data/lty/code/LIBERO-plus"))
    parser.add_argument("--oft-root", type=Path,
                        default=Path("/data/lty/code/openvla-oft-official"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source_steps.is_file():
        raise FileNotFoundError(args.source_steps)
    if not (args.libero_root / "libero" / "libero").is_dir() or not args.oft_root.is_dir():
        raise FileNotFoundError(args.libero_root)
    if args.fork_timestep < 0:
        raise ValueError("fork timestep must be non-negative")

    rows = [json.loads(line) for line in args.source_steps.read_text().splitlines()]
    if args.fork_timestep >= len(rows):
        raise ValueError(
            f"fork timestep {args.fork_timestep} is outside {len(rows)} recorded actions"
        )
    for expected, row in enumerate(rows):
        if int(row["timestep"]) != expected:
            raise ValueError("source steps must have contiguous timesteps")
        if int(row["task_id"]) != args.task_id or int(row["episode_id"]) != args.episode_id:
            raise ValueError("source task/episode differs from requested fork")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    # Plus has to precede OFT's bundled legacy LIBERO package; OFT still
    # supplies the evaluator helper imported below.
    sys.path[:0] = [str(args.libero_root), str(args.oft_root), str(Path(__file__).resolve().parents[1])]
    os.chdir(args.oft_root)

    import imageio.v2 as imageio
    import numpy as np
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        get_libero_wrist_image,
    )
    from experiments.robot.libero.run_libero_eval import GenerateConfig

    # This config is only used to reproduce the exact LIBERO warm-up action
    # convention.  No policy inference occurs in this script.
    cfg = GenerateConfig(task_suite_name=args.task_suite, num_trials_per_task=1)
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.episode_id < len(initial_states):
        raise ValueError("invalid episode id")
    env, language = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    dense_path = args.dense_bundle
    if dense_path is None:
        candidate = args.source_steps.with_name("dense-audit.npz")
        dense_path = candidate if candidate.is_file() else None
    dense = None
    if dense_path is not None:
        if not dense_path.is_file():
            raise FileNotFoundError(dense_path)
        with np.load(dense_path, allow_pickle=False) as bundle:
            dense = {key: bundle[key] for key in bundle.files}
        required = {
            "official_initial_state", "mujoco_state", "numpy_rng_keys",
            "numpy_rng_position", "numpy_rng_has_gaussian", "numpy_rng_cached_gaussian",
        }
        if not required.issubset(dense):
            raise ValueError(f"archival bundle is missing {sorted(required - set(dense))}")

    try:
        env.reset()
        initial_state = (
            initial_states[args.episode_id]
            if dense is None else dense["official_initial_state"]
        )
        obs = env.set_init_state(initial_state)
        for _ in range(cfg.num_steps_wait):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        max_prefix_state_error = 0.0
        for index, row in enumerate(rows[:args.fork_timestep]):
            if dense is not None:
                current = env.sim.get_state().flatten()
                error = float(np.max(np.abs(current - dense["mujoco_state"][index])))
                max_prefix_state_error = max(max_prefix_state_error, error)
                if error > 1e-10:
                    raise RuntimeError(
                        f"archival prefix diverged at step {index}: max state error {error}"
                    )
                np.random.set_state((
                    "MT19937", dense["numpy_rng_keys"][index],
                    int(dense["numpy_rng_position"][index]),
                    int(dense["numpy_rng_has_gaussian"][index]),
                    float(dense["numpy_rng_cached_gaussian"][index]),
                ))
            action = np.asarray(row["executed_action"], dtype=np.float32)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                raise RuntimeError("source prefix completed before requested fork")
        if dense is not None:
            fork_error = float(np.max(np.abs(
                env.sim.get_state().flatten() - dense["mujoco_state"][args.fork_timestep]
            )))
            max_prefix_state_error = max(max_prefix_state_error, fork_error)
            if fork_error > 1e-10:
                raise RuntimeError(
                    f"archival prefix diverged at fork: max state error {fork_error}"
                )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        state = env.sim.get_state().flatten()
        np.savez_compressed(
            args.output_dir / "mujoco_state.npz",
            mujoco_state=state,
            fork_timestep=np.asarray(args.fork_timestep, dtype=np.int32),
        )
        imageio.imwrite(args.output_dir / "main_rgb.png", get_libero_image(obs))
        imageio.imwrite(args.output_dir / "wrist_rgb.png", get_libero_wrist_image(obs))
        manifest = {
            "kind": "offline_recovery_fork_snapshot_v1",
            "teacher_only": True,
            "task_suite": args.task_suite,
            "task_id": args.task_id,
            "episode_id": args.episode_id,
            "task_language": language,
            "fork_timestep": args.fork_timestep,
            "source_steps": str(args.source_steps.resolve()),
            "dense_bundle": None if dense_path is None else str(dense_path.resolve()),
            "exact_archival_prefix_validated": dense is not None,
            "max_prefix_state_error": max_prefix_state_error if dense is not None else None,
            "state_file": "mujoco_state.npz",
            "online_inputs_permitted": [
                "main_rgb", "wrist_rgb", "language", "proprio", "executed_action_history"
            ],
            "teacher_only_fields": ["mujoco_state"],
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(json.dumps(manifest, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
