#!/usr/bin/env python
"""Replay current LIBERO-10 task-9 rollouts for role-mask supervision only.

The resulting masks are simulator teacher labels for training/evaluation.  They
are never part of the online SlotSSM or recovery-policy input contract.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = REPO_ROOT / "experiments" / "robot" / "libero"
sys.path.insert(0, str(LIBERO_ROOT))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_ROOT / "baseline_libero_config"))

from run_openvla_baseline import write_json  # noqa: E402


TASK_SUITE = "libero_10"
TASK_ID = 9
ROLES = ("white_yellow_mug_1", "microwave_1", "porcelain_mug_1")
MASK_KEYS = ("target_mug_masks", "microwave_masks", "distractor_mug_masks")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-ids", type=int, nargs="+", required=True)
    parser.add_argument("--wait-steps", type=int, default=20)
    parser.add_argument("--mask-size", type=int, default=64)
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Use at most this many aligned source frames per episode (0 keeps all).",
    )
    parser.add_argument("--max-replay-error-m", type=float, default=.002)
    parser.add_argument(
        "--stop-on-drift", action=argparse.BooleanOptionalAction, default=True,
        help="Keep only the verified prefix when replay diverges from the source trace.",
    )
    return parser.parse_args()


def segmentation_key(observation: dict) -> str:
    keys = [
        key for key in observation
        if key.startswith("agentview_segmentation") and "eye_in_hand" not in key
    ]
    if len(keys) != 1:
        raise RuntimeError(f"expected one agentview segmentation key, got {keys}")
    return keys[0]


def position_error(observation: dict, recorded: dict) -> float:
    errors = []
    for key, expected in recorded.items():
        if key.endswith("_pos") and key in observation:
            errors.append(float(np.linalg.norm(
                np.asarray(observation[key], np.float32) - np.asarray(expected, np.float32)
            )))
    return max(errors, default=0.)


def resize_masks(masks: list[np.ndarray], size: int) -> torch.Tensor:
    tensor = torch.from_numpy(np.stack(masks).astype(np.float32))[:, None]
    return F.interpolate(tensor, size=(size, size), mode="nearest")[:, 0].bool()


def main() -> None:
    args = arguments()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(TASK_ID)
    states = suite.get_task_init_states(TASK_ID)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for episode_id in args.episode_ids:
        source = args.source_root / f"task9-episode{episode_id}" / "steps.jsonl"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
        if args.max_frames:
            rows = rows[:args.max_frames]
        env = SegmentationRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(0)
        env.reset()
        observation = env.set_init_state(states[episode_id])
        noop = np.asarray([0, 0, 0, 0, 0, 0, -1], np.float32)
        for _ in range(args.wait_steps):
            observation, _, _, _ = env.step(noop.tolist())
        missing = set(ROLES) - set(env.instance_to_id)
        if missing:
            env.close()
            raise RuntimeError(f"missing segmentation roles: {sorted(missing)}")
        role_masks = {role: [] for role in ROLES}
        max_error = 0.
        first_drift_timestep = None
        try:
            for index, row in enumerate(rows):
                current_error = position_error(
                    observation, row["privileged_object_state_before"]
                )
                max_error = max(max_error, current_error)
                if current_error > args.max_replay_error_m:
                    first_drift_timestep = int(row["timestep"])
                    if args.stop_on_drift:
                        print(
                            f"episode={episode_id} stopping before timestep="
                            f"{first_drift_timestep} drift={current_error:.6f}",
                            flush=True,
                        )
                        break
                    raise RuntimeError(
                        f"episode {episode_id} drift at timestep "
                        f"{first_drift_timestep}={current_error:.6f} m"
                    )
                segmentation = np.asarray(observation[segmentation_key(observation)]).squeeze()
                segmentation = np.ascontiguousarray(segmentation[::-1, ::-1])
                for role in ROLES:
                    role_masks[role].append(segmentation == env.instance_to_id[role])
                observation, _, _, _ = env.step(row["executed_action"])
                if index % 100 == 0 or index + 1 == len(rows):
                    print(
                        f"episode={episode_id} replayed={index + 1}/{len(rows)} "
                        f"max_error_m={max_error:.6f}", flush=True
                    )
        finally:
            env.close()
        if not role_masks[ROLES[0]]:
            raise RuntimeError(f"episode {episode_id} has no verified replay frames")
        tensors = {
            key: resize_masks(role_masks[role], args.mask_size)
            for role, key in zip(ROLES, MASK_KEYS)
        }
        tensors["timesteps"] = torch.arange(len(rows), dtype=torch.int64)
        output = args.output_dir / f"task9-episode{episode_id}-role-masks.safetensors"
        save_file(tensors, output)
        summary = {
            "episode_id": episode_id,
            "frames": len(rows),
            "visible_frames": {
                role: int(tensors[key].flatten(1).any(1).sum())
                for role, key in zip(ROLES, MASK_KEYS)
            },
            "max_replay_object_position_error_m": max_error,
            "first_drift_timestep": first_drift_timestep,
            "replay_complete": first_drift_timestep is None,
            "file": str(output),
        }
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    write_json(args.output_dir / "metadata.json", {
        "schema_version": 1,
        "task_suite": TASK_SUITE,
        "task_id": TASK_ID,
        "role_order": list(ROLES),
        "mask_keys": list(MASK_KEYS),
        "mask_size": args.mask_size,
        "privileged_masks_are_supervision_only": True,
        "episodes": summaries,
    })


if __name__ == "__main__":
    main()
