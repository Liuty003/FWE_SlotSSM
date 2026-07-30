#!/usr/bin/env python
"""Replay task-6 trajectories and export cup/plate masks for supervision."""

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
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", str(LIBERO_ROOT / "baseline_libero_config")
)

from run_openvla_baseline import write_json  # noqa: E402

TASK_SUITE = "libero_10"
TASK_ID = 6
ROLES = ("porcelain_mug_1", "plate_1", "chocolate_pudding_1")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-ids", type=int, nargs="+", required=True)
    parser.add_argument("--wait-steps", type=int, default=20)
    parser.add_argument("--mask-size", type=int, default=64)
    parser.add_argument("--max-replay-error-m", type=float, default=0.002)
    return parser.parse_args()


def segmentation_key(obs):
    keys = [
        key for key in obs
        if key.startswith("agentview_segmentation") and "eye_in_hand" not in key
    ]
    if len(keys) != 1:
        raise RuntimeError(f"expected one agentview segmentation key, got {keys}")
    return keys[0]


def position_error(obs, recorded):
    errors = []
    for key, expected in recorded.items():
        if key.endswith("_pos") and key in obs:
            errors.append(
                float(
                    np.linalg.norm(
                        np.asarray(obs[key], np.float32)
                        - np.asarray(expected, np.float32)
                    )
                )
            )
    return max(errors, default=0.0)


def resize_masks(masks, size):
    tensor = torch.from_numpy(np.stack(masks).astype(np.float32))[:, None]
    return F.interpolate(tensor, size=(size, size), mode="nearest")[:, 0].bool()


def main():
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
        source = args.source_root / f"task6-episode{episode_id}" / "steps.jsonl"
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        env = SegmentationRenderEnv(
            bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
        )
        env.seed(0)
        env.reset()
        obs = env.set_init_state(states[episode_id])
        noop = np.asarray([0, 0, 0, 0, 0, 0, -1], np.float32)
        for _ in range(args.wait_steps):
            obs, _, _, _ = env.step(noop.tolist())
        missing = set(ROLES) - set(env.instance_to_id)
        if missing:
            raise RuntimeError(f"missing segmentation roles: {sorted(missing)}")
        role_masks = {role: [] for role in ROLES}
        labels = []
        max_error = 0.0
        try:
            for row in rows:
                max_error = max(
                    max_error,
                    position_error(obs, row["privileged_object_state_before"]),
                )
                segmentation = np.asarray(obs[segmentation_key(obs)]).squeeze()
                segmentation = np.ascontiguousarray(segmentation[::-1, ::-1])
                for role in ROLES:
                    role_masks[role].append(
                        segmentation == env.instance_to_id[role]
                    )
                labels.append(bool(row["mug_on_plate_oracle"]))
                obs, _, _, _ = env.step(row["executed_action"])
        finally:
            env.close()
        if max_error > args.max_replay_error_m:
            raise RuntimeError(
                f"episode {episode_id} drift={max_error:.6f} m"
            )
        tensors = {
            "mug_masks": resize_masks(
                role_masks["porcelain_mug_1"], args.mask_size
            ),
            "plate_masks": resize_masks(role_masks["plate_1"], args.mask_size),
            "pudding_masks": resize_masks(
                role_masks["chocolate_pudding_1"], args.mask_size
            ),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "timesteps": torch.arange(len(rows), dtype=torch.int64),
        }
        output = args.output_dir / f"task6-episode{episode_id}-role-masks.safetensors"
        save_file(tensors, output)
        summary = {
            "episode_id": episode_id,
            "frames": len(rows),
            "mug_visible_frames": int(tensors["mug_masks"].flatten(1).any(1).sum()),
            "plate_visible_frames": int(
                tensors["plate_masks"].flatten(1).any(1).sum()
            ),
            "max_replay_object_position_error_m": max_error,
            "file": str(output),
        }
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    write_json(
        args.output_dir / "metadata.json",
        {
            "schema_version": 1,
            "task_suite": TASK_SUITE,
            "task_id": TASK_ID,
            "roles": list(ROLES),
            "mask_size": args.mask_size,
            "privileged_masks_are_supervision_only": True,
            "episodes": summaries,
        },
    )


if __name__ == "__main__":
    main()
