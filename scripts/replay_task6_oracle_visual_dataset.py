#!/usr/bin/env python
"""Replay task-6 actions and export raw RGB with exact oracle predicate labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = REPO_ROOT / "experiments" / "robot" / "libero"
sys.path.insert(0, str(LIBERO_ROOT))
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", str(LIBERO_ROOT / "baseline_libero_config")
)

from run_openvla_baseline import extract_privileged_object_state, write_json  # noqa: E402

TASK_SUITE = "libero_10"
TASK_ID = 6


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--episode-ids", type=int, nargs="+", default=list(range(10))
    )
    parser.add_argument("--wait-steps", type=int, default=20)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-replay-error-m", type=float, default=0.002)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def max_object_position_error(obs: dict, recorded: dict) -> float:
    errors = []
    for key, expected in recorded.items():
        if not key.endswith("_pos") or key not in obs:
            continue
        errors.append(
            float(
                np.linalg.norm(
                    np.asarray(obs[key], np.float32)
                    - np.asarray(expected, np.float32)
                )
            )
        )
    return max(errors, default=0.0)


def main() -> None:
    args = arguments()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(TASK_ID)
    initial_states = suite.get_task_init_states(TASK_ID)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.jsonl"
    manifest_path.write_text("", encoding="utf-8")
    summary = []

    for episode_id in args.episode_ids:
        source = args.source_root / f"task6-episode{episode_id}" / "steps.jsonl"
        records = load_records(source)
        episode_dir = args.output_root / f"task6-episode{episode_id}"
        rgb_dir = episode_dir / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)

        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
        )
        env.seed(0)
        env.reset()
        obs = env.set_init_state(initial_states[episode_id])
        noop = np.asarray([0, 0, 0, 0, 0, 0, -1], np.float32)
        for _ in range(args.wait_steps):
            obs, _, done, _ = env.step(noop.tolist())

        max_error = 0.0
        labels = []
        try:
            for record in records:
                error = max_object_position_error(
                    obs, record["privileged_object_state_before"]
                )
                max_error = max(max_error, error)
                image = np.ascontiguousarray(
                    obs["agentview_image"][::-1, ::-1]
                )
                image_name = f"{record['timestep']:05d}.jpg"
                image_path = rgb_dir / image_name
                imageio.imwrite(
                    image_path,
                    image,
                    quality=args.jpeg_quality,
                )
                mug_on_plate = bool(
                    env.env._eval_predicate(
                        ["on", "porcelain_mug_1", "plate_1"]
                    )
                )
                row = {
                    "task_suite": TASK_SUITE,
                    "task_id": TASK_ID,
                    "episode_id": episode_id,
                    "timestep": record["timestep"],
                    "image": str(image_path.resolve()),
                    "mug_on_plate": mug_on_plate,
                    "oracle_switched": bool(record["oracle_switched"]),
                    "policy_instruction": record["policy_instruction"],
                    "source_steps_jsonl": str(source.resolve()),
                    "replay_object_position_error_m": error,
                    "privileged_object_state": extract_privileged_object_state(
                        obs
                    ),
                }
                labels.append(row)
                with manifest_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                obs, _, _, _ = env.step(record["executed_action"])
        finally:
            env.close()

        if max_error > args.max_replay_error_m:
            raise RuntimeError(
                f"episode {episode_id} replay drift {max_error:.6f} m exceeds "
                f"{args.max_replay_error_m:.6f} m"
            )
        episode_metadata = {
            "episode_id": episode_id,
            "num_frames": len(labels),
            "positive_frames": sum(row["mug_on_plate"] for row in labels),
            "max_replay_object_position_error_m": max_error,
            "source_steps_jsonl": str(source.resolve()),
        }
        write_json(episode_dir / "metadata.json", episode_metadata)
        summary.append(episode_metadata)
        print(json.dumps(episode_metadata), flush=True)

    write_json(
        args.output_root / "metadata.json",
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_suite": TASK_SUITE,
            "task_id": TASK_ID,
            "label": "exact simulator On(porcelain_mug_1, plate_1) predicate",
            "policy_input_excludes_oracle": True,
            "episodes": summary,
            "num_frames": sum(row["num_frames"] for row in summary),
            "manifest": str(manifest_path.resolve()),
        },
    )


if __name__ == "__main__":
    main()
