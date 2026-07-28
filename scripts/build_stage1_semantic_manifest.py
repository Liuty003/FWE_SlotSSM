#!/usr/bin/env python
"""Build episode-isolated Stage-B semantic Slot training manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROLE_TO_INSTANCE = {
    "target": "akita_black_bowl_1",
    "distractor": "akita_black_bowl_2",
    "goal": "plate_1",
    "robot": "robot",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/stage-b-task9-instance-data"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/stage-b-task9-instance-data/manifest.json"),
    )
    parser.add_argument("--train-episodes", default="0,1,2,5")
    parser.add_argument("--validation-episodes", default="3,6,8")
    parser.add_argument("--test-episodes", default="4,7,9")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--min-visible-pixels", type=int, default=16)
    return parser.parse_args()


def episode_ids(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    args = arguments()
    if args.window < 2 or args.stride < 1:
        raise ValueError("window must be >=2 and stride must be >=1")
    splits = {
        "train": episode_ids(args.train_episodes),
        "validation": episode_ids(args.validation_episodes),
        "test": episode_ids(args.test_episodes),
    }
    flattened = [episode for episodes in splits.values() for episode in episodes]
    if len(flattened) != len(set(flattened)):
        raise ValueError(f"Episode leakage across splits: {splits}")

    manifest = {
        "schema_version": 1,
        "task_suite": "libero_spatial",
        "task_id": 9,
        "instruction": "pick up the black bowl on the wooden cabinet and place it on the plate",
        "role_to_instance": ROLE_TO_INSTANCE,
        "slot_contract": {"target": 0, "distractor": 1, "goal": 2, "robot": 3},
        "window": args.window,
        "stride": args.stride,
        "splits": {},
    }
    for split, episodes in splits.items():
        windows = []
        episode_reports = []
        for episode in episodes:
            directory = args.input_dir / f"task9-episode{episode}"
            report_path = directory / "report.json"
            if not report_path.is_file():
                raise FileNotFoundError(f"Missing episode report: {report_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if not report.get("passed_replay_check"):
                raise RuntimeError(f"Replay check failed for episode {episode}")
            frames = report["frames"]
            max_eef_error = max(frame["eef_replay_error_m"] for frame in frames)
            max_object_error = max(
                max(frame["object_replay_error_m"].values(), default=0.0) for frame in frames
            )
            episode_reports.append({
                "episode": episode,
                "num_frames": len(frames),
                "max_eef_replay_error_m": max_eef_error,
                "max_object_replay_error_m": max_object_error,
            })
            starts = list(range(0, max(1, len(frames) - args.window + 1), args.stride))
            final_start = max(0, len(frames) - args.window)
            if final_start not in starts:
                starts.append(final_start)
            for start in sorted(set(starts)):
                selected = frames[start : start + args.window]
                if len(selected) != args.window:
                    continue
                visible = {
                    role: all(
                        frame["visible_pixels"].get(instance, 0) >= args.min_visible_pixels
                        for frame in selected
                    )
                    for role, instance in ROLE_TO_INSTANCE.items()
                }
                windows.append({
                    "episode": episode,
                    "start": start,
                    "stop": start + args.window,
                    "rgb_pattern": str(directory / "frame-{frame:03d}-rgb.png"),
                    "segmentation_pattern": str(directory / "frame-{frame:03d}-segmentation.png"),
                    "role_visible_entire_window": visible,
                })
        manifest["splits"][split] = {
            "episodes": episodes,
            "episode_reports": episode_reports,
            "num_windows": len(windows),
            "windows": windows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "windows": {split: data["num_windows"] for split, data in manifest["splits"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
