#!/usr/bin/env python
"""Extract frame-aligned RGB images from saved natural-baseline rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episode-ids", type=int, nargs="+", required=True)
    return parser.parse_args()


def main():
    args = arguments()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for episode in args.episode_ids:
        source = (
            args.source_root / f"task{args.task_id}"
            / f"task{args.task_id}-episode{episode}"
        )
        steps = [
            json.loads(line)
            for line in (source / "steps.jsonl").read_text().splitlines()
        ]
        reader = imageio.get_reader(source / "rollout.mp4")
        output = args.output_root / f"task{args.task_id}-episode{episode}"
        output.mkdir(parents=True, exist_ok=True)
        count = 0
        try:
            for index, frame in enumerate(reader):
                if index >= len(steps):
                    raise ValueError(
                        f"video has more frames than steps for episode {episode}"
                    )
                timestep = int(steps[index]["timestep"])
                if timestep != index:
                    raise ValueError(
                        f"non-contiguous timestep {timestep} at frame {index}"
                    )
                path = output / f"frame-{index + 1:06d}.png"
                imageio.imwrite(path, frame)
                manifest.append({
                    "task_id": args.task_id,
                    "episode_id": episode,
                    "timestep": timestep,
                    "image": str(path),
                })
                count += 1
        finally:
            reader.close()
        if count != len(steps):
            raise ValueError(
                f"episode {episode}: extracted {count} frames for {len(steps)} steps"
            )
        print(json.dumps({
            "task_id": args.task_id,
            "episode_id": episode,
            "frames": count,
        }), flush=True)
    (args.output_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in manifest) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "source_root": str(args.source_root),
        "task_id": args.task_id,
        "episode_ids": args.episode_ids,
        "frames": len(manifest),
        "alignment": "video frame index equals policy timestep",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
