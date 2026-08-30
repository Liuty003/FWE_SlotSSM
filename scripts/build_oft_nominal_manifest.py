#!/usr/bin/env python3
"""Convert successful recorded OFT rollouts into causal NORMAL windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--chunk-length", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[dict] = []
    for transition_path in sorted(args.input_root.glob("task*-episode*/transitions.jsonl")):
        rows = [json.loads(line) for line in transition_path.read_text().splitlines() if line]
        if not rows:
            continue
        metadata = json.loads((transition_path.parent / "metadata.json").read_text())
        if not metadata.get("success"):
            continue
        for start in range(0, len(rows) - args.chunk_length + 1, args.stride):
            target = [rows[index]["oft_action_chunk_start"] for index in range(start, start + args.chunk_length)]
            records.append({
                "kind": "oft_nominal_window", "task_suite": rows[start]["task_suite"],
                "task_id": rows[start]["task_id"], "gate_label": 0,
                "branch_eventually_regrasped": False,
                "branch_timestep": start,
                "recovery_target_oft_action_chunk": target,
                "provenance": {"raw_trajectory": str(transition_path.resolve()), "task_language": metadata["task_language"], "episode_id": rows[start]["episode_id"]},
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for record in records:
            out.write(json.dumps(record) + "\n")
    print(json.dumps({"windows": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
