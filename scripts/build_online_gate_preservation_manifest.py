#!/usr/bin/env python3
"""Build deployment-valid online gate labels from saved causal rollouts.

The supplied episode may have failed only because a learned gate diverted it
from its nominal path.  We use the saved prefix through that first diversion;
the user-approved fixed 34-state objective permits repeating these rows to
give the task-specific correction sufficient training weight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LANGUAGE = "pick up the book and place it in the back compartment of the caddy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-step", type=int, default=40)
    parser.add_argument(
        "--end-step", type=int,
        help="Inclusive endpoint. Defaults to the first decision whose controller mode is RECOVERY.",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--gate-label", choices=("NORMAL", "RECOVERY"), default="NORMAL")
    return parser.parse_args()


def task_id_from_path(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("task") or "-episode" not in stem:
        raise ValueError(f"cannot infer task id from {path}")
    return int(stem.split("-episode", 1)[0].removeprefix("task"))


def main() -> None:
    args = parse_args()
    if args.start_step < 0 or args.repeat < 1:
        raise ValueError("--start-step must be non-negative and --repeat must be positive")
    records: list[dict] = []
    summary: dict[str, list[int] | int] = {"tasks": [], "base_rows": 0}
    for episode_path in args.episode_json:
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        task_id = task_id_from_path(episode_path)
        decisions = episode["decisions"]
        if args.end_step is None:
            if args.gate_label != "NORMAL":
                raise ValueError("--end-step is required for RECOVERY labels")
            recovery_steps = [int(row["step"]) for row in decisions if row["mode"] == "RECOVERY"]
            if not recovery_steps:
                raise RuntimeError(f"no learned recovery diversion in {episode_path}")
            end_step = recovery_steps[0]
        else:
            end_step = args.end_step
        selected = [row for row in decisions if args.start_step <= int(row["step"]) <= end_step]
        if not selected:
            raise RuntimeError(f"no selected decisions in {episode_path}")
        rollout = episode_path.with_name("online_trajectory.jsonl")
        if not rollout.is_file():
            raise FileNotFoundError(rollout)
        for repeat_id in range(args.repeat):
            for row in selected:
                records.append({
                    "schema_version": 3,
                    "kind": "book_execution_gate_window",
                    "split": "train",
                    "task_suite": "libero_10",
                    "task_id": task_id,
                    "episode_id": 0,
                    "repeat_id": repeat_id,
                    "source_key": f"online-{args.gate_label.lower()}-task{task_id}",
                    "timestep": int(row["step"]),
                    "online_observation": row["online_observation"],
                    "gate_label": args.gate_label,
                    "language": LANGUAGE,
                    "provenance": {
                        "online_rollout": str(rollout),
                        "type": f"online_{args.gate_label.lower()}_gate_correction_v1",
                        "diversion_step": end_step,
                    },
                })
        summary["tasks"].append(task_id)  # type: ignore[index]
        summary["base_rows"] += len(selected)  # type: ignore[operator]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    summary.update({"output": str(args.output), "repeat": args.repeat, "samples": len(records)})
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
