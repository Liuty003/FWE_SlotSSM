#!/usr/bin/env python3
"""Select comparable regrasp forks from failed Robot-initial-state OFT rollouts.

The selector uses only the nominal policy's visible action timeline: it chooses
the starts of its first few close-gripper attempts, then maps them to saved
pre-action forks.  This deliberately avoids task-id-specific recovery times.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--replays-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-forks-per-task", type=int, default=3)
    return parser.parse_args()


def nearest_snapshot(step: int, available: list[int]) -> int:
    return min(available, key=lambda candidate: (abs(candidate - step), candidate))


def main() -> None:
    args = arguments()
    if args.max_forks_per_task < 1:
        raise ValueError("--max-forks-per-task must be positive")
    baseline = json.loads(args.baseline_summary.read_text())["rows"]
    failures = [row for row in baseline if not row["success"]]
    selected = []
    for row in failures:
        task_id, episode_id = int(row["task_id"]), int(row["episode_id"])
        source_dir = args.replays_root / "forks" / f"task{task_id}-episode{episode_id}"
        trajectory_path = source_dir / "trajectory.jsonl"
        if not trajectory_path.is_file():
            raise FileNotFoundError(trajectory_path)
        snapshots = sorted(
            int(path.parent.name.removeprefix("step"))
            for path in source_dir.glob("step*/mujoco_state.npz")
        )
        if not snapshots:
            raise RuntimeError(f"no snapshots for task {task_id}")
        trajectory = [json.loads(line) for line in trajectory_path.read_text().splitlines() if line]
        close_starts = []
        was_closed = False
        for event in trajectory:
            # LIBERO convention after OFT's official action processing:
            # positive is close, negative is open.
            closed = float(event["executed_action"][-1]) > 0.0
            if closed and not was_closed:
                close_starts.append(int(event["policy_step"]))
            was_closed = closed
        if not close_starts:
            close_starts = [64, 112, 160]
        reasons = []
        for close_step in close_starts:
            reasons.extend(
                [
                    (max(0, close_step - 16), "pre_close"),
                    (close_step, "close_start"),
                    (close_step + 16, "post_close"),
                ]
            )
        task_choices = []
        for desired_step, reason in reasons:
            chosen_step = nearest_snapshot(desired_step, snapshots)
            if chosen_step in {record["fork_timestep"] for record in task_choices}:
                continue
            task_choices.append(
                {
                    "task_suite": "libero_10",
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "fork_timestep": chosen_step,
                    "fork_dir": str((source_dir / f"step{chosen_step:04d}").resolve()),
                    "selection_rule": reason,
                    "nominal_close_step": desired_step if reason != "pre_close" else desired_step + 16,
                }
            )
            if len(task_choices) >= args.max_forks_per_task:
                break
        selected.extend(task_choices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record) + "\n" for record in selected))
    report = {
        "failed_tasks": len(failures),
        "selected_forks": len(selected),
        "per_task": {str(row["task_id"]): sum(r["task_id"] == row["task_id"] for r in selected) for row in failures},
    }
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
