#!/usr/bin/env python3
"""Convert a recorded successful OFT trajectory to strict replay input.

The recorded trajectory stores already-processed LIBERO gripper commands
(-1=open, +1=close), while collect_v4_action_trace_replay expects the policy
representation before process_action (1=open, 0=close).  This conversion is
explicit so strict replay reconstructs the exact executed 7-D actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline_report.read_text())
    if baseline.get("success") is not True:
        raise ValueError("only verified successful baseline trajectories are teachers")
    rows = [json.loads(line) for line in args.trajectory.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty trajectory")
    decisions = []
    for index, row in enumerate(rows):
        executed = row["executed_action"]
        if len(executed) != 7 or executed[-1] not in (-1.0, 1.0):
            raise ValueError(f"malformed executed action at row {index}")
        raw = list(executed)
        raw[-1] = (1.0 - raw[-1]) / 2.0
        decisions.append({
            "timestep": index,
            "mode": "RECOVERY",
            "action": raw,
            "online_observation": row.get("online_observation"),
        })
    payload = {
        "kind": "recorded_successful_oft_as_recovery_teacher_v1",
        "success": True,
        "task": baseline["task_description"],
        "task_id": baseline["task_id"],
        "episode_id": baseline["episode_id"],
        "source_trajectory": str(args.trajectory.resolve()),
        "source_baseline_report": str(args.baseline_report.resolve()),
        "decisions": decisions,
        "scope": "successful nominal OFT trace reused as full-action teacher",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "decisions": len(decisions)}))


if __name__ == "__main__":
    main()
