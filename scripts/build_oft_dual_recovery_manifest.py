#!/usr/bin/env python3
"""Convert dual-camera branch trajectories into OFT recovery supervision.

The output deliberately contains only online inputs and teacher *labels*.
Each record predicts the next 8 actions in OFT action-head space from a causal
history ending at ``branch_timestep``.
"""

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


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def env_to_oft_action(action: list[float]) -> list[float]:
    """Undo OFT's LIBERO gripper post-processing; Cartesian dimensions match."""
    if len(action) != 7:
        raise ValueError(f"expected 7D action, got {len(action)}")
    converted = list(map(float, action))
    converted[-1] = (1.0 - converted[-1]) / 2.0
    return converted


def main() -> None:
    args = parse_args()
    if args.stride < 1 or args.chunk_length < 1:
        raise ValueError("stride and chunk-length must be positive")
    trajectories = sorted(args.input_root.glob("**/raw-transitions.jsonl"))
    if not trajectories:
        raise FileNotFoundError(f"no raw trajectories under {args.input_root}")

    records: list[dict] = []
    summaries: list[dict] = []
    for trajectory in trajectories:
        rows = load_rows(trajectory)
        if len(rows) < args.chunk_length:
            continue
        common = rows[0]
        if any(row["branch_timestep"] != index for index, row in enumerate(rows)):
            raise ValueError(f"non-contiguous branch timesteps: {trajectory}")
        for row in rows:
            observation = row["online_observation"]
            for key in ("rgb", "wrist_rgb"):
                if not observation.get(key) or not Path(observation[key]).is_file():
                    raise FileNotFoundError(f"missing {key} in {trajectory}")

        attached_ever = any(row["teacher_only"]["attached_after"] for row in rows)
        # A brief contact that is lost before the chunk ends is not a positive
        # recovery-action target. Keep it separately for future temporal
        # outcome supervision.
        branch_success = bool(rows[-1]["teacher_only"]["attached_after"])
        samples = 0
        for start in range(0, len(rows) - args.chunk_length + 1, args.stride):
            current = rows[start]
            target = [env_to_oft_action(row["online_observation"]["recovery_action"]) for row in rows[start : start + args.chunk_length]]
            # At the source failure the policy must enter RECOVERY.  Once an
            # object is already attached, the same data acts as a conservative
            # RECOVERY_COMPLETE supervision example rather than another grasp.
            gate_label = 2 if current["teacher_only"]["attached_before"] else 1
            records.append({
                "schema_version": 1,
                "kind": "oft_dual_camera_recovery_window",
                "task_suite": current["task_suite"],
                "task_id": current["task_id"],
                "source_episode_id": current["source_episode_id"],
                "branch_episode_id": current["branch_episode_id"],
                "failure_timestep": current["failure_timestep"],
                "candidate_name": current["candidate_name"],
                "branch_timestep": start,
                "history_end_observation": current["online_observation"],
                "recovery_target_oft_action_chunk": target,
                "gate_label": gate_label,
                "branch_eventually_regrasped": branch_success,
                "branch_attached_at_any_time": attached_ever,
                "teacher_label_only": {
                    "attached_before": current["teacher_only"]["attached_before"],
                    "attached_after_chunk": rows[min(start + args.chunk_length - 1, len(rows) - 1)]["teacher_only"]["attached_after"],
                },
                "provenance": {"raw_trajectory": str(trajectory)},
            })
            samples += 1
        summaries.append({
            "trajectory": str(trajectory),
            "task_id": common["task_id"],
            "source_episode_id": common["source_episode_id"],
            "candidate_name": common["candidate_name"],
            "steps": len(rows),
            "samples": samples,
            "eventually_regrasped": branch_success,
            "attached_at_any_time": attached_ever,
        })

    if not records:
        raise RuntimeError("no valid recovery windows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    report = {
        "windows": len(records),
        "trajectories": len(summaries),
        "eventually_regrasped_trajectories": sum(item["eventually_regrasped"] for item in summaries),
        "gate_counts": {str(label): sum(record["gate_label"] == label for record in records) for label in (1, 2)},
        "branches": summaries,
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
