#!/usr/bin/env python
"""Build an episode-level manifest for recovery expert transitions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--validation-episodes", default="1",
        help="Comma-separated episode IDs reserved for validation.",
    )
    parser.add_argument(
        "--test-episodes", default="",
        help="Comma-separated episode IDs reserved for test.",
    )
    return parser.parse_args()


def ids(text):
    return {int(value) for value in text.split(",") if value.strip()}


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    args = arguments()
    validation_ids, test_ids = ids(args.validation_episodes), ids(args.test_episodes)
    if validation_ids & test_ids:
        raise ValueError("validation and test episode IDs overlap")
    records = []
    for directory in args.episodes:
        transitions = directory / "transitions.jsonl"
        rows = load_rows(transitions)
        if not rows:
            raise RuntimeError(f"empty episode: {directory}")
        episode_id = int(rows[0]["episode_id"])
        split = (
            "test" if episode_id in test_ids
            else "validation" if episode_id in validation_ids
            else "train"
        )
        phases = Counter(row["phase"] for row in rows)
        expert_rows = [row for row in rows if row["is_recovery_expert_action"]]
        recovery_completed = any(row["recovery_completed"] for row in rows)
        final_task_success = bool(rows[-1]["success_after"])
        missing_files = []
        for row in rows:
            for key in ("rgb", "depth", "target_mask_64"):
                value = row.get(key)
                if value is not None and not Path(value).exists():
                    missing_files.append(value)
        records.append({
            "task_suite": rows[0]["task_suite"],
            "task_id": int(rows[0]["task_id"]),
            "episode_id": episode_id,
            "split": split,
            "directory": str(directory.resolve()),
            "transitions": str(transitions.resolve()),
            "num_rows": len(rows),
            "num_expert_rows": len(expert_rows),
            "phase_counts": dict(phases),
            "failure_detected": any(
                row["is_failure_detection_event"] for row in rows
            ),
            "recovery_completed": recovery_completed,
            "final_task_success": final_task_success,
            "eligible_for_recovery_bc": (
                recovery_completed and bool(expert_rows) and not missing_files
            ),
            "eligible_for_end_to_end_success_bc": (
                recovery_completed and final_task_success and not missing_files
            ),
            "missing_file_count": len(missing_files),
        })
    split_counts = Counter(record["split"] for record in records)
    result = {
        "schema_version": 1,
        "purpose": "oracle recovery behavior cloning and DAgger",
        "split_unit": "episode",
        "episodes": records,
        "summary": {
            "num_episodes": len(records),
            "split_counts": dict(split_counts),
            "num_recovery_bc_eligible": sum(
                record["eligible_for_recovery_bc"] for record in records
            ),
            "num_end_to_end_success_eligible": sum(
                record["eligible_for_end_to_end_success_bc"] for record in records
            ),
            "num_expert_rows": sum(record["num_expert_rows"] for record in records),
            "num_eligible_expert_rows": sum(
                record["num_expert_rows"]
                for record in records if record["eligible_for_recovery_bc"]
            ),
            "has_nonempty_train_validation_test": all(
                split_counts[name] > 0 for name in ("train", "validation", "test")
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
