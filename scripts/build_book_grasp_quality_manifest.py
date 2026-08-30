#!/usr/bin/env python3
"""Build causal CLOSE/LIFT supervision for the book grasp-quality head.

Object contact and lift are teacher-only labels.  Every model input remains
the recorded dual cameras, EEF proprio, and the immediately preceding action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def split(source_key: str) -> str:
    # Source-level split avoids frames from one recovery fork entering both
    # train and validation.  It is an audit split, not a claim of broad task
    # generalization.
    bucket = int(hashlib.sha256(f"book-grasp-quality-v1::{source_key}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "val"


def main() -> None:
    args = arguments()
    if args.stride < 1:
        raise ValueError("--stride must be positive")
    records = []
    skipped = 0
    for root in args.branches_root:
        for metadata_path in sorted(root.rglob("metadata.json")):
            metadata = json.loads(metadata_path.read_text())
            steps_path = metadata_path.parent / "steps.jsonl"
            if metadata.get("kind") != "offline_book_regrasp_branch_v2" or not steps_path.is_file():
                continue
            fork = json.loads(Path(metadata["source_fork"]).joinpath("manifest.json").read_text())
            source_key = f"{fork['task_suite']}-task{fork['task_id']}-episode{fork['episode_id']}"
            rows = [json.loads(line) for line in steps_path.read_text().splitlines() if line]
            for index, row in enumerate(rows):
                teacher = row.get("teacher_only", {})
                # The quality decision is useful only after fingers close;
                # BACKOFF/ALIGN/DESCEND have no attachment label yet.
                if int(teacher.get("recovery_phase", -1)) < 3:
                    continue
                if "target_attached" not in teacher:
                    skipped += 1
                    continue
                previous = rows[index - 1].get("executed_action") if index else None
                records.append({
                    "schema_version": 1,
                    "kind": "book_grasp_quality_window",
                    "split": split(source_key),
                    "source_key": source_key,
                    "task_id": int(fork["task_id"]),
                    "branch_timestep": index,
                    "language": fork.get("task_language", "pick up the black book and place it in the caddy"),
                    "online_observation": {
                        **row["online_observation"],
                        "previous_executed_action": previous,
                    },
                    "grasp_attached_label": bool(teacher["target_attached"]),
                    "teacher_only": {
                        "target_lift_from_fork_m": float(teacher.get("target_lift_from_fork_m", 0.0)),
                        "target_quaternion_wxyz": teacher.get("target_quaternion_wxyz"),
                    },
                    "provenance": {"branch_metadata": str(metadata_path), "source_fork": metadata["source_fork"]},
                })
    if not records:
        raise RuntimeError("no dense attached labels found; recollect branches with the upgraded collector")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record) + "\n" for record in records))
    report = {
        "samples": len(records), "attached": sum(record["grasp_attached_label"] for record in records),
        "detached": sum(not record["grasp_attached_label"] for record in records),
        "skipped_legacy_unlabeled": skipped,
        "sources": sorted({record["source_key"] for record in records}),
        "splits": {name: sum(record["split"] == name for record in records) for name in ("train", "val")},
    }
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
