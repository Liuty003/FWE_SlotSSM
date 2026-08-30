#!/usr/bin/env python3
"""Build the canonical 393-task LIBERO-Plus Robot evaluation manifest.

LIBERO-Plus classification IDs are one-based while ``task_id`` used by the
benchmark and rollout filenames is zero-based.  Keeping that conversion in a
single script prevents silently assigning a rollout to the wrong difficulty.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classification", type=Path,
        default=Path("/data/lty/code/LIBERO-plus/libero/libero/benchmark/task_classification.json"),
    )
    parser.add_argument(
        "--baseline-root", type=Path, action="append",
        default=[
            Path("/data2/lty/naive-e-slotssm-runs/libero-plus/oft-plus-robot-levels1-4-v1"),
            Path("/data2/lty/naive-e-slotssm-runs/libero-plus/oft-plus-robot-level5-v1"),
        ],
        help="Earlier roots take precedence for duplicate task rollouts.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/oft-plus-robot-global-v1/robot_manifest.json"),
    )
    return parser.parse_args()


def task_id_from_path(path: Path) -> int | None:
    match = re.search(r"libero_10-task(\d+)-episode0\.json$", path.name)
    return None if match is None else int(match.group(1))


def semantic_task_name(name: str) -> str:
    """Strip LIBERO-Plus viewpoint/initial-state perturbation suffixes."""
    result = re.sub(r"_view_\d+_\d+_\d+_\d+_\d+_initstate_\d+$", "", name)
    if result == name:
        raise ValueError(f"unrecognized Robot task name format: {name}")
    return result


def main() -> None:
    args = parse_args()
    classification = json.loads(args.classification.read_text())["libero_10"]
    robot = {
        entry["id"] - 1: entry
        for entry in classification
        if entry["category"] == "Robot Initial States"
    }
    if len(robot) != 393:
        raise RuntimeError(f"expected 393 Robot Initial States tasks, got {len(robot)}")

    baseline = {}
    for root in args.baseline_root:
        for path in sorted(root.rglob("libero_10-task*-episode0.json")):
            task_id = task_id_from_path(path)
            if task_id not in robot or task_id in baseline:
                continue
            record = json.loads(path.read_text())
            baseline[task_id] = {
                "success": bool(record["success"]),
                "source_json": str(path),
                "source_video": record.get("video"),
            }
    missing = sorted(set(robot) - set(baseline))
    if missing:
        raise RuntimeError(f"missing baseline rollouts for task IDs: {missing}")

    semantic_names = sorted({semantic_task_name(entry["name"]) for entry in robot.values()})
    semantic_id = {name: index for index, name in enumerate(semantic_names)}
    records = []
    per_level = defaultdict(list)
    for task_id, entry in sorted(robot.items()):
        level = int(entry["difficulty_level"])
        semantic_name = semantic_task_name(entry["name"])
        record = {
            "task_id": task_id,
            "episode_id": 0,
            "classification_id": int(entry["id"]),
            "difficulty_level": level,
            "category": entry["category"],
            "name": entry["name"],
            "semantic_task_id": semantic_id[semantic_name],
            "semantic_task": semantic_name,
            "oft_plus_baseline": baseline[task_id],
        }
        records.append(record)
        per_level[level].append(record)
    summary = {
        str(level): {
            "total": len(rows),
            "successes": sum(row["oft_plus_baseline"]["success"] for row in rows),
            "success_rate": sum(row["oft_plus_baseline"]["success"] for row in rows) / len(rows),
        }
        for level, rows in sorted(per_level.items())
    }
    payload = {
        "kind": "libero_plus_robot_global_manifest_v1",
        "task_id_convention": "task_id = classification_id - 1",
        "num_tasks": len(records),
        "difficulty_counts": dict(sorted(Counter(row["difficulty_level"] for row in records).items())),
        "semantic_tasks": [
            {
                "semantic_task_id": semantic_id[name],
                "semantic_task": name,
                "num_robot_initial_states": sum(row["semantic_task"] == name for row in records),
            }
            for name in semantic_names
        ],
        "oft_plus_baseline_summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
