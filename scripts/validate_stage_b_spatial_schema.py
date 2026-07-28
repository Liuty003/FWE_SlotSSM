#!/usr/bin/env python
"""Validate the Stage-B relation schema against LIBERO Spatial BDDL files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = REPO_ROOT / "experiments" / "robot" / "libero"
sys.path.insert(0, str(LIBERO_ROOT))
os.environ["LIBERO_CONFIG_PATH"] = str(LIBERO_ROOT / "baseline_libero_config")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema", type=Path,
        default=LIBERO_ROOT / "stage_b_spatial_role_schema.json",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[schema["task_suite"]]()
    if set(schema["tasks"]) != {str(index) for index in range(suite.n_tasks)}:
        raise RuntimeError("Schema task IDs do not exactly cover the benchmark")
    shared = schema["shared_roles"]
    rows = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        spec = schema["tasks"][str(task_id)]
        bddl_path = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        text = bddl_path.read_text(encoding="utf-8")
        predicate = spec["target_init_predicate"]
        if predicate not in text:
            raise RuntimeError(f"task {task_id}: target predicate absent: {predicate}")
        for instance in (
            shared["target"], shared["distractor"], shared["goal"], *spec["references"]
        ):
            if instance != "robot" and instance not in text:
                raise RuntimeError(f"task {task_id}: instance absent from BDDL: {instance}")
        counterfactual = spec["counterfactual"]
        if counterfactual["init_predicate"] not in text:
            raise RuntimeError(
                f"task {task_id}: counterfactual predicate absent: "
                f"{counterfactual['init_predicate']}"
            )
        for instance in (counterfactual["target"], *counterfactual["references"]):
            if instance not in text:
                raise RuntimeError(
                    f"task {task_id}: counterfactual instance absent: {instance}"
                )
        row = {
            "task_id": task_id,
            "language": task.language,
            "relation": spec["relation"],
            "references": spec["references"],
            "target_predicate": predicate,
            "counterfactual": counterfactual,
            "bddl": str(bddl_path),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    print(json.dumps({
        "valid": True,
        "num_tasks": len(rows),
        "relations": sorted({row["relation"] for row in rows}),
    }, indent=2))


if __name__ == "__main__":
    main()
