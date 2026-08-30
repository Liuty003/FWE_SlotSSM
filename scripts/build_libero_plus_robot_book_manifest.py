#!/usr/bin/env python3
"""Freeze the official LIBERO-Plus Robot-initial-state book task manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/data2/lty/naive-e-slotssm-runs/oft-plus-book-local-repair-v1/"
            "official-robot-initial-state-book-34-manifest.json"
        ),
    )
    args = parser.parse_args()

    from libero import libero as libero_package
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    categories_path = Path(libero_package.__file__).resolve().parent / "benchmark" / "task_classification.json"
    categories = json.loads(categories_path.read_text())["libero_10"]
    category_by_task_id = {entry["id"] - 1: entry for entry in categories}
    records = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        bddl = task.bddl_file
        classification = category_by_task_id.get(task_id, {})
        if (
            classification.get("category") == "Robot Initial States"
            and "pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
            in bddl
        ):
            records.append(
                {
                    "task_id": task_id,
                    "episode_id": 0,
                    "perturbation": classification["category"],
                    "difficulty_level": classification["difficulty_level"],
                    "bddl_file": bddl,
                    "task_language": task.language,
                    "num_initial_states": len(suite.get_task_init_states(task_id)),
                }
            )

    if len(records) != 34 or any(r["num_initial_states"] != 50 for r in records):
        raise RuntimeError(f"Expected 34 Robot tasks with 50 initial states, got: {records}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "num_cases": len(records), "records": records}, indent=2))


if __name__ == "__main__":
    main()
