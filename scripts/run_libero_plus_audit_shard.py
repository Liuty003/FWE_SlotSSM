#!/usr/bin/env python3
"""Run one resumable shard of the official LIBERO-Plus missing-baseline queue."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument(
        "--source-worker-index", type=int,
        help="Optionally reshard only this worker from an earlier modulo partition.",
    )
    parser.add_argument(
        "--source-num-workers", type=int,
        help="Number of workers in the earlier modulo partition.",
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--max-instances", type=int)
    parser.add_argument(
        "--selection", choices=("missing", "all"), default="missing",
        help="Collect only missing compatible baselines, or recollect every official instance.",
    )
    parser.add_argument(
        "--pilot-per-cell", type=int,
        help=(
            "Before sharding, keep this many deterministic examples from every "
            "suite x perturbation x difficulty cell."
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker-index must be in [0, num-workers)")
    if (args.source_worker_index is None) != (args.source_num_workers is None):
        raise ValueError("provide source-worker-index and source-num-workers together")
    if args.source_num_workers is not None and not (
        args.source_num_workers > 0
        and 0 <= args.source_worker_index < args.source_num_workers
    ):
        raise ValueError("source-worker-index must be in [0, source-num-workers)")
    if args.max_instances is not None and args.max_instances < 1:
        raise ValueError("max-instances must be positive")
    if args.pilot_per_cell is not None and args.pilot_per_cell < 1:
        raise ValueError("pilot-per-cell must be positive")

    inventory = json.loads(args.inventory.read_text())
    queue = inventory["records"]
    if args.selection == "missing":
        queue = [row for row in queue if row["baseline"] is None]
    if args.pilot_per_cell is not None:
        cell_counts: dict[tuple[str, str, str], int] = defaultdict(int)
        pilot = []
        for row in queue:
            cell = (
                row["suite"], row["perturbation_category"],
                "unassigned" if row["difficulty_level"] is None else str(row["difficulty_level"]),
            )
            if cell_counts[cell] < args.pilot_per_cell:
                pilot.append(row)
                cell_counts[cell] += 1
        queue = pilot
    if args.source_num_workers is not None:
        queue = [
            row for index, row in enumerate(queue)
            if index % args.source_num_workers == args.source_worker_index
        ]
    selected = [row for index, row in enumerate(queue) if index % args.num_workers == args.worker_index]
    if args.max_instances is not None:
        selected = selected[:args.max_instances]
    by_suite: dict[str, list[int]] = defaultdict(list)
    for row in selected:
        by_suite[row["suite"]].append(int(row["task_id"]))

    args.output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "kind": "libero_plus_audit_shard_plan_v1",
        "inventory": str(args.inventory.resolve()),
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "source_worker_index": args.source_worker_index,
        "source_num_workers": args.source_num_workers,
        "gpu": args.gpu,
        "selection": args.selection,
        "pilot_per_cell": args.pilot_per_cell,
        "global_queue_instances": len(queue),
        "instances": len(selected),
        "suite_counts": {suite: len(task_ids) for suite, task_ids in sorted(by_suite.items())},
        "sample_ids": [row["sample_id"] for row in selected],
    }
    (args.output_root / f"worker-{args.worker_index}-plan.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    for suite, task_ids in sorted(by_suite.items()):
        command = [
            str(args.python_bin), str(ROOT / "scripts" / "collect_oft_nominal_trajectories.py"),
            "--task-suite", suite,
            "--task-ids", ",".join(map(str, task_ids)),
            "--episodes", "0", "--gpu", str(args.gpu),
            "--libero-root", "/data/lty/code/LIBERO-plus",
            "--checkpoint", "/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata",
            "--output-root", str(args.output_root / suite),
            "--save-privileged-audit", "--save-dense-audit",
            "--skip-checkpoint-config-sync", "--resume",
        ]
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
