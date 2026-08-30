#!/usr/bin/env python3
"""Materialize exact, resumable recovery forks from an onset manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onset-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def complete(output: Path, row: dict) -> bool:
    path = output / "manifest.json"
    if not path.is_file() or not (output / "mujoco_state.npz").is_file():
        return False
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return bool(
        manifest.get("exact_archival_prefix_validated") is True
        and int(manifest.get("fork_timestep", -1)) == int(row["fork_timestep"])
        and int(manifest.get("task_id", -1)) == int(row["sample_id"].split(":task")[1].split(":")[0])
    )


def main() -> None:
    args = parse_args()
    if args.num_workers < 1 or not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker-index must be in [0, num-workers)")
    payload = json.loads(args.onset_manifest.read_text())
    located = [row for row in payload["records"] if row.get("onset_status") == "located"]
    selected = [row for index, row in enumerate(located) if index % args.num_workers == args.worker_index]
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "kind": "libero_plus_recovery_fork_worker_plan_v1",
        "onset_manifest": str(args.onset_manifest.resolve()),
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "global_located": len(located),
        "selected": len(selected),
        "sample_ids": [row["sample_id"] for row in selected],
    }
    (args.output_root / f"worker-{args.worker_index}-plan.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return

    completed = 0
    for row in selected:
        suite, task_token, episode_token = row["sample_id"].split(":")
        task_id = int(task_token.removeprefix("task"))
        episode_id = int(episode_token.removeprefix("episode"))
        output = args.output_root / f"{suite}-task{task_id}-episode{episode_id}"
        if complete(output, row):
            completed += 1
            continue
        command = [
            str(args.python_bin), str(ROOT / "scripts" / "save_oft_recovery_fork_snapshot.py"),
            "--source-steps", row["steps"],
            "--dense-bundle", row["dense_bundle"],
            "--fork-timestep", str(row["fork_timestep"]),
            "--output-dir", str(output),
            "--task-suite", suite,
            "--task-id", str(task_id),
            "--episode-id", str(episode_id),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        if not complete(output, row):
            raise RuntimeError(f"fork did not pass completion audit: {row['sample_id']}")
        completed += 1
        print(json.dumps({
            "sample_id": row["sample_id"], "fork_timestep": row["fork_timestep"],
            "completed": completed, "worker_total": len(selected),
        }), flush=True)


if __name__ == "__main__":
    main()
