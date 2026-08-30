#!/usr/bin/env python3
"""Collect routed relational teachers without aborting a shard on candidate failure."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--fork-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--linear-limit", type=float, default=0.5)
    parser.add_argument("--rotation-limit", type=float, default=0.5)
    parser.add_argument("--position-gain", type=float, default=6.0)
    parser.add_argument("--rotation-gain", type=float, default=3.0)
    parser.add_argument("--linear-slew", type=float, default=0.12)
    parser.add_argument("--rotation-slew", type=float, default=0.12)
    parser.add_argument("--stable-confirmation-steps", type=int, default=2)
    parser.add_argument("--grasp-profile-file", type=Path)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def identity(record: dict) -> tuple[str, int, int]:
    suite, task_token, episode_token = record["sample_id"].split(":")
    return suite, int(task_token.removeprefix("task")), int(episode_token.removeprefix("episode"))


def audit(output: Path, record: dict) -> tuple[bool, str, dict | None]:
    metadata_path = output / "metadata.json"
    if not metadata_path.is_file():
        return False, "metadata_missing", None
    try:
        metadata = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False, "metadata_invalid", None
    _, task_id, episode_id = identity(record)
    expected = record["selected_predicate"]
    if int(metadata.get("task_id", -1)) != task_id or int(metadata.get("episode_id", -1)) != episode_id:
        return False, "identity_mismatch", metadata
    if int(metadata.get("fork_timestep", -1)) != int(record["fork_timestep"]):
        return False, "fork_mismatch", metadata
    if metadata.get("verification_predicate") != expected:
        return False, "verification_predicate_mismatch", metadata
    if not metadata.get("success"):
        return False, "teacher_unsuccessful", metadata
    for path in (output / "raw-transitions.jsonl", output / "rollout.mp4"):
        if not path.is_file() or path.stat().st_size == 0:
            return False, f"{path.name}_missing", metadata
    return True, "complete", metadata


def main() -> None:
    args = arguments()
    if args.num_workers < 1 or not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker-index must be in [0,num-workers)")
    route = json.loads(args.route_manifest.read_text())
    relational = [
        record for record in route["records"]
        if record.get("route_status") == "routed"
        and record.get("teacher_skill") == "relational_pick_place"
    ]
    if args.sample_id:
        requested = set(args.sample_id)
        relational = [record for record in relational if record["sample_id"] in requested]
        found = {record["sample_id"] for record in relational}
        if found != requested:
            raise ValueError(f"requested sample ids are not routed relational records: {sorted(requested - found)}")
    selected = [
        record for index, record in enumerate(relational)
        if index % args.num_workers == args.worker_index
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "kind": "routed_relational_teacher_worker_plan_v1",
        "route_manifest": str(args.route_manifest.resolve()),
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "global_relational": len(relational),
        "selected": len(selected),
        "sample_ids": [record["sample_id"] for record in selected],
    }
    (args.output_root / f"worker-{args.worker_index}-plan.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return

    results = []
    for count, record in enumerate(selected, start=1):
        suite, task_id, episode_id = identity(record)
        output = args.output_root / f"task{task_id}-episode{episode_id}"
        valid, reason, metadata = audit(output, record)
        if valid:
            status = "skipped_complete"
        elif output.exists() and any(output.iterdir()):
            status = f"skipped_existing_{reason}"
        else:
            fork = args.fork_root / f"{suite}-task{task_id}-episode{episode_id}"
            if not (fork / "manifest.json").is_file():
                status = "fork_missing"
            else:
                predicate = record["selected_predicate"]
                command = [
                    str(args.python_bin),
                    str(ROOT / "scripts" / "collect_relational_pick_place_feedback_recovery.py"),
                    "--fork-dir", str(fork), "--output-dir", str(output),
                    "--target-name", record["target_name"],
                    "--destination-site", record["destination_name"],
                    "--predicate-name", predicate[0].lower(),
                    "--failure-type", record["failure_type"],
                    "--linear-limit", str(args.linear_limit),
                    "--rotation-limit", str(args.rotation_limit),
                    "--position-gain", str(args.position_gain),
                    "--rotation-gain", str(args.rotation_gain),
                    "--linear-slew", str(args.linear_slew),
                    "--rotation-slew", str(args.rotation_slew),
                    "--stable-confirmation-steps", str(args.stable_confirmation_steps),
                ]
                if args.grasp_profile_file is not None:
                    command.extend([
                        "--grasp-profile-file", str(args.grasp_profile_file),
                        "--grasp-profile-key", f"{record['family_name']}:{record['target_name']}",
                    ])
                log_path = args.output_root / f"task{task_id}-episode{episode_id}.log"
                with log_path.open("w") as log:
                    completed = subprocess.run(
                        command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True,
                    )
                valid, reason, metadata = audit(output, record)
                status = "collected" if valid else f"failed_{reason}_exit{completed.returncode}"
        result = {
            "sample_id": record["sample_id"], "family_name": record.get("family_name"),
            "failure_type": record["failure_type"], "status": status,
            "success": valid, "output": str(output.resolve()),
            "transitions": None if metadata is None else metadata.get("transitions"),
            "task_success": None if metadata is None else metadata.get("task_success"),
            "completed": count, "worker_total": len(selected),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {
        **plan,
        "status_counts": dict(sorted(Counter(result["status"] for result in results).items())),
        "successes": sum(result["success"] for result in results),
        "results": results,
    }
    (args.output_root / f"worker-{args.worker_index}-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
