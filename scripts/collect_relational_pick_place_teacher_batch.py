#!/usr/bin/env python3
"""Shard and audit relational pick-place recovery-teacher collection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_STEP_DIRS = (
    "rgb", "wrist_rgb", "depth", "wrist_depth", "instance", "target_masks",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onset-manifest", type=Path, required=True)
    parser.add_argument("--fork-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--destination-site", required=True)
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
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def task_tokens(row: dict) -> tuple[str, int, int]:
    suite, task_token, episode_token = row["sample_id"].split(":")
    return (
        suite,
        int(task_token.removeprefix("task")),
        int(episode_token.removeprefix("episode")),
    )


def audit(output: Path, row: dict) -> tuple[bool, str]:
    metadata_path = output / "metadata.json"
    if not metadata_path.is_file():
        return False, "metadata_missing"
    try:
        metadata = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False, "metadata_invalid"
    _, task_id, episode_id = task_tokens(row)
    if not metadata.get("success"):
        return False, "teacher_unsuccessful"
    if int(metadata.get("task_id", -1)) != task_id:
        return False, "task_id_mismatch"
    if int(metadata.get("episode_id", -1)) != episode_id:
        return False, "episode_id_mismatch"
    if int(metadata.get("fork_timestep", -1)) != int(row["fork_timestep"]):
        return False, "fork_timestep_mismatch"
    transitions = int(metadata.get("transitions", -1))
    if transitions < 1:
        return False, "transition_count_invalid"
    raw = output / "raw-transitions.jsonl"
    if not raw.is_file() or sum(1 for line in raw.open() if line.strip()) != transitions:
        return False, "raw_transition_count_mismatch"
    if sum(1 for _ in (output / "states").glob("*.npy")) != transitions + 1:
        return False, "state_count_mismatch"
    for directory in REQUIRED_STEP_DIRS:
        if sum(1 for _ in (output / directory).iterdir()) != transitions:
            return False, f"{directory}_count_mismatch"
    video = output / "rollout.mp4"
    if not video.is_file() or video.stat().st_size == 0:
        return False, "video_missing"
    return True, "complete"


def main() -> None:
    args = parse_args()
    if args.num_workers < 1 or not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker-index must be in [0, num-workers)")
    payload = json.loads(args.onset_manifest.read_text())
    located = [row for row in payload["records"] if row.get("onset_status") == "located"]
    selected = [row for index, row in enumerate(located) if index % args.num_workers == args.worker_index]
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "kind": "relational_pick_place_teacher_worker_plan_v1",
        "onset_manifest": str(args.onset_manifest.resolve()),
        "fork_root": str(args.fork_root.resolve()),
        "output_root": str(args.output_root.resolve()),
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
        suite, task_id, episode_id = task_tokens(row)
        output = args.output_root / f"task{task_id}-episode{episode_id}"
        valid, reason = audit(output, row)
        if valid:
            completed += 1
            print(json.dumps({
                "sample_id": row["sample_id"], "status": "skipped_complete",
                "completed": completed, "worker_total": len(selected),
            }), flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(
                f"refusing to overwrite incomplete output {output} ({reason}); "
                "move it aside before resuming"
            )
        fork = args.fork_root / f"{suite}-task{task_id}-episode{episode_id}"
        command = [
            str(args.python_bin),
            str(ROOT / "scripts" / "collect_relational_pick_place_feedback_recovery.py"),
            "--fork-dir", str(fork), "--output-dir", str(output),
            "--target-name", args.target_name,
            "--destination-site", args.destination_site,
            "--linear-limit", str(args.linear_limit),
            "--rotation-limit", str(args.rotation_limit),
            "--position-gain", str(args.position_gain),
            "--rotation-gain", str(args.rotation_gain),
            "--linear-slew", str(args.linear_slew),
            "--rotation-slew", str(args.rotation_slew),
            "--stable-confirmation-steps", str(args.stable_confirmation_steps),
        ]
        log_path = args.output_root / f"task{task_id}-episode{episode_id}.log"
        with log_path.open("w") as log:
            subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                text=True, check=True,
            )
        valid, reason = audit(output, row)
        if not valid:
            raise RuntimeError(f"teacher completion audit failed for {row['sample_id']}: {reason}")
        completed += 1
        metadata = json.loads((output / "metadata.json").read_text())
        print(json.dumps({
            "sample_id": row["sample_id"], "status": "collected",
            "transitions": metadata["transitions"],
            "video": str((output / "rollout.mp4").resolve()),
            "completed": completed, "worker_total": len(selected),
        }), flush=True)


if __name__ == "__main__":
    main()
