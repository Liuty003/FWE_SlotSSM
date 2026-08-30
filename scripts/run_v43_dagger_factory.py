#!/usr/bin/env python3
"""Run V4.3 on certified tasks and collect teacher corrections from rollout states.

This is a resumable factory driver.  It derives task ids and recovery target
roles from the ownership dataset, saves exact online recovery forks, samples a
small set from the first recovery activation, and accepts only feedback-teacher
branches that satisfy the verified local lift predicate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory-root", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--task-ids", help="Optional comma-separated certified-task shard.")
    parser.add_argument("--max-teacher-forks-per-task", type=int, default=3)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--release-threshold", type=float, default=0.5)
    parser.add_argument("--execution-horizon", type=int, default=4)
    parser.add_argument("--recovery-horizon", type=int, default=72)
    parser.add_argument("--minimum-recovery-steps", type=int, default=40)
    parser.add_argument("--cooldown-steps", type=int, default=16)
    parser.add_argument("--wait-steps", type=int, default=10)
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    if count == 1:
        return [paths[0]]
    indices = [round(index * (len(paths) - 1) / (count - 1)) for index in range(count)]
    return [paths[index] for index in indices]


def main() -> None:
    args = arguments()
    if args.max_teacher_forks_per_task < 1:
        raise ValueError("--max-teacher-forks-per-task must be positive")
    for path in (args.factory_root, args.adapter_dir, args.controller_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest_path = args.factory_root / "ownership-dataset" / "handoff_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    requested = None if not args.task_ids else {
        int(value) for value in args.task_ids.split(",") if value.strip()
    }
    records = [record for record in manifest["records"]
               if record.get("status") == "certified_local_recovery"
               and (requested is None or int(record["task_id"]) in requested)]
    if not records:
        raise ValueError("no certified tasks selected")
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_records = []
    for record in records:
        task_id = int(record["task_id"])
        source_metadata = json.loads((Path(record["source"]) / "metadata.json").read_text())
        target_role = source_metadata["target_role"]
        task_root = args.output_root / f"task{task_id}-episode0"
        rollout_root = task_root / "online"
        forks_root = rollout_root / "recovery-forks"
        online_report = rollout_root / "report.json"
        if not online_report.is_file():
            run([
                sys.executable, str(ROOT / "scripts" / "run_v4_relational_online.py"),
                "--task-id", str(task_id), "--gpu", str(args.gpu),
                "--adapter-dir", str(args.adapter_dir), "--adapter-checkpoint", "best.safetensors",
                "--output-dir", str(rollout_root), "--no-paired-baseline",
                "--gate-threshold", str(args.gate_threshold),
                "--release-threshold", str(args.release_threshold),
                "--execution-horizon", str(args.execution_horizon),
                "--recovery-horizon", str(args.recovery_horizon),
                "--minimum-recovery-steps", str(args.minimum_recovery_steps),
                "--cooldown-steps", str(args.cooldown_steps), "--wait-steps", str(args.wait_steps),
                "--save-recovery-forks-dir", str(forks_root),
                "--recovery-target-role", target_role,
            ])
        online = json.loads(online_report.read_text())["results"][-1]
        activation_one = sorted(forks_root.glob("activation-01-step-*"))
        selected = [] if online.get("success") else evenly_spaced(
            activation_one, args.max_teacher_forks_per_task
        )
        teachers = []
        for fork in selected:
            teacher_dir = task_root / "teachers" / fork.name
            metadata_path = teacher_dir / "metadata.json"
            if not metadata_path.is_file():
                run([
                    sys.executable, str(ROOT / "scripts" / "collect_mug_pudding_feedback_recovery.py"),
                    "--fork-dir", str(fork),
                    "--controller-checkpoint", str(args.controller_checkpoint),
                    "--output-dir", str(teacher_dir),
                ])
            metadata = json.loads(metadata_path.read_text())
            teachers.append({
                "fork": str(fork.resolve()), "teacher": str(teacher_dir.resolve()),
                "success": metadata.get("success") is True,
                "transitions": metadata.get("transitions"),
            })
        result = {
            "task_id": task_id, "target_role": target_role,
            "online_success": online.get("success") is True,
            "recovery_activations": online.get("recovery_activations", 0),
            "saved_forks": len(list(forks_root.glob("activation-*-step-*"))),
            "selected_teacher_forks": len(selected), "teachers": teachers,
            "dataset_ready": bool(online.get("success") or (teachers and all(x["success"] for x in teachers))),
        }
        report_records.append(result)
        (task_root / "dagger-report.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.output_root / "factory-report.json").write_text(json.dumps({
            "kind": "v43_dagger_recovery_factory_v1",
            "manifest": str(manifest_path.resolve()),
            "adapter": str(args.adapter_dir.resolve()),
            "records": report_records,
        }, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
