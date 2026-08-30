#!/usr/bin/env python3
"""Resumable fixed-configuration V4.3 batch evaluation shard."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-ids", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gate-threshold", type=float, default=0.8)
    parser.add_argument("--release-threshold", type=float, default=0.5)
    parser.add_argument("--trigger-confirmation-decisions", type=int, default=2)
    parser.add_argument("--minimum-recovery-steps", type=int, default=24)
    parser.add_argument("--recovery-horizon", type=int, default=72)
    parser.add_argument("--execution-horizon", type=int, default=4)
    parser.add_argument("--max-env-steps", type=int)
    parser.add_argument(
        "--grasp-only-controller", action="store_true",
        help="Run the closed-loop grasp controller and permanently hand back to OFT after lift.",
    )
    args = parser.parse_args()
    tasks = [int(value) for value in args.task_ids.split(",") if value.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for task_id in tasks:
        output = args.output_root / f"task{task_id}"
        report_path = output / "report.json"
        if not report_path.is_file():
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "run_v4_relational_online.py"),
                "--task-id", str(task_id), "--gpu", str(args.gpu),
                "--adapter-dir", str(args.adapter_dir), "--adapter-checkpoint", "best.safetensors",
                "--output-dir", str(output), "--no-paired-baseline",
                "--gate-threshold", str(args.gate_threshold),
                "--release-threshold", str(args.release_threshold),
                "--trigger-confirmation-decisions", str(args.trigger_confirmation_decisions),
                "--execution-horizon", str(args.execution_horizon),
                "--recovery-horizon", str(args.recovery_horizon),
                "--minimum-recovery-steps", str(args.minimum_recovery_steps),
                "--cooldown-steps", "16", "--wait-steps", "10",
            ] + (["--grasp-only-controller"] if args.grasp_only_controller else []) +
                ([] if args.max_env_steps is None else ["--max-env-steps", str(args.max_env_steps)]),
                cwd=ROOT, check=True)
        result = json.loads(report_path.read_text())["results"][-1]
        activations = [event for event in result["events"] if event.get("new_recovery_activation")]
        records.append({
            "task_id": task_id, "success": result["success"], "env_steps": result["env_steps"],
            "recovery_activations": result["recovery_activations"],
            "adaptive_handoffs": result["adaptive_handoffs"],
            "first_activation_step": None if not activations else activations[0]["env_step"],
        })
        payload = {
            "kind": "v43_batch_eval_shard_v1",
            "configuration": {
                "gate_threshold": args.gate_threshold, "release_threshold": args.release_threshold,
                "trigger_confirmation_decisions": args.trigger_confirmation_decisions,
                "minimum_recovery_steps": args.minimum_recovery_steps,
                "recovery_horizon": args.recovery_horizon,
                "execution_horizon": args.execution_horizon,
                "max_env_steps": args.max_env_steps,
                "grasp_only_controller": args.grasp_only_controller,
            },
            "tasks": len(records), "successes": sum(row["success"] for row in records),
            "records": records,
        }
        (args.output_root / "batch-report.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(records[-1]), flush=True)


if __name__ == "__main__":
    main()
