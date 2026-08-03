#!/usr/bin/env python
"""Collect a small, reproducible set of natural-failure recovery branches.

This is deliberately a *branch trajectory* collector, not a claim that the
four source states form a benchmark.  Each plan item restores the same natural
OpenVLA failure state for every candidate, executes the full recovery macro,
and writes the macro, terminal simulator labels and an MP4 per branch.  The
result is suitable for building candidate-ranking and recovery-policy training
examples without mixing privileged simulator labels into online inputs.

The task-specific collectors own the simulator details.  Keeping this file as
the plan/manifest layer makes it easy to add a newly audited natural failure
without copying an ad-hoc shell command into a notebook.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROLLOUT_ROOT = Path(
    "/data2/lty/naive-e-slotssm-runs/libero10-natural-baseline-v1"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect four small matched groups from natural failures."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-rollout-root", type=Path,
                        default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument(
        "--python", type=Path,
        default=REPO_ROOT / ".conda/envs/openvla-a/bin/python",
        help="Python environment containing LIBERO/OpenVLA dependencies.",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--groups", nargs="*", default=["task6_ep1", "task6_ep6", "task9_ep5", "task9_ep6"],
        choices=("task6_ep1", "task6_ep6", "task9_ep5", "task9_ep6"),
        help="Named natural failure states to collect.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def plan(source_root: Path, fps: int) -> dict[str, dict[str, Any]]:
    """Return a compact mix of successful and unsuccessful candidates.

    The selected states were visually audited natural failures.  Task-6 uses
    externally measured visual relative xyz values captured in the prior audit;
    Task-9 retains the branch teacher geometry only for *candidate generation*.
    The output manifest explicitly records this distinction so these labels can
    never be mistaken for deployment-time inputs.
    """
    task6 = REPO_ROOT / "scripts/collect_task6_failure_candidate_branches.py"
    task9 = REPO_ROOT / "scripts/collect_task9_failure_candidate_branches.py"
    return {
        "task6_ep1": {
            "failure_type": "missed_grasp_empty_close",
            "source": source_root / "task6/task6-episode1",
            "command": [
                task6, "--episode-id", "1", "--failure-timestep", "183",
                "--target-role", "pudding", "--macro-kind", "relative-regrasp",
                "--relative-target-xyz", "0.045253", "-0.015600", "-0.011691",
                "--candidate-name", "relative_x020_y060_z085",
                "--candidate-name", "relative_x080_y040_z085",
                "--candidate-name", "relative_x100_y060_z085",
                "--candidate-name", "relative_xy100_z114",
                "--fps", str(fps),
            ],
            "candidate_geometry_source": "external_observable_prediction",
        },
        "task6_ep6": {
            "failure_type": "edge_grasp_slip_or_missed_lift",
            "source": source_root / "task6/task6-episode6",
            "command": [
                task6, "--episode-id", "6", "--failure-timestep", "230",
                "--target-role", "pudding", "--macro-kind", "relative-regrasp",
                "--relative-target-xyz", "0.056029", "-0.034471", "-0.027912",
                "--candidate-name", "relative_x020_y060_z085",
                "--candidate-name", "relative_x020_y080_z114",
                "--candidate-name", "relative_x040_y060_z114",
                "--candidate-name", "relative_x040_y080_z085",
                "--fps", str(fps),
            ],
            "candidate_geometry_source": "external_observable_prediction",
        },
        "task9_ep5": {
            "failure_type": "transport_drop",
            "source": source_root / "task9/task9-episode5",
            "command": [
                task9, "--episode-id", "5", "--failure-timestep", "100",
                "--grasp-relative-reference", "0.007", "0.042", "-0.069",
                "--stress-transport-steps", "20", "--stress-transport-action", ".30",
                "--candidate-name", "relative_x020_y060_z085",
                "--candidate-name", "relative_x040_y060_z085",
                "--candidate-name", "relative_x040_y080_z085",
                "--candidate-name", "relative_x040_y080_z114",
                "--fps", str(fps),
            ],
            "candidate_geometry_source": "privileged_branch_teacher",
        },
        "task9_ep6": {
            "failure_type": "missed_grasp_height_or_alignment",
            "source": source_root / "task9/task9-episode6",
            "command": [
                task9, "--episode-id", "6", "--failure-timestep", "110",
                "--candidate-name", "relative_x020_y060_z114",
                "--candidate-name", "relative_x020_y080_z085",
                "--candidate-name", "relative_x040_y080_z085",
                "--candidate-name", "relative_x040_y080_z114",
                "--fps", str(fps),
            ],
            "candidate_geometry_source": "privileged_branch_teacher",
        },
    }


def summarise(group_name: str, item: dict[str, Any], branch_path: Path) -> dict[str, Any]:
    payload = json.loads(branch_path.read_text(encoding="utf-8"))
    candidates = payload["candidates"]
    best = max(candidates, key=lambda row: float(row["teacher_value"]))
    return {
        "group": group_name,
        "task_suite": payload["task_suite"],
        "task_id": payload["task_id"],
        "episode_id": payload["episode_id"],
        "failure_timestep": payload["failure_timestep"],
        "failure_type": item["failure_type"],
        "source_rollout": payload["source_rollout"],
        "candidate_geometry_source": item["candidate_geometry_source"],
        "branch_manifest": str(branch_path),
        "candidate_count": len(candidates),
        "best_candidate": best["candidate_name"],
        "best_teacher_value": best["teacher_value"],
        "any_attached_after": any(bool(row["attached_after"]) for row in candidates),
        "candidate_videos": [row["video"] for row in candidates],
        "candidate_raw_trajectories": [
            row.get("raw_trajectory") for row in candidates
        ],
    }


def main() -> None:
    args = arguments()
    if not args.python.is_file():
        raise FileNotFoundError(f"LIBERO Python not found: {args.python}")
    if not args.source_rollout_root.is_dir():
        raise FileNotFoundError(
            f"natural rollout root not found: {args.source_rollout_root}"
        )
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"refusing to mix runs in non-empty output root: {args.output_root}; "
            "use a new directory or --overwrite"
        )

    selected_plan = plan(args.source_rollout_root, args.fps)
    selected = {name: selected_plan[name] for name in args.groups}
    for name, item in selected.items():
        source = item["source"]
        if not (source / "steps.jsonl").is_file():
            raise FileNotFoundError(f"{name}: missing source trace {source}")

    commands = []
    for name, item in selected.items():
        output = args.output_root / name
        command = [str(args.python), str(item["command"][0]), "--source-rollout", str(item["source"])]
        command.extend(["--failure-type", str(item["failure_type"])])
        command.extend(str(value) for value in item["command"][1:])
        command.extend([
            "--output-dir", str(output),
            "--trajectory-dir", str(output / "raw-trajectories"),
        ])
        commands.append((name, command))

    if args.dry_run:
        print(json.dumps({"commands": commands}, indent=2))
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    records = []
    for name, command in commands:
        print("RUN", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, env=environment)
        if completed.returncode:
            raise SystemExit(f"collection failed for {name}: {completed.returncode}")
        branch_path = args.output_root / name / "candidate_branches.json"
        if not branch_path.is_file():
            raise FileNotFoundError(f"collector did not write {branch_path}")
        records.append(summarise(name, selected[name], branch_path))

    index = {
        "schema_version": 1,
        "kind": "natural_failure_recovery_branch_trajectory_index",
        "purpose": (
            "small training/development collection; terminal simulator outcomes "
            "are teacher labels and never online model inputs"
        ),
        "independent_controller_per_candidate": True,
        "same_sim_state_per_group": True,
        "groups": records,
        "totals": {
            "groups": len(records),
            "candidate_trajectories": sum(row["candidate_count"] for row in records),
            "groups_with_regrasp": sum(row["any_attached_after"] for row in records),
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.output_root / "training-index.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            payload = json.loads(Path(record["branch_manifest"]).read_text())
            for candidate in payload["candidates"]:
                handle.write(json.dumps({
                    **record,
                    "candidate_name": candidate["candidate_name"],
                    "action_chunk": candidate["action_chunk"],
                    "teacher_outcome": {
                        key: value for key, value in candidate.items()
                        if key not in {"candidate_name", "action_chunk", "video"}
                    },
                    "video": candidate["video"],
                }, ensure_ascii=False) + "\n")
    print(json.dumps(index, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
