#!/usr/bin/env python3
"""Unified, resumable counterfactual recovery data factory.

This orchestrates the existing audited components rather than reimplementing
their simulator or action contracts.  Per-task state is persisted after every
stage, so a large job can be resumed without repeating accepted work.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("mine", "fork", "teacher", "handoff", "dataset")


def parse_tasks(values: list[str]) -> list[int]:
    tasks: set[int] = set()
    for value in values:
        for piece in value.split(","):
            if "-" in piece:
                start, end = (int(item) for item in piece.split("-", 1))
                if end < start:
                    raise ValueError(f"invalid task range: {piece}")
                tasks.update(range(start, end + 1))
            else:
                tasks.add(int(piece))
    if not tasks:
        raise ValueError("at least one task is required")
    return sorted(tasks)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["525-566"])
    parser.add_argument("--through", choices=STAGES, default="teacher")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--rollout-root", type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/oft-plus-robot-global-v1/forks"),
    )
    parser.add_argument(
        "--controller-checkpoint", type=Path,
        default=Path(
            "/data2/lty/naive-e-slotssm-runs/libero-plus/mug-pudding-checkpoint-v1/"
            "checkpoints/privileged-cartesian-v3.pt"
        ),
    )
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--adapter-checkpoint", default="best.safetensors")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-stage", choices=STAGES)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--review-overrides", type=Path,
                        help="Optional human review decisions keyed by task id.")
    return parser.parse_args()


def load(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(command: list[str], *, dry_run: bool) -> None:
    print(json.dumps({"command": command}), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def complete(path: Path, force: bool) -> bool:
    return path.is_file() and not force


def phase_audit_steps(metadata: dict[str, Any], stride: int = 8) -> list[int]:
    phases = metadata["phase_by_step"]
    close_start = next((i for i, phase in enumerate(phases) if phase == "close"), 0)
    final = int(metadata["steps"])
    values = list(range(close_start, final + 1, stride))
    values.extend([final, max(close_start, final - 1)])
    return sorted(set(values))


def audit_outcomes(report: dict[str, Any]) -> dict[int, bool]:
    return {
        int(result["teacher_relative_timestep"]): bool(result["success"])
        for result in report["results"]
    }


def dense_audit_steps(coarse: dict[int, bool], final_step: int) -> list[int]:
    successful = sorted(step for step, success in coarse.items() if success)
    if not successful:
        return []
    first = successful[0]
    previous = max((step for step in coarse if step < first and not coarse[step]), default=first)
    return list(range(previous, min(final_step, first + 2) + 1))


def stable_handoff(outcomes: dict[int, bool], width: int = 3) -> int | None:
    for step in sorted(outcomes):
        if all(outcomes.get(step + offset) is True for offset in range(width)):
            return step
    return None


def audit_command(args: argparse.Namespace, task: int, states: Path,
                  steps: list[int], output: Path) -> list[str]:
    if args.adapter_dir is None:
        raise ValueError("--adapter-dir is required through the handoff stage")
    command = [
        str(args.python), "scripts/run_v4_relational_online.py",
        "--task-id", str(task), "--episode", "0",
        "--adapter-dir", str(args.adapter_dir),
        "--adapter-checkpoint", args.adapter_checkpoint,
        "--output-dir", str(output), "--gpu", str(args.gpu),
        "--wait-steps", "10", "--nominal-only",
        "--audit-state-dir", str(states), "--audit-state-steps",
        *[str(step) for step in steps],
    ]
    if args.no_video:
        command.append("--no-save-video")
    return command


def main() -> None:
    args = arguments()
    tasks = parse_tasks(args.tasks)
    through_index = STAGES.index(args.through)
    args.output_root.mkdir(parents=True, exist_ok=True)
    review_overrides = load(args.review_overrides, {}) if args.review_overrides else {}
    review_queue: list[dict[str, Any]] = []
    global_records: list[dict[str, Any]] = []

    # Failure mining initializes the full LIBERO benchmark.  For contiguous
    # large jobs, initialize it once and split the resulting manifest instead
    # of paying that cost in one subprocess per task.
    missing_mines = [task for task in tasks if not (
        args.output_root / "tasks" / f"task{task}-episode0" /
        "failure-mining" / "failure-manifest.json"
    ).is_file() or args.force_stage == "mine"]
    bulk_mined: set[int] = set()
    contiguous_groups: list[list[int]] = []
    for task in missing_mines:
        if not contiguous_groups or task != contiguous_groups[-1][-1] + 1:
            contiguous_groups.append([task])
        else:
            contiguous_groups[-1].append(task)
    for group in contiguous_groups:
        if args.dry_run or len(group) < 2:
            continue
        bulk_dir = args.output_root / "bulk-failure-mining" / f"tasks{group[0]}-{group[-1]}"
        run([
            str(args.python), "scripts/mine_mug_pudding_failures.py",
            "--rollout-root", str(args.rollout_root),
            "--task-start", str(group[0]), "--task-end", str(group[-1]),
            "--output-dir", str(bulk_dir),
        ], dry_run=False)
        bulk = load(bulk_dir / "failure-manifest.json")
        for record in bulk["records"]:
            task = int(record["task_id"])
            mine_dir = args.output_root / "tasks" / f"task{task}-episode0" / "failure-mining"
            one = {**bulk, "task_range": [task, task], "records": [record], "summary": {
                "tasks": 1, "baseline_successes": int(record["baseline_success"]),
                "auto_high_confidence": int(record["status"] == "auto_high_confidence"),
                "needs_review": int(record["status"].startswith("needs_review")),
            }}
            write(mine_dir / "failure-manifest.json", one)
            bulk_mined.add(task)

    for task in tasks:
        task_root = args.output_root / "tasks" / f"task{task}-episode0"
        task_root.mkdir(parents=True, exist_ok=True)
        state_path = task_root / "pipeline-state.json"
        state = load(state_path, {
            "task_id": task, "episode": 0, "stages": {}, "status": "pending"
        })

        def forced(stage: str) -> bool:
            return args.force_stage == stage and not (stage == "mine" and task in bulk_mined)

        try:
            mine_dir = task_root / "failure-mining"
            mine_manifest = mine_dir / "failure-manifest.json"
            if not complete(mine_manifest, forced("mine")):
                run([
                    str(args.python), "scripts/mine_mug_pudding_failures.py",
                    "--rollout-root", str(args.rollout_root),
                    "--task-start", str(task), "--task-end", str(task),
                    "--output-dir", str(mine_dir),
                ], dry_run=args.dry_run)
            if args.dry_run:
                state["status"] = "dry_run_planned"
                write(state_path, state)
                global_records.append(state)
                continue
            failure_payload = load(mine_manifest)
            record = failure_payload["records"][0]
            state["stages"]["mine"] = {"status": record["status"], "manifest": str(mine_manifest)}
            review = review_overrides.get(str(task), {})
            if review.get("failure_decision") == "no_local_failure":
                state["stages"]["mine"]["human_review"] = review
                state["status"] = "human_review_no_local_failure"
                write(state_path, state)
                global_records.append(state)
                continue
            if record["baseline_success"]:
                state["status"] = "normal_source_ready"
                write(state_path, state)
                global_records.append(state)
                continue
            if record["selected_failure"] is None or record["status"].startswith("needs_review"):
                state["status"] = "needs_failure_review"
                review_queue.append({
                    "task_id": task, "reason": state["status"],
                    "video": record.get("baseline_video"), "manifest": str(mine_manifest),
                })
                write(state_path, state)
                global_records.append(state)
                continue
            if through_index == 0:
                state["status"] = "failure_ready"
                write(state_path, state); global_records.append(state); continue

            fork_dir = task_root / "failure-fork"
            fork_manifest = fork_dir / "manifest.json"
            if not complete(fork_manifest, forced("fork")):
                run([
                    str(args.python), "scripts/restore_mug_pudding_failure_fork.py",
                    "--failure-manifest", str(mine_manifest), "--task-id", str(task),
                    "--output-dir", str(fork_dir),
                ], dry_run=False)
            state["stages"]["fork"] = {"status": "ready", "manifest": str(fork_manifest)}
            if through_index == 1:
                state["status"] = "fork_ready"
                write(state_path, state); global_records.append(state); continue

            teacher_dir = task_root / "feedback-teacher-v1"
            teacher_metadata = teacher_dir / "metadata.json"
            if not complete(teacher_metadata, forced("teacher")):
                try:
                    run([
                        str(args.python), "scripts/collect_mug_pudding_feedback_recovery.py",
                        "--fork-dir", str(fork_dir),
                        "--controller-checkpoint", str(args.controller_checkpoint),
                        "--output-dir", str(teacher_dir),
                    ], dry_run=False)
                except subprocess.CalledProcessError:
                    state["status"] = "needs_teacher_review"
                    review_queue.append({
                        "task_id": task, "reason": state["status"],
                        "fork": str(fork_manifest), "teacher_dir": str(teacher_dir),
                    })
                    write(state_path, state); global_records.append(state); continue
            teacher = load(teacher_metadata)
            if teacher.get("success") is not True:
                if review.get("teacher_decision") == "reject_failed_teacher":
                    state["stages"]["teacher"] = {
                        "status": "human_rejected", "metadata": str(teacher_metadata),
                        "video": teacher.get("rollout"), "human_review": review,
                    }
                    state["status"] = "failed_teacher_excluded"
                    write(state_path, state); global_records.append(state); continue
                state["status"] = "needs_teacher_review"
                review_queue.append({
                    "task_id": task, "reason": state["status"],
                    "video": teacher.get("rollout"), "metadata": str(teacher_metadata),
                })
                write(state_path, state); global_records.append(state); continue
            state["stages"]["teacher"] = {
                "status": "accepted", "metadata": str(teacher_metadata),
                "video": teacher["rollout"], "steps": teacher["steps"],
            }
            if through_index == 2:
                state["status"] = "teacher_ready"
                write(state_path, state); global_records.append(state); continue

            handoff_manifest = task_root / "handoff-manifest.json"
            if not complete(handoff_manifest, forced("handoff")):
                states = teacher_dir / "states"
                phases = teacher["phase_by_step"]
                close_start = next((i for i, phase in enumerate(phases) if phase == "close"), 0)
                probe_steps = list(range(close_start, min(int(teacher["steps"]), close_start + 2) + 1))
                probe_dir = task_root / "handoff-probe"
                run(audit_command(args, task, states, probe_steps, probe_dir), dry_run=False)
                probe = audit_outcomes(load(probe_dir / "report.json"))
                certified = stable_handoff(probe)
                coarse_steps: list[int] = []
                dense_steps = probe_steps
                coarse: dict[int, bool] = {}
                dense: dict[int, bool] = {}
                outcomes = dict(probe)
                if certified is None:
                    coarse_steps = [step for step in phase_audit_steps(teacher) if step not in probe]
                    if coarse_steps:
                        coarse_dir = task_root / "handoff-coarse"
                        run(audit_command(args, task, states, coarse_steps, coarse_dir), dry_run=False)
                        coarse = audit_outcomes(load(coarse_dir / "report.json"))
                    combined = {**probe, **coarse}
                    extra_dense_steps = [step for step in dense_audit_steps(
                        combined, int(teacher["steps"])) if step not in combined]
                    dense_steps = sorted(set(probe_steps + extra_dense_steps))
                    if extra_dense_steps:
                        dense_dir = task_root / "handoff-dense"
                        run(audit_command(args, task, states, extra_dense_steps, dense_dir), dry_run=False)
                        dense = audit_outcomes(load(dense_dir / "report.json"))
                    outcomes = {**combined, **dense}
                    certified = stable_handoff(outcomes)
                write(handoff_manifest, {
                    "kind": "mug_pudding_counterfactual_handoff_manifest_v1",
                    "task_id": task, "episode": 0,
                    "failure_timestep": teacher["failure_timestep"],
                    "teacher": str(teacher_metadata),
                    "audit_semantics": "true means Frozen OFT completes the full task",
                    "probe_steps": probe_steps, "coarse_steps": coarse_steps, "dense_steps": dense_steps,
                    "outcomes": {str(key): value for key, value in sorted(outcomes.items())},
                    "stable_width": 3, "certified_handoff": certified,
                })
            handoff = load(handoff_manifest)
            state["stages"]["handoff"] = {
                "status": "certified" if handoff["certified_handoff"] is not None else "uncertified",
                "manifest": str(handoff_manifest), "timestep": handoff["certified_handoff"],
            }
            state["status"] = (
                "certified" if handoff["certified_handoff"] is not None else "needs_handoff_review"
            )
            if state["status"] != "certified":
                review_queue.append({
                    "task_id": task, "reason": state["status"], "manifest": str(handoff_manifest),
                })
            if through_index == 4 and state["status"] == "certified":
                # The global builder runs once after every per-task state has
                # been persisted, so all certified sources are merged atomically.
                state["status"] = "certified_pending_dataset_merge"
        except Exception as error:
            state["status"] = "pipeline_error"
            state["error"] = f"{type(error).__name__}: {error}"
            write(state_path, state)
            write(args.output_root / "review-queue.json", review_queue)
            raise
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write(state_path, state)
        global_records.append(state)

    if through_index == 4 and not args.dry_run:
        # Make the intermediate report visible to the builder.
        intermediate = {"kind": "counterfactual_recovery_factory_report_v1", "tasks": tasks,
                        "through": args.through, "records": global_records}
        write(args.output_root / "factory-report.json", intermediate)
        dataset_dir = args.output_root / "ownership-dataset"
        run([str(args.python), "scripts/build_ownership_dataset.py",
             "--factory-root", str(args.output_root), "--output-dir", str(dataset_dir)], dry_run=False)
        for state in global_records:
            if state["status"] == "certified_pending_dataset_merge":
                state["status"] = "dataset_ready"
                state["stages"]["dataset"] = {
                    "status": "ready", "quality_report": str(dataset_dir / "quality_report.json")}
                task_root = args.output_root / "tasks" / f"task{state['task_id']}-episode{state.get('episode', 0)}"
                write(task_root / "pipeline-state.json", state)

    summary = {
        "kind": "counterfactual_recovery_factory_report_v1",
        "tasks": tasks, "through": args.through, "records": global_records,
        "counts": {
            status: sum(row["status"] == status for row in global_records)
            for status in sorted({row["status"] for row in global_records})
        },
    }
    write(args.output_root / "factory-report.json", summary)
    write(args.output_root / "review-queue.json", review_queue)
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
