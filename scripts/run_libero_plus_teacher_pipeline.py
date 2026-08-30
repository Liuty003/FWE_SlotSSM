#!/usr/bin/env python3
"""Run routed LIBERO-Plus fork/teacher collection and build training manifests.

This is the process-level orchestration layer.  Individual teacher failures are
data outcomes, not pipeline failures: workers continue, and only trajectories
that pass the selected-predicate audit enter the recovery-action manifest.
"""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--grasp-profile-file", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpus", default="0,1,4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument(
        "--stage", choices=("all", "forks", "teachers", "summarize"), default="all"
    )
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def parse_gpus(value: str) -> list[str]:
    gpus = [token.strip() for token in value.split(",") if token.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU index")
    if len(set(gpus)) != len(gpus):
        raise ValueError("--gpus contains duplicate indices")
    return gpus


def load_routes(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("route manifest does not contain a records list")
    return records


def launch_workers(
    *,
    stage: str,
    script: Path,
    base_args: list[str],
    output_root: Path,
    python_bin: Path,
    gpus: list[str],
    workers_per_gpu: int,
    plan_only: bool,
) -> list[dict]:
    num_workers = len(gpus) * workers_per_gpu
    logs = output_root / "pipeline-logs"
    logs.mkdir(parents=True, exist_ok=True)
    specs = []
    for worker_index in range(num_workers):
        gpu = gpus[worker_index % len(gpus)]
        command = [
            str(python_bin), str(script), *base_args,
            "--worker-index", str(worker_index),
            "--num-workers", str(num_workers),
            "--python-bin", str(python_bin),
        ]
        if plan_only:
            command.append("--plan-only")
        specs.append({
            "worker_index": worker_index,
            "gpu": gpu,
            "command": command,
            "log": str((logs / f"{stage}-worker-{worker_index}.log").resolve()),
        })
    if plan_only:
        for spec in specs:
            print(json.dumps(spec), flush=True)
        return specs

    running = []
    for spec in specs:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = spec["gpu"]
        env["MUJOCO_GL"] = "egl"
        log = open(spec["log"], "w")
        process = subprocess.Popen(
            spec["command"], cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        running.append((spec, process, log))
        print(json.dumps({
            "stage": stage, "worker_index": spec["worker_index"],
            "gpu": spec["gpu"], "pid": process.pid, "log": spec["log"],
        }), flush=True)

    failures = []
    for spec, process, log in running:
        returncode = process.wait()
        log.close()
        spec["returncode"] = returncode
        if returncode:
            failures.append(spec)
    if failures:
        failed = ", ".join(
            f"worker{spec['worker_index']}=exit{spec['returncode']}" for spec in failures
        )
        raise RuntimeError(f"{stage} process failures: {failed}")
    return specs


def summarize(route_manifest: Path, fork_root: Path, teacher_root: Path) -> dict:
    records = load_routes(route_manifest)
    routed = [record for record in records if record.get("route_status") == "routed"]
    relational = [
        record for record in routed
        if record.get("teacher_skill") == "relational_pick_place"
    ]
    fork_complete = 0
    for record in routed:
        suite, task_token, episode_token = record["sample_id"].split(":")
        fork = fork_root / (
            f"{suite}-{task_token}-{episode_token}".replace(":", "-")
        )
        if (fork / "manifest.json").is_file() and (fork / "mujoco_state.npz").is_file():
            fork_complete += 1

    successes = []
    failures = []
    for record in relational:
        _, task_token, episode_token = record["sample_id"].split(":")
        output = teacher_root / f"{task_token}-{episode_token}"
        metadata_path = output / "metadata.json"
        metadata = None
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        valid = bool(
            metadata
            and metadata.get("success") is True
            and metadata.get("verification_predicate") == record.get("selected_predicate")
            and (output / "raw-transitions.jsonl").is_file()
            and (output / "rollout.mp4").is_file()
        )
        item = {
            "sample_id": record["sample_id"],
            "family_name": record.get("family_name"),
            "failure_type": record.get("failure_type"),
            "teacher_skill": record.get("teacher_skill"),
            "selected_predicate": record.get("selected_predicate"),
            "output": str(output.resolve()),
            "metadata": str(metadata_path.resolve()) if metadata_path.is_file() else None,
            "transitions": None if metadata is None else metadata.get("transitions"),
            "task_success": None if metadata is None else metadata.get("task_success"),
        }
        if valid:
            item.update({
                "raw_transitions": str((output / "raw-transitions.jsonl").resolve()),
                "video": str((output / "rollout.mp4").resolve()),
            })
            successes.append(item)
        else:
            item["failure_reason"] = (
                "not_collected" if metadata is None
                else metadata.get("failure_reason", "selected_predicate_unsuccessful")
            )
            failures.append(item)

    skill_counts = Counter(record.get("teacher_skill") for record in routed)
    report = {
        "kind": "libero_plus_teacher_pipeline_summary_v1",
        "route_manifest": str(route_manifest.resolve()),
        "fork_root": str(fork_root.resolve()),
        "teacher_root": str(teacher_root.resolve()),
        "records": len(records),
        "routed": len(routed),
        "route_status_counts": dict(Counter(record.get("route_status") for record in records)),
        "teacher_skill_counts": {str(k): v for k, v in skill_counts.items()},
        "forks_complete": fork_complete,
        "relational_attempted": len(relational) - sum(
            item["failure_reason"] == "not_collected" for item in failures
        ),
        "relational_successes": len(successes),
        "relational_failures": len(failures),
        "relational_success_rate_over_attempted": (
            len(successes) / max(1, len(relational) - sum(
                item["failure_reason"] == "not_collected" for item in failures
            ))
        ),
        "successful_action_manifest": str(
            (teacher_root / "successful-action-teachers.json").resolve()
        ),
        "failed_or_pending_manifest": str(
            (teacher_root / "failed-or-pending-teachers.json").resolve()
        ),
    }
    teacher_root.mkdir(parents=True, exist_ok=True)
    (teacher_root / "successful-action-teachers.json").write_text(
        json.dumps({"kind": "successful_recovery_action_teachers_v1", "records": successes}, indent=2)
        + "\n"
    )
    (teacher_root / "failed-or-pending-teachers.json").write_text(
        json.dumps({"kind": "failed_or_pending_recovery_teachers_v1", "records": failures}, indent=2)
        + "\n"
    )
    (teacher_root / "pipeline-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    args = arguments()
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be positive")
    gpus = parse_gpus(args.gpus)
    records = load_routes(args.route_manifest)
    relational = [
        record for record in records
        if record.get("route_status") == "routed"
        and record.get("teacher_skill") == "relational_pick_place"
    ]
    missing_family = [record["sample_id"] for record in relational if not record.get("family_name")]
    if missing_family:
        raise ValueError(
            f"{len(missing_family)} relational routes lack family_name; "
            "rerun route_libero_plus_teacher_skills.py with --family-index"
        )
    profiles = json.loads(args.grasp_profile_file.read_text()).get("profiles", {})
    required_profiles = {
        f"{record['family_name']}:{record['target_name']}" for record in relational
    }
    missing_profiles = sorted(required_profiles - set(profiles))
    if missing_profiles:
        raise ValueError(f"missing grasp profiles: {missing_profiles}")
    plan = {
        "kind": "libero_plus_teacher_pipeline_plan_v1",
        "stage": args.stage,
        "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "num_workers": len(gpus) * args.workers_per_gpu,
        "route_records": len(records),
        "route_status_counts": dict(Counter(r.get("route_status") for r in records)),
        "teacher_skill_counts": dict(Counter(
            str(r.get("teacher_skill")) for r in records if r.get("route_status") == "routed"
        )),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "pipeline-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2), flush=True)

    if args.stage in {"all", "forks"}:
        # Routing may move a teacher fork earlier than the original onset, so
        # materialize from the route manifest rather than the stale onset file.
        launch_workers(
            stage="forks", script=ROOT / "scripts/materialize_libero_plus_recovery_forks.py",
            base_args=["--onset-manifest", str(args.route_manifest), "--output-root", str(args.fork_root)],
            output_root=args.output_root, python_bin=args.python_bin, gpus=gpus,
            workers_per_gpu=args.workers_per_gpu, plan_only=args.plan_only,
        )
    if args.stage in {"all", "teachers"}:
        launch_workers(
            stage="teachers", script=ROOT / "scripts/collect_routed_relational_teacher_batch.py",
            base_args=[
                "--route-manifest", str(args.route_manifest),
                "--fork-root", str(args.fork_root),
                "--output-root", str(args.output_root),
                "--grasp-profile-file", str(args.grasp_profile_file),
            ],
            output_root=args.output_root, python_bin=args.python_bin, gpus=gpus,
            workers_per_gpu=args.workers_per_gpu, plan_only=args.plan_only,
        )
    if not args.plan_only and args.stage in {"all", "teachers", "summarize"}:
        print(json.dumps(summarize(
            args.route_manifest, args.fork_root, args.output_root
        ), indent=2), flush=True)


if __name__ == "__main__":
    main()
