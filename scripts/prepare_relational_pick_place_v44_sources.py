#!/usr/bin/env python3
"""Adapt audited branch trajectories to the V4.4 feature-source contract.

The adapter writes only JSON manifests.  RGB-D and segmentation files remain in
the immutable teacher directory and are referenced by absolute path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--teacher-root", type=Path)
    source.add_argument(
        "--teacher-manifest", type=Path,
        help="successful-action-teachers.json emitted by the batch pipeline",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--action-query-horizon", type=int, default=8)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    if args.action_query_horizon < 1:
        raise ValueError("action-query-horizon must be positive")
    if args.teacher_manifest is not None:
        manifest = json.loads(args.teacher_manifest.read_text())
        metadata_paths = sorted(
            Path(record["metadata"]) for record in manifest["records"]
        )
        teacher_source = str(args.teacher_manifest.resolve())
    else:
        metadata_paths = sorted(args.teacher_root.glob("task*-episode*/metadata.json"))
        teacher_source = str(args.teacher_root.resolve())
    if not metadata_paths:
        raise ValueError(f"no completed teacher metadata below {args.teacher_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_records = []
    for metadata_path in metadata_paths:
        teacher_dir = metadata_path.parent
        teacher = json.loads(metadata_path.read_text())
        if teacher.get("success") is not True:
            raise ValueError(f"teacher is not successful: {teacher_dir}")
        raw_path = teacher_dir / "raw-transitions.jsonl"
        rows = load_rows(raw_path)
        if len(rows) != int(teacher["transitions"]):
            raise ValueError(f"transition count mismatch: {teacher_dir}")
        output = args.output_root / teacher_dir.name
        output.mkdir(parents=True, exist_ok=True)
        adapted = []
        for expected_step, row in enumerate(rows):
            if int(row.get("branch_timestep", -1)) != expected_step:
                raise ValueError(f"non-contiguous branch timestep: {raw_path}:{expected_step}")
            observation = row.get("online_observation")
            if not isinstance(observation, dict) or observation.get("recovery_action") is None:
                raise ValueError(f"missing recovery action: {raw_path}:{expected_step}")
            value = dict(row)
            value["executed_action"] = observation["recovery_action"]
            value["gate_label"] = "RECOVERY"
            value["source_branch_timestep"] = expected_step
            adapted.append(value)
        steps_path = output / "steps.jsonl"
        steps_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in adapted)
        )
        source_fork_timestep = int(teacher["fork_timestep"])
        metadata = {
            **teacher,
            "kind": "relational_pick_place_v44_feature_source_v1",
            "success": True,
            # The feature-source sequence begins exactly at the restored fork.
            "fork_timestep": 0,
            "source_fork_timestep": source_fork_timestep,
            "source_teacher": str(teacher_dir.resolve()),
            "source_raw_transitions": str(raw_path.resolve()),
            "source_rollout": str((teacher_dir / "rollout.mp4").resolve()),
            "rollout_id": (
                f"relational-pick-place-v44:{teacher['task_suite']}:"
                f"task{teacher['task_id']}:episode{teacher['episode_id']}:"
                f"source-fork{source_fork_timestep}"
            ),
            "wait_steps": 0,
            "open_loop_steps": 1,
            "action_query_horizon": args.action_query_horizon,
            "recovery_sampling_stride": 1,
            "recovery_target_horizon": 1,
            "coordinate_contract": (
                "local step 0 is the exact restored archival fork; source_fork_timestep "
                "retains the baseline-rollout coordinate"
            ),
            "action_contract": (
                "executed_action is the physical teacher action copied from "
                "online_observation.recovery_action"
            ),
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        report_records.append({
            "task_id": int(teacher["task_id"]),
            "episode_id": int(teacher["episode_id"]),
            "source_fork_timestep": source_fork_timestep,
            "rows": len(adapted),
            "teacher": str(teacher_dir.resolve()),
            "feature_source": str(output.resolve()),
        })
    report = {
        "kind": "relational_pick_place_v44_feature_source_report_v1",
        "teacher_source": teacher_source,
        "output_root": str(args.output_root.resolve()),
        "action_query_horizon": args.action_query_horizon,
        "num_sources": len(report_records),
        "total_rows": sum(record["rows"] for record in report_records),
        "records": report_records,
    }
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("num_sources", "total_rows", "output_root")}, indent=2))


if __name__ == "__main__":
    main()
