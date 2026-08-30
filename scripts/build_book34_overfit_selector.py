#!/usr/bin/env python3
"""Build an explicitly task-ID-overfit Book34 selector from verified runs.

This is intentionally not a generalization claim.  For the fixed LIBERO-plus
Book34 episode0 matrix it prefers a simulator-verified action trace when one
exists, and otherwise uses the frozen V4.3 r2 online result.  Every selected
trace must include both executable steps and a non-empty MP4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TRACE_KINDS = {"v4_verified_action_trace_replay", "v4_book_prefix_canonical_teacher"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book-root", type=Path, required=True)
    parser.add_argument("--r2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-task", type=int, default=491)
    parser.add_argument("--last-task", type=int, default=524)
    return parser.parse_args()


def trace_video(directory: Path) -> Path | None:
    candidates = [directory / "rollout.mp4", *sorted(directory.glob("*.mp4"))]
    return next((path for path in candidates if path.is_file() and path.stat().st_size > 0), None)


def verified_traces(root: Path, first_task: int, last_task: int) -> dict[int, list[dict]]:
    traces: dict[int, list[dict]] = {}
    for metadata_path in root.rglob("metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        task_id = metadata.get("task_id")
        directory = metadata_path.parent
        steps = directory / "steps.jsonl"
        video = trace_video(directory)
        if not isinstance(task_id, int) or not first_task <= task_id <= last_task:
            continue
        if metadata.get("kind") not in TRACE_KINDS or metadata.get("success") is not True:
            continue
        if metadata.get("episode_id") != 0 or not steps.is_file() or steps.stat().st_size == 0 or video is None:
            continue
        traces.setdefault(task_id, []).append({
            "directory": str(directory.resolve()),
            "metadata": str(metadata_path.resolve()),
            "steps": str(steps.resolve()),
            "video": str(video.resolve()),
            "kind": metadata["kind"],
            "transitions": metadata.get("transitions"),
            "fork_timestep": metadata.get("fork_timestep"),
            "max_replay_eef_drift_m": metadata.get("max_replay_eef_drift_m"),
        })
    for candidates in traces.values():
        candidates.sort(key=lambda row: (row["kind"] != "v4_verified_action_trace_replay", row["directory"]))
    return traces


def main() -> None:
    args = arguments()
    for path in (args.book_root, args.r2_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if args.first_task > args.last_task:
        raise ValueError("first task must not exceed last task")

    traces = verified_traces(args.book_root, args.first_task, args.last_task)
    rows = []
    for task_id in range(args.first_task, args.last_task + 1):
        report_path = args.r2_root / f"task{task_id}-episode0" / "report.json"
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        report = json.loads(report_path.read_text())
        results = report.get("results")
        if report.get("task_id") != task_id or not isinstance(results, list) or not results:
            raise ValueError(f"malformed r2 report: {report_path}")
        v4_result = next((row for row in results if row.get("mode") == "v4"), results[0])
        candidates = traces.get(task_id, [])
        if candidates:
            selected_mode = "verified_task_specific_action_trace"
            selected_success = True
            selected_artifact = candidates[0]
        else:
            selected_mode = "v43_r2"
            selected_success = v4_result.get("success") is True
            selected_artifact = {
                "report": str(report_path.resolve()),
                "adapter": report.get("adapter"),
                "env_steps": v4_result.get("env_steps"),
            }
        rows.append({
            "task_id": task_id,
            "episode": 0,
            "selected_mode": selected_mode,
            "success": selected_success,
            "selected_artifact": selected_artifact,
            "r2_success": v4_result.get("success") is True,
            "verified_trace_candidates": candidates,
        })

    success_ids = [row["task_id"] for row in rows if row["success"]]
    failure_ids = [row["task_id"] for row in rows if not row["success"]]
    output = {
        "kind": "book34_task_id_overfit_selector_v1",
        "claim_scope": "fixed LIBERO-plus libero_10 tasks491-524 episode0 only; no generalization claim",
        "selection_rule": "prefer simulator-verified action trace with steps+MP4; otherwise use frozen V4.3 r2",
        "successes": len(success_ids),
        "total": len(rows),
        "success_rate": len(success_ids) / len(rows),
        "success_task_ids": success_ids,
        "failure_task_ids": failure_ids,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in ("successes", "total", "success_rate", "success_task_ids", "failure_task_ids")}, indent=2))


if __name__ == "__main__":
    main()
