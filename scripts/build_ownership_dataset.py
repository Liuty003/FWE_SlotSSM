#!/usr/bin/env python3
"""Build state-centric gate/action supervision from certified recoveries."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))


def recovery_chunk(rows: list[dict[str, Any]], start: int, horizon: int) -> list[list[float]] | None:
    end = start + horizon
    if end > len(rows):
        return None
    actions = [row.get("online_observation", {}).get("recovery_action") for row in rows[start:end]]
    return actions if all(isinstance(action, list) and len(action) == 7 for action in actions) else None


def state_row(task: int, episode: int, failure: int, timestep: int, handoff: int,
              raw: dict[str, Any], chunk: list[list[float]] | None) -> dict[str, Any]:
    gate = int(timestep < handoff)
    action_mask = int(gate == 1 and chunk is not None and timestep + len(chunk) <= handoff)
    return {
        "schema": "counterfactual_ownership_state_v1",
        "source": f"task{task}-episode{episode}-failure{failure}",
        "task_id": task, "episode_id": episode, "failure_timestep": failure,
        "teacher_timestep": timestep, "certified_handoff": handoff,
        "state_class": "teacher_recovery" if gate else "resumable",
        "gate_label": gate, "action_mask": action_mask,
        "recovery_action_chunk": chunk if action_mask else None,
        "online_observation": raw["online_observation"],
        "audit": {
            "kind": "frozen_oft_full_task_rollout",
            "resumable": not bool(gate),
            "label_semantics": "gate=1 recovery owns control; gate=0 Frozen OFT owns control",
        },
    }


def build(factory_root: Path, output_dir: Path, *, action_horizon: int = 8) -> dict[str, Any]:
    report = json.loads((factory_root / "factory-report.json").read_text())
    certified_statuses = {"certified", "certified_pending_dataset_merge", "dataset_ready"}
    certified = [record for record in report["records"] if record.get("status") in certified_statuses]
    if not certified:
        raise ValueError("factory report contains no certified recoveries")
    output_dir.mkdir(parents=True, exist_ok=True)
    states: list[dict[str, Any]] = []
    v4_handoff_records = []
    sources = []
    for record in certified:
        task, episode = int(record["task_id"]), int(record.get("episode", 0))
        task_root = factory_root / "tasks" / f"task{task}-episode{episode}"
        teacher_dir = task_root / "feedback-teacher-v1"
        metadata = json.loads((teacher_dir / "metadata.json").read_text())
        handoff_payload = json.loads((task_root / "handoff-manifest.json").read_text())
        handoff = int(handoff_payload["certified_handoff"])
        raw_rows = read_jsonl(teacher_dir / "raw-transitions.jsonl")
        if metadata.get("success") is not True or not 0 < handoff < len(raw_rows):
            raise ValueError(f"task {task}: invalid certified teacher/handoff")
        source_dir = output_dir / "sources" / f"task{task}-episode{episode}"
        source_dir.mkdir(parents=True, exist_ok=True)
        v4_rows = []
        for timestep, raw in enumerate(raw_rows):
            chunk = recovery_chunk(raw_rows, timestep, action_horizon)
            states.append(state_row(
                task, episode, int(metadata["failure_timestep"]), timestep,
                handoff, raw, chunk,
            ))
            executed = raw["online_observation"]["recovery_action"]
            v4 = {
                "kind": "counterfactual_v43_recovery_transition",
                "timestep": timestep,
                "phase": "TEACHER_RECOVERY" if timestep < handoff else "OFT_RESUMABLE",
                "gate_label": "RECOVERY" if timestep < handoff else "NORMAL",
                "online_observation": raw["online_observation"],
                "training_only": raw.get("training_only", {}),
                "executed_action": executed,
            }
            if chunk is not None and timestep + action_horizon <= handoff:
                v4["recovery_target_action_chunk"] = chunk
            v4_rows.append(v4)
        write_jsonl(source_dir / "steps.jsonl", v4_rows)
        source_metadata = {
            **metadata,
            "kind": "counterfactual_v43_certified_local_recovery",
            "schema": "v4_rgbd_rollout_v2", "fork_timestep": 0,
            "transitions": len(v4_rows), "recovery_branch": str(teacher_dir.resolve()),
            "source_rollout": metadata["source_fork"],
        }
        (source_dir / "metadata.json").write_text(json.dumps(source_metadata, indent=2) + "\n")
        v4_handoff_records.append({
            "task_id": task, "episode_id": episode, "status": "certified_local_recovery",
            "source": str(source_dir.resolve()), "fork_timestep": 0,
            "certified_local_handoff": handoff, "failure_timestep": metadata["failure_timestep"],
        })
        sources.append({"task_id": task, "episode_id": episode, "rows": len(v4_rows), "handoff": handoff})
    gate_rows = [{key: row[key] for key in row if key != "recovery_action_chunk"} for row in states]
    action_rows = [row for row in states if row["action_mask"]]
    write_jsonl(output_dir / "state_dataset.jsonl", states)
    write_jsonl(output_dir / "gate_dataset.jsonl", gate_rows)
    write_jsonl(output_dir / "action_dataset.jsonl", action_rows)
    handoff_manifest = {
        "kind": "v43_local_recovery_handoff_manifest", "records": v4_handoff_records,
        "fixed_latch_all_sources_certified": False,
    }
    (output_dir / "handoff_manifest.json").write_text(json.dumps(handoff_manifest, indent=2) + "\n")
    counts = Counter(row["state_class"] for row in states)
    quality = {
        "kind": "counterfactual_ownership_quality_report_v1",
        "certified_sources": len(sources), "sources": sources,
        "state_rows": len(states), "gate_rows": len(gate_rows), "action_rows": len(action_rows),
        "state_classes": dict(sorted(counts.items())), "action_horizon": action_horizon,
        "leakage_guard": "teacher_only and MuJoCo state are excluded from model inputs",
        "v43_feature_input_root": str((output_dir / "sources").resolve()),
        "v43_handoff_manifest": str((output_dir / "handoff_manifest.json").resolve()),
    }
    (output_dir / "quality_report.json").write_text(json.dumps(quality, indent=2) + "\n")
    return quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=8)
    args = parser.parse_args()
    if args.action_horizon < 1:
        raise ValueError("action horizon must be positive")
    print(json.dumps(build(args.factory_root, args.output_dir, action_horizon=args.action_horizon), indent=2))


if __name__ == "__main__":
    main()
