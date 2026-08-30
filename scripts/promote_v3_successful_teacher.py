#!/usr/bin/env python3
"""Promote an audited historical successful branch to a V3 teacher trace.

Historical book-recovery branches were generated offline with privileged state,
but the recorded trace itself contains causal RGB, wrist RGB, proprioception,
and the exact actions executed by the environment.  This tool retains only
those deploy-time fields and emits explicit provenance.  It never changes an
action, observation, or success result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument(
        "--source-metadata", type=Path, required=True,
        help="Current nominal-rollout metadata, used only for task language and physical source identity.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def valid_action(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 7 and all(isinstance(x, (int, float)) for x in value)


def main() -> None:
    args = parse_args()
    teacher_meta_path = args.teacher_dir / "metadata.json"
    steps_path = args.teacher_dir / "steps.jsonl"
    if not teacher_meta_path.is_file() or not steps_path.is_file():
        raise FileNotFoundError("teacher-dir must contain metadata.json and steps.jsonl")
    teacher_meta = load_json(teacher_meta_path)
    source_meta = load_json(args.source_metadata)
    if teacher_meta.get("task_success") is not True and teacher_meta.get("success") is not True:
        raise ValueError("refusing to promote a branch without explicit successful completion")
    identity = ("task_suite", "task_id", "episode_id")
    if any(key not in teacher_meta or key not in source_meta for key in identity):
        raise ValueError(f"teacher and source metadata require {identity}")
    if any(teacher_meta[key] != source_meta[key] for key in identity):
        raise ValueError("teacher branch and nominal source describe different physical episodes")
    language = source_meta.get("task_language")
    if not isinstance(language, str) or not language:
        raise ValueError("source metadata lacks task_language")

    retained: list[dict[str, Any]] = []
    for line_no, line in enumerate(steps_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        observation = row.get("online_observation")
        if not isinstance(observation, dict) or not isinstance(observation.get("proprio"), dict):
            raise ValueError(f"line {line_no}: missing causal online observation")
        if not all(isinstance(observation.get(key), str) and Path(observation[key]).is_file() for key in ("rgb", "wrist_rgb")):
            raise ValueError(f"line {line_no}: missing saved causal images")
        if not valid_action(row.get("executed_action")):
            raise ValueError(f"line {line_no}: missing exact executed_action")
        # Deliberately construct the allowed schema instead of copying the
        # offline FSM/object-pose payload stored under teacher_only.
        retained.append({
            "kind": "v3_canonical_recovery_transition",
            "timestep": len(retained),
            "online_observation": observation,
            "executed_action": row["executed_action"],
            "gate_label": "RECOVERY",
        })
    if not retained:
        raise ValueError("teacher trace has no transitions")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "steps.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in retained), encoding="utf-8"
    )
    metadata = {
        "kind": "v3_canonical_successful_recovery_teacher",
        "schema": "v3_canonical_teacher_v1",
        **{key: source_meta[key] for key in identity},
        "task_language": language,
        "success": True,
        "num_actions": len(retained),
        "teacher_policy": "historical_offline_full_completion",
        "teacher_only_generation": True,
        "online_inputs_permitted": ["main_rgb", "wrist_rgb", "language", "proprio", "executed_action_history"],
        "provenance": {
            "historical_teacher_dir": str(args.teacher_dir.resolve()),
            "historical_metadata": str(teacher_meta_path.resolve()),
            "current_source_metadata": str(args.source_metadata.resolve()),
            "historical_task_success": True,
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "actions": len(retained), "source": [metadata[key] for key in identity]}, indent=2))


if __name__ == "__main__":
    main()
