#!/usr/bin/env python3
"""Build dense, auditable NORMAL/RECOVERY supervision from saved OFT rollouts.

The trigger timestep is an offline label from the recovery audit, never an
online input.  Every emitted row retains the original RGB, wrist RGB, proprio,
and executed-action history so the frozen-feature precompute path can be
reused unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ROOT = Path("/data2/lty/naive-e-slotssm-runs/oft-plus-book-local-repair-v1/natural-v2")
# Audited first unsafe/collision-recovery times for the stable book failures.
DEFAULT_TRIGGERS = "494:76,501:55,504:62,506:60,510:55,511:44,513:44,521:52,523:57"


def args_() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--triggers", default=DEFAULT_TRIGGERS)
    parser.add_argument("--positive-before", type=int, default=10)
    parser.add_argument("--positive-after", type=int, default=14)
    parser.add_argument("--normal-stride", type=int, default=4)
    return parser.parse_args()


def parse_triggers(value: str) -> dict[int, int]:
    result = {}
    for item in value.split(","):
        task, timestep = item.split(":")
        result[int(task)] = int(timestep)
    return result


def record_with_label(record: dict, label: str, provenance: dict, source_key: str, language: str) -> dict:
    output = dict(record)
    output.update({
        "schema_version": 1,
        "kind": "book_execution_gate_window",
        "source_key": source_key,
        "language": language,
        "gate_label": label,
        "gate_label_provenance": provenance,
    })
    return output


def main() -> None:
    args = args_()
    triggers = parse_triggers(args.triggers)
    if args.normal_stride < 1 or args.positive_before < 0 or args.positive_after < 0:
        raise ValueError("window sizes and stride must be non-negative / positive")
    if not args.rollout_root.exists():
        raise FileNotFoundError(args.rollout_root)
    train, val = [], []
    counts = {"NORMAL": 0, "RECOVERY": 0}
    for steps_path in sorted(args.rollout_root.glob("task*-episode0/steps.jsonl")):
        task_id = int(steps_path.parent.name.split("-")[0].removeprefix("task"))
        metadata_path = steps_path.with_name("metadata.json")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        source_key = f"libero_10-task{task_id}-episode0"
        language = metadata.get("task_language")
        if not language:
            raise ValueError(f"missing task language in {metadata_path}")
        rows = [json.loads(line) for line in steps_path.read_text().splitlines()]
        if task_id in triggers:
            trigger = triggers[task_id]
            for row in rows:
                timestep = int(row["timestep"])
                positive = trigger - args.positive_before <= timestep <= trigger + args.positive_after
                # Keep failure trajectories well before the intervention as
                # hard NORMAL negatives; omit the ambiguous buffer.
                normal = timestep < trigger - args.positive_before - 4 and timestep % args.normal_stride == 0
                if not (positive or normal):
                    continue
                label = "RECOVERY" if positive else "NORMAL"
                item = record_with_label(row, label, {
                    "type": "audited_failure_trigger_window_v1", "trigger_timestep": trigger,
                    "positive_window": [trigger - args.positive_before, trigger + args.positive_after],
                }, source_key, language)
                # Use the trigger frame itself and every fifth positive as a
                # within-source validation probe; all remaining frames train.
                (val if positive and (timestep - trigger) % 5 == 0 else train).append(item)
                counts[label] += 1
        elif metadata.get("success"):
            # A complete nominal task supplies reliable NORMAL examples over
            # approach, grasp, transport and placement.
            for row in rows[::args.normal_stride]:
                train.append(record_with_label(row, "NORMAL", {"type": "completed_nominal_rollout_v1"}, source_key, language))
                counts["NORMAL"] += 1
    if counts["RECOVERY"] == 0 or counts["NORMAL"] == 0:
        raise ValueError(f"missing class after construction: {counts}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with (args.output_dir / f"{name}.jsonl").open("w") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")
    report = {"kind": "book_dense_gate_manifest_v1", "rollout_root": str(args.rollout_root),
              "triggers": triggers, "counts": counts, "train_rows": len(train), "val_rows": len(val),
              "online_inputs_permitted": ["main_rgb", "wrist_rgb", "language", "proprio", "executed_action_history"]}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
