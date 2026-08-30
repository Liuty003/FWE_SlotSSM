#!/usr/bin/env python3
"""Convert saved autonomous recovery rollouts into 3-state gate supervision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ROOT = Path("/data2/lty/naive-e-slotssm-runs/oft-plus-book-local-repair-v1/autonomous-full-recovery-data-v1")
DEFAULT_VAL_TASKS = {521, 523}


def args_() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normal-stride", type=int, default=4)
    return parser.parse_args()


def environment_action(raw: list[float]) -> list[float]:
    result = list(map(float, raw))
    result[-1] = -float((2.0 * result[-1] - 1.0 > 0.0) - (2.0 * result[-1] - 1.0 < 0.0))
    return result


def main() -> None:
    args = args_()
    if args.normal_stride < 1:
        raise ValueError("--normal-stride must be positive")
    train, val, splits = [], [], []
    counts = {"NORMAL": 0, "RECOVERY": 0, "RECOVERY_COMPLETE": 0}
    for report_path in sorted(args.input_root.glob("task*-episode0/task*-episode0.json")):
        report = json.loads(report_path.read_text())
        task_id = int(report_path.parent.name.split("-")[0].removeprefix("task"))
        if not report["success"]:
            continue
        source_key = f"autonomous-book-task{task_id}-episode0"
        steps = []
        first_recovery = next((int(item["step"]) for item in report["decisions"] if item["mode"] == "RECOVERY"), None)
        if first_recovery is None:
            continue
        for item in report["decisions"]:
            observation = item.get("online_observation")
            if observation is None:
                raise ValueError(f"{report_path} lacks --save-online-images records")
            phase = item.get("anchor_fsm_phase")
            if item["mode"] != "RECOVERY":
                label = "NORMAL"
            elif phase == 7:
                label = "RECOVERY_COMPLETE"
            else:
                label = "RECOVERY"
            row = {
                "kind": "oft_nominal_transition", "task_suite": "libero_10", "task_id": task_id,
                "episode_id": 0, "timestep": int(item["step"]), "online_observation": observation,
                "planned_action": item["action"], "executed_action": environment_action(item["action"]),
                "gate_label": label,
            }
            steps.append(row)
        # ``steps.jsonl`` retains every causal frame.  The manifest below can
        # subsample NORMAL rows without corrupting SlotSSM/action history.
        rollout_dir = args.output_dir / "rollouts" / source_key
        rollout_dir.mkdir(parents=True, exist_ok=True)
        (rollout_dir / "metadata.json").write_text(json.dumps({"task_language": report["task"], "success": True}, indent=2) + "\n")
        with (rollout_dir / "steps.jsonl").open("w") as file:
            for row in steps:
                file.write(json.dumps(row) + "\n")
                label = row["gate_label"]
                if label == "NORMAL" and int(row["timestep"]) % args.normal_stride:
                    continue
                record = {"schema_version": 1, "kind": "book_execution_gate_window", "source_key": source_key,
                          "timestep": row["timestep"], "language": report["task"],
                          "online_observation": row["online_observation"], "gate_label": label,
                          "gate_label_provenance": {"type": "autonomous_fsm_phase_v1", "phase": next(d.get("anchor_fsm_phase") for d in report["decisions"] if int(d["step"]) == row["timestep"])} }
                (val if task_id in DEFAULT_VAL_TASKS else train).append(record)
                counts[label] += 1
        splits.append({"source_key": source_key, "split": "val" if task_id in DEFAULT_VAL_TASKS else "train", "metadata": str(rollout_dir / "metadata.json")})
    if not counts["RECOVERY_COMPLETE"]:
        raise RuntimeError("no RECOVERY_COMPLETE labels")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with (args.output_dir / f"{name}.jsonl").open("w") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")
    (args.output_dir / "source_splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    report = {"kind": "autonomous_completion_gate_manifest_v1", "counts": counts, "train_rows": len(train), "val_rows": len(val), "sources": len(splits)}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
