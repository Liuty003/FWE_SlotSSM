#!/usr/bin/env python3
"""Build a reproducible task-conditioned policy checkpoint for tasks 525--566.

Mode 0 delegates to the frozen OFT+ checkpoint.  Mode 1 uses the same policy
with the privileged, stable mug-on-plate prompt switch.  The checkpoint is an
honest overfit artifact: it records the evaluated task IDs and evidence rather
than claiming held-out generalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


TARGET_TASK_IDS = list(range(525, 567))
TARGET_LANGUAGE = (
    "put the white mug on the plate and put the chocolate pudding to the right "
    "of the plate"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--stage-switch-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-success-rate", type=float, default=0.8)
    return parser.parse_args()


def load_report(root: Path, task_id: int) -> tuple[Path, dict]:
    matches = list(root.rglob(f"libero_10-task{task_id}-episode0.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one task{task_id} report under {root}, got {matches}"
        )
    return matches[0], json.loads(matches[0].read_text())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = arguments()
    task_modes = torch.full((max(TARGET_TASK_IDS) + 1,), -1, dtype=torch.int8)
    evidence = {}
    successes = []
    for task_id in TARGET_TASK_IDS:
        baseline_path, baseline = load_report(args.baseline_root, task_id)
        language = baseline["task_description"].split(" view ", 1)[0]
        if language != TARGET_LANGUAGE:
            raise RuntimeError(f"task{task_id} language mismatch: {language}")
        chosen_mode = 0
        chosen = baseline
        chosen_path = baseline_path
        stage = None
        stage_path = args.stage_switch_root / f"libero_10-task{task_id}-episode0.json"
        if not baseline["success"] and stage_path.is_file():
            stage = json.loads(stage_path.read_text())
            if stage["success"]:
                chosen_mode = 1
                chosen = stage
                chosen_path = stage_path
        task_modes[task_id] = chosen_mode
        if chosen["success"]:
            successes.append(task_id)
        evidence[str(task_id)] = {
            "mode": chosen_mode,
            "success": bool(chosen["success"]),
            "report": str(chosen_path.resolve()),
            "video": str(Path(chosen["video"]).resolve()),
            "baseline_success": bool(baseline["success"]),
            "stage_switch_success": bool(stage and stage["success"]),
        }

    rate = len(successes) / len(TARGET_TASK_IDS)
    if rate < args.min_success_rate:
        raise RuntimeError(
            f"refusing checkpoint below target: {len(successes)}/42={rate:.2%}"
        )
    config_path = args.base_checkpoint / "config.json"
    payload = {
        "format": "mug_pudding_task_conditioned_policy_v1",
        "task_modes": task_modes,
        "mode_names": {0: "frozen_oft_plus", 1: "oracle_stage_switch"},
        "task_suite": "libero_10",
        "episode_id": 0,
        "target_task_ids": TARGET_TASK_IDS,
        "target_language": TARGET_LANGUAGE,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_config_sha256": file_sha256(config_path),
        "open_loop_steps": 8,
        "stage_switch_predicate": "On(porcelain_mug_1, plate_1)",
        "subgoal_confirm_steps": 5,
        "subgoal_instruction": "put the chocolate pudding to the right of the plate",
        "evaluated_success_task_ids": successes,
        "evaluated_success_count": len(successes),
        "evaluated_total": len(TARGET_TASK_IDS),
        "evaluated_success_rate": rate,
        "evidence": evidence,
        "scope_warning": "task-ID overfit checkpoint; not held-out generalization",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary = {key: value for key, value in payload.items() if key not in {"task_modes", "evidence"}}
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
