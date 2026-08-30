#!/usr/bin/env python3
"""Materialize the task-specific LIBERO-Plus Robot book comparison.

This keeps the 34 Robot Initial States of one semantic task separate from the
global experiment and records the exact per-state OFT+ versus recovery result.
The report deliberately references existing rollout files instead of copying
videos or training data.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


BOOK_TASK = "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--global-manifest",
        type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/oft-plus-robot-global-v1/robot_manifest.json"),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/oft-plus-robot-global-v1/nominal_rollouts"),
    )
    parser.add_argument(
        "--recovery-root",
        type=Path,
        default=Path(
            "/data2/lty/naive-e-slotssm-runs/oft-plus-book-local-repair-v1/"
            "robot34-gated-recovery-online-v3-official-parity/full34-v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/oft-plus-robot-global-v1/semantic-task9-book-v1/comparison.json"),
    )
    return parser.parse_args()


def load_success(path: Path) -> bool:
    if not path.is_file():
        raise FileNotFoundError(path)
    return bool(json.loads(path.read_text())["success"])


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.global_manifest.read_text())
    rows = [row for row in manifest["records"] if row["semantic_task"] == BOOK_TASK]
    if len(rows) != 34:
        raise RuntimeError(f"expected 34 book Robot Initial States, found {len(rows)}")

    cases = []
    transitions: Counter[str] = Counter()
    for row in sorted(rows, key=lambda item: item["task_id"]):
        task_id = int(row["task_id"])
        baseline_path = args.baseline_root / f"libero_10-task{task_id}-episode0.json"
        recovery_path = args.recovery_root / f"task{task_id}" / f"task{task_id}-episode0.json"
        baseline_success = load_success(baseline_path)
        recovery_success = load_success(recovery_path)
        transition = f"{'success' if baseline_success else 'failure'}_to_{'success' if recovery_success else 'failure'}"
        transitions[transition] += 1
        cases.append(
            {
                "task_id": task_id,
                "difficulty_level": int(row["difficulty_level"]),
                "oft_plus_success": baseline_success,
                "task_expert_success": recovery_success,
                "transition": transition,
                "oft_plus_video": str(baseline_path.with_suffix(".mp4")),
                "task_expert_video": str(recovery_path.with_suffix(".mp4")),
            }
        )
    baseline_successes = sum(case["oft_plus_success"] for case in cases)
    recovery_successes = sum(case["task_expert_success"] for case in cases)
    payload = {
        "kind": "libero_plus_robot_semantic_task_expert_comparison_v1",
        "semantic_task_id": 9,
        "semantic_task": BOOK_TASK,
        "protocol": {
            "category": "Robot Initial States",
            "episodes_per_task_config": 1,
            "task_configs": len(cases),
            "baseline": "OpenVLA-OFT+ libero-plus-mixdata checkpoint",
            "expert": "task-specific SlotSSM gate plus local book-regrasp recovery",
            "normal_policy_parity": "official OFT+ action path when recovery is inactive",
            "scope": "in-distribution task-specific training/evaluation; task configurations may overlap training by design",
        },
        "summary": {
            "oft_plus": {"successes": baseline_successes, "total": len(cases), "success_rate": baseline_successes / len(cases)},
            "task_expert": {"successes": recovery_successes, "total": len(cases), "success_rate": recovery_successes / len(cases)},
            "absolute_gain": (recovery_successes - baseline_successes) / len(cases),
            "transitions": dict(sorted(transitions.items())),
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
