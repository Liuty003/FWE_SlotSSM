#!/usr/bin/env python3
"""Automatically assign coarse, task-agnostic failure labels to audit rollouts.

The classifier consumes only offline audit fields.  It intentionally emits a
low-confidence bucket rather than pretending ambiguous physics failures are
known; those rows can later be clustered for sparse human review.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--restrict-to-manifest", action="store_true",
        help="Ignore rollout records whose sample_id is absent from --manifest.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def vec_distance(a, b) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5


def classify(metadata: dict, steps: list[dict]) -> dict:
    if metadata.get("success"):
        return {"failure_type": "success", "confidence": "high", "candidate_skill": None}
    if not steps or "privileged_audit" not in steps[0]:
        return {
            "failure_type": "missing_privileged_audit", "confidence": "none",
            "candidate_skill": None,
        }

    audits = [step["privileged_audit"] for step in steps]
    goal_specs = [item["predicate"] for item in audits[-1]["goal_predicates"]]
    goal_values = [
        [item["value"] for item in audit["goal_predicates"]] for audit in audits
    ]
    supported = all(all(value is not None for value in values) for values in goal_values)
    max_goals = max((sum(value is True for value in values) for values in goal_values), default=0)
    final_goals = sum(value is True for value in goal_values[-1])
    total_goals = len(goal_values[-1])
    any_goal_regression = any(
        earlier is True and later is False
        for index in range(1, len(goal_values))
        for earlier, later in zip(goal_values[index - 1], goal_values[index])
    )
    ever_satisfied = [
        any(values[index] is True for values in goal_values)
        for index in range(total_goals)
    ]
    finally_satisfied = [value is True for value in goal_values[-1]]

    gripper = [step["online_observation"]["proprio"]["robot0_gripper_qpos"] for step in steps]
    aperture = [sum(abs(float(value)) for value in pair) for pair in gripper]
    close_attempt = bool(aperture and max(aperture) - min(aperture) >= 0.015)
    eef = [step["online_observation"]["proprio"]["robot0_eef_pos"] for step in steps]
    eef_path = sum(vec_distance(a, b) for a, b in zip(eef, eef[1:]))

    interests = set(audits[0].get("objects_of_interest", []))
    initial_entities = audits[0]["entities"]
    max_interest_motion = 0.0
    for name in interests:
        if name not in initial_entities:
            continue
        start = initial_entities[name]["position"]
        for audit in audits[1:]:
            if name in audit["entities"]:
                max_interest_motion = max(
                    max_interest_motion,
                    vec_distance(start, audit["entities"][name]["position"]),
                )

    # Only one-DoF joints are treated as articulated fixtures. Free object
    # joints also contain pose/quaternion values and must not enter this signal.
    max_fixture_joint_motion = 0.0
    initial_scalar_joints = {
        (entity, joint): float(qpos[0])
        for entity, state in initial_entities.items()
        for joint, qpos in (state.get("joint_qpos") or {}).items()
        if len(qpos) == 1
    }
    for audit in audits[1:]:
        for (entity, joint), start in initial_scalar_joints.items():
            qpos = (audit["entities"].get(entity, {}).get("joint_qpos") or {}).get(joint)
            if qpos and len(qpos) == 1:
                max_fixture_joint_motion = max(max_fixture_joint_motion, abs(float(qpos[0]) - start))

    predicate_names = [spec[0].lower() for spec in goal_specs]
    articulation_goal = any(name in {"open", "close", "turnon"} for name in predicate_names)
    relational_goal = any(name in {"on", "in"} for name in predicate_names)
    goal_status = [
        {
            "predicate": spec,
            "ever_satisfied": ever_satisfied[index],
            "final_satisfied": finally_satisfied[index],
        }
        for index, spec in enumerate(goal_specs)
    ]

    evidence = {
        "supported_goal_predicates": supported,
        "total_goal_predicates": total_goals,
        "max_satisfied_goal_predicates": max_goals,
        "final_satisfied_goal_predicates": final_goals,
        "goal_regression": any_goal_regression,
        "gripper_close_attempt": close_attempt,
        "eef_path_length_m": eef_path,
        "max_object_of_interest_motion_m": max_interest_motion,
        "max_fixture_joint_motion": max_fixture_joint_motion,
        "goal_status": goal_status,
        "remaining_goal_predicates": [
            spec for spec, done in zip(goal_specs, finally_satisfied) if not done
        ],
    }
    if not supported:
        label = ("unsupported_goal_predicate", "none", None)
    elif any_goal_regression:
        label = ("goal_regression", "high", "reposition_or_recover_subgoal")
    elif total_goals > 1 and 0 < max_goals < total_goals:
        label = ("partial_goal_completion", "high", "continue_remaining_subgoal")
    elif articulation_goal and not relational_goal and max_fixture_joint_motion < 0.01:
        label = ("articulation_contact_failure", "medium", "relocalize_and_recontact_fixture")
    elif articulation_goal and not relational_goal:
        label = ("articulation_incomplete", "medium", "continue_articulation")
    elif not close_attempt and eef_path < 0.08:
        label = ("progress_stall_before_manipulation", "high", "relocalize_and_approach")
    elif close_attempt and max_interest_motion < 0.025:
        label = ("empty_grasp_or_failed_contact", "medium", "regrasp")
    elif max_interest_motion >= 0.025 and max_goals == 0:
        label = ("transfer_or_placement_failure", "medium", "reposition")
    elif not close_attempt:
        label = ("no_grasp_attempt_or_articulation_failure", "low", "inspect_goal_type")
    else:
        label = ("unresolved_physical_failure", "low", None)
    return {
        "failure_type": label[0], "confidence": label[1],
        "candidate_skill": label[2], "evidence": evidence,
    }


def main() -> None:
    args = parse_args()
    if args.restrict_to_manifest and args.manifest is None:
        raise ValueError("--restrict-to-manifest requires --manifest")
    manifest = None
    if args.manifest:
        payload = json.loads(args.manifest.read_text())
        manifest = {row["sample_id"]: row for row in payload["records"]}
    records = []
    for metadata_path in sorted(args.rollout_root.rglob("metadata.json")):
        steps_path = metadata_path.with_name("steps.jsonl")
        if not steps_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        sample_id = (
            f"{metadata['task_suite']}:task{metadata['task_id']}:episode{metadata['episode_id']}"
        )
        if manifest is not None and args.restrict_to_manifest and sample_id not in manifest:
            continue
        row = {
            "sample_id": sample_id,
            "metadata": str(metadata_path.resolve()),
            "steps": str(steps_path.resolve()),
            **classify(metadata, load_jsonl(steps_path)),
        }
        if manifest and sample_id in manifest:
            official = manifest[sample_id]
            row.update({
                "suite": official["suite"],
                "perturbation_category": official["perturbation_category"],
                "difficulty_level": official["difficulty_level"],
            })
        records.append(row)
    counts = Counter(row["failure_type"] for row in records)
    confidence = Counter(row["confidence"] for row in records)
    payload = {
        "kind": "libero_plus_automatic_failure_manifest_v1",
        "rollout_root": str(args.rollout_root.resolve()),
        "rollouts": len(records),
        "failure_type_counts": dict(sorted(counts.items())),
        "confidence_counts": dict(sorted(confidence.items())),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: payload[k] for k in (
        "rollouts", "failure_type_counts", "confidence_counts"
    )}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
