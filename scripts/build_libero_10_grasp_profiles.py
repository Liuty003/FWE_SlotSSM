#!/usr/bin/env python3
"""Mine task-agnostic object-relative grasp profiles from successful OFT rollouts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument("--minimum-examples", type=int, default=5)
    parser.add_argument("--lift-height", type=float, default=0.02)
    parser.add_argument("--held-distance", type=float, default=0.15)
    parser.add_argument("--pre-lift-window", type=int, default=40)
    return parser.parse_args()


def rotation_wxyz(q) -> np.ndarray:
    w, x, y, z = (float(value) for value in q)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def rotation_xyzw(q) -> np.ndarray:
    x, y, z, w = (float(value) for value in q)
    return rotation_wxyz((w, x, y, z))


def matrix_to_xyzw(matrix: np.ndarray) -> np.ndarray:
    # Stable branch-based conversion, returned in robosuite's xyzw convention.
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
            w = (matrix[2, 1] - matrix[1, 2]) / scale
        elif axis == 1:
            scale = math.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
            w = (matrix[0, 2] - matrix[2, 0]) / scale
        else:
            scale = math.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
            w = (matrix[1, 0] - matrix[0, 1]) / scale
    quaternion = np.asarray((x, y, z, w), dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def average_quaternion(quaternions: list[np.ndarray]) -> np.ndarray:
    scatter = sum(np.outer(q, q) for q in quaternions)
    _, vectors = np.linalg.eigh(scatter)
    result = vectors[:, -1]
    if result[3] < 0:
        result = -result
    return result / np.linalg.norm(result)


def load_rows(path: str) -> list[dict]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def extract(rows: list[dict], target: str, args: argparse.Namespace):
    try:
        initial_z = float(rows[0]["privileged_audit"]["entities"][target]["position"][2])
    except (KeyError, IndexError, TypeError):
        return None
    lift = None
    for index, row in enumerate(rows):
        entity = row["privileged_audit"]["entities"].get(target)
        if entity is None:
            return None
        target_position = entity["position"]
        eef = row["online_observation"]["proprio"]["robot0_eef_pos"]
        if (
            float(target_position[2]) >= initial_z + args.lift_height
            and math.dist(target_position, eef) <= args.held_distance
        ):
            lift = index
            break
    if lift is None:
        return None
    candidates = []
    for index in range(max(0, lift - args.pre_lift_window), lift + 1):
        row = rows[index]
        if float(row["executed_action"][6]) <= 0:
            continue
        target_position = row["privileged_audit"]["entities"][target]["position"]
        eef = row["online_observation"]["proprio"]["robot0_eef_pos"]
        candidates.append((math.dist(target_position, eef), index))
    if not candidates:
        return None
    _, anchor = min(candidates)
    row = rows[anchor]
    entity = row["privileged_audit"]["entities"][target]
    target_position = np.asarray(entity["position"], dtype=np.float64)
    target_rotation = rotation_wxyz(entity["quaternion_wxyz"])
    proprio = row["online_observation"]["proprio"]
    eef_position = np.asarray(proprio["robot0_eef_pos"], dtype=np.float64)
    eef_rotation = rotation_xyzw(proprio["robot0_eef_quat"])
    return {
        "offset": target_rotation.T @ (eef_position - target_position),
        "relative_quaternion": matrix_to_xyzw(target_rotation.T @ eef_rotation),
        "anchor": anchor,
    }


def main() -> None:
    args = arguments()
    if min(args.max_examples, args.minimum_examples, args.pre_lift_window) < 1:
        raise ValueError("profile counts and window must be positive")
    route = json.loads(args.route_manifest.read_text())
    requested = sorted({
        (record["family_name"], record["target_name"])
        for record in route["records"]
        if record.get("teacher_skill") == "relational_pick_place"
    })
    profiles = {}
    for family, target in requested:
        summary_path = args.family_root / family / "baseline-summary.json"
        summary = json.loads(summary_path.read_text())
        examples = []
        for artifact in summary["artifacts"]:
            if not artifact["success"] or len(examples) >= args.max_examples:
                continue
            extracted = extract(load_rows(artifact["steps"]), target, args)
            if extracted is not None:
                examples.append({**extracted, "task_id": artifact["task_id"]})
        if len(examples) < args.minimum_examples:
            continue
        offsets = np.stack([example["offset"] for example in examples])
        median = np.median(offsets, axis=0)
        profiles[f"{family}:{target}"] = {
            "family_name": family,
            "target_name": target,
            "frame": "target_body",
            "grasp_offset": median.tolist(),
            "grasp_offset_mad": np.median(np.abs(offsets - median), axis=0).tolist(),
            "relative_grasp_quaternion_xyzw": average_quaternion([
                example["relative_quaternion"] for example in examples
            ]).tolist(),
            "num_examples": len(examples),
            "source_task_ids": [example["task_id"] for example in examples],
            "anchor_rule": "closest closed EEF pose in 40 steps before first verified target lift",
        }
    payload = {
        "kind": "libero_10_success_demonstration_grasp_profiles_v1",
        "schema_version": 1,
        "route_manifest": str(args.route_manifest.resolve()),
        "family_root": str(args.family_root.resolve()),
        "online_input_contract": "offline teacher generation only; profile is not an online model input",
        "num_profiles": len(profiles),
        "profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "num_profiles": len(profiles)}, indent=2))


if __name__ == "__main__":
    main()
