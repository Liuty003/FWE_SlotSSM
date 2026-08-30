#!/usr/bin/env python3
"""Validate that archival LIBERO-Plus rollouts satisfy the no-recollection contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


SCHEMA = "oft_archival_rollout_v4"
TIMESTEP_KEYS = {
    "rgb", "wrist_rgb", "depth", "wrist_depth",
    "agentview_instance", "wrist_instance", "mujoco_state", "ctrl",
    "mocap_pos", "mocap_quat", "camera_xpos", "camera_xmat",
    "numpy_rng_keys", "numpy_rng_position", "numpy_rng_has_gaussian",
    "numpy_rng_cached_gaussian",
}
TERMINAL_KEYS = {
    "terminal_mujoco_state", "terminal_ctrl", "terminal_mocap_pos",
    "terminal_mocap_quat", "terminal_camera_xpos", "terminal_camera_xmat",
    "terminal_numpy_rng_keys", "terminal_numpy_rng_position",
    "terminal_numpy_rng_has_gaussian", "terminal_numpy_rng_cached_gaussian",
    "official_initial_state",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_rollout(metadata_path: Path, verify_checksum: bool) -> dict:
    directory = metadata_path.parent
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema") != SCHEMA or metadata.get("complete") is not True:
        raise ValueError(f"{directory}: incomplete or wrong schema")
    steps_path = directory / "steps.jsonl"
    bundle_path = directory / "dense-audit.npz"
    for path in (steps_path, bundle_path, directory / "rollout.mp4", directory / "wrist-rollout.mp4"):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = [json.loads(line) for line in steps_path.read_text().splitlines() if line.strip()]
    transitions = int(metadata["transitions"])
    if len(rows) != transitions or transitions < 1:
        raise ValueError(f"{directory}: transition count mismatch")
    if verify_checksum and sha256(bundle_path) != metadata.get("dense_bundle_sha256"):
        raise ValueError(f"{directory}: dense bundle checksum mismatch")
    for index, row in enumerate(rows):
        if int(row["timestep"]) != index:
            raise ValueError(f"{directory}: non-contiguous timestep {index}")
        if int(row["action_chunk_offset"]) == 0 and row["planned_action_chunk"] is None:
            raise ValueError(f"{directory}: missing action chunk at {index}")
        if int(row["action_chunk_offset"]) != 0 and row["planned_action_chunk"] is not None:
            raise ValueError(f"{directory}: duplicate action chunk at {index}")
        proprio = row["online_observation"]["proprio"]
        required_proprio = {
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
            "robot0_gripper_qvel", "robot0_joint_pos", "robot0_joint_vel",
        }
        if not required_proprio.issubset(proprio):
            raise ValueError(f"{directory}: incomplete proprio at {index}")
        if "privileged_audit" not in row:
            raise ValueError(f"{directory}: missing privileged audit at {index}")
    with np.load(bundle_path, allow_pickle=False) as bundle:
        missing = (TIMESTEP_KEYS | TERMINAL_KEYS) - set(bundle.files)
        if missing:
            raise ValueError(f"{directory}: missing dense keys {sorted(missing)}")
        for key in TIMESTEP_KEYS:
            if bundle[key].shape[0] != transitions:
                raise ValueError(f"{directory}: {key} length mismatch")
        if bundle["rgb"].dtype != np.uint8 or bundle["wrist_rgb"].dtype != np.uint8:
            raise ValueError(f"{directory}: RGB must be lossless uint8")
        if bundle["depth"].dtype != np.float32 or bundle["wrist_depth"].dtype != np.float32:
            raise ValueError(f"{directory}: depth must be float32")
        if bundle["mujoco_state"].shape[1:] != bundle["terminal_mujoco_state"].shape:
            raise ValueError(f"{directory}: terminal MuJoCo state shape mismatch")
    terminal = metadata.get("terminal_privileged_audit")
    if not terminal or bool(terminal.get("task_success")) != bool(metadata["success"]):
        raise ValueError(f"{directory}: terminal success label mismatch")
    return {
        "sample_id": f"{metadata['task_suite']}:task{metadata['task_id']}:episode{metadata['episode_id']}",
        "success": bool(metadata["success"]),
        "transitions": transitions,
        "bytes": bundle_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int)
    parser.add_argument("--skip-checksum", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metadata_paths = sorted(args.root.rglob("metadata.json"))
    if args.expected_rollouts is not None and len(metadata_paths) != args.expected_rollouts:
        raise ValueError(f"expected {args.expected_rollouts} rollouts, found {len(metadata_paths)}")
    records = [validate_rollout(path, not args.skip_checksum) for path in metadata_paths]
    if len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("duplicate sample ids")
    payload = {
        "kind": "libero_plus_archival_validation_v1",
        "root": str(args.root.resolve()),
        "rollouts": len(records),
        "successes": sum(row["success"] for row in records),
        "transitions": sum(row["transitions"] for row in records),
        "bundle_bytes": sum(row["bytes"] for row in records),
        "schema_counts": dict(Counter([SCHEMA] * len(records))),
        "records": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "rollouts", "successes", "transitions", "bundle_bytes", "schema_counts"
    )}, indent=2))


if __name__ == "__main__":
    main()
