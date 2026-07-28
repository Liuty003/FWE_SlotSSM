#!/usr/bin/env python
"""Collect relation-aware instance masks from all LIBERO Spatial tasks.

Unlike the task-9 prototype, masks are stored as named arrays instead of a
single colour image.  This preserves both references required by ``between``
and keeps the machine-readable role schema as the only source of role labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = REPO_ROOT / "experiments" / "robot" / "libero"
sys.path.insert(0, str(LIBERO_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ["LIBERO_CONFIG_PATH"] = str(LIBERO_ROOT / "baseline_libero_config")

COLORS = {
    "target": np.array([255, 55, 55], dtype=np.uint8),
    "distractor": np.array([55, 155, 255], dtype=np.uint8),
    "goal": np.array([80, 230, 100], dtype=np.uint8),
    "reference_0": np.array([210, 90, 240], dtype=np.uint8),
    "reference_1": np.array([30, 220, 220], dtype=np.uint8),
    "counterfactual_target": np.array([255, 135, 30], dtype=np.uint8),
    "counterfactual_reference_0": np.array([160, 80, 255], dtype=np.uint8),
    "counterfactual_reference_1": np.array([20, 190, 190], dtype=np.uint8),
    "robot": np.array([245, 200, 45], dtype=np.uint8),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema", type=Path,
        default=LIBERO_ROOT / "stage_b_spatial_role_schema.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/stage-b-spatial-relation-data"),
    )
    parser.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--states", default="", help="State IDs; empty means every state")
    parser.add_argument("--max-states-per-task", type=int, default=0)
    parser.add_argument("--wait-steps", type=int, default=20)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--min-visible-pixels", type=int, default=16)
    parser.add_argument(
        "--fast-settle", action=argparse.BooleanOptionalAction, default=True,
        help="Disable camera observables during no-op settling and render only the final state",
    )
    return parser.parse_args()


def parse_ids(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def segmentation_key(obs: dict) -> str:
    keys = [
        key for key in obs
        if key.startswith("agentview_segmentation") and "eye_in_hand" not in key
    ]
    if len(keys) != 1:
        raise RuntimeError(f"Expected one agentview segmentation, got {keys}")
    return keys[0]


def orient(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array[::-1, ::-1])


def robot_ids(env) -> tuple[list[str], list[int]]:
    instances = list(env.env.model.instances_to_ids.keys())
    names = [
        name for name in instances
        if any(token in name.lower() for token in ("panda", "gripper", "mount"))
    ]
    return names, [instances.index(name) + 1 for name in names]


def make_masks(
    segmentation, env, roles, reference_names, counterfactual, robot_instance_ids
):
    masks = {
        role: orient(segmentation == env.instance_to_id[instance])
        for role, instance in roles.items()
        if role != "robot"
    }
    for index, instance in enumerate(reference_names):
        masks[f"reference_{index}"] = orient(
            segmentation == env.instance_to_id[instance]
        )
    masks["counterfactual_target"] = orient(
        segmentation == env.instance_to_id[counterfactual["target"]]
    )
    for index, instance in enumerate(counterfactual["references"]):
        masks[f"counterfactual_reference_{index}"] = orient(
            segmentation == env.instance_to_id[instance]
        )
    masks["robot"] = orient(np.isin(segmentation, robot_instance_ids))
    return masks


def visualization(rgb, masks):
    result = rgb.astype(np.float32).copy()
    for role, mask in masks.items():
        color = COLORS[role]
        result[mask] = 0.55 * result[mask] + 0.45 * color
    return np.clip(result, 0, 255).astype(np.uint8)


def camera_observable_names(env):
    return [
        name for name in env.env.observation_names
        if "image" in name or "depth" in name or "segmentation" in name
    ]


def set_camera_observables(env, enabled: bool):
    for name in camera_observable_names(env):
        env.env.modify_observable(name, "enabled", enabled)


def settle(env, obs, steps: int, fast: bool):
    """Advance identical physics while avoiding redundant intermediate renders."""
    no_op = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    if fast:
        set_camera_observables(env, False)
    try:
        for _ in range(steps):
            obs, _, done, _ = env.step(no_op)
            if done:
                break
    finally:
        if fast:
            set_camera_observables(env, True)
    if fast:
        env._update_observables(force=True)
        obs = env.env._get_observations()
    return obs


def main() -> None:
    args = arguments()
    if (
        args.wait_steps < 0 or args.camera_size < 1
        or args.max_states_per_task < 0 or args.min_visible_pixels < 1
    ):
        raise ValueError("Invalid wait/camera/max-state argument")
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    requested_tasks = parse_ids(args.tasks)
    requested_states = parse_ids(args.states)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv

    suite = benchmark.get_benchmark_dict()[schema["task_suite"]]()
    invalid_tasks = set(requested_tasks) - set(range(suite.n_tasks))
    if invalid_tasks:
        raise ValueError(f"Invalid task IDs: {sorted(invalid_tasks)}")
    shared = schema["shared_roles"]
    roles = {name: shared[name] for name in ("target", "distractor", "goal")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for task_id in requested_tasks:
        task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        state_ids = requested_states or list(range(len(states)))
        if args.max_states_per_task:
            state_ids = state_ids[: args.max_states_per_task]
        invalid_states = set(state_ids) - set(range(len(states)))
        if invalid_states:
            raise ValueError(f"task {task_id}: invalid state IDs {sorted(invalid_states)}")
        spec = schema["tasks"][str(task_id)]
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = SegmentationRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=args.camera_size,
            camera_widths=args.camera_size,
            camera_names=["agentview"],
            camera_depths=False,
            hard_reset=False,
        )
        env.seed(0)
        try:
            # SegmentationRenderEnv builds its instance mapping on first reset.
            env.reset()
            required = [
                *roles.values(), *spec["references"], spec["counterfactual"]["target"],
                *spec["counterfactual"]["references"],
            ]
            missing = [name for name in required if name not in env.instance_to_id]
            if missing:
                raise RuntimeError(f"task {task_id}: missing instances {missing}")
            robot_names, robot_instance_ids = robot_ids(env)
            if not robot_instance_ids:
                raise RuntimeError(f"task {task_id}: could not resolve robot instances")
            for state_id in state_ids:
                env.reset()
                if args.fast_settle:
                    set_camera_observables(env, False)
                obs = env.set_init_state(states[state_id])
                obs = settle(env, obs, args.wait_steps, args.fast_settle)
                rgb = orient(np.asarray(obs["agentview_image"]))
                segmentation = np.asarray(obs[segmentation_key(obs)]).squeeze()
                masks = make_masks(
                    segmentation, env, roles, spec["references"],
                    spec["counterfactual"], robot_instance_ids,
                )
                too_small = {
                    role: int(mask.sum()) for role, mask in masks.items()
                    if int(mask.sum()) < args.min_visible_pixels
                }
                if too_small:
                    raise RuntimeError(
                        f"task {task_id} state {state_id}: insufficient visibility {too_small}"
                    )
                prefix = args.output_dir / f"task{task_id}-state{state_id:03d}"
                imageio.imwrite(prefix.with_name(prefix.name + "-rgb.png"), rgb)
                imageio.imwrite(
                    prefix.with_name(prefix.name + "-overlay.png"),
                    visualization(rgb, masks),
                )
                mask_path = prefix.with_name(prefix.name + "-masks.npz")
                np.savez_compressed(mask_path, **{key: value.astype(np.uint8) for key, value in masks.items()})
                record = {
                    "task_id": task_id,
                    "state_id": state_id,
                    "language": task.language,
                    "relation": spec["relation"],
                    "role_to_instance": roles,
                    "references": spec["references"],
                    "counterfactual": spec["counterfactual"],
                    "rgb": str(prefix.with_name(prefix.name + "-rgb.png")),
                    "masks": str(mask_path),
                    "visible_pixels": {key: int(value.sum()) for key, value in masks.items()},
                    "robot_instances": robot_names,
                }
                records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
        finally:
            env.close()

    manifest = {
        "schema_version": 2,
        "task_suite": schema["task_suite"],
        "source_schema": str(args.schema),
        "sample_kind": "settled_initial_state",
        "wait_steps": args.wait_steps,
        "num_samples": len(records),
        "samples": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), "num_samples": len(records)}, indent=2))


if __name__ == "__main__":
    main()
