#!/usr/bin/env python
"""Preflight LIBERO instance segmentation and deterministic rollout replay.

This script deliberately does not load OpenVLA or the Stage-1 model.  It first
establishes that the simulator-provided instance masks, image orientation, and
recorded action replay are trustworthy enough to serve as Stage-A ground truth.
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
os.environ.setdefault(
    "LIBERO_CONFIG_PATH",
    str(LIBERO_ROOT / "baseline_libero_config"),
)

DEFAULT_SOURCES = (
    "/data2/lty/naive-e-slotssm-runs/openvla-online-control/task9-episode0",
    "/data2/lty/naive-e-slotssm-runs/openvla-standalone-repeat-v1/task9-episode6",
    "/data2/lty/naive-e-slotssm-runs/target-recovery-online-v2-replan/task9-episode7",
)
OBJECTS = ("akita_black_bowl_1", "akita_black_bowl_2", "plate_1")
COLORS = {
    "akita_black_bowl_1": np.array([255, 55, 55], dtype=np.uint8),
    "akita_black_bowl_2": np.array([55, 155, 255], dtype=np.uint8),
    "plate_1": np.array([80, 230, 100], dtype=np.uint8),
    "robot": np.array([245, 200, 45], dtype=np.uint8),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/stage-a-task9-segmentation-preflight"),
    )
    parser.add_argument("--max-frames", type=int, default=2)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--eef-tolerance-m", type=float, default=0.003)
    parser.add_argument("--object-tolerance-m", type=float, default=0.003)
    parser.add_argument(
        "--stop-on-drift", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def segmentation_key(obs: dict) -> str:
    candidates = [
        key for key in obs
        if key.startswith("agentview_segmentation") and "eye_in_hand" not in key
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one agentview instance segmentation key, got {candidates}")
    return candidates[0]


def orient(array: np.ndarray) -> np.ndarray:
    """Match the RGB convention used by the existing OpenVLA rollout scripts."""
    return np.ascontiguousarray(array[::-1, ::-1])


def masks_from_ids(
    segmentation: np.ndarray, instance_to_id: dict[str, int], robot_ids: list[int]
):
    segmentation = np.asarray(segmentation).squeeze().copy()
    masks = {name: segmentation == instance_to_id[name] for name in OBJECTS}
    masks["robot"] = np.isin(segmentation, robot_ids)
    return masks


def color_segmentation(masks: dict[str, np.ndarray]) -> np.ndarray:
    height, width = next(iter(masks.values())).shape
    output = np.zeros((height, width, 3), dtype=np.uint8)
    for name, mask in masks.items():
        output[mask] = COLORS[name]
    return output


def overlay(rgb: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    result = rgb.astype(np.float32).copy()
    for name, mask in masks.items():
        result[mask] = 0.55 * result[mask] + 0.45 * COLORS[name]
    return np.clip(result, 0, 255).astype(np.uint8)


def vector_error(actual, expected) -> float:
    return float(np.linalg.norm(np.asarray(actual, np.float64) - np.asarray(expected, np.float64)))


def state_errors(obs: dict, row: dict) -> tuple[float, dict[str, float]]:
    robot = row.get("robot_state_before", {})
    eef_error = vector_error(obs["robot0_eef_pos"], robot["robot0_eef_pos"])
    privileged = row.get("privileged_object_state_before", {})
    errors = {}
    for name in OBJECTS:
        key = f"{name}_pos"
        if key in privileged and key in obs:
            errors[name] = vector_error(obs[key], privileged[key])
    return eef_error, errors


def action_from_row(row: dict) -> np.ndarray:
    if "executed_action" not in row:
        raise KeyError("Recorded row has no executed_action")
    action = np.asarray(row["executed_action"], dtype=np.float32)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError(f"Invalid executed_action shape/value: {action}")
    return action


def run_source(source: Path, args: argparse.Namespace) -> dict:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv

    metadata = load_json(source / "metadata.json")
    rows = load_rows(source / "steps.jsonl")
    task_suite = metadata.get("task_suite", "libero_spatial")
    task_id = int(metadata.get("task_id", 9))
    episode_id = int(metadata.get("episode_id", rows[0].get("episode_id", -1)))
    if episode_id < 0:
        raise ValueError(f"Could not determine episode id for {source}")

    suite = benchmark.get_benchmark_dict()[task_suite]()
    task = suite.get_task(task_id)
    initial_states = suite.get_task_init_states(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
    )
    env.seed(0)
    env.reset()
    obs = env.set_init_state(initial_states[episode_id])
    no_op = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    wait_steps = int(metadata.get("wait_steps", 20))
    for _ in range(wait_steps):
        obs, _, done, _ = env.step(no_op)
        if done:
            break

    missing = [name for name in OBJECTS if name not in env.instance_to_id]
    if missing:
        raise RuntimeError(
            f"Missing required instances {missing}; available={sorted(env.instance_to_id)}"
        )
    simulator_instances = list(env.env.model.instances_to_ids.keys())
    robot_instances = [
        name for name in simulator_instances
        if "panda" in name.lower() or "gripper" in name.lower() or "mount" in name.lower()
    ]
    robot_instance_ids = [simulator_instances.index(name) + 1 for name in robot_instances]
    if env.segmentation_robot_id is None:
        print(
            "WARNING: SegmentationRenderEnv did not resolve Panda0 robot id; "
            f"simulator instances={simulator_instances}",
            flush=True,
        )

    output = args.output_dir / f"task{task_id}-episode{episode_id}"
    output.mkdir(parents=True, exist_ok=True)
    frame_reports = []
    try:
        for frame_id, row in enumerate(rows[: args.max_frames]):
            key = segmentation_key(obs)
            rgb = orient(np.asarray(obs["agentview_image"]))
            raw_seg = np.asarray(obs[key]).squeeze().copy()
            raw_masks = masks_from_ids(
                raw_seg, env.instance_to_id, robot_instance_ids
            )
            masks = {name: orient(mask) for name, mask in raw_masks.items()}
            eef_error, object_errors = state_errors(obs, row)
            drift = eef_error > args.eef_tolerance_m or any(
                value > args.object_tolerance_m for value in object_errors.values()
            )
            report = {
                "frame": frame_id,
                "segmentation_key": key,
                "rgb_shape": list(rgb.shape),
                "segmentation_shape": list(raw_seg.shape),
                "unique_segmentation_ids": sorted(int(value) for value in np.unique(raw_seg)),
                "visible_pixels": {name: int(mask.sum()) for name, mask in masks.items()},
                "eef_replay_error_m": eef_error,
                "object_replay_error_m": object_errors,
                "drift": drift,
            }
            frame_reports.append(report)
            imageio.imwrite(output / f"frame-{frame_id:03d}-rgb.png", rgb)
            imageio.imwrite(
                output / f"frame-{frame_id:03d}-segmentation.png",
                color_segmentation(masks),
            )
            imageio.imwrite(
                output / f"frame-{frame_id:03d}-overlay.png", overlay(rgb, masks)
            )
            if frame_id < 2 or frame_id % 25 == 0 or drift or frame_id + 1 == min(len(rows), args.max_frames):
                print(
                    f"episode={episode_id} frame={frame_id} eef_error={eef_error:.6f} "
                    f"object_max={max(object_errors.values(), default=0.0):.6f} drift={drift}",
                    flush=True,
                )
            if drift and args.stop_on_drift:
                break
            obs, _, done, _ = env.step(action_from_row(row).tolist())
            if done:
                break
    finally:
        env.close()

    result = {
        "source": str(source),
        "task_suite": task_suite,
        "task_id": task_id,
        "episode_id": episode_id,
        "task_language": task.language,
        "wait_steps": wait_steps,
        "instance_to_id": {name: int(value) for name, value in env.instance_to_id.items()},
        "segmentation_robot_id": (
            int(env.segmentation_robot_id) if env.segmentation_robot_id is not None else None
        ),
        "simulator_instances": simulator_instances,
        "robot_instances": robot_instances,
        "robot_instance_ids": robot_instance_ids,
        "orientation": "both axes reversed for RGB and segmentation",
        "frames": frame_reports,
        "passed_replay_check": bool(frame_reports) and not any(r["drift"] for r in frame_reports),
    }
    (output / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    args = arguments()
    if args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [run_source(Path(source), args) for source in args.sources]
    summary = {
        "schema_version": 1,
        "purpose": "Stage-A instance segmentation and replay preflight",
        "all_passed_replay_check": all(r["passed_replay_check"] for r in reports),
        "episodes": reports,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "all_passed_replay_check": summary["all_passed_replay_check"],
    }, indent=2))


if __name__ == "__main__":
    main()
