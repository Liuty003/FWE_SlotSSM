#!/usr/bin/env python
"""Audit mask+depth target 3D estimates against teacher-only simulator pose."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = REPO_ROOT / "experiments" / "robot" / "libero"
sys.path.insert(0, str(LIBERO_ROOT))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_ROOT / "baseline_libero_config"))


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def camera_parameters():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(9)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
        camera_depths=True,
    )
    env.reset()
    sim = env.env.sim
    camera_to_world = np.linalg.inv(
        get_camera_transform_matrix(sim, "agentview", 256, 256)
    )
    extent = sim.model.stat.extent
    far = sim.model.vis.map.zfar * extent
    near = sim.model.vis.map.znear * extent
    env.close()
    return camera_to_world, near, far


def real_depth(depth, near, far):
    return near / (1.0 - depth * (1.0 - near / far))


def point_cloud(row, camera_to_world, near, far):
    depth = real_depth(np.load(row["depth"])["depth"], near, far)
    mask = np.load(row["target_mask_64"])["mask"]
    mask = np.repeat(np.repeat(mask, 4, axis=0), 4, axis=1).astype(bool)
    ys, xs = np.where(mask)
    z = depth[ys, xs]
    # Saved RGB/depth/mask use both-axis reversal. Robosuite's camera utility
    # already expresses rows in display convention, so only x is restored.
    camera_x = 255.0 - xs.astype(np.float64)
    camera_y = ys.astype(np.float64)
    homogeneous = np.stack(
        (camera_x * z, camera_y * z, z, np.ones_like(z)), axis=1
    )
    world = (camera_to_world @ homogeneous.T).T[:, :3]
    return world, z


def estimates(world, depth):
    order = np.argsort(depth)
    count = len(order)
    return {
        "all_median": np.median(world, axis=0),
        "nearest_75_median": np.median(world[order[:max(1, int(.75 * count))]], axis=0),
        "nearest_50_median": np.median(world[order[:max(1, int(.50 * count))]], axis=0),
        "nearest_25_median": np.median(world[order[:max(1, int(.25 * count))]], axis=0),
        "all_mean": world.mean(axis=0),
    }


def summary(errors):
    values = np.asarray(errors)
    return {
        "num_frames": len(values),
        "mean_error_cm": float(100 * values.mean()),
        "median_error_cm": float(100 * np.median(values)),
        "p90_error_cm": float(100 * np.quantile(values, .9)),
        "max_error_cm": float(100 * values.max()),
    }


def main():
    args = arguments()
    manifest = json.loads(args.manifest.read_text())
    camera_to_world, near, far = camera_parameters()
    by_split = {}
    signed_by_split = {}
    per_episode = []
    frame_estimates = {}
    for record in manifest["episodes"]:
        if not record["eligible_for_recovery_bc"]:
            continue
        rows = [
            json.loads(line)
            for line in Path(record["transitions"]).read_text().splitlines() if line
        ]
        rows = [row for row in rows if row["is_recovery_expert_action"]]
        episode_errors = {}
        for row in rows:
            world, depth = point_cloud(row, camera_to_world, near, far)
            candidates = estimates(world, depth)
            teacher = np.asarray(
                row["teacher_privileged_object_state"]["akita_black_bowl_1_pos"]
            )
            key = f"{record['episode_id']}:{row['timestep']}"
            frame_estimates[key] = {
                name: value.tolist() for name, value in candidates.items()
            }
            for name, value in candidates.items():
                error = float(np.linalg.norm(value - teacher))
                episode_errors.setdefault(name, []).append(error)
                by_split.setdefault(record["split"], {}).setdefault(name, []).append(error)
                signed_by_split.setdefault(record["split"], {}).setdefault(
                    name, []
                ).append(value - teacher)
        per_episode.append({
            "episode_id": record["episode_id"], "split": record["split"],
            "methods": {
                name: summary(values) for name, values in episode_errors.items()
            },
        })
    development = {
        name: by_split["train"][name] + by_split["validation"][name]
        for name in by_split["train"]
    }
    selected_method = min(
        development, key=lambda name: np.mean(development[name])
    )
    calibration_bias = np.mean(
        signed_by_split["train"][selected_method]
        + signed_by_split["validation"][selected_method],
        axis=0,
    )
    calibrated = {}
    for split in signed_by_split:
        calibrated[split] = [
            float(np.linalg.norm(error - calibration_bias))
            for error in signed_by_split[split][selected_method]
        ]
    result = {
        "schema_version": 1,
        "kind": "teacher_only_visual_3d_audit",
        "manifest": str(args.manifest),
        "camera": {
            "name": "agentview", "width": 256, "height": 256,
            "saved_to_camera_orientation": "camera_y=saved_y; camera_x=255-saved_x",
            "near": near, "far": far,
        },
        "selection": {
            "selected_method": selected_method,
            "criterion": "minimum train+validation mean xyz error",
            "teacher_pose_used_for_method_selection": True,
        },
        "splits": {
            split: {
                name: summary(values) for name, values in methods.items()
            }
            for split, methods in by_split.items()
        },
        "selected_test": summary(by_split["test"][selected_method]),
        "constant_bias_calibration": {
            "estimated_on": "train+validation",
            "bias_estimate_minus_teacher_m": calibration_bias.tolist(),
            "splits": {
                split: summary(values) for split, values in calibrated.items()
            },
        },
        "episodes": per_episode,
        "frame_estimates": frame_estimates,
        "calibrated_frame_estimates": {
            key: (
                np.asarray(methods[selected_method]) - calibration_bias
            ).tolist()
            for key, methods in frame_estimates.items()
        },
        "claim_limit": (
            "teacher object pose is used only for estimator audit and method "
            "selection; estimates themselves consume predicted mask and depth"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "selection": result["selection"],
        "selected_test": result["selected_test"],
        "splits": result["splits"],
    }, indent=2))


if __name__ == "__main__":
    main()
