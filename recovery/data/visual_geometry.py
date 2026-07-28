"""Observable mask+depth geometry used by recovery data and online policies."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def metric_depth(depth_buffer: np.ndarray, near: float, far: float) -> np.ndarray:
    return near / (1.0 - depth_buffer * (1.0 - near / far))


def target_point_cloud(
    depth_path: str | Path,
    mask_path: str | Path,
    *,
    camera_to_world: np.ndarray,
    near: float,
    far: float,
) -> np.ndarray:
    depth_buffer = np.load(depth_path)["depth"]
    depth = metric_depth(depth_buffer, near, far)
    mask = np.load(mask_path)["mask"]
    if mask.shape != depth.shape:
        if depth.shape[0] % mask.shape[0] or depth.shape[1] % mask.shape[1]:
            raise ValueError(
                f"mask {mask.shape} cannot be expanded to depth {depth.shape}"
            )
        mask = np.repeat(
            np.repeat(mask, depth.shape[0] // mask.shape[0], axis=0),
            depth.shape[1] // mask.shape[1],
            axis=1,
        )
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        raise ValueError(f"empty target mask: {mask_path}")
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > 0)
    ys, xs, z = ys[valid], xs[valid], z[valid]
    if len(z) == 0:
        raise ValueError(f"target mask has no finite positive depth: {depth_path}")

    # Saved RGB/depth/mask use both-axis reversal. Robosuite's camera utility
    # expresses rows in display convention, so only x is restored.
    camera_x = (depth.shape[1] - 1) - xs.astype(np.float64)
    camera_y = ys.astype(np.float64)
    homogeneous = np.stack(
        (camera_x * z, camera_y * z, z, np.ones_like(z)), axis=1
    )
    return (camera_to_world @ homogeneous.T).T[:, :3]


def estimate_target_xyz(
    depth_path: str | Path,
    mask_path: str | Path,
    *,
    camera_to_world: np.ndarray,
    near: float,
    far: float,
    calibration_bias: np.ndarray | None = None,
) -> np.ndarray:
    world = target_point_cloud(
        depth_path,
        mask_path,
        camera_to_world=camera_to_world,
        near=near,
        far=far,
    )
    estimate = np.median(world, axis=0)
    if calibration_bias is not None:
        estimate = estimate - np.asarray(calibration_bias, dtype=np.float64)
    return estimate.astype(np.float32)
