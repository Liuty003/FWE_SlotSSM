"""Policy-visible RGB-D geometry utilities used by the V4 collector.

LIBERO / robosuite depth images are OpenGL depth-buffer values, not metres.
The camera transform emitted by ``robosuite.utils.camera_utils`` maps the
homogeneous vector ``[pixel_x * z, pixel_y * z, z, 1]`` into a target frame.
All functions below preserve the saved-image convention used by OFT: RGB and
depth are rotated by 180 degrees before being written to disk.
"""

from __future__ import annotations

import numpy as np


def metric_depth(depth_buffer: np.ndarray, near: float, far: float) -> np.ndarray:
    """Convert a robosuite normalized depth buffer to metric depth."""
    if depth_buffer.ndim != 2 or not np.isfinite(depth_buffer).all():
        raise ValueError("depth_buffer must be a finite [H,W] array")
    if not 0 < near < far:
        raise ValueError("expected 0 < near < far")
    return near / (1.0 - depth_buffer * (1.0 - near / far))


def saved_depth_to_robot_points(depth_buffer: np.ndarray, camera_to_robot: np.ndarray,
                                near: float, far: float) -> tuple[np.ndarray, np.ndarray]:
    """Return robot-frame point image and a conservative validity mask.

    ``camera_to_robot`` is a fixed 4x4 calibration matrix whose source camera
    uses robosuite's native orientation.  The input uses OFT's saved
    orientation, hence ``camera_x = W - 1 - saved_x`` and
    ``camera_y = saved_y``.
    """
    if camera_to_robot.shape != (4, 4):
        raise ValueError("camera_to_robot must be [4,4]")
    height, width = depth_buffer.shape
    depth = metric_depth(depth_buffer, near, far)
    ys, xs = np.indices((height, width), dtype=np.float64)
    camera_x = width - 1.0 - xs
    homogeneous = np.stack((camera_x * depth, ys * depth, depth, np.ones_like(depth)), axis=-1)
    transformed = homogeneous @ camera_to_robot.T
    w = transformed[..., 3:4]
    points = transformed[..., :3] / np.where(np.abs(w) > 1e-12, w, 1.0)
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(depth) & (depth_buffer > 0.0) & (depth_buffer < 1.0)
    points = np.where(valid[..., None], points, 0.0).astype(np.float32)
    return points, valid


def patch_median_points(points: np.ndarray, valid: np.ndarray, grid_height: int = 16,
                        grid_width: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool a dense point image into tokenizer-aligned patch geometry.

    The returned xy coordinates are normalized in the *saved* primary-camera
    image frame.  They are the 2D-relation baseline input; 3D uses points.
    """
    if points.ndim != 3 or points.shape[-1] != 3 or valid.shape != points.shape[:2]:
        raise ValueError("points must be [H,W,3] and valid must be [H,W]")
    height, width = valid.shape
    if height % grid_height or width % grid_width:
        raise ValueError("image dimensions must divide the patch grid")
    patch_h, patch_w = height // grid_height, width // grid_width
    pooled, pooled_valid, xy = [], [], []
    for row in range(grid_height):
        for col in range(grid_width):
            region_valid = valid[row * patch_h:(row + 1) * patch_h, col * patch_w:(col + 1) * patch_w]
            region = points[row * patch_h:(row + 1) * patch_h, col * patch_w:(col + 1) * patch_w]
            if region_valid.any():
                pooled.append(np.median(region[region_valid], axis=0))
                pooled_valid.append(True)
            else:
                pooled.append(np.zeros(3, dtype=np.float32))
                pooled_valid.append(False)
            xy.append(((col + 0.5) / grid_width, (row + 0.5) / grid_height))
    return np.asarray(pooled, dtype=np.float32), np.asarray(pooled_valid, dtype=bool), np.asarray(xy, dtype=np.float32)


def robot_point_to_saved_image_xy(point_robot: np.ndarray, camera_to_robot: np.ndarray,
                                  height: int, width: int) -> tuple[np.ndarray, bool]:
    """Project a robot-frame point into normalized saved-image coordinates."""
    if point_robot.shape != (3,) or camera_to_robot.shape != (4, 4):
        raise ValueError("point_robot must be [3] and camera_to_robot must be [4,4]")
    camera = np.linalg.inv(camera_to_robot) @ np.concatenate((point_robot, np.ones(1)))
    if not np.isfinite(camera).all() or camera[2] <= 1e-8:
        return np.zeros(2, dtype=np.float32), False
    native_x, native_y = camera[0] / camera[2], camera[1] / camera[2]
    saved = np.asarray(((width - 1.0 - native_x) / width, native_y / height), dtype=np.float32)
    return saved, bool(np.isfinite(saved).all() and np.all((saved >= 0.0) & (saved <= 1.0)))


def patch_instance_labels(instance: np.ndarray | None, grid_height: int = 16,
                          grid_width: int = 16) -> np.ndarray:
    """Majority-pool simulator instance IDs for training-only Slot targets.

    ``-1`` denotes unavailable supervision, ``0`` background, and positive
    values identify an instance.  The IDs are not exposed to deployment.
    """
    if instance is None:
        return np.full(grid_height * grid_width, -1, dtype=np.int64)
    if instance.ndim != 2:
        raise ValueError("instance segmentation must be [H,W]")
    height, width = instance.shape
    if height % grid_height or width % grid_width:
        raise ValueError("instance image dimensions must divide the patch grid")
    patch_h, patch_w = height // grid_height, width // grid_width
    labels = []
    for row in range(grid_height):
        for col in range(grid_width):
            values = instance[row * patch_h:(row + 1) * patch_h, col * patch_w:(col + 1) * patch_w].reshape(-1)
            positive = values[values > 0]
            if positive.size == 0:
                labels.append(0)
            else:
                identities, counts = np.unique(positive, return_counts=True)
                labels.append(int(identities[counts.argmax()]))
    return np.asarray(labels, dtype=np.int64)
