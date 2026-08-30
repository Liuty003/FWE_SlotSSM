"""Raw, aligned observations for natural-failure recovery branches.

This intentionally stores a pre-action observation together with the recovery
action executed from it.  It is a collection format, not a policy input schema:
the frozen OpenVLA nominal action is unavailable after a counterfactual fork,
so a later enrichment pass must compute it rather than silently copying the
recovery action into a field called ``nominal_action``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np


def oriented_rgb(observation: Mapping[str, Any]) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(observation["agentview_image"])[::-1, ::-1]
    )


def oriented_wrist_rgb(observation: Mapping[str, Any]) -> np.ndarray:
    """Return the wrist view in the same orientation as OpenVLA-OFT input."""
    return np.ascontiguousarray(
        np.asarray(observation["robot0_eye_in_hand_image"])[::-1, ::-1]
    )


def oriented_depth(observation: Mapping[str, Any]) -> np.ndarray | None:
    value = observation.get("agentview_depth")
    if value is None:
        return None
    return np.ascontiguousarray(np.asarray(value)[::-1, ::-1].squeeze())


def oriented_wrist_depth(observation: Mapping[str, Any]) -> np.ndarray | None:
    value = observation.get("robot0_eye_in_hand_depth")
    if value is None:
        return None
    return np.ascontiguousarray(np.asarray(value)[::-1, ::-1].squeeze())


def oriented_instance_segmentation(observation: Mapping[str, Any]) -> np.ndarray | None:
    value = observation.get("agentview_segmentation_instance")
    if value is None:
        return None
    return np.ascontiguousarray(np.asarray(value)[::-1, ::-1].squeeze())


def target_mask_measurement(
    observation: Mapping[str, Any], instance_id: int | None
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """Return a 64x64 instance mask in the same orientation as RGB.

    ``SegmentationRenderEnv`` uses a single agentview instance segmentation
    observation.  A missing target is represented by an all-zero mask and a
    zero-pixel measurement, rather than by a privileged object position.
    """
    if instance_id is None:
        return None, None
    keys = [
        key for key in observation
        if key.startswith("agentview_segmentation") and "eye_in_hand" not in key
    ]
    if len(keys) != 1:
        raise RuntimeError(
            f"expected one agentview segmentation observation, got {keys}"
        )
    segmentation = np.asarray(observation[keys[0]]).squeeze()[::-1, ::-1]
    mask = segmentation == instance_id
    height, width = mask.shape
    if height % 64 or width % 64:
        raise RuntimeError(f"cannot downsample mask {mask.shape} to 64x64")
    mask_64 = mask.reshape(64, height // 64, 64, width // 64).any(
        axis=(1, 3)
    ).astype(np.uint8)
    ys, xs = np.where(mask)
    return mask_64, {
        "visible_pixels": int(len(xs)),
        "centroid_xy": (
            [float(xs.mean()), float(ys.mean())] if len(xs) else None
        ),
    }


def robot_state(observation: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_gripper_qvel",
        "robot0_joint_pos",
        "robot0_joint_vel",
    )
    return {
        key: np.asarray(observation[key], np.float32).tolist()
        for key in keys
        if key in observation
    }


class BranchTrajectoryWriter:
    """Write one raw sequence per candidate branch with strict provenance."""

    def __init__(
        self,
        root: Path,
        *,
        task_suite: str,
        task_id: int,
        source_episode_id: int,
        failure_timestep: int,
        candidate_name: str,
        candidate_index: int,
        source_rollout: str,
        failure_type: str | None,
        target_name: str,
        target_instance_id: int | None,
        source_nominal_action: list[float] | None,
    ):
        # The source episode remains in provenance.  A branch needs its own
        # sequence identity so multiple counterfactual branches never collide
        # at the same (task, episode, timestep) key downstream.
        self.branch_episode_id = source_episode_id * 1000 + candidate_index
        self.target_name = target_name
        self.target_instance_id = target_instance_id
        self.source_nominal_action = source_nominal_action
        self.directory = root / candidate_name
        self.rgb_dir = self.directory / "rgb"
        self.wrist_rgb_dir = self.directory / "wrist_rgb"
        self.depth_dir = self.directory / "depth"
        self.wrist_depth_dir = self.directory / "wrist_depth"
        self.instance_dir = self.directory / "instance"
        self.mask_dir = self.directory / "target_masks"
        for path in (self.rgb_dir, self.wrist_rgb_dir, self.depth_dir,
                     self.wrist_depth_dir, self.instance_dir, self.mask_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "raw-transitions.jsonl"
        self.path.write_text("", encoding="utf-8")
        self.common = {
            "schema_version": 1,
            "kind": "natural_failure_recovery_branch_transition",
            "task_suite": task_suite,
            "task_id": task_id,
            "branch_episode_id": self.branch_episode_id,
            "source_episode_id": source_episode_id,
            "failure_timestep": failure_timestep,
            "candidate_name": candidate_name,
            "candidate_index": candidate_index,
            "source_rollout": source_rollout,
            "failure_type": failure_type,
            "target_name": target_name,
            "online_input_contract": (
                "agentview_rgb/wrist_rgb/depth/mask/proprio/previous action/"
                "recovery action only; "
                "teacher fields are never deployment inputs"
            ),
        }

    def record(
        self,
        *,
        step: int,
        observation: Mapping[str, Any],
        recovery_action: np.ndarray,
        previous_recovery_action: np.ndarray | None,
        teacher_contact_before: Mapping[str, Any],
        teacher_attached_before: bool,
        teacher_target_position_before: np.ndarray,
        teacher_contact_after: Mapping[str, Any],
        teacher_attached_after: bool,
        teacher_target_position_after: np.ndarray,
        task_success_after: bool,
        done_after: bool,
    ) -> None:
        frame = f"step-{step:04d}"
        rgb_path = self.rgb_dir / f"{frame}.png"
        wrist_rgb_path = self.wrist_rgb_dir / f"{frame}.png"
        depth_path = self.depth_dir / f"{frame}.npz"
        wrist_depth_path = self.wrist_depth_dir / f"{frame}.npz"
        instance_path = self.instance_dir / f"{frame}.npz"
        mask_path = self.mask_dir / f"{frame}.npz"
        imageio.imwrite(rgb_path, oriented_rgb(observation))
        imageio.imwrite(wrist_rgb_path, oriented_wrist_rgb(observation))
        depth = oriented_depth(observation)
        if depth is not None:
            np.savez_compressed(depth_path, depth=depth.astype(np.float32))
        else:
            depth_path = None
        wrist_depth = oriented_wrist_depth(observation)
        if wrist_depth is not None:
            np.savez_compressed(wrist_depth_path, depth=wrist_depth.astype(np.float32))
        else:
            wrist_depth_path = None
        instance = oriented_instance_segmentation(observation)
        if instance is not None:
            np.savez_compressed(instance_path, instance=instance.astype(np.int32))
        else:
            instance_path = None
        mask, measurement = target_mask_measurement(
            observation, self.target_instance_id
        )
        if mask is not None:
            np.savez_compressed(mask_path, mask=mask)
        else:
            mask_path = None
        record = {
            **self.common,
            "branch_timestep": step,
            "online_observation": {
                "rgb": str(rgb_path.resolve()),
                "wrist_rgb": str(wrist_rgb_path.resolve()),
                "depth": str(depth_path.resolve()) if depth_path else None,
                "wrist_depth": str(wrist_depth_path.resolve()) if wrist_depth_path else None,
                "target_mask": str(mask_path.resolve()) if mask_path else None,
                "target_mask_measurement": measurement,
                "proprio": robot_state(observation),
                "previous_recovery_action": (
                    None if previous_recovery_action is None
                    else np.asarray(previous_recovery_action, np.float32).tolist()
                ),
                "recovery_action": np.asarray(recovery_action, np.float32).tolist(),
                # This action is intentionally only a provenance reference;
                # it is the baseline action at fork time, not a fabricated
                # OpenVLA prediction for later counterfactual observations.
                "source_nominal_action_at_fork": self.source_nominal_action,
            },
            "training_only": {
                "instance_segmentation": str(instance_path.resolve()) if instance_path else None,
            },
            "teacher_only": {
                "contact_before": dict(teacher_contact_before),
                "attached_before": bool(teacher_attached_before),
                "target_position_before": np.asarray(
                    teacher_target_position_before, np.float32
                ).tolist(),
                "contact_after": dict(teacher_contact_after),
                "attached_after": bool(teacher_attached_after),
                "target_position_after": np.asarray(
                    teacher_target_position_after, np.float32
                ).tolist(),
                "target_lift_after_m": float(
                    np.asarray(teacher_target_position_after)[2]
                    - np.asarray(teacher_target_position_before)[2]
                ),
                "task_success_after": bool(task_success_after),
                "done_after": bool(done_after),
            },
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
