"""Tensor sequence dataset for schema-v2 recovery transitions.

The first unified model intentionally starts from policy-observable low-dimensional
features. RGB/depth paths remain available for a later visual encoder, while this
dataset exposes stable modality masks for missing legacy fields.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from recovery.data.schema import RecoveryTransition
from recovery.data.sequence_builder import SequenceWindow, build_windows


PHASES = (
    "NOMINAL",
    "LOST GRASP",
    "FAILURE DETECTED",
    "RETURN TO BOWL",
    "DESCEND",
    "CLOSE GRIPPER",
    "VERIFY LIFT",
    "VISUAL SETTLE",
    "RESUMED TASK",
)
PHASE_TO_ID = {name: index for index, name in enumerate(PHASES)}
MODES = ("normal", "failure", "recovery", "verify", "handoff", "resumed", "unknown")
MODE_TO_ID = {name: index for index, name in enumerate(MODES)}
MODEL_INPUT_DIM = 76
CANDIDATE_GEOMETRY_SLICE = slice(65, 76)
CANDIDATE_GEOMETRY_DIM = 11

PROPRIO_LAYOUT = (
    ("robot0_eef_pos", 3),
    ("robot0_eef_quat", 4),
    ("robot0_gripper_qpos", 2),
    ("robot0_gripper_qvel", 2),
    ("robot0_joint_pos", 7),
    ("robot0_joint_vel", 7),
)
PROPRIO_DIM = sum(size for _, size in PROPRIO_LAYOUT)


def _finite_vector(value: Any, size: int) -> tuple[np.ndarray, float]:
    if value is None:
        return np.zeros(size, np.float32), 0.0
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if len(array) != size or not np.isfinite(array).all():
        return np.zeros(size, np.float32), 0.0
    return array, 1.0


def tensorize_transition(row: RecoveryTransition) -> dict[str, torch.Tensor]:
    online = row.policy_inputs()
    labels = row.labels
    proprio = online.get("proprio")
    proprio_parts = []
    proprio_masks = []
    if not isinstance(proprio, dict):
        proprio = {}
    for key, size in PROPRIO_LAYOUT:
        value, present = _finite_vector(proprio.get(key), size)
        proprio_parts.append(value)
        proprio_masks.extend([present] * size)

    nominal, _ = _finite_vector(online.get("nominal_action"), 7)
    previous, previous_valid = _finite_vector(
        online.get("previous_executed_action"), 7
    )
    executed, _ = _finite_vector(labels.get("executed_action"), 7)
    expert, expert_valid = _finite_vector(labels.get("expert_action"), 7)
    visual_xyz, visual_valid = _finite_vector(
        online.get("visual_target_xyz"), 3
    )
    visual_valid *= float(bool(online.get("visual_target_xyz_valid", False)))

    eef, eef_valid = _finite_vector(proprio.get("robot0_eef_pos"), 3)
    target_to_eef = visual_xyz - eef if visual_valid and eef_valid else np.zeros(3, np.float32)
    geometry_valid = visual_valid * eef_valid

    measurement = online.get("target_mask_measurement")
    measurement = measurement if isinstance(measurement, dict) else {}
    centroid, centroid_valid = _finite_vector(measurement.get("centroid_xy"), 2)
    centroid = centroid / 256.0
    visible = measurement.get("visible_pixels")
    margin = measurement.get("selection_margin")
    measurement_scalars = np.asarray(
        [
            float(visible) / 4096.0 if visible is not None else 0.0,
            float(np.tanh(float(margin) / 10.0)) if margin is not None else 0.0,
        ],
        dtype=np.float32,
    )
    measurement_mask = np.asarray(
        [float(visible is not None), float(margin is not None)], np.float32
    )

    phase_id = PHASE_TO_ID.get(str(labels.get("phase")), -1)
    mode_id = MODE_TO_ID.get(row.mode, MODE_TO_ID["unknown"])
    teacher_events_valid = float(bool(labels.get("teacher_event_labels_valid", False)))
    failure_onset = labels.get("failure_onset_teacher")
    failure_active = labels.get("failure_active")
    attached = labels.get("attached")
    lifted = labels.get("lifted")
    slipped = labels.get("slipped")
    verified = labels.get("verified")
    handoff_ready = labels.get("handoff_ready")
    expert_residual = expert - nominal if expert_valid else np.zeros(7, np.float32)
    return {
        "proprio": torch.from_numpy(np.concatenate(proprio_parts)),
        "proprio_mask": torch.tensor(proprio_masks, dtype=torch.float32),
        "nominal_action": torch.from_numpy(nominal),
        "previous_action": torch.from_numpy(previous),
        "previous_action_valid": torch.tensor(previous_valid, dtype=torch.float32),
        "executed_action": torch.from_numpy(executed),
        "expert_action": torch.from_numpy(expert),
        "expert_action_valid": torch.tensor(expert_valid, dtype=torch.float32),
        "expert_residual": torch.from_numpy(expert_residual),
        "visual_target_xyz": torch.from_numpy(visual_xyz),
        "visual_target_xyz_valid": torch.tensor(visual_valid, dtype=torch.float32),
        "target_to_eef": torch.from_numpy(target_to_eef),
        "target_to_eef_valid": torch.tensor(geometry_valid, dtype=torch.float32),
        "target_centroid": torch.from_numpy(centroid),
        "target_centroid_valid": torch.tensor(centroid_valid, dtype=torch.float32),
        "target_measurement": torch.from_numpy(measurement_scalars),
        "target_measurement_mask": torch.from_numpy(measurement_mask),
        "phase_id": torch.tensor(phase_id, dtype=torch.long),
        "phase_valid": torch.tensor(float(phase_id >= 0), dtype=torch.float32),
        "mode_id": torch.tensor(mode_id, dtype=torch.long),
        "failure_onset_target": torch.tensor(
            float(bool(failure_onset)) if failure_onset is not None else 0.0,
            dtype=torch.float32,
        ),
        "failure_onset_valid": torch.tensor(
            teacher_events_valid if failure_onset is not None else 0.0,
            dtype=torch.float32,
        ),
        "current_failure_target": torch.tensor(
            float(bool(failure_active)) if failure_active is not None
            else float(row.mode == "failure"),
            dtype=torch.float32,
        ),
        "current_failure_valid": torch.tensor(
            teacher_events_valid if failure_active is not None else 1.0,
            dtype=torch.float32,
        ),
        "recovery_active_target": torch.tensor(
            float(row.mode in ("recovery", "verify", "handoff")),
            dtype=torch.float32,
        ),
        "verification_target": torch.tensor(
            float(bool(verified)) if verified is not None
            else float(row.mode == "verify"),
            dtype=torch.float32,
        ),
        "verification_valid": torch.tensor(
            teacher_events_valid if verified is not None else 1.0,
            dtype=torch.float32,
        ),
        "handoff_target": torch.tensor(
            float(bool(handoff_ready)) if handoff_ready is not None
            else float(
                row.mode in ("handoff", "resumed")
                and bool(labels.get("recovery_completed"))
            ),
            dtype=torch.float32,
        ),
        "handoff_valid": torch.tensor(
            teacher_events_valid if handoff_ready is not None else 1.0,
            dtype=torch.float32,
        ),
        "grasped_target": torch.tensor(
            float(bool(attached)) if attached is not None else 0.0,
            dtype=torch.float32,
        ),
        "grasped_valid": torch.tensor(
            teacher_events_valid if attached is not None else 0.0,
            dtype=torch.float32,
        ),
        "lifted_target": torch.tensor(
            float(bool(lifted)) if lifted is not None else 0.0,
            dtype=torch.float32,
        ),
        "lifted_valid": torch.tensor(
            teacher_events_valid if lifted is not None else 0.0,
            dtype=torch.float32,
        ),
        "slip_target": torch.tensor(
            float(bool(slipped)) if slipped is not None else 0.0,
            dtype=torch.float32,
        ),
        "slip_valid": torch.tensor(
            teacher_events_valid if slipped is not None else 0.0,
            dtype=torch.float32,
        ),
        "failure_injection": torch.tensor(
            float(bool(labels.get("failure_injection"))), dtype=torch.float32
        ),
        "failure_detection_event": torch.tensor(
            float(bool(labels.get("failure_detection_event"))), dtype=torch.float32
        ),
        "recovery_completed": torch.tensor(
            float(bool(labels.get("recovery_completed"))), dtype=torch.float32
        ),
        "success_after": torch.tensor(
            float(bool(labels.get("success_after"))), dtype=torch.float32
        ),
        "timestep": torch.tensor(row.timestep, dtype=torch.long),
    }


class RecoverySequenceDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: Sequence[RecoveryTransition],
        *,
        window_length: int,
        stride: int = 1,
        event_only: bool = False,
    ):
        event_labels = (
            "failure_injection",
            "failure_detection_event",
            "recovery_expert_action",
            "recovery_completed",
        ) if event_only else ()
        self.windows = build_windows(
            rows,
            length=window_length,
            stride=stride,
            event_labels=event_labels,
        )
        if not self.windows:
            raise ValueError("no valid sequence windows were constructed")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window: SequenceWindow = self.windows[index]
        steps = [tensorize_transition(row) for row in window.rows]
        keys = steps[0].keys()
        return {
            "sequence": {
                key: torch.stack([step[key] for step in steps])
                for key in keys
            },
            "metadata": {
                "task_suite": window.task_suite,
                "task_id": window.task_id,
                "episode_id": window.episode_id,
                "split": window.split,
                "start_timestep": window.start_timestep,
                "end_timestep": window.end_timestep,
            },
        }


def collate_recovery_sequences(
    samples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    keys = samples[0]["sequence"].keys()
    return {
        "sequence": {
            key: torch.stack([sample["sequence"][key] for sample in samples])
            for key in keys
        },
        "metadata": [sample["metadata"] for sample in samples],
    }


def build_recovery_model_inputs(
    sequence: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Concatenate only policy-observable tensors into the v1 temporal input."""
    parts = (
        sequence["proprio"],
        sequence["proprio_mask"],
        sequence["nominal_action"],
        sequence["previous_action"],
        sequence["previous_action_valid"].unsqueeze(-1),
        sequence["target_to_eef"],
        sequence["target_to_eef_valid"].unsqueeze(-1),
        sequence["target_centroid"],
        sequence["target_centroid_valid"].unsqueeze(-1),
        sequence["target_measurement"],
        sequence["target_measurement_mask"],
    )
    result = torch.cat(parts, dim=-1)
    if result.shape[-1] != MODEL_INPUT_DIM:
        raise ValueError(
            f"unexpected recovery model input dim {result.shape[-1]} "
            f"(expected {MODEL_INPUT_DIM})"
        )
    return result
