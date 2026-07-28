from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import json

import torch

from recovery.data.conversion import convert_legacy_expert_episode
from recovery.data.dataset import read_jsonl, write_jsonl
from recovery.data.schema import RecoveryTransition, SCHEMA_VERSION
from recovery.data.sequence_builder import build_windows
from recovery.data.tensor_dataset import (
    PROPRIO_DIM,
    RecoverySequenceDataset,
    build_recovery_model_inputs,
    collate_recovery_sequences,
)
from recovery.models.recovery_slotssm import RecoverySlotSSM


def transition(timestep: int, *, split: str = "train") -> RecoveryTransition:
    return RecoveryTransition(
        schema_version=SCHEMA_VERSION,
        source_format="test",
        source_path="/tmp/source.jsonl",
        task_suite="libero_spatial",
        task_id=9,
        episode_id=0,
        timestep=timestep,
        split=split,
        mode="normal",
        online_inputs={
            "rgb": None,
            "depth": None,
            "language_instruction": "test",
            "proprio": {"eef": [0.0, 0.0, 0.0]},
            "previous_executed_action": None if timestep == 0 else [0.0] * 7,
            "nominal_action": [0.0] * 7,
            "target_mask": None,
            "target_mask_measurement": None,
            "visual_target_xyz": None,
            "visual_target_xyz_valid": False,
            "observation_validity": {},
        },
        labels={"executed_action": [0.0] * 7},
        teacher_only={"raw_privileged_object_state": {"target": [1, 2, 3]}},
    )


class DataContractTest(unittest.TestCase):
    def test_collector_v2_contact_labels_survive_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "transitions.jsonl"
            raw = {
                "schema_version": 2,
                "task_suite": "libero_spatial",
                "task_id": 9,
                "episode_id": 2,
                "timestep": 0,
                "phase": "CLOSE GRIPPER",
                "is_failure_injection": False,
                "is_failure_detection_event": False,
                "is_recovery_expert_action": True,
                "recovery_completed": False,
                "rgb": None,
                "depth": None,
                "target_mask_64": None,
                "target_mask_measurement": None,
                "robot_state": {},
                "nominal_action": [0.0] * 7,
                "expert_or_executed_action": [0.0] * 6 + [1.0],
                "teacher_privileged_object_state": {},
                "teacher_contact_state": {
                    "target_gripper_contact": True,
                    "bilateral_finger_contact": True,
                },
                "teacher_attached_truth": True,
                "success_after": False,
            }
            source.write_text(json.dumps(raw) + "\n", encoding="utf-8")
            rows = convert_legacy_expert_episode(source, split="train")
            self.assertTrue(rows[0].labels["attached"])
            self.assertTrue(rows[0].labels["grasped"])
            self.assertTrue(rows[0].teacher_only["attached_truth"])
            self.assertEqual(
                rows[0].teacher_only["contact_state"],
                raw["teacher_contact_state"],
            )

    def test_policy_inputs_exclude_teacher_state(self) -> None:
        row = transition(0)
        row.validate()
        self.assertNotIn("raw_privileged_object_state", row.policy_inputs())

    def test_rejects_privileged_leak(self) -> None:
        row = transition(0)
        row.online_inputs["object_pose"] = [0.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "non-allowlisted"):
            row.validate()

    def test_jsonl_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            write_jsonl(path, [transition(0), transition(1)])
            restored = read_jsonl(path)
            self.assertEqual([row.timestep for row in restored], [0, 1])
            self.assertEqual(restored[1].online_inputs["nominal_action"], [0.0] * 7)

    def test_windows_never_cross_episode_or_split(self) -> None:
        rows = [transition(index) for index in range(4)]
        other = transition(0, split="validation")
        other.episode_id = 1
        rows.append(other)
        windows = build_windows(rows, length=3)
        self.assertEqual(len(windows), 2)
        self.assertTrue(all(window.episode_id == 0 for window in windows))

    def test_non_contiguous_rows_are_not_windowed(self) -> None:
        rows = [transition(0), transition(2)]
        self.assertEqual(build_windows(rows, length=2), [])

    def test_tensor_dataset_shapes_and_missing_masks(self) -> None:
        rows = [transition(index) for index in range(4)]
        dataset = RecoverySequenceDataset(rows, window_length=3)
        sample = dataset[0]
        self.assertEqual(sample["sequence"]["proprio"].shape, (3, PROPRIO_DIM))
        self.assertEqual(sample["sequence"]["nominal_action"].shape, (3, 7))
        self.assertEqual(
            sample["sequence"]["visual_target_xyz_valid"].sum().item(), 0
        )
        batch = collate_recovery_sequences([dataset[0], dataset[1]])
        self.assertEqual(batch["sequence"]["nominal_action"].shape, (2, 3, 7))
        self.assertEqual(
            build_recovery_model_inputs(batch["sequence"]).shape, (2, 3, 76)
        )

    def test_reference_temporal_model_forward_backward(self) -> None:
        model = RecoverySlotSSM(model_dim=32, num_layers=1, backend="gru")
        inputs = torch.randn(2, 4, 76)
        outputs = model(inputs)
        self.assertEqual(outputs["phase_logits"].shape, (2, 4, 9))
        self.assertEqual(outputs["failure_onset_logits"].shape, (2, 4))
        self.assertEqual(outputs["current_failure_logits"].shape, (2, 4))
        self.assertEqual(outputs["residual_action"].shape, (2, 4, 7))
        loss = (
            outputs["phase_logits"].mean()
            + outputs["failure_onset_logits"].mean()
            + outputs["residual_action"].mean()
        )
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
