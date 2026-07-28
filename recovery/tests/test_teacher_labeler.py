from __future__ import annotations

import unittest

from recovery.data.schema import RecoveryTransition, SCHEMA_VERSION
from recovery.teachers.recovery_labeler import label_episode


def row(timestep: int, target_z: float, eef_z: float) -> RecoveryTransition:
    return RecoveryTransition(
        schema_version=SCHEMA_VERSION,
        source_format="test",
        source_path="/tmp/test",
        task_suite="libero_spatial",
        task_id=9,
        episode_id=0,
        timestep=timestep,
        split="train",
        mode="normal",
        online_inputs={
            "rgb": None,
            "depth": None,
            "language_instruction": "test",
            "proprio": {"robot0_eef_pos": [0.0, 0.0, eef_z]},
            "previous_executed_action": None if timestep == 0 else [0.0] * 7,
            "nominal_action": [0.0] * 7,
            "target_mask": None,
            "target_mask_measurement": None,
            "visual_target_xyz": None,
            "visual_target_xyz_valid": False,
            "observation_validity": {},
        },
        labels={
            "phase": "NOMINAL",
            "executed_action": [0.0] * 7,
            "failure_detection_event": False,
            "recovery_completed": False,
        },
        teacher_only={
            "raw_privileged_object_state": {
                "akita_black_bowl_1_pos": [0.0, 0.0, target_z]
            }
        },
    )


class TeacherLabelerTest(unittest.TestCase):
    def test_comotion_confirms_attachment_and_lift(self) -> None:
        rows = [
            row(0, 0.00, 0.06),
            row(1, 0.01, 0.07),
            row(2, 0.02, 0.08),
            row(3, 0.05, 0.11),
            row(4, 0.06, 0.12),
        ]
        report = label_episode(rows)
        self.assertGreaterEqual(report["counts"]["attached"], 2)
        self.assertTrue(rows[-1].labels["lifted"])
        self.assertTrue(rows[-1].labels["verified"])

    def test_failure_active_ends_on_recovery_complete(self) -> None:
        rows = [row(index, 0.0, 0.06) for index in range(3)]
        rows[0].labels["failure_detection_event"] = True
        rows[2].labels["recovery_completed"] = True
        label_episode(rows)
        self.assertTrue(rows[0].labels["failure_active"])
        self.assertTrue(rows[1].labels["failure_active"])
        self.assertFalse(rows[2].labels["failure_active"])

    def test_injection_starts_failure_without_legacy_detection(self) -> None:
        rows = [row(0, 0.00, 0.04), row(1, 0.00, 0.04), row(2, 0.00, 0.04)]
        rows[1].labels["failure_injection"] = True
        label_episode(rows, confirm_steps=1)
        self.assertFalse(rows[0].labels["failure_active"])
        self.assertFalse(rows[1].labels["failure_onset_teacher"])
        self.assertTrue(rows[2].labels["failure_onset_teacher"])
        self.assertEqual(
            rows[2].labels["failure_onset_reason"],
            "post_intervention_observation",
        )
        self.assertTrue(rows[2].labels["failure_active"])
        self.assertTrue(rows[2].teacher_only["oracle_failure_onset"])


if __name__ == "__main__":
    unittest.main()
