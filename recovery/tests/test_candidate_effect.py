from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import torch

from recovery.data.candidate_schema import (
    CANDIDATE_SCHEMA_VERSION,
    RecoveryCandidateGroup,
    read_candidate_jsonl,
    write_candidate_jsonl,
)
from recovery.models.candidate_effect_scorer import (
    RecoveryCandidateEffectScorer,
    candidate_effect_losses,
    generate_recovery_candidates,
)
from recovery.data.candidate_tensor_dataset import (
    RecoveryCandidateDataset,
    UnifiedRecoveryCandidateDataset,
    collate_recovery_candidates,
)
from recovery.tests.test_data_contract import transition
from recovery.models.recovery_slotssm import RecoverySlotSSM
from recovery.models.unified_recovery import UnifiedRecoveryModel


def candidate_group() -> RecoveryCandidateGroup:
    outcomes = [
        {
            "eef_delta_m": [0.0, 0.0, 0.0],
            "target_delta_m": [0.0, 0.0, 0.0],
            "attached_after": False,
            "lifted_after": False,
            "slipped_after": False,
            "failure_persistent_after": True,
            "handoff_ready_after": False,
            "teacher_value": -1.0,
        },
        {
            "eef_delta_m": [0.0, 0.0, 0.01],
            "target_delta_m": [0.0, 0.0, 0.01],
            "attached_after": True,
            "lifted_after": True,
            "slipped_after": False,
            "failure_persistent_after": False,
            "handoff_ready_after": True,
            "teacher_value": 2.0,
        },
    ]
    return RecoveryCandidateGroup(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        task_suite="libero_spatial",
        task_id=9,
        episode_id=0,
        timestep=42,
        split="train",
        context_transition_path="/tmp/transitions.jsonl#42",
        nominal_action=[0.0] * 7,
        candidate_names=["nominal", "lift"],
        candidate_actions=[[0.0] * 7, [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, -1.0]],
        outcome_valid=[True, True],
        teacher_outcomes=outcomes,
        provenance={"branch_horizon": 4},
    )


class CandidateEffectTest(unittest.TestCase):
    def test_candidate_generator_is_bounded_and_distinct(self) -> None:
        nominal = torch.tensor([0.95, 0.0, 0.0, 0.0, 0.0, 0.9, -1.0])
        residual = torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0])
        names, actions = generate_recovery_candidates(nominal, residual)
        self.assertGreaterEqual(len(names), 2)
        self.assertEqual(actions.shape, (len(names), 7))
        self.assertTrue(torch.all(actions.abs() <= 1.0))
        self.assertEqual(len({tuple(row.tolist()) for row in actions}), len(names))
        self.assertTrue(torch.equal(actions[names.index("nominal")], nominal))

    def test_candidate_schema_round_trip_and_duplicate_rejection(self) -> None:
        group = candidate_group()
        group.validate()
        restored = RecoveryCandidateGroup.from_dict(group.to_dict())
        self.assertEqual(restored.candidate_names, ["nominal", "lift"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.jsonl"
            write_candidate_jsonl(path, [group])
            self.assertEqual(read_candidate_jsonl(path)[0].timestep, 42)
        restored.candidate_actions[1] = restored.candidate_actions[0]
        with self.assertRaisesRegex(ValueError, "distinct"):
            restored.validate()

    def test_scorer_shapes_permutation_and_action_sensitivity(self) -> None:
        torch.manual_seed(7)
        model = RecoveryCandidateEffectScorer(context_dim=76, hidden_dim=32).eval()
        context = torch.randn(2, 76)
        nominal = torch.randn(2, 7).clamp(-1.0, 1.0)
        candidates = torch.randn(2, 5, 7).clamp(-1.0, 1.0)
        outputs = model(context, nominal, candidates)
        self.assertEqual(outputs["eef_delta_m"].shape, (2, 5, 3))
        self.assertEqual(outputs["uncertainty_raw"].shape, (2, 5, 2))
        self.assertEqual(outputs["candidate_value"].shape, (2, 5))

        permutation = torch.tensor([3, 0, 4, 1, 2])
        permuted = model(context, nominal, candidates[:, permutation])
        for name, value in outputs.items():
            self.assertTrue(torch.allclose(permuted[name], value[:, permutation]))

        self.assertGreater(
            torch.std(outputs["candidate_value"][0]).item(),
            0.0,
        )

    def test_dataset_collation_and_loss_backward(self) -> None:
        group = candidate_group()
        key = ("libero_spatial", 9, 0, 42)
        row = transition(42)
        dataset = RecoveryCandidateDataset([group], {key: row})
        batch = collate_recovery_candidates([dataset[0]])
        self.assertEqual(batch["context"].shape, (1, 76))
        self.assertEqual(batch["candidate_actions"].shape, (1, 2, 7))
        model = RecoveryCandidateEffectScorer(context_dim=76, hidden_dim=32)
        outputs = model(
            batch["context"],
            batch["nominal_action"],
            batch["candidate_actions"],
        )
        losses = candidate_effect_losses(outputs, batch)
        losses["total"].backward()
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertGreater(float(losses["ranking"]), 0.0)

    def test_unified_model_shares_temporal_state_without_changing_old_heads(
        self,
    ) -> None:
        group = candidate_group()
        transitions = {
            ("libero_spatial", 9, 0, timestep): transition(timestep)
            for timestep in range(43)
        }
        dataset = UnifiedRecoveryCandidateDataset(
            [group], transitions, window_length=8
        )
        batch = collate_recovery_candidates([dataset[0]])
        recovery = RecoverySlotSSM(
            model_dim=32, num_layers=1, backend="gru"
        ).eval()
        baseline = recovery(batch["context"])
        unified = UnifiedRecoveryModel(
            recovery, candidate_hidden_dim=32
        ).eval()
        outputs = unified(
            batch["context"],
            nominal_action=batch["nominal_action"],
            candidate_actions=batch["candidate_actions"],
            candidate_timestep=batch["candidate_timestep"],
        )
        self.assertTrue(
            torch.equal(outputs["phase_logits"], baseline["phase_logits"])
        )
        candidate = outputs["candidate"]
        self.assertEqual(candidate["candidate_value"].shape, (1, 2))
        losses = candidate_effect_losses(candidate, batch)
        losses["total"].backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in unified.candidate_head.parameters()
            )
        )
        geometry_model = UnifiedRecoveryModel(
            RecoverySlotSSM(model_dim=32, num_layers=1, backend="gru"),
            candidate_hidden_dim=32,
            geometry_skip=True,
        )
        geometry_outputs = geometry_model(
            batch["context"],
            nominal_action=batch["nominal_action"],
            candidate_actions=batch["candidate_actions"],
            candidate_timestep=batch["candidate_timestep"],
        )
        self.assertEqual(
            geometry_outputs["candidate"]["candidate_value"].shape, (1, 2)
        )
        self.assertIsNotNone(
            geometry_model.candidate_head.geometry_encoder
        )


if __name__ == "__main__":
    unittest.main()
