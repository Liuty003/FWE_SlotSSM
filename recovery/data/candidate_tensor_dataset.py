"""Tensor dataset for matched, variable-count recovery action candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from recovery.data.candidate_schema import RecoveryCandidateGroup
from recovery.data.schema import RecoveryTransition
from recovery.data.tensor_dataset import (
    build_recovery_model_inputs,
    tensorize_transition,
)


class RecoveryCandidateDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        groups: Sequence[RecoveryCandidateGroup],
        transitions: Mapping[
            tuple[str, int, int, int], RecoveryTransition
        ],
    ):
        self.groups = list(groups)
        self.transitions = transitions
        if not self.groups:
            raise ValueError("candidate dataset is empty")
        for group in self.groups:
            group.validate()
            key = (
                group.task_suite,
                group.task_id,
                group.episode_id,
                group.timestep,
            )
            if key not in transitions:
                raise ValueError(f"candidate context transition not found: {key}")

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        group = self.groups[index]
        key = (
            group.task_suite,
            group.task_id,
            group.episode_id,
            group.timestep,
        )
        context_tensors = tensorize_transition(self.transitions[key])
        context = build_recovery_model_inputs(context_tensors)
        outcomes = group.teacher_outcomes
        return {
            "context": context,
            "nominal_action": torch.tensor(
                group.nominal_action, dtype=torch.float32
            ),
            "candidate_actions": torch.tensor(
                group.candidate_actions, dtype=torch.float32
            ),
            "outcome_valid": torch.tensor(
                group.outcome_valid, dtype=torch.bool
            ),
            "eef_delta_m": torch.tensor(
                [item.get("eef_delta_m", [0.0] * 3) for item in outcomes],
                dtype=torch.float32,
            ),
            "target_delta_m": torch.tensor(
                [item.get("target_delta_m", [0.0] * 3) for item in outcomes],
                dtype=torch.float32,
            ),
            "binary_targets": torch.tensor(
                [
                    [
                        float(bool(item.get("attached_after", False))),
                        float(bool(item.get("lifted_after", False))),
                        float(bool(item.get("slipped_after", False))),
                        float(bool(item.get("failure_persistent_after", False))),
                        float(bool(item.get("handoff_ready_after", False))),
                    ]
                    for item in outcomes
                ],
                dtype=torch.float32,
            ),
            "teacher_value": torch.tensor(
                [float(item.get("teacher_value", 0.0)) for item in outcomes],
                dtype=torch.float32,
            ),
            "metadata": {
                "key": key,
                "candidate_names": group.candidate_names,
            },
        }


class UnifiedRecoveryCandidateDataset(RecoveryCandidateDataset):
    """Candidate groups with a causal history ending at the branch state."""

    def __init__(
        self,
        groups: Sequence[RecoveryCandidateGroup],
        transitions: Mapping[
            tuple[str, int, int, int], RecoveryTransition
        ],
        *,
        window_length: int = 32,
    ):
        if window_length <= 0:
            raise ValueError("window_length must be positive")
        super().__init__(groups, transitions)
        self.window_length = window_length
        self.history_keys: list[list[tuple[str, int, int, int]]] = []
        for group in self.groups:
            start = group.timestep - window_length + 1
            if start < 0:
                raise ValueError(
                    f"candidate timestep {group.timestep} has insufficient history"
                )
            keys = [
                (
                    group.task_suite,
                    group.task_id,
                    group.episode_id,
                    timestep,
                )
                for timestep in range(start, group.timestep + 1)
            ]
            missing = [key for key in keys if key not in transitions]
            if missing:
                raise ValueError(
                    f"candidate history is non-contiguous; first missing={missing[0]}"
                )
            self.history_keys.append(keys)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        steps = [
            tensorize_transition(self.transitions[key])
            for key in self.history_keys[index]
        ]
        sequence = {
            name: torch.stack([step[name] for step in steps])
            for name in steps[0]
        }
        sample["context"] = build_recovery_model_inputs(sequence)
        sample["candidate_timestep"] = torch.tensor(
            self.window_length - 1, dtype=torch.long
        )
        return sample


def collate_recovery_candidates(
    samples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Pad only the candidate axis; invalid padding never contributes to loss."""

    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(samples)
    max_candidates = max(len(sample["candidate_actions"]) for sample in samples)
    result = {
        "context": torch.stack([sample["context"] for sample in samples]),
        "nominal_action": torch.stack(
            [sample["nominal_action"] for sample in samples]
        ),
        "candidate_actions": torch.zeros(batch_size, max_candidates, 7),
        "outcome_valid": torch.zeros(
            batch_size, max_candidates, dtype=torch.bool
        ),
        "candidate_mask": torch.zeros(
            batch_size, max_candidates, dtype=torch.bool
        ),
        "eef_delta_m": torch.zeros(batch_size, max_candidates, 3),
        "target_delta_m": torch.zeros(batch_size, max_candidates, 3),
        "binary_targets": torch.zeros(batch_size, max_candidates, 5),
        "teacher_value": torch.zeros(batch_size, max_candidates),
        "metadata": [sample["metadata"] for sample in samples],
    }
    if "candidate_timestep" in samples[0]:
        result["candidate_timestep"] = torch.stack(
            [sample["candidate_timestep"] for sample in samples]
        )
    for batch_index, sample in enumerate(samples):
        count = len(sample["candidate_actions"])
        result["candidate_mask"][batch_index, :count] = True
        for name in (
            "candidate_actions",
            "outcome_valid",
            "eef_delta_m",
            "target_delta_m",
            "binary_targets",
            "teacher_value",
        ):
            result[name][batch_index, :count] = sample[name]
    return result
