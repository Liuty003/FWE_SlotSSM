"""Shared candidate-outcome head over task-adapted canonical role slots."""

from __future__ import annotations

import torch
from torch import nn


class CanonicalUnifiedCandidateHead(nn.Module):
    """Score a regrasp candidate after task-specific slot adaptation.

    Slots from separate task encoders are not assumed to share coordinates.
    The small adapters map them into a shared canonical role space before one
    shared score head consumes target/context/distractor slots and action
    summary.  This is deliberately a smoke-scale model, not a shared visual
    encoder or an action-conditioned future predictor.
    """

    def __init__(
        self, *, slot_dim: int = 96, adapted_dim: int = 16,
        action_summary_dim: int = 28, task_ids: tuple[int, ...] = (6, 9),
    ) -> None:
        super().__init__()
        self.task_ids = tuple(int(task_id) for task_id in task_ids)
        self.adapters = nn.ModuleDict({
            str(task_id): nn.Sequential(
                nn.LayerNorm(slot_dim), nn.Linear(slot_dim, adapted_dim), nn.GELU()
            )
            for task_id in self.task_ids
        })
        self.task_embedding = nn.Embedding(len(self.task_ids), 4)
        input_dim = 3 * adapted_dim + 9 + action_summary_dim + 4
        self.score = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, 48), nn.GELU(),
            nn.Dropout(.1), nn.Linear(48, 1),
        )

    def forward(
        self, role_slots: torch.Tensor, role_geometry: torch.Tensor,
        action_summary: torch.Tensor, task_index: torch.Tensor,
    ) -> torch.Tensor:
        if role_slots.ndim != 3 or role_slots.shape[1:] != (3, 96):
            raise ValueError(f"expected role_slots [B,3,96], got {tuple(role_slots.shape)}")
        if role_geometry.shape[1:] != (3, 3):
            raise ValueError("role_geometry must have shape [B,3,3]")
        if action_summary.ndim != 2 or action_summary.shape[0] != role_slots.shape[0]:
            raise ValueError("action_summary batch shape does not match role slots")
        adapted = torch.empty(
            role_slots.shape[0], 3, 16, device=role_slots.device, dtype=role_slots.dtype
        )
        for index, task_id in enumerate(self.task_ids):
            selected = task_index == index
            if selected.any():
                adapted[selected] = self.adapters[str(task_id)](role_slots[selected])
        features = torch.cat((
            adapted.flatten(1), role_geometry.flatten(1), action_summary,
            self.task_embedding(task_index),
        ), dim=1)
        return self.score(features).squeeze(1)
