"""Candidate future scorer conditioned on frozen shared SlotSSM state."""

from __future__ import annotations

import torch
from torch import nn


class TemporalCandidateHead(nn.Module):
    """Shared action-conditioned scorer; no post-candidate observations enter."""

    def __init__(self, state_dim: int = 128, action_dim: int = 28) -> None:
        super().__init__()
        self.action = nn.Sequential(nn.LayerNorm(action_dim), nn.Linear(action_dim, 96), nn.GELU())
        self.task = nn.Embedding(2, 8)
        self.score = nn.Sequential(
            nn.LayerNorm(state_dim + 96 + 8), nn.Linear(state_dim + 96 + 8, 128),
            nn.GELU(), nn.Dropout(.1), nn.Linear(128, 1),
        )

    def forward(self, temporal_state: torch.Tensor, action_summary: torch.Tensor,
                task_index: torch.Tensor) -> torch.Tensor:
        if temporal_state.ndim != 2 or action_summary.shape != (len(temporal_state), 28):
            raise ValueError("expected state [B,H] and action summary [B,28]")
        return self.score(torch.cat((temporal_state, self.action(action_summary),
                                     self.task(task_index)), dim=1)).squeeze(1)
