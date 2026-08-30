"""Small learned residual for distribution-specific neural recovery repair."""

from __future__ import annotations

import torch
from torch import nn


class NeuralRecoveryResidual(nn.Module):
    """Predict whether and how to correct a frozen recovery action.

    It consumes only the deployment contract: frozen OFT action tokens,
    causal SlotSSM slots, robot proprio/action history, and the learned
    discrete approach mode.  The residual is deliberately bounded so that a
    positive correction cannot replace the base recovery policy wholesale.
    """

    def __init__(self, action_dim: int = 4096, slot_dim: int = 512, hidden: int = 256) -> None:
        super().__init__()
        self.action = nn.Sequential(nn.LayerNorm(action_dim), nn.Linear(action_dim, hidden), nn.GELU())
        self.slots = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, hidden), nn.GELU())
        self.robot = nn.Sequential(nn.LayerNorm(15), nn.Linear(15, hidden // 2), nn.GELU())
        self.mode = nn.Embedding(5, hidden // 2)
        self.trunk = nn.Sequential(
            nn.LayerNorm(3 * hidden), nn.Linear(3 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.gate = nn.Linear(hidden, 1)
        self.delta = nn.Linear(hidden, 7)
        # A residual starts as an exact no-op until supervised correction data
        # establishes a distribution where it is beneficial.
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(
        self,
        action_hidden: torch.Tensor,
        temporal_slots: torch.Tensor,
        proprio: torch.Tensor,
        previous_action: torch.Tensor,
        mode: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if mode.ndim != 1:
            raise ValueError("mode must have shape [B]")
        fused = torch.cat((
            self.action(action_hidden.mean(dim=1)),
            self.slots(temporal_slots.mean(dim=1)),
            self.robot(torch.cat((proprio, previous_action), dim=-1)),
            self.mode(mode.to(dtype=torch.long)),
        ), dim=-1)
        hidden = self.trunk(fused)
        # Normalized OFT actions lie near [-1, 1]; cap a correction to avoid
        # an accidental classifier false-positive creating a new controller.
        return {"gate_logit": self.gate(hidden).squeeze(-1), "delta": 0.35 * torch.tanh(self.delta(hidden))}
