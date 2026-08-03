"""Lightweight short-horizon object-slot dynamics for recovery candidates."""

from __future__ import annotations

import torch
from torch import nn


class ActionConditionedRoleDynamics(nn.Module):
    """Predict next grounded role slots from current slots, action and EEF pose.

    The model predicts a residual so the identity transition is an explicit
    baseline.  It is deliberately small: this first version is used to audit
    whether the newly collected recovery branches contain an action-dependent
    object signal before coupling dynamics to the risk scorer.
    """

    def __init__(
        self,
        *,
        num_roles: int = 3,
        slot_dim: int = 96,
        proprio_dim: int = 3,
        action_horizon: int = 1,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_roles = num_roles
        self.slot_dim = slot_dim
        self.proprio_dim = proprio_dim
        self.action_horizon = action_horizon
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        state_dim = num_roles * slot_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(state_dim + 7 * action_horizon + proprio_dim),
            nn.Linear(state_dim + 7 * action_horizon + proprio_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(hidden_dim, state_dim)

    def forward(
        self,
        role_slots: torch.Tensor,
        actions: torch.Tensor,
        eef_positions: torch.Tensor,
    ) -> torch.Tensor:
        if role_slots.ndim != 3 or role_slots.shape[1:] != (
            self.num_roles, self.slot_dim
        ):
            raise ValueError(
                f"role_slots must be [B,{self.num_roles},{self.slot_dim}]"
            )
        if actions.shape != (
            role_slots.shape[0], self.action_horizon, 7
        ):
            raise ValueError(
                f"actions must be [B,{self.action_horizon},7]"
            )
        if eef_positions.shape != (role_slots.shape[0], self.proprio_dim):
            raise ValueError(f"eef_positions must be [B,{self.proprio_dim}]")
        flattened = role_slots.flatten(1)
        delta = self.delta_head(
            self.encoder(torch.cat(
                (flattened, actions.flatten(1), eef_positions), dim=-1
            ))
        )
        return (flattened + delta).reshape_as(role_slots)
