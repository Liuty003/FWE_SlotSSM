"""Level-1 unified recovery model with one shared temporal memory."""

from __future__ import annotations

import torch
from torch import nn

from recovery.models.candidate_effect_scorer import OUTCOME_DIM
from recovery.models.recovery_slotssm import RecoverySlotSSM
from recovery.data.tensor_dataset import (
    CANDIDATE_GEOMETRY_DIM,
    CANDIDATE_GEOMETRY_SLICE,
)


class TemporalCandidateEffectHead(nn.Module):
    """Predict candidate futures from Recovery SlotSSM temporal state."""

    def __init__(
        self,
        *,
        state_dim: int = 128,
        hidden_dim: int = 128,
        geometry_dim: int = 0,
    ):
        super().__init__()
        self.geometry_dim = geometry_dim
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.geometry_encoder = (
            nn.Sequential(
                nn.LayerNorm(geometry_dim),
                nn.Linear(geometry_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if geometry_dim else None
        )
        trunk_inputs = 3 if geometry_dim else 2
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_dim * trunk_inputs),
            nn.Linear(hidden_dim * trunk_inputs, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.outcome_head = nn.Linear(hidden_dim, OUTCOME_DIM)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        temporal_state: torch.Tensor,
        nominal_action: torch.Tensor,
        candidate_actions: torch.Tensor,
        observable_geometry: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if temporal_state.ndim != 2:
            raise ValueError("temporal_state must be [B,H]")
        if nominal_action.shape != (temporal_state.shape[0], 7):
            raise ValueError("nominal_action must be [B,7]")
        if (
            candidate_actions.ndim != 3
            or candidate_actions.shape[0] != temporal_state.shape[0]
            or candidate_actions.shape[2] != 7
        ):
            raise ValueError("candidate_actions must be [B,K,7]")
        count = candidate_actions.shape[1]
        state = self.state_encoder(temporal_state)[:, None].expand(
            -1, count, -1
        )
        nominal = nominal_action[:, None].expand(-1, count, -1)
        action = self.action_encoder(
            torch.cat(
                (candidate_actions, candidate_actions - nominal), dim=-1
            )
        )
        features = [state, action]
        if self.geometry_encoder is not None:
            if observable_geometry is None or observable_geometry.shape != (
                temporal_state.shape[0],
                self.geometry_dim,
            ):
                raise ValueError(
                    f"observable_geometry must be [B,{self.geometry_dim}]"
                )
            geometry = self.geometry_encoder(observable_geometry)
            features.insert(1, geometry[:, None].expand(-1, count, -1))
        elif observable_geometry is not None:
            raise ValueError("this candidate head has no geometry encoder")
        hidden = self.trunk(torch.cat(features, dim=-1))
        outcome = self.outcome_head(hidden)
        return {
            "eef_delta_m": outcome[..., 0:3],
            "target_delta_m": outcome[..., 3:6],
            "attached_logits": outcome[..., 6],
            "lifted_logits": outcome[..., 7],
            "slip_logits": outcome[..., 8],
            "failure_persistence_logits": outcome[..., 9],
            "handoff_logits": outcome[..., 10],
            "uncertainty_raw": outcome[..., 11:13],
            "candidate_value": self.value_head(hidden).squeeze(-1),
        }


class UnifiedRecoveryModel(nn.Module):
    """Current-state and future-candidate heads sharing Recovery SlotSSM."""

    def __init__(
        self,
        recovery: RecoverySlotSSM,
        *,
        candidate_hidden_dim: int = 128,
        geometry_skip: bool = False,
    ):
        super().__init__()
        self.recovery = recovery
        self.geometry_skip = geometry_skip
        state_dim = int(recovery.phase_head.in_features)
        self.candidate_head = TemporalCandidateEffectHead(
            state_dim=state_dim,
            hidden_dim=candidate_hidden_dim,
            geometry_dim=CANDIDATE_GEOMETRY_DIM if geometry_skip else 0,
        )

    def freeze_recovery(self) -> None:
        for parameter in self.recovery.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        nominal_action: torch.Tensor | None = None,
        candidate_actions: torch.Tensor | None = None,
        candidate_timestep: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        outputs = self.recovery(inputs)
        if nominal_action is None and candidate_actions is None:
            return outputs
        if nominal_action is None or candidate_actions is None:
            raise ValueError(
                "nominal_action and candidate_actions must be provided together"
            )
        temporal = outputs["temporal_state"]
        if candidate_timestep is None:
            state = temporal[:, -1]
            selected_inputs = inputs[:, -1]
        else:
            if candidate_timestep.shape != (temporal.shape[0],):
                raise ValueError("candidate_timestep must be [B]")
            state = temporal[
                torch.arange(temporal.shape[0], device=temporal.device),
                candidate_timestep,
            ]
            selected_inputs = inputs[
                torch.arange(inputs.shape[0], device=inputs.device),
                candidate_timestep,
            ]
        # tensor_dataset v1 layout indices 65:76 contain only observable
        # target-to-EEF geometry, centroid/measurement, and their validity.
        geometry = (
            selected_inputs[:, CANDIDATE_GEOMETRY_SLICE]
            if self.geometry_skip else None
        )
        outputs["candidate"] = self.candidate_head(
            state,
            nominal_action,
            candidate_actions,
            observable_geometry=geometry,
        )
        return outputs
