"""Action-conditioned short-horizon effects for matched recovery candidates."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from recovery.data.tensor_dataset import MODEL_INPUT_DIM


OUTCOME_DIM = 13


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1)


def candidate_effect_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    ranking_margin: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Supervise effects and relative utility using only valid teacher branches."""

    valid = batch["outcome_valid"] & batch["candidate_mask"]
    motion = _masked_mean(
        F.smooth_l1_loss(
            outputs["eef_delta_m"], batch["eef_delta_m"], reduction="none"
        ),
        valid,
    ) + _masked_mean(
        F.smooth_l1_loss(
            outputs["target_delta_m"],
            batch["target_delta_m"],
            reduction="none",
        ),
        valid,
    )
    binary_logits = torch.stack(
        (
            outputs["attached_logits"],
            outputs["lifted_logits"],
            outputs["slip_logits"],
            outputs["failure_persistence_logits"],
            outputs["handoff_logits"],
        ),
        dim=-1,
    )
    binary = _masked_mean(
        F.binary_cross_entropy_with_logits(
            binary_logits, batch["binary_targets"], reduction="none"
        ),
        valid,
    )
    value = _masked_mean(
        F.smooth_l1_loss(
            outputs["candidate_value"],
            batch["teacher_value"],
            reduction="none",
        ),
        valid,
    )

    predicted_difference = (
        outputs["candidate_value"][:, :, None]
        - outputs["candidate_value"][:, None, :]
    )
    target_difference = (
        batch["teacher_value"][:, :, None]
        - batch["teacher_value"][:, None, :]
    )
    pair_valid = valid[:, :, None] & valid[:, None, :]
    ordered = target_difference > 1e-6
    ranking_mask = pair_valid & ordered
    ranking = _masked_mean(
        F.relu(ranking_margin - predicted_difference),
        ranking_mask,
    )
    total = motion + binary + value + 0.5 * ranking
    return {
        "total": total,
        "motion": motion,
        "binary": binary,
        "value": value,
        "ranking": ranking,
    }


def generate_recovery_candidates(
    nominal_action: torch.Tensor,
    residual_action: torch.Tensor,
    *,
    translation_delta: float = 0.20,
    rotation_delta: float = 0.15,
) -> tuple[list[str], torch.Tensor]:
    """Create deterministic, bounded and distinct 7-D recovery candidates."""

    if nominal_action.shape != (7,) or residual_action.shape != (7,):
        raise ValueError("nominal_action and residual_action must both be [7]")
    center = (nominal_action + residual_action).clamp(-1.0, 1.0)
    names = ["nominal", "recovery_center"]
    actions = [nominal_action.clamp(-1.0, 1.0), center]
    for index, axis in enumerate(("x", "y", "z")):
        for sign, suffix in ((-1.0, "minus"), (1.0, "plus")):
            candidate = center.clone()
            candidate[index] = (
                candidate[index] + sign * translation_delta
            ).clamp(-1.0, 1.0)
            names.append(f"translate_{axis}_{suffix}")
            actions.append(candidate)
    for sign, suffix in ((-1.0, "minus"), (1.0, "plus")):
        candidate = center.clone()
        candidate[5] = (
            candidate[5] + sign * rotation_delta
        ).clamp(-1.0, 1.0)
        names.append(f"rotate_z_{suffix}")
        actions.append(candidate)
    gripper = center.clone()
    gripper[6] = -center[6] if abs(float(center[6])) > 1e-6 else 1.0
    names.append("flip_gripper")
    actions.append(gripper)
    hold = torch.zeros_like(center)
    hold[6] = center[6]
    names.append("hold")
    actions.append(hold)

    unique_names: list[str] = []
    unique_actions: list[torch.Tensor] = []
    for name, action in zip(names, actions):
        if any(torch.equal(action, existing) for existing in unique_actions):
            continue
        unique_names.append(name)
        unique_actions.append(action)
    if len(unique_actions) < 2:
        raise ValueError("candidate generator produced fewer than two actions")
    return unique_names, torch.stack(unique_actions)


class RecoveryCandidateEffectScorer(nn.Module):
    """Permutation-equivariant candidate scorer conditioned on observable state."""

    def __init__(
        self,
        *,
        context_dim: int = MODEL_INPUT_DIM,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.outcome_head = nn.Linear(hidden_dim, OUTCOME_DIM)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        context: torch.Tensor,
        nominal_action: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if context.ndim != 2:
            raise ValueError("context must be [B,D]")
        if nominal_action.shape != (context.shape[0], 7):
            raise ValueError("nominal_action must be [B,7]")
        if candidate_actions.ndim != 3 or candidate_actions.shape[2] != 7:
            raise ValueError("candidate_actions must be [B,K,7]")
        count = candidate_actions.shape[1]
        context_feature = self.context_encoder(context)[:, None].expand(
            -1, count, -1
        )
        nominal = nominal_action[:, None].expand(-1, count, -1)
        action_feature = self.action_encoder(
            torch.cat((candidate_actions, candidate_actions - nominal), dim=-1)
        )
        hidden = self.trunk(torch.cat((context_feature, action_feature), dim=-1))
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
