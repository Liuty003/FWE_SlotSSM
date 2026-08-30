"""Unified OpenVLA-OFT action routing conditioned on temporal object slots.

The frozen OFT normal head consumes the original action-token features.  A
separate recovery head, initialized as a copy of the normal head, consumes a
zero-initialized residual cross-attention fusion of those features with
SlotSSM tokens.  A shared three-state gate predicts NORMAL, RECOVERY, or DONE.

This module deliberately does not import OpenVLA-OFT.  The official OFT action
head exposes ``predict_action(actions_hidden_states)`` and can be passed in
directly, while unit tests can use a small compatible head.
"""

from __future__ import annotations

import copy
from enum import IntEnum

import torch
from torch import nn


class RecoveryMode(IntEnum):
    NORMAL = 0
    RECOVERY = 1
    RECOVERY_COMPLETE = 2


class RecoveryPhase(IntEnum):
    """Causal local-regrasp phases supervised from offline teacher branches."""

    BACKOFF = 0
    ALIGN = 1
    DESCEND = 2
    CLOSE = 3
    LIFT_VERIFY = 4


class RecoveryCandidateMode(IntEnum):
    """Discrete approach modes distilled from the five FSM teachers.

    These are *latent policy modes*, not controller programs.  At deployment a
    learned selector chooses one mode once at recovery onset, and the
    mode-conditioned neural action head produces every subsequent action.
    """

    CENTER = 0
    Y_NEGATIVE = 1
    Y_POSITIVE = 2
    X_NEGATIVE = 3
    X_POSITIVE = 4


def _predict_with_head(head: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    predict = getattr(head, "predict_action", None)
    return predict(hidden) if predict is not None else head(hidden)


class OFTSlotCrossAttention(nn.Module):
    """Low-rank residual fusion from persistent slot tokens into OFT tokens."""

    def __init__(
        self,
        action_hidden_dim: int,
        slot_dim: int,
        fusion_dim: int = 512,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if fusion_dim % num_heads:
            raise ValueError("fusion_dim must be divisible by num_heads")
        self.action_norm = nn.LayerNorm(action_hidden_dim)
        self.slot_norm = nn.LayerNorm(slot_dim)
        self.action_projection = nn.Linear(action_hidden_dim, fusion_dim)
        self.slot_projection = nn.Linear(slot_dim, fusion_dim)
        self.cross_attention = nn.MultiheadAttention(
            fusion_dim, num_heads, batch_first=True
        )
        self.output_projection = nn.Linear(fusion_dim, action_hidden_dim)
        # Exact behavior preservation at initialization: the recovery branch
        # initially sees the same features as the pretrained normal branch.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        action_hidden: torch.Tensor,
        temporal_slots: torch.Tensor,
        slot_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if action_hidden.ndim != 3 or temporal_slots.ndim != 3:
            raise ValueError("expected action_hidden [B,A,D] and slots [B,N,S]")
        if action_hidden.shape[0] != temporal_slots.shape[0]:
            raise ValueError("action and slot batch sizes differ")
        query = self.action_projection(self.action_norm(action_hidden))
        context = self.slot_projection(self.slot_norm(temporal_slots))
        attended, _ = self.cross_attention(
            query,
            context,
            context,
            key_padding_mask=slot_padding_mask,
            need_weights=False,
        )
        fused = action_hidden + self.output_projection(attended)
        return fused, attended


class OpenVLAOFTSlotSSMAdapter(nn.Module):
    """One-checkpoint normal/recovery action heads with a three-state gate."""

    def __init__(
        self,
        normal_action_head: nn.Module,
        *,
        action_hidden_dim: int,
        slot_dim: int,
        fusion_dim: int = 512,
        num_heads: int = 8,
        gate_hidden_dim: int = 256,
        proprio_dim: int = 8,
        action_dim: int = 7,
        action_chunk: int = 8,
        action_output_dim: int | None = None,
        phase_expert_hidden_dim: int = 512,
        mode_embedding_dim: int | None = None,
        freeze_normal_head: bool = True,
    ) -> None:
        super().__init__()
        self.normal_action_head = normal_action_head
        self.recovery_action_head = copy.deepcopy(normal_action_head)
        if freeze_normal_head:
            self.normal_action_head.requires_grad_(False)
        self.slot_fusion = OFTSlotCrossAttention(
            action_hidden_dim,
            slot_dim,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
        )
        self.gate_action_projection = nn.Sequential(
            nn.LayerNorm(action_hidden_dim),
            nn.Linear(action_hidden_dim, gate_hidden_dim),
            nn.GELU(),
        )
        self.gate_slot_projection = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, gate_hidden_dim),
            nn.GELU(),
        )
        # Preserve existing adapter checkpoints exactly: this contribution is
        # zero until it has been trained on recorded proprio/action history.
        self.gate_robot_projection = nn.Sequential(
            nn.LayerNorm(proprio_dim + action_dim),
            nn.Linear(proprio_dim + action_dim, gate_hidden_dim),
            nn.GELU(),
        )
        nn.init.zeros_(self.gate_robot_projection[1].weight)
        nn.init.zeros_(self.gate_robot_projection[1].bias)
        # Recovery actions need the same causal robot/action-history signal as
        # the gate.  In particular, a local recovery starts with a backoff
        # before it may align, descend, close, or lift; RGB features alone can
        # make these otherwise similar states ambiguous.  Start at an exact
        # no-op so checkpoints trained before this branch preserve their
        # recovery actions until it is explicitly supervised.
        self.recovery_robot_action_projection = nn.Sequential(
            nn.LayerNorm(proprio_dim + action_dim),
            nn.Linear(proprio_dim + action_dim, action_hidden_dim),
            nn.GELU(),
        )
        nn.init.zeros_(self.recovery_robot_action_projection[1].weight)
        nn.init.zeros_(self.recovery_robot_action_projection[1].bias)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(2 * gate_hidden_dim),
            nn.Linear(2 * gate_hidden_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, len(RecoveryMode)),
        )
        self.phase_head = nn.Sequential(
            nn.LayerNorm(2 * gate_hidden_dim),
            nn.Linear(2 * gate_hidden_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, len(RecoveryPhase)),
        )
        # Teacher target pose is used only as an offline regression label.
        # At deployment this head receives the same visual/slot/proprio
        # execution context as the action path and estimates target minus EEF.
        self.geometry_head = nn.Sequential(
            nn.LayerNorm(2 * gate_hidden_dim),
            nn.Linear(2 * gate_hidden_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 3),
        )
        # Unlike ``geometry_head``, this branch never averages object slots.
        # The action/proprio context queries the individual causal SlotSSM
        # slots, preserving the object-selection information needed for a
        # target-minus-EEF estimate.
        self.geometry_slot_projection = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, gate_hidden_dim),
            nn.GELU(),
        )
        self.geometry_slot_attention = nn.MultiheadAttention(
            gate_hidden_dim, num_heads, batch_first=True,
        )
        self.spatial_geometry_head = nn.Sequential(
            nn.LayerNorm(2 * gate_hidden_dim),
            nn.Linear(2 * gate_hidden_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 3),
        )
        # A learned target-relative vector can condition the recovery action
        # without exposing simulator pose at deployment.  Zero initialization
        # preserves legacy action behavior until geometry supervision has
        # established a useful visual estimate.
        self.geometry_action_projection = nn.Linear(3, action_hidden_dim)
        nn.init.zeros_(self.geometry_action_projection.weight)
        nn.init.zeros_(self.geometry_action_projection.bias)
        # Soft phase conditioning starts as a no-op, preserving the copied
        # OFT recovery head until phase supervision supplies a signal.
        self.phase_embedding = nn.Embedding(len(RecoveryPhase), action_hidden_dim)
        nn.init.zeros_(self.phase_embedding.weight)
        # The five recovery modes disambiguate demonstrations which start
        # from the same failure state but approach the object from different
        # sides.  Zero initialization preserves old checkpoints exactly until
        # the distilled mode-conditioned action loss is applied.
        self.mode_embedding = nn.Embedding(
            len(RecoveryCandidateMode),
            action_hidden_dim if mode_embedding_dim is None else mode_embedding_dim,
        )
        if self.mode_embedding.embedding_dim != action_hidden_dim:
            raise ValueError("mode_embedding_dim must equal action_hidden_dim")
        nn.init.zeros_(self.mode_embedding.weight)
        self.action_chunk = action_chunk
        self.action_output_dim = action_dim if action_output_dim is None else action_output_dim
        # A phase-specific residual is deliberately much smaller than copying
        # the full OFT action head five times.  It gives closing/lifting their
        # own local controller while the shared recovery head remains a stable
        # global prior.  Its final layer starts at zero, so old checkpoints
        # retain exactly their previous behavior when loaded with strict=False.
        self.phase_action_experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(action_hidden_dim),
                nn.Linear(action_hidden_dim, phase_expert_hidden_dim),
                nn.GELU(),
                nn.Linear(phase_expert_hidden_dim, action_chunk * self.action_output_dim),
            )
            for _ in RecoveryPhase
        ])
        for expert in self.phase_action_experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)
        # Conservative initial router: use the pretrained policy until the
        # failure gate has learned causal evidence for intervention.
        final_gate = self.gate_head[-1]
        nn.init.zeros_(final_gate.weight)
        with torch.no_grad():
            final_gate.bias.copy_(torch.tensor((3.0, -3.0, -3.0)))

    def forward(
        self,
        action_hidden: torch.Tensor,
        temporal_slots: torch.Tensor,
        slot_padding_mask: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        previous_action: torch.Tensor | None = None,
        recovery_phase_override: torch.Tensor | None = None,
        recovery_mode_override: torch.Tensor | None = None,
        disable_phase_conditioning: bool = False,
        geometry_conditioning: bool = False,
    ) -> dict[str, torch.Tensor]:
        recovery_hidden, attended = self.slot_fusion(
            action_hidden, temporal_slots, slot_padding_mask
        )
        normal_actions = _predict_with_head(
            self.normal_action_head, action_hidden
        )
        action_context = self.gate_action_projection(action_hidden.mean(dim=1))
        slot_context = self.gate_slot_projection(temporal_slots.mean(dim=1))
        recovery_robot_action_residual = torch.zeros_like(recovery_hidden[:, 0])
        if (proprio is None) != (previous_action is None):
            raise ValueError("proprio and previous_action must be provided together")
        if proprio is not None and previous_action is not None:
            if proprio.ndim != 2 or previous_action.ndim != 2:
                raise ValueError("proprio and previous_action must have shape [B,D]")
            if proprio.shape[0] != action_hidden.shape[0]:
                raise ValueError("proprio batch size differs from action hidden")
            robot_context = self.gate_robot_projection(
                torch.cat((proprio, previous_action), dim=-1)
            )
            recovery_robot_action_residual = self.recovery_robot_action_projection(
                torch.cat((proprio, previous_action), dim=-1)
            )
            action_context = action_context + robot_context
        execution_context = torch.cat((action_context, slot_context), dim=-1)
        gate_logits = self.gate_head(execution_context)
        phase_logits = self.phase_head(execution_context)
        target_relative_position = self.geometry_head(execution_context)
        geometry_slots = self.geometry_slot_projection(temporal_slots)
        geometry_context, geometry_slot_weights = self.geometry_slot_attention(
            action_context[:, None, :], geometry_slots, geometry_slots,
            key_padding_mask=slot_padding_mask, need_weights=True,
        )
        spatial_target_relative_position = self.spatial_geometry_head(
            torch.cat((action_context, geometry_context[:, 0]), dim=-1)
        )
        if disable_phase_conditioning:
            # The phase label is an offline teacher annotation.  A pure
            # neural deployment may deliberately ablate it rather than
            # mixing phase experts using an uncertain predicted phase.
            phase_weights = torch.zeros_like(phase_logits)
        elif recovery_phase_override is None:
            phase_weights = phase_logits.softmax(dim=-1)
        else:
            if recovery_phase_override.ndim != 1 or recovery_phase_override.shape[0] != action_hidden.shape[0]:
                raise ValueError("recovery_phase_override must have shape [B]")
            if recovery_phase_override.min() < 0 or recovery_phase_override.max() >= len(RecoveryPhase):
                raise ValueError("recovery_phase_override contains an invalid phase")
            phase_weights = torch.nn.functional.one_hot(
                recovery_phase_override.to(dtype=torch.long), num_classes=len(RecoveryPhase)
            ).to(dtype=phase_logits.dtype)
        phase_residual = phase_weights @ self.phase_embedding.weight
        if recovery_mode_override is None:
            # Mode is deliberately a no-op unless an explicit learned
            # selector supplies one.  This keeps legacy checkpoints and the
            # normal path behavior-preserving.
            mode_residual = torch.zeros_like(phase_residual)
        else:
            if recovery_mode_override.ndim != 1 or recovery_mode_override.shape[0] != action_hidden.shape[0]:
                raise ValueError("recovery_mode_override must have shape [B]")
            if (recovery_mode_override.min() < 0 or
                    recovery_mode_override.max() >= len(RecoveryCandidateMode)):
                raise ValueError("recovery_mode_override contains an invalid mode")
            mode_residual = self.mode_embedding(recovery_mode_override.to(dtype=torch.long))
        geometry_residual = (
            self.geometry_action_projection(spatial_target_relative_position) if geometry_conditioning
            else torch.zeros_like(phase_residual)
        )
        phase_conditioned_hidden = (
            recovery_hidden + phase_residual[:, None, :] + mode_residual[:, None, :]
            + geometry_residual[:, None, :] + recovery_robot_action_residual[:, None, :]
        )
        recovery_actions = _predict_with_head(
            self.recovery_action_head, phase_conditioned_hidden
        )
        if recovery_actions.shape[1:] != (self.action_chunk, self.action_output_dim):
            raise ValueError(
                "normal action head output does not match action_chunk/action_dim; "
                "configure the adapter with the head's action shape"
            )
        expert_input = phase_conditioned_hidden.mean(dim=1)
        expert_deltas = torch.stack([
            expert(expert_input).view(-1, self.action_chunk, self.action_output_dim)
            for expert in self.phase_action_experts
        ], dim=1)
        if not disable_phase_conditioning:
            recovery_actions = recovery_actions + (phase_weights[:, :, None, None] * expert_deltas).sum(dim=1)
        return {
            "normal_actions": normal_actions,
            "recovery_actions": recovery_actions,
            "gate_logits": gate_logits,
            "recovery_phase_logits": phase_logits,
            "target_relative_position": target_relative_position,
            "spatial_target_relative_position": spatial_target_relative_position,
            "geometry_slot_attention": geometry_slot_weights,
            "recovery_hidden": recovery_hidden,
            "phase_conditioned_recovery_hidden": phase_conditioned_hidden,
            "recovery_mode_embedding": mode_residual,
            "geometry_action_embedding": geometry_residual,
            "phase_expert_deltas": expert_deltas,
            "attended_slot_context": attended,
        }

    @staticmethod
    def route_actions(
        normal_actions: torch.Tensor,
        recovery_actions: torch.Tensor,
        mode: torch.Tensor,
    ) -> torch.Tensor:
        """Hard-route actions; DONE hands control back to the normal head."""
        if mode.ndim != 1 or mode.shape[0] != normal_actions.shape[0]:
            raise ValueError("mode must have shape [B]")
        recovery = mode.eq(int(RecoveryMode.RECOVERY))
        while recovery.ndim < normal_actions.ndim:
            recovery = recovery.unsqueeze(-1)
        return torch.where(recovery, recovery_actions, normal_actions)
