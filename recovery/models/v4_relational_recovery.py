"""V4: relational corrective interface for a frozen OpenVLA-OFT action head.

This module starts after frozen OFT feature extraction.  It does not import or
own the OFT decoder: callers decode ``recovery_action_hidden`` with the same
frozen action head that generated the nominal action.  At deployment, use
``hard_route`` so a non-triggered decision is an exact nominal bypass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


Representation = Literal["global", "slot", "relation_2d", "relation_3d"]


@dataclass(frozen=True)
class V4ModelConfig:
    visual_dim: int = 2176
    action_hidden_dim: int = 4096
    proprio_dim: int = 8
    action_chunk: int = 8
    action_dim: int = 7
    num_slots: int = 16
    slot_dim: int = 512
    relation_dim: int = 512
    num_heads: int = 8
    residual_scale: float = 0.10
    representation: Representation = "relation_3d"

    def to_dict(self) -> dict:
        return asdict(self)


def _require_rank(value: Tensor, rank: int, name: str) -> None:
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got {tuple(value.shape)}")


class OFTTokenSlotAttention(nn.Module):
    """Object slots plus token masks from frozen, token-aligned OFT features."""

    def __init__(self, config: V4ModelConfig) -> None:
        super().__init__()
        if config.slot_dim % config.num_heads:
            raise ValueError("slot_dim must be divisible by num_heads")
        self.token = nn.Sequential(nn.LayerNorm(config.visual_dim), nn.Linear(config.visual_dim, config.slot_dim))
        self.queries = nn.Parameter(torch.empty(1, config.num_slots, config.slot_dim))
        self.attention = nn.MultiheadAttention(config.slot_dim, config.num_heads, batch_first=True)
        self.update = nn.Sequential(
            nn.LayerNorm(config.slot_dim), nn.Linear(config.slot_dim, 2 * config.slot_dim), nn.GELU(),
            nn.Linear(2 * config.slot_dim, config.slot_dim),
        )
        nn.init.normal_(self.queries, std=config.slot_dim ** -0.5)

    def forward(self, visual_tokens: Tensor, token_valid: Tensor) -> tuple[Tensor, Tensor]:
        _require_rank(visual_tokens, 3, "visual_tokens")
        if token_valid.shape != visual_tokens.shape[:2] or token_valid.dtype != torch.bool:
            raise ValueError("token_valid must be boolean [B,V]")
        tokens = self.token(visual_tokens)
        query = self.queries.expand(tokens.shape[0], -1, -1)
        attended, masks = self.attention(
            query, tokens, tokens, key_padding_mask=~token_valid,
            need_weights=True, average_attn_weights=True,
        )
        # Attention already ignores padded keys.  Renormalising keeps this true
        # even for PyTorch implementations that return tiny numerical weights.
        masks = masks * token_valid[:, None].to(masks.dtype)
        masks = masks / masks.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return attended + self.update(attended), masks


def _masked_centers(masks: Tensor, values: Tensor, valid: Tensor) -> Tensor:
    if values.shape[:2] != valid.shape or masks.shape[0] != values.shape[0] or masks.shape[-1] != values.shape[1]:
        raise ValueError("masks, values, and valid tokens are not aligned")
    weights = masks * valid[:, None].to(masks.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.einsum("bkv,bvd->bkd", weights, values)


def _one_to_one_instance_mask_loss(masks: Tensor, instance_labels: Tensor) -> Tensor:
    """Permutation-invariant object supervision without reusing a slot."""
    from scipy.optimize import linear_sum_assignment

    losses = []
    for batch_index in range(instance_labels.shape[0]):
        labels = instance_labels[batch_index]
        targets = [labels.eq(instance).float() for instance in labels.unique() if instance > 0]
        if not targets:
            continue
        targets = torch.stack([target / target.sum() for target in targets])
        scores = masks[batch_index] @ targets.T
        slot_indices, object_indices = linear_sum_assignment(-scores.detach().cpu().numpy())
        for slot_index, object_index in zip(slot_indices, object_indices):
            target = targets[object_index]
            losses.append(-(target * masks[batch_index, slot_index].clamp_min(1e-8).log()).sum())
    return masks.sum() * 0.0 if not losses else torch.stack(losses).mean()


class CandidateRelations(nn.Module):
    """Construct gripper-object and directed object-object candidate tokens."""

    def __init__(self, config: V4ModelConfig, geometry_dim: int) -> None:
        super().__init__()
        self.config, self.geometry_dim = config, geometry_dim
        self.gripper_object = nn.Sequential(
            nn.LayerNorm(config.slot_dim + geometry_dim + 1),
            nn.Linear(config.slot_dim + geometry_dim + 1, config.relation_dim), nn.GELU(),
            nn.Linear(config.relation_dim, config.relation_dim),
        )
        self.object_object = nn.Sequential(
            nn.LayerNorm(2 * config.slot_dim + geometry_dim),
            nn.Linear(2 * config.slot_dim + geometry_dim, config.relation_dim), nn.GELU(),
            nn.Linear(config.relation_dim, config.relation_dim),
        )
        self.type_embedding = nn.Embedding(2, config.relation_dim)

    def forward(self, slots: Tensor, positions: Tensor, gripper_position: Tensor) -> Tensor:
        batch, count, _ = slots.shape
        if positions.shape != (batch, count, self.geometry_dim):
            raise ValueError("positions do not match slot geometry dimension")
        if gripper_position.shape != (batch, self.geometry_dim):
            raise ValueError("gripper_position does not match slot geometry dimension")
        delta_go = positions - gripper_position[:, None]
        distance = delta_go.norm(dim=-1, keepdim=True)
        go = self.gripper_object(torch.cat((slots, delta_go, distance), dim=-1))
        source = slots[:, :, None].expand(-1, -1, count, -1)
        destination = slots[:, None, :].expand(-1, count, -1, -1)
        delta_oo = positions[:, None, :, :] - positions[:, :, None, :]
        oo = self.object_object(torch.cat((source, destination, delta_oo), dim=-1))
        keep = ~torch.eye(count, device=slots.device, dtype=torch.bool).reshape(-1)
        oo = oo.reshape(batch, count * count, -1)[:, keep]
        return torch.cat((go + self.type_embedding.weight[0], oo + self.type_embedding.weight[1]), dim=1)


class V4RelationalRecovery(nn.Module):
    """Produces a gate and an OFT-hidden corrective residual.

    The returned ``recovery_action_hidden`` is not decoded here on purpose:
    retaining the external frozen OFT decoder prevents this adapter becoming a
    second action policy.  ``hard_route`` guarantees an exact nominal bypass.
    """

    def __init__(self, config: V4ModelConfig = V4ModelConfig()) -> None:
        super().__init__()
        if config.relation_dim % config.num_heads:
            raise ValueError("relation_dim must be divisible by num_heads")
        if config.representation not in ("global", "slot", "relation_2d", "relation_3d"):
            raise ValueError(f"unsupported representation: {config.representation}")
        self.config = config
        self.slot_attention = OFTTokenSlotAttention(config)
        self.relations_2d = CandidateRelations(config, geometry_dim=2)
        self.relations_3d = CandidateRelations(config, geometry_dim=3)
        self.action_query = nn.Sequential(nn.LayerNorm(config.action_hidden_dim), nn.Linear(config.action_hidden_dim, config.relation_dim))
        self.global_context = nn.Sequential(nn.LayerNorm(config.action_hidden_dim), nn.Linear(config.action_hidden_dim, config.relation_dim), nn.GELU())
        self.context_attention = nn.MultiheadAttention(config.relation_dim, config.num_heads, batch_first=True)
        self.adapter = nn.Sequential(
            nn.LayerNorm(config.relation_dim), nn.Linear(config.relation_dim, config.relation_dim), nn.GELU(),
            nn.Linear(config.relation_dim, config.action_hidden_dim),
        )
        # V4.2 deliberately gives detection and correction exactly the same
        # task-conditioned relational context C_t.  Do not add a separate
        # action / proprio / failure-feature bypass to this gate.
        self.gate = nn.Sequential(
            nn.LayerNorm(config.relation_dim), nn.Linear(config.relation_dim, config.relation_dim), nn.GELU(),
            nn.Linear(config.relation_dim, 1),
        )
        # The initial adapter is an exact identity transformation.  This is a
        # safety property, not merely an optimisation convenience.
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def _context(self, action_hidden: Tensor, slots: Tensor, masks: Tensor, token_points: Tensor,
                 token_image_xy: Tensor, token_valid: Tensor, gripper_position: Tensor,
                 gripper_image_xy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        representation = self.config.representation
        if representation == "global":
            context = self.global_context(action_hidden)
            relation_tokens = context
            return context, relation_tokens, _masked_centers(masks, token_points, token_valid)
        if representation == "slot":
            relation_tokens = slots
        elif representation == "relation_2d":
            centers = _masked_centers(masks, token_image_xy, token_valid)
            relation_tokens = self.relations_2d(slots, centers, gripper_image_xy)
        else:
            centers = _masked_centers(masks, token_points, token_valid)
            relation_tokens = self.relations_3d(slots, centers, gripper_position)
        queries = self.action_query(action_hidden)
        context, _ = self.context_attention(queries, relation_tokens, relation_tokens, need_weights=False)
        return context, relation_tokens, _masked_centers(masks, token_points, token_valid)

    def forward(self, *, visual_tokens: Tensor, action_hidden: Tensor, token_points: Tensor,
                token_valid: Tensor, token_image_xy: Tensor, gripper_position: Tensor,
                gripper_image_xy: Tensor) -> dict[str, Tensor]:
        _require_rank(action_hidden, 3, "action_hidden")
        batch, tokens = visual_tokens.shape[:2]
        if token_points.shape != (batch, tokens, 3) or token_image_xy.shape != (batch, tokens, 2):
            raise ValueError("token geometry must align with visual_tokens")
        if gripper_position.shape != (batch, 3) or gripper_image_xy.shape != (batch, 2):
            raise ValueError("gripper geometry must be [B,3] and [B,2]")
        slots, masks = self.slot_attention(visual_tokens, token_valid)
        context, relation_tokens, slot_positions = self._context(
            action_hidden, slots, masks, token_points, token_image_xy, token_valid,
            gripper_position, gripper_image_xy,
        )
        delta = self.adapter(context)
        # Pool C_t, not R_t: both heads must see the same language-selected
        # relational evidence.  This is the central V4.2 hypothesis.
        gate_logit = self.gate(context.mean(dim=1)).squeeze(-1)
        probability = torch.sigmoid(gate_logit)
        recovery_hidden = action_hidden + self.config.residual_scale * delta
        return {
            "gate_logit": gate_logit, "gate_probability": probability, "delta_action_hidden": delta,
            "recovery_action_hidden": recovery_hidden, "slots": slots, "slot_masks": masks,
            "slot_positions": slot_positions, "relation_tokens": relation_tokens,
            "relational_context": context,
        }

    @staticmethod
    def hard_route(action_hidden: Tensor, recovery_action_hidden: Tensor, gate_probability: Tensor,
                   threshold: float) -> Tensor:
        """Return the exact nominal hidden state for every non-triggered row."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        if action_hidden.shape != recovery_action_hidden.shape or gate_probability.shape != action_hidden.shape[:1]:
            raise ValueError("hard-route inputs have incompatible shapes")
        return torch.where(gate_probability[:, None, None] >= threshold, recovery_action_hidden, action_hidden)

    @staticmethod
    def teacher_route(action_hidden: Tensor, recovery_action_hidden: Tensor, gate_labels: Tensor) -> Tensor:
        """Use the ground-truth gate during training, as specified by V4.2.

        Unknown / rejected labels (``-1``) are routed through nominal OFT and
        must not accidentally receive a recovery correction.
        """
        if action_hidden.shape != recovery_action_hidden.shape or gate_labels.shape != action_hidden.shape[:1]:
            raise ValueError("teacher-route inputs have incompatible shapes")
        return torch.where(gate_labels.eq(1)[:, None, None], recovery_action_hidden, action_hidden)

    @staticmethod
    def losses(outputs: dict[str, Tensor], *, gate_labels: Tensor, instance_labels: Tensor | None = None,
               lambda_gate: float = 1.0, lambda_slot: float = 0.0,
               gate_label_smoothing: float = 0.0) -> dict[str, Tensor]:
        """Losses independent of the external frozen OFT action decoder.

        Recovery action loss is computed by the trainer *after* feeding
        ``recovery_action_hidden`` to the original frozen OFT action head.
        """
        valid = gate_labels.ge(0)
        if not 0.0 <= gate_label_smoothing < 1.0:
            raise ValueError("gate_label_smoothing must lie in [0, 1)")
        targets = gate_labels[valid].float()
        targets = targets * (1.0 - gate_label_smoothing) + 0.5 * gate_label_smoothing
        gate = outputs["gate_logit"].sum() * 0.0 if not valid.any() else F.binary_cross_entropy_with_logits(
            outputs["gate_logit"][valid].float(), targets
        )
        slot = outputs["gate_logit"].sum() * 0.0
        if instance_labels is not None and lambda_slot:
            if instance_labels.shape != outputs["slot_masks"].shape[:1] + outputs["slot_masks"].shape[-1:]:
                raise ValueError("instance_labels must align with [B,V] slot masks")
            # Slot order has no semantic meaning.  For every simulator object
            # instance, match its normalized patch mask to the best slot, then
            # apply a distributional mask loss.  The discrete matching is only
            # a training-time target assignment; gradients flow through the
            # selected soft slot mask and never require simulator masks online.
            masks = outputs["slot_masks"].float().clamp_min(1e-8)
            slot = _one_to_one_instance_mask_loss(masks, instance_labels)
        return {"total": lambda_gate * gate + lambda_slot * slot, "gate": gate, "slot": slot}
