"""V3 action-aware object memory and neural recovery policy.

The module intentionally starts *after* frozen OpenVLA-OFT feature extraction.
It never imports OFT and therefore makes the training/deployment boundary
explicit: ``visual_tokens`` and ``action_hidden`` are frozen OFT outputs,
whereas every parameter in this file belongs to the learned recovery policy.

``FutureSlotPredictor`` and ``SlotFeatureDecoder`` are auxiliary training
heads.  They are returned by :meth:`forward_sequence` for loss computation but
are not needed by :meth:`step`, which is the deployment graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class V3ModelConfig:
    """Dimensions of frozen OFT features and the learned v3 policy."""

    visual_dim: int = 2176
    action_hidden_dim: int = 4096
    proprio_dim: int = 8
    action_dim: int = 7
    action_chunk: int = 8
    num_slots: int = 16
    slot_dim: int = 512
    hidden_dim: int = 512
    num_heads: int = 8

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _require_rank(tensor: Tensor, rank: int, name: str) -> None:
    if tensor.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got {tuple(tensor.shape)}")


class OFTTokenSlotAttention(nn.Module):
    """Small learnable Slot Attention operating on frozen OFT visual tokens.

    The slot queries are persistent learnable object queries.  Cross-attention
    is deliberately used instead of assigning a fixed slot index to an object;
    the following temporal memory is what carries identity across decisions.
    """

    def __init__(self, visual_dim: int, slot_dim: int, num_slots: int, num_heads: int) -> None:
        super().__init__()
        if slot_dim % num_heads:
            raise ValueError("slot_dim must be divisible by num_heads")
        self.num_slots = num_slots
        self.token = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, slot_dim))
        self.slot_queries = nn.Parameter(torch.empty(1, num_slots, slot_dim))
        self.cross_attention = nn.MultiheadAttention(slot_dim, num_heads, batch_first=True)
        self.update = nn.Sequential(
            nn.LayerNorm(slot_dim), nn.Linear(slot_dim, 2 * slot_dim), nn.GELU(),
            nn.Linear(2 * slot_dim, slot_dim),
        )
        nn.init.normal_(self.slot_queries, std=slot_dim ** -0.5)

    def forward(self, visual_tokens: Tensor) -> Tensor:
        _require_rank(visual_tokens, 3, "visual_tokens")
        tokens = self.token(visual_tokens)
        queries = self.slot_queries.expand(tokens.shape[0], -1, -1)
        attended, _ = self.cross_attention(queries, tokens, tokens, need_weights=False)
        return attended + self.update(attended)


class ActionAwareSlotSSM(nn.Module):
    """Causal per-slot memory conditioned on previous chunk and proprio.

    This is a gated state-space update rather than a recurrent visual-only
    encoder.  Every slot receives the *same* robot action/proprio context while
    retaining its own hidden state, exactly implementing
    ``s_n^k = SSM(s_{n-1}^k, [z_n^k, A_{n-1}, p_n])``.
    """

    def __init__(self, config: V3ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(config.action_chunk * config.action_dim),
            nn.Linear(config.action_chunk * config.action_dim, config.hidden_dim), nn.GELU(),
        )
        self.proprio_encoder = nn.Sequential(
            nn.LayerNorm(config.proprio_dim), nn.Linear(config.proprio_dim, config.hidden_dim), nn.GELU(),
        )
        self.input_projection = nn.Sequential(
            nn.LayerNorm(config.slot_dim + 2 * config.hidden_dim),
            nn.Linear(config.slot_dim + 2 * config.hidden_dim, config.slot_dim), nn.GELU(),
        )
        self.cell = nn.GRUCell(config.slot_dim, config.slot_dim)

    def encode_action(self, action_chunk: Tensor) -> Tensor:
        if action_chunk.ndim != 3 or action_chunk.shape[1:] != (
            self.config.action_chunk, self.config.action_dim,
        ):
            raise ValueError(
                "action_chunk must be [B, action_chunk, action_dim], got "
                f"{tuple(action_chunk.shape)}"
            )
        return self.action_encoder(action_chunk.flatten(1))

    def forward(self, instantaneous_slots: Tensor, previous_action: Tensor, proprio: Tensor,
                previous_state: Tensor | None = None) -> Tensor:
        _require_rank(instantaneous_slots, 3, "instantaneous_slots")
        batch, slots, dim = instantaneous_slots.shape
        if slots != self.config.num_slots or dim != self.config.slot_dim:
            raise ValueError("instantaneous_slots does not match model config")
        if proprio.shape != (batch, self.config.proprio_dim):
            raise ValueError("proprio does not match [B, proprio_dim]")
        if previous_state is None:
            previous_state = instantaneous_slots.new_zeros(batch, slots, dim)
        if previous_state.shape != instantaneous_slots.shape:
            raise ValueError("previous_state must have the same shape as instantaneous_slots")
        action_context = self.encode_action(previous_action)[:, None].expand(-1, slots, -1)
        proprio_context = self.proprio_encoder(proprio)[:, None].expand(-1, slots, -1)
        update_input = self.input_projection(torch.cat((instantaneous_slots, action_context, proprio_context), dim=-1))
        return self.cell(update_input.flatten(0, 1), previous_state.flatten(0, 1)).view_as(instantaneous_slots)


class SlotFeatureDecoder(nn.Module):
    """Training-only slot-to-frozen-token reconstruction head."""

    def __init__(self, config: V3ModelConfig) -> None:
        super().__init__()
        if config.slot_dim % config.num_heads:
            raise ValueError("slot_dim must be divisible by num_heads")
        self.query = nn.Sequential(nn.LayerNorm(config.visual_dim), nn.Linear(config.visual_dim, config.slot_dim))
        self.attention = nn.MultiheadAttention(config.slot_dim, config.num_heads, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(config.slot_dim), nn.Linear(config.slot_dim, config.visual_dim))

    def forward(self, visual_tokens: Tensor, slots: Tensor) -> Tensor:
        queries = self.query(visual_tokens)
        decoded, _ = self.attention(queries, slots, slots, need_weights=False)
        return self.output(decoded)


class FutureSlotPredictor(nn.Module):
    """Training-only action-conditioned one-decision future slot predictor."""

    def __init__(self, config: V3ModelConfig, action_encoder: nn.Module) -> None:
        super().__init__()
        self.config = config
        # Reuse exactly the same action representation consumed by SlotSSM.
        # Keep a non-registered reference: the encoder is owned by
        # ``slot_ssm``.  Registering it again creates aliased safetensors keys
        # and makes a deployment checkpoint impossible to serialize.
        object.__setattr__(self, "action_encoder", action_encoder)
        self.transition = nn.Sequential(
            nn.LayerNorm(2 * config.slot_dim + config.hidden_dim),
            nn.Linear(2 * config.slot_dim + config.hidden_dim, 2 * config.slot_dim), nn.GELU(),
            nn.Linear(2 * config.slot_dim, config.slot_dim),
        )

    def forward(self, slots: Tensor, action_chunk: Tensor) -> Tensor:
        _require_rank(slots, 3, "slots")
        action = self.action_encoder(action_chunk.flatten(1))[:, None].expand(-1, slots.shape[1], -1)
        global_state = slots.mean(dim=1, keepdim=True).expand_as(slots)
        delta = self.transition(torch.cat((slots, global_state, action), dim=-1))
        return slots + delta


class ActionEffectGate(nn.Module):
    """Binary gate over current slots, slot changes, previous action and OFT state."""

    def __init__(self, config: V3ModelConfig, action_encoder: nn.Module) -> None:
        super().__init__()
        self.config = config
        object.__setattr__(self, "action_encoder", action_encoder)
        self.slot_transition = nn.Sequential(
            nn.LayerNorm(2 * config.slot_dim + config.hidden_dim),
            nn.Linear(2 * config.slot_dim + config.hidden_dim, config.hidden_dim), nn.GELU(),
        )
        self.pool_query = nn.Parameter(torch.empty(1, 1, config.hidden_dim))
        self.pool = nn.MultiheadAttention(config.hidden_dim, config.num_heads, batch_first=True)
        self.oft = nn.Sequential(nn.LayerNorm(config.action_hidden_dim), nn.Linear(config.action_hidden_dim, config.hidden_dim), nn.GELU())
        self.proprio = nn.Sequential(nn.LayerNorm(config.proprio_dim), nn.Linear(config.proprio_dim, config.hidden_dim), nn.GELU())
        self.head = nn.Sequential(
            nn.LayerNorm(3 * config.hidden_dim), nn.Linear(3 * config.hidden_dim, config.hidden_dim), nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        nn.init.normal_(self.pool_query, std=config.hidden_dim ** -0.5)

    def forward(self, current_slots: Tensor, previous_slots: Tensor, previous_action: Tensor,
                action_hidden: Tensor, proprio: Tensor) -> Tensor:
        _require_rank(action_hidden, 3, "action_hidden")
        if current_slots.shape != previous_slots.shape:
            raise ValueError("current_slots and previous_slots must have the same shape")
        action = self.action_encoder(previous_action.flatten(1))[:, None].expand(-1, current_slots.shape[1], -1)
        transitions = self.slot_transition(torch.cat((current_slots, current_slots - previous_slots, action), dim=-1))
        query = self.pool_query.expand(current_slots.shape[0], -1, -1)
        pooled, _ = self.pool(query, transitions, transitions, need_weights=False)
        oft = self.oft(action_hidden.mean(dim=1))
        robot = self.proprio(proprio)
        return self.head(torch.cat((pooled[:, 0], oft, robot), dim=-1)).squeeze(-1)


class SlotConditionedRecoveryHead(nn.Module):
    """Generate each recovery action token by cross-attending to object slots."""

    def __init__(self, config: V3ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.oft = nn.Sequential(nn.LayerNorm(config.action_hidden_dim), nn.Linear(config.action_hidden_dim, config.slot_dim))
        self.positions = nn.Parameter(torch.empty(1, config.action_chunk, config.slot_dim))
        self.cross_attention = nn.MultiheadAttention(config.slot_dim, config.num_heads, batch_first=True)
        self.proprio = nn.Sequential(nn.LayerNorm(config.proprio_dim), nn.Linear(config.proprio_dim, config.slot_dim), nn.GELU())
        self.action = nn.Sequential(
            nn.LayerNorm(2 * config.slot_dim), nn.Linear(2 * config.slot_dim, config.hidden_dim), nn.GELU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )
        nn.init.normal_(self.positions, std=config.slot_dim ** -0.5)

    def forward(self, action_hidden: Tensor, slots: Tensor, proprio: Tensor) -> Tensor:
        _require_rank(action_hidden, 3, "action_hidden")
        seed = self.oft(action_hidden.mean(dim=1))[:, None]
        queries = seed + self.positions
        attended, _ = self.cross_attention(queries, slots, slots, need_weights=False)
        robot = self.proprio(proprio)[:, None].expand(-1, self.config.action_chunk, -1)
        return self.action(torch.cat((attended, robot), dim=-1))


class V3ActionAwareRecoveryPolicy(nn.Module):
    """The complete v3 learned graph after frozen OFT feature extraction."""

    def __init__(self, config: V3ModelConfig = V3ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.slot_attention = OFTTokenSlotAttention(config.visual_dim, config.slot_dim, config.num_slots, config.num_heads)
        self.slot_ssm = ActionAwareSlotSSM(config)
        self.slot_decoder = SlotFeatureDecoder(config)
        self.future_predictor = FutureSlotPredictor(config, self.slot_ssm.action_encoder)
        self.gate = ActionEffectGate(config, self.slot_ssm.action_encoder)
        self.recovery_head = SlotConditionedRecoveryHead(config)

    def step(self, *, visual_tokens: Tensor, action_hidden: Tensor, proprio: Tensor,
             previous_action: Tensor, previous_memory: Tensor | None = None,
             previous_slots: Tensor | None = None) -> dict[str, Tensor]:
        """Run the deployment graph for one decision without auxiliary heads."""
        instantaneous_slots = self.slot_attention(visual_tokens)
        slots = self.slot_ssm(instantaneous_slots, previous_action, proprio, previous_memory)
        if previous_slots is None:
            previous_slots = torch.zeros_like(slots)
        gate_logit = self.gate(slots, previous_slots, previous_action, action_hidden, proprio)
        recovery_actions = self.recovery_head(action_hidden, slots, proprio)
        return {
            "instantaneous_slots": instantaneous_slots,
            "slots": slots,
            "gate_logit": gate_logit,
            "recovery_actions": recovery_actions,
        }

    def forward_sequence(self, *, visual_tokens: Tensor, action_hidden: Tensor, proprio: Tensor,
                         actions: Tensor, initial_memory: Tensor | None = None) -> dict[str, Tensor]:
        """Run a causal decision sequence and retain training-only auxiliary heads.

        Args:
            visual_tokens: ``[B,T,N_visual,D_visual]`` frozen dual-view OFT tokens.
            action_hidden: ``[B,T,N_action,D_action]`` frozen OFT action tokens.
            proprio: ``[B,T,D_proprio]`` policy-visible robot state.
            actions: actual normalized chunks ``[B,T,H,7]``.  The action at
                decision ``t`` is the previous-action input at decision ``t+1``.
        """
        _require_rank(visual_tokens, 4, "visual_tokens")
        _require_rank(action_hidden, 4, "action_hidden")
        _require_rank(proprio, 3, "proprio")
        _require_rank(actions, 4, "actions")
        batch, steps = visual_tokens.shape[:2]
        if action_hidden.shape[:2] != (batch, steps) or proprio.shape[:2] != (batch, steps) or actions.shape[:2] != (batch, steps):
            raise ValueError("all sequence tensors must share [B,T]")
        if visual_tokens.shape[-1] != self.config.visual_dim or action_hidden.shape[-1] != self.config.action_hidden_dim:
            raise ValueError("frozen feature dimensions do not match V3ModelConfig")
        if proprio.shape[-1] != self.config.proprio_dim or actions.shape[2:] != (self.config.action_chunk, self.config.action_dim):
            raise ValueError("robot/action dimensions do not match V3ModelConfig")
        memory = initial_memory
        states, instantaneous, reconstructions, gate_logits, recovery_actions = [], [], [], [], []
        previous_slots: Tensor | None = None
        zero_action = actions.new_zeros(batch, self.config.action_chunk, self.config.action_dim)
        for index in range(steps):
            previous_action = zero_action if index == 0 else actions[:, index - 1]
            instant = self.slot_attention(visual_tokens[:, index])
            memory = self.slot_ssm(instant, previous_action, proprio[:, index], memory)
            gate_previous = torch.zeros_like(memory) if previous_slots is None else previous_slots
            states.append(memory)
            instantaneous.append(instant)
            reconstructions.append(self.slot_decoder(visual_tokens[:, index], instant))
            gate_logits.append(self.gate(memory, gate_previous, previous_action, action_hidden[:, index], proprio[:, index]))
            recovery_actions.append(self.recovery_head(action_hidden[:, index], memory, proprio[:, index]))
            previous_slots = memory
        state_sequence = torch.stack(states, dim=1)
        future_slots = self.future_predictor(
            state_sequence[:, :-1].reshape(-1, self.config.num_slots, self.config.slot_dim),
            actions[:, :-1].reshape(-1, self.config.action_chunk, self.config.action_dim),
        ).view(batch, max(steps - 1, 0), self.config.num_slots, self.config.slot_dim)
        return {
            "instantaneous_slots": torch.stack(instantaneous, dim=1),
            "slots": state_sequence,
            "reconstructed_visual_tokens": torch.stack(reconstructions, dim=1),
            "future_slots": future_slots,
            "gate_logits": torch.stack(gate_logits, dim=1),
            "recovery_actions": torch.stack(recovery_actions, dim=1),
        }

    @staticmethod
    def losses(outputs: Mapping[str, Tensor], *, visual_tokens: Tensor, gate_labels: Tensor | None = None,
               recovery_targets: Tensor | None = None, recovery_mask: Tensor | None = None,
               lambda_future: float = 1.0, lambda_gate: float = 1.0,
               lambda_recovery: float = 1.0, lambda_slot: float = 1.0,
               gate_positive_weight: float = 1.0) -> dict[str, Tensor]:
        """Compute v3 losses; a ``gate_label`` of ``-1`` is ignored."""
        slot_loss = F.mse_loss(outputs["reconstructed_visual_tokens"].float(), visual_tokens.detach().float())
        states = outputs["slots"]
        if states.shape[1] < 2:
            future_loss = states.sum() * 0.0
        else:
            future_loss = F.mse_loss(outputs["future_slots"].float(), states[:, 1:].detach().float())
        gate_loss = states.sum() * 0.0
        if gate_labels is not None:
            valid = gate_labels.ge(0)
            if valid.any():
                gate_loss = F.binary_cross_entropy_with_logits(
                    outputs["gate_logits"][valid].float(), gate_labels[valid].float(),
                    pos_weight=outputs["gate_logits"].new_tensor(gate_positive_weight, dtype=torch.float32),
                )
        recovery_loss = states.sum() * 0.0
        if recovery_targets is not None:
            if recovery_mask is None:
                raise ValueError("recovery_mask is required with recovery_targets")
            valid = recovery_mask.bool()
            if valid.any():
                recovery_loss = F.l1_loss(
                    outputs["recovery_actions"][valid].float(), recovery_targets[valid].float()
                )
        total = lambda_slot * slot_loss + lambda_future * future_loss + lambda_gate * gate_loss + lambda_recovery * recovery_loss
        return {
            "total": total, "slot": slot_loss, "future": future_loss,
            "gate": gate_loss, "recovery": recovery_loss,
        }
