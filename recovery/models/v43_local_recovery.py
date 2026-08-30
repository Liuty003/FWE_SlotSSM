"""V4.3 local recovery policy and its deployment latch.

V4.3 keeps the V4.2 object/geometry/language-conditioned relational
frontend, but deliberately removes the frozen OFT hidden-residual bottleneck
from the recovery branch.  The branch predicts a short normalized action
chunk directly.  Frozen OFT remains the exact NORMAL policy at deployment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from recovery.models.v4_relational_recovery import V4ModelConfig, V4RelationalRecovery


class V43LocalRecovery(V4RelationalRecovery):
    """Relational gate plus a direct, short-horizon recovery action head."""

    def __init__(self, config: V4ModelConfig = V4ModelConfig(), *,
                 direct_head_version: str = "context_action_query_v2",
                 recovery_output_horizon: int | None = None) -> None:
        super().__init__(config)
        expected_tokens = config.action_chunk * config.action_dim
        self.expected_action_tokens = expected_tokens
        self.recovery_output_horizon = (
            config.action_chunk if recovery_output_horizon is None else recovery_output_horizon
        )
        if not 1 <= self.recovery_output_horizon <= config.action_chunk:
            raise ValueError("recovery_output_horizon must be in [1, action_chunk]")
        output_tokens = self.recovery_output_horizon * config.action_dim
        self.direct_head_version = direct_head_version
        if (self.recovery_output_horizon != config.action_chunk
                and direct_head_version != "pooled_chunk_mlp_v3"):
            raise ValueError(
                "short recovery horizons require pooled_chunk_mlp_v3 so input token count "
                "and output action count can differ"
            )
        if direct_head_version == "context_only_v1":
            # Kept only so archived V4.3 checkpoints remain loadable.  This
            # head was observed to collapse toward one scalar per action token
            # because CrossAttn values do not preserve enough OFT query/action
            # identity on their own.
            self.direct_action = nn.Sequential(
                nn.LayerNorm(config.relation_dim),
                nn.Linear(config.relation_dim, 1),
            )
        elif direct_head_version in {"context_action_query_v2", "pooled_chunk_mlp_v3"}:
            # Recovery generation is conditioned on both the selected
            # relational evidence C_t and the OFT action query H_t^a.  The gate
            # remains context-only, preserving the shared-evidence hypothesis.
            self.direct_action = nn.Sequential(
                nn.LayerNorm(2 * config.relation_dim),
                nn.Linear(2 * config.relation_dim, config.relation_dim),
                nn.GELU(),
                nn.Linear(
                    config.relation_dim,
                    1 if direct_head_version == "context_action_query_v2" else output_tokens,
                ),
            )
        else:
            raise ValueError(f"unsupported direct head version: {direct_head_version}")

    def forward(self, **inputs: Tensor) -> dict[str, Tensor]:
        outputs = super().forward(**inputs)
        context = outputs["relational_context"]
        if context.shape[1] != self.expected_action_tokens:
            raise ValueError(
                "action_hidden token count must equal action_chunk * action_dim: "
                f"got {context.shape[1]} and expected {self.expected_action_tokens}"
            )
        direct_input = context
        if self.direct_head_version in {"context_action_query_v2", "pooled_chunk_mlp_v3"}:
            action_query = self.action_query(inputs["action_hidden"])
            direct_input = torch.cat((context, action_query), dim=-1)
            outputs["recovery_action_query"] = action_query
        if self.direct_head_version == "pooled_chunk_mlp_v3":
            direct_input = direct_input.mean(dim=1)
        outputs["recovery_actions"] = self.direct_action(direct_input).reshape(
            context.shape[0], self.recovery_output_horizon, self.config.action_dim,
        )
        return outputs

    @staticmethod
    def recovery_loss(outputs: dict[str, Tensor], targets: Tensor, recovery_mask: Tensor,
                      action_dimension_weights: Tensor | None = None) -> Tensor:
        """Mean L1 over recovery rows only, in checkpoint-normalized coordinates."""
        predicted = outputs["recovery_actions"]
        if (targets.ndim != 3 or targets.shape[0] != predicted.shape[0]
                or targets.shape[2] != predicted.shape[2]
                or targets.shape[1] < predicted.shape[1]):
            raise ValueError("recovery targets must contain predicted [B,H,A] horizon")
        targets = targets[:, :predicted.shape[1]]
        if recovery_mask.shape != predicted.shape[:1] or recovery_mask.dtype != torch.bool:
            raise ValueError("recovery_mask must be boolean [B]")
        if not recovery_mask.any():
            return predicted.sum() * 0.0
        difference = (predicted[recovery_mask].float() - targets[recovery_mask].float()).abs()
        if action_dimension_weights is not None:
            if action_dimension_weights.shape != predicted.shape[-1:] or not torch.all(action_dimension_weights > 0):
                raise ValueError("action_dimension_weights must be positive [action_dim]")
            weights = action_dimension_weights.float() / action_dimension_weights.float().mean()
            difference = difference * weights
        return difference.mean()

    @staticmethod
    def route_actions(nominal_actions: Tensor, recovery_actions: Tensor, recovering: Tensor) -> Tensor:
        """Route their common executable horizon with an exact NORMAL bypass.

        A closed-loop recovery head may emit one action while native OFT emits
        eight.  Mixed batches therefore route the common prefix; the online
        latch independently decides how many of those actions to enqueue.
        """
        if (nominal_actions.ndim != 3 or recovery_actions.ndim != 3
                or nominal_actions.shape[0] != recovery_actions.shape[0]
                or nominal_actions.shape[2] != recovery_actions.shape[2]):
            raise ValueError("nominal and recovery actions must be compatible [B,H,A] tensors")
        if recovering.shape != nominal_actions.shape[:1] or recovering.dtype != torch.bool:
            raise ValueError("recovering must be boolean [B]")
        horizon = min(nominal_actions.shape[1], recovery_actions.shape[1])
        return torch.where(
            recovering[:, None, None], recovery_actions[:, :horizon], nominal_actions[:, :horizon],
        )


@dataclass(frozen=True)
class RecoveryLatchConfig:
    trigger_threshold: float = 0.5
    recovery_horizon_steps: int = 52
    minimum_recovery_steps: int = 40
    release_threshold: float = 0.5
    cooldown_steps: int = 16
    trigger_confirmation_decisions: int = 1
    release_confirmation_decisions: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.trigger_threshold <= 1.0:
            raise ValueError("trigger_threshold must lie in [0, 1]")
        if not 0.0 <= self.release_threshold <= 1.0:
            raise ValueError("release_threshold must lie in [0, 1]")
        if (self.recovery_horizon_steps < 1 or self.cooldown_steps < 0 or
                min(self.trigger_confirmation_decisions, self.release_confirmation_decisions) < 1 or
                not 0 <= self.minimum_recovery_steps <= self.recovery_horizon_steps):
            raise ValueError("recovery_horizon_steps must be positive and cooldown_steps nonnegative")


class RecoveryLatch:
    """One gate event starts a bounded recovery episode instead of one action chunk.

    Call ``select`` once at each policy decision, cap execution with
    ``executable_steps``, then call ``advance`` with the number of simulator
    actions actually executed.  This prevents gate flicker from interrupting a
    multi-decision regrasp and prevents immediate retriggering after handoff.
    """

    def __init__(self, config: RecoveryLatchConfig = RecoveryLatchConfig()) -> None:
        self.config = config
        self.recovery_steps_remaining = 0
        self.cooldown_steps_remaining = 0
        self.activation_count = 0
        self.executed_recovery_steps = 0
        self.handoff_count = 0
        self.pending_trigger_decisions = 0
        self.pending_release_decisions = 0

    @property
    def recovering(self) -> bool:
        return self.recovery_steps_remaining > 0

    def select(self, gate_probability: float, *, completion_probability: float | None = None) -> bool:
        if not 0.0 <= gate_probability <= 1.0:
            raise ValueError("gate_probability must lie in [0, 1]")
        if self.recovering:
            self.pending_trigger_decisions = 0
            release = (gate_probability < self.config.release_threshold if completion_probability is None
                       else completion_probability >= self.config.release_threshold)
            eligible = self.executed_recovery_steps >= self.config.minimum_recovery_steps
            self.pending_release_decisions = self.pending_release_decisions + 1 if eligible and release else 0
            if self.pending_release_decisions >= self.config.release_confirmation_decisions:
                self.recovery_steps_remaining = 0
                self.cooldown_steps_remaining = self.config.cooldown_steps
                self.handoff_count += 1
                self.pending_release_decisions = 0
                return False
            return True
        if self.cooldown_steps_remaining == 0:
            if gate_probability >= self.config.trigger_threshold:
                self.pending_trigger_decisions += 1
                if self.pending_trigger_decisions >= self.config.trigger_confirmation_decisions:
                    self.pending_trigger_decisions = 0
                    self.recovery_steps_remaining = self.config.recovery_horizon_steps
                    self.executed_recovery_steps = 0
                    self.pending_release_decisions = 0
                    self.activation_count += 1
                    return True
            else:
                self.pending_trigger_decisions = 0
        else:
            self.pending_trigger_decisions = 0
        return False

    def executable_steps(self, requested_steps: int) -> int:
        if requested_steps < 1:
            raise ValueError("requested_steps must be positive")
        if not self.recovering:
            return requested_steps
        if self.executed_recovery_steps < self.config.minimum_recovery_steps:
            until_search = self.config.minimum_recovery_steps - self.executed_recovery_steps
            return min(requested_steps, self.recovery_steps_remaining, until_search)
        # Search for the non-monotonic resumable boundary at single-step
        # resolution; task511 succeeds at 93 but already fails again at 96.
        return 1

    def advance(self, executed_steps: int) -> None:
        if executed_steps < 1:
            raise ValueError("executed_steps must be positive")
        if self.recovering:
            if executed_steps > self.recovery_steps_remaining:
                raise ValueError("recovery execution crossed the latched handoff boundary")
            self.recovery_steps_remaining -= executed_steps
            self.executed_recovery_steps += executed_steps
            if self.recovery_steps_remaining == 0:
                self.cooldown_steps_remaining = self.config.cooldown_steps
        elif self.cooldown_steps_remaining:
            self.cooldown_steps_remaining = max(0, self.cooldown_steps_remaining - executed_steps)

    def force_handoff(self) -> None:
        """Return ownership to the nominal policy after an external verified event."""
        if self.recovering:
            self.recovery_steps_remaining = 0
            self.cooldown_steps_remaining = self.config.cooldown_steps
            self.handoff_count += 1
        self.pending_trigger_decisions = 0
        self.pending_release_decisions = 0
