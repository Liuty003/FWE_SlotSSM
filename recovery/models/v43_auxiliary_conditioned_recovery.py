"""V4.3 recovery whose auxiliary state representation conditions actions softly."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from recovery.models.v4_relational_recovery import V4ModelConfig
from recovery.models.v43_auxiliary_state_recovery import V43AuxiliaryStateRecovery


class V43AuxiliaryConditionedRecovery(V43AuxiliaryStateRecovery):
    """Keep gate/OFT frozen while allowing auxiliary supervision to shape actions."""

    def __init__(self, config: V4ModelConfig = V4ModelConfig()):
        super().__init__(config)
        d = 2 * config.relation_dim
        self.task_trunk = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d),
        )
        self.aux_to_action = nn.Sequential(
            nn.Linear(9, 128), nn.GELU(), nn.Linear(128, d + 3),
        )
        # Start as the archived policy and learn the coupling during fine-tuning.
        nn.init.zeros_(self.task_trunk[-1].weight)
        nn.init.zeros_(self.task_trunk[-1].bias)
        nn.init.zeros_(self.aux_to_action[-1].weight)
        nn.init.zeros_(self.aux_to_action[-1].bias)

    def forward(self, **inputs: Tensor) -> dict[str, Tensor]:
        # Call the geometry parent directly so this class owns all task heads.
        o = super(V43AuxiliaryStateRecovery, self).forward(**inputs)
        pooled0 = torch.cat((o["relational_context"].mean(1),
                             o["recovery_action_query"].mean(1)), 1)
        pooled = pooled0 + self.task_trunk(pooled0)
        delta = self.target_delta_head(pooled)
        phase_logits = self.phase_head(pooled)
        state_logits = self.state_head(pooled)
        drawer = self.drawer_head(pooled).squeeze(1)
        auxiliary = torch.cat((phase_logits.softmax(1), state_logits.sigmoid(),
                               drawer.unsqueeze(1)), 1)
        condition0 = torch.cat((pooled, delta), 1)
        condition = condition0 + self.aux_to_action(auxiliary)
        residual = self.unified_action_residual(condition).reshape(-1, self.config.action_chunk, 6)
        gripper = self.gripper_head(condition)
        base = o["recovery_actions"]
        o.update({
            "predicted_target_delta": delta,
            "phase_logits": phase_logits,
            "phase_probability": phase_logits.softmax(1),
            "state_logits": state_logits,
            "predicted_drawer_qpos": drawer,
            "unified_action_residual": residual,
            "gripper_logits": gripper,
        })
        o["recovery_actions"] = torch.cat((base[..., :6] + residual,
                                            gripper.sigmoid().unsqueeze(-1)), 2)
        return o
