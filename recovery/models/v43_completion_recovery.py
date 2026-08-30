"""Auxiliary-conditioned recovery with an independent learned handoff head."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from recovery.models.v4_relational_recovery import V4ModelConfig
from recovery.models.v43_auxiliary_conditioned_recovery import V43AuxiliaryConditionedRecovery


class V43CompletionRecovery(V43AuxiliaryConditionedRecovery):
    def __init__(self, config: V4ModelConfig = V4ModelConfig()):
        super().__init__(config)
        d = 2 * config.relation_dim
        self.completion_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, **inputs: Tensor) -> dict[str, Tensor]:
        outputs = super().forward(**inputs)
        pooled0 = torch.cat((outputs['relational_context'].mean(1),
                             outputs['recovery_action_query'].mean(1)), 1)
        pooled = pooled0 + self.task_trunk(pooled0)
        outputs['completion_logit'] = self.completion_head(pooled).squeeze(1)
        outputs['completion_probability'] = outputs['completion_logit'].sigmoid()
        return outputs
