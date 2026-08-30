"""V4.3 recovery with a training-only object-direction auxiliary objective."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from recovery.models.v4_relational_recovery import V4ModelConfig
from recovery.models.v43_local_recovery import V43LocalRecovery


class V43GeometryAuxRecovery(V43LocalRecovery):
    """Preserve a V4.3 policy and learn a geometry-conditioned action residual.

    The privileged bowl-minus-gripper vector supervises ``target_delta_head``
    during training only.  At deployment the vector is predicted solely from
    policy-visible RGB-D/OFT features and conditions the residual action head.
    """

    def __init__(self, config: V4ModelConfig = V4ModelConfig(), *,
                 direct_head_version: str = "pooled_chunk_mlp_v3") -> None:
        super().__init__(config, direct_head_version=direct_head_version)
        pooled_dim = 2 * config.relation_dim
        self.target_delta_head = nn.Sequential(
            nn.LayerNorm(pooled_dim), nn.Linear(pooled_dim, config.relation_dim // 2),
            nn.GELU(), nn.Linear(config.relation_dim // 2, 3),
        )
        self.geometry_action_residual = nn.Sequential(
            nn.LayerNorm(pooled_dim + 3),
            nn.Linear(pooled_dim + 3, config.relation_dim), nn.GELU(),
            nn.Linear(config.relation_dim, config.action_chunk * config.action_dim),
        )
        # Loading V4.3 starts from exactly the archived action policy.
        nn.init.zeros_(self.geometry_action_residual[-1].weight)
        nn.init.zeros_(self.geometry_action_residual[-1].bias)

    def forward(self, **inputs: Tensor) -> dict[str, Tensor]:
        outputs = super().forward(**inputs)
        context = outputs["relational_context"].mean(dim=1)
        query = outputs["recovery_action_query"].mean(dim=1)
        pooled = torch.cat((context, query), dim=-1)
        predicted_delta = self.target_delta_head(pooled)
        residual = self.geometry_action_residual(torch.cat((pooled, predicted_delta), dim=-1))
        residual = residual.reshape_as(outputs["recovery_actions"])
        outputs["base_recovery_actions"] = outputs["recovery_actions"]
        outputs["predicted_target_delta"] = predicted_delta
        outputs["geometry_action_residual"] = residual
        outputs["recovery_actions"] = outputs["recovery_actions"] + residual
        return outputs

