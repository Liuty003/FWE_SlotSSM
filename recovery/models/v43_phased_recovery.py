"""Phase-routed V4.3 recovery policy for long multi-stage manipulation."""

from __future__ import annotations

import copy
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from recovery.models.v43_local_recovery import V43LocalRecovery
from recovery.models.v4_relational_recovery import V4ModelConfig


PHASE_NAMES = ("clear_gripper", "approach_grasp", "grasp_lift", "transfer_place", "close_drawer")


class V43PhasedRecovery(V43LocalRecovery):
    def __init__(self, config: V4ModelConfig, *, direct_head_version: str = "context_action_query_v2"):
        super().__init__(config, direct_head_version=direct_head_version)
        self.phase_classifier = nn.Sequential(
            nn.LayerNorm(2 * config.relation_dim),
            nn.Linear(2 * config.relation_dim, config.relation_dim), nn.GELU(),
            nn.Linear(config.relation_dim, len(PHASE_NAMES)),
        )
        self.phase_actions = nn.ModuleList([copy.deepcopy(self.direct_action) for _ in PHASE_NAMES])

    def forward(self, **inputs: Tensor) -> dict[str, Tensor]:
        outputs = super().forward(**inputs)
        context = outputs["relational_context"]
        query = outputs["recovery_action_query"]
        phase_input = torch.cat((context.mean(1), query.mean(1)), dim=-1)
        logits = self.phase_classifier(phase_input)
        action_input = torch.cat((context, query), dim=-1)
        if self.direct_head_version == "pooled_chunk_mlp_v3":
            action_input = action_input.mean(dim=1)
        candidates = torch.stack([
            head(action_input).reshape(
                context.shape[0], self.config.action_chunk, self.config.action_dim)
            for head in self.phase_actions
        ], dim=1)
        phase = logits.argmax(-1)
        selected = candidates[torch.arange(len(phase), device=phase.device), phase]
        outputs.update(phase_logits=logits, phase_probability=logits.softmax(-1),
                       predicted_phase=phase, phase_recovery_candidates=candidates,
                       recovery_actions=selected)
        return outputs

    @staticmethod
    def phased_loss(outputs: dict[str, Tensor], targets: Tensor, phase_labels: Tensor):
        phase_loss = F.cross_entropy(outputs["phase_logits"], phase_labels)
        selected = outputs["phase_recovery_candidates"][
            torch.arange(len(phase_labels), device=phase_labels.device), phase_labels]
        action_loss = F.l1_loss(selected.float(), targets.float())
        return phase_loss + action_loss, phase_loss, action_loss
