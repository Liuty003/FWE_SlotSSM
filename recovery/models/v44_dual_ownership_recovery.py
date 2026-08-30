"""V4.4.2 local recovery with independent enter-risk and resume-value heads."""

from __future__ import annotations

from torch import Tensor, nn

from recovery.models.v4_relational_recovery import V4ModelConfig
from recovery.models.v43_local_recovery import V43LocalRecovery


class V44DualOwnershipRecovery(V43LocalRecovery):
    """Share the recovery representation while keeping ownership semantics separate.

    ``gate_probability`` remains the enter-risk probability inherited from
    V4.3. ``resume_probability`` estimates whether a fresh nominal-policy
    replan from the current state is safe.  The two terminal MLPs have no
    shared parameters, so calibrating resume cannot move the takeover gate.
    """

    def __init__(
        self,
        config: V4ModelConfig = V4ModelConfig(),
        *,
        direct_head_version: str = "pooled_chunk_mlp_v3",
        recovery_output_horizon: int | None = 1,
    ) -> None:
        super().__init__(
            config,
            direct_head_version=direct_head_version,
            recovery_output_horizon=recovery_output_horizon,
        )
        self.resume_head = nn.Sequential(
            nn.LayerNorm(config.relation_dim),
            nn.Linear(config.relation_dim, config.relation_dim),
            nn.GELU(),
            nn.Linear(config.relation_dim, 1),
        )

    def forward(self, **inputs: Tensor) -> dict[str, Tensor]:
        outputs = super().forward(**inputs)
        resume_logit = self.resume_head(outputs["relational_context"].mean(dim=1)).squeeze(-1)
        outputs["resume_logit"] = resume_logit
        outputs["resume_probability"] = resume_logit.sigmoid()
        return outputs
