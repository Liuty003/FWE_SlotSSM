"""Shared temporal backbone and auxiliary heads for recovery.

The reference GRU backend is kept for deterministic CPU contract tests. The
Mamba2 backend is the intended Recovery SlotSSM implementation for training and
online use.
"""

from __future__ import annotations

import torch
from torch import nn

from recovery.data.tensor_dataset import MODEL_INPUT_DIM, PHASES

try:
    from mamba_ssm import Mamba2
except ImportError:  # pragma: no cover - exercised only without mamba installed
    Mamba2 = None


class RecoverySlotSSM(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = MODEL_INPUT_DIM,
        model_dim: int = 128,
        num_layers: int = 2,
        backend: str = "mamba2",
    ):
        super().__init__()
        if backend not in ("mamba2", "gru"):
            raise ValueError(f"unsupported backend: {backend}")
        self.backend = backend
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
        )
        if backend == "mamba2":
            if Mamba2 is None:
                raise RuntimeError("mamba_ssm.Mamba2 is unavailable")
            self.temporal_layers = nn.ModuleList(
                [
                    Mamba2(
                        d_model=model_dim,
                        d_state=64,
                        d_conv=4,
                        expand=2,
                        # With model_dim=64, the default headdim=64 yields two
                        # expanded heads and a projection stride not divisible
                        # by 8, which causal_conv1d rejects. Sixteen keeps the
                        # projected channel-last stride kernel-compatible.
                        headdim=16,
                        # The fused path requires a channel-last stride layout
                        # that is not guaranteed after our feature projection.
                        # The reference selective-scan path is still Mamba2 and
                        # is reliable for variable batch/window shapes.
                        use_mem_eff_path=False,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.temporal_norms = nn.ModuleList(
                [nn.LayerNorm(model_dim) for _ in range(num_layers)]
            )
            self.gru = None
        else:
            self.temporal_layers = nn.ModuleList()
            self.temporal_norms = nn.ModuleList()
            self.gru = nn.GRU(
                model_dim,
                model_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        self.output_norm = nn.LayerNorm(model_dim)
        self.phase_head = nn.Linear(model_dim, len(PHASES))
        self.failure_onset_head = nn.Linear(model_dim, 1)
        self.current_failure_head = nn.Linear(model_dim, 1)
        self.recovery_active_head = nn.Linear(model_dim, 1)
        self.verification_head = nn.Linear(model_dim, 1)
        self.handoff_head = nn.Linear(model_dim, 1)
        self.grasped_head = nn.Linear(model_dim, 1)
        self.lifted_head = nn.Linear(model_dim, 1)
        self.slip_head = nn.Linear(model_dim, 1)
        self.residual_action_head = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.GELU(),
            nn.Linear(64, 7),
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 3:
            raise ValueError(f"expected [B,T,D] inputs, got {tuple(inputs.shape)}")
        hidden = self.input_projection(inputs)
        if self.backend == "mamba2":
            for mixer, norm in zip(self.temporal_layers, self.temporal_norms):
                hidden = hidden + mixer(norm(hidden))
        else:
            hidden, _ = self.gru(hidden)
        hidden = self.output_norm(hidden)
        return {
            "temporal_state": hidden,
            "phase_logits": self.phase_head(hidden),
            "failure_onset_logits": self.failure_onset_head(hidden).squeeze(-1),
            "current_failure_logits": self.current_failure_head(hidden).squeeze(-1),
            "recovery_active_logits": self.recovery_active_head(hidden).squeeze(-1),
            "verification_logits": self.verification_head(hidden).squeeze(-1),
            "handoff_logits": self.handoff_head(hidden).squeeze(-1),
            "grasped_logits": self.grasped_head(hidden).squeeze(-1),
            "lifted_logits": self.lifted_head(hidden).squeeze(-1),
            "slip_logits": self.slip_head(hidden).squeeze(-1),
            "residual_action": self.residual_action_head(hidden),
        }
