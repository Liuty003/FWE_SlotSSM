"""Shared temporal failure-risk model over task-aligned RGB role slots."""

from __future__ import annotations

import torch
from torch import nn

try:
    from mamba_ssm import Mamba2
except ImportError:  # pragma: no cover
    Mamba2 = None


class SharedRiskSlotSSM(nn.Module):
    """Task adapters followed by one shared causal SlotSSM and risk head.

    Inputs must already use the canonical role order
    ``(recovery_target, context, distractor)``.  The only task-specific
    parameters are adapters for frozen task-specific visual encoders; temporal
    dynamics and the failure probability are genuinely shared.
    """

    def __init__(
        self, *, input_dim: int = 305, model_dim: int = 128,
        num_layers: int = 2, backend: str = "mamba2",
    ) -> None:
        super().__init__()
        if backend not in {"mamba2", "gru"}:
            raise ValueError(f"unsupported backend: {backend}")
        self.backend = backend
        self.input_dim = input_dim
        self.task_adapters = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, model_dim), nn.GELU())
            for _ in range(2)
        ])
        if backend == "mamba2":
            if Mamba2 is None:
                raise RuntimeError("mamba_ssm.Mamba2 is unavailable")
            self.layers = nn.ModuleList([
                Mamba2(d_model=model_dim, d_state=64, d_conv=4, expand=2,
                       headdim=16, use_mem_eff_path=False)
                for _ in range(num_layers)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(num_layers)])
            self.gru = None
        else:
            self.layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.gru = nn.GRU(model_dim, model_dim, num_layers=num_layers, batch_first=True)
        self.output_norm = nn.LayerNorm(model_dim)
        self.current_failure_head = nn.Linear(model_dim, 1)

    def forward(self, inputs: torch.Tensor, task_index: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"inputs must be [B,T,{self.input_dim}]")
        if task_index.shape != (inputs.shape[0],):
            raise ValueError("task_index must be [B]")
        if task_index.min() < 0 or task_index.max() >= len(self.task_adapters):
            raise ValueError("task_index is out of range")
        hidden = torch.empty(
            *inputs.shape[:2], self.current_failure_head.in_features,
            device=inputs.device, dtype=inputs.dtype,
        )
        for index, adapter in enumerate(self.task_adapters):
            selected = task_index == index
            if selected.any():
                hidden[selected] = adapter(inputs[selected])
        if self.backend == "mamba2":
            for layer, norm in zip(self.layers, self.norms):
                hidden = hidden + layer(norm(hidden))
        else:
            hidden, _ = self.gru(hidden)
        hidden = self.output_norm(hidden)
        return {
            "temporal_state": hidden,
            "current_failure_logits": self.current_failure_head(hidden).squeeze(-1),
        }
