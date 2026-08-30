"""Checkpoint-compatible OpenVLA-OFT L1 action head.

The official ``action_heads.py`` imports Diffusers at module import time even
when only the deterministic L1 head is used.  Recovery extraction and online
evaluation should not inherit that unrelated optional dependency.  Keep this
small implementation structurally identical to the official head so archived
``action_head--*_checkpoint.pt`` files load strictly.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MLPResNetBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, value: Tensor) -> Tensor:
        return value + self.ffn(value)


class MLPResNet(nn.Module):
    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(
            [MLPResNetBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, value: Tensor) -> Tensor:
        value = self.relu(self.fc1(self.layer_norm1(value)))
        for block in self.mlp_resnet_blocks:
            value = block(value)
        return self.fc2(self.layer_norm2(value))


class OFTL1RegressionActionHead(nn.Module):
    """The official deterministic 8x7 OpenVLA-OFT action decoder."""

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 4096,
                 action_dim: int = 7, action_chunk: int = 8) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk = action_chunk
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * action_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def predict_action(self, actions_hidden_states: Tensor) -> Tensor:
        if (actions_hidden_states.ndim != 3
                or actions_hidden_states.shape[1] != self.action_chunk * self.action_dim):
            raise ValueError("OFT action hidden must be [B, action_chunk*action_dim, D]")
        batch_size = actions_hidden_states.shape[0]
        rearranged = actions_hidden_states.reshape(batch_size, self.action_chunk, -1)
        return self.model(rearranged)

