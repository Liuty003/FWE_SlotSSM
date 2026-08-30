"""Validated frozen-feature contract for v3 action-aware SlotSSM training.

V3 trains on *causal sequences*, not shuffled single recovery windows.  The
feature file is deliberately independent from OpenVLA-OFT code so expensive
feature extraction can be run once and training can run on any GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor


REQUIRED_KEYS = ("visual_tokens", "action_hidden", "proprio", "actions")
OPTIONAL_KEYS = ("gate_labels", "recovery_targets", "recovery_mask", "source_ids")


@dataclass(frozen=True)
class V3SequenceFeatures:
    """A source-isolated batch of frozen OFT feature sequences.

    Shapes:

    * ``visual_tokens``: ``[N,T,V,Dv]``;
    * ``action_hidden``: ``[N,T,A,Dh]``;
    * ``proprio``: ``[N,T,Dp]``;
    * ``actions``: actual normalized OFT chunks ``[N,T,H,7]``;
    * ``gate_labels``: optional ``[N,T]`` labels in ``{0,1,-1}``;
    * ``recovery_targets`` / ``recovery_mask``: optional ``[N,T,H,7]`` and
      ``[N,T]`` canonical successful-recovery supervision.

    A row is one complete causal source sequence.  Never create a row by
    randomly mixing adjacent windows from different simulator trajectories.
    """

    visual_tokens: Tensor
    action_hidden: Tensor
    proprio: Tensor
    actions: Tensor
    gate_labels: Tensor | None = None
    recovery_targets: Tensor | None = None
    recovery_mask: Tensor | None = None
    source_ids: Tensor | None = None

    @property
    def num_sequences(self) -> int:
        return int(self.visual_tokens.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.visual_tokens.shape[1])

    def validate(self) -> None:
        if self.visual_tokens.ndim != 4:
            raise ValueError("visual_tokens must be [N,T,V,Dv]")
        n, t = self.visual_tokens.shape[:2]
        if self.action_hidden.ndim != 4 or self.action_hidden.shape[:2] != (n, t):
            raise ValueError("action_hidden must be [N,T,A,Dh] with the same [N,T]")
        if self.proprio.ndim != 3 or self.proprio.shape[:2] != (n, t):
            raise ValueError("proprio must be [N,T,Dp] with the same [N,T]")
        if self.actions.ndim != 4 or self.actions.shape[:2] != (n, t):
            raise ValueError("actions must be [N,T,H,Da] with the same [N,T]")
        if self.gate_labels is not None:
            if self.gate_labels.shape != (n, t) or not bool(((self.gate_labels == -1) | (self.gate_labels == 0) | (self.gate_labels == 1)).all()):
                raise ValueError("gate_labels must be [N,T] with values -1, 0, or 1")
        if (self.recovery_targets is None) != (self.recovery_mask is None):
            raise ValueError("recovery_targets and recovery_mask must be present together")
        if self.recovery_targets is not None:
            if self.recovery_targets.shape != self.actions.shape:
                raise ValueError("recovery_targets must match actions shape")
            if self.recovery_mask is None or self.recovery_mask.shape != (n, t):
                raise ValueError("recovery_mask must be [N,T]")
        if self.source_ids is not None and self.source_ids.shape != (n,):
            raise ValueError("source_ids must be [N]")

    def index(self, indices: Tensor) -> "V3SequenceFeatures":
        return V3SequenceFeatures(
            visual_tokens=self.visual_tokens[indices], action_hidden=self.action_hidden[indices],
            proprio=self.proprio[indices], actions=self.actions[indices],
            gate_labels=None if self.gate_labels is None else self.gate_labels[indices],
            recovery_targets=None if self.recovery_targets is None else self.recovery_targets[indices],
            recovery_mask=None if self.recovery_mask is None else self.recovery_mask[indices],
            source_ids=None if self.source_ids is None else self.source_ids[indices],
        )

    def to(self, device: torch.device | str) -> "V3SequenceFeatures":
        return V3SequenceFeatures(
            visual_tokens=self.visual_tokens.to(device), action_hidden=self.action_hidden.to(device),
            proprio=self.proprio.to(device), actions=self.actions.to(device),
            gate_labels=None if self.gate_labels is None else self.gate_labels.to(device),
            recovery_targets=None if self.recovery_targets is None else self.recovery_targets.to(device),
            recovery_mask=None if self.recovery_mask is None else self.recovery_mask.to(device),
            source_ids=None if self.source_ids is None else self.source_ids.to(device),
        )

    def payload(self) -> dict[str, Tensor]:
        result = {key: getattr(self, key) for key in REQUIRED_KEYS}
        result.update({key: value for key in OPTIONAL_KEYS if (value := getattr(self, key)) is not None})
        return result


def from_tensors(payload: Mapping[str, Tensor]) -> V3SequenceFeatures:
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise KeyError(f"v3 feature payload is missing required keys: {missing}")
    unknown = set(payload) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS)
    if unknown:
        raise KeyError(f"v3 feature payload contains unsupported keys: {sorted(unknown)}")
    features = V3SequenceFeatures(**{key: payload.get(key) for key in (*REQUIRED_KEYS, *OPTIONAL_KEYS)})
    features.validate()
    return features


def load(path: Path) -> V3SequenceFeatures:
    from safetensors.torch import load_file

    return from_tensors(load_file(str(path)))


def save(features: V3SequenceFeatures, path: Path) -> None:
    from safetensors.torch import save_file

    features.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.detach().cpu().contiguous() for key, value in features.payload().items()}, str(path))
