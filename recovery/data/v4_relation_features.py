"""Frozen, policy-visible per-decision inputs for V4 relational recovery.

V4 deliberately has no temporal state.  Each row is one physical decision and
contains frozen OFT features plus geometry reconstructed from *RGB-D and fixed
camera calibration*.  ``token_points`` must therefore never contain simulator
object poses; it is the robot-frame 3D point associated with each visual token.

The contract is intentionally separate from V3: V3 features do not contain
depth-derived geometry and must not be silently reused for this method.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor


REQUIRED_KEYS = (
    "visual_tokens", "action_hidden", "token_points", "token_valid",
    "token_image_xy", "gripper_position", "gripper_image_xy", "actions",
    "gate_labels", "recovery_targets", "recovery_mask", "source_ids",
)
OPTIONAL_KEYS = ("instance_labels",)


@dataclass(frozen=True)
class V4RelationFeatures:
    """One source-labelled decision per row.

    Shapes are ``[N,V,Dv]`` visual tokens, ``[N,A,Dh]`` action hidden states,
    token-aligned robot-frame points ``[N,V,3]``, token validity ``[N,V]``,
    normalized primary-camera token/gripper coordinates ``[N,V,2]`` / ``[N,2]``,
    gripper robot-frame position ``[N,3]``, nominal action chunks
    ``[N,Hn,Da]``, and recovery targets ``[N,Hr,Da]``.  ``Hn`` and ``Hr``
    may differ: OFT always supplies its native context while a closed-loop
    recovery policy may supervise only the next action.
    ``instance_labels`` is optional training-only segmentation supervision:
    ``-1`` ignores a token, ``0`` is background, and positive values are object
    instances.  It is never an online input.
    """

    visual_tokens: Tensor
    action_hidden: Tensor
    token_points: Tensor
    token_valid: Tensor
    token_image_xy: Tensor
    gripper_position: Tensor
    gripper_image_xy: Tensor
    actions: Tensor
    gate_labels: Tensor
    recovery_targets: Tensor
    recovery_mask: Tensor
    source_ids: Tensor
    instance_labels: Tensor | None = None

    @property
    def num_rows(self) -> int:
        return int(self.visual_tokens.shape[0])

    def validate(self) -> None:
        if self.visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must be [N,V,Dv]")
        n, tokens = self.visual_tokens.shape[:2]
        if self.action_hidden.ndim != 3 or self.action_hidden.shape[0] != n:
            raise ValueError("action_hidden must be [N,A,Dh]")
        expected = (n, tokens)
        if self.token_points.shape != (*expected, 3):
            raise ValueError("token_points must be [N,V,3]")
        if self.token_valid.shape != expected or self.token_valid.dtype != torch.bool:
            raise ValueError("token_valid must be boolean [N,V]")
        if not bool(self.token_valid.any(dim=1).all()):
            raise ValueError("every row must contain at least one valid geometry token")
        if self.token_image_xy.shape != (*expected, 2):
            raise ValueError("token_image_xy must be [N,V,2]")
        if self.gripper_position.shape != (n, 3):
            raise ValueError("gripper_position must be [N,3]")
        if self.gripper_image_xy.shape != (n, 2):
            raise ValueError("gripper_image_xy must be [N,2]")
        if self.actions.ndim != 3 or self.actions.shape[0] != n:
            raise ValueError("actions must be [N,H,Da]")
        if (self.recovery_targets.ndim != 3 or self.recovery_targets.shape[0] != n
                or self.recovery_targets.shape[1] < 1
                or self.recovery_targets.shape[2] != self.actions.shape[2]):
            raise ValueError(
                "recovery_targets must be [N,Hr,Da] with Hr>=1 and the actions dimension"
            )
        if self.recovery_mask.shape != (n,) or self.recovery_mask.dtype != torch.bool:
            raise ValueError("recovery_mask must be boolean [N]")
        if self.gate_labels.shape != (n,) or not bool(
            ((self.gate_labels == -1) | (self.gate_labels == 0) | (self.gate_labels == 1)).all()
        ):
            raise ValueError("gate_labels must be [N] with values -1, 0, or 1")
        # V4.2 computes the corrective action loss only on y=1 examples.
        # A row without a complete target chunk may still be labelled y=1,
        # but a target must never be attached to NORMAL or ignored evidence.
        if bool((self.recovery_mask & self.gate_labels.ne(1)).any()):
            raise ValueError("recovery_mask may be true only for gate_labels == 1")
        if self.source_ids.shape != (n,):
            raise ValueError("source_ids must be [N]")
        if self.instance_labels is not None and self.instance_labels.shape != expected:
            raise ValueError("instance_labels must be [N,V]")
        for name in ("visual_tokens", "action_hidden", "token_points", "token_image_xy",
                     "gripper_position", "gripper_image_xy", "actions", "recovery_targets"):
            if not bool(torch.isfinite(getattr(self, name)).all()):
                raise ValueError(f"{name} contains non-finite values")

    def index(self, indices: Tensor) -> "V4RelationFeatures":
        values = {key: getattr(self, key)[indices] for key in REQUIRED_KEYS}
        if self.instance_labels is not None:
            values["instance_labels"] = self.instance_labels[indices]
        return V4RelationFeatures(**values)

    def to(self, device: torch.device | str) -> "V4RelationFeatures":
        values = {key: getattr(self, key).to(device) for key in REQUIRED_KEYS}
        if self.instance_labels is not None:
            values["instance_labels"] = self.instance_labels.to(device)
        return V4RelationFeatures(**values)

    def payload(self) -> dict[str, Tensor]:
        values = {key: getattr(self, key) for key in REQUIRED_KEYS}
        if self.instance_labels is not None:
            values["instance_labels"] = self.instance_labels
        return values


def from_tensors(payload: Mapping[str, Tensor]) -> V4RelationFeatures:
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise KeyError(f"v4 feature payload is missing required keys: {missing}")
    unknown = set(payload) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS)
    if unknown:
        raise KeyError(f"v4 feature payload contains unsupported keys: {sorted(unknown)}")
    features = V4RelationFeatures(**{key: payload.get(key) for key in (*REQUIRED_KEYS, *OPTIONAL_KEYS)})
    features.validate()
    return features


def load(path: Path) -> V4RelationFeatures:
    from safetensors.torch import load_file

    return from_tensors(load_file(str(path)))


def save(features: V4RelationFeatures, path: Path) -> None:
    from safetensors.torch import save_file

    features.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.detach().cpu().contiguous() for key, value in features.payload().items()}, str(path))
