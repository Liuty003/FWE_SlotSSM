"""Versioned recovery transition schema with explicit privilege separation.

The schema deliberately stores policy-observable inputs, supervision labels, and
simulator-only teacher state in separate dictionaries. Training code must call
``policy_inputs()`` rather than consuming a raw JSON row.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 2
ACTION_DIM = 7


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    DEVELOPMENT_TEST = "development_test"
    FINAL_TEST = "final_test"


class Mode(str, Enum):
    NORMAL = "normal"
    FAILURE = "failure"
    RECOVERY = "recovery"
    VERIFY = "verify"
    HANDOFF = "handoff"
    RESUMED = "resumed"
    UNKNOWN = "unknown"


# This is the only set of fields the learned online policy may consume.
ONLINE_INPUT_ALLOWLIST = frozenset(
    {
        "rgb",
        "depth",
        "language_instruction",
        "proprio",
        "previous_executed_action",
        "nominal_action",
        "target_mask",
        "target_mask_measurement",
        "visual_target_xyz",
        "visual_target_xyz_valid",
        "target_feature",
        "observation_validity",
    }
)

TEACHER_ONLY_FIELDS = frozenset(
    {
        "object_pose",
        "target_pose",
        "goal_pose",
        "object_height",
        "contact_state",
        "instance_segmentation",
        "grasped_truth",
        "attached_truth",
        "oracle_phase",
        "oracle_action",
        "oracle_failure_type",
        "oracle_failure_onset",
        "oracle_placement_relation",
        "raw_privileged_object_state",
    }
)


def _require_vector(name: str, value: Any, size: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of length {size}")
    if len(value) != size:
        raise ValueError(f"{name} must have length {size}, got {len(value)}")
    result = [float(item) for item in value]
    if any(item != item or abs(item) == float("inf") for item in result):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _validate_optional_path(name: str, value: Any, check_files: bool) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string or null")
    if check_files and not Path(value).is_file():
        raise ValueError(f"{name} does not exist: {value}")


@dataclass
class RecoveryTransition:
    schema_version: int
    source_format: str
    source_path: str
    task_suite: str
    task_id: int
    episode_id: int
    timestep: int
    split: str
    mode: str
    online_inputs: dict[str, Any]
    labels: dict[str, Any]
    teacher_only: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, check_files: bool = False) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version={self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if not self.task_suite:
            raise ValueError("task_suite is required")
        if self.task_id < 0 or self.episode_id < 0 or self.timestep < 0:
            raise ValueError("task_id, episode_id, and timestep must be non-negative")
        if self.split not in {item.value for item in Split}:
            raise ValueError(f"invalid split: {self.split}")
        if self.mode not in {item.value for item in Mode}:
            raise ValueError(f"invalid mode: {self.mode}")

        unexpected = set(self.online_inputs) - ONLINE_INPUT_ALLOWLIST
        if unexpected:
            raise ValueError(f"online_inputs contains non-allowlisted fields: {sorted(unexpected)}")
        leaked = set(self.online_inputs) & TEACHER_ONLY_FIELDS
        if leaked:
            raise ValueError(f"teacher-only fields leaked into online_inputs: {sorted(leaked)}")
        teacher_unknown = set(self.teacher_only) - TEACHER_ONLY_FIELDS
        if teacher_unknown:
            raise ValueError(f"unknown teacher_only fields: {sorted(teacher_unknown)}")

        for name in ("rgb", "depth", "target_mask"):
            _validate_optional_path(name, self.online_inputs.get(name), check_files)
        if "nominal_action" not in self.online_inputs:
            raise ValueError("online_inputs.nominal_action is required")
        self.online_inputs["nominal_action"] = _require_vector(
            "online_inputs.nominal_action", self.online_inputs["nominal_action"], ACTION_DIM
        )
        previous = self.online_inputs.get("previous_executed_action")
        if previous is not None:
            self.online_inputs["previous_executed_action"] = _require_vector(
                "online_inputs.previous_executed_action", previous, ACTION_DIM
            )
        if "executed_action" not in self.labels:
            raise ValueError("labels.executed_action is required")
        self.labels["executed_action"] = _require_vector(
            "labels.executed_action", self.labels["executed_action"], ACTION_DIM
        )
        if "expert_action" in self.labels and self.labels["expert_action"] is not None:
            self.labels["expert_action"] = _require_vector(
                "labels.expert_action", self.labels["expert_action"], ACTION_DIM
            )

    def policy_inputs(self) -> dict[str, Any]:
        """Return a defensive copy containing only legal online inputs."""
        unexpected = set(self.online_inputs) - ONLINE_INPUT_ALLOWLIST
        if unexpected:
            raise ValueError(f"non-policy fields present: {sorted(unexpected)}")
        return {key: self.online_inputs[key] for key in self.online_inputs}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, check_files: bool = False
    ) -> "RecoveryTransition":
        transition = cls(
            schema_version=int(value["schema_version"]),
            source_format=str(value["source_format"]),
            source_path=str(value["source_path"]),
            task_suite=str(value["task_suite"]),
            task_id=int(value["task_id"]),
            episode_id=int(value["episode_id"]),
            timestep=int(value["timestep"]),
            split=str(value["split"]),
            mode=str(value["mode"]),
            online_inputs=dict(value["online_inputs"]),
            labels=dict(value["labels"]),
            teacher_only=dict(value.get("teacher_only", {})),
            provenance=dict(value.get("provenance", {})),
        )
        transition.validate(check_files=check_files)
        return transition


def audit_privileged_leakage(row: Mapping[str, Any]) -> list[str]:
    """Return dotted paths where known privileged names occur under online_inputs."""
    online = row.get("online_inputs", {})
    if not isinstance(online, Mapping):
        return ["online_inputs:not-a-mapping"]
    return [
        f"online_inputs.{name}"
        for name in sorted(set(online) & TEACHER_ONLY_FIELDS)
    ]
