"""Matched recovery-candidate groups with strict privilege separation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from recovery.data.schema import ACTION_DIM


CANDIDATE_SCHEMA_VERSION = 1


def _action_vector(name: str, value: Any, size: int = ACTION_DIM) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a {size}-D sequence")
    result = [float(item) for item in value]
    if len(result) != size:
        raise ValueError(f"{name} must have length {size}")
    if any(item != item or abs(item) == float("inf") for item in result):
        raise ValueError(f"{name} contains non-finite values")
    return result


@dataclass
class RecoveryCandidateGroup:
    schema_version: int
    task_suite: str
    task_id: int
    episode_id: int
    timestep: int
    split: str
    context_transition_path: str
    nominal_action: list[float]
    candidate_names: list[str]
    candidate_actions: list[list[float]]
    outcome_valid: list[bool]
    teacher_outcomes: list[dict[str, Any]]
    provenance: dict[str, Any]

    def validate(self) -> None:
        if self.schema_version != CANDIDATE_SCHEMA_VERSION:
            raise ValueError("unsupported candidate schema version")
        self.nominal_action = _action_vector("nominal_action", self.nominal_action)
        if len(self.candidate_names) < 2:
            raise ValueError("a matched group requires at least two candidates")
        count = len(self.candidate_names)
        if not (
            len(self.candidate_actions)
            == len(self.outcome_valid)
            == len(self.teacher_outcomes)
            == count
        ):
            raise ValueError("candidate arrays have inconsistent lengths")
        if len(set(self.candidate_names)) != count:
            raise ValueError("candidate names must be unique")
        self.candidate_actions = [
            _action_vector(f"candidate_actions[{index}]", action)
            for index, action in enumerate(self.candidate_actions)
        ]
        if len({tuple(action) for action in self.candidate_actions}) != count:
            raise ValueError("candidate actions must be numerically distinct")
        required = {
            "eef_delta_m",
            "target_delta_m",
            "attached_after",
            "lifted_after",
            "slipped_after",
            "failure_persistent_after",
            "handoff_ready_after",
            "teacher_value",
        }
        for index, (valid, outcome) in enumerate(
            zip(self.outcome_valid, self.teacher_outcomes)
        ):
            if valid and not required.issubset(outcome):
                missing = sorted(required - set(outcome))
                raise ValueError(
                    f"teacher_outcomes[{index}] missing required fields: {missing}"
                )
            if valid:
                _action_vector("eef_delta_m", outcome["eef_delta_m"], 3)
                _action_vector("target_delta_m", outcome["target_delta_m"], 3)
                teacher_value = float(outcome["teacher_value"])
                if teacher_value != teacher_value or abs(teacher_value) == float("inf"):
                    raise ValueError(
                        f"teacher_outcomes[{index}].teacher_value is non-finite"
                    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecoveryCandidateGroup":
        group = cls(
            schema_version=int(value["schema_version"]),
            task_suite=str(value["task_suite"]),
            task_id=int(value["task_id"]),
            episode_id=int(value["episode_id"]),
            timestep=int(value["timestep"]),
            split=str(value["split"]),
            context_transition_path=str(value["context_transition_path"]),
            nominal_action=list(value["nominal_action"]),
            candidate_names=[str(item) for item in value["candidate_names"]],
            candidate_actions=[
                list(action) for action in value["candidate_actions"]
            ],
            outcome_valid=[bool(item) for item in value["outcome_valid"]],
            teacher_outcomes=[
                dict(item) for item in value["teacher_outcomes"]
            ],
            provenance=dict(value.get("provenance", {})),
        )
        group.validate()
        return group


def read_candidate_jsonl(path: Path) -> list[RecoveryCandidateGroup]:
    groups = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                groups.append(RecoveryCandidateGroup.from_dict(json.loads(line)))
            except Exception as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return groups


def write_candidate_jsonl(
    path: Path, groups: Sequence[RecoveryCandidateGroup]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for group in groups:
            group.validate()
            handle.write(
                json.dumps(group.to_dict(), ensure_ascii=False) + "\n"
            )
