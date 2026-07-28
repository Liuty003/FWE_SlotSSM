"""Converters from legacy recovery expert data to schema v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from recovery.data.schema import Mode, RecoveryTransition, SCHEMA_VERSION


PHASE_TO_MODE = {
    "NOMINAL": Mode.NORMAL.value,
    "FORCED ERROR": Mode.FAILURE.value,
    "MISSED GRASP": Mode.FAILURE.value,
    "LOST GRASP": Mode.FAILURE.value,
    "FAILURE DETECTED": Mode.FAILURE.value,
    "RETURN TO BOWL": Mode.RECOVERY.value,
    "DESCEND": Mode.RECOVERY.value,
    "CLOSE GRIPPER": Mode.RECOVERY.value,
    "VERIFY LIFT": Mode.VERIFY.value,
    "VISUAL SETTLE": Mode.HANDOFF.value,
    "RESUMED TASK": Mode.RESUMED.value,
}


def load_legacy_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def convert_legacy_expert_episode(
    transitions_path: Path,
    *,
    split: str,
    language_instruction: str | None = None,
    source_revision: str = "legacy-recovery-expert-v1",
) -> list[RecoveryTransition]:
    legacy = load_legacy_rows(transitions_path)
    if not legacy:
        raise ValueError(f"empty legacy episode: {transitions_path}")
    result: list[RecoveryTransition] = []
    previous_action: list[float] | None = None
    previous_timestep: int | None = None
    for raw in legacy:
        timestep = int(raw["timestep"])
        if previous_timestep is not None and timestep <= previous_timestep:
            raise ValueError(
                f"legacy timesteps must be strictly increasing: "
                f"{previous_timestep} then {timestep}"
            )
        phase = str(raw.get("phase", "UNKNOWN"))
        is_expert = bool(raw.get("is_recovery_expert_action", False))
        executed = list(raw["expert_or_executed_action"])
        attached_truth = raw.get("teacher_attached_truth")
        online_inputs = {
            "rgb": raw.get("rgb"),
            "depth": raw.get("depth"),
            "language_instruction": language_instruction,
            "proprio": raw.get("robot_state"),
            "previous_executed_action": previous_action,
            "nominal_action": raw["nominal_action"],
            "target_mask": raw.get("target_mask_64"),
            "target_mask_measurement": raw.get("target_mask_measurement"),
            "visual_target_xyz": None,
            "visual_target_xyz_valid": False,
            "observation_validity": {
                "rgb": raw.get("rgb") is not None,
                "depth": raw.get("depth") is not None,
                "target_mask": raw.get("target_mask_64") is not None,
                "visual_target_xyz": False,
            },
        }
        labels = {
            "phase": phase,
            "failure_injection": bool(raw.get("is_failure_injection", False)),
            "failure_detection_event": bool(
                raw.get("is_failure_detection_event", False)
            ),
            "recovery_expert_action": is_expert,
            "recovery_completed": bool(raw.get("recovery_completed", False)),
            "success_after": bool(raw.get("success_after", False)),
            "executed_action": executed,
            "expert_action": executed if is_expert else None,
            "expert_residual": (
                [
                    float(expert) - float(nominal)
                    for expert, nominal in zip(executed, raw["nominal_action"])
                ]
                if is_expert else None
            ),
            # Legacy files predate explicit event labels. These remain unknown
            # rather than being silently inferred from privileged pose.
            "failure_type": None,
            "grasped": (
                bool(attached_truth) if attached_truth is not None else None
            ),
            "attached": (
                bool(attached_truth) if attached_truth is not None else None
            ),
            "slipped": None,
            "verified": None,
            "handoff_ready": None,
        }
        transition = RecoveryTransition(
            schema_version=SCHEMA_VERSION,
            source_format=(
                "recovery_expert_collector_v2"
                if int(raw.get("schema_version", 1)) >= 2
                else "legacy_recovery_expert_transition_v1"
            ),
            source_path=str(transitions_path.resolve()),
            task_suite=str(raw["task_suite"]),
            task_id=int(raw["task_id"]),
            episode_id=int(raw["episode_id"]),
            timestep=timestep,
            split=split,
            mode=PHASE_TO_MODE.get(phase, Mode.UNKNOWN.value),
            online_inputs=online_inputs,
            labels=labels,
            teacher_only={
                "raw_privileged_object_state": raw.get(
                    "teacher_privileged_object_state"
                ),
                "contact_state": raw.get("teacher_contact_state"),
                "attached_truth": attached_truth,
            },
            provenance={
                "source_revision": source_revision,
                "legacy_schema_version": raw.get("schema_version"),
                "conversion_is_lossless": False,
                "missing_from_legacy": [
                    "visual_target_xyz",
                    "failure_type",
                    "slipped",
                    "verified",
                    "handoff_ready",
                ] + (
                    ["grasped", "attached", "contact_state"]
                    if attached_truth is None else []
                ),
            },
        )
        transition.validate(check_files=False)
        result.append(transition)
        previous_action = [float(value) for value in executed]
        previous_timestep = timestep
    return result


def choose_latest_episode_sources(paths: Iterable[Path]) -> list[Path]:
    """Deduplicate task/episode sources, preferring lexically newer dataset roots."""
    selected: dict[tuple[str, int, int], Path] = {}
    for path in sorted(paths):
        rows = load_legacy_rows(path)
        if not rows:
            continue
        first = rows[0]
        key = (
            str(first["task_suite"]),
            int(first["task_id"]),
            int(first["episode_id"]),
        )
        selected[key] = path
    return [selected[key] for key in sorted(selected)]


def convert_online_trace(
    steps_path: Path,
    *,
    task_suite: str,
    task_id: int,
    episode_id: int,
    split: str,
    language_instruction: str | None,
    source_revision: str,
) -> list[RecoveryTransition]:
    """Convert a rich online steps trace without claiming unavailable media."""
    legacy = load_legacy_rows(steps_path)
    if not legacy:
        raise ValueError(f"empty online trace: {steps_path}")
    result: list[RecoveryTransition] = []
    previous_action: list[float] | None = None
    for raw in legacy:
        phase = str(raw.get("phase", "UNKNOWN"))
        executed = list(raw["executed_action"])
        visual_xyz = raw.get("visual_target_xyz")
        target_measurement = raw.get("target_mask")
        transition = RecoveryTransition(
            schema_version=SCHEMA_VERSION,
            source_format="online_recovery_steps_v1",
            source_path=str(steps_path.resolve()),
            task_suite=task_suite,
            task_id=task_id,
            episode_id=episode_id,
            timestep=int(raw["timestep"]),
            split=split,
            mode=PHASE_TO_MODE.get(phase, Mode.UNKNOWN.value),
            online_inputs={
                # Legacy online traces contain a rollout video but no aligned
                # per-step media files. Null is intentional and audited.
                "rgb": None,
                "depth": None,
                "language_instruction": language_instruction,
                "proprio": raw.get("robot_state_before"),
                "previous_executed_action": previous_action,
                "nominal_action": raw["nominal_action"],
                "target_mask": None,
                "target_mask_measurement": target_measurement,
                "visual_target_xyz": visual_xyz,
                "visual_target_xyz_valid": visual_xyz is not None,
                "observation_validity": {
                    "rgb": False,
                    "depth": False,
                    "target_mask": target_measurement is not None,
                    "visual_target_xyz": visual_xyz is not None,
                },
            },
            labels={
                "phase": phase,
                "failure_injection": bool(raw.get("failure_injected", False)),
                "failure_detection_event": bool(raw.get("failure_detected", False)),
                "recovery_expert_action": False,
                "recovery_completed": bool(raw.get("recovery_completed", False)),
                "success_after": bool(raw.get("success", False)),
                "executed_action": executed,
                "expert_action": None,
                "expert_residual": None,
                "learned_recovery_residual": raw.get("learned_recovery_residual"),
                "failure_type": (
                    "lost_grasp"
                    if phase == "LOST GRASP" or bool(raw.get("failure_detected"))
                    else None
                ),
                "grasped": None,
                "attached": None,
                "slipped": phase == "LOST GRASP",
                "verified": (
                    float(raw["visual_regrasp_probability"])
                    if raw.get("visual_regrasp_probability") is not None else None
                ),
                "handoff_ready": phase in ("VISUAL SETTLE", "RESUMED TASK"),
                "reward": raw.get("reward"),
                "done": bool(raw.get("done", False)),
            },
            teacher_only={
                "raw_privileged_object_state": raw.get(
                    "privileged_object_state_before"
                )
            },
            provenance={
                "source_revision": source_revision,
                "conversion_is_lossless": False,
                "missing_from_legacy": [
                    "aligned_rgb_path",
                    "aligned_depth_path",
                    "target_mask_array_path",
                    "explicit_grasped_attached_labels",
                ],
                "online_trace_only": True,
            },
        )
        transition.validate(check_files=False)
        result.append(transition)
        previous_action = [float(value) for value in executed]
    return result
