"""Conservative temporal teacher labels from simulator object state.

The labeler uses privileged state only to produce supervision. It records label
validity/confidence explicitly and never modifies ``online_inputs``.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np

from recovery.data.schema import RecoveryTransition


TARGET_KEY = "akita_black_bowl_1_pos"


def _vector(row: RecoveryTransition, key: str) -> np.ndarray | None:
    state = row.teacher_only.get("raw_privileged_object_state")
    if not isinstance(state, dict) or state.get(key) is None:
        return None
    value = np.asarray(state[key], dtype=np.float32)
    return value if value.shape == (3,) and np.isfinite(value).all() else None


def _eef(row: RecoveryTransition) -> np.ndarray | None:
    proprio = row.online_inputs.get("proprio")
    if not isinstance(proprio, dict) or proprio.get("robot0_eef_pos") is None:
        return None
    value = np.asarray(proprio["robot0_eef_pos"], dtype=np.float32)
    return value if value.shape == (3,) and np.isfinite(value).all() else None


def label_episode(
    rows: Sequence[RecoveryTransition],
    *,
    lift_threshold_m: float = 0.015,
    verify_lift_threshold_m: float = 0.04,
    attach_distance_m: float = 0.10,
    detach_distance_m: float = 0.12,
    comotion_error_m: float = 0.008,
    comotion_min_motion_m: float = 0.001,
    slip_drop_m: float = 0.025,
    confirm_steps: int = 2,
) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot label an empty episode")
    ordered = sorted(rows, key=lambda row: row.timestep)
    target_positions = [_vector(row, TARGET_KEY) for row in ordered]
    eef_positions = [_eef(row) for row in ordered]
    first_target = next((value for value in target_positions if value is not None), None)
    if first_target is None:
        raise ValueError("episode has no valid privileged target position")
    initial_z = float(first_target[2])

    comotion_history: deque[bool] = deque(maxlen=max(confirm_steps, 1))
    lift_history: deque[bool] = deque(maxlen=max(confirm_steps, 1))
    recent_z: deque[float] = deque(maxlen=6)
    attached = False
    previous_explicit_attached: bool | None = None
    ever_lifted = False
    failure_active = False
    previous_failure_injection = False
    counts = {
        "lifted": 0,
        "attached": 0,
        "slip_event": 0,
        "verified": 0,
        "failure_active": 0,
        "failure_onset": 0,
        "handoff_ready": 0,
        "invalid_state": 0,
    }

    for index, row in enumerate(ordered):
        target = target_positions[index]
        eef = eef_positions[index]
        valid = target is not None and eef is not None
        labels = row.labels
        if not valid:
            labels.update(
                {
                    "teacher_event_labels_valid": False,
                    "lifted": None,
                    "grasped": None,
                    "attached": None,
                    "slipped": None,
                    "verified": None,
                    "handoff_ready": None,
                    "failure_active": None,
                }
            )
            counts["invalid_state"] += 1
            continue

        lift = float(target[2] - initial_z)
        distance = float(np.linalg.norm(target - eef))
        lifted = lift >= lift_threshold_m
        ever_lifted = ever_lifted or lifted
        recent_z.append(float(target[2]))

        comoving = False
        separation_growth = 0.0
        if index > 0 and target_positions[index - 1] is not None and eef_positions[index - 1] is not None:
            target_delta = target - target_positions[index - 1]
            eef_delta = eef - eef_positions[index - 1]
            comoving = bool(
                distance <= attach_distance_m
                and np.linalg.norm(target_delta) >= comotion_min_motion_m
                and np.linalg.norm(target_delta - eef_delta) <= comotion_error_m
            )
            previous_distance = float(
                np.linalg.norm(target_positions[index - 1] - eef_positions[index - 1])
            )
            separation_growth = distance - previous_distance
        comotion_history.append(comoving)
        lift_history.append(lift >= verify_lift_threshold_m)

        drop_from_recent_peak = (
            max(recent_z) - float(target[2]) if recent_z else 0.0
        )
        explicit_attached = row.teacher_only.get("attached_truth")
        slip_event = bool(
            ever_lifted
            and drop_from_recent_peak >= slip_drop_m
            and (separation_growth > 0.005 or distance >= attach_distance_m)
        )
        if explicit_attached is not None:
            attached = bool(explicit_attached)
            slip_event = bool(
                slip_event
                or (
                    ever_lifted
                    and previous_explicit_attached is True
                    and not attached
                )
            )
            previous_explicit_attached = attached
            attachment_source = "sim_bilateral_finger_contact"
        else:
            if slip_event or distance >= detach_distance_m:
                attached = False
            elif sum(comotion_history) >= confirm_steps:
                attached = True
            attachment_source = "privileged_comotion_proxy"

        verified = bool(
            attached
            and len(lift_history) == lift_history.maxlen
            and all(lift_history)
        )
        failure_onset = False
        failure_onset_reason = None
        if previous_failure_injection and not failure_active:
            failure_active = True
            failure_onset = True
            failure_onset_reason = "post_intervention_observation"
        elif slip_event and not failure_active:
            failure_active = True
            failure_onset = True
            failure_onset_reason = "lifted_contact_loss"
        elif labels.get("failure_detection_event") and not failure_active:
            failure_active = True
            failure_onset = True
            failure_onset_reason = "legacy_detector"
        if labels.get("recovery_completed"):
            failure_active = False
        handoff_ready = bool(labels.get("recovery_completed"))
        failure_type = (
            "lost_grasp"
            if failure_active
            or failure_onset
            or labels.get("failure_detection_event")
            or labels.get("phase") in ("LOST GRASP", "MISSED GRASP")
            else None
        )
        if labels.get("phase") == "MISSED GRASP":
            failure_type = "missed_grasp"
        confidence = (
            1.0
            if explicit_attached is not None
            else 1.0 if distance < 0.30 else 0.5
        )
        labels.update(
            {
                "teacher_event_labels_valid": True,
                "teacher_event_label_confidence": confidence,
                "attachment_label_source": attachment_source,
                "target_lift_m_teacher": lift,
                "target_eef_distance_m_teacher": distance,
                "target_eef_comoving_teacher": comoving,
                "lifted": lifted,
                "grasped": attached,
                "attached": attached,
                "slipped": slip_event,
                "verified": verified,
                "handoff_ready": handoff_ready,
                "failure_active": failure_active,
                "failure_onset_teacher": failure_onset,
                "failure_onset_reason": failure_onset_reason,
                "failure_type": failure_type,
            }
        )
        row.teacher_only["oracle_failure_onset"] = failure_onset
        row.teacher_only["oracle_failure_type"] = failure_type
        for key, value in (
            ("lifted", lifted),
            ("attached", attached),
            ("slip_event", slip_event),
            ("verified", verified),
            ("failure_active", failure_active),
            ("failure_onset", failure_onset),
            ("handoff_ready", handoff_ready),
        ):
            counts[key] += int(value)
        previous_failure_injection = bool(labels.get("failure_injection"))

    return {
        "episode_id": ordered[0].episode_id,
        "num_rows": len(ordered),
        "initial_target_z": initial_z,
        "thresholds": {
            "lift_threshold_m": lift_threshold_m,
            "verify_lift_threshold_m": verify_lift_threshold_m,
            "attach_distance_m": attach_distance_m,
            "detach_distance_m": detach_distance_m,
            "comotion_error_m": comotion_error_m,
            "comotion_min_motion_m": comotion_min_motion_m,
            "slip_drop_m": slip_drop_m,
            "confirm_steps": confirm_steps,
        },
        "counts": counts,
    }
