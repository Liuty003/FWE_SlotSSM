"""Dataset-level validation and audit summaries."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from recovery.data.schema import RecoveryTransition
from recovery.data.sequence_builder import build_windows, group_episodes


def audit_dataset(
    rows: Iterable[RecoveryTransition],
    *,
    check_files: bool = True,
    window_lengths: tuple[int, ...] = (32, 64),
) -> dict[str, Any]:
    materialized = list(rows)
    errors: list[str] = []
    missing_files: list[str] = []
    for index, row in enumerate(materialized):
        try:
            row.validate(check_files=False)
        except Exception as error:
            errors.append(f"row {index}: {error}")
        if check_files:
            for name in ("rgb", "depth", "target_mask"):
                value = row.online_inputs.get(name)
                if value is not None and not Path(value).is_file():
                    missing_files.append(value)

    episodes = group_episodes(materialized)
    split_episodes = Counter(
        episode[0].split for episode in episodes.values() if episode
    )
    modes = Counter(row.mode for row in materialized)
    phases = Counter(str(row.labels.get("phase")) for row in materialized)
    sources = Counter(row.source_format for row in materialized)
    input_availability = {
        name: sum(row.online_inputs.get(name) is not None for row in materialized)
        for name in ("rgb", "depth", "target_mask", "visual_target_xyz")
    }
    teacher_availability = {
        name: sum(row.teacher_only.get(name) is not None for row in materialized)
        for name in ("contact_state", "attached_truth")
    }
    window_counts = {
        str(length): len(build_windows(materialized, length=length))
        for length in window_lengths
    }
    event_window_counts = {
        str(length): len(
            build_windows(
                materialized,
                length=length,
                event_labels=(
                    "failure_injection",
                    "failure_detection_event",
                    "recovery_expert_action",
                    "recovery_completed",
                ),
            )
        )
        for length in window_lengths
    }
    return {
        "schema_version": 2,
        "valid": not errors and not missing_files,
        "num_rows": len(materialized),
        "num_episodes": len(episodes),
        "split_episode_counts": dict(split_episodes),
        "source_format_counts": dict(sources),
        "mode_counts": dict(modes),
        "phase_counts": dict(phases),
        "online_input_availability": input_availability,
        "teacher_label_availability": teacher_availability,
        "num_attached_truth_rows": sum(
            row.teacher_only.get("attached_truth") is True for row in materialized
        ),
        "num_recovery_expert_rows": sum(
            bool(row.labels.get("recovery_expert_action")) for row in materialized
        ),
        "num_failure_detection_events": sum(
            bool(row.labels.get("failure_detection_event")) for row in materialized
        ),
        "num_final_success_episodes": sum(
            bool(episode[-1].labels.get("success_after"))
            for episode in episodes.values() if episode
        ),
        "window_counts": window_counts,
        "event_window_counts": event_window_counts,
        "missing_file_count": len(missing_files),
        "missing_files_sample": missing_files[:20],
        "errors": errors[:100],
    }
