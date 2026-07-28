"""Build causal, episode-contained windows from recovery transitions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from recovery.data.schema import RecoveryTransition


@dataclass(frozen=True)
class SequenceWindow:
    task_suite: str
    task_id: int
    episode_id: int
    split: str
    start_timestep: int
    end_timestep: int
    rows: tuple[RecoveryTransition, ...]


def group_episodes(
    rows: Iterable[RecoveryTransition],
) -> dict[tuple[str, int, int], list[RecoveryTransition]]:
    grouped: dict[tuple[str, int, int], list[RecoveryTransition]] = defaultdict(list)
    for row in rows:
        grouped[(row.task_suite, row.task_id, row.episode_id)].append(row)
    for key, episode in grouped.items():
        episode.sort(key=lambda item: item.timestep)
        timesteps = [item.timestep for item in episode]
        if len(timesteps) != len(set(timesteps)):
            raise ValueError(f"duplicate timestep in episode {key}")
        splits = {item.split for item in episode}
        if len(splits) != 1:
            raise ValueError(f"split leakage within episode {key}: {sorted(splits)}")
    return dict(grouped)


def build_windows(
    rows: Iterable[RecoveryTransition],
    *,
    length: int,
    stride: int = 1,
    require_contiguous: bool = True,
    event_labels: Sequence[str] = (),
) -> list[SequenceWindow]:
    if length <= 0 or stride <= 0:
        raise ValueError("length and stride must be positive")
    windows: list[SequenceWindow] = []
    for (suite, task_id, episode_id), episode in group_episodes(rows).items():
        for start in range(0, max(0, len(episode) - length + 1), stride):
            selected = episode[start:start + length]
            if len(selected) != length:
                continue
            timesteps = [row.timestep for row in selected]
            if require_contiguous and any(
                right != left + 1 for left, right in zip(timesteps, timesteps[1:])
            ):
                continue
            if event_labels and not any(
                any(bool(row.labels.get(label, False)) for label in event_labels)
                for row in selected
            ):
                continue
            windows.append(
                SequenceWindow(
                    task_suite=suite,
                    task_id=task_id,
                    episode_id=episode_id,
                    split=selected[0].split,
                    start_timestep=selected[0].timestep,
                    end_timestep=selected[-1].timestep,
                    rows=tuple(selected),
                )
            )
    return windows
