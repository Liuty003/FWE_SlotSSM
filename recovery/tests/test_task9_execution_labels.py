import json
from pathlib import Path

from scripts.build_task9_execution_labels import (
    classify_attempt,
    find_grasp_attempts,
    first_transport_drop,
)


ROOT = Path(
    "/data2/lty/naive-e-slotssm-runs/libero10-natural-baseline-v1/task9"
)


def load(episode):
    path = ROOT / f"task9-episode{episode}" / "steps.jsonl"
    if not path.is_file():
        return None
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_episode5_acquires_then_drops():
    rows = load(5)
    if rows is None:
        return
    attempts = find_grasp_attempts(rows)
    assert attempts == [76]
    acquisition = classify_attempt(rows, attempts[0])
    assert acquisition["acquired"]
    drop = first_transport_drop(
        rows, attempts[0], acquisition["assessment_timestep"], True
    )
    assert drop is not None
    assert 90 <= drop["drop_timestep"] <= 100
    assert drop["fall_from_peak_m"] > 0.05


def test_episode6_is_failed_acquisition_not_transport_drop():
    rows = load(6)
    if rows is None:
        return
    attempts = find_grasp_attempts(rows)
    # OpenVLA briefly commands close at 95, reopens, then makes a sustained
    # second attempt at 156.  Neither attempt establishes transport.
    assert attempts == [95, 156]
    for attempt in attempts:
        acquisition = classify_attempt(rows, attempt)
        assert acquisition["failed_acquisition"]
        assert first_transport_drop(
            rows, attempt, acquisition["assessment_timestep"], False
        ) is None
