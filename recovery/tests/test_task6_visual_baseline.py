import pytest

from scripts.train_task6_visual_state_baseline import (
    confirmed_trigger,
    parse_ids,
    split_rows,
)


def test_split_rows_rejects_episode_leakage():
    rows = [{"episode_id": 0}, {"episode_id": 1}]
    with pytest.raises(ValueError, match="leakage"):
        split_rows(rows, {"train": (0,), "validation": (0,), "test": (1,)})


def test_split_rows_preserves_whole_episodes():
    rows = [
        {"episode_id": 0, "timestep": 0},
        {"episode_id": 0, "timestep": 1},
        {"episode_id": 1, "timestep": 0},
        {"episode_id": 2, "timestep": 0},
    ]
    splits = split_rows(
        rows, {"train": (0,), "validation": (1,), "test": (2,)}
    )
    assert [row["episode_id"] for row in splits["train"]] == [0, 0]
    assert [row["episode_id"] for row in splits["validation"]] == [1]


def test_confirmed_trigger_requires_consecutive_frames():
    assert confirmed_trigger([False, True, False, True, True], 2) == 4
    assert confirmed_trigger([True, False, True], 2) is None


def test_parse_ids_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        parse_ids("1,1")
