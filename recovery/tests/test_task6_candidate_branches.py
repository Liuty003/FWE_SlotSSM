import numpy as np

from scripts.collect_task6_failure_candidate_branches import (
    candidate_macros,
    cup_place_macros,
    stable_cup_grasp_macros,
)


def test_task6_candidate_macros_are_matched_and_distinct():
    candidates = candidate_macros()
    assert len(candidates) >= 5
    assert {chunk.shape for chunk in candidates.values()} == {(28, 7)}
    assert len({chunk.tobytes() for chunk in candidates.values()}) == len(candidates)


def test_task6_regrasp_candidates_finish_closed_and_lifting():
    for name, chunk in candidate_macros().items():
        assert np.all(chunk[-6:, 2] > 0)
        assert np.all(chunk[-6:, 6] > 0)
        assert np.isfinite(chunk).all()


def test_cup_place_candidates_finish_open_and_are_distinct():
    candidates = cup_place_macros()
    assert {chunk.shape for chunk in candidates.values()} == {(126, 7)}
    assert len({chunk.tobytes() for chunk in candidates.values()}) == len(candidates)
    for chunk in candidates.values():
        assert np.all(chunk[-10:, 6] < 0)


def test_stable_cup_grasps_finish_with_slow_closed_lift():
    candidates = stable_cup_grasp_macros()
    assert {chunk.shape for chunk in candidates.values()} == {(48, 7)}
    for chunk in candidates.values():
        assert np.allclose(chunk[-20:, 2], .15)
        assert np.all(chunk[-20:, 6] > 0)
