import numpy as np

from scripts.collect_task6_failure_candidate_branches import candidate_macros


def test_task6_candidate_macros_are_matched_and_distinct():
    candidates = candidate_macros()
    assert len(candidates) >= 5
    assert {chunk.shape for chunk in candidates.values()} == {(28, 7)}
    assert len({chunk.tobytes() for chunk in candidates.values()}) == len(candidates)


def test_task6_regrasp_candidates_finish_closed_and_lifting():
    for name, chunk in candidate_macros().items():
        assert np.all(chunk[-8:, 2] > 0)
        assert np.all(chunk[-8:, 6] > 0)
        assert np.isfinite(chunk).all()
