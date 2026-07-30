import pytest

from experiments.robot.libero.run_task6_oracle_recovery import (
    ReversibleStageLatch,
    StableBooleanLatch,
)


def test_stable_boolean_latch_requires_consecutive_confirmations():
    latch = StableBooleanLatch(confirm_steps=3)
    assert not latch.update(True)
    assert not latch.update(False)
    assert not latch.update(True)
    assert not latch.update(True)
    assert latch.update(True)


def test_stable_boolean_latch_remains_latched():
    latch = StableBooleanLatch(confirm_steps=1)
    assert latch.update(True)
    assert latch.update(False)


def test_stable_boolean_latch_rejects_nonpositive_window():
    with pytest.raises(ValueError):
        StableBooleanLatch(confirm_steps=0)


def test_reversible_stage_latch_can_rollback_and_reenter():
    latch = ReversibleStageLatch(enter_steps=2, rollback_steps=2)
    assert not latch.update(True, False)
    assert latch.update(True, False)
    assert latch.update(False, True)
    assert not latch.update(False, True)
    assert not latch.update(True, False)
    assert latch.update(True, False)
