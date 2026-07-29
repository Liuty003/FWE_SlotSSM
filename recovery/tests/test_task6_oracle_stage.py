import pytest

from experiments.robot.libero.run_task6_oracle_recovery import StableBooleanLatch


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
