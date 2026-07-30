import torch

from scripts.train_task6_grounded_role_mamba import (
    first_rollback,
    stable_history,
)


def test_stable_history_requires_confirmed_completion():
    labels = torch.tensor([0, 1, 0, 1, 1, 1, 0])
    assert stable_history(labels, 3).tolist() == [
        False, False, False, False, False, True, True
    ]


def test_first_rollback_after_entry():
    values = [False, False, False, True, True, True]
    assert first_rollback(values, start=1, confirm_frames=3) == 5
