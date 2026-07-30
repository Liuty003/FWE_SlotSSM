import numpy as np
import torch

from scripts.replay_task6_role_masks import position_error, resize_masks


def test_resize_masks_preserves_binary_contract():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:4, 3:5] = True
    resized = resize_masks([mask], 4)
    assert resized.shape == (1, 4, 4)
    assert resized.dtype == torch.bool


def test_role_mask_position_error():
    obs = {"cup_pos": np.asarray([0.0, 0.0, 0.0])}
    recorded = {"cup_pos": [0.0, 0.003, 0.004]}
    assert np.isclose(position_error(obs, recorded), 0.005)
