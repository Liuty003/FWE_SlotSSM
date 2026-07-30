import numpy as np

from scripts.replay_task6_oracle_visual_dataset import (
    max_object_position_error,
)


def test_max_object_position_error_uses_position_keys_only():
    obs = {
        "object_1_pos": np.asarray([1.0, 2.0, 3.0]),
        "object_1_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
    }
    recorded = {
        "object_1_pos": [1.0, 2.003, 3.004],
        "object_1_quat": [1.0, 0.0, 0.0, 0.0],
    }
    assert np.isclose(max_object_position_error(obs, recorded), 0.005)


def test_max_object_position_error_ignores_missing_keys():
    assert max_object_position_error({}, {"object_1_pos": [1, 2, 3]}) == 0.0
