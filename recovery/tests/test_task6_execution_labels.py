import numpy as np

from scripts.build_task6_execution_labels import distance_to_region


def test_distance_to_region_is_zero_inside():
    assert distance_to_region([0.15, 0.10, 0.4]) == 0.0


def test_distance_to_region_uses_nearest_boundary():
    assert np.isclose(distance_to_region([0.07, 0.01, 0.4]), 0.05)
