import numpy as np

from scripts.run_v4_relational_online import bound_local_actions, recovery_execution_count


def test_single_step_output_cannot_enqueue_four_actions():
    assert recovery_execution_count(requested=4, available=1, latch_available=4) == 1


def test_local_action_bounds_translation_and_rotation_but_not_gripper():
    actions = np.asarray([[3.0, 4.0, 0.0, 0.0, 0.0, 2.0, 0.37]], dtype=np.float32)
    bounded, clipped = bound_local_actions(actions, translation_limit=0.5, rotation_limit=0.25)
    assert clipped
    assert np.isclose(np.linalg.norm(bounded[0, :3]), 0.5)
    assert np.isclose(np.linalg.norm(bounded[0, 3:6]), 0.25)
    assert bounded[0, 6] == actions[0, 6]
