import numpy as np

from scripts.mine_mug_pudding_failures import (
    MUG,
    PLATE,
    PUDDING,
    RED_MUG,
    choose_failure,
    event_candidates,
    stable_true,
)


def test_stable_true_requires_confirmation_and_latches():
    assert stable_true([False, True, True, False, True, True, True], confirm=3) == [
        False, False, False, False, False, False, True,
    ]


def test_success_is_never_selected_for_recovery():
    candidate = {"timestep": 10, "confidence": "high"}
    selected, status = choose_failure([candidate], {"stage2_reached": True}, True)
    assert selected is None
    assert status == "normal_success"


def test_high_confidence_event_wins_over_earlier_medium_event():
    medium = {"timestep": 10, "confidence": "medium"}
    high = {"timestep": 20, "confidence": "high"}
    selected, status = choose_failure([medium, high], {"stage2_reached": True}, False)
    assert selected is high
    assert status == "auto_high_confidence"


def test_ambiguous_incomplete_stage_is_sent_to_review():
    selected, status = choose_failure([], {"stage2_reached": False}, False)
    assert selected is None
    assert status == "needs_review_mug_stage_incomplete_without_causal_event"


def test_wrong_object_close_is_a_high_confidence_causal_event():
    def row(timestep, gripper_action):
        objects = {
            MUG: [0.17, 0.0, 0.45],
            RED_MUG: [0.04, 0.0, 0.45],
            PLATE: [0.3, 0.0, 0.44],
            PUDDING: [-0.2, 0.0, 0.45],
        }
        robot = {
            "robot0_eef_pos": [0.0, 0.0, 0.45],
            "robot0_eef_quat": [1.0, 0.0, 0.0, 0.0],
            "robot0_gripper_qpos": [0.04, -0.04],
        }
        return {
            "timestep": timestep,
            "executed_action": [0, 0, 0, 0, 0, 0, gripper_action],
            "robot_state_before": robot,
            "privileged_object_state_before": objects,
            "mug_on_plate_before": False,
            "mug_on_plate_after": False,
            "pudding_in_right_region_after": False,
        }

    rows = [row(i, 1.0 if i == 2 else -1.0) for i in range(4)]
    candidates, _ = event_candidates(rows)
    assert [(item["timestep"], item["failure_type"]) for item in candidates] == [
        (2, "wrong_object_acquisition")
    ]
    assert candidates[0]["confidence"] == "high"


def test_close_far_from_intended_target_is_empty_close_failure():
    objects = {
        MUG: [0.24, 0.0, 0.45], RED_MUG: [-0.3, 0.0, 0.45],
        PLATE: [0.0, 0.0, 0.44], PUDDING: [0.0, 0.2, 0.45],
    }
    robot = {
        "robot0_eef_pos": [0.0, 0.0, 0.45],
        "robot0_eef_quat": [1.0, 0.0, 0.0, 0.0],
        "robot0_gripper_qpos": [0.04, -0.04],
    }
    rows = [{
        "timestep": i,
        "executed_action": [0, 0, 0, 0, 0, 0, 1.0 if i == 1 else -1.0],
        "robot_state_before": robot,
        "privileged_object_state_before": objects,
        "mug_on_plate_before": False,
        "mug_on_plate_after": False,
        "pudding_in_right_region_after": False,
    } for i in range(3)]
    candidates, _ = event_candidates(rows)
    assert candidates[0]["timestep"] == 1
    assert candidates[0]["failure_type"] == "premature_empty_close"
