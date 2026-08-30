import argparse
import json

import numpy as np

from scripts.build_libero_10_grasp_profiles import matrix_to_xyzw, rotation_xyzw
from scripts.locate_libero_plus_failure_onsets_v2 import locate
from scripts.route_libero_plus_teacher_skills import (
    choose_predicate,
    failed_release_fork,
    pre_irreversible_tilt_fork,
)


PREDICATE = ["in", "target_1", "basket_1_contain_region"]


def row(*, goal=False, target_z=0.4, eef_z=0.45, gripper=-1.0):
    return {
        "executed_action": [0.0] * 6 + [gripper],
        "online_observation": {
            "proprio": {"robot0_eef_pos": [0.0, 0.0, eef_z]},
            "dense_bundle": "/tmp/not-used.npz",
        },
        "privileged_audit": {
            "entities": {"target_1": {"position": [0.0, 0.0, target_z]}},
            "goal_predicates": [{"predicate": PREDICATE, "value": goal}],
        },
    }


def test_goal_regression_teacher_fork_is_after_regression(tmp_path):
    rows = [row(goal=False), row(goal=True), row(goal=True), row(goal=False)]
    steps = tmp_path / "steps.jsonl"
    steps.write_text("".join(json.dumps(item) + "\n" for item in rows))
    result = locate(
        {"steps": str(steps), "failure_type": "goal_regression"},
        argparse.Namespace(
            pre_event_context_steps=2,
            confirmation_steps=2,
            partial_recovery_budget=10,
        ),
    )
    assert result["event_timestep"] == 3
    assert result["fork_timestep"] == 3
    assert result["teacher_fork_timestep"] == 3
    assert result["gate_context_timestep"] == 1
    assert result["onset_strategy"] == "post_goal_regression"


def test_failed_release_fork_starts_while_target_is_still_held():
    rows = [
        row(),
        row(target_z=0.43, eef_z=0.46, gripper=1.0),
        row(target_z=0.47, eef_z=0.50, gripper=1.0),
        row(target_z=0.48, eef_z=0.51, gripper=1.0),
        row(target_z=0.48, eef_z=0.51, gripper=-1.0),
        row(target_z=0.46, eef_z=0.52, gripper=-1.0),
        row(target_z=0.42, eef_z=0.54, gripper=-1.0),
        row(target_z=0.40, eef_z=0.56, gripper=-1.0),
    ]
    assert failed_release_fork(rows, PREDICATE, pre_context=2) == 2


def test_transient_gripper_flicker_is_not_a_release():
    rows = [
        row(),
        row(target_z=0.45, eef_z=0.48, gripper=1.0),
        row(target_z=0.47, eef_z=0.50, gripper=-1.0),
        row(target_z=0.47, eef_z=0.50, gripper=1.0),
        row(target_z=0.47, eef_z=0.50, gripper=1.0),
        row(target_z=0.47, eef_z=0.50, gripper=-1.0),
        row(target_z=0.44, eef_z=0.52, gripper=-1.0),
        row(target_z=0.41, eef_z=0.55, gripper=-1.0),
        row(target_z=0.40, eef_z=0.56, gripper=-1.0),
    ]
    assert failed_release_fork(rows, PREDICATE, pre_context=1) == 4


def test_successful_release_is_not_mined_as_failed_placement():
    rows = [
        row(),
        row(target_z=0.45, eef_z=0.48, gripper=1.0),
        row(target_z=0.47, eef_z=0.50, gripper=1.0),
        row(goal=True, target_z=0.47, eef_z=0.50, gripper=-1.0),
        row(goal=True, target_z=0.43, eef_z=0.52, gripper=-1.0),
        row(goal=True, target_z=0.40, eef_z=0.55, gripper=-1.0),
        row(goal=True, target_z=0.40, eef_z=0.56, gripper=-1.0),
    ]
    assert failed_release_fork(rows, PREDICATE, pre_context=1) is None


def test_partial_goal_selection_excludes_goal_completed_before_fork():
    other = ["in", "other_1", "basket_1_contain_region"]
    rows = [row(), row(), row()]
    for index, item in enumerate(rows):
        item["privileged_audit"]["entities"]["other_1"] = {
            "position": [0.2, 0.0, 0.4]
        }
        item["privileged_audit"]["goal_predicates"].append({
            "predicate": other,
            "value": index >= 1,
        })
    selected, _ = choose_predicate(
        {"failure_type": "partial_goal_completion", "fork_timestep": 2},
        rows,
        event=0,
        context=2,
    )
    assert selected == PREDICATE


def test_upright_target_tilt_is_forked_before_it_falls():
    rows = [row() for _ in range(10)]
    # Identity starts with local Z upright.  Four persistent 90-degree rotations
    # about X begin at step 6; recovery receives two clean context steps.
    for index, item in enumerate(rows):
        item["privileged_audit"]["entities"]["target_1"]["quaternion_wxyz"] = (
            [1.0, 0.0, 0.0, 0.0]
            if index < 6 else [0.70710678, 0.70710678, 0.0, 0.0]
        )
    assert pre_irreversible_tilt_fork(rows, PREDICATE, pre_context=2) == 4


def test_initially_slanted_object_is_not_treated_as_new_fall():
    rows = [row() for _ in range(8)]
    for item in rows:
        item["privileged_audit"]["entities"]["target_1"]["quaternion_wxyz"] = [
            0.8660254, 0.3535534, 0.3535534, 0.0,
        ]
    assert pre_irreversible_tilt_fork(rows, PREDICATE, pre_context=2) is None


def test_success_grasp_profile_quaternion_conversion_round_trips():
    angle = np.pi / 3
    quaternion = np.asarray([np.sin(angle / 2), 0.0, 0.0, np.cos(angle / 2)])
    rotation = rotation_xyzw(quaternion)
    recovered = matrix_to_xyzw(rotation)
    assert np.allclose(rotation_xyzw(recovered), rotation, atol=1e-7)
