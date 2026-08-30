import numpy as np
import pytest

from recovery.data.v4_geometry import (
    metric_depth, patch_instance_labels, patch_median_points, robot_point_to_saved_image_xy, saved_depth_to_robot_points,
)
from scripts.precompute_v4_oft_relation_features import (
    decision_starts,
    decoder_target_action,
    executed_to_planned_action,
    normalized_action_chunk,
    normalized_executed_action_chunk,
    resolve_handoff,
    source_id,
    source_instance_id,
)


def test_recovery_decision_grid_restarts_at_non_aligned_fork():
    starts = decision_starts(row_count=233, decision_stride=8, fork_timestep=49)
    assert starts[:7] == [0, 8, 16, 24, 32, 40, 48]
    assert starts[7:10] == [49, 57, 65]
    assert starts[-1] == 225


def test_nominal_decision_grid_remains_stride_aligned():
    assert decision_starts(row_count=23, decision_stride=8, fork_timestep=None) == [0, 8]


def test_nominal_feature_source_does_not_require_a_handoff_record():
    assert resolve_handoff(None, use_fixed_latch=False) is None
    assert resolve_handoff(None, use_fixed_latch=True) is None


def test_v43_local_grid_reobserves_every_four_and_never_crosses_handoff():
    assert decision_starts(
        row_count=240, decision_stride=8, fork_timestep=49,
        recovery_sampling_stride=4, recovery_handoff=94, recovery_only=True,
    ) == list(range(49, 87, 4))


def test_v44_grid_reobserves_every_step_and_single_target_reaches_handoff_tail():
    assert decision_starts(
        row_count=240, decision_stride=8, fork_timestep=49,
        recovery_sampling_stride=1, recovery_target_horizon=1,
        recovery_handoff=94, recovery_only=True,
    ) == list(range(49, 94))


def test_v43_gate_grid_keeps_tail_states_and_exact_variable_handoff():
    assert decision_starts(
        row_count=240, decision_stride=8, fork_timestep=44,
        recovery_sampling_stride=4, recovery_handoff=93, recovery_only=True,
        include_handoff_gate_rows=True,
    ) == [*range(44, 93, 4), 93]
from scripts.merge_v4_relation_features import merged_source_records


def test_metric_depth_and_saved_orientation_are_explicit():
    depth = np.full((2, 4), 0.5, dtype=np.float64)
    points, valid = saved_depth_to_robot_points(depth, np.eye(4), near=1.0, far=3.0)
    # z = 1 / (1 - .5 * (1 - 1/3)) = 1.5. Saved x=0 maps to native x=3.
    assert np.all(valid)
    assert np.isclose(points[0, 0, 0], 4.5)
    assert np.isclose(points[0, 0, 2], 1.5)
    assert np.allclose(metric_depth(depth, 1.0, 3.0), 1.5)


def test_patch_pooling_uses_only_valid_points_and_reports_saved_xy():
    points = np.zeros((4, 4, 3), dtype=np.float32)
    points[:2, :2] = np.array((2.0, 4.0, 6.0), dtype=np.float32)
    points[0, 0] = np.array((100.0, 100.0, 100.0), dtype=np.float32)
    valid = np.zeros((4, 4), dtype=bool)
    valid[:2, :2] = True
    valid[0, 0] = False
    pooled, pooled_valid, xy = patch_median_points(points, valid, grid_height=2, grid_width=2)
    assert pooled_valid.tolist() == [True, False, False, False]
    assert np.allclose(pooled[0], (2.0, 4.0, 6.0))
    assert np.allclose(xy[0], (0.25, 0.25))


def test_robot_point_projection_uses_the_same_saved_orientation():
    # Native pixel (3, 2) at z=2 becomes saved pixel (0, 2).
    xy, valid = robot_point_to_saved_image_xy(
        np.asarray((6.0, 4.0, 2.0)), np.eye(4), height=4, width=4,
    )
    assert valid
    assert np.allclose(xy, (0.0, 0.5))


def test_patch_instance_labels_is_training_only_majority_pooling():
    segmentation = np.zeros((4, 4), dtype=np.int64)
    segmentation[:2, :2] = np.asarray(((3, 3), (3, 9)))
    labels = patch_instance_labels(segmentation, grid_height=2, grid_width=2)
    assert labels.tolist() == [3, 0, 0, 0]
    assert np.all(patch_instance_labels(None, 2, 2) == -1)


def test_recovery_teacher_chunk_is_normalized_like_decoder_targets():
    stats = {"q01": np.zeros(7), "q99": np.full(7, 2.0), "mask": np.ones(7, dtype=bool)}
    physical = [[0.0] * 7, [2.0] * 7] * 4
    normalized = normalized_action_chunk(physical, stats, np)
    assert normalized.shape == (8, 7)
    assert np.all(normalized[::2] == -1.0)
    assert np.all(normalized[1::2] == 1.0)


def test_executed_gripper_is_inverted_back_to_planned_decoder_coordinates():
    opened = executed_to_planned_action([0, 0, 0, 0, 0, 0, -1], np)
    closed = executed_to_planned_action([0, 0, 0, 0, 0, 0, 1], np)
    assert opened[-1] == 1.0
    assert closed[-1] == 0.0
    stats = {"q01": np.zeros(7), "q99": np.ones(7), "mask": np.asarray([True] * 6 + [False])}
    chunk = [[0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 1]] * 4
    normalized = normalized_executed_action_chunk(chunk, stats, np)
    assert np.all(normalized[::2, -1] == 1.0)
    assert np.all(normalized[1::2, -1] == 0.0)


def test_single_step_recovery_target_normalization_is_independent_of_oft_chunk():
    stats = {"q01": np.zeros(7), "q99": np.ones(7), "mask": np.ones(7, dtype=bool)}
    normalized = normalized_executed_action_chunk(
        [[0, 0, 0, 0, 0, 0, -1]], stats, np, expected_horizon=1,
    )
    assert normalized.shape == (1, 7)


def test_stored_planned_action_is_preferred_over_postprocessed_action():
    row = {"planned_action": [0, 0, 0, 0, 0, 0, 0.37],
           "executed_action": [0, 0, 0, 0, 0, 0, -1]}
    assert decoder_target_action(row, np)[-1] == pytest.approx(0.37)


def test_multiple_forks_share_split_group_but_have_distinct_instance_ids(tmp_path):
    common = {
        "kind": "v4_book_prefix_canonical_teacher", "task_suite": "libero_10",
        "task_id": 504, "episode_id": 0, "source_rollout": "/rollouts/task504-episode0",
    }
    fork_a = {**common, "fork_timestep": 55, "recovery_branch": "/branches/fork55"}
    fork_b = {**common, "fork_timestep": 71, "recovery_branch": "/branches/fork71"}
    assert source_id(tmp_path / "a", fork_a) == source_id(tmp_path / "b", fork_b)
    assert source_instance_id(tmp_path / "a", fork_a) != source_instance_id(tmp_path / "b", fork_b)


def test_copied_canonical_fork_keeps_stable_instance_identity(tmp_path):
    metadata = {
        "kind": "v4_book_prefix_canonical_teacher", "task_suite": "libero_10",
        "task_id": 504, "episode_id": 0, "fork_timestep": 55,
        "source_rollout": "/rollouts/task504-episode0", "recovery_branch": "/branches/fork55",
    }
    assert source_instance_id(tmp_path / "copy-a", metadata) == source_instance_id(tmp_path / "copy-b", metadata)


def test_merge_allows_distinct_forks_in_one_split_group(tmp_path):
    reports = [{"sources": [{"source_instance_id": "fork-a"}]},
               {"sources": [{"source_instance_id": "fork-b"}]}]
    assert len(merged_source_records([tmp_path / "a", tmp_path / "b"], reports, overlap=True)) == 2
    reports[1]["sources"][0]["source_instance_id"] = "fork-a"
    with pytest.raises(ValueError, match="duplicate a concrete rollout"):
        merged_source_records([tmp_path / "a", tmp_path / "b"], reports, overlap=True)
