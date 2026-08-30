import pytest

from scripts.analyze_libero10_gate_by_family import (
    calibrated_sources,
    has_confirmation_run,
    source_metrics,
)
from scripts.audit_relational_oft_handoffs import resolve_normalization_key
from scripts.train_v44_resume_head import validate_resume_feature_contract
from scripts.run_v44_dual_batch import fingerprint as rollout_fingerprint


def test_gate_calibration_uses_complete_trajectory_and_confirmation_count():
    assert has_confirmation_run([0.9, 0.1, 0.9, 0.9], 0.8, 2)
    assert not has_confirmation_run([0.9, 0.1, 0.9], 0.8, 2)
    values = source_metrics([[0.8, 0.8]], [[0.9, 0.1, 0.9]], 0.75, 2)
    assert values["failure_source_hit_rate"] == 1.0
    assert values["normal_source_false_activation_rate"] == 0.0


def test_gate_calibration_constrains_normal_source_false_activation():
    result = calibrated_sources(
        positive_sequences=[[0.8, 0.8], [0.6, 0.6]],
        negative_sequences=[[0.7, 0.7], [0.1, 0.1]],
        target_fpr=0.0,
        minimum_threshold=0.5,
        confirmations=2,
    )
    assert result["target_satisfied"] is True
    assert result["threshold"] > 0.7
    assert result["normal_source_false_activation_rate"] == 0.0


def test_handoff_normalization_defaults_to_task_suite():
    stats = {"libero_10": {}, "libero_object": {}}
    assert resolve_normalization_key(stats, "libero_10", None) == "libero_10"
    assert resolve_normalization_key(stats, "libero_10", "libero_object") == "libero_object"
    with pytest.raises(KeyError, match="normalization key"):
        resolve_normalization_key({}, "libero_10", None)


def test_rollout_cache_fingerprint_changes_with_deployment_contract():
    original = {"adapter_sha256": "abc", "gate_threshold": 0.8, "task_id": 7}
    assert rollout_fingerprint(original) == rollout_fingerprint(dict(original))
    assert rollout_fingerprint(original) != rollout_fingerprint({**original, "gate_threshold": 0.81})


def resume_reports():
    common = {
        "handoff_manifest": "/audits/handoff.json",
        "fixed_recovery_window_steps": None,
        "recovery_sampling_stride": 1,
        "recovery_target_horizon": 1,
    }
    return (
        {**common, "include_handoff_gate_rows": False},
        {**common, "include_handoff_gate_rows": True},
    )


def test_resume_contract_requires_certified_cropped_features():
    recovery, boundary = resume_reports()
    validate_resume_feature_contract(recovery, boundary)
    with pytest.raises(ValueError, match="handoff manifest"):
        validate_resume_feature_contract({**recovery, "handoff_manifest": None}, boundary)
    with pytest.raises(ValueError, match="exclude"):
        validate_resume_feature_contract({**recovery, "include_handoff_gate_rows": True}, boundary)
    with pytest.raises(ValueError, match="single-step"):
        validate_resume_feature_contract({**recovery, "recovery_sampling_stride": 4}, boundary)
