from scripts.build_counterfactual_recovery_dataset import (
    dense_audit_steps,
    parse_tasks,
    phase_audit_steps,
    stable_handoff,
)


def test_parse_tasks_accepts_ranges_and_deduplicates():
    assert parse_tasks(["525-527", "527,530"]) == [525, 526, 527, 530]


def test_phase_audit_starts_at_close_and_includes_final_state():
    metadata = {
        "phase_by_step": ["approach"] * 3 + ["close"] * 4 + ["verify_lift"] * 5,
        "steps": 12,
    }
    assert phase_audit_steps(metadata, stride=4) == [3, 7, 11, 12]


def test_dense_scan_covers_boundary_and_stability_tail():
    assert dense_audit_steps({39: False, 47: False, 55: True, 63: True}, 70) == list(range(47, 58))


def test_stable_handoff_requires_consecutive_audited_successes():
    outcomes = {50: False, 51: True, 52: True, 53: True, 54: False}
    assert stable_handoff(outcomes) == 51
    assert stable_handoff({51: True, 52: True}) is None
