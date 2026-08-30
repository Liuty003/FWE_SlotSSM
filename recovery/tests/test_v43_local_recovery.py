import torch
from types import MethodType

from recovery.models.v4_relational_recovery import V4ModelConfig
from recovery.models.v43_local_recovery import (
    RecoveryLatch, RecoveryLatchConfig, V43LocalRecovery,
)
from recovery.models.v44_dual_ownership_recovery import V44DualOwnershipRecovery
from scripts.train_v43_local_recovery import adapt_short_horizon_state


def config():
    return V4ModelConfig(
        visual_dim=12, action_hidden_dim=16, action_chunk=2, action_dim=3,
        num_slots=4, slot_dim=8, relation_dim=8, num_heads=2,
    )


def inputs(rows=3):
    cfg = config()
    return {
        "visual_tokens": torch.randn(rows, 6, cfg.visual_dim),
        "action_hidden": torch.randn(rows, cfg.action_chunk * cfg.action_dim, cfg.action_hidden_dim),
        "token_points": torch.randn(rows, 6, 3),
        "token_valid": torch.ones(rows, 6, dtype=torch.bool),
        "token_image_xy": torch.randn(rows, 6, 2),
        "gripper_position": torch.randn(rows, 3),
        "gripper_image_xy": torch.randn(rows, 2),
    }


def test_direct_recovery_head_shape_loss_and_exact_normal_route():
    model = V43LocalRecovery(config())
    output = model(**inputs())
    assert output["recovery_actions"].shape == (3, 2, 3)
    targets = torch.randn(3, 2, 3)
    mask = torch.tensor([False, True, False])
    loss = model.recovery_loss(output, targets, mask)
    loss.backward()
    assert model.direct_action[-1].weight.grad is not None

    nominal = torch.randn(3, 2, 3)
    routed = model.route_actions(nominal, output["recovery_actions"].detach(), mask)
    assert torch.equal(routed[~mask], nominal[~mask])


def test_v2_direct_head_preserves_action_query_when_relational_context_is_fixed():
    def fixed_context(self, action_hidden, slots, masks, token_points, token_image_xy,
                      token_valid, gripper_position, gripper_image_xy):
        batch, action_tokens = action_hidden.shape[:2]
        context = torch.zeros(batch, action_tokens, self.config.relation_dim)
        relations = torch.zeros(batch, 1, self.config.relation_dim)
        positions = torch.zeros(batch, self.config.num_slots, 3)
        return context, relations, positions

    payload = inputs(rows=1)
    changed = {key: value.clone() for key, value in payload.items()}
    changed["action_hidden"] = changed["action_hidden"] + torch.randn_like(changed["action_hidden"])

    model = V43LocalRecovery(config(), direct_head_version="context_action_query_v2")
    model._context = MethodType(fixed_context, model)
    first = model(**payload)["recovery_actions"]
    second = model(**changed)["recovery_actions"]
    assert not torch.allclose(first, second)

    legacy = V43LocalRecovery(config(), direct_head_version="context_only_v1")
    legacy._context = MethodType(fixed_context, legacy)
    legacy_first = legacy(**payload)["recovery_actions"]
    legacy_second = legacy(**changed)["recovery_actions"]
    assert torch.equal(legacy_first, legacy_second)


def test_v3_chunk_head_has_explicit_action_coordinates_and_weighted_loss():
    model = V43LocalRecovery(config(), direct_head_version="pooled_chunk_mlp_v3")
    output = model(**inputs())
    assert output["recovery_actions"].shape == (3, 2, 3)
    assert model.direct_action[-1].out_features == 6

    predicted = torch.zeros(1, 2, 3)
    targets = torch.ones_like(predicted)
    mask = torch.tensor([True])
    unweighted = model.recovery_loss({"recovery_actions": predicted}, targets, mask)
    weighted_targets = targets.clone()
    weighted_targets[..., :2] = 0
    weighted = model.recovery_loss(
        {"recovery_actions": predicted}, weighted_targets, mask,
        action_dimension_weights=torch.tensor([1.0, 1.0, 4.0]),
    )
    assert torch.isclose(unweighted, torch.tensor(1.0))
    assert torch.isclose(weighted, torch.tensor(2.0 / 3.0))


def test_v44_single_step_head_keeps_full_oft_context_and_slices_first_target():
    cfg = config()
    model = V43LocalRecovery(
        cfg, direct_head_version="pooled_chunk_mlp_v3", recovery_output_horizon=1,
    )
    output = model(**inputs())
    assert output["relational_context"].shape[1] == cfg.action_chunk * cfg.action_dim
    assert output["recovery_actions"].shape == (3, 1, cfg.action_dim)
    assert model.direct_action[-1].out_features == cfg.action_dim

    targets = torch.randn(3, cfg.action_chunk, cfg.action_dim)
    mask = torch.tensor([True, False, True])
    expected = torch.nn.functional.l1_loss(
        output["recovery_actions"][mask], targets[mask, :1],
    )
    actual = model.recovery_loss(output, targets, mask)
    assert torch.allclose(actual, expected)


def test_v44_short_head_initialization_exactly_inherits_v43_first_action():
    cfg = config()
    old = V43LocalRecovery(cfg, direct_head_version="pooled_chunk_mlp_v3")
    short = V43LocalRecovery(
        cfg, direct_head_version="pooled_chunk_mlp_v3", recovery_output_horizon=1,
    )
    state = adapt_short_horizon_state(
        old.state_dict(), short.state_dict(), cfg.action_dim,
    )
    result = short.load_state_dict(state, strict=False)
    assert not result.missing_keys and not result.unexpected_keys
    payload = inputs()
    assert torch.equal(short.direct_action[-1].weight, old.direct_action[-1].weight[:cfg.action_dim])
    assert torch.equal(short.direct_action[-1].bias, old.direct_action[-1].bias[:cfg.action_dim])
    torch.testing.assert_close(
        short(**payload)["recovery_actions"],
        old(**payload)["recovery_actions"][:, :1],
        rtol=1e-6, atol=1e-7,
    )


def test_short_horizon_route_uses_common_prefix_and_exact_nominal_action():
    nominal = torch.randn(2, 8, 7)
    recovery = torch.randn(2, 1, 7)
    mask = torch.tensor([False, True])
    routed = V43LocalRecovery.route_actions(nominal, recovery, mask)
    assert routed.shape == (2, 1, 7)
    assert torch.equal(routed[0], nominal[0, :1])
    assert torch.equal(routed[1], recovery[1])


def test_v44_short_horizon_rejects_tokenwise_legacy_head():
    try:
        V43LocalRecovery(
            config(), direct_head_version="context_action_query_v2",
            recovery_output_horizon=1,
        )
    except ValueError as error:
        assert "short recovery horizons" in str(error)
    else:
        raise AssertionError("short outputs require a pooled action head")


def test_v44_dual_ownership_heads_are_independent_and_share_context():
    model = V44DualOwnershipRecovery(config())
    output = model(**inputs())
    assert output["gate_probability"].shape == (3,)
    assert output["resume_probability"].shape == (3,)
    gate_storage = {parameter.data_ptr() for parameter in model.gate.parameters()}
    resume_storage = {parameter.data_ptr() for parameter in model.resume_head.parameters()}
    assert not gate_storage & resume_storage


def test_latch_survives_gate_flicker_then_hands_back_and_cools_down():
    latch = RecoveryLatch(RecoveryLatchConfig(
        trigger_threshold=0.7, recovery_horizon_steps=10, minimum_recovery_steps=10,
        release_threshold=0.2, cooldown_steps=6,
    ))
    assert latch.select(0.8)
    assert latch.activation_count == 1
    assert latch.executable_steps(4) == 4
    latch.advance(4)
    assert latch.select(0.0)
    latch.advance(4)
    assert latch.executable_steps(4) == 2
    latch.advance(2)
    assert not latch.recovering
    assert not latch.select(1.0)
    latch.advance(4)
    assert not latch.select(1.0)
    latch.advance(2)
    assert latch.select(1.0)
    assert latch.activation_count == 2


def test_latch_rejects_chunk_that_crosses_handoff():
    latch = RecoveryLatch(RecoveryLatchConfig(recovery_horizon_steps=3, minimum_recovery_steps=3))
    assert latch.select(1.0)
    assert latch.executable_steps(8) == 3
    try:
        latch.advance(4)
    except ValueError as error:
        assert "handoff boundary" in str(error)
    else:
        raise AssertionError("executor must cap a recovery chunk at the handoff boundary")


def test_latch_searches_single_steps_and_releases_on_learned_boundary():
    latch = RecoveryLatch(RecoveryLatchConfig(
        recovery_horizon_steps=12, minimum_recovery_steps=8,
        trigger_threshold=0.7, release_threshold=0.3,
    ))
    assert latch.select(0.9)
    assert latch.executable_steps(4) == 4
    latch.advance(4)
    assert latch.select(0.9)
    latch.advance(4)
    assert latch.select(0.9)
    assert latch.executable_steps(4) == 1
    latch.advance(1)
    assert not latch.select(0.1)
    assert latch.handoff_count == 1
    assert not latch.recovering


def test_latch_requires_consecutive_trigger_confirmations():
    latch = RecoveryLatch(RecoveryLatchConfig(
        trigger_threshold=0.9, trigger_confirmation_decisions=2,
    ))
    assert not latch.select(0.95)
    assert not latch.select(0.2)
    assert not latch.select(0.95)
    assert latch.select(0.96)
    assert latch.activation_count == 1


def test_latch_can_force_handoff_after_verified_attachment():
    latch = RecoveryLatch(RecoveryLatchConfig(recovery_horizon_steps=100, cooldown_steps=8))
    assert latch.select(1.0)
    latch.advance(4)
    latch.force_handoff()
    assert not latch.recovering
    assert latch.handoff_count == 1
    assert latch.cooldown_steps_remaining == 8


def test_latch_requires_consecutive_resume_confirmations():
    latch = RecoveryLatch(RecoveryLatchConfig(
        trigger_threshold=0.7, recovery_horizon_steps=20, minimum_recovery_steps=2,
        release_threshold=0.8, release_confirmation_decisions=3,
    ))
    assert latch.select(0.9, completion_probability=0.0)
    latch.advance(2)
    assert latch.select(0.9, completion_probability=0.9)
    latch.advance(1)
    assert latch.select(0.9, completion_probability=0.7)
    latch.advance(1)
    assert latch.select(0.9, completion_probability=0.9)
    latch.advance(1)
    assert latch.select(0.9, completion_probability=0.9)
    latch.advance(1)
    assert not latch.select(0.9, completion_probability=0.9)
    assert latch.handoff_count == 1
