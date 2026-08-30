import torch

from recovery.data.v4_relation_features import V4RelationFeatures, from_tensors, load, save
from recovery.models.v4_relational_recovery import (
    V4ModelConfig, V4RelationalRecovery, _one_to_one_instance_mask_loss,
)


def tiny_config(representation="relation_3d"):
    return V4ModelConfig(
        visual_dim=12, action_hidden_dim=16, proprio_dim=5, action_chunk=2, action_dim=3,
        num_slots=4, slot_dim=8, relation_dim=8, num_heads=2, representation=representation,
    )


def payload(config, rows=3):
    torch.manual_seed(7)
    tokens = 6
    return {
        "visual_tokens": torch.randn(rows, tokens, config.visual_dim),
        "action_hidden": torch.randn(rows, config.action_chunk * config.action_dim, config.action_hidden_dim),
        "token_points": torch.randn(rows, tokens, 3),
        "token_valid": torch.tensor([[True, True, True, True, False, False]]).expand(rows, -1).clone(),
        "token_image_xy": torch.randn(rows, tokens, 2),
        "gripper_position": torch.randn(rows, 3),
        "gripper_image_xy": torch.randn(rows, 2),
        "actions": torch.randn(rows, config.action_chunk, config.action_dim),
        "gate_labels": torch.tensor([0, 1, -1])[:rows],
        "recovery_targets": torch.randn(rows, config.action_chunk, config.action_dim),
        "recovery_mask": torch.tensor([False, True, False])[:rows],
        "source_ids": torch.arange(rows),
        "instance_labels": torch.tensor([[0, 1, 1, 2, -1, -1]]).expand(rows, -1).clone(),
    }


def inputs(values):
    return {key: values[key] for key in (
        "visual_tokens", "action_hidden", "token_points", "token_valid", "token_image_xy",
        "gripper_position", "gripper_image_xy",
    )}


def test_v4_contract_roundtrip_and_rejects_v3_like_geometry_free_input(tmp_path):
    values = payload(tiny_config())
    features = from_tensors(values)
    assert isinstance(features, V4RelationFeatures)
    path = tmp_path / "v4.safetensors"
    save(features, path)
    assert load(path).num_rows == 3
    bad = dict(values)
    del bad["token_points"]
    try:
        from_tensors(bad)
    except KeyError as error:
        assert "token_points" in str(error)
    else:
        raise AssertionError("V4 must reject geometry-free V3-style features")


def test_v4_contract_allows_only_recovery_labels_to_have_action_targets():
    values = payload(tiny_config())
    values["recovery_mask"] = torch.tensor([True, False, False])
    try:
        from_tensors(values)
    except ValueError as error:
        assert "gate_labels == 1" in str(error)
    else:
        raise AssertionError("NORMAL examples must not contribute recovery action loss")


def test_v4_contract_decouples_native_oft_chunk_from_single_step_target():
    values = payload(tiny_config())
    values["recovery_targets"] = values["recovery_targets"][:, :1]
    features = from_tensors(values)
    assert features.actions.shape[1] == 2
    assert features.recovery_targets.shape[1] == 1


@torch.no_grad()
def test_v4_initial_adapter_is_exact_noop_and_hard_route_bypasses_nominal_hidden():
    values = payload(tiny_config())
    model = V4RelationalRecovery(tiny_config())
    output = model(**inputs(values))
    assert torch.equal(output["recovery_action_hidden"], values["action_hidden"])
    routed = model.hard_route(values["action_hidden"], output["recovery_action_hidden"], output["gate_probability"], 1.0)
    assert torch.equal(routed, values["action_hidden"])
    teacher_routed = model.teacher_route(
        values["action_hidden"], output["recovery_action_hidden"], values["gate_labels"],
    )
    # Recovery uses the residual only for y=1; NORMAL and ignored rows are
    # strict nominal bypasses during joint training.
    assert torch.equal(teacher_routed[0], values["action_hidden"][0])
    assert torch.equal(teacher_routed[2], values["action_hidden"][2])


def test_v4_all_representation_variants_shape_and_loss_contract():
    for representation in ("global", "slot", "relation_2d", "relation_3d"):
        config = tiny_config(representation)
        values = payload(config)
        model = V4RelationalRecovery(config)
        output = model(**inputs(values))
        assert output["relation_tokens"].shape[0] == 3
        assert output["relational_context"].shape[:2] == values["action_hidden"].shape[:2]
        assert output["recovery_action_hidden"].shape == values["action_hidden"].shape
        losses = model.losses(output, gate_labels=values["gate_labels"], instance_labels=values["instance_labels"], lambda_slot=0.1)
        assert all(torch.isfinite(value) for value in losses.values())
        losses["total"].backward()
        assert model.gate[-1].weight.grad is not None
        # V4.2 forbids an action / gripper bypass into detection: this gate is
        # parameterized solely over pooled C_t.
        assert model.gate[0].normalized_shape == (config.relation_dim,)


def test_slot_supervision_assigns_distinct_slots_to_distinct_instances():
    masks = torch.tensor([[[.45, .45, .05, .05], [.40, .40, .10, .10]]], requires_grad=True)
    labels = torch.tensor([[1, 1, 2, 2]])
    loss = _one_to_one_instance_mask_loss(masks, labels)
    expected = (-torch.log(torch.tensor(.45)) - torch.log(torch.tensor(.10))) / 2
    assert torch.allclose(loss, expected)
    loss.backward()
    assert masks.grad is not None
