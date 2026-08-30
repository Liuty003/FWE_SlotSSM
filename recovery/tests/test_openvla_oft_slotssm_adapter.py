import torch
from torch import nn

from recovery.models.openvla_oft_slotssm_adapter import (
    OpenVLAOFTSlotSSMAdapter,
    RecoveryPhase,
    RecoveryMode,
)


class TinyOFTActionHead(nn.Module):
    def __init__(self, hidden_dim=16, chunk=2, action_dim=3):
        super().__init__()
        self.chunk = chunk
        self.action_dim = action_dim
        self.projection = nn.Linear(action_dim * hidden_dim, action_dim)

    def predict_action(self, hidden):
        batch = hidden.shape[0]
        return self.projection(
            hidden.reshape(batch, self.chunk, self.action_dim * hidden.shape[-1])
        )


def make_model():
    return OpenVLAOFTSlotSSMAdapter(
        TinyOFTActionHead(),
        action_hidden_dim=16,
        slot_dim=8,
        fusion_dim=16,
        num_heads=4,
        gate_hidden_dim=12,
        action_chunk=2,
        action_output_dim=3,
    )


def test_initialization_preserves_normal_actions_and_routes_normal():
    torch.manual_seed(0)
    model = make_model()
    action_hidden = torch.randn(2, 6, 16)
    slots = torch.randn(2, 4, 8)
    output = model(action_hidden, slots)
    assert output["normal_actions"].shape == (2, 2, 3)
    assert output["target_relative_position"].shape == (2, 3)
    assert torch.equal(output["normal_actions"], output["recovery_actions"])
    assert output["gate_logits"].argmax(dim=-1).eq(RecoveryMode.NORMAL).all()
    routed = model.route_actions(
        output["normal_actions"],
        output["recovery_actions"] + 1,
        output["gate_logits"].argmax(dim=-1),
    )
    assert torch.equal(routed, output["normal_actions"])


def test_normal_head_is_frozen_but_recovery_path_is_trainable():
    model = make_model()
    assert all(not parameter.requires_grad for parameter in model.normal_action_head.parameters())
    assert all(parameter.requires_grad for parameter in model.recovery_action_head.parameters())
    action_hidden = torch.randn(2, 6, 16)
    slots = torch.randn(2, 4, 8)
    output = model(action_hidden, slots)
    loss = output["recovery_actions"].square().mean() + output["gate_logits"].square().mean()
    loss.backward()
    assert model.recovery_action_head.projection.weight.grad is not None
    assert model.slot_fusion.output_projection.weight.grad is not None
    assert model.normal_action_head.projection.weight.grad is None


def test_hard_router_uses_only_recovery_state():
    normal = torch.zeros(3, 2, 3)
    recovery = torch.ones_like(normal)
    mode = torch.tensor(
        [RecoveryMode.NORMAL, RecoveryMode.RECOVERY, RecoveryMode.RECOVERY_COMPLETE]
    )
    routed = OpenVLAOFTSlotSSMAdapter.route_actions(normal, recovery, mode)
    assert torch.equal(routed[0], normal[0])
    assert torch.equal(routed[1], recovery[1])
    assert torch.equal(routed[2], normal[2])


def test_zero_initialized_robot_context_preserves_gate_and_is_trainable():
    model = make_model()
    action_hidden = torch.randn(2, 6, 16)
    slots = torch.randn(2, 4, 8)
    baseline = model(action_hidden, slots)
    contextual = model(
        action_hidden,
        slots,
        proprio=torch.randn(2, 8),
        previous_action=torch.randn(2, 7),
    )
    assert torch.equal(baseline["gate_logits"], contextual["gate_logits"])
    assert torch.equal(baseline["recovery_actions"], contextual["recovery_actions"])
    (contextual["gate_logits"].square().mean() + contextual["recovery_actions"].square().mean()).backward()
    assert model.gate_robot_projection[1].weight.grad is not None
    assert model.recovery_robot_action_projection[1].weight.grad is not None


def test_phase_override_selects_a_phase_embedding_without_changing_phase_logits():
    torch.manual_seed(0)
    model = make_model()
    with torch.no_grad():
        model.phase_embedding.weight[RecoveryPhase.BACKOFF].fill_(1.0)
        model.phase_embedding.weight[RecoveryPhase.CLOSE].fill_(-1.0)
    action_hidden = torch.randn(2, 6, 16)
    slots = torch.randn(2, 4, 8)
    backoff = model(action_hidden, slots, recovery_phase_override=torch.tensor([0, 0]))
    close = model(action_hidden, slots, recovery_phase_override=torch.tensor([3, 3]))
    assert torch.equal(backoff["recovery_phase_logits"], close["recovery_phase_logits"])
    assert not torch.equal(backoff["recovery_actions"], close["recovery_actions"])
