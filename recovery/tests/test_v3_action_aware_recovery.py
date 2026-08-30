import subprocess
import sys

import torch

from recovery.data.v3_sequence_features import V3SequenceFeatures, from_tensors, load, save
from recovery.models.v3_action_aware_recovery import V3ActionAwareRecoveryPolicy, V3ModelConfig


def tiny_config():
    return V3ModelConfig(
        visual_dim=12, action_hidden_dim=16, proprio_dim=5, action_dim=3,
        action_chunk=2, num_slots=4, slot_dim=8, hidden_dim=8, num_heads=2,
    )


def sequence(config, batch=2, steps=3):
    torch.manual_seed(11)
    return {
        "visual_tokens": torch.randn(batch, steps, 6, config.visual_dim),
        "action_hidden": torch.randn(batch, steps, 4, config.action_hidden_dim),
        "proprio": torch.randn(batch, steps, config.proprio_dim),
        "actions": torch.randn(batch, steps, config.action_chunk, config.action_dim),
        "gate_labels": torch.tensor([[0, 1, -1], [0, 0, 1]])[:, :steps],
        "recovery_targets": torch.randn(batch, steps, config.action_chunk, config.action_dim),
        "recovery_mask": torch.tensor([[False, True, False], [False, False, True]])[:, :steps],
        "source_ids": torch.tensor([10, 20]),
    }


def test_v3_sequence_forward_and_losses_are_finite():
    config = tiny_config()
    model = V3ActionAwareRecoveryPolicy(config)
    payload = sequence(config)
    output = model.forward_sequence(
        visual_tokens=payload["visual_tokens"], action_hidden=payload["action_hidden"],
        proprio=payload["proprio"], actions=payload["actions"],
    )
    assert output["slots"].shape == (2, 3, config.num_slots, config.slot_dim)
    assert output["future_slots"].shape == (2, 2, config.num_slots, config.slot_dim)
    assert output["reconstructed_visual_tokens"].shape == payload["visual_tokens"].shape
    assert output["recovery_actions"].shape == payload["actions"].shape
    losses = model.losses(output, visual_tokens=payload["visual_tokens"],
                           gate_labels=payload["gate_labels"],
                           recovery_targets=payload["recovery_targets"],
                           recovery_mask=payload["recovery_mask"])
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert model.slot_ssm.action_encoder[1].weight.grad is not None
    assert model.future_predictor.transition[-1].weight.grad is not None


def test_previous_action_changes_slot_memory_and_future_prediction():
    config = tiny_config()
    model = V3ActionAwareRecoveryPolicy(config)
    with torch.no_grad():
        # Remove the accidental possibility of a zero random action path.
        model.slot_ssm.action_encoder[1].weight.zero_()
        model.slot_ssm.action_encoder[1].weight[0, 0] = 1.0
        model.future_predictor.transition[-1].weight.fill_(0.1)
    visual = torch.randn(1, 6, config.visual_dim)
    hidden = torch.randn(1, 4, config.action_hidden_dim)
    proprio = torch.randn(1, config.proprio_dim)
    action_a = torch.zeros(1, config.action_chunk, config.action_dim)
    action_b = torch.arange(action_a.numel(), dtype=torch.float32).reshape_as(action_a)
    a = model.step(visual_tokens=visual, action_hidden=hidden, proprio=proprio, previous_action=action_a)
    b = model.step(visual_tokens=visual, action_hidden=hidden, proprio=proprio, previous_action=action_b)
    assert not torch.allclose(a["slots"], b["slots"])
    assert not torch.allclose(model.future_predictor(a["slots"], action_a), model.future_predictor(a["slots"], action_b))


def test_feature_contract_rejects_window_shaped_or_invalid_labels():
    config = tiny_config()
    payload = sequence(config)
    features = from_tensors(payload)
    assert isinstance(features, V3SequenceFeatures)
    bad = dict(payload)
    bad["gate_labels"] = torch.ones(2, 3, dtype=torch.int64) * 2
    try:
        from_tensors(bad)
    except ValueError as error:
        assert "gate_labels" in str(error)
    else:
        raise AssertionError("invalid gate labels must be rejected")


def test_feature_contract_roundtrip_retains_sequence_and_source_boundary(tmp_path):
    config = tiny_config()
    features = from_tensors(sequence(config))
    path = tmp_path / "v3.safetensors"
    save(features, path)
    restored = load(path)
    assert restored.sequence_length == 3
    assert torch.equal(restored.source_ids, torch.tensor([10, 20]))


def test_stage1_trainer_writes_deployment_checkpoint(tmp_path):
    config = tiny_config()
    features = from_tensors(sequence(config))
    train_path, val_path, output = tmp_path / "train.safetensors", tmp_path / "val.safetensors", tmp_path / "output"
    save(features.index(torch.tensor([0])), train_path)
    save(features.index(torch.tensor([1])), val_path)
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    command = [
        sys.executable, str(root / "scripts" / "train_v3_action_aware_recovery.py"),
        "--stage", "stage1", "--train-features", str(train_path), "--val-features", str(val_path),
        "--output-dir", str(output), "--steps", "1", "--batch-size", "1", "--eval-every", "1",
        "--save-every", "1", "--device", "cpu", "--num-slots", "4", "--slot-dim", "8",
        "--hidden-dim", "8", "--num-heads", "2",
    ]
    subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    assert (output / "train_best.safetensors").is_file()
    assert (output / "deploy_best.safetensors").is_file()
    stage2 = tmp_path / "stage2"
    stage2_command = command.copy()
    stage2_command[stage2_command.index("stage1")] = "stage2"
    stage2_command[stage2_command.index(str(output))] = str(stage2)
    stage2_command.extend(["--init-checkpoint", str(output / "train_best.safetensors")])
    subprocess.run(stage2_command, cwd=root, check=True, capture_output=True, text=True)
    assert (stage2 / "deploy_best.safetensors").is_file()
