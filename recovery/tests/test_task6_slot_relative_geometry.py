import torch

from scripts.train_task6_slot_relative_geometry import SlotRelativeGeometry


def test_slot_relative_geometry_has_observable_metric_output():
    model = SlotRelativeGeometry(hidden_dim=32)
    prediction = model(
        torch.randn(2, 3, 96),
        torch.randn(2, 3, 3),
        torch.randn(2, 4),
    )
    assert prediction.shape == (2, 3)
    prediction.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
