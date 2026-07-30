import numpy as np
import torch

from scripts.train_task6_slot_temporal_state import (
    SlotTemporalClassifier,
    calibrate_threshold,
)


def test_slot_temporal_classifier_is_shape_preserving():
    model = SlotTemporalClassifier(model_dim=32)
    assert model(torch.randn(2, 7, 16, 512)).shape == (2, 7)


def test_threshold_calibration_uses_episode_triggers():
    predictions = [
        {
            "episode_id": 0,
            "labels": torch.tensor([0, 0, 1, 1]),
            "scores": np.asarray([0.1, 0.2, 0.8, 0.9]),
        },
        {
            "episode_id": 1,
            "labels": torch.zeros(4),
            "scores": np.asarray([0.1, 0.2, 0.3, 0.4]),
        },
    ]
    threshold, report = calibrate_threshold(predictions, confirm_frames=2)
    assert 0.4 < threshold <= 0.8
    assert not any(item["false_trigger"] for item in report["episodes"])
    assert not any(item["missed_trigger"] for item in report["episodes"])
