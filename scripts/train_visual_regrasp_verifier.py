#!/usr/bin/env python
"""Train a non-privileged temporal verifier for successful re-grasp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-traces", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-traces", type=Path, nargs="+", required=True)
    parser.add_argument("--test-traces", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positive-lift-m", type=float, default=0.05)
    parser.add_argument("--probability-threshold", type=float, default=0.6)
    parser.add_argument("--confirm-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


class VisualRegraspVerifier(nn.Module):
    def __init__(self, input_dim=11):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 12), nn.GELU(),
            nn.Linear(12, 1),
        )

    def forward(self, inputs):
        return self.network(inputs).squeeze(-1)


def feature(row, target_start, eef_start, elapsed):
    target = np.asarray(row["visual_target_xyz"], np.float32)
    robot = row["robot_state_before"]
    eef = np.asarray(robot["robot0_eef_pos"], np.float32)
    target_delta = target - target_start
    eef_delta = eef - eef_start
    gripper = np.asarray(robot["robot0_gripper_qpos"], np.float32)
    gripper_velocity = np.asarray(robot["robot0_gripper_qvel"], np.float32)
    relative_drift = np.linalg.norm(
        (target - eef) - (target_start - eef_start)
    )
    xy_drift = np.linalg.norm(
        ((target - eef) - (target_start - eef_start))[:2]
    )
    geometry = np.asarray([
        target_delta[2],
        eef_delta[2],
        relative_drift,
        xy_drift,
        abs(target_delta[2] - eef_delta[2]),
        np.linalg.norm(target_delta),
        min(elapsed, 50) / 50.0,
    ], np.float32)
    return np.concatenate((geometry, gripper, gripper_velocity))


def load_trace(path, positive_lift_m):
    all_rows = [
        json.loads(line) for line in path.read_text().splitlines() if line
    ]
    rows = [
        row for row in all_rows
        if row.get("phase") == "VERIFY LIFT"
        and row.get("visual_target_xyz") is not None
    ]
    if not rows:
        raise RuntimeError(f"no visual VERIFY frames in {path}")
    trace_recovered = any(
        row.get("recovery_completed", False) for row in all_rows
    )
    target_start = np.asarray(rows[0]["visual_target_xyz"], np.float32)
    eef_start = np.asarray(
        rows[0]["robot_state_before"]["robot0_eef_pos"], np.float32
    )
    object_initial = float(
        all_rows[0]["privileged_object_state_before"]["akita_black_bowl_1_pos"][2]
    )
    features, labels = [], []
    for elapsed, row in enumerate(rows):
        object_z = float(
            row["privileged_object_state_before"]["akita_black_bowl_1_pos"][2]
        )
        features.append(feature(row, target_start, eef_start, elapsed))
        labels.append(
            trace_recovered and object_z - object_initial >= positive_lift_m
        )
    return {
        "path": str(path),
        "trace_recovered": trace_recovered,
        "features": torch.from_numpy(np.stack(features)),
        "labels": torch.tensor(labels, dtype=torch.float32),
    }


@torch.inference_mode()
def probabilities(model, trace, mean, std):
    return torch.sigmoid(model((trace["features"] - mean) / std))


@torch.inference_mode()
def evaluate(model, traces, mean, std, threshold, confirm_steps):
    reports = []
    total_correct = total_frames = false_positives = false_negatives = 0
    for trace in traces:
        scores = probabilities(model, trace, mean, std)
        above = scores >= threshold
        predictions = torch.zeros_like(above)
        streak = 0
        for index, value in enumerate(above):
            streak = streak + 1 if bool(value) else 0
            predictions[index] = streak >= confirm_steps
        labels = trace["labels"].bool()
        false_positives += int((predictions & ~labels).sum())
        false_negatives += int((~predictions & labels).sum())
        total_correct += int((predictions == labels).sum())
        total_frames += len(labels)
        first_prediction = torch.where(predictions)[0]
        first_label = torch.where(labels)[0]
        predicted_step = (
            int(first_prediction[0]) if len(first_prediction) else None
        )
        label_step = int(first_label[0]) if len(first_label) else None
        reports.append({
            "trace": trace["path"],
            "num_verify_frames": len(labels),
            "num_positive_labels": int(labels.sum()),
            "first_positive_label": label_step,
            "first_positive_prediction": predicted_step,
            "transition_delay_steps": (
                predicted_step - label_step
                if predicted_step is not None and label_step is not None else None
            ),
            "max_probability": float(scores.max()),
            "false_positives": int((predictions & ~labels).sum()),
            "false_negatives": int((~predictions & labels).sum()),
        })
    return {
        "num_frames": total_frames,
        "accuracy": total_correct / total_frames,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "traces": reports,
    }


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    traces = {
        "train": [
            load_trace(path, args.positive_lift_m) for path in args.train_traces
        ],
        "validation": [
            load_trace(path, args.positive_lift_m)
            for path in args.validation_traces
        ],
        "test": [
            load_trace(path, args.positive_lift_m) for path in args.test_traces
        ],
    }
    train_features = torch.cat([
        trace["features"] for trace in traces["train"]
    ])
    train_labels = torch.cat([trace["labels"] for trace in traces["train"]])
    mean = train_features.mean(0)
    std = train_features.std(0).clamp_min(0.01)
    model = VisualRegraspVerifier(train_features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    positive_weight = torch.tensor(
        [(len(train_labels) - train_labels.sum()) / train_labels.sum().clamp_min(1)]
    )
    generator = torch.Generator().manual_seed(args.seed)
    best_state = None
    best_score = (-10**9, -1.0)
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            len(train_features), (32,), generator=generator
        )
        logits = model((train_features[indices] - mean) / std)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, train_labels[indices], pos_weight=positive_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 20 == 0 or step == args.steps:
            validation = evaluate(
                model, traces["validation"], mean, std,
                args.probability_threshold, args.confirm_steps,
            )
            score = (
                -validation["false_positives"],
                validation["accuracy"],
            )
            if score > best_score:
                best_score = score
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
    model.load_state_dict(best_state)
    model.eval()
    threshold = args.probability_threshold
    metrics = {
        split: evaluate(
            model, values, mean, std, threshold, args.confirm_steps
        )
        for split, values in traces.items()
    }
    passed = all(
        metrics[split]["false_positives"] == 0
        and all(
            report["first_positive_label"] is None
            or (
                report["first_positive_prediction"] is not None
                and abs(report["transition_delay_steps"]) <= 3
            )
            for report in metrics[split]["traces"]
        )
        for split in ("validation", "test")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "visual-regrasp-verifier.safetensors"
    save_file({
        **{
            f"model.{key}": value
            for key, value in model.state_dict().items()
        },
        "feature_mean": mean,
        "feature_std": std,
        "probability_threshold": torch.tensor([threshold]),
        "confirm_steps": torch.tensor([args.confirm_steps]),
    }, checkpoint)
    result = {
        "schema_version": 1,
        "kind": "non_privileged_temporal_visual_regrasp_verifier",
        "input_contract": (
            "VERIFY-relative visual target/eef displacement and relation + "
            "gripper qpos/qvel + elapsed"
        ),
        "teacher_label_only": (
            "trace eventually recovery_completed and privileged object lift "
            f"from episode initial >= {args.positive_lift_m} m"
        ),
        "probability_threshold": threshold,
        "confirm_steps": args.confirm_steps,
        "metrics": metrics,
        "offline_gate_passed": passed,
        "checkpoint": str(checkpoint),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
