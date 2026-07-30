#!/usr/bin/env python
"""Regress target-to-EEF xyz from observable role slots and proprioception."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--train-episodes",
        default="0,1,2,5,8,9,20,21,22,23,24,25,26,27,28,29",
    )
    parser.add_argument("--validation-episodes", default="4,6")
    parser.add_argument("--test-episodes", default="3,7")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def parse_ids(value):
    return [int(item) for item in value.split(",") if item]


class SlotRelativeGeometry(nn.Module):
    """Decode metric target-relative geometry without privileged runtime input."""

    def __init__(self, hidden_dim=192):
        super().__init__()
        # All role slots retain relational scene context; explicit mask
        # geometry and EEF xyz/aperture make the fixed-camera metric mapping
        # easier to learn.
        input_dim = 3 * 96 + 3 * 3 + 4
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, role_slots, role_geometry, robot_features):
        features = torch.cat((
            role_slots.flatten(-2),
            role_geometry.flatten(-2),
            robot_features,
        ), dim=-1)
        return self.network(features)


def load_targets(path):
    targets = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (int(row["episode_id"]), int(row["timestep"]))
        targets[key] = row["pudding_relative_to_eef_xyz_m"]
    return targets


def load_split(root, episode_ids, targets):
    inputs = []
    labels = []
    episode_index = []
    for episode in episode_ids:
        tensors = load_file(
            root / f"task6-episode{episode}-execution-roles.safetensors"
        )
        timesteps = tensors["timesteps"].tolist()
        inputs.append((
            tensors["role_slots"],
            tensors["role_geometry"],
            tensors["robot_features"],
        ))
        labels.append(torch.tensor(
            [targets[(episode, int(t))] for t in timesteps],
            dtype=torch.float32,
        ))
        episode_index.extend([episode] * len(timesteps))
    return (
        tuple(torch.cat([item[column] for item in inputs]) for column in range(3)),
        torch.cat(labels),
        torch.tensor(episode_index),
    )


@torch.inference_mode()
def evaluate(model, split, device):
    inputs, targets, episodes = split
    prediction = model(*(tensor.to(device) for tensor in inputs)).cpu()
    error = torch.linalg.vector_norm(prediction - targets, dim=-1)
    metrics = {
        "mean_error_m": float(error.mean()),
        "median_error_m": float(error.median()),
        "p90_error_m": float(error.quantile(.9)),
        "axis_mae_m": (prediction - targets).abs().mean(0).tolist(),
        "episodes": {},
    }
    for episode in episodes.unique().tolist():
        selected = error[episodes == episode]
        metrics["episodes"][str(episode)] = {
            "frames": int(len(selected)),
            "mean_error_m": float(selected.mean()),
            "p90_error_m": float(selected.quantile(.9)),
        }
    return metrics


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_ids = {
        "train": parse_ids(args.train_episodes),
        "validation": parse_ids(args.validation_episodes),
        "test": parse_ids(args.test_episodes),
    }
    targets = load_targets(args.labels)
    splits = {
        name: load_split(args.sequence_dir, ids, targets)
        for name, ids in split_ids.items()
    }
    model = SlotRelativeGeometry().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    train_inputs, train_targets, _ = splits["train"]
    train_inputs = tuple(tensor.to(device) for tensor in train_inputs)
    train_targets = train_targets.to(device)
    best_error = float("inf")
    best_state = None
    best_epoch = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_targets), device=device)
        losses = []
        for indices in permutation.split(512):
            prediction = model(*(tensor[indices] for tensor in train_inputs))
            loss = nn.functional.smooth_l1_loss(
                prediction, train_targets[indices], beta=.01
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        validation = evaluate(model.eval(), splits["validation"], device)
        if validation["mean_error_m"] < best_error:
            best_error = validation["mean_error_m"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"epoch={epoch} loss={np.mean(losses):.5f} "
                f"val_error_m={validation['mean_error_m']:.5f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    metrics = {
        "schema_version": 1,
        "kind": "observable_slot_relative_geometry",
        "runtime_inputs": ["role_slots", "role_geometry", "robot_features"],
        "privileged_runtime_input": False,
        "target": "pudding_xyz_minus_eef_xyz",
        "episode_split": split_ids,
        "best_epoch": best_epoch,
        "metrics": {
            name: evaluate(model.eval(), split, device)
            for name, split in splits.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        model.state_dict(),
        args.output_dir / "task6-slot-relative-geometry.safetensors",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
