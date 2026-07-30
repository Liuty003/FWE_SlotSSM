#!/usr/bin/env python
"""Train matched macro-candidate values from frozen execution-Mamba state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.train_task6_execution_mamba import ExecutionMamba  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--execution-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--validation-json",
        type=Path,
        nargs="+",
        help="Explicit held-out candidate groups, including same-episode states.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-episode", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--robot-features", action="store_true")
    return parser.parse_args()


class MacroCandidateScorer(nn.Module):
    def __init__(self, context_dim=128, chunk_steps=28):
        super().__init__()
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, 128), nn.GELU()
        )
        self.candidate = nn.Sequential(
            nn.LayerNorm(chunk_steps * 7),
            nn.Linear(chunk_steps * 7, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
        )
        self.value = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1)
        )

    def forward(self, context, chunks):
        batch, count, steps, action_dim = chunks.shape
        context_features = self.context(context)[:, None].expand(-1, count, -1)
        candidate_features = self.candidate(
            chunks.reshape(batch, count, steps * action_dim)
        )
        return self.value(
            torch.cat((context_features, candidate_features), dim=-1)
        ).squeeze(-1)


def pairwise_ranking_loss(prediction, target, margin=.05):
    target_difference = target[:, :, None] - target[:, None, :]
    prediction_difference = prediction[:, :, None] - prediction[:, None, :]
    comparable = target_difference.abs() > margin
    if not comparable.any():
        return prediction.sum() * 0
    return F.softplus(
        -target_difference[comparable].sign()
        * prediction_difference[comparable]
    ).mean()


@torch.inference_mode()
def context_at(
    execution, sequence_dir, episode, timestep, device, robot_features=False
):
    tensors = load_file(
        sequence_dir / f"task6-episode{episode}-execution-roles.safetensors"
    )
    _, _, hidden = execution(
        tensors["role_slots"][None, :timestep + 1].to(device),
        tensors["role_geometry"][None, :timestep + 1].to(device),
        tensors["actions"][None, :timestep + 1].to(device),
        (
            tensors["robot_features"][None, :timestep + 1].to(device)
            if robot_features else None
        ),
        return_hidden=True,
    )
    return hidden[0, -1].cpu()


def load_groups(
    paths, execution, sequence_dir, device, robot_features=False
):
    groups = []
    for path in paths:
        row = json.loads(path.read_text())
        candidates = row["candidates"]
        groups.append({
            "episode_id": int(row["episode_id"]),
            "timestep": int(row["failure_timestep"]),
            "names": [item["candidate_name"] for item in candidates],
            "chunks": torch.tensor(
                [item["action_chunk"] for item in candidates],
                dtype=torch.float32,
            ),
            "values": torch.tensor(
                [item["teacher_value"] for item in candidates],
                dtype=torch.float32,
            ),
            "context": context_at(
                execution, sequence_dir, int(row["episode_id"]),
                int(row["failure_timestep"]), device,
                robot_features,
            ),
            "source": str(path),
        })
    return groups


@torch.inference_mode()
def evaluate(model, groups, device):
    result = []
    for group in groups:
        prediction = model(
            group["context"][None].to(device),
            group["chunks"][None].to(device),
        )[0].cpu()
        expected = int(group["values"].argmax())
        selected = int(prediction.argmax())
        result.append({
            "episode_id": group["episode_id"],
            "timestep": group["timestep"],
            "expected_name": group["names"][expected],
            "selected_name": group["names"][selected],
            "top1_correct": selected == expected,
            "teacher_regret": float(
                group["values"][expected] - group["values"][selected]
            ),
            "candidate_names": group["names"],
            "teacher_values": group["values"].tolist(),
            "predicted_values": prediction.tolist(),
        })
    return result


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    execution = ExecutionMamba(
        robot_feature_dim=4 if args.robot_features else 0
    ).to(device).eval()
    execution.load_state_dict(load_file(args.execution_checkpoint))
    groups = load_groups(
        args.candidate_json, execution, args.sequence_dir, device,
        args.robot_features,
    )
    if args.validation_json:
        train = groups
        validation = load_groups(
            args.validation_json, execution, args.sequence_dir, device,
            args.robot_features,
        )
    else:
        train = [
            group for group in groups
            if group["episode_id"] != args.validation_episode
        ]
        validation = [
            group for group in groups
            if group["episode_id"] == args.validation_episode
        ]
    if not train or not validation:
        raise ValueError("both training and validation groups are required")
    model = MacroCandidateScorer(
        chunk_steps=train[0]["chunks"].shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for group in train:
            context = group["context"][None].to(device)
            chunks = group["chunks"][None].to(device)
            values = group["values"][None].to(device)
            # Ranking is within a matched state, so remove group offsets.
            normalized = values - values.mean(1, keepdim=True)
            prediction = model(context, chunks)
            losses.append(
                F.smooth_l1_loss(prediction, normalized)
                + pairwise_ranking_loss(prediction, normalized)
            )
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        report = evaluate(model, validation, device)
        score = (
            sum(item["top1_correct"] for item in report),
            -sum(item["teacher_regret"] for item in report),
            -float(loss),
        )
        if epoch == 1 or epoch % 200 == 0:
            print(
                f"epoch={epoch} loss={float(loss):.5f} "
                f"validation={report[0]['selected_name']}", flush=True
            )
        if best is None or score > best["score"]:
            best = {
                "score": score, "epoch": epoch,
                "state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }
    model.load_state_dict(best["state"])
    reports = {
        "train": evaluate(model, train, device),
        "validation": evaluate(model, validation, device),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "task6-macro-candidate-scorer.safetensors"
    save_file(best["state"], checkpoint)
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "schema_version": 1,
        "best_epoch": best["epoch"],
        "training_episodes": [item["episode_id"] for item in train],
        "validation_episodes": [item["episode_id"] for item in validation],
        "reports": reports,
        "checkpoint": str(checkpoint),
        "sequence_dir": str(args.sequence_dir),
        "execution_checkpoint": str(args.execution_checkpoint),
        "candidate_json": [str(path) for path in args.candidate_json],
        "validation_json": (
            [str(path) for path in args.validation_json]
            if args.validation_json else None
        ),
        "robot_features": args.robot_features,
        "limitation": "development smoke with five matched states",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
