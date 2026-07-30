#!/usr/bin/env python
"""Train shared causal Mamba2 execution heads over three grounded role slots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.train_task6_execution_slots import HEADS  # noqa: E402
from scripts.train_task6_visual_state_baseline import (  # noqa: E402
    binary_metrics,
    parse_ids,
)

FAILED_ACQUISITION_THRESHOLD = .08
MOTION_GATE_CENTER = .02
MOTION_GATE_SCALE = 200.


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--train-episodes", default="0,2,3,4,5,20,21,22,23,24,25,26,27,28,29"
    )
    parser.add_argument("--validation-episodes", default="1,6,8")
    parser.add_argument("--test-episodes", default="7,9")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def load_episodes(root, ids):
    result = []
    for episode_id in ids:
        tensors = load_file(
            root / f"task6-episode{episode_id}-execution-roles.safetensors"
        )
        result.append({"episode_id": episode_id, **tensors})
    return result


class ExecutionMamba(nn.Module):
    def __init__(self, model_dim=128):
        super().__init__()
        # Three roles, three directed relational differences, and the action.
        # Geometry plus explicit causal finite differences preserve motion
        # evidence that appearance-slot pooling tends to discard.
        self.input_projection = nn.Sequential(
            nn.LayerNorm(96 * 6 + 3 * 3 * 4 + 7),
            nn.Linear(96 * 6 + 3 * 3 * 4 + 7, model_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList([
            Mamba2(
                d_model=model_dim, d_state=64, d_conv=4, expand=2,
                headdim=16, use_mem_eff_path=False,
            )
            for _ in range(2)
        ])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(model_dim) for _ in range(2)]
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.binary_heads = nn.Linear(model_dim, len(HEADS))
        self.value_head = nn.Linear(model_dim, 1)

    def forward(self, roles, geometry, actions):
        mug, plate, pudding = roles[:, :, 0], roles[:, :, 1], roles[:, :, 2]
        geometry_features = [geometry]
        for delay in (5, 10, 20):
            difference = torch.zeros_like(geometry)
            difference[:, delay:] = geometry[:, delay:] - geometry[:, :-delay]
            geometry_features.append(difference)
        inputs = torch.cat((
            mug, plate, pudding,
            pudding - mug, pudding - plate, mug - plate,
            torch.cat(geometry_features, dim=-1).flatten(2),
            actions,
        ), dim=-1)
        hidden = self.input_projection(inputs)
        original = hidden.shape[1]
        padding = (-original) % 8
        if padding:
            hidden = F.pad(hidden, (0, 0, 0, padding))
        for layer, norm in zip(self.layers, self.norms):
            hidden = hidden + layer(norm(hidden))
        hidden = self.output_norm(hidden[:, :original])
        return self.binary_heads(hidden), self.value_head(hidden).squeeze(-1)


def pad(episodes, device):
    length = max(len(item["actions"]) for item in episodes)
    batch = len(episodes)
    roles = torch.zeros(batch, length, 3, 96)
    geometry = torch.zeros(batch, length, 3, 3)
    actions = torch.zeros(batch, length, 7)
    targets = torch.zeros(batch, length, len(HEADS))
    values = torch.zeros(batch, length)
    active = torch.zeros(batch, length, dtype=torch.bool)
    assessment = torch.zeros(batch, length, dtype=torch.bool)
    valid = torch.zeros(batch, length, dtype=torch.bool)
    for index, item in enumerate(episodes):
        count = len(item["actions"])
        roles[index, :count] = item["role_slots"]
        geometry[index, :count] = item["role_geometry"]
        actions[index, :count] = item["actions"]
        targets[index, :count] = item["targets"]
        values[index, :count] = item["candidate_values"]
        active[index, :count] = item["stage2_active"]
        assessment[index, :count] = item["acquisition_assessment_active"]
        valid[index, :count] = True
    return tuple(
        tensor.to(device)
        for tensor in (
            roles, geometry, actions, targets, values, active, assessment, valid
        )
    )


@torch.inference_mode()
def evaluate(model, episodes, device):
    model.eval()
    all_scores = {name: [] for name in HEADS}
    all_targets = {name: [] for name in HEADS}
    all_values, all_value_targets = [], []
    episode_reports = []
    for item in episodes:
        logits, values = model(
            item["role_slots"][None].to(device),
            item["role_geometry"][None].to(device),
            item["actions"][None].to(device),
        )
        active = item["stage2_active"].bool()
        logits = logits[0].cpu()
        probabilities = logits.sigmoid()
        geometry = item["role_geometry"][:, 2, :2]
        motion = torch.zeros(len(geometry))
        motion[15:] = (geometry[15:] - geometry[:-15]).norm(dim=-1)
        no_motion_gate = torch.sigmoid(
            (MOTION_GATE_CENTER - motion) * MOTION_GATE_SCALE
        )
        probabilities[:, HEADS.index("failed_acquisition")] = torch.minimum(
            probabilities[:, HEADS.index("failed_acquisition")],
            no_motion_gate,
        )
        active_values = values[0].cpu()[active]
        target_values = item["candidate_values"][active]
        all_values.append(active_values)
        all_value_targets.append(target_values)
        report = {"episode_id": item["episode_id"], "stage2_frames": int(active.sum())}
        for column, name in enumerate(HEADS):
            head_mask = (
                item["acquisition_assessment_active"].bool()
                if name == "failed_acquisition" else active
            )
            labels = item["targets"][head_mask, column]
            scores = probabilities[head_mask, column]
            threshold = (
                FAILED_ACQUISITION_THRESHOLD
                if name == "failed_acquisition" else .5
            )
            report[name] = binary_metrics(
                labels.numpy().astype(int), scores.numpy(), threshold
            )
            all_targets[name].append(labels)
            all_scores[name].append(scores)
        episode_reports.append(report)
    values = torch.cat(all_values)
    value_targets = torch.cat(all_value_targets)
    report = {"stage2_frames": int(sum(item["stage2_frames"] for item in episode_reports))}
    for column, name in enumerate(HEADS):
        threshold = (
            FAILED_ACQUISITION_THRESHOLD
            if name == "failed_acquisition" else .5
        )
        report[name] = binary_metrics(
            torch.cat(all_targets[name]).numpy().astype(int),
            torch.cat(all_scores[name]).numpy(),
            threshold,
        )
    report["candidate_value_mae_m"] = float(
        (values - value_targets).abs().mean()
    )
    report["episodes"] = episode_reports
    return report


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    split_ids = {
        "train": parse_ids(args.train_episodes),
        "validation": parse_ids(args.validation_episodes),
        "test": parse_ids(args.test_episodes),
    }
    splits = {
        name: load_episodes(args.sequence_dir, ids)
        for name, ids in split_ids.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExecutionMamba(args.model_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    roles, geometry, actions, targets, values, active, assessment, valid = pad(
        splits["train"], device
    )
    head_masks = active[:, :, None].expand_as(targets).clone()
    head_masks[:, :, HEADS.index("failed_acquisition")] = assessment
    positive = (targets * head_masks).sum((0, 1))
    negative = head_masks.sum((0, 1)) - positive
    pos_weight = (negative / positive.clamp_min(1)).clamp(1, 30)
    best, history = None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        binary, predicted_values = model(roles, geometry, actions)
        element_loss = F.binary_cross_entropy_with_logits(
            binary, targets, pos_weight=pos_weight, reduction="none"
        )
        binary_loss = element_loss[head_masks].mean()
        value_loss = F.smooth_l1_loss(
            predicted_values[active] * 100, values[active] * 100
        )
        loss = binary_loss + .1 * value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        validation = evaluate(model, splits["validation"], device)
        score = sum(
            validation[name]["balanced_accuracy"]
            for name in ("effective_progress", "stall", "controlled",
                         "failed_acquisition")
        )
        history.append({
            "epoch": epoch, "loss": float(loss),
            "validation_score": score,
        })
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch} loss={float(loss):.4f} score={score:.4f}",
                flush=True,
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
    metrics = {
        name: evaluate(model, episodes, device)
        for name, episodes in splits.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "task6-execution-mamba.safetensors"
    save_file(best["state"], checkpoint)
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "schema_version": 1,
        "kind": "three_grounded_roles_action_conditioned_execution_mamba2",
        "head_order": HEADS,
        "failed_acquisition_threshold": FAILED_ACQUISITION_THRESHOLD,
        "motion_gate_center": MOTION_GATE_CENTER,
        "motion_gate_scale": MOTION_GATE_SCALE,
        "episode_split": split_ids,
        "best_epoch": best["epoch"],
        "head_positive_counts": {
            name: int(positive[index])
            for index, name in enumerate(HEADS)
        },
        "head_pos_weights": {
            name: float(pos_weight[index])
            for index, name in enumerate(HEADS)
        },
        "metrics": metrics,
        "history": history,
        "checkpoint": str(checkpoint),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
