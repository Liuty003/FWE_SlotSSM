#!/usr/bin/env python
"""Train persistent completion and rollback state over grounded role slots."""

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

from scripts.train_task6_visual_state_baseline import (
    binary_metrics,
    confirmed_trigger,
    parse_ids,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--train-episodes", default="0,1,2,3,4,5,20,21,22,23,24,25,26,27,28,29"
    )
    parser.add_argument("--validation-episodes", default="6,8")
    parser.add_argument("--test-episodes", default="7,9")
    parser.add_argument("--backend", choices=("gru", "mamba2"), default="mamba2")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--confirm-frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def stable_history(labels, confirm_frames):
    latched = torch.zeros_like(labels, dtype=torch.bool)
    streak = 0
    active = False
    for index, value in enumerate(labels.bool()):
        streak = streak + 1 if bool(value) else 0
        active = active or streak >= confirm_frames
        latched[index] = active
    return latched


def load_episodes(root, episode_ids, confirm_frames):
    result = []
    for episode in episode_ids:
        tensors = load_file(
            root / f"task6-episode{episode}-grounded-roles.safetensors"
        )
        labels = tensors["labels"].float()
        completed_before = stable_history(labels, confirm_frames)
        rollback = completed_before & ~labels.bool()
        result.append({
            "episode_id": episode,
            "role_slots": tensors["role_slots"].float(),
            "single_probability": tensors["single_frame_probability"].float(),
            "labels": labels,
            "rollback": rollback.float(),
        })
    return result


class GroundedRoleMamba(nn.Module):
    def __init__(self, model_dim=128, backend="mamba2"):
        super().__init__()
        self.backend = backend
        self.input_projection = nn.Sequential(
            nn.LayerNorm(96 * 4 + 1),
            nn.Linear(96 * 4 + 1, model_dim),
            nn.GELU(),
        )
        if backend == "mamba2":
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
            self.gru = None
        else:
            self.layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.gru = nn.GRU(
                model_dim, model_dim, num_layers=2,
                batch_first=True, dropout=.1,
            )
        self.output_norm = nn.LayerNorm(model_dim)
        self.completion_head = nn.Linear(model_dim, 1)
        self.rollback_head = nn.Linear(model_dim, 1)

    def forward(self, role_slots, single_probability):
        first, second = role_slots[:, :, 0], role_slots[:, :, 1]
        inputs = torch.cat(
            (
                first, second, first - second, first * second,
                single_probability[..., None],
            ),
            dim=-1,
        )
        hidden = self.input_projection(inputs)
        if self.backend == "mamba2":
            original = hidden.shape[1]
            padding = (-original) % 8
            if padding:
                hidden = F.pad(hidden, (0, 0, 0, padding))
            for layer, norm in zip(self.layers, self.norms):
                hidden = hidden + layer(norm(hidden))
            hidden = hidden[:, :original]
        else:
            hidden, _ = self.gru(hidden)
        hidden = self.output_norm(hidden)
        return (
            self.completion_head(hidden).squeeze(-1),
            self.rollback_head(hidden).squeeze(-1),
        )


def pad(episodes, device):
    length = max(len(item["labels"]) for item in episodes)
    batch = len(episodes)
    roles = torch.zeros(batch, length, 2, 96)
    single = torch.zeros(batch, length)
    labels = torch.zeros(batch, length)
    rollback = torch.zeros(batch, length)
    mask = torch.zeros(batch, length, dtype=torch.bool)
    for index, item in enumerate(episodes):
        count = len(item["labels"])
        roles[index, :count] = item["role_slots"]
        single[index, :count] = item["single_probability"]
        labels[index, :count] = item["labels"]
        rollback[index, :count] = item["rollback"]
        mask[index, :count] = True
    return tuple(
        tensor.to(device)
        for tensor in (roles, single, labels, rollback, mask)
    )


@torch.inference_mode()
def predict(model, episodes, device):
    model.eval()
    outputs = []
    for item in episodes:
        completion, rollback = model(
            item["role_slots"][None].to(device),
            item["single_probability"][None].to(device),
        )
        outputs.append({
            **item,
            # The grounded single-frame relation is the precision gate; the
            # temporal model may confirm it, but may not override it based on
            # task progress alone.
            "completion_scores": torch.minimum(
                completion[0].sigmoid().cpu(),
                item["single_probability"],
            ).numpy(),
            "rollback_scores": torch.minimum(
                rollback[0].sigmoid().cpu(),
                1 - item["single_probability"],
            ).numpy(),
        })
    return outputs


def first_rollback(values, start, confirm_frames):
    if start is None:
        return None
    streak = 0
    for index in range(start + 1, len(values)):
        streak = streak + 1 if bool(values[index]) else 0
        if streak >= confirm_frames:
            return index
    return None


def evaluate(
    predictions, threshold, confirm_frames, rollback_threshold=.8
):
    labels = np.concatenate(
        [item["labels"].numpy().astype(np.int64) for item in predictions]
    )
    scores = np.concatenate([item["completion_scores"] for item in predictions])
    report = binary_metrics(labels, scores, threshold)
    episodes = []
    for item in predictions:
        truth = item["labels"].numpy().astype(bool)
        expected_entry = confirmed_trigger(truth, confirm_frames)
        predicted_entry = confirmed_trigger(
            item["completion_scores"] >= threshold, confirm_frames
        )
        expected_rollback = first_rollback(
            ~truth, expected_entry, confirm_frames
        )
        predicted_rollback = first_rollback(
            item["rollback_scores"] >= rollback_threshold,
            predicted_entry,
            confirm_frames,
        )
        episodes.append({
            "episode_id": item["episode_id"],
            "expected_entry": expected_entry,
            "predicted_entry": predicted_entry,
            "false_entry": predicted_entry is not None and expected_entry is None,
            "missed_entry": predicted_entry is None and expected_entry is not None,
            "entry_delay": (
                predicted_entry - expected_entry
                if predicted_entry is not None and expected_entry is not None else None
            ),
            "expected_rollback": expected_rollback,
            "predicted_rollback": predicted_rollback,
            "rollback_delay": (
                predicted_rollback - expected_rollback
                if predicted_rollback is not None and expected_rollback is not None
                else None
            ),
        })
    report["episodes"] = episodes
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
        name: load_episodes(
            args.sequence_dir, ids, args.confirm_frames
        )
        for name, ids in split_ids.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GroundedRoleMamba(
        model_dim=args.model_dim, backend=args.backend
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    roles, single, labels, rollback, mask = pad(splits["train"], device)
    completion_weight = (mask.sum() - labels[mask].sum()) / labels[mask].sum()
    rollback_positive = rollback[mask].sum()
    rollback_weight = (
        (mask.sum() - rollback_positive) / rollback_positive.clamp_min(1)
    ).clamp(max=50)
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        completion_logits, rollback_logits = model(roles, single)
        completion_loss = F.binary_cross_entropy_with_logits(
            completion_logits[mask], labels[mask],
            pos_weight=completion_weight,
        )
        rollback_loss = F.binary_cross_entropy_with_logits(
            rollback_logits[mask], rollback[mask],
            pos_weight=rollback_weight,
        )
        loss = completion_loss + .5 * rollback_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            validation = evaluate(
                predict(model, splits["validation"], device),
                .5, args.confirm_frames,
            )
            train_report = evaluate(
                predict(model, splits["train"], device),
                .5, args.confirm_frames,
            )
            errors = sum(
                item["false_entry"] or item["missed_entry"]
                for item in validation["episodes"]
            )
            delays = sum(
                abs(item["entry_delay"])
                for item in validation["episodes"]
                if item["entry_delay"] is not None
            )
            rollback_errors = sum(
                (item["expected_rollback"] is None)
                != (item["predicted_rollback"] is None)
                for item in train_report["episodes"]
            )
            rollback_delays = sum(
                abs(item["rollback_delay"])
                for item in train_report["episodes"]
                if item["rollback_delay"] is not None
            )
            key = (
                errors, rollback_errors, rollback_delays, delays,
                -validation["balanced_accuracy"],
            )
            history.append({
                "epoch": epoch, "loss": float(loss),
                "completion_loss": float(completion_loss),
                "rollback_loss": float(rollback_loss),
                "validation": validation,
                "train_rollback_errors": rollback_errors,
                "train_rollback_delay": rollback_delays,
            })
            print(
                f"epoch={epoch} loss={loss.item():.4f} "
                f"val_ba={validation['balanced_accuracy']:.4f} "
                f"errors={errors} rollback_errors={rollback_errors} "
                f"rollback_delay={rollback_delays} delays={delays}",
                flush=True,
            )
            if best is None or key < best["key"]:
                best = {
                    "key": key, "epoch": epoch,
                    "state": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                }
    model.load_state_dict(best["state"])
    metrics = {
        name: evaluate(predict(model, values, device), .5, args.confirm_frames)
        for name, values in splits.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "task6-grounded-role-mamba.safetensors"
    save_file(best["state"], checkpoint)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({
            "schema_version": 1,
            "kind": f"grounded_role_{args.backend}_completion_and_rollback",
            "backend": args.backend,
            "episode_split": split_ids,
            "best_epoch": best["epoch"],
            "metrics": metrics,
            "history": history,
            "checkpoint": str(checkpoint),
        }, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
