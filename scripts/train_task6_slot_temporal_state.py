#!/usr/bin/env python
"""Train a causal temporal subgoal classifier over continuous Stage-1 slots."""

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

from scripts.train_task6_visual_state_baseline import (
    binary_metrics,
    confirmed_trigger,
    parse_ids,
)

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-episodes", default="0,1,2,3,4,5")
    parser.add_argument("--validation-episodes", default="6,8")
    parser.add_argument("--test-episodes", default="7,9")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--backend", choices=("gru", "mamba2"), default="gru")
    parser.add_argument(
        "--pooling", choices=("single_attention", "role_relation"),
        default="single_attention",
    )
    parser.add_argument("--confirm-frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def load_split(slot_dirs: list[Path], episodes: tuple[int, ...]) -> list[dict]:
    result = []
    for episode in episodes:
        name = f"task6-episode{episode}-slots.safetensors"
        matches = [directory / name for directory in slot_dirs if (directory / name).is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one shard for episode {episode}, found {matches}"
            )
        tensors = load_file(matches[0])
        result.append(
            {
                "episode_id": episode,
                "slots": tensors["slots"].float(),
                "labels": tensors["labels"].float(),
            }
        )
    return result


class SlotTemporalClassifier(nn.Module):
    def __init__(
        self, slot_dim=512, model_dim=128, backend="gru",
        pooling="single_attention",
    ):
        super().__init__()
        self.backend = backend
        self.pooling = pooling
        self.slot_norm = nn.LayerNorm(slot_dim)
        if pooling == "single_attention":
            self.slot_score = nn.Sequential(
                nn.Linear(slot_dim, model_dim), nn.Tanh(), nn.Linear(model_dim, 1)
            )
            self.role_queries = None
            self.role_attention = None
            projection_dim = slot_dim
        elif pooling == "role_relation":
            self.slot_score = None
            self.role_queries = nn.Parameter(torch.randn(2, slot_dim) * 0.02)
            self.role_attention = nn.MultiheadAttention(
                slot_dim, 8, batch_first=True
            )
            projection_dim = slot_dim * 4
        else:
            raise ValueError(f"unsupported pooling: {pooling}")
        self.input_projection = nn.Sequential(
            nn.Linear(projection_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        if backend == "gru":
            self.temporal = nn.GRU(
                model_dim, model_dim, num_layers=2,
                batch_first=True, dropout=0.1,
            )
            self.temporal_layers = nn.ModuleList()
            self.temporal_norms = nn.ModuleList()
        elif backend == "mamba2":
            if Mamba2 is None:
                raise RuntimeError("mamba_ssm.Mamba2 is unavailable")
            self.temporal = None
            self.temporal_layers = nn.ModuleList(
                [
                    Mamba2(
                        d_model=model_dim, d_state=64, d_conv=4, expand=2,
                        headdim=16, use_mem_eff_path=False,
                    )
                    for _ in range(2)
                ]
            )
            self.temporal_norms = nn.ModuleList(
                [nn.LayerNorm(model_dim) for _ in range(2)]
            )
        else:
            raise ValueError(f"unsupported backend: {backend}")
        self.head = nn.Linear(model_dim, 1)

    def forward(self, slots):
        values = self.slot_norm(slots)
        if self.pooling == "single_attention":
            weights = self.slot_score(values).squeeze(-1).softmax(dim=-1)
            pooled = torch.einsum("bts,btsd->btd", weights, values)
        else:
            batch, timesteps, num_slots, dim = values.shape
            flat = values.reshape(batch * timesteps, num_slots, dim)
            queries = self.role_queries[None].expand(len(flat), -1, -1)
            roles, _ = self.role_attention(
                queries, flat, flat, need_weights=False
            )
            first, second = roles[:, 0], roles[:, 1]
            pooled = torch.cat(
                (first, second, first - second, first * second), dim=-1
            ).reshape(batch, timesteps, -1)
        hidden = self.input_projection(pooled)
        if self.backend == "gru":
            hidden, _ = self.temporal(hidden)
        else:
            original_length = hidden.shape[1]
            padding = (-original_length) % 8
            if padding:
                hidden = F.pad(hidden, (0, 0, 0, padding))
            for mixer, norm in zip(self.temporal_layers, self.temporal_norms):
                hidden = hidden + mixer(norm(hidden))
            hidden = hidden[:, :original_length]
        return self.head(hidden).squeeze(-1)


def pad_episodes(episodes: list[dict], device):
    length = max(len(item["labels"]) for item in episodes)
    batch = len(episodes)
    slots = torch.zeros(batch, length, 16, 512)
    labels = torch.zeros(batch, length)
    mask = torch.zeros(batch, length, dtype=torch.bool)
    for index, item in enumerate(episodes):
        count = len(item["labels"])
        slots[index, :count] = item["slots"]
        labels[index, :count] = item["labels"]
        mask[index, :count] = True
    return slots.to(device), labels.to(device), mask.to(device)


@torch.inference_mode()
def predict(model, episodes, device):
    model.eval()
    results = []
    for item in episodes:
        scores = torch.sigmoid(model(item["slots"][None].to(device)))[0].cpu().numpy()
        results.append({**item, "scores": scores})
    return results


def evaluate_predictions(predictions, threshold, confirm_frames):
    all_labels = np.concatenate(
        [item["labels"].numpy().astype(np.int64) for item in predictions]
    )
    all_scores = np.concatenate([item["scores"] for item in predictions])
    report = binary_metrics(all_labels, all_scores, threshold)
    episode_reports = []
    for item in predictions:
        labels = item["labels"].numpy().astype(bool)
        predicted = confirmed_trigger(item["scores"] >= threshold, confirm_frames)
        expected = confirmed_trigger(labels, confirm_frames)
        episode_reports.append(
            {
                "episode_id": item["episode_id"],
                "frames": len(labels),
                "positive_frames": int(labels.sum()),
                "expected_trigger_index": expected,
                "predicted_trigger_index": predicted,
                "trigger_delay_frames": (
                    predicted - expected
                    if predicted is not None and expected is not None else None
                ),
                "false_trigger": predicted is not None and expected is None,
                "missed_trigger": predicted is None and expected is not None,
            }
        )
    report["episodes"] = episode_reports
    return report


def calibrate_threshold(predictions, confirm_frames):
    candidates = np.linspace(0.50, 0.995, 100)
    best = None
    for threshold in candidates:
        report = evaluate_predictions(predictions, float(threshold), confirm_frames)
        false = sum(item["false_trigger"] for item in report["episodes"])
        missed = sum(item["missed_trigger"] for item in report["episodes"])
        delays = [
            abs(item["trigger_delay_frames"])
            for item in report["episodes"]
            if item["trigger_delay_frames"] is not None
        ]
        # A detector that never fires is not a useful recovery model.  First
        # require positive validation trajectories to be detected, then reduce
        # false triggers and timing error.
        key = (missed, false, sum(delays), -report["balanced_accuracy"])
        if best is None or key < best[0]:
            best = (key, float(threshold), report)
    return best[1], best[2]


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    split_ids = {
        "train": parse_ids(args.train_episodes),
        "validation": parse_ids(args.validation_episodes),
        "test": parse_ids(args.test_episodes),
    }
    if set(split_ids["train"]) & (
        set(split_ids["validation"]) | set(split_ids["test"])
    ) or set(split_ids["validation"]) & set(split_ids["test"]):
        raise ValueError("episode leakage across splits")
    splits = {
        name: load_split(args.slot_dir, ids) for name, ids in split_ids.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SlotTemporalClassifier(
        model_dim=args.model_dim, backend=args.backend, pooling=args.pooling
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    slots, labels, mask = pad_episodes(splits["train"], device)
    positives = labels[mask].sum()
    pos_weight = ((mask.sum() - positives) / positives.clamp_min(1)).detach()
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(slots)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits[mask], labels[mask], pos_weight=pos_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            validation_predictions = predict(model, splits["validation"], device)
            threshold, validation = calibrate_threshold(
                validation_predictions, args.confirm_frames
            )
            false = sum(x["false_trigger"] for x in validation["episodes"])
            missed = sum(x["missed_trigger"] for x in validation["episodes"])
            delays = [
                abs(x["trigger_delay_frames"])
                for x in validation["episodes"]
                if x["trigger_delay_frames"] is not None
            ]
            key = (
                missed, false, sum(delays), -validation["balanced_accuracy"],
            )
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss),
                    "threshold": threshold,
                    "validation": validation,
                }
            )
            print(
                f"epoch={epoch} loss={loss.item():.4f} threshold={threshold:.3f} "
                f"val_bal_acc={validation['balanced_accuracy']:.4f} "
                f"trajectory_errors={false + missed}",
                flush=True,
            )
            if best is None or key < best["key"]:
                best = {
                    "key": key,
                    "epoch": epoch,
                    "threshold": threshold,
                    "state": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                }
    model.load_state_dict(best["state"])
    predictions = {
        name: predict(model, episodes, device) for name, episodes in splits.items()
    }
    metrics = {
        name: evaluate_predictions(values, best["threshold"], args.confirm_frames)
        for name, values in predictions.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "task6-slot-temporal-state.safetensors"
    save_file(
        {
            **{f"model.{key}": value.contiguous() for key, value in best["state"].items()},
            "threshold": torch.tensor([best["threshold"]]),
            "confirm_frames": torch.tensor([args.confirm_frames]),
        },
        checkpoint,
    )
    result = {
        "schema_version": 1,
        "kind": (
            f"causal_continuous_slot_{args.pooling}_{args.backend}_"
            "subgoal_classifier"
        ),
        "backend": args.backend,
        "pooling": args.pooling,
        "input_contract": "16x512 continuous slots only; no simulator state",
        "episode_split": split_ids,
        "best_epoch": best["epoch"],
        "threshold": best["threshold"],
        "confirm_frames": args.confirm_frames,
        "metrics": metrics,
        "history": history,
        "checkpoint": str(checkpoint.resolve()),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
