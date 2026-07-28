#!/usr/bin/env python
"""Train candidate futures from a frozen Recovery SlotSSM temporal memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovery.data.candidate_schema import read_candidate_jsonl
from recovery.data.candidate_tensor_dataset import (
    UnifiedRecoveryCandidateDataset,
    collate_recovery_candidates,
)
from recovery.data.dataset import read_jsonl
from recovery.models.candidate_effect_scorer import candidate_effect_losses
from recovery.models.recovery_slotssm import RecoverySlotSSM
from recovery.models.unified_recovery import UnifiedRecoveryModel


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--recovery-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-hidden-dim", type=int, default=128)
    parser.add_argument("--geometry-skip", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ranking-margin", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def load_data(args):
    groups = [
        group
        for path in sorted(args.candidates.rglob("candidates.jsonl"))
        for group in read_candidate_jsonl(path)
    ]
    rows = [
        row
        for path in sorted(args.transitions.rglob("transitions-v2.jsonl"))
        for row in read_jsonl(path)
    ]
    transitions = {
        (row.task_suite, row.task_id, row.episode_id, row.timestep): row
        for row in rows
    }
    datasets = {}
    for split in ("train", "validation"):
        datasets[split] = UnifiedRecoveryCandidateDataset(
            [group for group in groups if group.split == split],
            transitions,
            window_length=args.window_length,
        )
    return datasets


def main() -> None:
    args = arguments()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    datasets = load_data(args)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate_recovery_candidates,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_recovery_candidates,
        ),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recovery = RecoverySlotSSM(
        model_dim=128, num_layers=2, backend="mamba2"
    )
    recovery.load_state_dict(
        load_file(str(args.recovery_checkpoint)), strict=True
    )
    model = UnifiedRecoveryModel(
        recovery,
        candidate_hidden_dim=args.candidate_hidden_dim,
        geometry_skip=args.geometry_skip,
    ).to(device)
    model.freeze_recovery()
    model.recovery.eval()
    optimizer = torch.optim.AdamW(
        model.candidate_head.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-3,
    )

    def move(batch):
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    @torch.inference_mode()
    def evaluate():
        model.eval()
        totals = {}
        shuffled_total = regret = nominal_regret = random_regret = 0.0
        correct = count = 0
        for raw in loaders["validation"]:
            batch = move(raw)
            outputs = model(
                batch["context"],
                nominal_action=batch["nominal_action"],
                candidate_actions=batch["candidate_actions"],
                candidate_timestep=batch["candidate_timestep"],
            )["candidate"]
            losses = candidate_effect_losses(
                outputs, batch, ranking_margin=args.ranking_margin
            )
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value)
            shuffled_actions = batch["candidate_actions"].clone()
            for batch_index in range(len(shuffled_actions)):
                indices = batch["candidate_mask"][batch_index].nonzero().flatten()
                shuffled_actions[batch_index, indices] = shuffled_actions[
                    batch_index, indices.flip(0)
                ]
            shuffled = model(
                batch["context"],
                nominal_action=batch["nominal_action"],
                candidate_actions=shuffled_actions,
                candidate_timestep=batch["candidate_timestep"],
            )["candidate"]
            shuffled_total += float(
                candidate_effect_losses(
                    shuffled, batch, ranking_margin=args.ranking_margin
                )["total"]
            )
            prediction = outputs["candidate_value"].masked_fill(
                ~batch["outcome_valid"], float("-inf")
            )
            target = batch["teacher_value"].masked_fill(
                ~batch["outcome_valid"], float("-inf")
            )
            selected = prediction.argmax(1)
            best = target.max(1).values
            chosen = batch["teacher_value"].gather(
                1, selected[:, None]
            ).squeeze(1)
            correct += int((selected == target.argmax(1)).sum())
            regret += float((best - chosen).sum())
            for index, metadata in enumerate(raw["metadata"]):
                candidate_count = len(metadata["candidate_names"])
                values = batch["teacher_value"][index, :candidate_count]
                oracle = values.max()
                nominal = metadata["candidate_names"].index("nominal")
                nominal_regret += float(oracle - values[nominal])
                random_regret += float((oracle - values).mean())
            count += len(batch["context"])
        batches = max(len(loaders["validation"]), 1)
        model.train()
        model.recovery.eval()
        return {
            **{
                f"validation_{name}": value / batches
                for name, value in totals.items()
            },
            "validation_shuffled_total": shuffled_total / batches,
            "validation_action_shuffle_loss_delta": (
                shuffled_total - totals["total"]
            )
            / batches,
            "validation_best_candidate_accuracy": correct / max(count, 1),
            "validation_mean_regret": regret / max(count, 1),
            "validation_nominal_mean_regret": nominal_regret / max(count, 1),
            "validation_uniform_random_expected_regret": (
                random_regret / max(count, 1)
            ),
        }

    iterator = iter(loaders["train"])
    history = []
    best_score = (float("inf"), float("inf"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        try:
            batch = move(next(iterator))
        except StopIteration:
            iterator = iter(loaders["train"])
            batch = move(next(iterator))
        outputs = model(
            batch["context"],
            nominal_action=batch["nominal_action"],
            candidate_actions=batch["candidate_actions"],
            candidate_timestep=batch["candidate_timestep"],
        )["candidate"]
        losses = candidate_effect_losses(
            outputs, batch, ranking_margin=args.ranking_margin
        )
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.candidate_head.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            metrics = evaluate()
            record = {
                "step": step,
                **{f"train_{key}": float(value) for key, value in losses.items()},
                **metrics,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            score = (
                metrics["validation_mean_regret"],
                metrics["validation_total"],
            )
            if score < best_score:
                best_score = score
                save_file(
                    {
                        key: value.detach().cpu().contiguous()
                        for key, value in model.candidate_head.state_dict().items()
                    },
                    str(args.output_dir / "candidate-head-best.safetensors"),
                )
                save_file(
                    {
                        key: value.detach().cpu().contiguous()
                        for key, value in model.state_dict().items()
                    },
                    str(args.output_dir / "unified-recovery-best.safetensors"),
                )
    save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in model.candidate_head.state_dict().items()
        },
        str(args.output_dir / "candidate-head-last.safetensors"),
    )
    save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in model.state_dict().items()
        },
        str(args.output_dir / "unified-recovery-last.safetensors"),
    )
    report = {
        "kind": "frozen_recovery_slotssm_shared_memory_candidate_head",
        "recovery_checkpoint": str(args.recovery_checkpoint.resolve()),
        "recovery_frozen": True,
        "train_groups": len(datasets["train"]),
        "validation_groups": len(datasets["validation"]),
        "window_length": args.window_length,
        "geometry_skip": args.geometry_skip,
        "best_selection_score": list(best_score),
        "history": history,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
