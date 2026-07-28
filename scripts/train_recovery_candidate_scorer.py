#!/usr/bin/env python
"""Train the recovery action-effect scorer on matched simulator branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovery.data.candidate_schema import read_candidate_jsonl
from recovery.data.candidate_tensor_dataset import (
    RecoveryCandidateDataset,
    collate_recovery_candidates,
)
from recovery.data.dataset import read_jsonl
from recovery.models.candidate_effect_scorer import (
    RecoveryCandidateEffectScorer,
    candidate_effect_losses,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", type=Path, required=True)
    parser.add_argument("--transitions", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ranking-margin", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def expand(paths: list[Path], filename: str) -> list[Path]:
    result = []
    for path in paths:
        result.extend([path] if path.is_file() else sorted(path.rglob(filename)))
    return result


def main() -> None:
    args = arguments()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    groups = [
        group
        for path in expand(args.candidates, "candidates.jsonl")
        for group in read_candidate_jsonl(path)
    ]
    rows = [
        row
        for path in expand(args.transitions, "transitions-v2.jsonl")
        for row in read_jsonl(path)
    ]
    transition_map = {
        (row.task_suite, row.task_id, row.episode_id, row.timestep): row
        for row in rows
    }
    train_groups = [group for group in groups if group.split == "train"]
    validation_groups = [
        group for group in groups if group.split == "validation"
    ]
    if not validation_groups:
        raise ValueError("at least one validation candidate group is required")
    train = RecoveryCandidateDataset(train_groups, transition_map)
    validation = RecoveryCandidateDataset(validation_groups, transition_map)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_recovery_candidates,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_recovery_candidates,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecoveryCandidateEffectScorer(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3
    )

    def move(batch):
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    @torch.inference_mode()
    def evaluate() -> dict[str, float]:
        model.eval()
        totals: dict[str, float] = {}
        correct = count = 0
        for raw in validation_loader:
            batch = move(raw)
            outputs = model(
                batch["context"],
                batch["nominal_action"],
                batch["candidate_actions"],
            )
            values = candidate_effect_losses(
                outputs, batch, ranking_margin=args.ranking_margin
            )
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value)
            masked_prediction = outputs["candidate_value"].masked_fill(
                ~batch["outcome_valid"], float("-inf")
            )
            masked_target = batch["teacher_value"].masked_fill(
                ~batch["outcome_valid"], float("-inf")
            )
            correct += int(
                (masked_prediction.argmax(1) == masked_target.argmax(1)).sum()
            )
            count += len(batch["context"])
        model.train()
        return {
            **{
                f"validation_{name}": value / max(len(validation_loader), 1)
                for name, value in totals.items()
            },
            "validation_best_candidate_accuracy": correct / max(count, 1),
        }

    iterator = iter(train_loader)
    history = []
    for step in range(1, args.steps + 1):
        try:
            batch = move(next(iterator))
        except StopIteration:
            iterator = iter(train_loader)
            batch = move(next(iterator))
        outputs = model(
            batch["context"],
            batch["nominal_action"],
            batch["candidate_actions"],
        )
        values = candidate_effect_losses(
            outputs, batch, ranking_margin=args.ranking_margin
        )
        optimizer.zero_grad(set_to_none=True)
        values["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            record = {
                "step": step,
                **{f"train_{name}": float(value) for name, value in values.items()},
                **evaluate(),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in model.state_dict().items()
        },
        str(args.output_dir / "candidate-scorer.safetensors"),
    )
    report = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "train_groups": len(train),
        "validation_groups": len(validation),
        "history": history,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
