#!/usr/bin/env python
"""Train a short shared Recovery SlotSSM smoke run on schema-v2 sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovery.data.dataset import read_jsonl
from recovery.data.tensor_dataset import (
    PHASE_TO_ID,
    RecoverySequenceDataset,
    build_recovery_model_inputs,
    collate_recovery_sequences,
)
from recovery.models.recovery_slotssm import RecoverySlotSSM


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("gru", "mamba2"), default="gru")
    parser.add_argument("--window-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--balanced-window-sampling", action="store_true")
    parser.add_argument("--max-window-weight", type=float, default=4.0)
    return parser.parse_args()


def expand(value: Path) -> list[Path]:
    if value.is_file():
        return [value]
    return sorted(value.glob("task*-episode*/transitions-v2.jsonl"))


def load_rows(paths: list[Path]):
    rows = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def balanced_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    positives = (targets * valid).sum()
    negatives = ((1.0 - targets) * valid).sum()
    positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
    values = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=positive_weight,
    )
    return masked_mean(values, valid)


def losses(
    outputs: dict[str, torch.Tensor],
    sequence: dict[str, torch.Tensor],
    phase_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    phase = F.cross_entropy(
        outputs["phase_logits"].flatten(0, 1),
        sequence["phase_id"].flatten().clamp_min(0),
        weight=phase_weights,
        reduction="none",
    ).reshape_as(sequence["phase_valid"])
    phase_loss = masked_mean(phase, sequence["phase_valid"])
    failure_loss = balanced_binary_loss(
        outputs["current_failure_logits"],
        sequence["current_failure_target"],
        sequence["current_failure_valid"],
    )
    recovery_loss = balanced_binary_loss(
        outputs["recovery_active_logits"],
        sequence["recovery_active_target"],
        torch.ones_like(sequence["recovery_active_target"]),
    )
    verification_loss = balanced_binary_loss(
        outputs["verification_logits"],
        sequence["verification_target"],
        sequence["verification_valid"],
    )
    handoff_loss = balanced_binary_loss(
        outputs["handoff_logits"],
        sequence["handoff_target"],
        sequence["handoff_valid"],
    )
    grasped_loss = balanced_binary_loss(
        outputs["grasped_logits"],
        sequence["grasped_target"],
        sequence["grasped_valid"],
    )
    lifted_loss = balanced_binary_loss(
        outputs["lifted_logits"],
        sequence["lifted_target"],
        sequence["lifted_valid"],
    )
    slip_loss = balanced_binary_loss(
        outputs["slip_logits"],
        sequence["slip_target"],
        sequence["slip_valid"],
    )
    residual = F.smooth_l1_loss(
        outputs["residual_action"],
        sequence["expert_residual"],
        reduction="none",
    ).mean(-1)
    residual_loss = masked_mean(residual, sequence["expert_action_valid"])
    total = (
        phase_loss
        + failure_loss
        + recovery_loss
        + 0.5 * verification_loss
        + 0.5 * handoff_loss
        + 0.5 * grasped_loss
        + 0.5 * lifted_loss
        + 0.5 * slip_loss
        + residual_loss
    )
    return {
        "total": total,
        "phase": phase_loss,
        "failure": failure_loss,
        "recovery": recovery_loss,
        "verification": verification_loss,
        "handoff": handoff_loss,
        "grasped": grasped_loss,
        "lifted": lifted_loss,
        "slip": slip_loss,
        "residual": residual_loss,
    }


@torch.inference_mode()
def evaluate(model, loader, device, phase_weights):
    totals = {}
    count = 0
    phase_correct = phase_total = 0
    residual_sum = residual_count = 0.0
    for batch in loader:
        sequence = {
            key: value.to(device) for key, value in batch["sequence"].items()
        }
        outputs = model(build_recovery_model_inputs(sequence))
        batch_losses = losses(outputs, sequence, phase_weights)
        for name, value in batch_losses.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        count += 1
        prediction = outputs["phase_logits"].argmax(-1)
        valid = sequence["phase_valid"].bool()
        phase_correct += int((prediction[valid] == sequence["phase_id"][valid]).sum())
        phase_total += int(valid.sum())
        expert = sequence["expert_action_valid"].bool()
        residual_sum += float(
            (outputs["residual_action"] - sequence["expert_residual"])
            .abs().mean(-1)[expert].sum()
        )
        residual_count += int(expert.sum())
    return {
        "losses": {name: value / max(count, 1) for name, value in totals.items()},
        "phase_accuracy": phase_correct / max(phase_total, 1),
        "expert_residual_mae": residual_sum / max(residual_count, 1),
    }


def main() -> None:
    args = arguments()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    paths = [path for value in args.inputs for path in expand(value)]
    rows = load_rows(paths)
    train_rows = [row for row in rows if row.split == "train"]
    validation_rows = [row for row in rows if row.split == "validation"]
    train_dataset = RecoverySequenceDataset(
        train_rows,
        window_length=args.window_length,
        stride=args.stride,
    )
    validation_dataset = RecoverySequenceDataset(
        validation_rows,
        window_length=args.window_length,
        stride=args.stride,
    )
    generator = torch.Generator().manual_seed(args.seed)
    # Compute weights from independent rows rather than overlapping windows.
    phase_counts = torch.zeros(len(PHASE_TO_ID), dtype=torch.float32)
    for row in train_rows:
        phase_id = PHASE_TO_ID.get(str(row.labels.get("phase")))
        if phase_id is not None:
            phase_counts[phase_id] += 1
    present = phase_counts > 0
    phase_weights = torch.zeros_like(phase_counts)
    phase_weights[present] = (
        phase_counts[present].sum()
        / (present.sum() * phase_counts[present])
    ).clamp(0.25, 10.0)
    window_weights = torch.ones(len(train_dataset), dtype=torch.double)
    if args.balanced_window_sampling:
        for index, window in enumerate(train_dataset.windows):
            rare_weights = [
                float(phase_weights[PHASE_TO_ID[str(row.labels.get("phase"))]])
                for row in window.rows
                if str(row.labels.get("phase")) in PHASE_TO_ID
            ]
            event_boost = 1.5 if any(
                bool(row.labels.get("failure_detection_event"))
                or bool(row.labels.get("slipped"))
                for row in window.rows
            ) else 1.0
            rare = max(rare_weights or [1.0])
            # Mix rare-event emphasis with the original distribution instead
            # of sampling in direct inverse proportion.
            mixed = 1.0 + 0.5 * max(0.0, rare - 1.0)
            window_weights[index] = min(
                args.max_window_weight, mixed * event_boost
            )
        sampler = WeightedRandomSampler(
            window_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        generator=generator if sampler is None else None,
        collate_fn=collate_recovery_sequences,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_recovery_sequences,
    )
    device = torch.device("cuda" if args.backend == "mamba2" else "cpu")
    model = RecoverySlotSSM(
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        backend=args.backend,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3
    )
    phase_weights = phase_weights.to(device)
    iterator = iter(train_loader)
    history = []
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        sequence = {
            key: value.to(device) for key, value in batch["sequence"].items()
        }
        outputs = model(build_recovery_model_inputs(sequence))
        batch_losses = losses(outputs, sequence, phase_weights)
        optimizer.zero_grad(set_to_none=True)
        batch_losses["total"].backward()
        gradient_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if not gradient_finite:
            raise RuntimeError(f"non-finite gradient at step {step}")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step == args.steps or step % 25 == 0:
            history.append(
                {
                    "step": step,
                    **{name: float(value.detach()) for name, value in batch_losses.items()},
                    "grad_norm_before_clip": float(grad_norm),
                }
            )
    model.eval()
    validation = evaluate(model, validation_loader, device, phase_weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "recovery-slotssm-smoke.safetensors"
    save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in model.state_dict().items()
        },
        str(checkpoint),
    )
    report = {
        "kind": "shared_recovery_slotssm_smoke",
        "backend": args.backend,
        "model_dim": args.model_dim,
        "num_layers": args.num_layers,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "steps": args.steps,
        "phase_counts": phase_counts.tolist(),
        "phase_weights": phase_weights.detach().cpu().tolist(),
        "balanced_window_sampling": args.balanced_window_sampling,
        "window_weight_min": float(window_weights.min()),
        "window_weight_max": float(window_weights.max()),
        "history": history,
        "validation": validation,
        "checkpoint": str(checkpoint.resolve()),
        "claim_limit": (
            "training-path smoke only; labels are legacy phase/controller labels "
            "and this is not an online-qualified recovery policy"
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
