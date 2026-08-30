#!/usr/bin/env python3
"""Train the two-stage v3 Action-Aware SlotSSM recovery policy.

Input files use :mod:`recovery.data.v3_sequence_features` and therefore hold
complete causal sequences of frozen OFT features.  This trainer deliberately
does not load or update OpenVLA-OFT; normal-policy features are immutable.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.data.v3_sequence_features import V3SequenceFeatures, load
from recovery.models.v3_action_aware_recovery import V3ActionAwareRecoveryPolicy, V3ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Stage-1 checkpoint used to initialise stage 2.")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--slotssm-lr-scale", type=float, default=0.2,
                        help="Stage-2 LR multiplier for SlotAttention/SlotSSM.")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--lambda-slot", type=float, default=1.0)
    parser.add_argument("--lambda-future", type=float, default=1.0)
    parser.add_argument("--lambda-gate", type=float, default=1.0)
    parser.add_argument("--lambda-recovery", type=float, default=1.0)
    parser.add_argument("--num-slots", type=int, default=16)
    parser.add_argument("--slot-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    return parser.parse_args()


def config_from_features(features: V3SequenceFeatures, args: argparse.Namespace) -> V3ModelConfig:
    return V3ModelConfig(
        visual_dim=int(features.visual_tokens.shape[-1]),
        action_hidden_dim=int(features.action_hidden.shape[-1]),
        proprio_dim=int(features.proprio.shape[-1]),
        action_chunk=int(features.actions.shape[-2]),
        action_dim=int(features.actions.shape[-1]),
        num_slots=args.num_slots, slot_dim=args.slot_dim, hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
    )


def check_compatible(train: V3SequenceFeatures, val: V3SequenceFeatures) -> None:
    for key in ("visual_tokens", "action_hidden", "proprio", "actions"):
        if getattr(train, key).shape[2:] != getattr(val, key).shape[2:]:
            raise ValueError(f"train/val {key} feature shape differs")
    if train.source_ids is not None and val.source_ids is not None:
        overlap = set(train.source_ids.tolist()) & set(val.source_ids.tolist())
        if overlap:
            raise ValueError(f"source leakage: {len(overlap)} source_ids occur in both train and val")


def model_state(model: V3ActionAwareRecoveryPolicy, *, deploy_only: bool) -> dict:
    state = model.state_dict()
    if deploy_only:
        state = {key: value for key, value in state.items()
                 if not key.startswith("future_predictor.") and not key.startswith("slot_decoder.")}
    return {key: value.detach().cpu().contiguous() for key, value in state.items()}


def unique_parameters(*groups):
    """Keep shared action encoders in one optimizer parameter group."""
    seen = set()
    result = []
    for group in groups:
        for parameter in group:
            if id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return result


def jsonable_args(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def evaluate(model: V3ActionAwareRecoveryPolicy, features: V3SequenceFeatures, args: argparse.Namespace,
             device, gate_positive_weight: float):
    import torch

    model.eval()
    totals = {key: 0.0 for key in ("total", "slot", "future", "gate", "recovery")}
    batches = 0
    with torch.inference_mode():
        for start in range(0, features.num_sequences, args.batch_size):
            indices = torch.arange(start, min(start + args.batch_size, features.num_sequences))
            batch = features.index(indices).to(device)
            outputs = model.forward_sequence(
                visual_tokens=batch.visual_tokens, action_hidden=batch.action_hidden,
                proprio=batch.proprio, actions=batch.actions,
            )
            losses = V3ActionAwareRecoveryPolicy.losses(
                outputs, visual_tokens=batch.visual_tokens,
                # Stage 1 is representation-only.  Including untrained gate
                # and recovery losses here would select a checkpoint based on
                # random downstream heads rather than action-effect quality.
                gate_labels=None if args.stage == "stage1" else batch.gate_labels,
                recovery_targets=None if args.stage == "stage1" else batch.recovery_targets,
                recovery_mask=None if args.stage == "stage1" else batch.recovery_mask,
                lambda_slot=args.lambda_slot, lambda_future=args.lambda_future,
                lambda_gate=args.lambda_gate, lambda_recovery=args.lambda_recovery,
                gate_positive_weight=gate_positive_weight,
            )
            for key, value in losses.items():
                totals[key] += float(value)
            batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1 or args.save_every < 1:
        raise ValueError("steps, batch-size, eval-every and save-every must be positive")
    if args.stage == "stage2" and args.init_checkpoint is None:
        raise ValueError("stage2 requires --init-checkpoint from stage1")
    import torch
    from safetensors.torch import load_file, save_file

    torch.manual_seed(args.seed)
    train, val = load(args.train_features), load(args.val_features)
    check_compatible(train, val)
    config = config_from_features(train, args)
    model = V3ActionAwareRecoveryPolicy(config).to(args.device)
    if args.init_checkpoint is not None:
        result = model.load_state_dict(load_file(str(args.init_checkpoint)), strict=False)
        unexpected = [key for key in result.unexpected_keys if not key.startswith(("future_predictor.", "slot_decoder."))]
        if unexpected:
            raise RuntimeError(f"unexpected checkpoint keys: {unexpected}")
    if args.stage == "stage1":
        parameters = unique_parameters(
            model.slot_attention.parameters(), model.slot_ssm.parameters(),
            model.slot_decoder.parameters(), model.future_predictor.parameters(),
        )
    else:
        representation = unique_parameters(model.slot_attention.parameters(), model.slot_ssm.parameters())
        downstream = unique_parameters(
            model.gate.parameters(), model.recovery_head.parameters(), model.slot_decoder.parameters(),
            model.future_predictor.parameters(),
        )
        representation_ids = {id(parameter) for parameter in representation}
        downstream = [parameter for parameter in downstream if id(parameter) not in representation_ids]
        parameters = [
            {"params": representation, "lr": args.lr * args.slotssm_lr_scale},
            {"params": downstream, "lr": args.lr},
        ]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    labels = train.gate_labels
    positives = 0 if labels is None else int(labels.eq(1).sum())
    negatives = 0 if labels is None else int(labels.eq(0).sum())
    gate_positive_weight = float(negatives / max(positives, 1)) if positives else 1.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps({
        "kind": "v3_action_aware_slotssm", "stage": args.stage,
        "model": asdict(config), "train_features": str(args.train_features),
        "val_features": str(args.val_features), "args": jsonable_args(args),
        "gate_positive_weight": gate_positive_weight,
        "deployment_contract": ["frozen_oft_visual_tokens", "frozen_oft_action_hidden", "proprio", "previous_executed_action"],
    }, indent=2) + "\n")
    best = float("inf")
    history = []
    model.train()
    for step in range(1, args.steps + 1):
        index = torch.randint(train.num_sequences, (args.batch_size,))
        batch = train.index(index).to(args.device)
        outputs = model.forward_sequence(
            visual_tokens=batch.visual_tokens, action_hidden=batch.action_hidden,
            proprio=batch.proprio, actions=batch.actions,
        )
        losses = V3ActionAwareRecoveryPolicy.losses(
            outputs, visual_tokens=batch.visual_tokens,
            gate_labels=None if args.stage == "stage1" else batch.gate_labels,
            recovery_targets=None if args.stage == "stage1" else batch.recovery_targets,
            recovery_mask=None if args.stage == "stage1" else batch.recovery_mask,
            lambda_slot=args.lambda_slot, lambda_future=args.lambda_future,
            lambda_gate=args.lambda_gate, lambda_recovery=args.lambda_recovery,
            gate_positive_weight=gate_positive_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0:
            metrics = evaluate(model, val, args, args.device, gate_positive_weight)
            row = {"step": step, "train": {key: float(value.detach()) for key, value in losses.items()}, "val": metrics}
            history.append(row)
            print(json.dumps(row), flush=True)
            if metrics["total"] < best:
                best = metrics["total"]
                save_file(model_state(model, deploy_only=False), str(args.output_dir / "train_best.safetensors"))
                save_file(model_state(model, deploy_only=True), str(args.output_dir / "deploy_best.safetensors"))
            model.train()
        if step % args.save_every == 0:
            save_file(model_state(model, deploy_only=False), str(args.output_dir / "train_last.safetensors"))
            save_file(model_state(model, deploy_only=True), str(args.output_dir / "deploy_last.safetensors"))
    save_file(model_state(model, deploy_only=False), str(args.output_dir / "train_last.safetensors"))
    save_file(model_state(model, deploy_only=True), str(args.output_dir / "deploy_last.safetensors"))
    report = {"kind": "v3_action_aware_slotssm_training", "best_val_total": best, "history": history}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
