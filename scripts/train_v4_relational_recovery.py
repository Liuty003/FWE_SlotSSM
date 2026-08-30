#!/usr/bin/env python3
"""Train a V4 corrective adapter through a frozen OFT action decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.data.v4_relation_features import V4RelationFeatures, load
from recovery.models.v4_relational_recovery import V4ModelConfig, V4RelationalRecovery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--action-head", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Optional V4 safetensors state, e.g. Stage-1 Slot initialization for joint training.")
    parser.add_argument("--oft-root", type=Path, default=Path("/data/lty/code/openvla-oft-official"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representation", choices=("global", "slot", "relation_2d", "relation_3d"), default="relation_3d")
    parser.add_argument("--stage", choices=("slot_init", "joint"), default="joint",
                        help="V4.2 schedule: train Slot masks first, then jointly train gate and recovery residual.")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--lambda-gate", type=float, default=1.0)
    parser.add_argument("--lambda-recovery", type=float, default=1.0)
    parser.add_argument("--recovery-deviation-weight", type=float, default=0.0,
                        help="Upweight recovery chunks that differ strongly from frozen nominal OFT actions.")
    parser.add_argument("--lambda-slot", type=float, default=0.0)
    parser.add_argument("--gate-label-smoothing", type=float, default=0.0)
    parser.add_argument("--residual-scale", type=float, default=0.10)
    parser.add_argument("--num-slots", type=int, default=16)
    parser.add_argument("--slot-dim", type=int, default=512)
    parser.add_argument("--relation-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def config_from_features(features: V4RelationFeatures, args: argparse.Namespace) -> V4ModelConfig:
    return V4ModelConfig(
        visual_dim=int(features.visual_tokens.shape[-1]), action_hidden_dim=int(features.action_hidden.shape[-1]),
        action_chunk=int(features.actions.shape[-2]), action_dim=int(features.actions.shape[-1]),
        num_slots=args.num_slots, slot_dim=args.slot_dim, relation_dim=args.relation_dim,
        num_heads=args.num_heads, residual_scale=args.residual_scale, representation=args.representation,
    )


def check_split(train: V4RelationFeatures, val: V4RelationFeatures) -> None:
    for key in ("visual_tokens", "action_hidden", "token_points", "actions"):
        if getattr(train, key).shape[1:] != getattr(val, key).shape[1:]:
            raise ValueError(f"train/validation {key} shapes differ")
    overlap = set(train.source_ids.tolist()) & set(val.source_ids.tolist())
    if overlap:
        raise ValueError(f"source leakage: {len(overlap)} physical sources are in both splits")


def check_provenance(train_path: Path, val_path: Path, action_head: Path) -> dict:
    reports = []
    for path in (train_path, val_path):
        report_path = path.with_suffix(".json")
        if not report_path.is_file():
            raise FileNotFoundError(f"missing feature provenance: {report_path}")
        reports.append(json.loads(report_path.read_text()))
    keys = ("oft_checkpoint", "action_head_sha256", "proprio_projector_sha256",
            "action_normalization_key", "recovery_target_coordinates",
            "decoder_action_contract", "image_preprocessing", "decision_grid_contract",
            "recovery_target_contract")
    if any(reports[0].get(key) != reports[1].get(key) for key in keys):
        raise ValueError("train/validation feature provenance differs")
    if reports[0].get("action_head_sha256") != sha256(action_head):
        raise ValueError("frozen decoder does not match feature action-head SHA-256")
    if reports[0].get("recovery_target_coordinates") != "checkpoint-normalized action coordinates":
        raise ValueError("recovery targets are not in frozen decoder coordinates")
    if reports[0].get("decoder_action_contract") != \
            "pre-process_action planned coordinates; executed gripper inverted by (1-g)/2":
        raise ValueError("feature actions do not match the frozen decoder's pre-process_action contract")
    if reports[0].get("image_preprocessing") != \
            "official OFT JPEG+TF-Lanczos3+sqrt(0.9)-crop_and_resize":
        raise ValueError("feature images were not processed by the exact official OFT pipeline")
    if reports[0].get("decision_grid_contract") != \
            "nominal grid from zero; recovery grid restarts at fork_timestep":
        raise ValueError("recovery features are not aligned to the actual fork decision grid")
    if reports[0].get("recovery_target_contract") != \
            "successful RECOVERY rows use canonical executed chunks when explicit target is absent":
        raise ValueError("recovery features may omit the canonical first post-fork target chunk")
    return {key: reports[0][key] for key in keys}


def decoder(oft_root: Path, checkpoint: Path, hidden_dim: int, action_dim: int, device):
    import torch

    sys.path.insert(0, str(oft_root))
    from prismatic.models.action_heads import L1RegressionActionHead

    head = L1RegressionActionHead(input_dim=hidden_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    head.load_state_dict(state)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_inputs(batch: V4RelationFeatures) -> dict:
    return {key: getattr(batch, key) for key in (
        "visual_tokens", "action_hidden", "token_points", "token_valid", "token_image_xy",
        "gripper_position", "gripper_image_xy",
    )}


def losses(model: V4RelationalRecovery, action_decoder, batch: V4RelationFeatures, args):
    import torch
    import torch.nn.functional as F

    outputs = model(**model_inputs(batch))
    slot_init = args.stage == "slot_init"
    auxiliary = model.losses(
        outputs, gate_labels=batch.gate_labels, instance_labels=batch.instance_labels,
        lambda_gate=0.0 if slot_init else args.lambda_gate, lambda_slot=args.lambda_slot,
        gate_label_smoothing=args.gate_label_smoothing,
    )
    if slot_init:
        # Stage 1 is deliberately restricted to simulator-mask supervision.
        # It establishes object-centric slots before task-conditioned gate and
        # residual gradients are allowed to shape the representation.
        recovery = outputs["gate_logit"].sum() * 0.0
        return auxiliary["total"], {
            "total": auxiliary["total"], "gate": auxiliary["gate"], "slot": auxiliary["slot"],
            "recovery": recovery,
        }, outputs
    # Ground-truth routing during training prevents an immature gate from
    # suppressing recovery gradients.  Runtime uses hard_route instead.
    routed_hidden = model.teacher_route(batch.action_hidden, outputs["recovery_action_hidden"], batch.gate_labels)
    predicted = action_decoder.predict_action(routed_hidden)
    if batch.recovery_mask.any():
        selected_prediction = predicted[batch.recovery_mask].float()
        selected_target = batch.recovery_targets[batch.recovery_mask].float()
        row_error = (selected_prediction - selected_target).abs().mean(dim=(1, 2))
        if args.recovery_deviation_weight > 0:
            # The critical post-fork chunks are precisely those for which the
            # scripted teacher must move far away from frozen nominal OFT.
            # A plain elementwise mean lets numerous easy late-stage nudges
            # hide failure on those large, causal recovery motions.
            with torch.no_grad():
                nominal_prediction = action_decoder.predict_action(
                    batch.action_hidden[batch.recovery_mask],
                ).float()
                difficulty = (nominal_prediction - selected_target).abs().mean(dim=(1, 2))
                weights = 1.0 + args.recovery_deviation_weight * difficulty
            recovery = (row_error * weights).sum() / weights.sum()
        else:
            recovery = row_error.mean()
    else:
        recovery = predicted.sum() * 0.0
    total = auxiliary["total"] + args.lambda_recovery * recovery
    return total, {"total": total, "gate": auxiliary["gate"], "slot": auxiliary["slot"], "recovery": recovery}, outputs


def evaluate(model, action_decoder, features, args, device):
    import torch

    model.eval()
    sums = {key: 0.0 for key in ("gate", "slot", "recovery")}
    counts = {key: 0 for key in sums}
    with torch.inference_mode():
        for start in range(0, features.num_rows, args.batch_size):
            index = torch.arange(start, min(start + args.batch_size, features.num_rows))
            batch = features.index(index).to(device)
            _, values, _ = losses(model, action_decoder, batch, args)
            weights = {
                "gate": int(batch.gate_labels.ge(0).sum()),
                "recovery": int(batch.recovery_mask.sum()),
                "slot": 0 if batch.instance_labels is None else sum(
                    int(row.unique().gt(0).sum()) for row in batch.instance_labels
                ),
            }
            for key, weight in weights.items():
                if weight:
                    sums[key] += float(values[key]) * weight
                    counts[key] += weight
    means = {key: sums[key] / max(counts[key], 1) for key in sums}
    means["total"] = (
        (0.0 if args.stage == "slot_init" else args.lambda_gate * means["gate"])
        + args.lambda_slot * means["slot"]
        + (0.0 if args.stage == "slot_init" else args.lambda_recovery * means["recovery"])
    )
    return {key: means[key] for key in ("total", "gate", "slot", "recovery")}


def main() -> None:
    args = parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.save_every) < 1:
        raise ValueError("steps, batch-size, eval-every, and save-every must be positive")
    if args.recovery_deviation_weight < 0:
        raise ValueError("--recovery-deviation-weight must be nonnegative")
    if args.stage == "slot_init" and args.lambda_slot <= 0:
        raise ValueError("slot_init requires --lambda-slot > 0")
    for path in (args.train_features, args.val_features, args.action_head, args.oft_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.init_checkpoint is not None and not args.init_checkpoint.is_file():
        raise FileNotFoundError(args.init_checkpoint)
    import torch
    from safetensors.torch import load_file, save_file

    torch.manual_seed(args.seed)
    provenance = check_provenance(args.train_features, args.val_features, args.action_head)
    train, val = load(args.train_features), load(args.val_features)
    check_split(train, val)
    config = config_from_features(train, args)
    model = V4RelationalRecovery(config).to(args.device)
    if args.init_checkpoint is not None:
        model.load_state_dict(load_file(str(args.init_checkpoint), device=str(args.device)), strict=True)
    action_decoder = decoder(args.oft_root, args.action_head, config.action_hidden_dim, config.action_dim, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps({
        "kind": "v4_relational_recovery", "model": asdict(config), "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "frozen_decoder": str(args.action_head), "frozen_decoder_sha256": sha256(args.action_head),
        "init_checkpoint": None if args.init_checkpoint is None else str(args.init_checkpoint),
        "deployment_route": "hard gate; below threshold returns exact frozen OFT hidden/action",
        "feature_provenance": provenance,
    }, indent=2) + "\n")
    best = {"total": float("inf"), "gate": float("inf"), "recovery": float("inf")}
    best_steps = {key: None for key in best}
    history = []
    for step in range(1, args.steps + 1):
        index = torch.randint(train.num_rows, (args.batch_size,))
        batch = train.index(index).to(args.device)
        total, values, _ = losses(model, action_decoder, batch, args)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0:
            validation = evaluate(model, action_decoder, val, args, args.device)
            row = {"step": step, "train": {key: float(value.detach()) for key, value in values.items()}, "val": validation}
            history.append(row)
            print(json.dumps(row), flush=True)
            state = None
            for metric in best:
                if validation[metric] < best[metric]:
                    best[metric], best_steps[metric] = validation[metric], step
                    if state is None:
                        state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
                    save_file(state, str(args.output_dir / f"best-{metric}.safetensors"))
                    # Preserve the original deployment filename as an alias for
                    # the checkpoint selected by the full validation objective.
                    if metric == "total":
                        save_file(state, str(args.output_dir / "best.safetensors"))
            model.train()
        if step % args.save_every == 0:
            save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(args.output_dir / "last.safetensors"))
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(args.output_dir / "last.safetensors"))
    (args.output_dir / "report.json").write_text(json.dumps({
        "kind": "v4_relational_recovery_training", "best_val": best,
        "best_steps": best_steps, "best_val_total": best["total"], "history": history,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
