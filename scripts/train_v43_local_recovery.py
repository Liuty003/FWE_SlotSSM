#!/usr/bin/env python3
"""Train the V4.3 gate and direct local-recovery action head."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.data.v4_relation_features import REQUIRED_KEYS, V4RelationFeatures, load
from recovery.models.v4_relational_recovery import V4ModelConfig
from recovery.models.v43_local_recovery import V43LocalRecovery
from recovery.models.v43_detailed_phase_recovery import (
    DETAILED_PHASE_NAMES,
    V43DetailedPhaseRecovery,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--train-normal-features", type=Path,
                        help="Optional NORMAL-only shard; kept separate because its sampling contract differs.")
    parser.add_argument("--val-normal-features", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-kind", choices=("local", "detailed"), default="local",
                        help="Architecture to preserve while training; detailed keeps all auxiliary heads.")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Optional V4/V4.3 checkpoint; V4 direct-head keys are initialized fresh.")
    parser.add_argument("--representation", choices=("global", "slot", "relation_2d", "relation_3d"),
                        default="relation_3d")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--lambda-gate", type=float, default=1.0)
    parser.add_argument("--lambda-recovery", type=float, default=1.0)
    parser.add_argument("--lambda-slot", type=float, default=0.0)
    parser.add_argument("--gate-label-smoothing", type=float, default=0.0)
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--recovery-output-horizon", type=int,
        help=("Number of recovery actions predicted per decision. Omit to preserve the "
              "V4.3 feature/action horizon; V4.4.2 uses 1."),
    )
    parser.add_argument("--gate-only-finetune", action="store_true",
                        help="Freeze the relational representation and recovery head; update only the ownership gate.")
    parser.add_argument(
        "--recovery-head-only-finetune", action="store_true",
        help=("Freeze the relational representation and ownership gate; update only direct_action. "
              "This preserves the initialized competence scores exactly while expanding recovery actions."),
    )
    parser.add_argument("--num-slots", type=int, default=16)
    parser.add_argument("--slot-dim", type=int, default=512)
    parser.add_argument("--relation-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--allow-in-sample-validation", action="store_true",
        help=("Permit train/validation source overlap for fixed-task memorization. "
              "Reported validation metrics are then explicitly in-sample."),
    )
    return parser.parse_args()


def config_from(features: V4RelationFeatures, args: argparse.Namespace) -> V4ModelConfig:
    return V4ModelConfig(
        visual_dim=int(features.visual_tokens.shape[-1]),
        action_hidden_dim=int(features.action_hidden.shape[-1]),
        action_chunk=int(features.actions.shape[-2]),
        action_dim=int(features.actions.shape[-1]),
        num_slots=args.num_slots, slot_dim=args.slot_dim, relation_dim=args.relation_dim,
        num_heads=args.num_heads, representation=args.representation,
    )


def model_inputs(batch: V4RelationFeatures) -> dict:
    return {key: getattr(batch, key) for key in (
        "visual_tokens", "action_hidden", "token_points", "token_valid", "token_image_xy",
        "gripper_position", "gripper_image_xy",
    )}


def check_data(train: V4RelationFeatures, val: V4RelationFeatures, *, allow_overlap: bool = False,
               require_recovery: bool = True) -> None:
    for key in ("visual_tokens", "action_hidden", "token_points", "recovery_targets"):
        if getattr(train, key).shape[1:] != getattr(val, key).shape[1:]:
            raise ValueError(f"train/validation {key} shapes differ")
    overlap = set(train.source_ids.tolist()) & set(val.source_ids.tolist())
    if overlap and not allow_overlap:
        raise ValueError(f"source leakage: {len(overlap)} physical sources occur in both splits")
    if require_recovery and (not train.recovery_mask.any() or not val.recovery_mask.any()):
        raise ValueError("both splits require recovery action rows")
    if not require_recovery and any(
            not features.gate_labels.eq(label).any()
            for features in (train, val) for label in (0, 1)):
        raise ValueError("gate-only train/validation splits require both ownership labels")


def adapt_short_horizon_state(init_state: dict, model_state: dict, output_tokens: int) -> dict:
    """Preserve a pooled V4.3 head and truncate only its final output rows.

    Row-major flattening stores action 0 in the first ``action_dim`` rows, so
    this makes a one-step model exactly reproduce the archived first action at
    initialization instead of silently reinitializing the entire MLP.
    """
    adapted = dict(init_state)
    final_weight, final_bias = "direct_action.3.weight", "direct_action.3.bias"
    for key in (final_weight, final_bias):
        if key not in adapted or key not in model_state:
            raise ValueError(f"short-horizon initialization requires {key}")
    source_weight, target_weight = adapted[final_weight], model_state[final_weight]
    source_bias, target_bias = adapted[final_bias], model_state[final_bias]
    if (source_weight.ndim != 2 or source_bias.ndim != 1
            or source_weight.shape[1:] != target_weight.shape[1:]
            or source_weight.shape[0] < output_tokens or source_bias.shape[0] < output_tokens
            or target_weight.shape[0] != output_tokens or target_bias.shape[0] != output_tokens):
        raise ValueError("checkpoint is not a compatible pooled_chunk_mlp_v3 action head")
    adapted[final_weight] = source_weight[:output_tokens].clone()
    adapted[final_bias] = source_bias[:output_tokens].clone()
    return adapted


def concatenate(parts: list[V4RelationFeatures], torch) -> V4RelationFeatures:
    if len(parts) == 1:
        return parts[0]
    target_horizon = parts[0].recovery_targets.shape[1]
    normalized_parts = []
    for part in parts:
        if part.recovery_targets.shape[1] != target_horizon:
            if part.recovery_mask.any():
                raise ValueError("recovery feature shards must have one target horizon")
            if part.recovery_targets.shape[1] < target_horizon:
                raise ValueError("a NORMAL shard has fewer target placeholders than the recovery shard")
            values = part.payload()
            values["recovery_targets"] = part.recovery_targets[:, :target_horizon]
            part = V4RelationFeatures(**values)
        normalized_parts.append(part)
    parts = normalized_parts
    payload = {key: torch.cat([getattr(part, key) for part in parts], dim=0) for key in REQUIRED_KEYS}
    if all(part.instance_labels is not None for part in parts):
        payload["instance_labels"] = torch.cat([part.instance_labels for part in parts], dim=0)
    elif any(part.instance_labels is not None for part in parts):
        raise ValueError("all feature shards must agree on instance-label availability")
    return V4RelationFeatures(**payload)


def provenance(path: Path) -> dict:
    report_path = path.with_suffix(".json")
    if not report_path.is_file():
        raise FileNotFoundError(f"missing feature provenance: {report_path}")
    report = json.loads(report_path.read_text())
    if report.get("recovery_target_coordinates") != "checkpoint-normalized action coordinates":
        raise ValueError("V4.3 direct targets must use checkpoint-normalized action coordinates")
    return report


def objective(model, batch: V4RelationFeatures, args):
    import torch

    outputs = model(**model_inputs(batch))
    auxiliary = model.losses(
        outputs, gate_labels=batch.gate_labels, instance_labels=batch.instance_labels,
        lambda_gate=args.lambda_gate, lambda_slot=args.lambda_slot,
        gate_label_smoothing=args.gate_label_smoothing,
    )
    action_weights = torch.ones(batch.recovery_targets.shape[-1], device=batch.recovery_targets.device)
    action_weights[-1] = args.gripper_loss_weight
    recovery = model.recovery_loss(
        outputs, batch.recovery_targets, batch.recovery_mask,
        action_dimension_weights=action_weights,
    )
    if args.gate_only_finetune:
        total = auxiliary["total"]
    elif args.recovery_head_only_finetune:
        # Frozen gate/representation metrics remain useful diagnostics, but a
        # constant frozen loss must not obscure the action-only objective.
        total = args.lambda_recovery * recovery
    else:
        total = auxiliary["total"] + args.lambda_recovery * recovery
    return total, {"total": total, "gate": auxiliary["gate"], "slot": auxiliary["slot"],
                   "recovery": recovery}


def balanced_indices(features: V4RelationFeatures, count: int, torch) -> object:
    positive = torch.where(features.recovery_mask)[0]
    other = torch.where(~features.recovery_mask)[0]
    positive_count = count if not len(other) else max(1, count // 2)
    parts = [positive[torch.randint(len(positive), (positive_count,))]]
    if positive_count < count:
        parts.append(other[torch.randint(len(other), (count - positive_count,))])
    return torch.cat(parts)[torch.randperm(count)]


def balanced_gate_indices(features: V4RelationFeatures, count: int, torch) -> object:
    positive = torch.where(features.gate_labels.eq(1))[0]
    negative = torch.where(features.gate_labels.eq(0))[0]
    if not len(positive) or not len(negative):
        raise ValueError("gate-only fine-tuning requires both ownership labels")
    positive_count = count // 2
    parts = [positive[torch.randint(len(positive), (positive_count,))],
             negative[torch.randint(len(negative), (count - positive_count,))]]
    return torch.cat(parts)[torch.randperm(count)]


def recovery_indices(features: V4RelationFeatures, count: int, torch) -> object:
    recovery = torch.where(features.recovery_mask)[0]
    if not len(recovery):
        raise ValueError("recovery-head-only fine-tuning requires recovery targets")
    return recovery[torch.randint(len(recovery), (count,))]


def evaluate(model, features, args, torch) -> dict[str, float]:
    model.eval()
    weighted = {key: 0.0 for key in ("gate", "slot", "recovery")}
    counts = {key: 0 for key in weighted}
    with torch.inference_mode():
        for start in range(0, features.num_rows, args.batch_size):
            index = torch.arange(start, min(start + args.batch_size, features.num_rows))
            batch = features.index(index).to(args.device)
            _, values = objective(model, batch, args)
            row_counts = {
                "gate": int(batch.gate_labels.ge(0).sum()),
                "recovery": int(batch.recovery_mask.sum()),
                "slot": 0 if batch.instance_labels is None else int(batch.instance_labels.gt(0).sum()),
            }
            for key, count in row_counts.items():
                if count:
                    weighted[key] += float(values[key]) * count
                    counts[key] += count
    means = {key: weighted[key] / max(counts[key], 1) for key in weighted}
    if args.gate_only_finetune:
        means["total"] = args.lambda_gate * means["gate"] + args.lambda_slot * means["slot"]
    elif args.recovery_head_only_finetune:
        means["total"] = args.lambda_recovery * means["recovery"]
    else:
        means["total"] = (args.lambda_gate * means["gate"] + args.lambda_slot * means["slot"]
                          + args.lambda_recovery * means["recovery"])
    return {key: means[key] for key in ("total", "gate", "slot", "recovery")}


def main() -> None:
    args = arguments()
    if min(args.steps, args.batch_size, args.eval_every, args.save_every) < 1:
        raise ValueError("step and batch arguments must be positive")
    if args.gripper_loss_weight <= 0:
        raise ValueError("--gripper-loss-weight must be positive")
    if args.recovery_output_horizon is not None and args.recovery_output_horizon < 1:
        raise ValueError("--recovery-output-horizon must be positive")
    if args.gate_only_finetune and args.recovery_head_only_finetune:
        raise ValueError("gate-only and recovery-head-only fine-tuning are mutually exclusive")
    if (args.train_normal_features is None) != (args.val_normal_features is None):
        raise ValueError("provide both --train-normal-features and --val-normal-features")
    feature_paths = [args.train_features, args.val_features]
    if args.train_normal_features is not None:
        feature_paths += [args.train_normal_features, args.val_normal_features]
    for path in feature_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.init_checkpoint is not None and not args.init_checkpoint.is_file():
        raise FileNotFoundError(args.init_checkpoint)

    import torch
    from safetensors.torch import load_file, save_file

    torch.manual_seed(args.seed)
    train_report, val_report = provenance(args.train_features), provenance(args.val_features)
    if train_report.get("native_input_only") != val_report.get("native_input_only"):
        raise ValueError("train/validation native-input contracts differ")
    if train_report.get("native_input_only") is True and args.representation not in {"global", "slot"}:
        raise ValueError("native-input-only features require --representation global or slot")
    train_main = load(args.train_features)
    val_main = train_main if args.train_features.resolve() == args.val_features.resolve() else load(args.val_features)
    train_parts, val_parts = [train_main], [val_main]
    if args.train_normal_features is not None:
        provenance(args.train_normal_features)
        provenance(args.val_normal_features)
        train_normal = load(args.train_normal_features)
        val_normal = (train_normal if args.train_normal_features.resolve() == args.val_normal_features.resolve()
                      else load(args.val_normal_features))
        train_parts.append(train_normal)
        val_parts.append(val_normal)
    train = concatenate(train_parts, torch)
    same_dataset = (args.train_features.resolve() == args.val_features.resolve() and
                    (args.train_normal_features is None or
                     args.train_normal_features.resolve() == args.val_normal_features.resolve()))
    val = train if same_dataset else concatenate(val_parts, torch)
    check_data(train, val, allow_overlap=args.allow_in_sample_validation,
               require_recovery=not args.gate_only_finetune)
    cfg = config_from(train, args)
    effective_output_horizon = cfg.action_chunk if args.recovery_output_horizon is None else args.recovery_output_horizon
    if effective_output_horizon > train.recovery_targets.shape[-2]:
        raise ValueError(
            "recovery output horizon exceeds the feature recovery target horizon; "
            "pass --recovery-output-horizon for short-target features"
        )
    direct_head_version = "pooled_chunk_mlp_v3"
    if args.model_kind == "detailed" and args.recovery_output_horizon not in (None, cfg.action_chunk):
        raise ValueError("short-horizon ablation currently supports --model-kind local only")
    model = (V43DetailedPhaseRecovery(cfg) if args.model_kind == "detailed"
             else V43LocalRecovery(
                 cfg, direct_head_version=direct_head_version,
                 recovery_output_horizon=args.recovery_output_horizon,
             )).to(args.device)
    init_result = None
    if args.init_checkpoint is not None:
        init_state = load_file(str(args.init_checkpoint), device=str(args.device))
        # Preserve the complete pooled MLP and truncate only its output rows.
        # Thus a V4.4.2 H=1 initialization exactly equals the archived V4.3
        # first action on every input.
        if args.recovery_output_horizon not in (None, cfg.action_chunk):
            init_state = adapt_short_horizon_state(
                init_state, model.state_dict(), effective_output_horizon * cfg.action_dim,
            )
        init_result = model.load_state_dict(init_state, strict=False)
        if init_result.missing_keys or init_result.unexpected_keys:
            raise ValueError(f"incompatible init checkpoint: {init_result}")

    if args.gate_only_finetune or args.recovery_head_only_finetune:
        if args.init_checkpoint is None:
            raise ValueError("restricted fine-tuning requires --init-checkpoint")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith("gate.") if args.gate_only_finetune else name.startswith("direct_action.")
            )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    short_horizon = args.recovery_output_horizon not in (None, cfg.action_chunk)
    configuration = {
        "kind": ("v43_detailed_phase_recovery_v1" if args.model_kind == "detailed"
                 else "v44_local_closed_loop_recovery_v1" if short_horizon
                 else "v43_relational_local_recovery"), "model": asdict(cfg),
        "evaluation_scope": ("in_sample_fixed_tasks" if args.allow_in_sample_validation
                             else "source_disjoint_validation"),
        "direct_head_version": direct_head_version,
        "recovery_output_horizon": (
            cfg.action_chunk if args.recovery_output_horizon is None
            else args.recovery_output_horizon
        ),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "normal_policy": "exact frozen OFT action bypass outside the recovery latch",
        "recovery_policy": (
            f"direct normalized {cfg.action_chunk if args.recovery_output_horizon is None else args.recovery_output_horizon}x"
            f"{cfg.action_dim} actions from relational context plus OFT action query"
        ),
        "sampling": (
            "50% gate-positive and 50% gate-negative rows"
            if args.gate_only_finetune else
            "100% recovery-target rows"
            if args.recovery_head_only_finetune else
            "50% recovery-target and 50% non-recovery rows when both exist"
        ),
        "train_feature_report": str(args.train_features.with_suffix('.json')),
        "val_feature_report": str(args.val_features.with_suffix('.json')),
        "feature_contract_equal": train_report.get("recovery_target_contract") == val_report.get("recovery_target_contract"),
        "feature_provenance": {key: train_report.get(key) for key in (
            "oft_checkpoint", "action_head_sha256", "proprio_projector_sha256",
            "action_normalization_key", "recovery_target_coordinates", "decoder_action_contract",
            "image_preprocessing", "decision_grid_contract", "recovery_target_contract",
            "sampling_contract", "recovery_sampling_stride", "handoff_manifest",
            "recovery_target_horizon", "native_input_only", "representation_input",
        )},
        "init_missing_keys": [] if init_result is None else init_result.missing_keys,
    }
    if args.model_kind == "detailed":
        configuration["phase_names"] = DETAILED_PHASE_NAMES
    (args.output_dir / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")

    best = float("inf")
    history = []
    for step in range(1, args.steps + 1):
        indices = (balanced_gate_indices(train, args.batch_size, torch) if args.gate_only_finetune
                   else recovery_indices(train, args.batch_size, torch) if args.recovery_head_only_finetune
                   else balanced_indices(train, args.batch_size, torch))
        batch = train.index(indices).to(args.device)
        total, values = objective(model, batch, args)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0:
            validation = evaluate(model, val, args, torch)
            row = {"step": step, "train": {key: float(value.detach()) for key, value in values.items()},
                   "val": validation}
            print(json.dumps(row), flush=True)
            history.append(row)
            if validation["total"] < best:
                best = validation["total"]
                save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
                          str(args.output_dir / "best.safetensors"))
            model.train()
        if step % args.save_every == 0:
            save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
                      str(args.output_dir / "last.safetensors"))
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
              str(args.output_dir / "last.safetensors"))
    (args.output_dir / "report.json").write_text(json.dumps({
        "best_val_total": best, "history": history,
        "evaluation_scope": ("in_sample_fixed_tasks" if args.allow_in_sample_validation
                             else "source_disjoint_validation"),
        "train_rows": train.num_rows, "train_recovery_rows": int(train.recovery_mask.sum()),
        "val_rows": val.num_rows, "val_recovery_rows": int(val.recovery_mask.sum()),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
