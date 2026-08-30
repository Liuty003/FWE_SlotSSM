#!/usr/bin/env python3
"""Train the independent V4.4.2 resume-value head on certified handoffs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.data.v4_relation_features import load
from recovery.models.v44_dual_ownership_recovery import V44DualOwnershipRecovery
from scripts.train_v43_local_recovery import config_from, model_inputs, provenance


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-features", type=Path, required=True)
    parser.add_argument("--boundary-features", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representation", choices=("global", "slot"), default="slot")
    parser.add_argument("--recovery-output-horizon", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--tail-negative-steps", type=int, default=24)
    parser.add_argument("--num-slots", type=int, default=16)
    parser.add_argument("--slot-dim", type=int, default=512)
    parser.add_argument("--relation-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def tail_negative_indices(features, tail_steps: int, torch):
    indices = []
    for source_id in features.source_ids.unique():
        source = torch.where(features.source_ids.eq(source_id))[0]
        indices.append(source[-tail_steps:])
    return torch.cat(indices)


def validate_resume_feature_contract(recovery_report: dict, boundary_report: dict) -> None:
    handoff = recovery_report.get("handoff_manifest")
    if not handoff or boundary_report.get("handoff_manifest") != handoff:
        raise ValueError("resume features must share one explicit certified handoff manifest")
    if recovery_report.get("fixed_recovery_window_steps") is not None:
        raise ValueError("resume negatives cannot use an unaudited fixed recovery window")
    if recovery_report.get("include_handoff_gate_rows") is True:
        raise ValueError("recovery negatives must exclude the certified handoff boundary")
    if boundary_report.get("include_handoff_gate_rows") is not True:
        raise ValueError("boundary features must include one exact certified handoff row")
    for name, report in (("recovery", recovery_report), ("boundary", boundary_report)):
        if report.get("recovery_sampling_stride") != 1:
            raise ValueError(f"{name} features must use V4.4.2 single-step recovery sampling")
        if report.get("recovery_target_horizon") != 1:
            raise ValueError(f"{name} features must use a one-action recovery target horizon")


def initialize_resume_as_gate_complement(model) -> None:
    """Start with p(resume)=1-p(enter) without tying either head."""
    model.resume_head[0].load_state_dict(model.gate[0].state_dict())
    model.resume_head[1].load_state_dict(model.gate[1].state_dict())
    model.resume_head[3].weight.data.copy_(-model.gate[3].weight.data)
    model.resume_head[3].bias.data.copy_(-model.gate[3].bias.data)


def evaluate(model, negative, positive, batch_size: int, device: str, torch):
    import torch.nn.functional as F

    model.eval()
    values = {}
    with torch.inference_mode():
        for name, features, target in (("negative", negative, 0.0), ("positive", positive, 1.0)):
            losses, probabilities = [], []
            for start in range(0, features.num_rows, batch_size):
                index = torch.arange(start, min(start + batch_size, features.num_rows))
                output = model(**model_inputs(features.index(index).to(device)))
                logits = output["resume_logit"].float()
                labels = torch.full_like(logits, target)
                losses.append(F.binary_cross_entropy_with_logits(logits, labels, reduction="sum"))
                probabilities.append(output["resume_probability"].float())
            probability = torch.cat(probabilities)
            values[f"{name}_loss"] = float(torch.stack(losses).sum() / features.num_rows)
            values[f"{name}_accuracy"] = float(
                (probability < 0.5).float().mean() if target == 0 else (probability >= 0.5).float().mean()
            )
            values[f"{name}_probability_mean"] = float(probability.mean())
            values[f"{name}_probability_min"] = float(probability.min())
            values[f"{name}_probability_max"] = float(probability.max())
    values["balanced_loss"] = 0.5 * (values["negative_loss"] + values["positive_loss"])
    return values


def main() -> None:
    args = arguments()
    if min(args.steps, args.batch_size, args.eval_every, args.save_every, args.tail_negative_steps) < 1:
        raise ValueError("training and sampling arguments must be positive")
    for path in (args.recovery_features, args.boundary_features, args.init_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file

    torch.manual_seed(args.seed)
    recovery_report = provenance(args.recovery_features)
    boundary_report = provenance(args.boundary_features)
    validate_resume_feature_contract(recovery_report, boundary_report)
    if not recovery_report.get("native_input_only") or not boundary_report.get("native_input_only"):
        raise ValueError("resume-value training requires native-input-only features")
    negative = load(args.recovery_features)
    boundary = load(args.boundary_features)
    if not bool(negative.gate_labels.eq(1).all() and negative.recovery_mask.all()):
        raise ValueError("resume-negative recovery features must contain only pre-handoff teacher rows")
    positive_index = torch.where(boundary.gate_labels.eq(0))[0]
    if len(positive_index) != len(boundary.source_ids.unique()):
        raise ValueError("boundary features require exactly one certified resume-positive row per source")
    positive = boundary.index(positive_index)
    negative_sources = set(negative.source_ids.tolist())
    positive_sources = set(positive.source_ids.tolist())
    if negative_sources != positive_sources:
        raise ValueError("resume positive and negative features must cover exactly the same sources")
    if positive.num_rows < 2 or negative.num_rows < 2:
        raise ValueError("resume training requires positive and negative examples")

    cfg = config_from(negative, args)
    model = V44DualOwnershipRecovery(
        cfg, direct_head_version="pooled_chunk_mlp_v3",
        recovery_output_horizon=args.recovery_output_horizon,
    ).to(args.device)
    state = load_file(str(args.init_checkpoint), device=args.device)
    result = model.load_state_dict(state, strict=False)
    expected_missing = {name for name in model.state_dict() if name.startswith("resume_head.")}
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise ValueError(f"incompatible initialization: {result}")
    initialize_resume_as_gate_complement(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("resume_head."))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    tail_negative = tail_negative_indices(negative, args.tail_negative_steps, torch)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "kind": "v44_dual_ownership_recovery_v1",
        "model": asdict(cfg),
        "direct_head_version": "pooled_chunk_mlp_v3",
        "recovery_output_horizon": args.recovery_output_horizon,
        "enter_risk_head": "gate",
        "resume_value_head": "resume_head",
        "resume_target": "fresh-replan OFT success at certified teacher handoff",
        "normal_policy": "exact frozen OFT action bypass outside the recovery latch",
        "feature_provenance": {key: recovery_report.get(key) for key in (
            "oft_checkpoint", "action_head_sha256", "proprio_projector_sha256",
            "action_normalization_key", "recovery_target_coordinates", "decoder_action_contract",
            "image_preprocessing", "decision_grid_contract", "recovery_target_contract",
            "native_input_only", "representation_input",
        )},
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "negative_rows": negative.num_rows,
        "tail_negative_rows": int(len(tail_negative)),
        "positive_rows": positive.num_rows,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    best, history = float("inf"), []
    model.train()
    for step in range(1, args.steps + 1):
        positive_count = args.batch_size // 2
        positive_rows = torch.randint(positive.num_rows, (positive_count,))
        negative_count = args.batch_size - positive_count
        # Certified-tail negatives are the hard distinction immediately before
        # resume. Keep a smaller global-negative component for calibration.
        tail_count = max(1, 3 * negative_count // 4)
        negative_rows = torch.cat((
            tail_negative[torch.randint(len(tail_negative), (tail_count,))],
            torch.randint(negative.num_rows, (negative_count - tail_count,)),
        ))
        positive_output = model(**model_inputs(positive.index(positive_rows).to(args.device)))
        negative_output = model(**model_inputs(negative.index(negative_rows).to(args.device)))
        positive_loss = F.binary_cross_entropy_with_logits(
            positive_output["resume_logit"].float(),
            torch.ones_like(positive_output["resume_logit"], dtype=torch.float32),
        )
        negative_loss = F.binary_cross_entropy_with_logits(
            negative_output["resume_logit"].float(),
            torch.zeros_like(negative_output["resume_logit"], dtype=torch.float32),
        )
        loss = 0.5 * (positive_loss + negative_loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0:
            validation = evaluate(model, negative, positive, args.batch_size, args.device, torch)
            row = {"step": step, "train_loss": float(loss.detach()), "val": validation}
            print(json.dumps(row), flush=True)
            history.append(row)
            if validation["balanced_loss"] < best:
                best = validation["balanced_loss"]
                save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
                          str(args.output_dir / "best.safetensors"))
            model.train()
        if step % args.save_every == 0:
            save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
                      str(args.output_dir / "last.safetensors"))
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
              str(args.output_dir / "last.safetensors"))
    (args.output_dir / "report.json").write_text(json.dumps({
        "best_val_balanced_loss": best, "history": history,
        "evaluation_scope": "in_sample_fixed_certified_handoffs",
        "negative_rows": negative.num_rows, "positive_rows": positive.num_rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
