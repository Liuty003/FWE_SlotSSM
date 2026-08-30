#!/usr/bin/env python3
"""Evaluate a V3 recovery checkpoint on source-disjoint frozen features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.data.v3_sequence_features import load
from recovery.models.v3_action_aware_recovery import V3ActionAwareRecoveryPolicy, V3ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Defaults to config.json beside checkpoint.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def binary_metrics(prediction, target) -> dict[str, float | int]:
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    tn = int((~prediction & ~target).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "fpr": fp / max(fp + tn, 1),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    import torch
    from safetensors.torch import load_file

    config_path = args.config or args.checkpoint.parent / "config.json"
    config = V3ModelConfig(**json.loads(config_path.read_text(encoding="utf-8"))["model"])
    model = V3ActionAwareRecoveryPolicy(config).to(args.device).eval()
    loaded = model.load_state_dict(load_file(str(args.checkpoint)), strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("v3 training checkpoint did not load strictly")
    features = load(args.features)
    if features.gate_labels is None or features.recovery_mask is None or features.recovery_targets is None:
        raise ValueError("evaluation requires gate labels and canonical recovery targets")

    probabilities, targets, action_l1 = [], [], []
    with torch.inference_mode():
        for start in range(0, features.num_sequences, args.batch_size):
            indices = torch.arange(start, min(start + args.batch_size, features.num_sequences))
            batch = features.index(indices).to(args.device)
            output = model.forward_sequence(
                visual_tokens=batch.visual_tokens, action_hidden=batch.action_hidden,
                proprio=batch.proprio, actions=batch.actions,
            )
            valid_gate = batch.gate_labels.ge(0)
            if valid_gate.any():
                probabilities.append(output["gate_logits"][valid_gate].sigmoid().float().cpu())
                targets.append(batch.gate_labels[valid_gate].bool().cpu())
            if batch.recovery_mask.any():
                action_l1.append(
                    (output["recovery_actions"][batch.recovery_mask].float()
                     - batch.recovery_targets[batch.recovery_mask].float()).abs().cpu()
                )
    if not probabilities:
        raise ValueError("evaluation features contain no valid gate labels")
    probability = torch.cat(probabilities)
    target = torch.cat(targets)
    report = {
        "kind": "v3_action_aware_recovery_evaluation",
        "features": str(args.features), "checkpoint": str(args.checkpoint),
        "threshold": args.threshold, "valid_gate_decisions": int(target.numel()),
        "positive_gate_decisions": int(target.sum()),
        "metrics": binary_metrics(probability >= args.threshold, target),
        "mean_gate_probability": {
            "recovery": float(probability[target].mean()) if target.any() else None,
            "normal": float(probability[~target].mean()) if (~target).any() else None,
        },
        "recovery_action_normalized_l1": (
            float(torch.cat(action_l1).mean()) if action_l1 else None
        ),
        "limitation": (
            "Offline source-held-out feature metrics only; they do not establish "
            "online recovery completion or robustness to unrecorded perturbations."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
