#!/usr/bin/env python3
"""Evaluate a V4.2 checkpoint without using training-loss model selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.data.v4_relation_features import load
from recovery.models.v4_relational_recovery import V4ModelConfig, V4RelationalRecovery
from scripts.train_v4_relational_recovery import decoder, model_inputs, sha256


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.safetensors")
    parser.add_argument("--action-head", type=Path, required=True)
    parser.add_argument("--oft-root", type=Path, default=Path("/data/lty/code/openvla-oft-official"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--residual-scale-override", type=float)
    return parser.parse_args()


def confusion(probability: torch.Tensor, truth: torch.Tensor, threshold: float) -> dict:
    guess = probability >= threshold
    positive = truth.eq(1)
    negative = truth.eq(0)
    tp = int((guess & positive).sum())
    fp = int((guess & negative).sum())
    tn = int((~guess & negative).sum())
    fn = int((~guess & positive).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
        "trigger_rate": (tp + fp) / max(tp + fp + tn + fn, 1),
    }


def main() -> None:
    args = arguments()
    if args.batch_size < 1 or not all(0 <= value <= 1 for value in args.thresholds):
        raise ValueError("batch size must be positive and thresholds must lie in [0, 1]")
    config_path = args.adapter_dir / "config.json"
    checkpoint_path = args.adapter_dir / args.checkpoint
    for path in (args.features, config_path, checkpoint_path, args.action_head, args.oft_root):
        if not path.exists():
            raise FileNotFoundError(path)
    training_config = json.loads(config_path.read_text())
    expected_decoder = training_config.get("frozen_decoder_sha256")
    if expected_decoder != sha256(args.action_head):
        raise ValueError("evaluation decoder differs from the adapter's frozen decoder")
    features = load(args.features)
    model_values = dict(training_config["model"])
    if args.residual_scale_override is not None:
        if args.residual_scale_override <= 0:
            raise ValueError("residual scale override must be positive")
        model_values["residual_scale"] = args.residual_scale_override
    model_config = V4ModelConfig(**model_values)
    model = V4RelationalRecovery(model_config).to(args.device).eval()
    model.load_state_dict(load_file(str(checkpoint_path), device=args.device), strict=True)
    action_decoder = decoder(args.oft_root, args.action_head, model_config.action_hidden_dim,
                             model_config.action_dim, args.device)

    probabilities, nominal_actions, recovered_actions = [], [], []
    with torch.inference_mode():
        for start in range(0, features.num_rows, args.batch_size):
            index = torch.arange(start, min(start + args.batch_size, features.num_rows))
            batch = features.index(index).to(args.device)
            outputs = model(**model_inputs(batch))
            probabilities.append(outputs["gate_probability"].cpu())
            nominal_actions.append(action_decoder.predict_action(batch.action_hidden).float().cpu())
            recovered_actions.append(action_decoder.predict_action(outputs["recovery_action_hidden"]).float().cpu())
    probability = torch.cat(probabilities)
    nominal = torch.cat(nominal_actions)
    recovered = torch.cat(recovered_actions)
    valid = features.gate_labels.ge(0)
    probability_valid = probability[valid]
    truth = features.gate_labels[valid]
    thresholds = [confusion(probability_valid, truth, value) for value in args.thresholds]

    per_source = {}
    for source in features.source_ids.unique().tolist():
        selected = features.source_ids.eq(source) & valid
        labels = features.gate_labels[selected]
        selected_recovery = features.source_ids.eq(source) & features.recovery_mask
        source_nominal_l1 = None
        source_recovery_l1 = None
        if selected_recovery.any():
            source_nominal_l1 = float(torch.nn.functional.l1_loss(
                nominal[selected_recovery], features.recovery_targets[selected_recovery].float(),
            ))
            source_recovery_l1 = float(torch.nn.functional.l1_loss(
                recovered[selected_recovery], features.recovery_targets[selected_recovery].float(),
            ))
        per_source[str(source)] = {
            "rows": int(selected.sum()), "normal": int(labels.eq(0).sum()), "recovery": int(labels.eq(1).sum()),
            "mean_probability": float(probability[selected].mean()),
            "at_0.5": confusion(probability[selected], labels, 0.5),
            "nominal_action_l1_on_recovery": source_nominal_l1,
            "recovery_action_l1": source_recovery_l1,
            "recovery_l1_improvement": None if source_nominal_l1 is None else source_nominal_l1 - source_recovery_l1,
        }
    recovery_mask = features.recovery_mask
    nominal_l1 = None if not recovery_mask.any() else float(
        torch.nn.functional.l1_loss(nominal[recovery_mask], features.recovery_targets[recovery_mask].float())
    )
    recovery_l1 = None if not recovery_mask.any() else float(
        torch.nn.functional.l1_loss(recovered[recovery_mask], features.recovery_targets[recovery_mask].float())
    )
    report = {
        "kind": "v4_relational_recovery_evaluation",
        "residual_scale": model_config.residual_scale,
        "features": str(args.features), "adapter": str(checkpoint_path),
        "rows": features.num_rows, "valid_gate_rows": int(valid.sum()),
        "normal_probability_mean": float(probability[features.gate_labels.eq(0)].mean()),
        "recovery_probability_mean": float(probability[features.gate_labels.eq(1)].mean()),
        "gate_brier": float(((probability_valid - truth.float()) ** 2).mean()),
        "recovery_target_rows": int(recovery_mask.sum()),
        "nominal_action_l1_on_recovery": nominal_l1,
        "recovery_action_l1": recovery_l1,
        "recovery_l1_improvement": None if nominal_l1 is None else nominal_l1 - recovery_l1,
        "thresholds": thresholds,
        "best_balanced_accuracy": max(thresholds, key=lambda row: (row["balanced_accuracy"], row["f1"])),
        "per_source": per_source,
    }
    destination = args.output or args.adapter_dir / f"evaluation-{args.features.stem}-{args.checkpoint.removesuffix('.safetensors')}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
