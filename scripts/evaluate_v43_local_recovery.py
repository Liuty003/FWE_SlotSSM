#!/usr/bin/env python3
"""Offline gate and per-source direct-action evaluation for V4.3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.data.v4_relation_features import load
from recovery.models.v4_relational_recovery import V4ModelConfig
from recovery.models.v43_local_recovery import V43LocalRecovery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.safetensors")
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from safetensors.torch import load_file

    config = json.loads((args.adapter_dir / "config.json").read_text())
    model = V43LocalRecovery(
        V4ModelConfig(**config["model"]),
        direct_head_version=config.get("direct_head_version", "context_only_v1"),
        recovery_output_horizon=config.get("recovery_output_horizon"),
    ).to(args.device).eval()
    model.load_state_dict(load_file(str(args.adapter_dir / args.checkpoint), device=args.device), strict=True)
    rows = []
    evaluated_horizons = []
    for path in args.features:
        features = load(path)
        with torch.inference_mode():
            for start in range(0, features.num_rows, args.batch_size):
                index = torch.arange(start, min(start + args.batch_size, features.num_rows))
                batch = features.index(index).to(args.device)
                output = model(**{key: getattr(batch, key) for key in (
                    "visual_tokens", "action_hidden", "token_points", "token_valid", "token_image_xy",
                    "gripper_position", "gripper_image_xy",
                )})
                predicted = output["recovery_actions"].float()
                evaluated_horizon = min(predicted.shape[1], batch.recovery_targets.shape[1])
                if evaluated_horizon < 1:
                    raise ValueError("adapter and features have no common action horizon")
                evaluated_horizons.append(evaluated_horizon)
                # A target-horizon-1 dataset is also the strict way to compare
                # an archived 8-step policy's first action against V4.4.2.
                difference = (
                    predicted[:, :evaluated_horizon]
                    - batch.recovery_targets[:, :evaluated_horizon].float()
                ).abs()
                error = difference.mean((1, 2))
                for offset in range(len(index)):
                    has_recovery_target = bool(batch.recovery_mask[offset])
                    rows.append({
                        "source_id": int(batch.source_ids[offset]),
                        "gate_label": int(batch.gate_labels[offset]),
                        "gate_probability": float(output["gate_probability"][offset]),
                        "recovery_target": has_recovery_target,
                        "action_l1": float(error[offset]) if has_recovery_target else None,
                        "action_l1_per_dim": (difference[offset].mean(0).cpu().tolist()
                                              if has_recovery_target else None),
                        "predicted_first_action": (output["recovery_actions"][offset, 0].float().cpu().tolist()
                                                   if has_recovery_target else None),
                        "target_first_action": (batch.recovery_targets[offset, 0].float().cpu().tolist()
                                                if has_recovery_target else None),
                    })
    sources = {}
    for identity in sorted({row["source_id"] for row in rows}):
        selected = [row for row in rows if row["source_id"] == identity]
        recovery = [row["action_l1"] for row in selected if row["action_l1"] is not None]
        recovery_per_dim = [row["action_l1_per_dim"] for row in selected
                            if row["action_l1_per_dim"] is not None]
        positive_probabilities = [
            row["gate_probability"] for row in selected if row["gate_label"] == 1
        ]
        negative_probabilities = [
            row["gate_probability"] for row in selected if row["gate_label"] == 0
        ]
        true_positive = sum(value >= args.gate_threshold for value in positive_probabilities)
        false_positive = sum(value >= args.gate_threshold for value in negative_probabilities)
        sources[str(identity)] = {
            "rows": len(selected), "recovery_rows": len(recovery),
            "recovery_action_l1": None if not recovery else sum(recovery) / len(recovery),
            "recovery_action_l1_per_dim": None if not recovery_per_dim else [
                sum(values[dimension] for values in recovery_per_dim) / len(recovery_per_dim)
                for dimension in range(len(recovery_per_dim[0]))
            ],
            "mean_gate_probability": sum(row["gate_probability"] for row in selected) / len(selected),
            "positive_gate_rows": len(positive_probabilities),
            "negative_gate_rows": len(negative_probabilities),
            "gate_recall": None if not positive_probabilities else true_positive / len(positive_probabilities),
            "gate_false_positive_rate": None if not negative_probabilities else false_positive / len(negative_probabilities),
            "positive_gate_probability": None if not positive_probabilities else {
                "mean": sum(positive_probabilities) / len(positive_probabilities),
                "min": min(positive_probabilities), "max": max(positive_probabilities),
            },
            "negative_gate_probability": None if not negative_probabilities else {
                "mean": sum(negative_probabilities) / len(negative_probabilities),
                "min": min(negative_probabilities), "max": max(negative_probabilities),
            },
            # Feature rows are stored in trajectory order.  Keep the compact
            # sequence so handoff release thresholds can be audited instead
            # of being hidden by aggregate NORMAL/recovery metrics.
            "gate_sequence": [
                {
                    "source_row": source_row,
                    "gate_label": row["gate_label"],
                    "gate_probability": row["gate_probability"],
                    "recovery_target": row["recovery_target"],
                }
                for source_row, row in enumerate(selected)
            ],
            "recovery_sequence": [
                {
                    "source_row": source_row,
                    "action_l1": row["action_l1"],
                    "action_l1_per_dim": row["action_l1_per_dim"],
                    "predicted_first_action": row["predicted_first_action"],
                    "target_first_action": row["target_first_action"],
                }
                for source_row, row in enumerate(selected) if row["recovery_target"]
            ],
        }
    valid = [row for row in rows if row["gate_label"] >= 0]
    true_positive = sum(row["gate_label"] == 1 and row["gate_probability"] >= args.gate_threshold for row in valid)
    false_positive = sum(row["gate_label"] == 0 and row["gate_probability"] >= args.gate_threshold for row in valid)
    positive = sum(row["gate_label"] == 1 for row in valid)
    negative = sum(row["gate_label"] == 0 for row in valid)
    recovery_errors = [row["action_l1"] for row in rows if row["action_l1"] is not None]
    recovery_errors_per_dim = [row["action_l1_per_dim"] for row in rows
                               if row["action_l1_per_dim"] is not None]
    report = {
        "kind": "v43_local_recovery_offline_evaluation", "adapter": str(args.adapter_dir / args.checkpoint),
        "features": [str(path) for path in args.features], "rows": len(rows),
        "gate_threshold": args.gate_threshold,
        "evaluated_action_horizon": min(evaluated_horizons),
        "gate_recall": true_positive / max(positive, 1),
        "gate_false_positive_rate": false_positive / max(negative, 1),
        "recovery_action_l1": sum(recovery_errors) / max(len(recovery_errors), 1),
        "recovery_action_l1_per_dim": [
            sum(values[dimension] for values in recovery_errors_per_dim) / len(recovery_errors_per_dim)
            for dimension in range(len(recovery_errors_per_dim[0]))
        ] if recovery_errors_per_dim else None,
        "per_source": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
