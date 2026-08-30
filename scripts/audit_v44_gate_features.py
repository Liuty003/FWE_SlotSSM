#!/usr/bin/env python3
"""Audit takeover-gate calibration on archived native-input V4 features."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.safetensors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--thresholds", default="0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated takeover thresholds.",
    )
    parser.add_argument(
        "--confirmations", default="1,2,3",
        help="Comma-separated consecutive-positive requirements.",
    )
    return parser.parse_args()


def model_inputs(features):
    return {
        "visual_tokens": features.visual_tokens,
        "action_hidden": features.action_hidden,
        "token_points": features.token_points,
        "token_valid": features.token_valid,
        "token_image_xy": features.token_image_xy,
        "gripper_position": features.gripper_position,
        "gripper_image_xy": features.gripper_image_xy,
    }


def has_run(values: list[float], threshold: float, confirmations: int) -> bool:
    run = 0
    for value in values:
        run = run + 1 if value >= threshold else 0
        if run >= confirmations:
            return True
    return False


def main() -> None:
    args = arguments()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    thresholds = [float(value) for value in args.thresholds.split(",")]
    confirmations = [int(value) for value in args.confirmations.split(",")]
    if (not thresholds or not confirmations or any(not 0 <= value <= 1 for value in thresholds)
            or any(value < 1 for value in confirmations)):
        raise ValueError("invalid thresholds or confirmations")
    feature_report_path = args.features.with_suffix(".json")
    for path in (args.features, feature_report_path, args.adapter_dir / "config.json",
                 args.adapter_dir / args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    import torch
    from safetensors.torch import load_file
    from recovery.data.v4_relation_features import load
    from recovery.models.v4_relational_recovery import V4ModelConfig
    from recovery.models.v44_dual_ownership_recovery import V44DualOwnershipRecovery

    config = json.loads((args.adapter_dir / "config.json").read_text())
    if config.get("kind") != "v44_dual_ownership_recovery_v1":
        raise ValueError("adapter must be a V4.4 dual-ownership checkpoint")
    features = load(args.features)
    model = V44DualOwnershipRecovery(
        V4ModelConfig(**config["model"]),
        direct_head_version=config.get("direct_head_version", "pooled_chunk_mlp_v3"),
        recovery_output_horizon=config.get("recovery_output_horizon", 1),
    ).to(args.device).eval()
    model.load_state_dict(
        load_file(str(args.adapter_dir / args.checkpoint), device=args.device), strict=True,
    )
    probabilities = []
    with torch.inference_mode():
        for start in range(0, features.num_rows, args.batch_size):
            batch = features.index(torch.arange(start, min(start + args.batch_size, features.num_rows)))
            output = model(**model_inputs(batch.to(args.device)))
            probabilities.extend(output["gate_probability"].float().cpu().tolist())

    metadata = json.loads(feature_report_path.read_text())
    source_metadata = {int(record["source_id"]): record for record in metadata["source_records"]}
    grouped: dict[int, list[float]] = {}
    labels: dict[int, list[int]] = {}
    if not (len(probabilities) == features.num_rows == len(features.source_ids)):
        raise RuntimeError("prediction and feature row counts differ")
    for source_id, probability, label in zip(
        features.source_ids.tolist(), probabilities, features.gate_labels.tolist(),
    ):
        grouped.setdefault(int(source_id), []).append(probability)
        labels.setdefault(int(source_id), []).append(int(label))

    source_records = []
    for source_id, values in grouped.items():
        source = source_metadata[source_id]
        match = re.search(r"task(\d+)-episode(\d+)$", source["source"])
        source_records.append({
            "source_id": source_id,
            "task_id": None if match is None else int(match.group(1)),
            "episode": None if match is None else int(match.group(2)),
            "success": source.get("success"),
            "rows": len(values),
            "gate_label_counts": {
                str(label): labels[source_id].count(label) for label in sorted(set(labels[source_id]))
            },
            "maximum_gate_probability": max(values),
            "mean_gate_probability": sum(values) / len(values),
            "gate_probabilities": values,
            "trigger_grid": {
                f"threshold={threshold}:confirmations={confirmation}": has_run(
                    values, threshold, confirmation,
                )
                for threshold in thresholds for confirmation in confirmations
            },
        })
    source_records.sort(key=lambda record: (record["task_id"] is None, record["task_id"]))
    summary = {}
    for threshold in thresholds:
        for confirmation in confirmations:
            key = f"threshold={threshold}:confirmations={confirmation}"
            triggered = [record for record in source_records if record["trigger_grid"][key]]
            summary[key] = {
                "triggered_sources": len(triggered),
                "trigger_rate": len(triggered) / len(source_records),
                "task_ids": [record["task_id"] for record in triggered],
            }
    report = {
        "kind": "v44_takeover_gate_feature_audit_v1",
        "native_input_only": metadata.get("native_input_only"),
        "features": str(args.features),
        "adapter": str(args.adapter_dir / args.checkpoint),
        "rows": features.num_rows,
        "sources": len(source_records),
        "summary": summary,
        "source_records": source_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rows": features.num_rows,
                      "sources": len(source_records), "summary": summary}))


if __name__ == "__main__":
    main()
