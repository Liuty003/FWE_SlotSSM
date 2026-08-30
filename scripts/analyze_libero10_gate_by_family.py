#!/usr/bin/env python3
"""Report and calibrate a shared LIBERO-10 Gate separately by task family."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TASK = re.compile(r"task(\d+)(?:-|:)episode(\d+)")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--failure-feature-report", type=Path, required=True)
    parser.add_argument("--normal-feature-report", type=Path, required=True)
    parser.add_argument("--failure-routes", type=Path, required=True)
    parser.add_argument("--normal-manifest", type=Path, required=True)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--target-source-fpr", type=float, default=0.05)
    parser.add_argument("--minimum-threshold", type=float, default=0.5)
    parser.add_argument("--trigger-confirmation-decisions", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def identity(path: str) -> tuple[int, int]:
    match = TASK.search(path)
    if match is None:
        raise ValueError(f"cannot parse task identity from {path}")
    return int(match.group(1)), int(match.group(2))


def auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = sum(p > n for p in positive for n in negative)
    ties = sum(p == n for p in positive for n in negative)
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def metrics(positive: list[float], negative: list[float], threshold: float) -> dict:
    tp = sum(value >= threshold for value in positive)
    fp = sum(value >= threshold for value in negative)
    return {
        "threshold": threshold,
        "recall": tp / len(positive) if positive else None,
        "false_positive_rate": fp / len(negative) if negative else None,
        "precision": tp / (tp + fp) if tp + fp else None,
    }


def calibrated(positive: list[float], negative: list[float], target_fpr: float,
               minimum_threshold: float) -> dict:
    candidates = sorted(
        {minimum_threshold, 1.0, *(value for value in positive + negative if value >= minimum_threshold)}
    )
    feasible = [metrics(positive, negative, value) for value in candidates]
    feasible = [row for row in feasible if row["false_positive_rate"] <= target_fpr]
    if not feasible:
        return metrics(positive, negative, 1.0)
    return max(feasible, key=lambda row: (row["recall"], -row["threshold"]))


def has_confirmation_run(values: list[float], threshold: float, confirmations: int) -> bool:
    consecutive = 0
    for value in values:
        consecutive = consecutive + 1 if value >= threshold else 0
        if consecutive >= confirmations:
            return True
    return False


def source_metrics(positive_sequences: list[list[float]], negative_sequences: list[list[float]],
                   threshold: float, confirmations: int) -> dict:
    hits = sum(has_confirmation_run(values, threshold, confirmations)
               for values in positive_sequences)
    false_activations = sum(has_confirmation_run(values, threshold, confirmations)
                            for values in negative_sequences)
    return {
        "threshold": threshold,
        "trigger_confirmation_decisions": confirmations,
        "failure_source_hit_rate": hits / len(positive_sequences) if positive_sequences else None,
        "normal_source_false_activation_rate": (
            false_activations / len(negative_sequences) if negative_sequences else None
        ),
        "failure_sources_hit": hits,
        "normal_sources_false_activated": false_activations,
    }


def calibrated_sources(positive_sequences: list[list[float]], negative_sequences: list[list[float]],
                       target_fpr: float, minimum_threshold: float,
                       confirmations: int) -> dict:
    values = [value for sequence in positive_sequences + negative_sequences for value in sequence]
    candidates = sorted({minimum_threshold, 1.0, *(value for value in values if value >= minimum_threshold)})
    rows = [source_metrics(positive_sequences, negative_sequences, threshold, confirmations)
            for threshold in candidates]
    feasible = [row for row in rows
                if row["normal_source_false_activation_rate"] <= target_fpr]
    if not feasible:
        result = source_metrics(positive_sequences, negative_sequences, 1.0, confirmations)
        result["target_satisfied"] = False
        return result
    result = max(feasible, key=lambda row: (row["failure_source_hit_rate"], -row["threshold"]))
    result["target_satisfied"] = True
    return result


def main() -> None:
    args = arguments()
    if (not 0 <= args.target_fpr <= 1 or not 0 <= args.target_source_fpr <= 1
            or not 0 <= args.minimum_threshold <= 1
            or args.trigger_confirmation_decisions < 1):
        raise ValueError("FPRs/threshold must be in [0, 1] and confirmations must be positive")
    evaluation = json.loads(args.evaluation.read_text())
    failure_features = json.loads(args.failure_feature_report.read_text())
    normal_features = json.loads(args.normal_feature_report.read_text())
    routes = json.loads(args.failure_routes.read_text())
    normal_manifest = json.loads(args.normal_manifest.read_text())

    family_by_identity = {
        identity(row["sample_id"]): row["family_name"]
        for row in routes["records"] if row.get("family_name")
    }
    family_by_identity.update({
        (int(row["task_id"]), int(row.get("episode_id", 0))): row["family_name"]
        for row in normal_manifest["records"]
    })
    family_by_source = {}
    for report in (failure_features, normal_features):
        for row in report["source_records"]:
            key = identity(row["source"])
            if key not in family_by_identity:
                raise KeyError(f"no family for source {key}")
            family_by_source[str(row["source_id"])] = family_by_identity[key]

    values: dict[str, dict[str, list[float]]] = {}
    for source_id, source in evaluation["per_source"].items():
        family = family_by_source[source_id]
        bucket = values.setdefault(family, {
            "positive": [], "negative": [],
            "failure_positive_sequences": [], "normal_negative_sequences": [],
        })
        positive_sequence = []
        negative_sequence = []
        for row in source["gate_sequence"]:
            if row["gate_label"] == 1:
                probability = float(row["gate_probability"])
                bucket["positive"].append(probability)
                positive_sequence.append(probability)
            elif row["gate_label"] == 0:
                probability = float(row["gate_probability"])
                bucket["negative"].append(probability)
                negative_sequence.append(probability)
        if positive_sequence:
            bucket["failure_positive_sequences"].append(positive_sequence)
        else:
            bucket["normal_negative_sequences"].append(negative_sequence)

    families = {}
    for family, bucket in sorted(values.items()):
        positive, negative = bucket["positive"], bucket["negative"]
        positive_sequences = bucket["failure_positive_sequences"]
        negative_sequences = bucket["normal_negative_sequences"]
        families[family] = {
            "positive_rows": len(positive),
            "negative_rows": len(negative),
            "auc": auc(positive, negative),
            "shared_threshold_0_5": metrics(positive, negative, 0.5),
            "shared_threshold_0_7": metrics(positive, negative, 0.7),
            "row_calibrated_diagnostic": calibrated(
                positive, negative, args.target_fpr, args.minimum_threshold,
            ),
            "trajectory_threshold_0_7": source_metrics(
                positive_sequences, negative_sequences, 0.7,
                args.trigger_confirmation_decisions,
            ),
            "calibrated": calibrated_sources(
                positive_sequences, negative_sequences, args.target_source_fpr,
                args.minimum_threshold, args.trigger_confirmation_decisions,
            ),
        }
    payload = {
        "kind": "libero10_gate_family_calibration_v2",
        "evaluation": str(args.evaluation.resolve()),
        "target_false_positive_rate": args.target_fpr,
        "target_normal_source_false_activation_rate": args.target_source_fpr,
        "minimum_threshold": args.minimum_threshold,
        "trigger_confirmation_decisions": args.trigger_confirmation_decisions,
        "families": families,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
