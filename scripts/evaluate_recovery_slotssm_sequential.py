#!/usr/bin/env python
"""Causal episode-level evaluation for a shared Recovery SlotSSM checkpoint."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

from safetensors.torch import load_file
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovery.data.dataset import read_jsonl
from recovery.data.tensor_dataset import (
    PHASES,
    build_recovery_model_inputs,
    tensorize_transition,
)
from recovery.models.recovery_slotssm import RecoverySlotSSM


ALLOWED_NEXT = {
    0: {0, 1, 2},
    1: {1, 2},
    2: {2, 3},
    3: {3, 4},
    4: {4, 5},
    5: {5, 6},
    6: {6, 7, 8},
    7: {7, 8},
    8: {8},
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", default="validation,development_test")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def expand(value: Path) -> list[Path]:
    if value.is_file():
        return [value]
    return sorted(value.glob("task*-episode*/transitions-v2.jsonl"))


def binary_metrics(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor):
    prediction = prediction[valid]
    target = target[valid]
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    tn = int((~prediction & ~target).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def constrained_phase_decode(logits: torch.Tensor) -> torch.Tensor:
    """Greedily decode legal phases using only current logits and past state."""
    previous = 0
    path = [previous]
    for timestep in range(1, logits.shape[0]):
        allowed = sorted(ALLOWED_NEXT.get(previous, {previous}))
        local = logits[timestep, allowed]
        previous = allowed[int(local.argmax())]
        path.append(previous)
    return torch.tensor(path, dtype=torch.long)


def failure_gated_phase_decode(
    logits: torch.Tensor,
    onset_probability: torch.Tensor,
    *,
    threshold: float,
    confirm_steps: int = 2,
) -> torch.Tensor:
    """Stay NOMINAL until a confirmed failure trigger, then follow legal phases."""
    path = []
    previous = 0
    consecutive = 0
    triggered = False
    for timestep in range(logits.shape[0]):
        if not triggered:
            consecutive = (
                consecutive + 1
                if float(onset_probability[timestep]) >= threshold else 0
            )
            if consecutive >= confirm_steps:
                triggered = True
                previous = 2
            else:
                path.append(0)
                continue
        allowed = sorted(ALLOWED_NEXT.get(previous, {previous}))
        local = logits[timestep, allowed]
        previous = allowed[int(local.argmax())]
        path.append(previous)
    return torch.tensor(path, dtype=torch.long)


def calibrate_threshold(probability, target, valid):
    best = None
    for threshold in [value / 20 for value in range(1, 20)]:
        metrics = binary_metrics(probability >= threshold, target, valid)
        candidate = (metrics["f1"], metrics["precision"], threshold)
        if best is None or candidate > best[0]:
            best = (candidate, threshold, metrics)
    return best[1], best[2]


def stack_episode(rows):
    steps = [tensorize_transition(row) for row in rows]
    return {
        key: torch.stack([step[key] for step in steps]).unsqueeze(0)
        for key in steps[0]
    }


@torch.inference_mode()
def main() -> None:
    args = arguments()
    requested_splits = {
        value.strip() for value in args.splits.split(",") if value.strip()
    }
    grouped = defaultdict(list)
    for value in args.inputs:
        for path in expand(value):
            for row in read_jsonl(path):
                if row.split in requested_splits:
                    grouped[(row.task_suite, row.task_id, row.episode_id)].append(row)
    if not grouped:
        raise ValueError("no episodes found for requested splits")

    training = json.loads(args.training_report.read_text())
    model = RecoverySlotSSM(
        backend=training["backend"],
        model_dim=int(training["model_dim"]),
        num_layers=int(training["num_layers"]),
    )
    model.load_state_dict(load_file(str(args.checkpoint)), strict=True)
    device = torch.device("cuda" if training["backend"] == "mamba2" else "cpu")
    model.to(device).eval()

    episode_reports = []
    aggregate_values = defaultdict(list)
    phase_correct = phase_total = violations = transitions = 0
    constrained_correct = constrained_total = 0
    episode_cache = []
    for (_, task_id, episode_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row.timestep)
        sequence = {
            key: value.to(device) for key, value in stack_episode(rows).items()
        }
        outputs = model(build_recovery_model_inputs(sequence))
        phase_logits = outputs["phase_logits"][0].cpu()
        phase_prediction = phase_logits.argmax(-1)
        constrained_prediction = constrained_phase_decode(phase_logits)
        phase_target = sequence["phase_id"][0].cpu()
        phase_valid = sequence["phase_valid"][0].bool().cpu()
        phase_correct += int((phase_prediction[phase_valid] == phase_target[phase_valid]).sum())
        phase_total += int(phase_valid.sum())
        constrained_correct += int(
            (constrained_prediction[phase_valid] == phase_target[phase_valid]).sum()
        )
        constrained_total += int(phase_valid.sum())
        episode_violations = sum(
            int(int(right) not in ALLOWED_NEXT.get(int(left), {int(left)}))
            for left, right in zip(phase_prediction[:-1], phase_prediction[1:])
        )
        violations += episode_violations
        transitions += max(len(rows) - 1, 0)

        head_specs = {
            "failure_onset": ("failure_onset_logits", "failure_onset_target", "failure_onset_valid"),
            "current_failure": ("current_failure_logits", "current_failure_target", "current_failure_valid"),
            "recovery_active": ("recovery_active_logits", "recovery_active_target", None),
            "verification": ("verification_logits", "verification_target", "verification_valid"),
            "handoff": ("handoff_logits", "handoff_target", "handoff_valid"),
            "grasped": ("grasped_logits", "grasped_target", "grasped_valid"),
            "lifted": ("lifted_logits", "lifted_target", "lifted_valid"),
            "slip": ("slip_logits", "slip_target", "slip_valid"),
        }
        head_reports = {}
        for name, (logit_key, target_key, valid_key) in head_specs.items():
            probability = outputs[logit_key][0].sigmoid().cpu()
            prediction = probability >= args.threshold
            target = sequence[target_key][0].bool().cpu()
            valid = (
                sequence[valid_key][0].bool().cpu()
                if valid_key is not None else torch.ones_like(target)
            )
            metrics = binary_metrics(prediction, target, valid)
            head_reports[name] = metrics
            aggregate_values[name].append(
                (probability, target, valid, rows[0].split)
            )

        failure_truth = sequence["current_failure_target"][0].bool().cpu()
        failure_prediction = (
            outputs["current_failure_logits"][0].sigmoid().cpu() >= args.threshold
        )
        true_indices = torch.where(failure_truth)[0]
        predicted_indices = torch.where(failure_prediction)[0]
        first_prediction_offset = (
            int(predicted_indices[0] - true_indices[0])
            if len(true_indices) and len(predicted_indices) else None
        )
        if len(true_indices):
            onset = int(true_indices[0])
            post_onset_predictions = predicted_indices[predicted_indices >= onset]
            post_onset_detection_delay = (
                int(post_onset_predictions[0] - onset)
                if len(post_onset_predictions) else None
            )
            pre_onset_false_positives = int(failure_prediction[:onset].sum())
            pre_onset_steps = onset
        else:
            onset = None
            post_onset_detection_delay = None
            pre_onset_false_positives = int(failure_prediction.sum())
            pre_onset_steps = len(failure_prediction)
        episode_reports.append(
            {
                "task_id": task_id,
                "episode_id": episode_id,
                "split": rows[0].split,
                "num_rows": len(rows),
                "phase_accuracy": float(
                    (phase_prediction[phase_valid] == phase_target[phase_valid]).float().mean()
                ),
                "constrained_phase_accuracy": float(
                    (
                        constrained_prediction[phase_valid]
                        == phase_target[phase_valid]
                    ).float().mean()
                ),
                "illegal_phase_transitions": episode_violations,
                "failure_onset_timestep_index": onset,
                "first_failure_prediction_offset_from_onset": first_prediction_offset,
                "post_onset_detection_delay_steps": post_onset_detection_delay,
                "pre_onset_false_positive_steps": pre_onset_false_positives,
                "pre_onset_false_positive_rate": (
                    pre_onset_false_positives / max(pre_onset_steps, 1)
                ),
                "heads": head_reports,
                "predicted_phase_counts": {
                    PHASES[index]: int((phase_prediction == index).sum())
                    for index in range(len(PHASES))
                },
                "constrained_phase_counts": {
                    PHASES[index]: int((constrained_prediction == index).sum())
                    for index in range(len(PHASES))
                },
            }
        )
        episode_cache.append(
            {
                "episode_id": episode_id,
                "split": rows[0].split,
                "phase_logits": phase_logits,
                "phase_target": phase_target,
                "phase_valid": phase_valid,
                "failure_probability": (
                    outputs["current_failure_logits"][0].sigmoid().cpu()
                ),
                "onset_probability": (
                    outputs["failure_onset_logits"][0].sigmoid().cpu()
                ),
            }
        )

    aggregate_heads = {}
    for name, values in aggregate_values.items():
        probability = torch.cat([value[0] for value in values])
        target = torch.cat([value[1] for value in values])
        valid = torch.cat([value[2] for value in values])
        aggregate_heads[name] = binary_metrics(
            probability >= args.threshold, target, valid
        )
    calibrated_heads = {}
    for name, values in aggregate_values.items():
        validation = [value for value in values if value[3] == "validation"]
        development = [
            value for value in values if value[3] == "development_test"
        ]
        if not validation:
            continue
        validation_probability = torch.cat([value[0] for value in validation])
        validation_target = torch.cat([value[1] for value in validation])
        validation_valid = torch.cat([value[2] for value in validation])
        threshold, validation_metrics = calibrate_threshold(
            validation_probability, validation_target, validation_valid
        )
        record = {
            "threshold_selected_on_validation": threshold,
            "validation": validation_metrics,
        }
        if development:
            development_probability = torch.cat(
                [value[0] for value in development]
            )
            development_target = torch.cat([value[1] for value in development])
            development_valid = torch.cat([value[2] for value in development])
            record["development_test"] = binary_metrics(
                development_probability >= threshold,
                development_target,
                development_valid,
            )
        calibrated_heads[name] = record
    gated_phase = {}
    onset_calibration = calibrated_heads.get("failure_onset")
    if onset_calibration is not None:
        threshold = onset_calibration["threshold_selected_on_validation"]
        for split in ("validation", "development_test"):
            selected = [item for item in episode_cache if item["split"] == split]
            correct = total = 0
            per_episode = []
            for item in selected:
                prediction = failure_gated_phase_decode(
                    item["phase_logits"],
                    item["onset_probability"],
                    threshold=threshold,
                    confirm_steps=1,
                )
                valid = item["phase_valid"]
                episode_correct = int(
                    (prediction[valid] == item["phase_target"][valid]).sum()
                )
                episode_total = int(valid.sum())
                correct += episode_correct
                total += episode_total
                per_episode.append(
                    {
                        "episode_id": item["episode_id"],
                        "accuracy": episode_correct / max(episode_total, 1),
                        "phase_counts": {
                            PHASES[index]: int((prediction == index).sum())
                            for index in range(len(PHASES))
                        },
                    }
                )
            if selected:
                gated_phase[split] = {
                    "accuracy": correct / max(total, 1),
                    "episodes": per_episode,
                }
    result = {
        "kind": "recovery_slotssm_causal_sequential_audit",
        "checkpoint": str(args.checkpoint.resolve()),
        "splits": sorted(requested_splits),
        "threshold": args.threshold,
        "num_episodes": len(episode_reports),
        "phase_accuracy": phase_correct / max(phase_total, 1),
        "constrained_phase_accuracy": (
            constrained_correct / max(constrained_total, 1)
        ),
        "illegal_phase_transition_rate": violations / max(transitions, 1),
        "aggregate_heads": aggregate_heads,
        "validation_calibrated_heads": calibrated_heads,
        "failure_gated_constrained_phase": {
            "gate_head": "failure_onset",
            "onset_threshold_selected_on_validation": (
                onset_calibration["threshold_selected_on_validation"]
                if onset_calibration is not None else None
            ),
            "confirm_steps": 1,
            "splits": gated_phase,
        },
        "episodes": episode_reports,
        "online_smoke_allowed": False,
        "claim_limit": (
            "offline causal audit only; thresholds are selected on validation "
            "episodes only and the online rollout safety gate is not complete"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
