#!/usr/bin/env python
"""Tune a failure-aware intervention gate on validation and freeze it for test."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from scripts.train_task_aware_action_scorer import TaskAwareActionScorer, load_split


def model_outputs(model, data, device):
    with torch.inference_mode():
        output = model(
            data["before_slots"].to(device).float(),
            data["executed_actions"].to(device).float(),
            data["joint_positions"].to(device).float(),
            data["language_tokens"].to(device),
        )
    return {
        "rank": output[:, :, -1, 9],
        "predicted_progress": output[:, :, -1, 10],
        "failure": output[:, :, -1, 8].sigmoid(),
        "true_progress": data["task_progress_m"][:, :, -1].to(device).float(),
    }


def policy_metrics(values, intervene):
    rank, true = values["rank"], values["true_progress"]
    alternative = rank.argmax(1)
    chosen = torch.where(intervene, alternative, torch.zeros_like(alternative))
    selected = true.gather(1, chosen[:, None]).squeeze(1)
    oracle = true.max(1).values
    return {
        "progress_mm": float(selected.mean() * 1000),
        "regret_mm": float((oracle - selected).mean() * 1000),
        "positive_progress_percent": float(100 * selected.gt(0).float().mean()),
        "intervention_percent": float(100 * intervene.float().mean()),
        "best_branch_accuracy_percent": float(100 * chosen.eq(true.argmax(1)).float().mean()),
    }


def choose_threshold(values, method):
    rank, predicted, failure = values["rank"], values["predicted_progress"], values["failure"]
    best = rank.argmax(1)
    row = torch.arange(len(best), device=best.device)
    predicted_gain = predicted[row, best] - predicted[:, 0]
    score_gain = rank[row, best] - rank[:, 0]
    failure_nominal = failure[:, 0]
    if method == "failure":
        candidates = [(float(t), failure_nominal >= t) for t in torch.linspace(0, 1, 101, device=rank.device)]
    elif method == "predicted_progress":
        thresholds = torch.linspace(-0.01, 0.03, 161, device=rank.device)
        candidates = [(float(t), predicted_gain >= t) for t in thresholds]
    elif method == "rank_margin":
        low, high = float(score_gain.min()), float(score_gain.max())
        candidates = [(float(t), score_gain >= t) for t in torch.linspace(low, high + 1e-6, 161, device=rank.device)]
    elif method == "failure_and_progress":
        candidates = []
        for failure_threshold in torch.linspace(0.1, 0.9, 17, device=rank.device):
            for progress_threshold in torch.linspace(-0.005, 0.02, 26, device=rank.device):
                gate = (failure_nominal >= failure_threshold) & (predicted_gain >= progress_threshold)
                candidates.append(((float(failure_threshold), float(progress_threshold)), gate))
    else:
        raise ValueError(method)
    scored = []
    for threshold, gate in candidates:
        metrics = policy_metrics(values, gate)
        scored.append((metrics["progress_mm"], -metrics["intervention_percent"], threshold, gate, metrics))
    return max(scored, key=lambda item: (item[0], item[1]))


def gate_with_threshold(values, method, threshold):
    rank, predicted, failure = values["rank"], values["predicted_progress"], values["failure"]
    best = rank.argmax(1); row = torch.arange(len(best), device=best.device)
    predicted_gain = predicted[row, best] - predicted[:, 0]
    score_gain = rank[row, best] - rank[:, 0]
    if method == "failure": return failure[:, 0] >= threshold
    if method == "predicted_progress": return predicted_gain >= threshold
    if method == "rank_margin": return score_gain >= threshold
    if method == "failure_and_progress":
        return (failure[:, 0] >= threshold[0]) & (predicted_gain >= threshold[1])
    raise ValueError(method)


def baselines(values):
    true = values["true_progress"]
    nominal = true[:, 0]; oracle = true.max(1).values
    always = policy_metrics(values, torch.ones(len(true), dtype=torch.bool, device=true.device))
    return {
        "nominal": {
            "progress_mm": float(nominal.mean() * 1000),
            "regret_mm": float((oracle - nominal).mean() * 1000),
            "positive_progress_percent": float(100 * nominal.gt(0).float().mean()),
            "intervention_percent": 0.0,
        },
        "random_action": {"progress_mm": float(true.mean() * 1000)},
        "oracle": {"progress_mm": float(oracle.mean() * 1000)},
        "always_scorer": always,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--progress-dir", required=True)
    p.add_argument("--checkpoint", required=True); p.add_argument("--val-task", type=int, default=8)
    p.add_argument("--broad-validation", action="store_true")
    p.add_argument("--test-task", type=int, default=9); p.add_argument("--episode-ids", default="0,1,2,3,4")
    p.add_argument("--output", required=True); args = p.parse_args()
    episodes = [int(value) for value in args.episode_ids.split(",") if value]
    data_dir, progress_dir = Path(args.data_dir), Path(args.progress_dir)
    if args.broad_validation:
        same_task_unseen_episode = load_split(data_dir, progress_dir, list(range(8)), [4])
        heldout_task = load_split(data_dir, progress_dir, [args.val_task], episodes)
        val = {
            key: torch.cat((same_task_unseen_episode[key], heldout_task[key]))
            for key in same_task_unseen_episode
        }
    else:
        val = load_split(data_dir, progress_dir, [args.val_task], episodes)
    test = load_split(data_dir, progress_dir, [args.test_task], episodes)
    model = TaskAwareActionScorer(torch.zeros(7), torch.ones(7), torch.zeros(7), torch.ones(7))
    model.load_state_dict(load_file(args.checkpoint), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device).eval()
    val_values, test_values = model_outputs(model, val, device), model_outputs(model, test, device)
    result = {"validation_task": args.val_task, "broad_validation": args.broad_validation,
              "validation_forks": int(len(val["before_slots"])),
              "test_task": args.test_task, "episode_ids": episodes,
              "validation_baselines": baselines(val_values), "test_baselines": baselines(test_values), "gates": {}}
    for method in ("failure", "predicted_progress", "rank_margin", "failure_and_progress"):
        _, _, threshold, _, validation_metrics = choose_threshold(val_values, method)
        test_gate = gate_with_threshold(test_values, method, threshold)
        result["gates"][method] = {
            "threshold": threshold, "validation": validation_metrics,
            "test": policy_metrics(test_values, test_gate),
        }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)); print(f"saved={output}")


if __name__ == "__main__": main()
