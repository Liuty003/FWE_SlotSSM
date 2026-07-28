#!/usr/bin/env python
"""Strict held-out-task evaluation of counterfactual action sensitivity."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from scripts.train_counterfactual_slot_predictor import CounterfactualSlotPredictor


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task-id", type=int, default=9)
    p.add_argument("--output", required=True)
    return p.parse_args()


def masked_mae(prediction, target, valid):
    mask = valid[..., None, None].float()
    return ((prediction - target).abs() * mask).sum() / (
        mask.sum() * target.shape[-2] * target.shape[-1]
    ).clamp_min(1)


def main():
    args = arguments()
    data = load_file(str(Path(args.data_dir) / f"libero_spatial_t{args.task_id}_e0_slots.safetensors"))
    state = load_file(args.checkpoint)
    model = CounterfactualSlotPredictor(
        torch.zeros(7), torch.ones(7), torch.zeros(7), torch.ones(7),
        steps=data["after_slots"].shape[2],
    )
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    before = data["before_slots"].to(device).float()
    actions = data["executed_actions"].to(device).float()
    joint = data["joint_positions"].to(device).float()
    target = data["after_slots"].to(device).float()
    valid = data["after_valid"].to(device)
    with torch.inference_mode():
        prediction = model(before, actions, joint)
        shuffled_prediction = model(before, actions.roll(1, dims=1), joint)
    persistence = before[:, None, None].expand_as(target)
    action_agnostic_oracle = target.mean(dim=1, keepdim=True).expand_as(target)
    correct_mae = masked_mae(prediction, target, valid)
    shuffled_mae = masked_mae(shuffled_prediction, target, valid)
    persistence_mae = masked_mae(persistence, target, valid)
    oracle_mean_mae = masked_mae(action_agnostic_oracle, target, valid)

    # For every predicted action branch, retrieve the closest true branch
    # outcome from the same initial state. Correct branch retrieval is 10% by chance.
    final_pred = prediction[:, :, -1].float()
    final_target = target[:, :, -1].float()
    distances = (final_pred[:, :, None] - final_target[:, None]).abs().mean(dim=(-1, -2))
    labels = torch.arange(actions.shape[1], device=device).view(1, -1)
    retrieval = distances.argmin(dim=-1).eq(labels).float().mean()
    correct_distance = distances.diagonal(dim1=1, dim2=2)
    shifted_index = ((labels + 1) % actions.shape[1]).expand(before.shape[0], -1)
    shifted_distance = distances.gather(2, shifted_index[:, :, None]).squeeze(-1)
    pairwise_win = correct_distance.lt(shifted_distance).float().mean()

    # Compare centered cross-branch outcome deltas, removing common state evolution.
    pred_centered = final_pred - final_pred.mean(dim=1, keepdim=True)
    target_centered = final_target - final_target.mean(dim=1, keepdim=True)
    branch_delta_cosine = F.cosine_similarity(
        pred_centered.flatten(2), target_centered.flatten(2), dim=-1
    ).mean()
    predicted_branch_spread = pred_centered.abs().mean()
    target_branch_spread = target_centered.abs().mean()
    result = {
        "task_id": args.task_id,
        "num_forks": int(before.shape[0]),
        "num_branches": int(actions.shape[1]),
        "metrics": {
            "persistence_mae": float(persistence_mae),
            "action_agnostic_oracle_mean_mae": float(oracle_mean_mae),
            "correct_action_mae": float(correct_mae),
            "shuffled_action_mae": float(shuffled_mae),
            "shuffle_relative_degradation": float((shuffled_mae - correct_mae) / correct_mae),
            "branch_retrieval_accuracy": float(retrieval),
            "branch_retrieval_chance": 1.0 / actions.shape[1],
            "correct_vs_shifted_pairwise_win_rate": float(pairwise_win),
            "branch_delta_cosine": float(branch_delta_cosine),
            "predicted_branch_spread": float(predicted_branch_spread),
            "target_branch_spread": float(target_branch_spread),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
