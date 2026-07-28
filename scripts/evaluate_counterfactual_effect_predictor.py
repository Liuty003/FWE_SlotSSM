#!/usr/bin/env python
"""Evaluate differential Slot action effects on a held-out task."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from scripts.train_counterfactual_effect_predictor import CounterfactualEffectPredictor, physical_target


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task-id", type=int, default=9)
    p.add_argument("--episode-ids", default="0,1,2,3,4")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    episodes = [int(value) for value in args.episode_ids.split(",") if value]
    parts = [load_file(str(Path(args.data_dir) / f"libero_spatial_t{args.task_id}_e{episode}_slots.safetensors")) for episode in episodes]
    keys = (
        "before_slots", "after_slots", "after_valid", "executed_actions", "joint_positions",
        "eef_delta_m", "max_object_delta_m", "max_object_displacement_m",
    )
    data = {key: torch.cat([part[key] for part in parts]) for key in keys}
    state = load_file(args.checkpoint)
    model = CounterfactualEffectPredictor(torch.ones(7), torch.zeros(7), torch.ones(7))
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    before = data["before_slots"].to(device).float()
    actions = data["executed_actions"].to(device).float()
    joint = data["joint_positions"].to(device).float()
    after = data["after_slots"].to(device).float()
    target = after - after[:, :1]
    physical = physical_target(data, torch.arange(len(before)), device)
    valid = data["after_valid"].to(device)
    with torch.inference_mode():
        prediction, physical_prediction = model(before, actions[:, 0], actions, joint)
        shuffled, shuffled_physical = model(before, actions[:, 0], actions.roll(1, 1), joint)
    mask = valid[..., None, None].float()
    denominator = (mask.sum() * target.shape[-2] * target.shape[-1]).clamp_min(1)
    mae = ((prediction - target).abs() * mask).sum() / denominator
    shuffled_mae = ((shuffled - target).abs() * mask).sum() / denominator
    zero_mae = (target.abs() * mask).sum() / denominator
    final_pred, final_target = prediction[:, :, -1], target[:, :, -1]
    distances = (final_pred[:, :, None] - final_target[:, None]).abs().mean((-1, -2))
    labels = torch.arange(actions.shape[1], device=device).view(1, -1)
    retrieval = distances.argmin(-1).eq(labels).float().mean()
    shifted = ((labels + 1) % actions.shape[1]).expand(before.shape[0], -1)
    correct_distance = distances.diagonal(dim1=1, dim2=2)
    shifted_distance = distances.gather(2, shifted[..., None]).squeeze(-1)
    win_rate = correct_distance.lt(shifted_distance).float().mean()
    pred_centered = final_pred - final_pred.mean(1, keepdim=True)
    target_centered = final_target - final_target.mean(1, keepdim=True)
    cosine = F.cosine_similarity(pred_centered.flatten(2), target_centered.flatten(2), dim=-1).mean()
    physical_denominator = (valid.sum() * physical.shape[-1]).clamp_min(1)
    physical_mae_cm = 100.0 * ((physical_prediction - physical).abs() * valid[..., None]).sum() / physical_denominator
    shuffled_physical_mae_cm = 100.0 * ((shuffled_physical - physical).abs() * valid[..., None]).sum() / physical_denominator
    zero_physical_mae_cm = 100.0 * (physical.abs() * valid[..., None]).sum() / physical_denominator
    result = {
        "task_id": args.task_id, "episode_ids": episodes, "num_forks": int(before.shape[0]),
        "metrics": {
            "zero_effect_mae": float(zero_mae), "effect_mae": float(mae),
            "shuffled_effect_mae": float(shuffled_mae),
            "relative_improvement_over_zero": float((zero_mae - mae) / zero_mae),
            "shuffle_relative_degradation": float((shuffled_mae - mae) / mae),
            "branch_retrieval_accuracy": float(retrieval), "branch_retrieval_chance": 0.1,
            "correct_vs_shifted_win_rate": float(win_rate), "branch_delta_cosine": float(cosine),
            "predicted_branch_spread": float(pred_centered.abs().mean()),
            "target_branch_spread": float(target_centered.abs().mean()),
            "zero_physical_mae_cm": float(zero_physical_mae_cm),
            "physical_mae_cm": float(physical_mae_cm),
            "shuffled_physical_mae_cm": float(shuffled_physical_mae_cm),
        },
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)); print(f"saved={output}")


if __name__ == "__main__":
    main()
