#!/usr/bin/env python
"""Train a non-privileged MLP recovery residual on episode-held-out data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file

PHASES = ("RETURN TO BOWL", "DESCEND", "CLOSE GRIPPER", "VERIFY LIFT")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--visual-3d-audit", type=Path)
    parser.add_argument("--feature-set", choices=("full", "geometry"), default="full")
    return parser.parse_args()


class RecoveryResidualMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 7),
        )

    def forward(self, inputs):
        return self.network(inputs)


def vectorize(row, visual_xyz=None, feature_set="full"):
    measurement = row["target_mask_measurement"]
    centroid = measurement["centroid_xy"]
    if centroid is None:
        raise RuntimeError(f"missing target centroid at timestep {row['timestep']}")
    phase = np.zeros(len(PHASES), np.float32)
    phase[PHASES.index(row["phase"])] = 1
    nominal_feature = np.asarray(row["nominal_action"], np.float32)
    if feature_set == "geometry":
        if visual_xyz is None:
            raise RuntimeError("geometry feature set requires visual xyz")
        visual_xyz = np.asarray(visual_xyz, np.float32)
        eef = np.asarray(row["robot_state"]["robot0_eef_pos"], np.float32)
        parts = [visual_xyz - eef, nominal_feature, phase]
    else:
        parts = [
            np.asarray(centroid, np.float32) / 256,
            np.asarray([
                measurement["visible_pixels"] / 4096,
                np.tanh(measurement["selection_margin"] / 10),
            ], np.float32),
            np.asarray(row["robot_state"]["robot0_proprio-state"], np.float32),
            nominal_feature,
            phase,
        ]
        if visual_xyz is not None:
            visual_xyz = np.asarray(visual_xyz, np.float32)
            eef = np.asarray(row["robot_state"]["robot0_eef_pos"], np.float32)
            parts.extend((visual_xyz, visual_xyz - eef))
    features = np.concatenate(parts)
    nominal = np.asarray(row["nominal_action"], np.float32)
    expert = np.asarray(row["expert_or_executed_action"], np.float32)
    return features, expert - nominal, nominal, expert, row["phase"]


def load_split(manifest, split, visual_estimates=None, feature_set="full"):
    values = []
    episode_ids = []
    for episode in manifest["episodes"]:
        if episode["split"] != split or not episode["eligible_for_recovery_bc"]:
            continue
        rows = [
            json.loads(line)
            for line in Path(episode["transitions"]).read_text().splitlines()
            if line
        ]
        selected = [row for row in rows if row["is_recovery_expert_action"]]
        values.extend(
            vectorize(
                row,
                visual_estimates[
                    f"{episode['episode_id']}:{row['timestep']}"
                ] if visual_estimates is not None else None,
                feature_set,
            )
            for row in selected
        )
        episode_ids.extend([episode["episode_id"]] * len(selected))
    if not values:
        raise RuntimeError(f"split {split} has no eligible expert rows")
    features, residuals, nominals, experts, phases = zip(*values)
    return {
        "features": torch.from_numpy(np.stack(features)),
        "residuals": torch.from_numpy(np.stack(residuals)),
        "nominals": torch.from_numpy(np.stack(nominals)),
        "experts": torch.from_numpy(np.stack(experts)),
        "phases": phases,
        "episode_ids": episode_ids,
    }


def metrics(predicted, data, phase_means):
    predicted_action = (data["nominals"] + predicted).clamp(-1, 1)
    expert = data["experts"]
    nominal_error = (data["nominals"] - expert).abs()
    model_error = (predicted_action - expert).abs()
    phase_prediction = torch.stack([
        data["nominals"][index] + phase_means[phase]
        for index, phase in enumerate(data["phases"])
    ]).clamp(-1, 1)
    phase_error = (phase_prediction - expert).abs()
    return {
        "num_rows": len(expert),
        "episode_ids": sorted(set(data["episode_ids"])),
        "action_mae": float(model_error.mean()),
        "translation_mae": float(model_error[:, :3].mean()),
        "rotation_mae": float(model_error[:, 3:6].mean()),
        "gripper_sign_accuracy": float(
            (predicted_action[:, 6].sign() == expert[:, 6].sign()).float().mean()
        ),
        "nominal_action_mae": float(nominal_error.mean()),
        "phase_mean_action_mae": float(phase_error.mean()),
        "improvement_over_nominal_percent": float(
            100 * (1 - model_error.mean() / nominal_error.mean().clamp_min(1e-8))
        ),
    }


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    manifest = json.loads(args.manifest.read_text())
    visual_estimates = None
    visual_method = None
    if args.visual_3d_audit is not None:
        audit = json.loads(args.visual_3d_audit.read_text())
        visual_method = audit["selection"]["selected_method"]
        visual_estimates = (
            audit["calibrated_frame_estimates"]
            if "calibrated_frame_estimates" in audit else {
                key: methods[visual_method]
                for key, methods in audit["frame_estimates"].items()
            }
        )
    splits = {
        name: load_split(manifest, name, visual_estimates, args.feature_set)
        for name in ("train", "validation", "test")
    }
    train = splits["train"]
    feature_mean = train["features"].mean(0)
    # Two training episodes make several proprio dimensions nearly constant.
    # A tiny denominator turns harmless episode shifts into huge extrapolation.
    feature_std = train["features"].std(0).clamp_min(0.05)
    residual_mean = train["residuals"].mean(0)
    residual_std = train["residuals"].std(0).clamp_min(1e-4)
    phase_means = {
        phase: train["residuals"][
            torch.tensor([value == phase for value in train["phases"]])
        ].mean(0)
        for phase in PHASES
    }
    model = RecoveryResidualMLP(train["features"].shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3
    )
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    best_validation_mae = float("inf")
    best_step = None
    best_state = None
    model.train()
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            len(train["features"]), (args.batch_size,), generator=generator
        )
        inputs = (train["features"][indices] - feature_mean) / feature_std
        targets = (train["residuals"][indices] - residual_mean) / residual_std
        prediction = model(inputs)
        loss = torch.nn.functional.smooth_l1_loss(prediction, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == args.steps or step % 25 == 0:
            model.eval()
            with torch.inference_mode():
                validation = splits["validation"]
                normalized = (
                    validation["features"] - feature_mean
                ) / feature_std
                validation_residual = (
                    model(normalized) * residual_std + residual_mean
                )
                validation_mae = metrics(
                    validation_residual, validation, phase_means
                )["action_mae"]
            if validation_mae < best_validation_mae:
                best_validation_mae = validation_mae
                best_step = step
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
            if step == 1 or step == args.steps or step % 250 == 0:
                history.append({
                    "step": step, "loss": float(loss),
                    "validation_action_mae": validation_mae,
                })
            model.train()
    model.load_state_dict(best_state)
    model.eval()
    reports = {}
    with torch.inference_mode():
        for name, data in splits.items():
            normalized = (data["features"] - feature_mean) / feature_std
            residual = model(normalized) * residual_std + residual_mean
            reports[name] = metrics(residual, data, phase_means)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "recovery-residual-mlp.safetensors"
    tensors = {f"model.{key}": value for key, value in model.state_dict().items()}
    tensors.update({
        "feature_mean": feature_mean, "feature_std": feature_std,
        "residual_mean": residual_mean, "residual_std": residual_std,
        **{f"phase_mean.{index}": phase_means[phase]
           for index, phase in enumerate(PHASES)},
    })
    save_file(tensors, checkpoint)
    result = {
        "schema_version": 1,
        "kind": "non_privileged_recovery_residual_smoke",
        "manifest": str(args.manifest),
        "input_contract": (
            "Stage-B mask centroid/area/margin + robot proprio + nominal action "
            "+ oracle recovery phase"
            + (
                " + mask-depth visual target xyz and target-eef delta"
                if visual_estimates is not None else ""
            )
            + "; no simulator object state"
        ),
        "visual_3d_audit": (
            str(args.visual_3d_audit) if args.visual_3d_audit is not None else None
        ),
        "visual_3d_method": visual_method,
        "feature_set": args.feature_set,
        "visual_3d_bias_calibrated": (
            visual_estimates is not None
            and args.visual_3d_audit is not None
            and "calibrated_frame_estimates" in audit
        ),
        "target": "expert_action - nominal_openvla_action",
        "history": history,
        "selection": {
            "criterion": "minimum validation action MAE",
            "best_step": best_step,
            "best_validation_action_mae": best_validation_mae,
        },
        "metrics": reports,
        "checkpoint": str(checkpoint),
        "claim_limit": (
            "offline smoke only; phase is oracle and test episodes share task9"
        ),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
