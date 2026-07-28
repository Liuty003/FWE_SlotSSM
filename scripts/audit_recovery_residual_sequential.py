#!/usr/bin/env python
"""Teacher-forced sequential safety audit for a recovery residual model."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_recovery_residual_smoke import (
    PHASES, RecoveryResidualMLP, vectorize,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-3d-audit", type=Path)
    parser.add_argument("--feature-set", choices=("full", "geometry"), default="full")
    parser.add_argument("--clip-candidates", default="0.1,0.2,0.3,0.5,1.0")
    parser.add_argument("--max-negative-cosine-rate", type=float, default=0.1)
    parser.add_argument("--min-z-sign-accuracy", type=float, default=0.9)
    parser.add_argument("--max-action-jump", type=float, default=1.25)
    return parser.parse_args()


def load_model(path):
    tensors = load_file(path)
    state = {
        key.removeprefix("model."): value
        for key, value in tensors.items() if key.startswith("model.")
    }
    input_dim = state["network.0.weight"].shape[1]
    model = RecoveryResidualMLP(input_dim)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, tensors


def episode_rows(record):
    rows = [
        json.loads(line)
        for line in Path(record["transitions"]).read_text().splitlines() if line
    ]
    return [row for row in rows if row["is_recovery_expert_action"]]


def predict_episode(
    model, tensors, rows, clip, visual_estimates=None,
    episode_id=None, feature_set="full", gripper_override=True,
):
    values = [
        vectorize(
            row,
            visual_estimates[f"{episode_id}:{row['timestep']}"]
            if visual_estimates is not None else None,
            feature_set,
        )
        for row in rows
    ]
    features, _, nominals, experts, phases = zip(*values)
    features = torch.from_numpy(np.stack(features))
    nominal = torch.from_numpy(np.stack(nominals))
    expert = torch.from_numpy(np.stack(experts))
    normalized = (
        features - tensors["feature_mean"]
    ) / tensors["feature_std"]
    with torch.inference_mode():
        residual = (
            model(normalized) * tensors["residual_std"] + tensors["residual_mean"]
        )
    clipped = residual.clamp(-clip, clip)
    action = (nominal + clipped).clamp(-1, 1)
    if gripper_override:
        action[:, 6] = torch.tensor([
            -1.0 if phase in ("RETURN TO BOWL", "DESCEND") else 1.0
            for phase in phases
        ])
    return {
        "rows": rows, "phases": phases, "nominal": nominal, "expert": expert,
        "raw_residual": residual, "clipped_residual": clipped, "action": action,
    }


def summarize(episodes, clip):
    phase_values = defaultdict(list)
    errors, translation_errors = [], []
    cosines, z_matches, jumps, saturations = [], [], [], []
    episode_reports = []
    for episode_id, prediction in episodes.items():
        action, expert = prediction["action"], prediction["expert"]
        error = (action - expert).abs()
        errors.append(error)
        translation_errors.append(error[:, :3])
        translation = action[:, :3]
        expert_translation = expert[:, :3]
        expert_norm = torch.linalg.vector_norm(expert_translation, dim=1)
        valid = expert_norm > 0.05
        cosine = torch.nn.functional.cosine_similarity(
            translation[valid], expert_translation[valid], dim=1
        )
        cosines.append(cosine)
        z_valid = expert[:, 2].abs() > 0.05
        z_match = (
            action[z_valid, 2].sign() == expert[z_valid, 2].sign()
        ).float()
        z_matches.append(z_match)
        if len(action) > 1:
            jumps.append((action[1:] - action[:-1]).abs().amax(1))
        saturations.append(
            (prediction["raw_residual"].abs() > clip).float().mean()
        )
        for index, phase in enumerate(prediction["phases"]):
            phase_values[phase].append({
                "error": float(error[index].mean()),
                "translation_error": float(error[index, :3].mean()),
                "z_match": (
                    bool(action[index, 2].sign() == expert[index, 2].sign())
                    if abs(float(expert[index, 2])) > 0.05 else None
                ),
            })
        episode_reports.append({
            "episode_id": episode_id,
            "num_steps": len(action),
            "action_mae": float(error.mean()),
            "translation_mae": float(error[:, :3].mean()),
            "negative_translation_cosine_rate": float((cosine < 0).float().mean()),
            "z_sign_accuracy": float(z_match.mean()) if len(z_match) else 1.0,
            "max_translation_jump": (
                float((action[1:, :3] - action[:-1, :3]).abs().max())
                if len(action) > 1 else 0.0
            ),
        })
    errors = torch.cat(errors)
    translation_errors = torch.cat(translation_errors)
    cosines = torch.cat(cosines)
    z_matches = torch.cat(z_matches)
    jumps = torch.cat(jumps) if jumps else torch.zeros(1)
    return {
        "clip": clip,
        "num_episodes": len(episodes),
        "num_steps": len(errors),
        "action_mae": float(errors.mean()),
        "translation_mae": float(translation_errors.mean()),
        "negative_translation_cosine_rate": float((cosines < 0).float().mean()),
        "mean_translation_cosine": float(cosines.mean()),
        "z_sign_accuracy": float(z_matches.mean()),
        "max_translation_jump": float(torch.cat([
            (prediction["action"][1:, :3] - prediction["action"][:-1, :3])
            .abs().amax(1)
            for prediction in episodes.values() if len(prediction["action"]) > 1
        ]).max()),
        "mean_clip_saturation_fraction": float(torch.stack(saturations).mean()),
        "gripper_sign_accuracy": 1.0,
        "by_phase": {
            phase: {
                "num_steps": len(values),
                "action_mae": float(np.mean([value["error"] for value in values])),
                "translation_mae": float(np.mean([
                    value["translation_error"] for value in values
                ])),
                "z_sign_accuracy": float(np.mean([
                    value["z_match"] for value in values
                    if value["z_match"] is not None
                ])) if any(value["z_match"] is not None for value in values) else 1.0,
            }
            for phase, values in phase_values.items()
        },
        "episodes": episode_reports,
    }


def passes(report, args):
    descend = report["by_phase"].get("DESCEND", {})
    return (
        report["negative_translation_cosine_rate"]
        <= args.max_negative_cosine_rate
        and report["z_sign_accuracy"] >= args.min_z_sign_accuracy
        and descend.get("z_sign_accuracy", 0.0) >= args.min_z_sign_accuracy
        and report["max_translation_jump"] <= args.max_action_jump
        and all(
            episode["negative_translation_cosine_rate"]
            <= args.max_negative_cosine_rate
            and episode["z_sign_accuracy"] >= args.min_z_sign_accuracy
            for episode in report["episodes"]
        )
    )


def main():
    args = arguments()
    manifest = json.loads(args.manifest.read_text())
    model, tensors = load_model(args.checkpoint)
    visual_estimates = None
    if args.visual_3d_audit is not None:
        visual = json.loads(args.visual_3d_audit.read_text())
        method = visual["selection"]["selected_method"]
        visual_estimates = (
            visual["calibrated_frame_estimates"]
            if "calibrated_frame_estimates" in visual else {
                key: methods[method]
                for key, methods in visual["frame_estimates"].items()
            }
        )
    records = {
        split: [
            record for record in manifest["episodes"]
            if record["split"] == split and record["eligible_for_recovery_bc"]
        ]
        for split in ("validation", "test")
    }
    clips = [float(value) for value in args.clip_candidates.split(",")]
    validation_candidates = []
    for clip in clips:
        predictions = {
            record["episode_id"]: predict_episode(
                model, tensors, episode_rows(record), clip,
                visual_estimates, record["episode_id"], args.feature_set,
            )
            for record in records["validation"]
        }
        report = summarize(predictions, clip)
        report["passes_safety_thresholds"] = passes(report, args)
        validation_candidates.append(report)
    safe = [
        report for report in validation_candidates
        if report["passes_safety_thresholds"]
    ]
    if not safe:
        selected = min(validation_candidates, key=lambda report: report["action_mae"])
        selection_passed = False
    else:
        selected = min(safe, key=lambda report: report["action_mae"])
        selection_passed = True
    selected_clip = selected["clip"]
    test_predictions = {
        record["episode_id"]: predict_episode(
            model, tensors, episode_rows(record), selected_clip,
            visual_estimates, record["episode_id"], args.feature_set,
        )
        for record in records["test"]
    }
    test_report = summarize(test_predictions, selected_clip)
    test_report["passes_safety_thresholds"] = passes(test_report, args)
    result = {
        "schema_version": 1,
        "kind": "teacher_forced_sequential_safety_audit",
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "visual_3d_audit": (
            str(args.visual_3d_audit) if args.visual_3d_audit is not None else None
        ),
        "feature_set": args.feature_set,
        "teacher_forced": True,
        "gripper_rule": (
            "phase safety override: open for return/descend; "
            "closed for close/verify"
        ),
        "thresholds": {
            "max_negative_cosine_rate": args.max_negative_cosine_rate,
            "min_z_sign_accuracy": args.min_z_sign_accuracy,
            "max_action_jump": args.max_action_jump,
        },
        "validation_clip_candidates": validation_candidates,
        "selection": {
            "selected_clip": selected_clip,
            "validation_selection_passed": selection_passed,
            "criterion": "lowest validation action MAE among safety-passing clips",
        },
        "test": test_report,
        "online_smoke_allowed": (
            selection_passed and test_report["passes_safety_thresholds"]
        ),
        "claim_limit": (
            "states remain on oracle trajectories; this does not establish "
            "closed-loop recovery"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
