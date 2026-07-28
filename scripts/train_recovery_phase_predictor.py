#!/usr/bin/env python
"""Train and audit a monotonic recovery phase predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file

PHASES = ("RETURN TO BOWL", "DESCEND", "CLOSE GRIPPER", "VERIFY LIFT")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual-3d-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--extra-online-teacher-traces", type=Path, nargs="*", default=[]
    )
    return parser.parse_args()


class PhasePredictor(nn.Module):
    def __init__(self, input_dim=11):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32), nn.GELU(),
            nn.Linear(32, 32), nn.GELU(),
            nn.Linear(32, len(PHASES)),
        )

    def forward(self, inputs):
        return self.network(inputs)


def phase_feature(row, target_xyz, previous_phase):
    robot = row.get("robot_state", row.get("robot_state_before"))
    eef = np.asarray(robot["robot0_eef_pos"], np.float32)
    gripper = np.asarray(robot["robot0_gripper_qpos"], np.float32)
    gripper_velocity = np.asarray(
        robot["robot0_gripper_qvel"], np.float32
    )
    previous = np.zeros(len(PHASES), np.float32)
    previous[previous_phase] = 1
    return np.concatenate((
        np.asarray(target_xyz, np.float32) - eef,
        gripper, gripper_velocity, previous,
    ))


def load_episodes(manifest, visual, split):
    episodes = []
    for record in manifest["episodes"]:
        if record["split"] != split or not record["eligible_for_recovery_bc"]:
            continue
        rows = [
            json.loads(line)
            for line in Path(record["transitions"]).read_text().splitlines() if line
        ]
        rows = [row for row in rows if row["is_recovery_expert_action"]]
        labels = [PHASES.index(row["phase"]) for row in rows]
        features = []
        for index, row in enumerate(rows):
            previous = labels[index - 1] if index else 0
            key = f"{record['episode_id']}:{row['timestep']}"
            features.append(phase_feature(row, visual[key], previous))
        episodes.append({
            "episode_id": record["episode_id"], "rows": rows,
            "features": torch.from_numpy(np.stack(features)),
            "labels": torch.tensor(labels, dtype=torch.long),
        })
    if not episodes:
        raise RuntimeError(f"no eligible episodes in {split}")
    return episodes


def flatten(episodes):
    return (
        torch.cat([episode["features"] for episode in episodes]),
        torch.cat([episode["labels"] for episode in episodes]),
    )


def load_online_teacher_trace(path, synthetic_episode_id):
    rows = [
        json.loads(line) for line in path.read_text().splitlines() if line
    ]
    rows = [
        row for row in rows
        if row.get("learned_recovery_residual") is not None
        and row["phase"] in PHASES
    ]
    labels = [PHASES.index(row["phase"]) for row in rows]
    features = []
    for index, row in enumerate(rows):
        previous = labels[index - 1] if index else 0
        features.append(phase_feature(
            row, row["visual_target_xyz"], previous
        ))
    return {
        "episode_id": synthetic_episode_id,
        "rows": rows,
        "features": torch.from_numpy(np.stack(features)),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def transition_steps(labels):
    result = {}
    for index, phase in enumerate(PHASES):
        positions = np.where(np.asarray(labels) == index)[0]
        result[phase] = int(positions[0]) if len(positions) else None
    return result


@torch.inference_mode()
def evaluate(model, episodes, mean, std, visual):
    teacher_correct = 0
    teacher_total = 0
    autoregressive_correct = 0
    episode_reports = []
    for episode in episodes:
        teacher_logits = model((episode["features"] - mean) / std)
        teacher_prediction = teacher_logits.argmax(1)
        labels = episode["labels"]
        teacher_correct += int((teacher_prediction == labels).sum())
        teacher_total += len(labels)
        previous = 0
        predictions = []
        for row in episode["rows"]:
            key = f"{episode['episode_id']}:{row['timestep']}"
            target_xyz = (
                row["visual_target_xyz"]
                if row.get("visual_target_xyz") is not None else visual[key]
            )
            feature = torch.from_numpy(
                phase_feature(row, target_xyz, previous)
            )[None]
            logits = model((feature - mean) / std)[0]
            legal = torch.full_like(logits, float("-inf"))
            legal[previous:min(previous + 2, len(PHASES))] = logits[
                previous:min(previous + 2, len(PHASES))
            ]
            previous = int(legal.argmax())
            predictions.append(previous)
        autoregressive_correct += sum(
            prediction == int(label)
            for prediction, label in zip(predictions, labels)
        )
        truth_steps = transition_steps(labels.tolist())
        predicted_steps = transition_steps(predictions)
        delays = {
            phase: (
                predicted_steps[phase] - truth_steps[phase]
                if predicted_steps[phase] is not None
                and truth_steps[phase] is not None else None
            )
            for phase in PHASES
        }
        episode_reports.append({
            "episode_id": episode["episode_id"],
            "num_steps": len(labels),
            "teacher_forced_accuracy": float(
                (teacher_prediction == labels).float().mean()
            ),
            "autoregressive_accuracy": float(np.mean(
                np.asarray(predictions) == labels.numpy()
            )),
            "truth_transition_steps": truth_steps,
            "predicted_transition_steps": predicted_steps,
            "transition_delay_steps": delays,
            "reached_verify": PHASES[-1] in [
                PHASES[index] for index in predictions
            ],
            "monotonic": all(
                current <= following
                for current, following in zip(predictions, predictions[1:])
            ),
        })
    total = sum(len(episode["labels"]) for episode in episodes)
    return {
        "num_episodes": len(episodes), "num_steps": total,
        "teacher_forced_accuracy": teacher_correct / teacher_total,
        "autoregressive_accuracy": autoregressive_correct / total,
        "all_reached_verify": all(
            report["reached_verify"] for report in episode_reports
        ),
        "all_monotonic": all(report["monotonic"] for report in episode_reports),
        "max_absolute_transition_delay": max(
            abs(delay)
            for report in episode_reports
            for delay in report["transition_delay_steps"].values()
            if delay is not None
        ),
        "episodes": episode_reports,
    }


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    manifest = json.loads(args.manifest.read_text())
    audit = json.loads(args.visual_3d_audit.read_text())
    visual = audit["calibrated_frame_estimates"]
    episodes = {
        split: load_episodes(manifest, visual, split)
        for split in ("train", "validation", "test")
    }
    for index, path in enumerate(args.extra_online_teacher_traces):
        episodes["train"].append(load_online_teacher_trace(path, -100 - index))
    train_features, train_labels = flatten(episodes["train"])
    mean = train_features.mean(0)
    std = train_features.std(0).clamp_min(0.02)
    counts = torch.bincount(train_labels, minlength=len(PHASES)).float()
    weights = counts.sum() / counts.clamp_min(1)
    weights = weights / weights.mean()
    model = PhasePredictor(train_features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    best_accuracy, best_step, best_state = -1.0, None, None
    history = []
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            len(train_features), (args.batch_size,), generator=generator
        )
        logits = model((train_features[indices] - mean) / std)
        loss = torch.nn.functional.cross_entropy(
            logits, train_labels[indices], weight=weights
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            model.eval()
            validation = evaluate(
                model, episodes["validation"], mean, std, visual
            )
            score = validation["autoregressive_accuracy"]
            if score > best_accuracy:
                best_accuracy, best_step = score, step
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
            if step == 1 or step % 250 == 0 or step == args.steps:
                history.append({
                    "step": step, "loss": float(loss),
                    "validation_autoregressive_accuracy": score,
                })
            model.train()
    model.load_state_dict(best_state)
    model.eval()
    metrics = {
        split: evaluate(model, values, mean, std, visual)
        for split, values in episodes.items()
    }
    test = metrics["test"]
    passed = (
        test["autoregressive_accuracy"] >= 0.9
        and test["all_reached_verify"]
        and test["all_monotonic"]
        and test["max_absolute_transition_delay"] <= 5
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "recovery-phase-predictor.safetensors"
    tensors = {
        **{f"model.{key}": value for key, value in model.state_dict().items()},
        "feature_mean": mean, "feature_std": std,
    }
    save_file(tensors, checkpoint)
    result = {
        "schema_version": 1,
        "kind": "monotonic_recovery_phase_predictor",
        "manifest": str(args.manifest),
        "visual_3d_audit": str(args.visual_3d_audit),
        "phases": list(PHASES),
        "input_contract": (
            "visual target-eef xyz + gripper qpos/qvel + previous phase"
        ),
        "extra_online_teacher_traces": [
            str(path) for path in args.extra_online_teacher_traces
        ],
        "history": history,
        "selection": {
            "best_step": best_step,
            "criterion": "validation autoregressive accuracy",
        },
        "metrics": metrics,
        "offline_gate_passed": passed,
        "checkpoint": str(checkpoint),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
