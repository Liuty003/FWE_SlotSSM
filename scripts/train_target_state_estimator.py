#!/usr/bin/env python
"""Train language-conditioned target grounding and phase estimation from Stage-1 slots."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.train_task_aware_action_scorer import language_tokens

TARGET = "akita_black_bowl_1_pos"
GOAL = "plate_1_pos"


def ids(value):
    return [int(x) for x in value.split(",") if x]


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--slot-dir", required=True)
    p.add_argument("--counterfactual-root", required=True)
    p.add_argument("--failure-root", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-tasks", default="0,1,2,3,4,5,6,7,8")
    p.add_argument("--val-tasks", default="9")
    p.add_argument("--train-episodes", default="0,1,2,3")
    p.add_argument("--val-episodes", default="4")
    p.add_argument("--train-failure-episodes", default="")
    p.add_argument("--val-failure-episodes", default="")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260721)
    return p.parse_args()


def load_split(slot_dir, raw_root, tasks, episodes, lift_threshold=0.015):
    parts = []
    for task in tasks:
        for episode in episodes:
            slot = load_file(str(slot_dir / f"libero_spatial_t{task}_e{episode}_slots.safetensors"))
            episode_dir = raw_root / f"libero_spatial_t{task}_e{episode}_h4_dual"
            metadata = json.loads((episode_dir / "metadata.json").read_text())
            grouped = defaultdict(dict)
            for line in (episode_dir / "transitions.jsonl").open():
                row = json.loads(line); grouped[int(row["fork_id"])][row["branch"]] = row
            first = grouped[min(grouped)]["nominal"]["privileged_object_state_before"]
            initial_z = float(first[TARGET][2])
            relative, target_goal, lifted = [], [], []
            for fork in slot["fork_ids"].tolist():
                row = grouped[int(fork)]["nominal"]
                objects = row["privileged_object_state_before"]
                robot = row["robot_state_before"]
                target = np.asarray(objects[TARGET], np.float32)
                goal = np.asarray(objects[GOAL], np.float32)
                eef = np.asarray(robot["robot0_eef_pos"], np.float32)
                relative.append(target - eef)
                target_goal.append(goal - target)
                lifted.append(target[2] > initial_z + lift_threshold)
            count = len(relative)
            parts.append({
                "slots": slot["before_slots"].float(),
                "joints": slot["joint_positions"].float(),
                "language": language_tokens(metadata["task_language"])[None].expand(count, -1),
                "target_minus_eef": torch.tensor(np.asarray(relative)),
                "goal_minus_target": torch.tensor(np.asarray(target_goal)),
                "lifted": torch.tensor(lifted, dtype=torch.float32),
            })
    return {k: torch.cat([x[k] for x in parts]) for k in parts[0]}


def load_failure_split(root, episodes, lift_threshold=0.015):
    parts = []
    for episode in episodes:
        episode_dir = root / f"task9-episode{episode}"
        trace = load_file(str(episode_dir / "slots.safetensors"))["slots"].float()
        rows = [json.loads(line) for line in (episode_dir / "steps.jsonl").open()]
        if len(rows) != len(trace):
            raise RuntimeError(f"failure episode {episode}: rows={len(rows)} slots={len(trace)}")
        metadata = json.loads((episode_dir / "metadata.json").read_text())
        initial_z = float(rows[0]["privileged_object_state_before"][TARGET][2])
        relative, target_goal, lifted, joints = [], [], [], []
        for row in rows:
            objects = row["privileged_object_state_before"]
            robot = row["robot_state_before"]
            target = np.asarray(objects[TARGET], np.float32)
            goal = np.asarray(objects[GOAL], np.float32)
            eef = np.asarray(robot["robot0_eef_pos"], np.float32)
            relative.append(target - eef); target_goal.append(goal - target)
            lifted.append(target[2] > initial_z + lift_threshold)
            joints.append(robot["robot0_joint_pos"])
        count = len(rows)
        parts.append({
            "slots": trace, "joints": torch.tensor(np.asarray(joints), dtype=torch.float32),
            "language": language_tokens(metadata["task_language"])[None].expand(count, -1),
            "target_minus_eef": torch.tensor(np.asarray(relative)),
            "goal_minus_target": torch.tensor(np.asarray(target_goal)),
            "lifted": torch.tensor(lifted, dtype=torch.float32),
        })
    return {k: torch.cat([x[k] for x in parts]) for k in parts[0]}


def concatenate(*datasets):
    datasets = [data for data in datasets if data is not None]
    return {key: torch.cat([data[key] for data in datasets]) for key in datasets[0]}


class TargetStateEstimator(torch.nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.slot_norm = torch.nn.LayerNorm(dim)
        self.language_embedding = torch.nn.Embedding(4096, 128, padding_idx=0)
        self.language_projection = torch.nn.Sequential(
            torch.nn.Linear(128, dim), torch.nn.GELU(), torch.nn.Linear(dim, dim)
        )
        self.joint_projection = torch.nn.Sequential(
            torch.nn.Linear(7, 128), torch.nn.GELU(), torch.nn.Linear(128, dim)
        )
        self.target_attention = torch.nn.MultiheadAttention(dim, 8, batch_first=True)
        self.goal_attention = torch.nn.MultiheadAttention(dim, 8, batch_first=True)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(dim * 3), torch.nn.Linear(dim * 3, 512),
            torch.nn.GELU(), torch.nn.Linear(512, 7),
        )

    def forward(self, slots, joints, language, return_attention=False):
        values = self.slot_norm(slots)
        mask = language.ne(0)[..., None]
        lang = (self.language_embedding(language) * mask).sum(1) / mask.sum(1).clamp_min(1)
        query = self.language_projection(lang)[:, None]
        target, target_weights = self.target_attention(query, values, values)
        goal_query = query + target
        goal, goal_weights = self.goal_attention(goal_query, values, values)
        joint = self.joint_projection(joints)[:, None]
        output = self.head(torch.cat((target, goal, joint), -1))[:, 0]
        if return_attention:
            return output, target_weights[:, 0], goal_weights[:, 0]
        return output


def losses(output, data, index):
    target = data["target_minus_eef"][index].to(output.device)
    goal = data["goal_minus_target"][index].to(output.device)
    lifted = data["lifted"][index].to(output.device)
    target_loss = F.smooth_l1_loss(output[:, :3] * 100, target * 100)
    goal_loss = F.smooth_l1_loss(output[:, 3:6] * 100, goal * 100)
    phase_loss = F.binary_cross_entropy_with_logits(output[:, 6], lifted)
    return target_loss + 0.5 * goal_loss + phase_loss, (target_loss, goal_loss, phase_loss)


@torch.inference_mode()
def evaluate(model, data, device):
    model.eval(); outputs = []
    for start in range(0, len(data["slots"]), 128):
        sl = slice(start, start + 128)
        outputs.append(model(data["slots"][sl].to(device), data["joints"][sl].to(device), data["language"][sl].to(device)).cpu())
    out = torch.cat(outputs); target = data["target_minus_eef"]; goal = data["goal_minus_target"]; lifted = data["lifted"]
    pred = out[:, 6].sigmoid() > .5
    result = {
        "samples": len(out),
        "target_relative_mae_cm": float((out[:, :3] - target).abs().mean() * 100),
        "target_distance_mae_cm": float(((out[:, :3].norm(dim=1) - target.norm(dim=1)).abs().mean()) * 100),
        "goal_relative_mae_cm": float((out[:, 3:6] - goal).abs().mean() * 100),
        "lifted_accuracy_percent": float((pred == lifted.bool()).float().mean() * 100),
        "lifted_positive_percent": float(lifted.mean() * 100),
    }
    model.train(); return result


def main():
    args = arguments(); torch.manual_seed(args.seed)
    slot_dir, raw_root = Path(args.slot_dir), Path(args.counterfactual_root)
    train_normal = load_split(slot_dir, raw_root, ids(args.train_tasks), ids(args.train_episodes))
    val_normal = load_split(slot_dir, raw_root, ids(args.val_tasks), ids(args.val_episodes))
    failure_root = Path(args.failure_root) if args.failure_root else None
    train_failure = load_failure_split(failure_root, ids(args.train_failure_episodes)) if failure_root and ids(args.train_failure_episodes) else None
    val_failure = load_failure_split(failure_root, ids(args.val_failure_episodes)) if failure_root and ids(args.val_failure_episodes) else None
    train = concatenate(train_normal, train_failure)
    val = concatenate(val_normal, val_failure)
    print(f"samples train_normal={len(train_normal['slots'])} train_failure={len(train_failure['slots']) if train_failure else 0} val_normal={len(val_normal['slots'])} val_failure={len(val_failure['slots']) if val_failure else 0}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TargetStateEstimator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for step in range(1, args.steps + 1):
        index = torch.randint(len(train["slots"]), (args.batch_size,))
        out = model(train["slots"][index].to(device), train["joints"][index].to(device), train["language"][index].to(device))
        loss, terms = losses(out, train, index)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1 or step % 50 == 0:
            print(f"step={step} loss={loss.item():.4f} target_cm={terms[0].item():.4f} phase={terms[2].item():.4f}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, val, device)
            metrics["normal"] = evaluate(model, val_normal, device)
            if val_failure is not None:
                metrics["failure"] = evaluate(model, val_failure, device)
            print(f"val step={step} {json.dumps(metrics)}", flush=True)
            if metrics["target_relative_mae_cm"] < best:
                best = metrics["target_relative_mae_cm"]
                save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}, str(output_dir / "target-state-best.safetensors"))
                (output_dir / "best-metrics.json").write_text(json.dumps(metrics, indent=2))
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}, str(output_dir / "target-state-final.safetensors"))


if __name__ == "__main__":
    main()
