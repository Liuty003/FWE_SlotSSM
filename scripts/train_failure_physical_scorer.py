#!/usr/bin/env python
"""Train an action-conditioned physical-effect and no-effect failure scorer."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-tasks", default="0,1,2,3,4,5,6,7")
    p.add_argument("--train-episodes", default="0,1,2,3")
    p.add_argument("--val-task", type=int, default=8)
    p.add_argument("--val-episodes", default="0,1,2,3,4")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-forks", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--physical-weight", type=float, default=1.0)
    p.add_argument("--interaction-bce-weight", type=float, default=0.5)
    p.add_argument("--failure-bce-weight", type=float, default=0.5)
    p.add_argument("--ranking-weight", type=float, default=0.5)
    p.add_argument("--ranking-temperature-m", type=float, default=0.002)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260721)
    return p.parse_args()


def parse_ids(text):
    return [int(value) for value in text.split(",") if value]


def load_split(root, tasks, episodes):
    parts = [
        load_file(str(root / f"libero_spatial_t{task}_e{episode}_slots.safetensors"))
        for task in tasks for episode in episodes
    ]
    keys = (
        "before_slots", "after_valid", "executed_actions", "joint_positions",
        "eef_delta_m", "max_object_delta_m", "max_object_displacement_m",
    )
    return {key: torch.cat([part[key] for part in parts]) for key in keys}


def normalization(data):
    actions = data["executed_actions"].float().reshape(-1, 7)
    joints = data["joint_positions"].float()
    return (
        actions.mean(0), actions.std(0).clamp_min(1e-3),
        joints.mean(0), joints.std(0).clamp_min(1e-3),
    )


class FailurePhysicalScorer(torch.nn.Module):
    def __init__(self, action_mean, action_std, joint_mean, joint_std, horizon=4, dim=512):
        super().__init__()
        self.horizon = horizon
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)
        self.register_buffer("joint_mean", joint_mean)
        self.register_buffer("joint_std", joint_std)
        self.slot_norm = torch.nn.LayerNorm(dim)
        self.action_encoder = torch.nn.Sequential(
            torch.nn.Linear(7, 256), torch.nn.GELU(), torch.nn.Linear(256, dim),
        )
        self.joint_encoder = torch.nn.Sequential(
            torch.nn.Linear(7, 256), torch.nn.GELU(), torch.nn.Linear(256, dim),
        )
        self.step_embedding = torch.nn.Embedding(horizon, dim)
        self.film = torch.nn.Linear(dim, dim * 2)
        self.attention = torch.nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm = torch.nn.LayerNorm(dim)
        # 7 physical values in meters + interaction/failure logits + ranking score.
        self.head = torch.nn.Sequential(
            torch.nn.Linear(dim, 512), torch.nn.GELU(), torch.nn.Linear(512, 10),
        )
        torch.nn.init.zeros_(self.head[-1].weight)
        torch.nn.init.zeros_(self.head[-1].bias)

    def forward(self, slots, actions, joints):
        batch, branches = actions.shape[:2]
        slot_values = self.slot_norm(slots)[:, None].expand(-1, branches, -1, -1)
        slot_values = slot_values.reshape(batch * branches, slots.shape[1], slots.shape[2])
        action = self.action_encoder((actions - self.action_mean) / self.action_std)
        joint = self.joint_encoder((joints - self.joint_mean) / self.joint_std)
        condition = action + joint[:, None]
        outputs = []
        for step in range(self.horizon):
            step_condition = condition + self.step_embedding.weight[step]
            scale, shift = self.film(step_condition).chunk(2, -1)
            value = slot_values * (1.0 + 0.1 * scale.reshape(-1, 1, 512).tanh())
            value = value + shift.reshape(-1, 1, 512)
            attended, _ = self.attention(value, value, value, need_weights=False)
            pooled = self.norm(value + attended).mean(1)
            outputs.append(self.head(pooled).reshape(batch, branches, 10))
        return torch.stack(outputs, dim=2)


def targets(data, index, device):
    eef_vector = data["eef_delta_m"][index].to(device).float()
    object_vector = data["max_object_delta_m"][index].to(device).float()
    object_distance = data["max_object_displacement_m"][index].to(device).float()
    eef_distance = eef_vector.norm(dim=-1)
    physical = torch.cat((eef_vector, object_vector, object_distance[..., None]), dim=-1)
    interaction = object_distance.gt(0.001).float()
    no_effect_failure = (eef_distance.gt(0.005) & object_distance.lt(0.001)).float()
    return physical, interaction, no_effect_failure


def loss_terms(output, physical, interaction, failure, valid, args):
    mask = valid[..., None].float()
    physical_loss = (F.smooth_l1_loss(output[..., :7] * 100, physical * 100, reduction="none") * mask).sum()
    physical_loss = physical_loss / (mask.sum() * 7).clamp_min(1)
    interaction_loss = (F.binary_cross_entropy_with_logits(output[..., 7], interaction, reduction="none") * valid).sum()
    interaction_loss = interaction_loss / valid.sum().clamp_min(1)
    failure_loss = (F.binary_cross_entropy_with_logits(output[..., 8], failure, reduction="none") * valid).sum()
    failure_loss = failure_loss / valid.sum().clamp_min(1)
    # Listwise ranking within each matched fork and horizon. The target distribution
    # concentrates on branches that actually move an object farther.
    ranking_logits = output[..., 9].masked_fill(~valid, -1e4)
    target_logits = (physical[..., 6] / args.ranking_temperature_m).masked_fill(~valid, -1e4)
    target_distribution = target_logits.softmax(dim=1)
    ranking_loss = -(target_distribution * ranking_logits.log_softmax(dim=1)).sum(dim=1)
    valid_distance = physical[..., 6].masked_fill(~valid, 0.0)
    max_distance = valid_distance.max(dim=1).values
    min_distance = valid_distance.masked_fill(~valid, torch.inf).min(dim=1).values
    # Do not teach arbitrary order when every branch has effectively the same outcome.
    informative = (valid.any(dim=1) & (max_distance - min_distance > 0.001)).float()
    ranking_loss = (ranking_loss * informative).sum() / informative.sum().clamp_min(1)
    total = (
        args.physical_weight * physical_loss
        + args.interaction_bce_weight * interaction_loss
        + args.failure_bce_weight * failure_loss
        + args.ranking_weight * ranking_loss
    )
    return total, physical_loss, interaction_loss, failure_loss, ranking_loss


def binary_auc(scores, labels):
    scores, labels = scores.flatten(), labels.flatten().bool()
    positive, negative = labels.sum(), (~labels).sum()
    if positive == 0 or negative == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, device=scores.device).float()
    auc = (ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative)
    return float(auc)


@torch.inference_mode()
def evaluate(model, data, device):
    model.eval()
    count = len(data["before_slots"])
    slots = data["before_slots"].to(device).float()
    actions = data["executed_actions"].to(device).float()
    joints = data["joint_positions"].to(device).float()
    valid = data["after_valid"].to(device).bool()
    physical, interaction, failure = targets(data, torch.arange(count), device)
    output = model(slots, actions, joints)
    shuffled = model(slots, actions.roll(1, 1), joints)
    mask = valid[..., None]
    physical_mae = ((output[..., :7] - physical).abs() * mask).sum() / (mask.sum() * 7)
    shuffled_mae = ((shuffled[..., :7] - physical).abs() * mask).sum() / (mask.sum() * 7)
    zero_mae = (physical.abs() * mask).sum() / (mask.sum() * 7)
    interaction_score = output[..., 7].sigmoid()[valid]
    failure_score = output[..., 8].sigmoid()[valid]
    interaction_label, failure_label = interaction[valid], failure[valid]
    final_valid = valid[..., -1]
    predicted_distance = output[..., -1, 9].masked_fill(~final_valid, -torch.inf)
    true_distance = physical[..., -1, 6].masked_fill(~final_valid, -torch.inf)
    best_match = predicted_distance.argmax(1).eq(true_distance.argmax(1)).float().mean()
    top3 = true_distance.gather(1, predicted_distance.topk(3, dim=1).indices).max(1).values
    oracle = true_distance.max(1).values.clamp_min(1e-6)
    combinations = torch.combinations(torch.arange(true_distance.shape[1], device=device), r=3)
    random_top3 = true_distance[:, combinations].max(dim=-1).values.mean(dim=-1)
    metrics = {
        "physical_mae_cm": float(physical_mae * 100),
        "zero_physical_mae_cm": float(zero_mae * 100),
        "shuffled_physical_mae_cm": float(shuffled_mae * 100),
        "interaction_accuracy_percent": float(100 * interaction_score.gt(0.5).eq(interaction_label.bool()).float().mean()),
        "interaction_auc": binary_auc(interaction_score, interaction_label),
        "no_effect_failure_accuracy_percent": float(100 * failure_score.gt(0.5).eq(failure_label.bool()).float().mean()),
        "no_effect_failure_auc": binary_auc(failure_score, failure_label),
        "best_branch_accuracy_percent": float(100 * best_match),
        "best_branch_chance_percent": 10.0,
        "top3_object_effect_oracle_ratio_percent": float(100 * (top3 / oracle).mean()),
        "random_top3_oracle_ratio_percent": float(100 * (random_top3 / oracle).mean()),
    }
    model.train()
    return metrics


def save(model, output, step, best=None):
    output.mkdir(parents=True, exist_ok=True)
    name = f"scorer-best-{best}.safetensors" if best else f"scorer-step-{step:06d}.safetensors"
    save_file({k: value.detach().cpu().contiguous() for k, value in model.state_dict().items()}, str(output / name))


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    root, output_dir = Path(args.data_dir), Path(args.output_dir)
    train = load_split(root, parse_ids(args.train_tasks), parse_ids(args.train_episodes))
    val = load_split(root, [args.val_task], parse_ids(args.val_episodes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FailurePhysicalScorer(*normalization(train)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"device={device} train_forks={len(train['before_slots'])} val_forks={len(val['before_slots'])} params={sum(x.numel() for x in model.parameters())}", flush=True)
    best_physical, best_ranking = float("inf"), -float("inf")
    for step in range(1, args.steps + 1):
        index = torch.randint(len(train["before_slots"]), (args.batch_forks,))
        slots = train["before_slots"][index].to(device).float()
        actions = train["executed_actions"][index].to(device).float()
        joints = train["joint_positions"][index].to(device).float()
        valid = train["after_valid"][index].to(device).bool()
        physical, interaction, failure = targets(train, index, device)
        result = model(slots, actions, joints)
        loss, physical_loss, interaction_loss, failure_loss, ranking_loss = loss_terms(
            result, physical, interaction, failure, valid, args
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % args.log_every == 0:
            print(f"step={step}/{args.steps} loss={float(loss):.6f} physical_cm_loss={float(physical_loss):.6f} interaction_bce={float(interaction_loss):.6f} failure_bce={float(failure_loss):.6f} ranking={float(ranking_loss):.6f} finite={finite}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, val, device)
            print(f"validation step={step} {json.dumps(metrics)}", flush=True)
            if metrics["physical_mae_cm"] < best_physical:
                best_physical = metrics["physical_mae_cm"]
                save(model, output_dir, step, best="physical")
            if metrics["top3_object_effect_oracle_ratio_percent"] > best_ranking:
                best_ranking = metrics["top3_object_effect_oracle_ratio_percent"]
                save(model, output_dir, step, best="ranking")
        if step % args.save_every == 0:
            save(model, output_dir, step)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"saved={output_dir} best_physical_mae_cm={best_physical:.6f} best_top3_oracle_percent={best_ranking:.6f}")


if __name__ == "__main__":
    main()
