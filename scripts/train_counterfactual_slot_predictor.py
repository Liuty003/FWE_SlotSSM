#!/usr/bin/env python
"""Train an action-sensitive predictor on matched counterfactual Slot branches."""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-tasks", default="0,1,2,3,4,5,6,7")
    p.add_argument("--val-task", type=int, default=8)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-forks", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--ranking-margin", type=float, default=0.002)
    p.add_argument("--ranking-weight", type=float, default=0.2)
    p.add_argument("--branch-delta-weight", type=float, default=0.5)
    p.add_argument("--identity-weight", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260721)
    return p.parse_args()


class CounterfactualSlotPredictor(torch.nn.Module):
    def __init__(self, action_mean, action_std, joint_mean, joint_std, steps=4, dim=512):
        super().__init__()
        self.steps = steps
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)
        self.register_buffer("joint_mean", joint_mean)
        self.register_buffer("joint_std", joint_std)
        self.action_encoder = torch.nn.Sequential(
            torch.nn.Linear(7, 256), torch.nn.GELU(), torch.nn.Linear(256, dim)
        )
        self.joint_encoder = torch.nn.Sequential(
            torch.nn.Linear(7, 256), torch.nn.GELU(), torch.nn.Linear(256, dim)
        )
        self.step_embedding = torch.nn.Embedding(steps, dim)
        self.input_norm = torch.nn.LayerNorm(dim)
        self.self_attention = torch.nn.MultiheadAttention(dim, 8, batch_first=True)
        self.attention_norm = torch.nn.LayerNorm(dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 2), torch.nn.GELU(), torch.nn.Linear(dim * 2, dim)
        )
        self.ffn_norm = torch.nn.LayerNorm(dim)
        self.delta_head = torch.nn.Sequential(
            torch.nn.LayerNorm(dim), torch.nn.Linear(dim, dim),
            torch.nn.GELU(), torch.nn.Linear(dim, dim),
        )
        torch.nn.init.zeros_(self.delta_head[-1].weight)
        torch.nn.init.zeros_(self.delta_head[-1].bias)

    def forward(self, before_slots, actions, joint_positions):
        # before [B,K,D], action [B,A,7], joint [B,7]
        b, branches = actions.shape[:2]
        current = before_slots[:, None].expand(-1, branches, -1, -1).reshape(
            b * branches, before_slots.shape[1], before_slots.shape[2]
        )
        action = (actions - self.action_mean) / self.action_std
        joint = (joint_positions - self.joint_mean) / self.joint_std
        condition = self.action_encoder(action).reshape(b * branches, 1, -1)
        condition = condition + self.joint_encoder(joint)[:, None, None, :].expand(
            -1, branches, -1, -1
        ).reshape(b * branches, 1, -1)
        outputs = []
        for step in range(self.steps):
            value = self.input_norm(
                current + condition + self.step_embedding.weight[step].view(1, 1, -1)
            )
            attended, _ = self.self_attention(value, value, value, need_weights=False)
            value = self.attention_norm(value + attended)
            value = self.ffn_norm(value + self.ffn(value))
            current = current + self.delta_head(value)
            outputs.append(current.reshape(b, branches, current.shape[1], current.shape[2]))
        return torch.stack(outputs, dim=2)


def load_tasks(data_dir, task_ids):
    parts = [load_file(str(data_dir / f"libero_spatial_t{task}_e0_slots.safetensors")) for task in task_ids]
    keys = ("before_slots", "after_slots", "after_valid", "executed_actions", "joint_positions")
    return {key: torch.cat([part[key] for part in parts], dim=0) for key in keys}


def normalization(data):
    action = data["executed_actions"].reshape(-1, 7).float()
    joint = data["joint_positions"].float()
    return action.mean(0), action.std(0).clamp_min(1e-3), joint.mean(0), joint.std(0).clamp_min(1e-3)


def identity_loss(prediction, target, valid):
    query = F.normalize(prediction.float(), dim=-1)
    key = F.normalize(target.float(), dim=-1)
    logits = torch.einsum("batkd,batjd->batkj", query, key)
    labels = torch.arange(query.shape[-2], device=query.device).view(1, 1, 1, -1)
    labels = labels.expand(query.shape[:-1]).reshape(-1)
    per = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels, reduction="none")
    slot_valid = valid.unsqueeze(-1).expand(*valid.shape, query.shape[-2]).reshape(-1)
    return (per * slot_valid).sum() / slot_valid.sum().clamp_min(1)


def losses(prediction, target, valid, args):
    valid_f = valid[..., None, None].float()
    element = F.smooth_l1_loss(prediction.float(), target.float(), reduction="none")
    correct = (element * valid_f).sum() / (valid_f.sum() * target.shape[-2] * target.shape[-1]).clamp_min(1)
    # The prediction for a different action from the same state must not fit
    # this branch target as well as its correctly paired prediction.
    wrong = prediction.roll(shifts=1, dims=1)
    correct_per = (prediction.float() - target.float()).abs().mean(dim=(-1, -2, -3))
    wrong_per = (wrong.float() - target.float()).abs().mean(dim=(-1, -2, -3))
    branch_valid = valid.any(dim=-1).float()
    ranking = (F.relu(args.ranking_margin + correct_per - wrong_per) * branch_valid).sum()
    ranking = ranking / branch_valid.sum().clamp_min(1)
    pred_delta = prediction[:, 1:] - prediction[:, :1]
    target_delta = target[:, 1:] - target[:, :1]
    delta_valid = (valid[:, 1:] & valid[:, :1]).float()[..., None, None]
    delta = (F.smooth_l1_loss(pred_delta.float(), target_delta.float(), reduction="none") * delta_valid).sum()
    delta = delta / (delta_valid.sum() * target.shape[-2] * target.shape[-1]).clamp_min(1)
    identity = identity_loss(prediction, target, valid.float())
    total = (
        correct + args.ranking_weight * ranking
        + args.branch_delta_weight * delta + args.identity_weight * identity
    )
    return total, {"correct": correct, "ranking": ranking, "delta": delta, "identity": identity}


@torch.inference_mode()
def evaluate(model, data, device, args):
    model.eval()
    before = data["before_slots"].to(device).float()
    actions = data["executed_actions"].to(device).float()
    joint = data["joint_positions"].to(device).float()
    target = data["after_slots"].to(device).float()
    valid = data["after_valid"].to(device)
    prediction = model(before, actions, joint)
    _, values = losses(prediction, target, valid, args)
    shuffled = model(before, actions.roll(1, dims=1), joint)
    mask = valid[..., None, None].float()
    shuffled_l1 = ((shuffled - target).abs() * mask).sum() / (
        mask.sum() * target.shape[-2] * target.shape[-1]
    ).clamp_min(1)
    correct_l1 = ((prediction - target).abs() * mask).sum() / (
        mask.sum() * target.shape[-2] * target.shape[-1]
    ).clamp_min(1)
    model.train()
    return {**{k: float(v) for k, v in values.items()}, "mae": float(correct_l1), "shuffled_mae": float(shuffled_l1)}


def save(model, optimizer, step, output_dir, args, best=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    name = "counterfactual-best.safetensors" if best else f"counterfactual-step-{step:06d}.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(output_dir / name))
    if not best:
        torch.save({"step": step, "optimizer": optimizer.state_dict()}, output_dir / f"optimizer-step-{step:06d}.pt")
        (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    train_tasks = [int(value) for value in args.train_tasks.split(",") if value]
    train = load_tasks(data_dir, train_tasks)
    val = load_tasks(data_dir, [args.val_task])
    stats = normalization(train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CounterfactualSlotPredictor(*stats).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(
        f"device={device} train_forks={len(train['before_slots'])} val_forks={len(val['before_slots'])} "
        f"trainable={sum(p.numel() for p in model.parameters())}", flush=True,
    )
    best = float("inf")
    running = 0.0
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(train["before_slots"]), (args.batch_forks,))
        before = train["before_slots"][indices].to(device).float()
        actions = train["executed_actions"][indices].to(device).float()
        joint = train["joint_positions"][indices].to(device).float()
        target = train["after_slots"][indices].to(device).float()
        valid = train["after_valid"][indices].to(device)
        prediction = model(before, actions, joint)
        loss, values = losses(prediction, target, valid, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        running += float(loss)
        if step == 1 or step % args.log_every == 0:
            divisor = 1 if step == 1 else args.log_every
            print(
                f"step={step}/{args.steps} loss={running/divisor:.6f} "
                f"correct={float(values['correct']):.6f} ranking={float(values['ranking']):.6f} "
                f"delta={float(values['delta']):.6f} finite={finite}", flush=True,
            )
            running = 0.0
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, val, device, args)
            print(f"validation step={step} {json.dumps(metrics)}", flush=True)
            if metrics["mae"] < best:
                best = metrics["mae"]
                save(model, optimizer, step, output_dir, args, best=True)
        if step % args.save_every == 0:
            save(model, optimizer, step, output_dir, args)
    save(model, optimizer, args.steps, output_dir, args)
    print(f"saved={output_dir} best_val_mae={best:.6f}")


if __name__ == "__main__":
    main()
