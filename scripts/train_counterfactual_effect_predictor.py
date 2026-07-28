#!/usr/bin/env python
"""Train differential action effects from matched counterfactual Slot outcomes."""

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
    p.add_argument("--batch-forks", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ranking-margin", type=float, default=0.002)
    p.add_argument("--ranking-weight", type=float, default=0.5)
    p.add_argument("--cosine-weight", type=float, default=0.1)
    p.add_argument("--physical-weight", type=float, default=0.25)
    p.add_argument("--interaction-weight", type=float, default=2.0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260721)
    return p.parse_args()


class CounterfactualEffectPredictor(torch.nn.Module):
    def __init__(self, delta_std, joint_mean, joint_std, steps=4, dim=512):
        super().__init__()
        self.steps = steps
        self.register_buffer("delta_std", delta_std)
        self.register_buffer("joint_mean", joint_mean)
        self.register_buffer("joint_std", joint_std)
        self.action_encoder = torch.nn.Sequential(
            torch.nn.Linear(7, 256, bias=False), torch.nn.GELU(),
            torch.nn.Linear(256, dim, bias=False),
        )
        self.joint_encoder = torch.nn.Sequential(
            torch.nn.Linear(7, 256), torch.nn.GELU(), torch.nn.Linear(256, dim),
        )
        self.slot_norm = torch.nn.LayerNorm(dim)
        self.step_embedding = torch.nn.Embedding(steps, dim)
        self.film = torch.nn.Linear(dim, dim * 2)
        self.self_attention = torch.nn.MultiheadAttention(dim, 8, batch_first=True)
        self.attention_norm = torch.nn.LayerNorm(dim)
        self.output = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 2), torch.nn.GELU(), torch.nn.Linear(dim * 2, dim),
        )
        torch.nn.init.zeros_(self.output[-1].weight)
        torch.nn.init.zeros_(self.output[-1].bias)
        self.physical_head = torch.nn.Sequential(
            torch.nn.Linear(dim, 256), torch.nn.GELU(), torch.nn.Linear(256, 7),
        )
        torch.nn.init.zeros_(self.physical_head[-1].weight)
        torch.nn.init.zeros_(self.physical_head[-1].bias)

    def forward(self, before_slots, nominal_actions, executed_actions, joint_positions):
        b, branches = executed_actions.shape[:2]
        normalized_delta = (executed_actions - nominal_actions[:, None]) / self.delta_std
        action = self.action_encoder(normalized_delta).reshape(b * branches, 1, -1)
        joint = (joint_positions - self.joint_mean) / self.joint_std
        joint = self.joint_encoder(joint)[:, None, None].expand(-1, branches, -1, -1)
        joint = joint.reshape(b * branches, 1, -1)
        slots = self.slot_norm(before_slots)[:, None].expand(-1, branches, -1, -1)
        slots = slots.reshape(b * branches, before_slots.shape[1], before_slots.shape[2])
        # Exact zero intervention must produce exact zero effect.
        gate = normalized_delta.norm(dim=-1).tanh().reshape(b * branches, 1, 1)
        outputs, physical_outputs = [], []
        for step in range(self.steps):
            condition = action + joint + self.step_embedding.weight[step].view(1, 1, -1)
            scale, shift = self.film(condition).chunk(2, dim=-1)
            value = slots * (1.0 + 0.1 * scale.tanh()) + shift
            attended, _ = self.self_attention(value, value, value, need_weights=False)
            value = self.attention_norm(value + attended)
            outputs.append((self.output(value) * gate).reshape(
                b, branches, before_slots.shape[1], before_slots.shape[2]
            ))
            physical_outputs.append(
                (self.physical_head(value.mean(1)) * gate.squeeze(1)).reshape(b, branches, 7)
            )
        return torch.stack(outputs, dim=2), torch.stack(physical_outputs, dim=2)


def load_tasks(root, tasks, episodes):
    parts = [
        load_file(str(root / f"libero_spatial_t{task}_e{episode}_slots.safetensors"))
        for task in tasks for episode in episodes
    ]
    keys = (
        "before_slots", "after_slots", "after_valid", "executed_actions", "joint_positions",
        "eef_delta_m", "max_object_delta_m", "max_object_displacement_m",
    )
    return {key: torch.cat([part[key] for part in parts]) for key in keys}


def stats(data):
    actions = data["executed_actions"].float()
    delta = (actions - actions[:, :1]).reshape(-1, 7)
    joint = data["joint_positions"].float()
    return delta.std(0).clamp_min(1e-3), joint.mean(0), joint.std(0).clamp_min(1e-3)


def target_effect(data, indices, device):
    target = data["after_slots"][indices].to(device).float()
    return target - target[:, :1], target


def physical_target(data, indices, device):
    eef = data["eef_delta_m"][indices].to(device).float()
    obj = data["max_object_delta_m"][indices].to(device).float()
    distance = data["max_object_displacement_m"][indices].to(device).float()[..., None]
    physical = torch.cat((eef, obj, distance), dim=-1)
    return physical - physical[:, :1]


def loss_terms(prediction, target, physical_prediction, physical, object_distance, valid, args):
    mask = valid[..., None, None].float()
    interaction = 1.0 + args.interaction_weight * (object_distance > 0.001).float()
    weighted_mask = mask * interaction[..., None, None]
    correct = (F.smooth_l1_loss(prediction, target, reduction="none") * weighted_mask).sum()
    correct = correct / (weighted_mask.sum() * target.shape[-2] * target.shape[-1]).clamp_min(1)
    # Express displacement in centimeters so the auxiliary loss has an interpretable scale.
    physical_mask = valid[..., None].float() * interaction[..., None]
    physical_loss = (
        F.smooth_l1_loss(physical_prediction * 100.0, physical * 100.0, reduction="none")
        * physical_mask
    ).sum() / (physical_mask.sum() * physical.shape[-1]).clamp_min(1)
    wrong = prediction.roll(1, dims=1)
    correct_per = (prediction - target).abs().mean(dim=(-1, -2, -3))
    wrong_per = (wrong - target).abs().mean(dim=(-1, -2, -3))
    valid_branch = valid.any(-1).float()
    ranking = (F.relu(args.ranking_margin + correct_per - wrong_per) * valid_branch).sum()
    ranking = ranking / valid_branch.sum().clamp_min(1)
    # Non-nominal effect direction; nominal target and prediction are exactly zero.
    pred_flat = prediction[:, 1:].flatten(3)
    target_flat = target[:, 1:].flatten(3)
    nonzero = target_flat.norm(dim=-1).gt(1e-6).float()
    cosine = ((1.0 - F.cosine_similarity(pred_flat, target_flat, dim=-1)) * nonzero).sum()
    cosine = cosine / nonzero.sum().clamp_min(1)
    return correct + args.ranking_weight * ranking + args.cosine_weight * cosine + args.physical_weight * physical_loss, {
        "effect": correct, "ranking": ranking, "cosine": cosine, "physical_cm": physical_loss,
    }


@torch.inference_mode()
def evaluate(model, data, device, args):
    model.eval()
    before = data["before_slots"].to(device).float()
    actions = data["executed_actions"].to(device).float()
    joint = data["joint_positions"].to(device).float()
    target, _ = target_effect(data, torch.arange(len(before)), device)
    physical = physical_target(data, torch.arange(len(before)), device)
    valid = data["after_valid"].to(device)
    prediction, physical_prediction = model(before, actions[:, 0], actions, joint)
    shuffled, _ = model(before, actions[:, 0], actions.roll(1, 1), joint)
    mask = valid[..., None, None].float()
    mae = ((prediction - target).abs() * mask).sum() / (mask.sum() * 16 * 512)
    shuffled_mae = ((shuffled - target).abs() * mask).sum() / (mask.sum() * 16 * 512)
    zero_mae = (target.abs() * mask).sum() / (mask.sum() * 16 * 512)
    final_pred, final_target = prediction[:, :, -1], target[:, :, -1]
    distances = (final_pred[:, :, None] - final_target[:, None]).abs().mean((-1, -2))
    labels = torch.arange(actions.shape[1], device=device).view(1, -1)
    retrieval = distances.argmin(-1).eq(labels).float().mean()
    centered_pred = final_pred - final_pred.mean(1, keepdim=True)
    centered_target = final_target - final_target.mean(1, keepdim=True)
    cosine = F.cosine_similarity(centered_pred.flatten(2), centered_target.flatten(2), dim=-1).mean()
    physical_mae_cm = ((physical_prediction - physical).abs() * valid[..., None]).sum()
    physical_mae_cm = 100.0 * physical_mae_cm / (valid.sum() * physical.shape[-1]).clamp_min(1)
    model.train()
    return {
        "effect_mae": float(mae), "shuffled_effect_mae": float(shuffled_mae),
        "zero_effect_mae": float(zero_mae), "branch_retrieval": float(retrieval),
        "branch_delta_cosine": float(cosine), "physical_mae_cm": float(physical_mae_cm),
    }


def save(model, step, output, best=False):
    output.mkdir(parents=True, exist_ok=True)
    name = "effect-best.safetensors" if best else f"effect-step-{step:06d}.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(output / name))


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    root, output = Path(args.data_dir), Path(args.output_dir)
    train_episodes = [int(x) for x in args.train_episodes.split(",") if x]
    val_episodes = [int(x) for x in args.val_episodes.split(",") if x]
    train = load_tasks(root, [int(x) for x in args.train_tasks.split(",") if x], train_episodes)
    val = load_tasks(root, [args.val_task], val_episodes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CounterfactualEffectPredictor(*stats(train)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"device={device} train={len(train['before_slots'])} val={len(val['before_slots'])} params={sum(p.numel() for p in model.parameters())}", flush=True)
    best = float("inf")
    running = 0.0
    for step in range(1, args.steps + 1):
        index = torch.randint(len(train["before_slots"]), (args.batch_forks,))
        before = train["before_slots"][index].to(device).float()
        actions = train["executed_actions"][index].to(device).float()
        joint = train["joint_positions"][index].to(device).float()
        target, _ = target_effect(train, index, device)
        physical = physical_target(train, index, device)
        object_distance = train["max_object_displacement_m"][index].to(device).float()
        valid = train["after_valid"][index].to(device)
        prediction, physical_prediction = model(before, actions[:, 0], actions, joint)
        loss, values = loss_terms(
            prediction, target, physical_prediction, physical, object_distance, valid, args
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        running += float(loss)
        if step == 1 or step % args.log_every == 0:
            divisor = 1 if step == 1 else args.log_every
            print(f"step={step}/{args.steps} loss={running/divisor:.6f} effect={float(values['effect']):.6f} ranking={float(values['ranking']):.6f} cosine={float(values['cosine']):.6f} physical_cm={float(values['physical_cm']):.6f} finite={finite}", flush=True)
            running = 0.0
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, val, device, args)
            print(f"validation step={step} {json.dumps(metrics)}", flush=True)
            score = metrics["effect_mae"] + args.physical_weight * metrics["physical_mae_cm"]
            if score < best:
                best = score; save(model, step, output, best=True)
        if step % args.save_every == 0:
            save(model, step, output)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"saved={output} best={best:.6f}")


if __name__ == "__main__":
    main()
