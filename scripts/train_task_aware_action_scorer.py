#!/usr/bin/env python
"""Train a language-conditioned scorer on dense task-progress labels."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from scripts.train_failure_physical_scorer import binary_auc, normalization, targets


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--progress-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-tasks", default="0,1,2,3,4,5,6,7")
    p.add_argument("--train-episodes", default="0,1,2,3")
    p.add_argument("--val-task", type=int, default=8)
    p.add_argument("--val-episodes", default="0,1,2,3,4")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-forks", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ranking-temperature-m", type=float, default=0.005)
    p.add_argument("--physical-weight", type=float, default=0.5)
    p.add_argument("--classification-weight", type=float, default=0.25)
    p.add_argument("--progress-weight", type=float, default=0.5)
    p.add_argument("--ranking-weight", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260721)
    return p.parse_args()


def ids(text):
    return [int(value) for value in text.split(",") if value]


def language_tokens(text, length=32, buckets=4096):
    words = re.findall(r"[a-z0-9]+", text.lower())[:length]
    token_ids = []
    for word in words:
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
        token_ids.append(1 + int.from_bytes(digest, "little") % (buckets - 1))
    token_ids += [0] * (length - len(token_ids))
    return torch.tensor(token_ids, dtype=torch.long)


def load_split(data_dir, progress_dir, tasks, episodes):
    manifest = json.loads((progress_dir / "manifest.json").read_text(encoding="utf-8"))
    languages = {(item["task_id"], item["episode_id"]): item["task_language"] for item in manifest["shards"]}
    parts = []
    for task in tasks:
        for episode in episodes:
            physical = load_file(str(data_dir / f"libero_spatial_t{task}_e{episode}_slots.safetensors"))
            progress = load_file(str(progress_dir / f"libero_spatial_t{task}_e{episode}_progress.safetensors"))
            if not torch.equal(physical["fork_ids"], progress["fork_ids"]):
                raise RuntimeError(f"fork mismatch task={task} episode={episode}")
            count = physical["before_slots"].shape[0]
            parts.append({
                "before_slots": physical["before_slots"], "after_valid": physical["after_valid"],
                "executed_actions": physical["executed_actions"], "joint_positions": physical["joint_positions"],
                "eef_delta_m": physical["eef_delta_m"], "max_object_delta_m": physical["max_object_delta_m"],
                "max_object_displacement_m": physical["max_object_displacement_m"],
                "task_progress_m": progress["task_progress_m"],
                "lifted_phase": progress["lifted_phase"],
                "language_tokens": language_tokens(languages[(task, episode)])[None].expand(count, -1),
            })
    return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}


class TaskAwareActionScorer(torch.nn.Module):
    def __init__(self, action_mean, action_std, joint_mean, joint_std, horizon=4, dim=512):
        super().__init__()
        self.horizon = horizon
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)
        self.register_buffer("joint_mean", joint_mean)
        self.register_buffer("joint_std", joint_std)
        self.slot_norm = torch.nn.LayerNorm(dim)
        self.action_encoder = torch.nn.Sequential(torch.nn.Linear(7, 256), torch.nn.GELU(), torch.nn.Linear(256, dim))
        self.joint_encoder = torch.nn.Sequential(torch.nn.Linear(7, 256), torch.nn.GELU(), torch.nn.Linear(256, dim))
        self.language_embedding = torch.nn.Embedding(4096, 128, padding_idx=0)
        self.language_projection = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.GELU(), torch.nn.Linear(256, dim))
        self.step_embedding = torch.nn.Embedding(horizon, dim)
        self.film = torch.nn.Linear(dim, dim * 2)
        self.attention = torch.nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm = torch.nn.LayerNorm(dim)
        # physical(7), interaction, no-effect failure, rank score, progress(m)
        self.head = torch.nn.Sequential(torch.nn.Linear(dim, 512), torch.nn.GELU(), torch.nn.Linear(512, 11))
        torch.nn.init.zeros_(self.head[-1].weight)
        torch.nn.init.zeros_(self.head[-1].bias)

    def forward(self, slots, actions, joints, language):
        batch, branches = actions.shape[:2]
        slot_values = self.slot_norm(slots)[:, None].expand(-1, branches, -1, -1)
        slot_values = slot_values.reshape(batch * branches, slots.shape[1], slots.shape[2])
        action = self.action_encoder((actions - self.action_mean) / self.action_std)
        joint = self.joint_encoder((joints - self.joint_mean) / self.joint_std)
        language_mask = language.ne(0)[..., None]
        language_value = (self.language_embedding(language) * language_mask).sum(1)
        language_value = language_value / language_mask.sum(1).clamp_min(1)
        language_value = self.language_projection(language_value)
        condition = action + joint[:, None] + language_value[:, None]
        outputs = []
        for step in range(self.horizon):
            step_condition = condition + self.step_embedding.weight[step]
            scale, shift = self.film(step_condition).chunk(2, -1)
            value = slot_values * (1 + 0.1 * scale.reshape(-1, 1, 512).tanh())
            value = value + shift.reshape(-1, 1, 512)
            attended, _ = self.attention(value, value, value, need_weights=False)
            pooled = self.norm(value + attended).mean(1)
            outputs.append(self.head(pooled).reshape(batch, branches, 11))
        return torch.stack(outputs, dim=2)


def loss_terms(output, physical, interaction, failure, progress, valid, args):
    mask = valid[..., None].float()
    physical_loss = (F.smooth_l1_loss(output[..., :7] * 100, physical * 100, reduction="none") * mask).sum()
    physical_loss = physical_loss / (mask.sum() * 7).clamp_min(1)
    interaction_loss = (F.binary_cross_entropy_with_logits(output[..., 7], interaction, reduction="none") * valid).sum() / valid.sum().clamp_min(1)
    failure_loss = (F.binary_cross_entropy_with_logits(output[..., 8], failure, reduction="none") * valid).sum() / valid.sum().clamp_min(1)
    progress_loss = (F.smooth_l1_loss(output[..., 10] * 1000, progress * 1000, reduction="none") * valid).sum() / valid.sum().clamp_min(1)
    ranking_logits = output[..., 9].masked_fill(~valid, -1e4)
    target_logits = (progress / args.ranking_temperature_m).masked_fill(~valid, -1e4)
    target_distribution = target_logits.softmax(dim=1)
    ranking = -(target_distribution * ranking_logits.log_softmax(dim=1)).sum(1)
    informative = (progress.max(1).values - progress.min(1).values > 0.001) & valid.any(1)
    ranking_loss = (ranking * informative).sum() / informative.sum().clamp_min(1)
    total = (
        args.physical_weight * physical_loss
        + args.classification_weight * (interaction_loss + failure_loss)
        + args.progress_weight * progress_loss
        + args.ranking_weight * ranking_loss
    )
    return total, {"physical_cm": physical_loss, "interaction": interaction_loss, "failure": failure_loss, "progress_mm": progress_loss, "ranking": ranking_loss}


@torch.inference_mode()
def evaluate(model, data, device):
    model.eval()
    count = len(data["before_slots"])
    slots = data["before_slots"].to(device).float(); actions = data["executed_actions"].to(device).float()
    joints = data["joint_positions"].to(device).float(); language = data["language_tokens"].to(device)
    valid = data["after_valid"].to(device).bool(); progress = data["task_progress_m"].to(device).float()
    physical, interaction, failure = targets(data, torch.arange(count), device)
    output = model(slots, actions, joints, language)
    shuffled = model(slots, actions.roll(1, 1), joints, language)
    mask = valid[..., None]
    physical_mae = ((output[..., :7] - physical).abs() * mask).sum() / (mask.sum() * 7)
    shuffled_mae = ((shuffled[..., :7] - physical).abs() * mask).sum() / (mask.sum() * 7)
    final_score = output[:, :, -1, 9]; final_progress_prediction = output[:, :, -1, 10]
    true = progress[:, :, -1]; best = final_score.argmax(1); top3_indices = final_score.topk(3, 1).indices
    selected = true.gather(1, best[:, None]).squeeze(1); top3 = true.gather(1, top3_indices).max(1).values
    oracle = true.max(1).values; nominal = true[:, 0]; random = true.mean(1)
    combinations = torch.combinations(torch.arange(10, device=device), r=3)
    random_top3 = true[:, combinations].max(-1).values.mean(1)
    interaction_score = output[..., 7].sigmoid()[valid]; failure_score = output[..., 8].sigmoid()[valid]
    result = {
        "physical_mae_cm": float(physical_mae * 100), "shuffled_physical_mae_cm": float(shuffled_mae * 100),
        "progress_mae_mm": float((final_progress_prediction - true).abs().mean() * 1000),
        "interaction_auc": binary_auc(interaction_score, interaction[valid]),
        "no_effect_failure_auc": binary_auc(failure_score, failure[valid]),
        "best_branch_accuracy_percent": float(100 * best.eq(true.argmax(1)).float().mean()),
        "selected_progress_mm": float(selected.mean() * 1000), "nominal_progress_mm": float(nominal.mean() * 1000),
        "random_action_progress_mm": float(random.mean() * 1000), "oracle_progress_mm": float(oracle.mean() * 1000),
        "top3_regret_mm": float((oracle - top3).mean() * 1000),
        "random_top3_regret_mm": float((oracle - random_top3).mean() * 1000),
        "selected_positive_progress_percent": float(100 * selected.gt(0).float().mean()),
    }
    model.train(); return result


def save(model, directory, step, label=None):
    directory.mkdir(parents=True, exist_ok=True)
    name = f"task-scorer-best-{label}.safetensors" if label else f"task-scorer-step-{step:06d}.safetensors"
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(directory / name))


def main():
    args = arguments(); torch.manual_seed(args.seed)
    data_dir, progress_dir, output = Path(args.data_dir), Path(args.progress_dir), Path(args.output_dir)
    train = load_split(data_dir, progress_dir, ids(args.train_tasks), ids(args.train_episodes))
    val = load_split(data_dir, progress_dir, [args.val_task], ids(args.val_episodes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaskAwareActionScorer(*normalization(train)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_regret, best_physical = float("inf"), float("inf")
    print(f"device={device} train_forks={len(train['before_slots'])} val_forks={len(val['before_slots'])} params={sum(p.numel() for p in model.parameters())}", flush=True)
    for step in range(1, args.steps + 1):
        index = torch.randint(len(train["before_slots"]), (args.batch_forks,))
        slots=train["before_slots"][index].to(device).float(); actions=train["executed_actions"][index].to(device).float()
        joints=train["joint_positions"][index].to(device).float(); language=train["language_tokens"][index].to(device)
        valid=train["after_valid"][index].to(device).bool(); progress=train["task_progress_m"][index].to(device).float()
        physical, interaction, failure = targets(train, index, device)
        result=model(slots, actions, joints, language)
        loss, terms=loss_terms(result, physical, interaction, failure, progress, valid, args)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % args.log_every == 0:
            print(f"step={step}/{args.steps} loss={float(loss):.6f} " + " ".join(f"{k}={float(v):.6f}" for k,v in terms.items()) + f" finite={finite}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics=evaluate(model,val,device); print(f"validation step={step} {json.dumps(metrics)}",flush=True)
            if metrics["top3_regret_mm"] < best_regret:
                best_regret=metrics["top3_regret_mm"]; save(model,output,step,"ranking")
            if metrics["physical_mae_cm"] < best_physical:
                best_physical=metrics["physical_mae_cm"]; save(model,output,step,"physical")
        if step % args.save_every == 0: save(model,output,step)
    output.mkdir(parents=True,exist_ok=True); (output/"config.json").write_text(json.dumps(vars(args),indent=2),encoding="utf-8")
    print(f"saved={output} best_top3_regret_mm={best_regret:.6f} best_physical_mae_cm={best_physical:.6f}")


if __name__ == "__main__": main()
