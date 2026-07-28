#!/usr/bin/env python
"""Train fixed-role Stage-B slots from precomputed OpenVLA vision patches.

This first Stage-B implementation intentionally uses truncated temporal
gradients: the previous frame's slots initialize the next frame but are
detached.  It makes the expensive 64x64 mask head trainable on a 24 GB GPU
while still supervising persistent role identity at every frame.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from prismatic.extern.hf.modeling_prismatic import MaskPredictionHead, SlotAttention


DEFAULT_CHECKPOINT = (
    "/data2/lty/naive-e-slotssm-runs/stage1-adapters/"
    "openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--"
    "full10tasks-h16-sync6-s20k-eb6-no-gpu3--image_aug--multiview1000--object/"
    "pretrained_slots-wo-filters_s16.safetensors"
)
ROLE_ORDER = ("target", "distractor", "goal", "robot")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/stage-b-task9-semantic-patches"),
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/stage-b-task9-semantic-smoke"),
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--mask-bce-weight", type=float, default=1.0)
    parser.add_argument("--mask-dice-weight", type=float, default=1.0)
    parser.add_argument(
        "--role-weights", default="3,2,1,1",
        help="Comma-separated weights for target,distractor,goal,robot",
    )
    parser.add_argument("--temporal-weight", type=float, default=0.1)
    parser.add_argument("--separation-weight", type=float, default=0.1)
    parser.add_argument(
        "--attention-weight", type=float, default=0.0,
        help="Direct role-mask supervision on the native 16x16 SlotAttention maps",
    )
    parser.add_argument("--separation-margin", type=float, default=0.2)
    parser.add_argument("--max-pos-weight", type=float, default=50.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-windows", type=int, default=2)
    return parser.parse_args()


def load_modules(checkpoint: str, device: torch.device):
    tokenizer = SlotAttention(n_slots=16, in_dim=2176, feature_dim=512)
    mask_head = MaskPredictionHead(slot_dim=512, hidden_dim=1024, mask_size=(64, 64))
    state = load_file(checkpoint)
    tokenizer_prefix = "object_centric_tokenizer."
    mask_prefix = "object_centric_mask_head."
    tokenizer_state = {
        key[len(tokenizer_prefix):]: value
        for key, value in state.items() if key.startswith(tokenizer_prefix)
    }
    mask_state = {
        key[len(mask_prefix):]: value
        for key, value in state.items() if key.startswith(mask_prefix)
    }
    tokenizer_result = tokenizer.load_state_dict(tokenizer_state, strict=True)
    mask_result = mask_head.load_state_dict(mask_state, strict=True)
    if tokenizer_result.missing_keys or tokenizer_result.unexpected_keys:
        raise RuntimeError(f"Tokenizer checkpoint mismatch: {tokenizer_result}")
    if mask_result.missing_keys or mask_result.unexpected_keys:
        raise RuntimeError(f"Mask checkpoint mismatch: {mask_result}")
    return tokenizer.to(device), mask_head.to(device)


def load_manifest(data_dir: Path):
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if tuple(manifest["role_order"]) != ROLE_ORDER:
        raise RuntimeError(f"Unexpected role order: {manifest['role_order']}")
    grouped = {"train": [], "validation": [], "test": []}
    for episode in manifest["episodes"]:
        grouped[episode["split"]].append(episode)
    if not all(grouped.values()):
        raise RuntimeError(f"Empty data split: {grouped}")
    return manifest, grouped


class EpisodeCache:
    def __init__(self):
        self.path = None
        self.tensors = None

    def get(self, path: str):
        if path != self.path:
            self.tensors = load_file(path)
            self.path = path
        return self.tensors


def sample_window(entries, cache, window, rng):
    eligible = [entry for entry in entries if entry["num_frames"] >= window]
    entry = rng.choice(eligible)
    start = rng.randint(0, entry["num_frames"] - window)
    tensors = cache.get(entry["output"])
    return (
        tensors["patches"][start : start + window],
        tensors["role_masks"][start : start + window],
        {"episode": entry["episode"], "start": start},
    )


def parse_role_weights(value: str, device: torch.device):
    weights = torch.tensor([float(item) for item in value.split(",")], device=device)
    if weights.shape != (len(ROLE_ORDER),) or not torch.all(weights > 0):
        raise ValueError(f"role-weights must contain {len(ROLE_ORDER)} positive values")
    return weights


def weighted_role_mean(values, role_weights):
    return (values * role_weights[None]).sum() / (values.shape[0] * role_weights.sum())


def dice_loss(logits, targets, role_weights):
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).flatten(2).sum(-1)
    denominator = probabilities.flatten(2).sum(-1) + targets.flatten(2).sum(-1)
    per_role = 1 - (2 * intersection + 1) / (denominator + 1)
    return weighted_role_mean(per_role, role_weights)


def mask_bce(logits, targets, max_pos_weight, role_weights):
    positives = targets.flatten(2).sum(-1)
    negatives = targets.shape[-1] * targets.shape[-2] - positives
    pos_weight = (negatives / positives.clamp_min(1)).clamp(1, max_pos_weight)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.where(targets.bool(), pos_weight[:, :, None, None], 1.0)
    per_role = (loss * weights).flatten(2).mean(-1)
    return weighted_role_mean(per_role, role_weights)


def fixed_role_iou(logits, targets):
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).flatten(2).sum(-1)
    union = probabilities.flatten(2).sum(-1) + targets.flatten(2).sum(-1) - intersection
    return intersection / union.clamp_min(1e-6)


def attention_alignment_loss(attention, targets, role_weights):
    # attention is the pre-reweighting slot competition map [B,S,heads,patches].
    attention = attention[:, : len(ROLE_ORDER)].mean(2).reshape(-1, len(ROLE_ORDER), 16, 16)
    target_patches = F.interpolate(targets, size=(16, 16), mode="area")
    with torch.autocast(device_type="cuda", enabled=False):
        probabilities = attention.float().clamp(1e-5, 1 - 1e-5)
        target_patches = target_patches.float()
        bce = F.binary_cross_entropy(probabilities, target_patches, reduction="none")
        bce = weighted_role_mean(bce.flatten(2).mean(-1), role_weights)
        intersection = (probabilities * target_patches).flatten(2).sum(-1)
        denominator = probabilities.flatten(2).sum(-1) + target_patches.flatten(2).sum(-1)
        dice = weighted_role_mean(1 - (2 * intersection + 1) / (denominator + 1), role_weights)
    return bce + dice


def forward_frame(tokenizer, mask_head, patch, target, past, args):
    slots, attention, _ = tokenizer(patch[None], past_slots=past)
    role_slots = slots[:, : len(ROLE_ORDER)]
    logits = mask_head(role_slots)
    target = target[None].float()
    role_weights = parse_role_weights(args.role_weights, logits.device)
    bce = mask_bce(logits, target, args.max_pos_weight, role_weights)
    dice = dice_loss(logits, target, role_weights)
    attention_loss = attention_alignment_loss(attention, target, role_weights)
    separation = F.relu(
        F.cosine_similarity(role_slots[:, 0], role_slots[:, 1], dim=-1)
        - args.separation_margin
    ).mean()
    if past is None:
        temporal = slots.new_zeros(())
    else:
        previous_roles = past[:, 0, : len(ROLE_ORDER)]
        temporal = (
            1 - F.cosine_similarity(role_slots, previous_roles, dim=-1)
        ).mean()
    total = (
        args.mask_bce_weight * bce
        + args.mask_dice_weight * dice
        + args.temporal_weight * temporal
        + args.separation_weight * separation
        + args.attention_weight * attention_loss
    )
    metrics = {
        "loss_bce": float(bce.detach()),
        "loss_dice": float(dice.detach()),
        "loss_temporal": float(temporal.detach()),
        "loss_separation": float(separation.detach()),
        "loss_attention": float(attention_loss.detach()),
        "role_soft_iou": fixed_role_iou(logits.detach(), target).mean(0).cpu().tolist(),
    }
    return total, slots, metrics


def train_step(tokenizer, mask_head, patches, masks, optimizer, args):
    tokenizer.train()
    mask_head.train()
    optimizer.zero_grad(set_to_none=True)
    past = None
    frame_metrics = []
    for patch, target in zip(patches, masks):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, slots, metrics = forward_frame(
                tokenizer, mask_head, patch, target, past, args
            )
            normalized = loss / len(patches)
        normalized.backward()
        frame_metrics.append(metrics)
        past = slots.detach()[:, None]
    parameters = list(tokenizer.parameters()) + list(mask_head.parameters())
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in parameters
    )
    if not finite_gradients:
        raise RuntimeError("Non-finite gradient in Stage-B semantic training")
    optimizer.step()
    return summarize_metrics(frame_metrics) | {"grad_norm": float(grad_norm)}


def summarize_metrics(frames):
    result = {}
    for key in (
        "loss_bce", "loss_dice", "loss_temporal", "loss_separation", "loss_attention"
    ):
        result[key] = sum(frame[key] for frame in frames) / len(frames)
    result["role_soft_iou"] = [
        sum(frame["role_soft_iou"][role] for frame in frames) / len(frames)
        for role in range(len(ROLE_ORDER))
    ]
    result["loss"] = (
        result["loss_bce"] + result["loss_dice"]
        + result["loss_temporal"] + result["loss_separation"] + result["loss_attention"]
    )
    return result


@torch.inference_mode()
def evaluate(tokenizer, mask_head, entries, cache, args, device, rng):
    tokenizer.eval()
    mask_head.eval()
    windows = []
    for _ in range(args.eval_windows):
        patches, masks, source = sample_window(entries, cache, args.window, rng)
        patches = patches.to(device)
        masks = masks.to(device)
        past, frame_metrics = None, []
        for patch, target in zip(patches, masks):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, slots, metrics = forward_frame(
                    tokenizer, mask_head, patch, target, past, args
                )
            past = slots[:, None]
            frame_metrics.append(metrics)
        windows.append(summarize_metrics(frame_metrics) | source)
    aggregate = summarize_metrics(windows)
    aggregate["windows"] = windows
    return aggregate


def save_checkpoint(tokenizer, mask_head, output: Path, metadata: dict):
    state = {}
    state.update({
        f"object_centric_tokenizer.{key}": value.detach().cpu()
        for key, value in tokenizer.state_dict().items()
    })
    state.update({
        f"object_centric_mask_head.{key}": value.detach().cpu()
        for key, value in mask_head.state_dict().items()
    })
    save_file(state, output, metadata={key: str(value) for key, value in metadata.items()})


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-B semantic training requires CUDA")
    if args.steps < 1 or args.window < 2:
        raise ValueError("steps must be >=1 and window must be >=2")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda:0")
    _, splits = load_manifest(args.data_dir)
    tokenizer, mask_head = load_modules(args.checkpoint, device)
    optimizer = torch.optim.AdamW(
        list(tokenizer.parameters()) + list(mask_head.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    cache = EpisodeCache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_seed = args.seed + 10000
    initial_validation = evaluate(
        tokenizer, mask_head, splits["validation"], cache, args, device,
        random.Random(validation_seed),
    )
    print(json.dumps({"initial_validation": initial_validation}), flush=True)
    history = []
    for step in range(1, args.steps + 1):
        patches, masks, source = sample_window(splits["train"], cache, args.window, rng)
        metrics = train_step(
            tokenizer, mask_head, patches.to(device), masks.to(device), optimizer, args
        )
        record = {"step": step, **source, **metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
    validation = evaluate(
        tokenizer, mask_head, splits["validation"], cache, args, device,
        random.Random(validation_seed),
    )
    checkpoint = args.output_dir / f"semantic-slots-step-{args.steps:06d}.safetensors"
    save_checkpoint(tokenizer, mask_head, checkpoint, {
        "steps": args.steps, "role_order": ",".join(ROLE_ORDER), "window": args.window,
    })
    report = {
        "schema_version": 1,
        "data_dir": str(args.data_dir),
        "initial_checkpoint": args.checkpoint,
        "output_checkpoint": str(checkpoint),
        "role_order": list(ROLE_ORDER),
        "truncated_temporal_gradient": True,
        "initial_validation": initial_validation,
        "history": history,
        "validation": validation,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"checkpoint": str(checkpoint), "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
