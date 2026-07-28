#!/usr/bin/env python
"""Train two language-independent bowl instance proposals on frozen patches."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(
        "/data2/lty/naive-e-slotssm-runs/stage-b-spatial-counterfactual-patches"))
    parser.add_argument("--split-manifest", type=Path, default=Path(
        "/data2/lty/naive-e-slotssm-runs/stage-b-spatial-relation-data/split_manifest.json"))
    parser.add_argument("--protocol", default="compositional_reference_seen_fold_b")
    parser.add_argument("--output-dir", type=Path, default=Path(
        "/data2/lty/naive-e-slotssm-runs/stage-b-bowl-proposals-fold-b"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--overlap-weight", type=float, default=0.5)
    parser.add_argument("--horizontal-flip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--patch-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


class BowlProposalSegmenter(nn.Module):
    def __init__(self, hidden_dim, heads):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.patch_norm = nn.LayerNorm(2176)
        self.patch_projection = nn.Linear(2176, hidden_dim)
        self.queries = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.refinement = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, patches):
        features = self.patch_projection(self.patch_norm(patches.float()))
        queries = self.queries[None].expand(len(patches), -1, -1)
        slots, _ = self.attention(queries, features, features, need_weights=False)
        slots = F.normalize(self.query_norm(slots), dim=-1)
        spatial = F.normalize(self.spatial_norm(features), dim=-1)
        logits = self.logit_scale.exp().clamp(max=100) * torch.einsum(
            "bqd,bpd->bqp", slots, spatial
        )
        coarse = F.interpolate(
            logits.reshape(-1, 1, 16, 16), size=(64, 64),
            mode="bilinear", align_corners=False,
        )
        return (coarse + self.refinement(coarse)).reshape(len(patches), 2, 64, 64)


def load_samples(data_dir):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    samples = {}
    for task in manifest["tasks"]:
        tensors = load_file(task["output"])
        for index, state in enumerate(tensors["state_ids"].tolist()):
            samples[(int(task["task_id"]), int(state))] = (
                tensors["patches"][index],
                torch.stack((tensors["role_masks"][index, 0],
                             tensors["counterfactual_target_masks"][index])),
            )
    return samples


def load_splits(path, protocol):
    data = json.loads(path.read_text())["protocols"][protocol]
    return {
        name: [(int(row["task_id"]), int(row["state_id"])) for row in rows]
        for name, rows in data.items()
    }


def batch(keys, samples, device):
    return (
        torch.stack([samples[key][0] for key in keys]).to(device),
        torch.stack([samples[key][1] for key in keys]).to(device).float(),
    )


def augment(patches, targets, horizontal_flip, patch_dropout):
    if horizontal_flip:
        selected = torch.rand(len(patches), device=patches.device) < 0.5
        if selected.any():
            grid = patches[selected].reshape(-1, 16, 16, patches.shape[-1])
            patches[selected] = grid.flip(2).reshape(-1, 256, patches.shape[-1])
            targets[selected] = targets[selected].flip(-1)
    if patch_dropout > 0:
        keep = torch.rand(patches.shape[:2], device=patches.device) >= patch_dropout
        patches = patches * keep[:, :, None]
    return patches, targets


def pair_cost(logits, targets):
    probabilities = logits.sigmoid()
    pair_logits = logits[:, :, None].expand(-1, -1, 2, -1, -1)
    pair_targets = targets[:, None].expand(-1, 2, -1, -1, -1)
    bce = F.binary_cross_entropy_with_logits(
        pair_logits, pair_targets, reduction="none"
    ).flatten(3).mean(-1)
    intersection = (probabilities[:, :, None] * targets[:, None]).flatten(3).sum(-1)
    denominator = probabilities[:, :, None].flatten(3).sum(-1) + targets[:, None].flatten(3).sum(-1)
    dice = 1 - (2 * intersection + 1) / (denominator + 1)
    return bce + dice


def proposal_loss(logits, targets, dice_weight, overlap_weight):
    probabilities = logits.sigmoid()
    pair_logits = logits[:, :, None].expand(-1, -1, 2, -1, -1)
    pair_targets = targets[:, None].expand(-1, 2, -1, -1, -1)
    bce = F.binary_cross_entropy_with_logits(
        pair_logits, pair_targets, reduction="none"
    ).flatten(3).mean(-1)
    intersection = (probabilities[:, :, None] * targets[:, None]).flatten(3).sum(-1)
    denominator = probabilities[:, :, None].flatten(3).sum(-1) + targets[:, None].flatten(3).sum(-1)
    dice = 1 - (2 * intersection + 1) / (denominator + 1)
    costs = bce + dice_weight * dice
    direct = costs[:, 0, 0] + costs[:, 1, 1]
    swapped = costs[:, 0, 1] + costs[:, 1, 0]
    matched = torch.minimum(direct, swapped).mean() / 2
    overlap = (probabilities[:, 0] * probabilities[:, 1]).mean()
    return matched + overlap_weight * overlap, matched, overlap


@torch.inference_mode()
def evaluate(model, keys, samples, device, batch_size):
    model.eval()
    matched_ious, overlaps = [], []
    by_task = {}
    for start in range(0, len(keys), batch_size):
        patches, targets = batch(keys[start:start + batch_size], samples, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probabilities = model(patches).sigmoid().float()
        intersection = (probabilities[:, :, None] * targets[:, None]).flatten(3).sum(-1)
        union = probabilities[:, :, None].flatten(3).sum(-1) + targets[:, None].flatten(3).sum(-1) - intersection
        ious = intersection / union.clamp_min(1e-6)
        direct = ious[:, 0, 0] + ious[:, 1, 1]
        swapped = ious[:, 0, 1] + ious[:, 1, 0]
        batch_ious = (torch.maximum(direct, swapped) / 2).cpu().tolist()
        matched_ious.extend(batch_ious)
        binary = probabilities > 0.5
        batch_overlaps = ((binary[:, 0] & binary[:, 1]).sum((1, 2)).float() /
                          (binary[:, 0] | binary[:, 1]).sum((1, 2)).clamp_min(1)).cpu().tolist()
        overlaps.extend(batch_overlaps)
        for key, iou, overlap in zip(keys[start:start + batch_size], batch_ious, batch_overlaps):
            values = by_task.setdefault(str(key[0]), {"ious": [], "overlaps": []})
            values["ious"].append(iou)
            values["overlaps"].append(overlap)
    return {
        "permutation_mean_soft_iou": sum(matched_ious) / len(matched_ious),
        "candidate_pair_binary_iou": sum(overlaps) / len(overlaps),
        "by_task": {
            task: {
                "permutation_mean_soft_iou": sum(values["ious"]) / len(values["ious"]),
                "candidate_pair_binary_iou": sum(values["overlaps"]) / len(values["overlaps"]),
                "num_samples": len(values["ious"]),
            }
            for task, values in by_task.items()
        },
    }


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("proposal training requires CUDA")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda:0")
    samples, splits = load_samples(args.data_dir), load_splits(args.split_manifest, args.protocol)
    model = BowlProposalSegmenter(args.hidden_dim, args.heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    for step in range(1, args.steps + 1):
        keys = rng.sample(splits["train"], min(args.batch_size, len(splits["train"])))
        patches, targets = batch(keys, samples, device)
        patches, targets = augment(
            patches, targets, args.horizontal_flip, args.patch_dropout
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(patches)
            loss, matched, overlap = proposal_loss(
                logits, targets, args.dice_weight, args.overlap_weight
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step == args.steps or step % args.log_every == 0:
            record = {"step": step, "loss": float(loss), "matched_loss": float(matched),
                      "soft_overlap": float(overlap)}
            history.append(record)
            print(json.dumps(record), flush=True)
    metrics = {name: evaluate(model, keys, samples, device, args.batch_size)
               for name, keys in splits.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"bowl-proposals-step-{args.steps:06d}.safetensors"
    save_file({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint)
    report = {"protocol": args.protocol, "checkpoint": str(checkpoint),
              "history": history, "metrics": metrics}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
