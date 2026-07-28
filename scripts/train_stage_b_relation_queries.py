#!/usr/bin/env python
"""Train a minimal relation/reference-conditioned target-query model."""

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


ROLE_ORDER = (
    "target", "distractor", "goal", "reference_0", "reference_1", "robot"
)
RELATION_ORDER = ("between", "in", "next_to", "on", "table_center")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(
            "/data2/lty/naive-e-slotssm-runs/"
            "stage-b-spatial-relation-patches"
        ),
    )
    parser.add_argument(
        "--split-manifest", type=Path,
        default=Path(
            "/data2/lty/naive-e-slotssm-runs/"
            "stage-b-spatial-relation-data/split_manifest.json"
        ),
    )
    parser.add_argument(
        "--language-features", type=Path,
        default=Path(
            "/data2/lty/naive-e-slotssm-runs/stage-b-spatial-relation-patches/"
            "language_features.safetensors"
        ),
    )
    parser.add_argument(
        "--protocol",
        choices=(
            "state_generalization", "compositional_task_generalization",
            "compositional_reference_seen_fold_a",
            "compositional_reference_seen_fold_b",
        ),
        default="state_generalization",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "/data2/lty/naive-e-slotssm-runs/"
            "stage-b-relation-query-smoke"
        ),
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-pos-weight", type=float, default=50.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--separation-weight", type=float, default=0.1)
    parser.add_argument("--switch-ranking-weight", type=float, default=0.5)
    parser.add_argument("--switch-ranking-margin", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--counterfactual-training", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


class RelationQuerySegmenter(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, language_dim: int = 4096):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.patch_norm = nn.LayerNorm(2176)
        self.patch_projection = nn.Linear(2176, hidden_dim)
        self.role_queries = nn.Parameter(torch.randn(len(ROLE_ORDER), hidden_dim) * 0.02)
        self.relation_embedding = nn.Embedding(len(RELATION_ORDER), hidden_dim)
        self.language_norm = nn.LayerNorm(language_dim)
        self.language_projection = nn.Linear(language_dim, hidden_dim)
        self.reference_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.target_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.other_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.reference_to_target = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.mask_refinement = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(
        self, patches, relation_ids, role_presence,
        instruction_features, relation_features, reference_features,
    ):
        features = self.patch_projection(self.patch_norm(patches.float()))
        batch = len(features)
        language = self.language_projection(self.language_norm(torch.cat((
            instruction_features[:, None], relation_features[:, None], reference_features,
        ), dim=1).float()))
        instruction_language = language[:, 0]
        relation = self.relation_embedding(relation_ids) + language[:, 1]
        reference_language = language[:, 2:4]
        reference_queries = self.role_queries[3:5][None].expand(batch, -1, -1)
        # Object identity is invariant to whether the target is on / next to / in
        # the reference.  Relation conditioning belongs only in the target query.
        reference_queries = reference_queries + reference_language
        reference_slots, _ = self.reference_attention(
            reference_queries, features, features, need_weights=False
        )
        reference_presence = role_presence[:, 3:5].float()
        reference_context = (
            reference_slots * reference_presence[:, :, None]
        ).sum(1) / reference_presence.sum(1, keepdim=True).clamp_min(1)
        target_query = (
            self.role_queries[0][None] + relation
            + instruction_language
            + self.reference_to_target(reference_context)
        )[:, None]
        target_slot, _ = self.target_attention(
            target_query, features, features, need_weights=False
        )
        other_indices = torch.tensor([1, 2, 5], device=patches.device)
        other_queries = self.role_queries[other_indices][None].expand(batch, -1, -1)
        other_slots, _ = self.other_attention(
            other_queries, features, features, need_weights=False
        )
        slots = torch.stack((
            target_slot[:, 0], other_slots[:, 0], other_slots[:, 1],
            reference_slots[:, 0], reference_slots[:, 1], other_slots[:, 2],
        ), dim=1)
        slots = F.normalize(self.query_norm(slots), dim=-1)
        spatial = F.normalize(self.spatial_norm(features), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100)
        logits_16 = scale * torch.einsum("brd,bpd->brp", slots, spatial)
        logits_16 = logits_16.reshape(batch * len(ROLE_ORDER), 1, 16, 16)
        coarse = F.interpolate(logits_16, size=(64, 64), mode="bilinear", align_corners=False)
        logits = (coarse + self.mask_refinement(coarse)).reshape(
            batch, len(ROLE_ORDER), 64, 64
        )
        return logits, slots


def load_dataset(data_dir: Path, language_features_path: Path):
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if tuple(manifest["role_order"]) != ROLE_ORDER:
        raise RuntimeError(f"Unexpected roles: {manifest['role_order']}")
    if tuple(manifest["relation_order"]) != RELATION_ORDER:
        raise RuntimeError(f"Unexpected relations: {manifest['relation_order']}")
    language_tensors = load_file(language_features_path)
    language_by_task = {
        int(task_id): {
            "instruction_features": language_tensors["instruction_features"][index],
            "relation_features": language_tensors["relation_features"][index],
            "reference_features": language_tensors["reference_features"][index],
            **({
                "counterfactual_instruction_features":
                    language_tensors["counterfactual_instruction_features"][index],
                "counterfactual_relation_features":
                    language_tensors["counterfactual_relation_features"][index],
                "counterfactual_reference_features":
                    language_tensors["counterfactual_reference_features"][index],
            } if "counterfactual_instruction_features" in language_tensors else {}),
        }
        for index, task_id in enumerate(language_tensors["task_ids"].tolist())
    }
    samples = {}
    for task in manifest["tasks"]:
        tensors = load_file(task["output"])
        for index, state_id in enumerate(tensors["state_ids"].tolist()):
            key = (int(task["task_id"]), int(state_id))
            if key in samples:
                raise RuntimeError(f"Duplicate sample {key}")
            samples[key] = {
                name: tensors[name][index]
                for name in ("patches", "role_masks", "role_presence", "relation_ids")
            }
            samples[key].update(language_by_task[int(task["task_id"])])
            if "counterfactual_target_masks" in tensors:
                samples[key].update({
                    "counterfactual_target_masks":
                        tensors["counterfactual_target_masks"][index],
                    "counterfactual_reference_masks":
                        tensors["counterfactual_reference_masks"][index],
                    "counterfactual_reference_presence":
                        tensors["counterfactual_reference_presence"][index],
                    "counterfactual_relation_ids":
                        tensors["counterfactual_relation_ids"][index],
                })
    return samples


def load_splits(path: Path, protocol: str, samples):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    splits = {}
    for split, rows in manifest["protocols"][protocol].items():
        keys = [(int(row["task_id"]), int(row["state_id"])) for row in rows]
        missing = set(keys) - set(samples)
        if missing:
            raise RuntimeError(f"{split} samples missing from patches: {sorted(missing)[:5]}")
        splits[split] = keys
    flat = [key for keys in splits.values() for key in keys]
    if len(flat) != len(set(flat)):
        raise RuntimeError(f"Leakage in protocol {protocol}")
    return splits


def query_sample(key, samples):
    if len(key) == 2:
        return samples[tuple(key)]
    task, state, variant = key
    sample = samples[(task, state)]
    if variant == "primary":
        return sample
    if variant != "counterfactual":
        raise ValueError(f"Unknown query variant: {variant}")
    role_masks = torch.stack((
        sample["counterfactual_target_masks"], sample["role_masks"][0],
        sample["role_masks"][2], sample["counterfactual_reference_masks"][0],
        sample["counterfactual_reference_masks"][1], sample["role_masks"][5],
    ))
    role_presence = torch.tensor([
        True, True, True,
        bool(sample["counterfactual_reference_presence"][0]),
        bool(sample["counterfactual_reference_presence"][1]), True,
    ])
    return {
        **sample, "role_masks": role_masks, "role_presence": role_presence,
        "relation_ids": sample["counterfactual_relation_ids"],
        "instruction_features": sample["counterfactual_instruction_features"],
        "relation_features": sample["counterfactual_relation_features"],
        "reference_features": sample["counterfactual_reference_features"],
    }


def make_batch(keys, samples, device):
    query_samples = [query_sample(key, samples) for key in keys]
    return tuple(
        torch.stack([sample[name] for sample in query_samples]).to(device)
        for name in (
            "patches", "role_masks", "role_presence", "relation_ids",
            "instruction_features", "relation_features", "reference_features",
        )
    )


def paired_query_keys(keys):
    return [item for key in keys for item in ((*key, "primary"), (*key, "counterfactual"))]


def loss_and_metrics(logits, targets, presence, max_pos_weight, dice_weight, separation_weight):
    targets = targets.float()
    valid = presence.float()
    positives = targets.flatten(2).sum(-1)
    pixels = targets.shape[-1] * targets.shape[-2]
    pos_weight = ((pixels - positives) / positives.clamp_min(1)).clamp(1, max_pos_weight)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.where(targets.bool(), pos_weight[:, :, None, None], 1.0)
    bce_role = (bce * weights).flatten(2).mean(-1)
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).flatten(2).sum(-1)
    denominator = probabilities.flatten(2).sum(-1) + positives
    dice_role = 1 - (2 * intersection + 1) / (denominator + 1)
    normalizer = valid.sum().clamp_min(1)
    bce_loss = (bce_role * valid).sum() / normalizer
    dice_loss = (dice_role * valid).sum() / normalizer
    overlap = (probabilities[:, 0] * probabilities[:, 1]).mean()
    loss = bce_loss + dice_weight * dice_loss + separation_weight * overlap
    union = probabilities.flatten(2).sum(-1) + positives - intersection
    soft_iou = intersection / union.clamp_min(1e-6)
    return loss, {
        "loss": float(loss.detach()), "loss_bce": float(bce_loss.detach()),
        "loss_dice": float(dice_loss.detach()), "target_distractor_overlap": float(overlap.detach()),
        "role_soft_iou": [
            float((soft_iou[:, role] * valid[:, role]).sum().detach() / valid[:, role].sum().clamp_min(1))
            for role in range(len(ROLE_ORDER))
        ],
    }


def paired_switch_ranking_loss(logits, targets, margin):
    """Rank the queried bowl above the alternate bowl for adjacent query pairs."""
    if logits.shape[0] % 2:
        raise ValueError("paired switch ranking requires an even query batch")
    target_logits = logits[:, 0]
    losses, margins = [], []
    for index in range(0, logits.shape[0], 2):
        primary_mask = targets[index, 0].bool()
        counterfactual_mask = targets[index + 1, 0].bool()
        for query_index, correct_mask, wrong_mask in (
            (index, primary_mask, counterfactual_mask),
            (index + 1, counterfactual_mask, primary_mask),
        ):
            correct_score = target_logits[query_index][correct_mask].mean()
            wrong_score = target_logits[query_index][wrong_mask].mean()
            score_margin = correct_score - wrong_score
            losses.append(F.relu(margin - score_margin))
            margins.append(score_margin.detach())
    return torch.stack(losses).mean(), torch.stack(margins).mean()


@torch.inference_mode()
def evaluate(model, keys, samples, args, device):
    model.eval()
    sums = torch.zeros(len(ROLE_ORDER), dtype=torch.float64)
    counts = torch.zeros(len(ROLE_ORDER), dtype=torch.float64)
    for start in range(0, len(keys), args.batch_size):
        patches, masks, presence, relations, instruction, relation_text, reference_text = make_batch(
            keys[start : start + args.batch_size], samples, device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(
                patches, relations, presence, instruction, relation_text, reference_text
            )
        probabilities, targets = logits.sigmoid().float(), masks.float()
        intersection = (probabilities * targets).flatten(2).sum(-1)
        union = probabilities.flatten(2).sum(-1) + targets.flatten(2).sum(-1) - intersection
        iou = intersection / union.clamp_min(1e-6)
        sums += (iou * presence).sum(0).double().cpu()
        counts += presence.sum(0).double().cpu()
    return {
        role: float(sums[index] / counts[index].clamp_min(1))
        for index, role in enumerate(ROLE_ORDER)
    }


@torch.inference_mode()
def evaluate_switch(model, keys, samples, args, device):
    """Measure whether changing only the language query switches bowls."""
    model.eval()
    totals = {
        "primary_correct_iou": 0.0, "primary_wrong_iou": 0.0,
        "counterfactual_correct_iou": 0.0, "counterfactual_wrong_iou": 0.0,
        "paired_switch_accuracy": 0.0, "prediction_change_l1": 0.0,
    }
    count = 0
    physical_batch = max(1, args.batch_size // 2)
    for start in range(0, len(keys), physical_batch):
        physical_keys = keys[start:start + physical_batch]
        patches, masks, presence, relations, instruction, relation_text, reference_text = make_batch(
            paired_query_keys(physical_keys), samples, device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(
                patches, relations, presence, instruction, relation_text, reference_text
            )
        predictions = logits.sigmoid().float()[:, 0]
        for index in range(len(physical_keys)):
            primary = predictions[2 * index]
            counterfactual = predictions[2 * index + 1]
            primary_mask = masks[2 * index, 0].float()
            counterfactual_mask = masks[2 * index + 1, 0].float()

            def soft_iou(prediction, target):
                intersection = (prediction * target).sum()
                union = prediction.sum() + target.sum() - intersection
                return float(intersection / union.clamp_min(1e-6))

            pc = soft_iou(primary, primary_mask)
            pw = soft_iou(primary, counterfactual_mask)
            cc = soft_iou(counterfactual, counterfactual_mask)
            cw = soft_iou(counterfactual, primary_mask)
            totals["primary_correct_iou"] += pc
            totals["primary_wrong_iou"] += pw
            totals["counterfactual_correct_iou"] += cc
            totals["counterfactual_wrong_iou"] += cw
            totals["paired_switch_accuracy"] += float(pc > pw and cc > cw)
            totals["prediction_change_l1"] += float((primary - counterfactual).abs().mean())
            count += 1
    return {name: value / max(count, 1) for name, value in totals.items()}


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("Relation query training requires CUDA")
    if args.steps < 1 or args.batch_size < 1 or args.log_every < 1:
        raise ValueError("steps, batch size, and log-every must be positive")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda:0")
    samples = load_dataset(args.data_dir, args.language_features)
    splits = load_splits(args.split_manifest, args.protocol, samples)
    if args.counterfactual_training:
        if args.batch_size < 2 or args.batch_size % 2:
            raise ValueError("counterfactual training requires an even batch size >= 2")
        required = (
            "counterfactual_target_masks", "counterfactual_reference_masks",
            "counterfactual_relation_ids", "counterfactual_instruction_features",
            "counterfactual_relation_features", "counterfactual_reference_features",
        )
        missing = [name for name in required if name not in next(iter(samples.values()))]
        if missing:
            raise RuntimeError(f"Counterfactual tensors missing: {missing}")
    language_dim = int(next(iter(samples.values()))["instruction_features"].shape[-1])
    model = RelationQuerySegmenter(args.hidden_dim, args.heads, language_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    initial_validation = evaluate(model, splits["validation"], samples, args, device)
    history = []
    model.train()
    for step in range(1, args.steps + 1):
        physical_batch = args.batch_size // 2 if args.counterfactual_training else args.batch_size
        physical_keys = rng.sample(splits["train"], k=min(physical_batch, len(splits["train"])))
        keys = paired_query_keys(physical_keys) if args.counterfactual_training else physical_keys
        patches, masks, presence, relations, instruction, relation_text, reference_text = make_batch(
            keys, samples, device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(
                patches, relations, presence, instruction, relation_text, reference_text
            )
            loss, metrics = loss_and_metrics(
                logits, masks, presence, args.max_pos_weight,
                args.dice_weight, args.separation_weight,
            )
            if args.counterfactual_training:
                ranking_loss, ranking_margin = paired_switch_ranking_loss(
                    logits, masks, args.switch_ranking_margin
                )
                loss = loss + args.switch_ranking_weight * ranking_loss
                metrics.update({
                    "loss": float(loss.detach()),
                    "switch_ranking_loss": float(ranking_loss.detach()),
                    "switch_ranking_margin": float(ranking_margin),
                })
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Non-finite relation-query gradient")
        optimizer.step()
        record = {"step": step, "keys": keys, "grad_norm": float(grad_norm), **metrics}
        history.append(record)
        if step == 1 or step == args.steps or step % args.log_every == 0:
            print(json.dumps(record), flush=True)
    validation = evaluate(model, splits["validation"], samples, args, device)
    test = evaluate(model, splits["test"], samples, args, device)
    validation_switch = (
        evaluate_switch(model, splits["validation"], samples, args, device)
        if args.counterfactual_training else None
    )
    test_switch = (
        evaluate_switch(model, splits["test"], samples, args, device)
        if args.counterfactual_training else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"relation-query-step-{args.steps:06d}.safetensors"
    save_file(
        {key: value.detach().cpu() for key, value in model.state_dict().items()}, checkpoint,
        metadata={"protocol": args.protocol, "steps": str(args.steps)},
    )
    # Strict reload is part of the smoke contract.
    reloaded = RelationQuerySegmenter(args.hidden_dim, args.heads, language_dim)
    reloaded.load_state_dict(load_file(checkpoint), strict=True)
    report = {
        "schema_version": 1, "protocol": args.protocol,
        "language_features": str(args.language_features), "language_dim": language_dim,
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "checkpoint": str(checkpoint), "initial_validation": initial_validation,
        "history": history, "validation": validation, "test": test,
        "counterfactual_training": args.counterfactual_training,
        "validation_switch": validation_switch, "test_switch": test_switch,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
