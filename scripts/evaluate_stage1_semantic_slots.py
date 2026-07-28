#!/usr/bin/env python
"""Evaluate fixed-role Stage-B Slot checkpoints on complete held-out episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_stage1_semantic_slots import ROLE_ORDER, load_manifest, load_modules


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/stage-b-task9-semantic-patches"),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sums_template():
    return {
        "frames": 0,
        "soft_iou_sum": torch.zeros(len(ROLE_ORDER), dtype=torch.float64),
        "hard_iou_sum": torch.zeros(len(ROLE_ORDER), dtype=torch.float64),
        "target_distractor_overlap_sum": 0.0,
    }


def update_sums(sums, probabilities, targets, threshold):
    soft_intersection = (probabilities * targets).flatten(2).sum(-1)
    soft_union = probabilities.flatten(2).sum(-1) + targets.flatten(2).sum(-1) - soft_intersection
    soft_iou = soft_intersection / soft_union.clamp_min(1e-6)
    hard = probabilities >= threshold
    truth = targets.bool()
    hard_intersection = (hard & truth).flatten(2).sum(-1).float()
    hard_union = (hard | truth).flatten(2).sum(-1).float()
    hard_iou = hard_intersection / hard_union.clamp_min(1)
    overlap_intersection = (hard[:, 0] & hard[:, 1]).flatten(1).sum(-1).float()
    overlap_union = (hard[:, 0] | hard[:, 1]).flatten(1).sum(-1).float()
    overlap = overlap_intersection / overlap_union.clamp_min(1)
    sums["frames"] += len(probabilities)
    sums["soft_iou_sum"] += soft_iou.sum(0).double().cpu()
    sums["hard_iou_sum"] += hard_iou.sum(0).double().cpu()
    sums["target_distractor_overlap_sum"] += float(overlap.sum())


def finalize(sums):
    frames = max(1, sums["frames"])
    return {
        "frames": sums["frames"],
        "roles": {
            role: {
                "mean_soft_iou": float(sums["soft_iou_sum"][index] / frames),
                "mean_hard_iou": float(sums["hard_iou_sum"][index] / frames),
            }
            for index, role in enumerate(ROLE_ORDER)
        },
        "mean_target_distractor_prediction_overlap": (
            sums["target_distractor_overlap_sum"] / frames
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("Semantic Slot evaluation requires CUDA")
    device = torch.device("cuda:0")
    _, splits = load_manifest(args.data_dir)
    tokenizer, mask_head = load_modules(args.checkpoint, device)
    tokenizer.eval()
    mask_head.eval()
    total = sums_template()
    episodes = []
    for entry in sorted(splits[args.split], key=lambda item: item["episode"]):
        tensors = load_file(entry["output"])
        patches = tensors["patches"]
        masks = tensors["role_masks"]
        episode_sums = sums_template()
        past = None
        for patch, target in zip(patches, masks):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                slots, _, _ = tokenizer(patch[None].to(device), past_slots=past)
                probabilities = mask_head(slots[:, : len(ROLE_ORDER)]).sigmoid().float()
            target = target[None].to(device).float()
            update_sums(episode_sums, probabilities, target, args.mask_threshold)
            update_sums(total, probabilities, target, args.mask_threshold)
            past = slots[:, None]
        result = {"episode": entry["episode"], **finalize(episode_sums)}
        episodes.append(result)
        print(json.dumps(result), flush=True)
    report = {
        "schema_version": 1,
        "checkpoint": args.checkpoint,
        "data_dir": str(args.data_dir),
        "split": args.split,
        "mask_threshold": args.mask_threshold,
        "summary": finalize(total),
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
