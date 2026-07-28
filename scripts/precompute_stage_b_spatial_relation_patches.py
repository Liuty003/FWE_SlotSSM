#!/usr/bin/env python
"""Precompute frozen OpenVLA patches for Stage-B v2 relation supervision."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from scripts.train_slot_stage3_openvla import max_memory_map


ROLE_ORDER = (
    "target", "distractor", "goal", "reference_0", "reference_1", "robot"
)
RELATION_ORDER = ("between", "in", "next_to", "on", "table_center")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial"
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path(
            "/data2/lty/naive-e-slotssm-runs/"
            "stage-b-spatial-relation-data/manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "/data2/lty/naive-e-slotssm-runs/"
            "stage-b-spatial-relation-patches"
        ),
    )
    parser.add_argument("--tasks", default="", help="Optional comma-separated task IDs")
    parser.add_argument("--max-samples-per-task", type=int, default=0)
    parser.add_argument("--vision-microbatch", type=int, default=8)
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    parser.add_argument("--free-fraction", type=float, default=0.75)
    return parser.parse_args()


def load_model(args):
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        device_map="balanced_low_0",
        max_memory=max_memory_map(args.reserve_gib, args.free_fraction),
    )
    model.requires_grad_(False)
    model.eval()
    return processor, model, next(model.vision_backbone.parameters()).device


def group_samples(manifest: dict, requested: set[int], limit: int):
    grouped = defaultdict(list)
    for sample in manifest["samples"]:
        task_id = int(sample["task_id"])
        if not requested or task_id in requested:
            grouped[task_id].append(sample)
    if requested - set(grouped):
        raise ValueError(f"Tasks absent from manifest: {sorted(requested - set(grouped))}")
    for task_id in grouped:
        grouped[task_id].sort(key=lambda item: int(item["state_id"]))
        if limit:
            grouped[task_id] = grouped[task_id][:limit]
    return dict(sorted(grouped.items()))


def load_task(samples, transform):
    pixels, masks, presence, state_ids, relation_ids = [], [], [], [], []
    for sample in samples:
        pixels.append(transform(Image.open(sample["rgb"]).convert("RGB")))
        sample_masks, sample_presence = [], []
        with np.load(sample["masks"]) as stored:
            for role in ROLE_ORDER:
                exists = role in stored.files
                sample_presence.append(exists)
                sample_masks.append(
                    stored[role].astype(np.uint8)
                    if exists else np.zeros((256, 256), dtype=np.uint8)
                )
        masks.append(np.stack(sample_masks))
        presence.append(sample_presence)
        state_ids.append(int(sample["state_id"]))
        relation_ids.append(RELATION_ORDER.index(sample["relation"]))
    mask_tensor = torch.from_numpy(np.stack(masks)).float()
    mask_tensor = F.interpolate(mask_tensor, size=(64, 64), mode="nearest").to(torch.uint8)
    return (
        torch.stack(pixels), mask_tensor,
        torch.tensor(presence, dtype=torch.bool),
        torch.tensor(state_ids, dtype=torch.int64),
        torch.tensor(relation_ids, dtype=torch.int64),
    )


@torch.inference_mode()
def encode(model, pixels, device, microbatch):
    patches = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(pixels), microbatch):
            encoded = model.vision_backbone(pixels[start : start + microbatch].to(device))
            patches.append(encoded.reshape(encoded.shape[0], -1, 2176).cpu())
    return torch.cat(patches).to(torch.bfloat16).contiguous()


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("Patch precomputation requires CUDA")
    if args.vision_microbatch < 1 or args.max_samples_per_task < 0:
        raise ValueError("Invalid microbatch/sample limit")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    requested = {int(item) for item in args.tasks.split(",") if item.strip()}
    grouped = group_samples(manifest, requested, args.max_samples_per_task)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor, model, device = load_model(args)
    records = []
    for task_id, samples in grouped.items():
        pixels, masks, presence, state_ids, relation_ids = load_task(
            samples, processor.image_processor.apply_transform
        )
        patches = encode(model, pixels, device, args.vision_microbatch)
        expected = (len(samples), 256, 2176)
        if tuple(patches.shape) != expected or not torch.isfinite(patches).all():
            raise RuntimeError(f"task {task_id}: invalid patches {tuple(patches.shape)}")
        output = args.output_dir / f"task{task_id}.safetensors"
        save_file({
            "patches": patches,
            "role_masks": masks.contiguous(),
            "role_presence": presence.contiguous(),
            "state_ids": state_ids,
            "relation_ids": relation_ids,
        }, output)
        record = {
            "task_id": task_id,
            "num_samples": len(samples),
            "output": str(output),
            "patches_shape": list(patches.shape),
            "role_masks_shape": list(masks.shape),
            "role_presence_counts": {
                role: int(presence[:, index].sum())
                for index, role in enumerate(ROLE_ORDER)
            },
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        del pixels, masks, presence, state_ids, relation_ids, patches
    output_manifest = {
        "schema_version": 2,
        "source_manifest": str(args.manifest),
        "model": args.model,
        "role_order": list(ROLE_ORDER),
        "relation_order": list(RELATION_ORDER),
        "patch_dim": 2176,
        "tasks": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved relation patch dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
