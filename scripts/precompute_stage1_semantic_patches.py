#!/usr/bin/env python
"""Precompute frozen OpenVLA vision patches for Stage-B semantic Slot training."""

from __future__ import annotations

import argparse
import json
import sys
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


COLORS = {
    "target": np.array([255, 55, 55], dtype=np.uint8),
    "distractor": np.array([55, 155, 255], dtype=np.uint8),
    "goal": np.array([80, 230, 100], dtype=np.uint8),
    "robot": np.array([245, 200, 45], dtype=np.uint8),
}
ROLE_ORDER = tuple(COLORS)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial")
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/stage-b-task9-instance-data/manifest.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/stage-b-task9-semantic-patches"),
    )
    parser.add_argument("--episodes", default="", help="Optional comma-separated subset")
    parser.add_argument("--vision-microbatch", type=int, default=16)
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


def collect_episode_specs(manifest: dict) -> dict[int, dict]:
    episodes = {}
    for split, split_data in manifest["splits"].items():
        for window in split_data["windows"]:
            episode = int(window["episode"])
            spec = episodes.setdefault(episode, {
                "split": split,
                "rgb_pattern": window["rgb_pattern"],
                "segmentation_pattern": window["segmentation_pattern"],
                "num_frames": 0,
            })
            if spec["split"] != split:
                raise RuntimeError(f"Episode {episode} appears in multiple splits")
            spec["num_frames"] = max(spec["num_frames"], int(window["stop"]))
        for report in split_data["episode_reports"]:
            episode = int(report["episode"])
            if episode in episodes:
                episodes[episode]["num_frames"] = int(report["num_frames"])
    return episodes


def load_episode(spec, transform):
    pixels, masks = [], []
    for frame in range(spec["num_frames"]):
        rgb_path = Path(spec["rgb_pattern"].format(frame=frame))
        seg_path = Path(spec["segmentation_pattern"].format(frame=frame))
        if not rgb_path.is_file() or not seg_path.is_file():
            raise FileNotFoundError(f"Missing frame pair: {rgb_path} / {seg_path}")
        rgb = Image.open(rgb_path).convert("RGB")
        segmentation = np.asarray(Image.open(seg_path).convert("RGB"))
        pixels.append(transform(rgb))
        masks.append(np.stack([np.all(segmentation == COLORS[role], axis=-1) for role in ROLE_ORDER]))
    masks = torch.from_numpy(np.stack(masks)).float()
    masks = F.interpolate(masks, size=(64, 64), mode="nearest").to(torch.uint8)
    return torch.stack(pixels), masks


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
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    episodes = collect_episode_specs(manifest)
    requested = {int(value) for value in args.episodes.split(",") if value.strip()}
    if requested:
        missing = requested - set(episodes)
        if missing:
            raise ValueError(f"Episodes absent from manifest: {sorted(missing)}")
        episodes = {key: value for key, value in episodes.items() if key in requested}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor, model, device = load_model(args)
    output_manifest = {
        "schema_version": 1,
        "source_manifest": str(args.manifest),
        "model": args.model,
        "role_order": list(ROLE_ORDER),
        "patch_dim": 2176,
        "episodes": [],
    }
    for episode, spec in sorted(episodes.items()):
        output = args.output_dir / f"task9-episode{episode}.safetensors"
        pixels, masks = load_episode(spec, processor.image_processor.apply_transform)
        patches = encode(model, pixels, device, args.vision_microbatch)
        if patches.shape[:2] != (spec["num_frames"], 256):
            raise RuntimeError(f"Unexpected patch shape for episode {episode}: {patches.shape}")
        if masks.shape != (spec["num_frames"], len(ROLE_ORDER), 64, 64):
            raise RuntimeError(f"Unexpected mask shape for episode {episode}: {masks.shape}")
        if not torch.isfinite(patches).all():
            raise RuntimeError(f"Non-finite vision patches in episode {episode}")
        save_file({"patches": patches, "role_masks": masks.contiguous()}, output)
        record = {
            "episode": episode,
            "split": spec["split"],
            "num_frames": spec["num_frames"],
            "output": str(output),
            "patches_shape": list(patches.shape),
            "role_masks_shape": list(masks.shape),
            "visible_frames": {
                role: int(masks[:, index].flatten(1).any(1).sum())
                for index, role in enumerate(ROLE_ORDER)
            },
        }
        output_manifest["episodes"].append(record)
        print(json.dumps(record), flush=True)
        del pixels, masks, patches
    (args.output_dir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved semantic patch dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
