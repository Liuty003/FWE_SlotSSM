#!/usr/bin/env python
"""Audit Stage-1 target grounding and identity on verified task-9 replays."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import (
    OpenVLAForActionPrediction,
    OpenVLAForActionPrediction_SlotAtt,
)
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from scripts.train_slot_stage3_openvla import max_memory_map, select_auxiliary_device


DEFAULT_CHECKPOINT = (
    "/data2/lty/naive-e-slotssm-runs/stage1-adapters/"
    "openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--"
    "full10tasks-h16-sync6-s20k-eb6-no-gpu3--image_aug--multiview1000--object/"
    "pretrained_slots-wo-filters_s16.safetensors"
)
OBJECTS = ("akita_black_bowl_1", "akita_black_bowl_2", "plate_1", "robot")
COLORS = {
    "akita_black_bowl_1": np.array([255, 55, 55], dtype=np.uint8),
    "akita_black_bowl_2": np.array([55, 155, 255], dtype=np.uint8),
    "plate_1": np.array([80, 230, 100], dtype=np.uint8),
    "robot": np.array([245, 200, 45], dtype=np.uint8),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--slot-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/stage-a-task9-segmentation-preflight"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/stage-a-task9-slot-audit"),
    )
    parser.add_argument("--episodes", default="0,6,7")
    parser.add_argument("--max-frames", type=int, default=0, help="0 uses every verified frame")
    parser.add_argument("--vision-microbatch", type=int, default=16)
    parser.add_argument("--slot-microbatch", type=int, default=32)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    parser.add_argument("--free-fraction", type=float, default=0.75)
    parser.add_argument("--video-fps", type=int, default=20)
    return parser.parse_args()


def load_verified_episode(root: Path, episode: int, transform, max_frames: int):
    directory = root / f"task9-episode{episode}"
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if not report.get("passed_replay_check"):
        raise RuntimeError(f"Replay preflight did not pass: {directory}")
    frames = report["frames"][: max_frames or None]
    rgbs, pixels, gt = [], [], []
    for frame in frames:
        index = int(frame["frame"])
        rgb = np.asarray(Image.open(directory / f"frame-{index:03d}-rgb.png").convert("RGB"))
        segmentation = np.asarray(
            Image.open(directory / f"frame-{index:03d}-segmentation.png").convert("RGB")
        )
        rgbs.append(rgb)
        pixels.append(transform(Image.fromarray(rgb)))
        gt.append(np.stack([np.all(segmentation == COLORS[name], axis=-1) for name in OBJECTS]))
    if not rgbs:
        raise RuntimeError(f"No verified frames found in {directory}")
    return np.stack(rgbs), torch.stack(pixels), torch.from_numpy(np.stack(gt)), report


@torch.inference_mode()
def encode_vision(base, pixels, vision_device, auxiliary_device, microbatch):
    chunks = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(pixels), microbatch):
            encoded = base.vision_backbone(pixels[start : start + microbatch].to(vision_device))
            chunks.append(encoded.reshape(encoded.shape[0], -1, 2176).to(auxiliary_device))
    return torch.cat(chunks)


@torch.inference_mode()
def infer_slots(slot_model, patches, auxiliary_device, microbatch, tracked):
    tokenizer = slot_model.object_centric_tokenizer
    mask_head = slot_model.object_centric_mask_head
    all_slots, all_masks = [], []
    past = None
    with torch.cuda.device(auxiliary_device), torch.autocast("cuda", dtype=torch.bfloat16):
        if tracked:
            for frame in patches:
                current, _, _ = tokenizer(frame[None], past_slots=past)
                masks = mask_head(current)
                all_slots.append(current[0].float().cpu())
                all_masks.append(masks[0].float().sigmoid().cpu())
                past = current[:, None]
        else:
            for start in range(0, len(patches), microbatch):
                current, _, _ = tokenizer(patches[start : start + microbatch], past_slots=None)
                masks = mask_head(current)
                all_slots.extend(current.float().cpu())
                all_masks.extend(masks.float().sigmoid().cpu())
    return torch.stack(all_slots), torch.stack(all_masks)


def iou_matrix(predicted: torch.Tensor, gt: torch.Tensor, threshold: float):
    gt64 = F.interpolate(gt.float(), size=predicted.shape[-2:], mode="nearest").bool()
    hard = predicted >= threshold
    hard_intersection = (hard[:, :, None] & gt64[:, None]).flatten(3).sum(-1).float()
    hard_union = (hard[:, :, None] | gt64[:, None]).flatten(3).sum(-1).float()
    hard_iou = hard_intersection / hard_union.clamp_min(1)
    soft_intersection = (predicted[:, :, None] * gt64[:, None]).flatten(3).sum(-1)
    soft_union = (
        predicted[:, :, None].flatten(3).sum(-1)
        + gt64[:, None].flatten(3).sum(-1)
        - soft_intersection
    )
    soft_iou = soft_intersection / soft_union.clamp_min(1e-6)
    visible = gt64.flatten(2).any(-1)
    return hard_iou, soft_iou, visible


def summarize_mode(mode: str, hard_iou, soft_iou, visible):
    best_soft, best_slot = soft_iou.max(dim=1)  # [T,O]
    best_hard = hard_iou.gather(1, best_slot[:, None]).squeeze(1)
    target_slot = best_slot[:, 0]
    distractor_slot = best_slot[:, 1]
    target_visible = visible[:, 0]
    switches = int(
        ((target_slot[1:] != target_slot[:-1]) & target_visible[1:] & target_visible[:-1]).sum()
    )
    transitions = int((target_visible[1:] & target_visible[:-1]).sum())
    collision = (target_slot == distractor_slot) & visible[:, 0] & visible[:, 1]
    both_visible = visible[:, 0] & visible[:, 1]
    per_object = {}
    for index, name in enumerate(OBJECTS):
        valid = visible[:, index]
        per_object[name] = {
            "visible_frames": int(valid.sum()),
            "mean_best_soft_iou": float(best_soft[valid, index].mean()) if valid.any() else None,
            "mean_best_hard_iou": float(best_hard[valid, index].mean()) if valid.any() else None,
        }
    summary = {
        "mode": mode,
        "num_frames": len(best_slot),
        "objects": per_object,
        "target_identity_switches": switches,
        "target_visible_transitions": transitions,
        "target_switch_rate": switches / max(1, transitions),
        "target_distractor_collision_frames": int(collision.sum()),
        "target_distractor_both_visible_frames": int(both_visible.sum()),
        "target_distractor_collision_rate": int(collision.sum()) / max(1, int(both_visible.sum())),
    }
    return summary, best_slot, best_soft, best_hard


def mask_to_image(mask: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    resized = F.interpolate(mask[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]
    return resized.numpy()


def render_mode(rgb, masks, best_slot, mode):
    output = rgb.astype(np.float32).copy()
    height, width = rgb.shape[:2]
    for object_index, name in enumerate(OBJECTS[:2]):
        probability = mask_to_image(masks[int(best_slot[object_index])], (height, width))
        region = probability >= 0.5
        output[region] = 0.55 * output[region] + 0.45 * COLORS[name]
    output = np.clip(output, 0, 255).astype(np.uint8)
    # A small mode marker: cyan for independent, magenta for tracked.
    output[:6] = np.array([0, 220, 220] if mode == "independent" else [220, 0, 220], np.uint8)
    return output


def write_episode_outputs(output_dir, episode, rgbs, gt, results, fps):
    episode_dir = output_dir / f"task9-episode{episode}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    writer = imageio.get_writer(episode_dir / "slot_audit_overlay.mp4", fps=fps)
    try:
        for frame in range(len(rgbs)):
            row = {"episode": episode, "frame": frame}
            panels = [rgbs[frame]]
            for mode, result in results.items():
                slots = result["best_slot"][frame]
                panels.append(render_mode(rgbs[frame], result["masks"][frame], slots, mode))
                for object_index, name in enumerate(OBJECTS):
                    row[f"{mode}_{name}_slot"] = int(slots[object_index])
                    row[f"{mode}_{name}_soft_iou"] = float(result["best_soft"][frame, object_index])
                    row[f"{mode}_{name}_hard_iou"] = float(result["best_hard"][frame, object_index])
            writer.append_data(np.concatenate(panels, axis=1))
            rows.append(row)
    finally:
        writer.close()
    with (episode_dir / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    tensors = {}
    for mode, result in results.items():
        tensors[f"{mode}_slots"] = result["slots"].to(torch.float16).contiguous()
        tensors[f"{mode}_best_slot"] = result["best_slot"].to(torch.int64).contiguous()
        tensors[f"{mode}_best_soft_iou"] = result["best_soft"].contiguous()
        tensors[f"{mode}_best_hard_iou"] = result["best_hard"].contiguous()
    tensors["gt_masks"] = gt.to(torch.uint8).contiguous()
    save_file(tensors, episode_dir / "audit_tensors.safetensors")


def load_models(args):
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    base = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        device_map="balanced_low_0",
        max_memory=max_memory_map(args.reserve_gib, args.free_fraction),
    )
    base.requires_grad_(False)
    vision_device = next(base.vision_backbone.parameters()).device
    auxiliary_device = select_auxiliary_device()
    slot_model = OpenVLAForActionPrediction_SlotAtt(model=base, number_of_slots=16)
    loaded = slot_model.load_state_dict(load_file(args.slot_checkpoint), strict=False)
    slot_model.requires_grad_(False)
    slot_model.object_centric_tokenizer.to(auxiliary_device, dtype=torch.bfloat16).eval()
    slot_model.object_centric_mask_head.to(auxiliary_device, dtype=torch.bfloat16).eval()
    print(
        f"vision={vision_device} auxiliary={auxiliary_device} "
        f"stage1_missing={len(loaded.missing_keys)} unexpected={len(loaded.unexpected_keys)}",
        flush=True,
    )
    return processor, base, slot_model, vision_device, auxiliary_device


def aggregate_reports(reports, mode):
    objects = {}
    for name in OBJECTS:
        entries = [report[mode]["objects"][name] for report in reports]
        visible = sum(entry["visible_frames"] for entry in entries)
        objects[name] = {
            "visible_frames": visible,
            "mean_best_soft_iou": sum(
                entry["mean_best_soft_iou"] * entry["visible_frames"] for entry in entries
                if entry["mean_best_soft_iou"] is not None
            ) / max(1, visible),
            "mean_best_hard_iou": sum(
                entry["mean_best_hard_iou"] * entry["visible_frames"] for entry in entries
                if entry["mean_best_hard_iou"] is not None
            ) / max(1, visible),
        }
    switches = sum(report[mode]["target_identity_switches"] for report in reports)
    transitions = sum(report[mode]["target_visible_transitions"] for report in reports)
    collisions = sum(report[mode]["target_distractor_collision_frames"] for report in reports)
    both_visible = sum(report[mode]["target_distractor_both_visible_frames"] for report in reports)
    return {
        "mode": mode,
        "num_episodes": len(reports),
        "num_frames": sum(report[mode]["num_frames"] for report in reports),
        "objects": objects,
        "target_identity_switches": switches,
        "target_visible_transitions": transitions,
        "target_switch_rate": switches / max(1, transitions),
        "target_distractor_collision_frames": collisions,
        "target_distractor_both_visible_frames": both_visible,
        "target_distractor_collision_rate": collisions / max(1, both_visible),
    }


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-1 audit requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor, base, slot_model, vision_device, auxiliary_device = load_models(args)
    episode_summaries = []
    for episode in [int(value) for value in args.episodes.split(",") if value]:
        rgbs, pixels, gt, preflight = load_verified_episode(
            args.input_dir, episode, processor.image_processor.apply_transform, args.max_frames
        )
        patches = encode_vision(
            base, pixels, vision_device, auxiliary_device, args.vision_microbatch
        )
        results = {}
        for mode, tracked in (("independent", False), ("tracked", True)):
            slots, masks = infer_slots(
                slot_model, patches, auxiliary_device, args.slot_microbatch, tracked
            )
            hard_iou, soft_iou, visible = iou_matrix(masks, gt, args.mask_threshold)
            summary, best_slot, best_soft, best_hard = summarize_mode(
                mode, hard_iou, soft_iou, visible
            )
            results[mode] = {
                "summary": summary, "slots": slots, "masks": masks,
                "best_slot": best_slot, "best_soft": best_soft, "best_hard": best_hard,
            }
            print(f"episode={episode} {json.dumps(summary, ensure_ascii=False)}", flush=True)
        write_episode_outputs(args.output_dir, episode, rgbs, gt, results, args.video_fps)
        report = {
            "episode": episode,
            "source": preflight["source"],
            "num_frames": len(rgbs),
            "independent": results["independent"]["summary"],
            "tracked": results["tracked"]["summary"],
        }
        episode_summaries.append(report)
        episode_dir = args.output_dir / f"task9-episode{episode}"
        (episode_dir / "metrics.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        del patches, pixels, results
        torch.cuda.empty_cache()
    final = {
        "schema_version": 1,
        "slot_checkpoint": args.slot_checkpoint,
        "mask_threshold": args.mask_threshold,
        "summary": {
            mode: aggregate_reports(episode_summaries, mode)
            for mode in ("independent", "tracked")
        },
        "episodes": episode_summaries,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved Stage-A Slot audit to {args.output_dir}")


if __name__ == "__main__":
    main()
