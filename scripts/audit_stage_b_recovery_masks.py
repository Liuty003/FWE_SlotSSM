#!/usr/bin/env python
"""Replay a recovery rollout and audit Stage-B target-mask temporal identity."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import (
    AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = REPO_ROOT / "experiments" / "robot" / "libero"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LIBERO_ROOT))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_ROOT / "baseline_libero_config"))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor, PrismaticProcessor,
)
from scripts.stage_b_spatial_selector import RELATION_ORDER, StageBSpatialSelector
from scripts.train_slot_stage3_openvla import max_memory_map


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--patch-dataset", type=Path)
    parser.add_argument(
        "--model", default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial"
    )
    parser.add_argument(
        "--selector-config", type=Path,
        default=Path("experiments/robot/libero/stage_b_selector_production.json"),
    )
    parser.add_argument(
        "--language-features", type=Path,
        default=Path(
            "/data2/lty/naive-e-slotssm-runs/stage-b-spatial-counterfactual-patches/"
            "siglip_language_features.safetensors"
        ),
    )
    parser.add_argument("--vision-microbatch", type=int, default=8)
    parser.add_argument("--selector-batch", type=int, default=32)
    parser.add_argument("--visual-lift-threshold-px", type=float, default=30.0)
    parser.add_argument("--visual-drop-return-px", type=float, default=16.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-dataset", type=Path)
    return parser.parse_args()


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def segmentation_key(obs):
    keys = [
        key for key in obs
        if key.startswith("agentview_segmentation") and "eye_in_hand" not in key
    ]
    if len(keys) != 1:
        raise RuntimeError(f"expected one agentview segmentation, got {keys}")
    return keys[0]


def centroid(mask):
    points = mask.bool().nonzero().float()
    if not len(points):
        return None
    return [float(points[:, 1].mean()), float(points[:, 0].mean())]


def iou(first, second):
    first, second = first.bool(), second.bool()
    return float((first & second).sum() / (first | second).sum().clamp_min(1))


def load_vision(args):
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=False, local_files_only=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=False, local_files_only=True,
        device_map="balanced_low_0", max_memory=max_memory_map(2.0, 0.75),
    ).eval()
    model.requires_grad_(False)
    return processor, model, next(model.vision_backbone.parameters()).device


@torch.inference_mode()
def encode_images(images, processor, model, device, microbatch):
    pixels = torch.stack([
        processor.image_processor.apply_transform(Image.fromarray(image))
        for image in images
    ])
    patches = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(pixels), microbatch):
            encoded = model.vision_backbone(pixels[start:start + microbatch].to(device))
            patches.append(encoded.reshape(encoded.shape[0], -1, 2176).cpu())
    return torch.cat(patches).to(torch.bfloat16)


def tracking_index(candidates, previous_mask, previous_centroid):
    overlaps = [iou(candidate, previous_mask) for candidate in candidates]
    if max(overlaps) > 0:
        return int(np.argmax(overlaps)), "mask_iou"
    centers = [centroid(candidate) for candidate in candidates]
    distances = [
        np.linalg.norm(np.asarray(center) - np.asarray(previous_centroid))
        if center is not None else float("inf")
        for center in centers
    ]
    return int(np.argmin(distances)), "centroid"


def detector_step(rows, key, lift_threshold, return_threshold):
    initial = rows[0][key]
    peak = 0.0
    for row in rows:
        value = row[key]
        if value is None:
            continue
        rise = initial - value
        peak = max(peak, rise)
        if (
            row["timestep"] >= row["mistake_end"]
            and peak >= lift_threshold and rise <= return_threshold
        ):
            return row["timestep"]
    return None


def main():
    args = arguments()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv

    metadata = json.loads((args.source / "metadata.json").read_text())
    action_rows = load_rows(args.source / "steps.jsonl")
    suite = benchmark.get_benchmark_dict()[metadata["task_suite"]]()
    task_id, episode_id = int(metadata["task_id"]), int(metadata["episode_id"])
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if args.patch_dataset is not None:
        cached = load_file(args.patch_dataset)
        patches = cached["patches"]
        gt_targets = cached["target_masks"].bool()
        gt_distractors = cached["distractor_masks"].bool()
    else:
        env = SegmentationRenderEnv(
            bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
        )
        env.seed(0)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(task_id)[episode_id])
        for _ in range(int(metadata.get("wait_steps", 20))):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        target_id = env.instance_to_id["akita_black_bowl_1"]
        distractor_id = env.instance_to_id["akita_black_bowl_2"]
        images, gt_targets, gt_distractors = [], [], []
        try:
            for row in action_rows:
                images.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                segmentation = np.asarray(obs[segmentation_key(obs)]).squeeze()[::-1, ::-1]
                gt_targets.append(torch.from_numpy(segmentation == target_id))
                gt_distractors.append(torch.from_numpy(segmentation == distractor_id))
                obs, _, _, _ = env.step(row["executed_action"])
        finally:
            env.close()
        gt_targets = F.interpolate(
            torch.stack(gt_targets)[:, None].float(), (64, 64), mode="nearest"
        )[:, 0].bool()
        gt_distractors = F.interpolate(
            torch.stack(gt_distractors)[:, None].float(), (64, 64), mode="nearest"
        )[:, 0].bool()
        processor, vision, vision_device = load_vision(args)
        patches = encode_images(
            images, processor, vision, vision_device, args.vision_microbatch
        )
        if args.save_dataset is not None:
            args.save_dataset.parent.mkdir(parents=True, exist_ok=True)
            save_file({
                "patches": patches.contiguous(),
                "target_masks": gt_targets.to(torch.uint8).contiguous(),
                "distractor_masks": gt_distractors.to(torch.uint8).contiguous(),
                "timesteps": torch.arange(len(patches), dtype=torch.int64),
            }, args.save_dataset)
        del vision, processor
        torch.cuda.empty_cache()

    device = torch.device("cuda:0")
    selector = StageBSpatialSelector.from_config(args.selector_config, device)
    language = load_file(args.language_features)
    task_index = language["task_ids"].tolist().index(task_id)
    relation_id = RELATION_ORDER.index("on")
    presence = torch.tensor([True, True, True, True, False, True])
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(patches), args.selector_batch):
            count = len(patches[start:start + args.selector_batch])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = selector(
                    patches[start:start + count].to(device),
                    torch.full((count,), relation_id, dtype=torch.long, device=device),
                    presence[None].expand(count, -1).to(device),
                    language["instruction_features"][task_index][None].expand(
                        count, -1
                    ).to(device),
                    language["relation_features"][task_index][None].expand(
                        count, -1
                    ).to(device),
                    language["reference_features"][task_index][None].expand(
                        count, -1, -1
                    ).to(device),
                )
            outputs.append({
                key: result[key].cpu()
                for key in ("target_masks", "candidate_masks", "selected_indices",
                            "selection_margin")
            })
    predicted = torch.cat([item["target_masks"] for item in outputs])
    candidates = torch.cat([item["candidate_masks"] for item in outputs])
    selected = torch.cat([item["selected_indices"] for item in outputs])
    margins = torch.cat([item["selection_margin"] for item in outputs])

    tracked_indices, tracking_modes = [], []
    previous_mask, previous_center = predicted[0], centroid(predicted[0])
    tracked_indices.append(int(selected[0]))
    tracking_modes.append("initial_grounding")
    for frame in range(1, len(candidates)):
        index, mode = tracking_index(
            candidates[frame], previous_mask, previous_center
        )
        tracked_indices.append(index)
        tracking_modes.append(mode)
        previous_mask = candidates[frame, index]
        previous_center = centroid(previous_mask)
    tracked = torch.stack([
        candidates[frame, index] for frame, index in enumerate(tracked_indices)
    ])

    mistake_end = int(metadata["mistake_start"]) + int(metadata["mistake_steps"])
    records = []
    for frame, action_row in enumerate(action_rows):
        stateless_center, tracked_center = centroid(predicted[frame]), centroid(tracked[frame])
        records.append({
            "timestep": frame, "phase": action_row["phase"],
            "mistake_end": mistake_end,
            "stateless_selected_index": int(selected[frame]),
            "tracked_index": tracked_indices[frame],
            "tracking_mode": tracking_modes[frame],
            "selection_margin": float(margins[frame]),
            "stateless_target_iou": iou(predicted[frame], gt_targets[frame]),
            "stateless_distractor_iou": iou(predicted[frame], gt_distractors[frame]),
            "tracked_target_iou": iou(tracked[frame], gt_targets[frame]),
            "tracked_distractor_iou": iou(tracked[frame], gt_distractors[frame]),
            "gt_row": centroid(gt_targets[frame])[1],
            "stateless_row": stateless_center[1] if stateless_center else None,
            "tracked_row": tracked_center[1] if tracked_center else None,
        })
    stateless_step = detector_step(
        records, "stateless_row",
        args.visual_lift_threshold_px / 4, args.visual_drop_return_px / 4,
    )
    tracked_step = detector_step(
        records, "tracked_row",
        args.visual_lift_threshold_px / 4, args.visual_drop_return_px / 4,
    )
    gt_step = detector_step(
        records, "gt_row",
        args.visual_lift_threshold_px / 4, args.visual_drop_return_px / 4,
    )
    report = {
        "schema_version": 1,
        "source": str(args.source),
        "num_frames": len(records),
        "detector_steps": {
            "ground_truth_mask": gt_step,
            "stage_b_stateless": stateless_step,
            "stage_b_temporal_tracking": tracked_step,
        },
        "mean_iou": {
            key: float(np.mean([row[key] for row in records]))
            for key in (
                "stateless_target_iou", "stateless_distractor_iou",
                "tracked_target_iou", "tracked_distractor_iou",
            )
        },
        "keyframes": [
            row for row in records
            if row["timestep"] in (0, 57, 64, 67, 85, 105, 130, 173)
        ],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "source", "num_frames", "detector_steps", "mean_iou", "keyframes"
    )}, indent=2))


if __name__ == "__main__":
    main()
