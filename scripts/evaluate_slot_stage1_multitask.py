#!/usr/bin/env python
"""Evaluate and visualize a Stage-1 Slot checkpoint across LIBERO-Mem tasks."""

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.debug_tools import denormalize_from_DINO
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import (
    OpenVLAForActionPrediction,
    OpenVLAForActionPrediction_SlotAtt,
)
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.losses.training_losses import build_bbox_matcher
from visualize_slot_stage1 import build_gt_tracks, draw_box


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--data-root", default="/data2/lty/libero-mem-rlds/full")
    parser.add_argument("--raw-data-root", default="/data2/lty/libero-mem-data/LIBERO-Mem")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--max-scan-windows", type=int, default=5000)
    parser.add_argument("--objectness-threshold", type=float, default=0.5)
    return parser.parse_args()


def load_raw_task_samples(raw_root, image_transform, horizon, num_tasks):
    """Load one deterministic midpoint window from every raw task file."""
    raw_root = Path(raw_root)
    with (raw_root / "metainfo.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    samples = []
    for path in sorted(raw_root.glob("*.hdf5"))[:num_tasks]:
        task_name = path.name[:-10]
        task_metadata = metadata[task_name]
        with h5py.File(path, "r") as handle:
            available = set(handle["data"].keys()) & set(task_metadata.keys())
            demo_name = sorted(available, key=lambda value: int(value.replace("demo_", "")))[0]
            demo = handle["data"][demo_name]
            length = int(demo["actions"].shape[0])
            start = max(0, min(length - horizon, length // 2 - horizon // 2))
            stop = start + horizon
            images = demo["obs"]["agentview_rgb"][start:stop]
            segmentation = demo["obs"]["agentview_seg"][start:stop]
        meta = task_metadata[demo_name]
        reasoning = []
        for frame in meta["exo_boxes"][start:stop]:
            parsed = {}
            for name, values in frame.items():
                seg_id, bbox, interaction = values
                bbox = np.asarray(bbox, dtype=np.float32).copy()
                if bbox.max(initial=0) > 2:
                    bbox /= 256.0
                bbox[:2] += bbox[2:] / 2
                parsed["robot" if name == "gripper" else name] = [
                    str(interaction), int(seg_id), bbox
                ]
            reasoning.append(parsed)
        samples.append(
            {
                "all_pixel_values": torch.stack(
                    [image_transform(Image.fromarray(image)) for image in images]
                ),
                "all_pixel_seg_values": torch.from_numpy(segmentation.copy()),
                "reasoning_on_image": reasoning,
                "task": {"instruction": meta["task_description"].lower()},
                "source": str(path),
                "demo": demo_name,
                "start_frame": start,
            }
        )
        print(f"Selected task {len(samples)}: {meta['task_description']} ({demo_name}, frame {start})")
    return samples


def box_iou_cxcywh(boxes1, boxes2):
    half1 = boxes1[..., 2:] / 2
    half2 = boxes2[..., 2:] / 2
    lo = torch.maximum(boxes1[..., :2] - half1, boxes2[..., :2] - half2)
    hi = torch.minimum(boxes1[..., :2] + half1, boxes2[..., :2] + half2)
    inter = (hi - lo).clamp_min(0).prod(dim=-1)
    area1 = boxes1[..., 2:].clamp_min(0).prod(dim=-1)
    area2 = boxes2[..., 2:].clamp_min(0).prod(dim=-1)
    return inter / (area1 + area2 - inter + 1e-6)


def build_gt_masks(reasoning, segmentation, object_names):
    masks = []
    for name in object_names:
        frames = []
        for frame_id, frame in enumerate(reasoning):
            if name in frame:
                frames.append(segmentation[frame_id] == int(frame[name][1]))
            else:
                frames.append(torch.zeros_like(segmentation[frame_id], dtype=torch.bool))
        masks.append(torch.stack(frames))
    return torch.stack(masks)  # [O,T,64,64]


def identity_metrics(pred_boxes, gt_tracks):
    consistencies, switches = [], []
    for object_id in range(gt_tracks.shape[0]):
        visible = (gt_tracks[object_id, :, 2:] > 0).all(dim=-1)
        chosen = []
        for frame_id in torch.where(visible)[0].tolist():
            gt = gt_tracks[object_id, frame_id, :4].expand(pred_boxes.shape[1], -1)
            ious = box_iou_cxcywh(pred_boxes[frame_id, :, :4], gt)
            chosen.append(int(ious.argmax()))
        if not chosen:
            continue
        counts = np.bincount(chosen, minlength=pred_boxes.shape[1])
        consistencies.append(float(counts.max() / len(chosen)))
        switches.append(sum(a != b for a, b in zip(chosen[:-1], chosen[1:])))
    return consistencies, switches


def evaluate_window(outputs, sample, threshold):
    pred_boxes = outputs["bboxes"][0].float().cpu()  # [T,S,5]
    pred_masks = outputs["masks"][0].float().cpu()  # [T,S,64,64]
    reasoning = sample["reasoning_on_image"]
    object_names, gt_tracks = build_gt_tracks(reasoning)

    segmentation = sample["all_pixel_seg_values"].float()[:, None]
    segmentation = F.interpolate(segmentation, size=(64, 64), mode="nearest")[:, 0].long()
    gt_masks = build_gt_masks(reasoning, segmentation, object_names)

    matcher = build_bbox_matcher(with_tracking=True)
    indices = matcher(pred_boxes[:, :, :4].permute(1, 0, 2).unsqueeze(0), [gt_tracks])[0]
    matched_slots = indices[0].tolist()
    matched_objects = indices[1].tolist()
    slot_to_object = {
        int(slot): object_names[int(obj)] for slot, obj in zip(matched_slots, matched_objects)
    }

    box_ious, mask_ious = [], []
    labels = torch.zeros(pred_boxes.shape[:2], dtype=torch.bool)  # [T,S]
    for slot_id, object_id in zip(matched_slots, matched_objects):
        visible = (gt_tracks[object_id, :, 2:] > 0).all(dim=-1)
        labels[:, slot_id] = visible
        if visible.any():
            box_ious.extend(
                box_iou_cxcywh(
                    pred_boxes[visible, slot_id, :4], gt_tracks[object_id, visible, :4]
                ).tolist()
            )
            predicted = pred_masks[visible, slot_id].sigmoid() >= 0.5
            target = gt_masks[object_id, visible]
            intersection = (predicted & target).flatten(1).sum(dim=1).float()
            union = (predicted | target).flatten(1).sum(dim=1).float()
            valid = union > 0
            mask_ious.extend((intersection[valid] / union[valid]).tolist())

    predictions = pred_boxes[:, :, 4] >= threshold
    tp = int((predictions & labels).sum())
    fp = int((predictions & ~labels).sum())
    fn = int((~predictions & labels).sum())
    tn = int((~predictions & ~labels).sum())
    consistencies, switches = identity_metrics(pred_boxes, gt_tracks)

    metrics = {
        "instruction": sample["task"]["instruction"],
        "num_objects": len(object_names),
        "mean_bbox_iou": float(np.mean(box_ious)) if box_ious else 0.0,
        "mean_mask_iou": float(np.mean(mask_ious)) if mask_ious else 0.0,
        "objectness_precision": tp / (tp + fp + 1e-8),
        "objectness_recall": tp / (tp + fn + 1e-8),
        "objectness_accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
        "identity_consistency": float(np.mean(consistencies)) if consistencies else 0.0,
        "identity_switches": int(sum(switches)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
    return pred_boxes, pred_masks, slot_to_object, metrics


def save_figure(sample, pred_boxes, pred_masks, slot_to_object, output, args):
    reasoning = sample["reasoning_on_image"]
    rgb = denormalize_from_DINO(sample["all_pixel_values"][:, :3].float()).clamp(0, 1)
    rgb = rgb.permute(0, 2, 3, 1).numpy()
    frame_ids = np.linspace(0, args.horizon - 1, args.num_frames, dtype=int)
    colors = plt.cm.tab20(np.linspace(0, 1, 16))
    figure, axes = plt.subplots(len(frame_ids), 3, figsize=(12, 4 * len(frame_ids)))
    if len(frame_ids) == 1:
        axes = axes[None, :]

    for row, frame_id in enumerate(frame_ids):
        for col in range(3):
            axes[row, col].imshow(rgb[frame_id])
            axes[row, col].set_xlim(0, 224)
            axes[row, col].set_ylim(224, 0)
            axes[row, col].axis("off")
        axes[row, 0].set_title(f"RGB frame {frame_id}")
        axes[row, 1].set_title("Ground-truth boxes")
        axes[row, 2].set_title("Predicted Slot boxes + masks")
        for name, datum in reasoning[frame_id].items():
            draw_box(axes[row, 1], datum[2], "lime", name)

        active = []
        for slot_id in range(pred_boxes.shape[1]):
            score = float(pred_boxes[frame_id, slot_id, 4])
            if score < args.objectness_threshold:
                continue
            active.append(slot_id)
            draw_box(
                axes[row, 2], pred_boxes[frame_id, slot_id, :4], colors[slot_id],
                f"s{slot_id}:{slot_to_object.get(slot_id, 'unmatched')} {score:.2f}",
            )
        if active:
            masks = F.interpolate(
                pred_masks[frame_id, active, None].sigmoid(),
                size=(224, 224), mode="bilinear", align_corners=False,
            )[:, 0]
            strength, index = masks.max(dim=0)
            overlay = np.zeros((224, 224, 4), dtype=np.float32)
            for local_id, slot_id in enumerate(active):
                region = (index == local_id) & (strength > 0.5)
                overlay[region.numpy(), :3] = colors[slot_id, :3]
                overlay[region.numpy(), 3] = 0.35
            axes[row, 2].imshow(overlay)

    figure.suptitle(f"Stage-1 multitask acceptance\n{sample['task']['instruction']}", fontsize=11)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-1 evaluation requires one CUDA GPU.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    base = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=False,
    )
    base.vision_backbone.requires_grad_(False)
    del base.projector
    del base.language_model
    model = OpenVLAForActionPrediction_SlotAtt(model=base, number_of_slots=16)
    missing, unexpected = model.load_state_dict(load_file(args.checkpoint), strict=False)
    print(f"Loaded Slot checkpoint: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device)
    model.train()  # prediction heads are only emitted in training mode

    selected = load_raw_task_samples(
        args.raw_data_root, processor.image_processor.apply_transform,
        args.horizon, args.num_tasks,
    )
    if len(selected) != args.num_tasks:
        raise RuntimeError(f"Found only {len(selected)} raw task files; expected {args.num_tasks}.")

    rows = []
    with torch.no_grad():
        for task_id, sample in enumerate(selected, start=1):
            pixels = sample["all_pixel_values"].unsqueeze(0).to(device, dtype=torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.get_obj_slots(pixels)
            pred_boxes, pred_masks, mapping, metrics = evaluate_window(
                outputs, sample, args.objectness_threshold
            )
            metrics["task_id"] = task_id
            rows.append(metrics)
            save_figure(
                sample, pred_boxes, pred_masks, mapping,
                output_dir / f"task_{task_id:02d}_slot_predictions.png", args,
            )
            (output_dir / f"task_{task_id:02d}_mapping.txt").write_text(
                "\n".join(f"slot {slot}: {mapping.get(slot, 'unmatched')}" for slot in range(16)) + "\n",
                encoding="utf-8",
            )
            print(f"Evaluated task {task_id}: {metrics}")

    totals = {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn", "tn")}
    summary = {
        "num_tasks": len(rows),
        "mean_bbox_iou": float(np.mean([r["mean_bbox_iou"] for r in rows])),
        "mean_mask_iou": float(np.mean([r["mean_mask_iou"] for r in rows])),
        "objectness_precision": totals["tp"] / (totals["tp"] + totals["fp"] + 1e-8),
        "objectness_recall": totals["tp"] / (totals["tp"] + totals["fn"] + 1e-8),
        "objectness_accuracy": (totals["tp"] + totals["tn"]) / max(1, sum(totals.values())),
        "mean_identity_consistency": float(np.mean([r["identity_consistency"] for r in rows])),
        "total_identity_switches": int(sum(r["identity_switches"] for r in rows)),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps({"summary": summary, "tasks": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("SUMMARY", json.dumps(summary, indent=2))
    print(f"Saved Stage-1 acceptance outputs to {output_dir}")


if __name__ == "__main__":
    main()
