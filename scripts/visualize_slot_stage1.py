#!/usr/bin/env python
"""Visualize Stage-1 Slot predictions against LIBERO-Mem annotations."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from safetensors.torch import load_file
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.debug_tools import denormalize_from_DINO
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import (
    OpenVLAForActionPrediction,
    OpenVLAForActionPrediction_SlotAtt,
)
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from prismatic.losses.training_losses import build_bbox_matcher
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransformV3, RLDSDatasetV3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial",
    )
    parser.add_argument("--data-root", default="/data2/lty/libero-mem-rlds")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output-dir",
        default="/data2/lty/naive-e-slotssm-runs/stage1-visualization",
    )
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--objectness-threshold", type=float, default=0.5)
    return parser.parse_args()


def build_gt_tracks(reasoning):
    names = []
    for frame in reasoning:
        for name in frame:
            if name not in names:
                names.append(name)

    tracks = []
    for name in names:
        boxes = []
        for frame in reasoning:
            if name in frame:
                boxes.append(np.asarray(frame[name][2], dtype=np.float32))
            else:
                boxes.append(np.zeros(4, dtype=np.float32))
        tracks.append(np.stack(boxes))
    return names, torch.from_numpy(np.stack(tracks))


def draw_box(ax, bbox, color, label, linewidth=1.5):
    cx, cy, width, height = [float(x) for x in bbox]
    if width <= 0 or height <= 0:
        return
    x0, y0 = (cx - width / 2) * 224, (cy - height / 2) * 224
    rect = Rectangle(
        (x0, y0),
        width * 224,
        height * 224,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
    )
    ax.add_patch(rect)
    ax.text(
        x0,
        max(0, y0 - 2),
        label,
        color="white",
        fontsize=6,
        bbox={"facecolor": color, "alpha": 0.75, "pad": 1, "edgecolor": "none"},
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This visualization requires one CUDA GPU.")

    device = torch.device("cuda:0")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    base_model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    base_model.vision_backbone.requires_grad_(False)
    del base_model.projector
    del base_model.language_model

    model = OpenVLAForActionPrediction_SlotAtt(model=base_model, number_of_slots=16)
    missing, unexpected = model.load_state_dict(load_file(args.checkpoint), strict=False)
    print(f"Loaded Slot checkpoint: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device)
    # The current Slot wrapper emits prediction heads only in training mode.
    # no_grad below prevents any parameter updates.
    model.train()

    transform = RLDSBatchTransformV3(
        ActionTokenizer(processor.tokenizer),
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    dataset = RLDSDatasetV3(
        args.data_root,
        "libero_mem",
        transform,
        resize_resolution=(224, 224),
        shuffle_buffer_size=1,
        train=False,
        image_aug=False,
        window_size=args.horizon,
        load_camera_views=("primary", "wrist"),
        load_depth=True,
        cropping=False,
    )
    sample = next(iter(dataset))
    pixels = sample["all_pixel_values"].unsqueeze(0).to(device, dtype=torch.bfloat16)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.get_obj_slots(pixels)

    pred_boxes = outputs["bboxes"][0].float().cpu()  # [T, S, 5]
    pred_masks = outputs["masks"][0].float().cpu()   # [T, S, 64, 64]
    reasoning = sample["reasoning_on_image"]
    object_names, gt_tracks = build_gt_tracks(reasoning)

    matcher = build_bbox_matcher(with_tracking=True)
    indices = matcher(
        pred_boxes[:, :, :4].permute(1, 0, 2).unsqueeze(0),
        [gt_tracks],
    )[0]
    slot_to_object = {
        int(slot): object_names[int(obj)]
        for slot, obj in zip(indices[0].tolist(), indices[1].tolist())
    }

    rgb = denormalize_from_DINO(
        sample["all_pixel_values"][:, :3].float()
    ).clamp(0, 1)
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
        axes[row, 1].set_title("Ground truth boxes")
        axes[row, 2].set_title("Predicted Slot boxes + masks")

        for name, datum in reasoning[frame_id].items():
            draw_box(axes[row, 1], datum[2], "lime", name)

        active_slots = []
        for slot_id in range(pred_boxes.shape[1]):
            score = float(pred_boxes[frame_id, slot_id, 4])
            if score < args.objectness_threshold:
                continue
            active_slots.append(slot_id)
            color = colors[slot_id]
            label = slot_to_object.get(slot_id, "unmatched")
            draw_box(
                axes[row, 2],
                pred_boxes[frame_id, slot_id, :4],
                color,
                f"s{slot_id}:{label} {score:.2f}",
            )

        if active_slots:
            masks = torch.sigmoid(pred_masks[frame_id, active_slots])
            masks = F.interpolate(
                masks[:, None], size=(224, 224), mode="bilinear", align_corners=False
            )[:, 0]
            best_strength, best_index = masks.max(dim=0)
            overlay = np.zeros((224, 224, 4), dtype=np.float32)
            for local_id, slot_id in enumerate(active_slots):
                region = (best_index == local_id) & (best_strength > 0.5)
                overlay[region.numpy(), :3] = colors[slot_id, :3]
                overlay[region.numpy(), 3] = 0.35
            axes[row, 2].imshow(overlay)

    figure.suptitle(
        "Stage-1 object Slot visualization\n"
        f"green=ground truth, colored=matched predicted Slot; instruction: "
        f"{sample['task']['instruction']}",
        fontsize=11,
    )
    figure.tight_layout()
    output_path = output_dir / "slot_predictions.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    mapping_path = output_dir / "slot_track_mapping.txt"
    mapping_path.write_text(
        "\n".join(
            [f"slot {slot_id}: {slot_to_object.get(slot_id, 'unmatched')}" for slot_id in range(16)]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved visualization to {output_path}")
    print(f"Saved Slot/object mapping to {mapping_path}")


if __name__ == "__main__":
    main()
