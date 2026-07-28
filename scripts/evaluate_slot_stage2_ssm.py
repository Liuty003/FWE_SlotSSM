#!/usr/bin/env python
"""Post-training diagnostics for Stage-2 temporal dynamics."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import (
    OpenVLAForActionPrediction,
    OpenVLAForActionPrediction_SlotAtt,
)
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransformV3, RLDSDatasetV3
from scripts.train_slot_stage2_ssm import Stage2Dynamics, window_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial")
    p.add_argument("--data-root", default="/data2/lty/libero-mem-rlds/full")
    p.add_argument("--slot-checkpoint", required=True)
    p.add_argument("--stage2-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-windows", type=int, default=20)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--vision-microbatch", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260720)
    p.add_argument("--residual-next", action="store_true")
    return p.parse_args()


def next_metrics(prediction, slots):
    pred = prediction[:, :-1].float()
    target = slots[:, 1:].float()
    loss = F.smooth_l1_loss(pred, target)
    cosine = F.cosine_similarity(pred, target, dim=-1).mean()
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    logits = torch.einsum("btkd,btjd->btkj", pred_n, target_n)
    labels = torch.arange(pred.shape[2], device=pred.device).view(1, 1, -1)
    identity_accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    return {
        "next_smooth_l1": float(loss),
        "next_cosine": float(cosine),
        "identity_retrieval_accuracy": float(identity_accuracy),
    }


def persistence_window_loss(slots):
    total = slots.new_zeros((), dtype=torch.float32)
    count = 0
    timesteps = slots.shape[1]
    for offset in Stage2Dynamics.OFFSETS:
        start = max(0, -offset)
        end = min(timesteps, timesteps - offset)
        if end <= start:
            continue
        total += F.smooth_l1_loss(slots[:, start:end].float(), slots[:, start + offset:end + offset].float())
        count += 1
    return float(total / max(count, 1))


def per_offset_losses(predicted_window, slots):
    results = {}
    timesteps = slots.shape[1]
    for index, offset in enumerate(Stage2Dynamics.OFFSETS):
        start = max(0, -offset)
        end = min(timesteps, timesteps - offset)
        pred = predicted_window[:, start:end, :, index].float()
        target = slots[:, start + offset:end + offset].float()
        results[f"offset_{offset:+03d}"] = float(F.smooth_l1_loss(pred, target))
    return results


def persistence_per_offset_losses(slots):
    results = {}
    timesteps = slots.shape[1]
    for offset in Stage2Dynamics.OFFSETS:
        start = max(0, -offset)
        end = min(timesteps, timesteps - offset)
        results[f"offset_{offset:+03d}"] = float(
            F.smooth_l1_loss(slots[:, start:end].float(), slots[:, start + offset:end + offset].float())
        )
    return results


def extract_slots(slot_model, pixels, microbatch):
    b, t = pixels.shape[:2]
    flat = pixels.flatten(0, 1)
    patch_chunks = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, flat.shape[0], microbatch):
            encoded = slot_model.model.vision_backbone(flat[start:start + microbatch])
            patch_chunks.append(encoded.reshape(encoded.shape[0], -1, 2176))
        patches = torch.cat(patch_chunks).reshape(b, t, -1, 2176)
        slots, past = [], None
        for timestep in range(t):
            current, _, _ = slot_model.object_centric_tokenizer(patches[:, timestep], past_slots=past)
            past = current[:, None]
            slots.append(current)
    # stack/clone outside inference_mode so trainable modules may consume these tensors.
    return torch.stack(slots, dim=1), patches.clone()


def average(records):
    keys = records[0].keys()
    return {key: sum(row[key] for row in records) / len(records) for key in keys}


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    base = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=False
    )
    base.vision_backbone.requires_grad_(False)
    del base.projector
    del base.language_model
    slot_model = OpenVLAForActionPrediction_SlotAtt(model=base, number_of_slots=16)
    slot_model.load_state_dict(load_file(args.slot_checkpoint), strict=False)
    slot_model.requires_grad_(False)
    slot_model = slot_model.to(device, dtype=torch.bfloat16).eval()

    transform = RLDSBatchTransformV3(
        ActionTokenizer(processor.tokenizer), processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    dataset = RLDSDatasetV3(
        args.data_root, "libero_mem", transform, resize_resolution=(224, 224),
        shuffle_buffer_size=1, image_aug=False, window_size=args.horizon,
        load_camera_views=("primary", "wrist"), load_depth=True, cropping=False,
    )
    iterator = iter(dataset)

    # Seed first, then instantiate both architectures; only trained receives the checkpoint.
    torch.manual_seed(args.seed)
    random_model = Stage2Dynamics("mamba2").to(device, dtype=torch.bfloat16).eval()
    torch.manual_seed(args.seed)
    trained_model = Stage2Dynamics("mamba2", residual_next=args.residual_next).to(device, dtype=torch.bfloat16).eval()
    incompatible = trained_model.load_state_dict(load_file(args.stage2_checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Stage-2 load mismatch: {incompatible}")

    records = {"persistence": [], "random": [], "trained": []}
    with torch.inference_mode():
        for index in range(args.num_windows):
            sample = next(iterator)
            pixels = sample["all_pixel_values"].unsqueeze(0).to(device, dtype=torch.bfloat16)
            slots, patches = extract_slots(slot_model, pixels, args.vision_microbatch)
            persistence = next_metrics(slots, slots)
            persistence["window_smooth_l1"] = persistence_window_loss(slots)
            persistence.update(persistence_per_offset_losses(slots))
            records["persistence"].append(persistence)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, random_next, random_window = random_model(slots, patches)
                _, trained_next, trained_window = trained_model(slots, patches)
            random_metrics = next_metrics(random_next, slots)
            random_metrics["window_smooth_l1"] = float(window_loss(random_window, slots))
            random_metrics.update(per_offset_losses(random_window, slots))
            trained_metrics = next_metrics(trained_next, slots)
            trained_metrics["window_smooth_l1"] = float(window_loss(trained_window, slots))
            trained_metrics.update(per_offset_losses(trained_window, slots))
            records["random"].append(random_metrics)
            records["trained"].append(trained_metrics)
            print(f"evaluated window {index + 1}/{args.num_windows}", flush=True)

    summary = {name: average(rows) for name, rows in records.items()}
    summary["metadata"] = {
        "num_windows": args.num_windows,
        "horizon": args.horizon,
        "strict_held_out": False,
        "note": "Post-training deterministic diagnostic on the training dataset; not a predeclared held-out split.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "per_window": records}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved diagnostics to {output}")


if __name__ == "__main__":
    main()
