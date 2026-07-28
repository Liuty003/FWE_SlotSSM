#!/usr/bin/env python
"""Held-out evaluation for the Failure-Aware action-conditioned slot predictor."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction, OpenVLAForActionPrediction_SlotAtt
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransformV3, RLDSDatasetV3
from scripts.train_failure_action_predictor import ActionConditionedSlotPredictor
from scripts.train_slot_stage2_ssm import Stage2Dynamics
from scripts.train_slot_stage3_openvla import extract_slots, max_memory_map, select_auxiliary_device


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial")
    p.add_argument("--data-root", default="/data2/lty/libero-mem-rlds/full")
    p.add_argument("--slot-checkpoint", required=True)
    p.add_argument("--stage2-checkpoint", required=True)
    p.add_argument("--predictor-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-windows", type=int, default=50)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--vision-microbatch", type=int, default=8)
    p.add_argument("--reserve-gib", type=float, default=2.5)
    p.add_argument("--free-fraction", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=20260720)
    return p.parse_args()


def metrics(predicted, target):
    query = F.normalize(predicted.float(), dim=-1)
    key = F.normalize(target.float(), dim=-1)
    similarity = torch.einsum("btkd,btjd->btkj", query, key)
    identity = similarity.argmax(dim=-1).eq(
        torch.arange(query.shape[2], device=query.device).view(1, 1, -1)
    ).float().mean()
    return {
        "mae": float(F.l1_loss(predicted.float(), target.float())),
        "smooth_l1": float(F.smooth_l1_loss(predicted.float(), target.float())),
        "identity_retrieval": float(identity),
    }


def summarize(rows):
    modes = rows[0].keys()
    return {
        mode: {
            metric: sum(row[mode][metric] for row in rows) / len(rows)
            for metric in rows[0][mode]
        }
        for mode in modes
    }


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    base = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=False,
        device_map="balanced_low_0", max_memory=max_memory_map(args.reserve_gib, args.free_fraction),
    )
    base.requires_grad_(False)
    vision_device = next(base.vision_backbone.parameters()).device
    auxiliary_device = select_auxiliary_device()
    slot_model = OpenVLAForActionPrediction_SlotAtt(model=base, number_of_slots=16)
    slot_model.load_state_dict(load_file(args.slot_checkpoint), strict=False)
    slot_model.requires_grad_(False)
    slot_model.object_centric_tokenizer.to(auxiliary_device, dtype=torch.bfloat16)
    slot_model.object_centric_mask_head.to(auxiliary_device, dtype=torch.float32)
    slot_model.eval()
    stage2 = Stage2Dynamics("mamba2", residual_next=True)
    stage2.load_state_dict(load_file(args.stage2_checkpoint), strict=True)
    stage2.requires_grad_(False)
    stage2.to(auxiliary_device, dtype=torch.bfloat16).eval()
    predictor = ActionConditionedSlotPredictor()
    predictor.load_state_dict(load_file(args.predictor_checkpoint), strict=True)
    predictor.requires_grad_(False)
    predictor.to(auxiliary_device, dtype=torch.bfloat16).eval()

    transform = RLDSBatchTransformV3(
        ActionTokenizer(processor.tokenizer), processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    # train=False selects the held-out final 5% of trajectories in dataset_v3.
    dataset = RLDSDatasetV3(
        args.data_root, "libero_mem", transform, resize_resolution=(224, 224),
        shuffle_buffer_size=max(args.num_windows, 64), train=False, image_aug=False,
        window_size=args.horizon, load_camera_views=("primary",), load_depth=False,
        load_proprio=True, cropping=False,
    )
    iterator = iter(dataset)
    rows = []
    with torch.inference_mode():
        for index in range(args.num_windows):
            sample = next(iterator)
            pixels = sample["all_pixel_values"].unsqueeze(0).to(dtype=torch.bfloat16)
            slots, patches = extract_slots(
                slot_model, base, pixels, vision_device, auxiliary_device, args.vision_microbatch
            )
            with torch.cuda.device(auxiliary_device), torch.autocast("cuda", dtype=torch.bfloat16):
                dynamics, stage2_next, _ = stage2(slots, patches, compute_window=False)
                current, target = slots[:, :-1], slots[:, 1:]
                actions = sample["continuous_actions"][:-1].unsqueeze(0).to(auxiliary_device, dtype=torch.bfloat16)
                proprio = sample["proprio"][:-1].unsqueeze(0).to(auxiliary_device, dtype=torch.bfloat16)
                conditioned = predictor(current, dynamics[:, :-1], stage2_next[:, :-1], actions, proprio)
                shuffled_action = predictor(
                    current, dynamics[:, :-1], stage2_next[:, :-1],
                    actions.roll(shifts=1, dims=1), proprio,
                )
                shuffled_proprio = predictor(
                    current, dynamics[:, :-1], stage2_next[:, :-1],
                    actions, proprio.roll(shifts=1, dims=1),
                )
            rows.append({
                "persistence": metrics(current, target),
                "stage2": metrics(stage2_next[:, :-1], target),
                "action_conditioned": metrics(conditioned, target),
                "shuffled_action": metrics(shuffled_action, target),
                "shuffled_proprio": metrics(shuffled_proprio, target),
            })
            print(f"evaluated={index + 1}/{args.num_windows}", flush=True)

    result = {
        "summary": summarize(rows),
        "metadata": {
            "split": "held-out final 5% trajectories", "num_windows": args.num_windows,
            "horizon": args.horizon, "predictor_checkpoint": args.predictor_checkpoint,
        },
        "per_window": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
