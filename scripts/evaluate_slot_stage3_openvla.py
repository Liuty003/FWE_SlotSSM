#!/usr/bin/env python
"""Offline action evaluation and ablations for the formal Stage-3 path."""

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
from scripts.train_slot_stage2_ssm import Stage2Dynamics
from scripts.train_slot_stage3_openvla import (
    Stage3ActionPath,
    encode_subgoals,
    extract_slots,
    max_memory_map,
    object_subgoal_strings,
    select_auxiliary_device,
)


MODES = (
    "full",
    "no_next_slot",
    "zero_subgoal",
    "zero_dynamics_token",
    "zero_relation_token",
    "zero_all_object_tokens",
)


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial")
    p.add_argument("--data-root", default="/data2/lty/libero-mem-rlds/full")
    p.add_argument("--slot-checkpoint", required=True)
    p.add_argument("--stage2-checkpoint", required=True)
    p.add_argument("--stage3-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-windows", type=int, default=20)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--action-chunk", type=int, default=16)
    p.add_argument("--vision-microbatch", type=int, default=8)
    p.add_argument("--reserve-gib", type=float, default=2.5)
    p.add_argument("--free-fraction", type=float, default=0.80)
    p.add_argument("--seed", type=int, default=20260720)
    return p.parse_args()


def load_stage3_only(model, checkpoint):
    state = load_file(checkpoint)
    own = model.state_dict()
    unknown = sorted(set(state) - set(own))
    if unknown:
        raise RuntimeError(f"Unknown Stage-3 keys: {unknown}")
    own.update(state)
    model.load_state_dict(own, strict=True)
    return len(state), sum(value.numel() for value in state.values())


def build_object_tokens(model, current, predicted_next, patches, subgoal_llm, mode):
    next_slots = current if mode == "no_next_slot" else predicted_next
    encoded_subgoal = model.subgoal_projection(subgoal_llm)
    if mode == "zero_subgoal":
        encoded_subgoal = torch.zeros_like(encoded_subgoal)
    fused = model.slot_fusion([current, next_slots, encoded_subgoal])
    fused = model.fusion_norm(model.fusion_residual(current) + fused)
    relation = model.relation_encoder(fused, patches)
    dynamics_token = model.dynamic_projector(fused) + model.object_type_embedding[0]
    relation_token = model.relation_projector(relation) + model.object_type_embedding[1]
    if mode in ("zero_dynamics_token", "zero_all_object_tokens"):
        dynamics_token = torch.zeros_like(dynamics_token)
    if mode in ("zero_relation_token", "zero_all_object_tokens"):
        relation_token = torch.zeros_like(relation_token)
    return relation_token, dynamics_token


def decode_action(model, base, embedding, tokenizer, instruction, relation, dynamics, embedding_device, final_device):
    prompt = f"What action should the robot take to {instruction}?"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(embedding_device)
    prompt_embeddings = embedding(ids)
    injected = torch.cat(
        (prompt_embeddings, relation.to(embedding_device), dynamics.to(embedding_device), model.action_queries), dim=1
    )
    attention_mask = torch.ones(injected.shape[:2], dtype=torch.long, device=embedding_device)
    output = base.language_model(
        input_ids=None, inputs_embeds=injected, attention_mask=attention_mask,
        output_hidden_states=True, use_cache=False, return_dict=True,
    )
    hidden = output.hidden_states[-1][:, -model.action_chunk:].to(final_device)
    return model.action_head(hidden).float()


def summarize(rows):
    return {
        mode: {
            "action_l1": sum(row[mode]["action_l1"] for row in rows) / len(rows),
            "mean_action_change_from_full": sum(
                row[mode]["mean_action_change_from_full"] for row in rows
            ) / len(rows),
        }
        for mode in MODES
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
    embedding = base.get_input_embeddings()
    embedding_device = next(embedding.parameters()).device
    final_device = next(base.language_model.model.layers[-1].parameters()).device
    auxiliary_device = select_auxiliary_device()
    print(f"vision={vision_device} embedding={embedding_device} final={final_device} auxiliary={auxiliary_device}")

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

    model = Stage3ActionPath(base, args.action_chunk)
    keys, params = load_stage3_only(model, args.stage3_checkpoint)
    model.requires_grad_(False)
    for module in (
        model.subgoal_projection, model.slot_fusion, model.fusion_residual, model.fusion_norm,
        model.relation_encoder, model.dynamic_projector, model.relation_projector,
    ):
        module.to(auxiliary_device, dtype=torch.bfloat16)
    model.object_type_embedding = torch.nn.Parameter(
        model.object_type_embedding.to(auxiliary_device, dtype=torch.bfloat16), requires_grad=False
    )
    model.action_queries = torch.nn.Parameter(
        model.action_queries.to(embedding_device, dtype=torch.bfloat16), requires_grad=False
    )
    model.action_head.to(final_device, dtype=torch.bfloat16)
    model.eval()
    print(f"Loaded Stage-3: keys={keys} params={params}")

    transform = RLDSBatchTransformV3(
        ActionTokenizer(processor.tokenizer), processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    dataset = RLDSDatasetV3(
        args.data_root, "libero_mem", transform, resize_resolution=(224, 224),
        shuffle_buffer_size=1, image_aug=False,
        window_size=args.horizon + args.action_chunk,
        load_camera_views=("primary", "wrist"), load_depth=True, cropping=False,
    )
    iterator = iter(dataset)
    rows = []
    with torch.inference_mode():
        for window_index in range(args.num_windows):
            sample = next(iterator)
            pixels = sample["all_pixel_values"][:args.horizon].unsqueeze(0).to(dtype=torch.bfloat16)
            slots, patches = extract_slots(
                slot_model, base, pixels, vision_device, auxiliary_device, args.vision_microbatch
            )
            with torch.cuda.device(auxiliary_device), torch.autocast("cuda", dtype=torch.bfloat16):
                _, next_slots, _ = stage2(slots, patches, compute_window=False)
            strings = object_subgoal_strings(
                sample, slot_model, slots[:, -1], auxiliary_device, time_index=args.horizon - 1
            )
            subgoal_llm = encode_subgoals(
                strings, processor.tokenizer, embedding, embedding_device, auxiliary_device
            )
            current = slots[:, -1]
            predicted_next = next_slots[:, -1]
            final_patches = patches[:, -1]
            target = sample["continuous_actions"][
                args.horizon:args.horizon + args.action_chunk
            ].unsqueeze(0).to(final_device).float()
            predictions = {}
            for mode in MODES:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    relation, dynamics = build_object_tokens(
                        model, current, predicted_next, final_patches, subgoal_llm, mode
                    )
                    predictions[mode] = decode_action(
                        model, base, embedding, processor.tokenizer, sample["task"]["instruction"],
                        relation, dynamics, embedding_device, final_device,
                    )
            full = predictions["full"]
            row = {}
            for mode, prediction in predictions.items():
                row[mode] = {
                    "action_l1": float(F.l1_loss(prediction, target)),
                    "mean_action_change_from_full": float((prediction - full).abs().mean()),
                }
            rows.append(row)
            print(f"evaluated window {window_index + 1}/{args.num_windows}", flush=True)

    result = {
        "summary": summarize(rows),
        "metadata": {
            "num_windows": args.num_windows,
            "context_horizon": args.horizon,
            "action_chunk": args.action_chunk,
            "dataset_window": args.horizon + args.action_chunk,
            "strict_held_out": False,
            "note": "Post-training deterministic diagnostic on LIBERO-Mem; ablations share each exact window.",
        },
        "per_window": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved Stage-3 evaluation to {output}")


if __name__ == "__main__":
    main()
