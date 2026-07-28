#!/usr/bin/env python
"""One-step OpenVLA LLM action-path smoke with injected Mamba2 Slot tokens."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import (
    EmbodiedSlotSSM,
    OpenVLAForActionPrediction,
    OpenVLAForActionPrediction_SlotAtt,
)
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransformV3, RLDSDatasetV3


class OpenVLASlotSSMActionPath(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.object_centric_ssm = EmbodiedSlotSSM(
            num_slots=16,
            num_blocks=2,
            d_model=512,
            d_input=None,
            visual_d_model=2176,
            textual_d_model=None,
            use_cross_attn=True,
            space_attn_num_heads=8,
            encoder_attn_num_heads=8,
            use_ffn=True,
            mamba_version="mamba2",
            attn_impl="eager",
            lookback_only=True,
        )
        self.object_centric_next_slot_head = torch.nn.Linear(1024, 512)
        self.object_centric_slot_projector = torch.nn.Sequential(
            torch.nn.LayerNorm(512),
            torch.nn.Linear(512, 4096),
        )
        self.object_centric_action_head = torch.nn.Sequential(
            torch.nn.LayerNorm(4096),
            torch.nn.Linear(4096, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 7),
        )

    def temporal_slots(self, slots, patch_features):
        ssm_device = next(self.object_centric_ssm.parameters()).device
        # Triton launches on torch.cuda.current_device(), not merely on the
        # device of its pointer arguments.  In model-parallel runs the default
        # device may remain cuda:0 while Mamba lives on another GPU.
        with torch.cuda.device(ssm_device):
            latent, _, _ = self.object_centric_ssm(slots, patch_features)
            return self.object_centric_next_slot_head(torch.cat([latent, slots], dim=-1))

    def decode_action(self, temporal_slots, input_ids, attention_mask):
        # Use the final memory state: 16 object tokens, each projected into the
        # native OpenVLA/Llama embedding dimension.
        injection_device = next(self.object_centric_slot_projector.parameters()).device
        temporal_slots = temporal_slots.to(injection_device)
        input_ids = input_ids.to(injection_device)
        attention_mask = attention_mask.to(injection_device)
        slot_embeddings = self.object_centric_slot_projector(temporal_slots[:, -1])
        text_embeddings = self.base_model.get_input_embeddings()(input_ids)
        embeddings = torch.cat(
            [text_embeddings[:, :1], slot_embeddings, text_embeddings[:, 1:]], dim=1
        )
        slot_mask = torch.ones(
            input_ids.shape[0],
            slot_embeddings.shape[1],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_mask = torch.cat(
            [attention_mask[:, :1], slot_mask, attention_mask[:, 1:]], dim=1
        )
        output = self.base_model.language_model(
            input_ids=None,
            attention_mask=full_mask,
            inputs_embeds=embeddings,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        # The final prompt token summarizes language plus injected object memory.
        action_device = next(self.object_centric_action_head.parameters()).device
        hidden = output.hidden_states[-1][:, -2].to(action_device)
        return self.object_centric_action_head(hidden)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--data-root", default="/data2/lty/libero-mem-rlds")
    parser.add_argument("--slot-checkpoint", required=True)
    parser.add_argument("--ssm-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--reserve-gib", type=float, default=3.0)
    parser.add_argument("--free-fraction", type=float, default=0.75)
    return parser.parse_args()


def build_max_memory(gpu_count, reserve_gib, free_fraction):
    max_memory = {}
    gib = 1024**3
    for index in range(min(gpu_count, torch.cuda.device_count())):
        free_bytes, _ = torch.cuda.mem_get_info(index)
        usable_bytes = min(
            int(free_bytes * free_fraction),
            max(0, free_bytes - int(reserve_gib * gib)),
        )
        usable_gib = usable_bytes // gib
        if usable_gib < 2:
            raise RuntimeError(
                f"GPU {index} has only {free_bytes / gib:.1f} GiB free; "
                f"less than the configured safe allocation."
            )
        max_memory[index] = f"{usable_gib}GiB"
    max_memory["cpu"] = "64GiB"
    print(f"Model-parallel max_memory: {max_memory}")
    return max_memory


def select_auxiliary_gpu(gpu_count):
    free_by_gpu = []
    for index in range(min(gpu_count, torch.cuda.device_count())):
        free_bytes, _ = torch.cuda.mem_get_info(index)
        free_by_gpu.append((free_bytes, index))
    free_bytes, index = max(free_by_gpu)
    device = torch.device(f"cuda:{index}")
    print(
        f"Auxiliary Slot/Mamba device: {device} "
        f"({free_bytes / 1024**3:.1f} GiB free after OpenVLA dispatch)"
    )
    return device


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("OpenVLA SlotSSM action-path smoke requires one CUDA GPU.")
    if torch.cuda.device_count() < args.gpu_count:
        raise RuntimeError(
            f"Requested {args.gpu_count} GPUs, but only {torch.cuda.device_count()} are visible."
        )
    max_memory = build_max_memory(
        args.gpu_count, args.reserve_gib, args.free_fraction
    )

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
        max_memory=max_memory,
    )
    base.requires_grad_(False)
    print(f"OpenVLA device map: {base.hf_device_map}")
    vision_device = next(base.vision_backbone.parameters()).device
    embedding_device = next(base.get_input_embeddings().parameters()).device
    try:
        final_lm_device = next(base.language_model.model.layers[-1].parameters()).device
    except AttributeError:
        final_lm_device = embedding_device
    print(
        f"vision={vision_device}, token_embedding={embedding_device}, "
        f"final_llm={final_lm_device}"
    )
    auxiliary_device = select_auxiliary_gpu(args.gpu_count)

    slot_model = OpenVLAForActionPrediction_SlotAtt(model=base, number_of_slots=16)
    slot_model.load_state_dict(load_file(args.slot_checkpoint), strict=False)
    slot_model.object_centric_tokenizer.requires_grad_(False)
    slot_model.object_centric_bbox_head.requires_grad_(False)
    slot_model.object_centric_mask_head.requires_grad_(False)

    model = OpenVLASlotSSMActionPath(base)
    missing, unexpected = model.load_state_dict(
        load_file(args.ssm_checkpoint), strict=False
    )
    print(f"Loaded Stage-2 Mamba2: missing={len(missing)} unexpected={len(unexpected)}")
    model.object_centric_ssm.requires_grad_(False)
    model.object_centric_next_slot_head.requires_grad_(False)
    model.object_centric_ssm.to(auxiliary_device, dtype=torch.bfloat16)
    model.object_centric_next_slot_head.to(auxiliary_device, dtype=torch.bfloat16)
    model.object_centric_slot_projector.to(embedding_device, dtype=torch.bfloat16)
    model.object_centric_action_head.to(final_lm_device, dtype=torch.bfloat16)
    slot_model.object_centric_tokenizer.to(auxiliary_device)
    slot_model.object_centric_bbox_head.to(auxiliary_device)
    slot_model.object_centric_mask_head.to(auxiliary_device)
    slot_model.train()

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
        image_aug=False,
        window_size=args.horizon,
        load_camera_views=("primary", "wrist"),
        load_depth=True,
        cropping=False,
    )
    sample = next(iter(dataset))
    pixels = sample["all_pixel_values"].unsqueeze(0).to(vision_device, dtype=torch.bfloat16)
    target_action = sample["continuous_actions"][-1].unsqueeze(0).to(final_lm_device)
    input_ids = sample["input_ids"].unsqueeze(0).to(embedding_device)
    attention_mask = input_ids.ne(processor.tokenizer.pad_token_id)

    slots, patches = [], []
    past = None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for timestep in range(args.horizon):
            patch = base.vision_backbone(pixels[:, timestep]).reshape(1, -1, 2176)
            patch = patch.to(auxiliary_device)
            current, _, _ = slot_model.object_centric_tokenizer(patch, past_slots=past)
            past = current[:, None].detach()
            slots.append(current)
            patches.append(patch)
        slots = torch.stack(slots, dim=1)
        patches = torch.stack(patches, dim=1)
        if slots.device.type != "cuda" or patches.device.type != "cuda":
            raise RuntimeError(
                f"Mamba2 inputs must be CUDA tensors, got slots={slots.device}, "
                f"patches={patches.device}"
            )
        ssm_devices = {parameter.device for parameter in model.object_centric_ssm.parameters()}
        if ssm_devices != {auxiliary_device}:
            raise RuntimeError(f"Mamba2 parameters span unexpected devices: {ssm_devices}")
        ssm_buffer_devices = {
            buffer.device
            for buffer in model.object_centric_ssm.buffers()
            if buffer.numel() > 0
        }
        if any(device.type != "cuda" or device != auxiliary_device for device in ssm_buffer_devices):
            raise RuntimeError(
                f"Mamba2 buffers span unexpected devices: {ssm_buffer_devices}"
            )
        print(
            f"Stage-2 devices verified: slots={slots.device}, patches={patches.device}, "
            f"Mamba2={next(iter(ssm_devices))}, buffers={ssm_buffer_devices or 'none'}, "
            f"current_cuda=cuda:{torch.cuda.current_device()}"
        )
        temporal_slots = model.temporal_slots(slots, patches)

    trainable = list(model.object_centric_slot_projector.parameters())
    trainable += list(model.object_centric_action_head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        predicted_action = model.decode_action(temporal_slots, input_ids, attention_mask)
        action_loss = F.l1_loss(predicted_action.float(), target_action.float())
    action_loss.backward()
    optimizer.step()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        zero_slot_action = model.decode_action(
            torch.zeros_like(temporal_slots), input_ids, attention_mask
        )
    slot_effect = (predicted_action.detach() - zero_slot_action).abs().mean()
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in trainable
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_state = {}
    for prefix, module in (
        ("object_centric_slot_projector", model.object_centric_slot_projector),
        ("object_centric_action_head", model.object_centric_action_head),
    ):
        for key, value in module.state_dict().items():
            output_state[f"{prefix}.{key}"] = value.detach().cpu()
    save_file(output_state, str(output))
    print(f"OpenVLA SlotSSM action L1 loss: {float(action_loss.detach()):.6f}")
    print(f"Predicted action shape: {tuple(predicted_action.shape)}")
    print(f"Mean action change when SlotSSM tokens are zeroed: {float(slot_effect):.6f}")
    print(f"Injection/action gradients finite: {gradients_finite}")
    print(f"Saved OpenVLA SlotSSM action-path checkpoint to {output}")


if __name__ == "__main__":
    main()
