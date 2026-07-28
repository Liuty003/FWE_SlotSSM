#!/usr/bin/env python
"""One-step Stage-2 Slot temporal-dynamics smoke test."""

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--data-root", default="/data2/lty/libero-mem-rlds")
    parser.add_argument("--slot-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--ssm-backend", choices=("reference", "mamba2"), default="mamba2")
    return parser.parse_args()


class SlotDynamicsSmoke(torch.nn.Module):
    def __init__(self, ssm_backend):
        super().__init__()
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
            mamba_version=ssm_backend,
            attn_impl="eager",
            lookback_only=True,
        )
        self.object_centric_next_slot_head = torch.nn.Linear(1024, 512)

    def forward(self, slots, patch_features):
        latent, _, _ = self.object_centric_ssm(slots, patch_features)
        return self.object_centric_next_slot_head(torch.cat([latent, slots], dim=-1))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2 smoke test requires one CUDA GPU.")
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
    slot_model = slot_model.to(device).train()

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
    pixels = sample["all_pixel_values"].unsqueeze(0).to(device, dtype=torch.bfloat16)

    slots = []
    patches = []
    past = None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for timestep in range(args.horizon):
            patch = slot_model.model.vision_backbone(pixels[:, timestep]).reshape(1, -1, 2176)
            current, _, _ = slot_model.object_centric_tokenizer(patch, past_slots=past)
            past = current[:, None].detach()
            slots.append(current)
            patches.append(patch)
    slots = torch.stack(slots, dim=1)
    patches = torch.stack(patches, dim=1)

    dynamics = SlotDynamicsSmoke(args.ssm_backend).to(device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(dynamics.parameters(), lr=1e-4)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        predicted_next = dynamics(slots[:, :-1], patches[:, :-1])
        loss = F.mse_loss(predicted_next.float(), slots[:, 1:].float())
    loss.backward()
    optimizer.step()

    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in dynamics.parameters()
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.detach().cpu() for key, value in dynamics.state_dict().items()},
        str(output),
    )
    print(f"Stage-2 {args.ssm_backend} SSM loss: {float(loss.detach()):.6f}")
    print(f"Gradients finite: {gradients_finite}")
    print(f"Saved Stage-2 smoke checkpoint to {output}")


if __name__ == "__main__":
    main()
