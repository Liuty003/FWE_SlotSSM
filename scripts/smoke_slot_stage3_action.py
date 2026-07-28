#!/usr/bin/env python
"""One-step action prediction smoke test conditioned on Mamba2 object Slots."""

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


class SlotDynamicsActionModel(torch.nn.Module):
    def __init__(self):
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
            mamba_version="mamba2",
            attn_impl="eager",
            lookback_only=True,
        )
        self.object_centric_next_slot_head = torch.nn.Linear(1024, 512)
        self.object_centric_action_head = torch.nn.Sequential(
            torch.nn.LayerNorm(1024),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 7),
        )

    def forward(self, slots, patch_features):
        temporal_slots, _, _ = self.object_centric_ssm(slots, patch_features)
        predicted_next = self.object_centric_next_slot_head(
            torch.cat([temporal_slots, slots], dim=-1)
        )
        fused = torch.cat([slots, predicted_next], dim=-1).mean(dim=2)
        return self.object_centric_action_head(fused), predicted_next


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--data-root", default="/data2/lty/libero-mem-rlds")
    parser.add_argument("--slot-checkpoint", required=True)
    parser.add_argument("--ssm-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-3 action smoke requires one CUDA GPU.")
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

    dynamics_action = SlotDynamicsActionModel()
    missing, unexpected = dynamics_action.load_state_dict(
        load_file(args.ssm_checkpoint), strict=False
    )
    print(f"Loaded Stage-2 Mamba2: missing={len(missing)} unexpected={len(unexpected)}")
    # Freeze the Stage-2 dynamics for this smoke; train only the action head.
    dynamics_action.object_centric_ssm.requires_grad_(False)
    dynamics_action.object_centric_next_slot_head.requires_grad_(False)
    dynamics_action = dynamics_action.to(device, dtype=torch.bfloat16)

    transform = RLDSBatchTransformV3(
        ActionTokenizer(processor.tokenizer),
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    dataset = RLDSDatasetV3(
        args.data_root, "libero_mem", transform,
        resize_resolution=(224, 224), shuffle_buffer_size=1,
        image_aug=False, window_size=args.horizon,
        load_camera_views=("primary", "wrist"), load_depth=True, cropping=False,
    )
    sample = next(iter(dataset))
    pixels = sample["all_pixel_values"].unsqueeze(0).to(device, dtype=torch.bfloat16)
    actions = sample["continuous_actions"].unsqueeze(0).to(device)

    slots, patches = [], []
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

    optimizer = torch.optim.AdamW(
        dynamics_action.object_centric_action_head.parameters(), lr=1e-4
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        predicted_actions, predicted_next = dynamics_action(slots, patches)
        action_loss = F.l1_loss(predicted_actions.float(), actions.float())
        slot_consistency = F.mse_loss(predicted_next[:, :-1].float(), slots[:, 1:].float())
        loss = action_loss + 0.01 * slot_consistency
    loss.backward()
    optimizer.step()

    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in dynamics_action.object_centric_action_head.parameters()
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            key: value.detach().cpu()
            for key, value in dynamics_action.state_dict().items()
            if "object_centric_action_head" in key
        },
        str(output),
    )
    print(f"Stage-3 action L1 loss: {float(action_loss.detach()):.6f}")
    print(f"Stage-3 Slot consistency loss: {float(slot_consistency.detach()):.6f}")
    print(f"Predicted action shape: {tuple(predicted_actions.shape)}")
    print(f"Action-head gradients finite: {gradients_finite}")
    print(f"Saved Stage-3 action smoke checkpoint to {output}")


if __name__ == "__main__":
    main()
