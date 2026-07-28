#!/usr/bin/env python
"""Train Stage-2 object-centric temporal dynamics on frozen Stage-1 slots."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
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


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial")
    p.add_argument("--data-root", default="/data2/lty/libero-mem-rlds/full")
    p.add_argument("--slot-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--contrast-weight", type=float, default=0.1)
    p.add_argument("--window-weight", type=float, default=1.0)
    p.add_argument("--next-weight", type=float, default=1.0)
    p.add_argument("--backend", choices=("mamba2", "reference"), default="mamba2")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--vision-microbatch", type=int, default=8)
    p.add_argument("--load-stage2", default=None)
    p.add_argument("--residual-next", action="store_true")
    p.add_argument("--reset-next-head", action="store_true")
    p.add_argument("--next-only", action="store_true")
    return p.parse_args()


class Stage2Dynamics(torch.nn.Module):
    """Author-code-compatible SlotSSM plus the paper's temporal prediction heads."""

    OFFSETS = tuple(range(-16, 16))

    def __init__(self, backend="mamba2", residual_next=False):
        super().__init__()
        self.residual_next = residual_next
        self.ssm = EmbodiedSlotSSM(
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
            mamba_version=backend,
            attn_impl="eager",
            lookback_only=True,
        )
        self.next_slot_head = torch.nn.Sequential(
            torch.nn.LayerNorm(1024), torch.nn.Linear(1024, 1024),
            torch.nn.GELU(), torch.nn.Linear(1024, 512),
        )
        # A shared head is applied for each offset, avoiding a very large 512*32 output layer.
        self.offset_embedding = torch.nn.Embedding(len(self.OFFSETS), 512)
        self.past_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(1536), torch.nn.Linear(1536, 1024),
            torch.nn.GELU(), torch.nn.Linear(1024, 512),
        )

    def forward(self, slots, patches, compute_window=True):
        dynamics, _, _ = self.ssm(slots, patches)
        predicted_next = self.next_slot_head(torch.cat((dynamics, slots), dim=-1))
        if self.residual_next:
            predicted_next = slots + predicted_next
        if not compute_window:
            return dynamics, predicted_next, None
        b, t, k, _ = slots.shape
        offsets = self.offset_embedding.weight.view(1, 1, 1, len(self.OFFSETS), 512)
        base = torch.cat((dynamics, slots), dim=-1).unsqueeze(3).expand(b, t, k, len(self.OFFSETS), 1024)
        predicted_window = self.past_mlp(torch.cat((base, offsets.expand(b, t, k, -1, -1)), dim=-1))
        return dynamics, predicted_next, predicted_window


def window_loss(prediction, target):
    """Masked Smooth-L1 over offsets -16..15 without crossing a sampled window boundary."""
    _, timesteps, _, _, _ = prediction.shape
    total = prediction.new_zeros((), dtype=torch.float32)
    count = 0
    for index, offset in enumerate(Stage2Dynamics.OFFSETS):
        source_start = max(0, -offset)
        source_end = min(timesteps, timesteps - offset)
        if source_end <= source_start:
            continue
        pred = prediction[:, source_start:source_end, :, index].float()
        truth = target[:, source_start + offset:source_end + offset].detach().float()
        total = total + F.smooth_l1_loss(pred, truth)
        count += 1
    return total / max(count, 1)


def temporal_contrastive_loss(predicted_next, next_slots):
    """InfoNCE: the same slot at t+1 is positive; other slots are negatives."""
    query = F.normalize(predicted_next[:, :-1].float(), dim=-1)
    key = F.normalize(next_slots[:, 1:].detach().float(), dim=-1)
    logits = torch.einsum("btkd,btjd->btkj", query, key)
    labels = torch.arange(query.shape[2], device=query.device).view(1, 1, -1).expand(query.shape[:3])
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    positive = logits.diagonal(dim1=-2, dim2=-1).mean()
    eye = torch.eye(logits.shape[-1], device=logits.device, dtype=torch.bool).view(1, 1, logits.shape[-1], -1)
    negative = logits.masked_select(~eye).mean()
    return loss, positive, negative


def save_checkpoint(model, optimizer, step, output_dir, args, rank):
    if rank != 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(output_dir / f"stage2-step-{step:06d}.safetensors"))
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, output_dir / f"optimizer-step-{step:06d}.pt")
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


def main():
    args = arguments()
    distributed = "RANK" in os.environ
    if distributed and not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

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
    incompatible = slot_model.load_state_dict(load_file(args.slot_checkpoint), strict=False)
    slot_model.requires_grad_(False)
    slot_model = slot_model.to(device, dtype=torch.bfloat16).eval()
    if rank == 0:
        print(f"Stage-1 loaded: missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}", flush=True)

    transform = RLDSBatchTransformV3(
        ActionTokenizer(processor.tokenizer), processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    dataset = RLDSDatasetV3(
        args.data_root, "libero_mem", transform, resize_resolution=(224, 224),
        shuffle_buffer_size=10000, image_aug=True, window_size=args.horizon,
        load_camera_views=("primary", "wrist"), load_depth=True, cropping=False,
    )
    iterator = iter(dataset)
    model = Stage2Dynamics(args.backend, residual_next=args.residual_next).to(device, dtype=torch.bfloat16)
    if args.load_stage2:
        loaded = model.load_state_dict(load_file(args.load_stage2), strict=True)
        if rank == 0:
            print(f"Loaded Stage-2 initialization from {args.load_stage2}: {loaded}", flush=True)
    if args.reset_next_head:
        final = model.next_slot_head[-1]
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)
    if args.next_only:
        model.requires_grad_(False)
        model.next_slot_head.requires_grad_(True)
    # EmbodiedSlotSSM carries this legacy parameter but forward() consumes the
    # Stage-1 slots passed by the caller, so init_slots is intentionally unused.
    model.ssm.init_slots.requires_grad_(False)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    output_dir = Path(args.output_dir)
    running = {"total": 0.0, "next": 0.0, "window": 0.0, "contrast": 0.0, "pos": 0.0, "neg": 0.0}

    for step in range(1, args.steps + 1):
        sample = next(iterator)
        pixels = sample["all_pixel_values"].unsqueeze(0).to(device, dtype=torch.bfloat16, non_blocking=True)
        slots, past = [], None
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            flat_pixels = pixels.flatten(0, 1)
            patch_chunks = []
            for start in range(0, args.horizon, args.vision_microbatch):
                encoded = slot_model.model.vision_backbone(flat_pixels[start:start + args.vision_microbatch])
                patch_chunks.append(encoded.reshape(encoded.shape[0], -1, 2176))
            patch_sequence = torch.cat(patch_chunks, dim=0).unsqueeze(0)
            for timestep in range(args.horizon):
                patch = patch_sequence[:, timestep]
                current, _, _ = slot_model.object_centric_tokenizer(patch, past_slots=past)
                past = current[:, None]
                slots.append(current)
        slots = torch.stack(slots, dim=1)
        # Leave inference-mode storage before trainable cross-attention saves K/V for backward.
        patches = patch_sequence.clone()
        sync_now = step % args.grad_accum == 0
        sync_context = model.no_sync() if distributed and not sync_now else torch.enable_grad()
        with sync_context:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, predicted_next, predicted_window = model(slots, patches, compute_window=not args.next_only)
                next_loss = F.smooth_l1_loss(predicted_next[:, :-1].float(), slots[:, 1:].detach().float())
                contrast, positive, negative = temporal_contrastive_loss(predicted_next, slots)
                win_loss = window_loss(predicted_window, slots) if predicted_window is not None else next_loss.new_zeros(())
                total = (args.next_weight * next_loss + args.contrast_weight * contrast + args.window_weight * win_loss) / args.grad_accum
            total.backward()
        if sync_now:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        values = (total.item() * args.grad_accum, next_loss.item(), win_loss.item(), contrast.item(), positive.item(), negative.item())
        for key, value in zip(running, values):
            running[key] += value
        if rank == 0 and step % args.log_every == 0:
            metrics = " ".join(f"{k}={v / args.log_every:.5f}" for k, v in running.items())
            print(f"step={step}/{args.steps} {metrics}", flush=True)
            running = {key: 0.0 for key in running}
        if step % args.save_every == 0:
            if distributed:
                dist.barrier()
            save_checkpoint(model, optimizer, step, output_dir, args, rank)
            if rank == 0:
                print(f"saved checkpoint at step {step}", flush=True)

    save_checkpoint(model, optimizer, args.steps, output_dir, args, rank)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
