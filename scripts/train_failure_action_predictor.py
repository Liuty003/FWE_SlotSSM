#!/usr/bin/env python
"""Train the Failure-Aware action-conditioned next-slot predictor.

The successful original OpenVLA remains the acting policy.  This sidecar model
predicts the object slots expected after the *executed* action, which later
provides a causally meaningful post-transition deviation signal.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction, OpenVLAForActionPrediction_SlotAtt
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransformV3, RLDSDatasetV3
from scripts.train_slot_stage2_ssm import Stage2Dynamics
from scripts.train_slot_stage3_openvla import extract_slots, max_memory_map, select_auxiliary_device


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial")
    p.add_argument("--data-root", default="/data2/lty/libero-mem-rlds/full")
    p.add_argument("--slot-checkpoint", required=True)
    p.add_argument("--stage2-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--vision-microbatch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--contrastive-weight", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--reserve-gib", type=float, default=2.5)
    p.add_argument("--free-fraction", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=20260720)
    return p.parse_args()


class ActionConditionedSlotPredictor(torch.nn.Module):
    """Residual correction to frozen Stage-2 using executed action/proprio."""

    def __init__(self, slot_dim=512, action_dim=7, proprio_dim=7):
        super().__init__()
        self.action_encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(action_dim), torch.nn.Linear(action_dim, 128),
            torch.nn.GELU(), torch.nn.Linear(128, slot_dim),
        )
        self.proprio_encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(proprio_dim), torch.nn.Linear(proprio_dim, 128),
            torch.nn.GELU(), torch.nn.Linear(128, slot_dim),
        )
        self.correction = torch.nn.Sequential(
            torch.nn.LayerNorm(slot_dim * 5),
            torch.nn.Linear(slot_dim * 5, 1024), torch.nn.GELU(),
            torch.nn.Linear(1024, slot_dim),
        )
        # Begin exactly at the already-trained Stage-2 baseline.
        torch.nn.init.zeros_(self.correction[-1].weight)
        torch.nn.init.zeros_(self.correction[-1].bias)

    def forward(self, current, dynamics, stage2_next, executed_action, proprio):
        action = self.action_encoder(executed_action).unsqueeze(2).expand_as(current)
        state = self.proprio_encoder(proprio).unsqueeze(2).expand_as(current)
        correction = self.correction(torch.cat((current, dynamics, stage2_next, action, state), dim=-1))
        return stage2_next + correction


def slot_identity_contrastive(predicted, target):
    query = F.normalize(predicted.float(), dim=-1)
    key = F.normalize(target.float(), dim=-1)
    logits = torch.einsum("btkd,btjd->btkj", query, key)
    labels = torch.arange(query.shape[2], device=query.device).view(1, 1, -1).expand(query.shape[:3])
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    positive = logits.diagonal(dim1=-2, dim2=-1).mean()
    eye = torch.eye(logits.shape[-1], device=logits.device, dtype=torch.bool).view(1, 1, logits.shape[-1], -1)
    negative = logits.masked_select(~eye).mean()
    return loss, positive, negative


def save(model, optimizer, step, output_dir, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(output_dir / f"action-predictor-step-{step:06d}.safetensors"))
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, output_dir / f"optimizer-step-{step:06d}.pt")
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


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
    load_result = slot_model.load_state_dict(load_file(args.slot_checkpoint), strict=False)
    slot_model.requires_grad_(False)
    slot_model.object_centric_tokenizer.to(auxiliary_device, dtype=torch.bfloat16)
    slot_model.object_centric_mask_head.to(auxiliary_device, dtype=torch.float32)
    slot_model.eval()
    stage2 = Stage2Dynamics("mamba2", residual_next=True)
    stage2.load_state_dict(load_file(args.stage2_checkpoint), strict=True)
    stage2.requires_grad_(False)
    stage2.to(auxiliary_device, dtype=torch.bfloat16).eval()
    predictor = ActionConditionedSlotPredictor().to(auxiliary_device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(
        f"vision={vision_device} auxiliary={auxiliary_device} "
        f"stage1_missing={len(load_result.missing_keys)} trainable={sum(p.numel() for p in predictor.parameters())}",
        flush=True,
    )

    transform = RLDSBatchTransformV3(
        ActionTokenizer(processor.tokenizer), processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    dataset = RLDSDatasetV3(
        args.data_root, "libero_mem", transform, resize_resolution=(224, 224),
        shuffle_buffer_size=10000, image_aug=True, window_size=args.horizon,
        load_camera_views=("primary",), load_depth=False, load_proprio=True, cropping=False,
    )
    iterator = iter(dataset)
    output_dir = Path(args.output_dir)
    running = {"loss": 0.0, "conditioned": 0.0, "stage2": 0.0}
    for step in range(1, args.steps + 1):
        sample = next(iterator)
        pixels = sample["all_pixel_values"].unsqueeze(0).to(dtype=torch.bfloat16)
        slots, patches = extract_slots(
            slot_model, base, pixels, vision_device, auxiliary_device, args.vision_microbatch
        )
        with torch.inference_mode(), torch.cuda.device(auxiliary_device), torch.autocast("cuda", dtype=torch.bfloat16):
            dynamics, stage2_next, _ = stage2(slots, patches, compute_window=False)
        current = slots[:, :-1].detach()
        target = slots[:, 1:].detach()
        actions = sample["continuous_actions"][:-1].unsqueeze(0).to(auxiliary_device, dtype=torch.bfloat16)
        proprio = sample["proprio"][:-1].unsqueeze(0).to(auxiliary_device, dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = predictor(
                current, dynamics[:, :-1].clone(), stage2_next[:, :-1].clone(), actions, proprio
            )
            next_l1 = F.smooth_l1_loss(predicted.float(), target.float())
            contrastive, positive, negative = slot_identity_contrastive(predicted, target)
            loss = next_l1 + args.contrastive_weight * contrastive
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.max_grad_norm)
        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in predictor.parameters())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            baseline = F.smooth_l1_loss(stage2_next[:, :-1].float(), target.float())
        running["loss"] += float(loss)
        running["conditioned"] += float(next_l1)
        running["stage2"] += float(baseline)
        if step == 1 and step % args.log_every != 0:
            print(
                f"step={step}/{args.steps} loss={float(loss):.6f} "
                f"conditioned_l1={float(next_l1):.6f} stage2_l1={float(baseline):.6f} "
                f"pos={float(positive):.4f} neg={float(negative):.4f} gradients_finite={finite}",
                flush=True,
            )
        if step % args.log_every == 0:
            print(
                f"step={step}/{args.steps} loss={running['loss']/args.log_every:.6f} "
                f"conditioned_l1={running['conditioned']/args.log_every:.6f} "
                f"stage2_l1={running['stage2']/args.log_every:.6f} "
                f"pos={float(positive):.4f} neg={float(negative):.4f} gradients_finite={finite}",
                flush=True,
            )
            running = {key: 0.0 for key in running}
        if step % args.save_every == 0:
            save(predictor, optimizer, step, output_dir, args)
    save(predictor, optimizer, args.steps, output_dir, args)
    print(f"saved={output_dir / f'action-predictor-step-{args.steps:06d}.safetensors'}")


if __name__ == "__main__":
    main()
