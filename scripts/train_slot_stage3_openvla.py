#!/usr/bin/env python
"""Stage-3: train Slot Fusion, relation tokens, and an OpenVLA action-chunk path."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import (
    OpenVLAForActionPrediction,
    OpenVLAForActionPrediction_SlotAtt,
    SlotFusion,
)
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransformV3, RLDSDatasetV3
from scripts.train_slot_stage2_ssm import Stage2Dynamics


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
    p.add_argument("--action-chunk", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--vision-microbatch", type=int, default=8)
    p.add_argument("--reserve-gib", type=float, default=2.5)
    p.add_argument("--free-fraction", type=float, default=0.80)
    p.add_argument("--seed", type=int, default=20260720)
    return p.parse_args()


class DynamicsQueryRelation(torch.nn.Module):
    """One lightweight relation block with object dynamics as the queries."""

    def __init__(self, slot_dim=512, patch_dim=2176, heads=8):
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(slot_dim)
        self.patch_norm = torch.nn.LayerNorm(slot_dim)
        self.patch_projection = torch.nn.Linear(patch_dim, slot_dim)
        self.cross_attention = torch.nn.MultiheadAttention(slot_dim, heads, batch_first=True)
        self.output_norm = torch.nn.LayerNorm(slot_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(slot_dim, slot_dim * 4), torch.nn.GELU(),
            torch.nn.Linear(slot_dim * 4, slot_dim),
        )
        self.ffn_norm = torch.nn.LayerNorm(slot_dim)

    def forward(self, dynamics, patches):
        query = self.query_norm(dynamics)
        context = self.patch_norm(self.patch_projection(patches))
        attended, _ = self.cross_attention(query, context, context, need_weights=False)
        relation = self.output_norm(dynamics + attended)
        return self.ffn_norm(relation + self.ffn(relation))


class Stage3ActionPath(torch.nn.Module):
    def __init__(self, base_model, action_chunk=16):
        super().__init__()
        self.base_model = base_model
        self.action_chunk = action_chunk
        self.subgoal_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(4096), torch.nn.Linear(4096, 512),
        )
        self.slot_fusion = SlotFusion(
            input_dims=[512, 512, 512], embed_dims=[512, 512, 512], output_dim=512, normalize=True
        )
        self.fusion_residual = torch.nn.Linear(512, 512)
        self.fusion_norm = torch.nn.LayerNorm(512)
        self.relation_encoder = DynamicsQueryRelation()
        self.dynamic_projector = torch.nn.Sequential(torch.nn.LayerNorm(512), torch.nn.Linear(512, 4096))
        self.relation_projector = torch.nn.Sequential(torch.nn.LayerNorm(512), torch.nn.Linear(512, 4096))
        self.object_type_embedding = torch.nn.Parameter(torch.zeros(2, 4096))
        self.action_queries = torch.nn.Parameter(torch.randn(1, action_chunk, 4096) * 0.02)
        self.action_head = torch.nn.Sequential(
            torch.nn.LayerNorm(4096), torch.nn.Linear(4096, 1024),
            torch.nn.GELU(), torch.nn.Linear(1024, 7),
        )

    def trainable_state_dict(self):
        state = {}
        modules = {
            "subgoal_projection": self.subgoal_projection,
            "slot_fusion": self.slot_fusion,
            "fusion_residual": self.fusion_residual,
            "fusion_norm": self.fusion_norm,
            "relation_encoder": self.relation_encoder,
            "dynamic_projector": self.dynamic_projector,
            "relation_projector": self.relation_projector,
            "action_head": self.action_head,
        }
        for prefix, module in modules.items():
            for key, value in module.state_dict().items():
                state[f"{prefix}.{key}"] = value.detach().cpu().contiguous()
        state["object_type_embedding"] = self.object_type_embedding.detach().cpu().contiguous()
        state["action_queries"] = self.action_queries.detach().cpu().contiguous()
        return state


def max_memory_map(reserve_gib, free_fraction):
    result = {}
    gib = 1024 ** 3
    for index in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(index)
        usable = min(int(free * free_fraction), max(0, free - int(reserve_gib * gib)))
        if usable < 2 * gib:
            raise RuntimeError(f"Logical GPU {index} has only {free / gib:.1f} GiB free")
        result[index] = f"{usable // gib}GiB"
    result["cpu"] = "64GiB"
    return result


def select_auxiliary_device(excluded=()):
    candidates = []
    for index in range(torch.cuda.device_count()):
        if index in excluded:
            continue
        free, _ = torch.cuda.mem_get_info(index)
        candidates.append((free, index))
    return torch.device(f"cuda:{max(candidates)[1]}")


def extract_slots(slot_model, base, pixels, vision_device, auxiliary_device, microbatch):
    b, t = pixels.shape[:2]
    flat = pixels.flatten(0, 1)
    chunks = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, flat.shape[0], microbatch):
            encoded = base.vision_backbone(flat[start:start + microbatch].to(vision_device))
            chunks.append(encoded.reshape(encoded.shape[0], -1, 2176).to(auxiliary_device))
        patches = torch.cat(chunks).reshape(b, t, -1, 2176)
        slots, past = [], None
        for timestep in range(t):
            current, _, _ = slot_model.object_centric_tokenizer(patches[:, timestep], past_slots=past)
            past = current[:, None]
            slots.append(current)
    return torch.stack(slots, dim=1), patches.clone()


def object_subgoal_strings(sample, slot_model, final_slots, auxiliary_device, time_index=-1):
    """Assign deterministic oracle object-stage strings to slots via GT masks."""
    reasoning = sample["reasoning_on_image"][time_index]
    instruction = sample["task"]["instruction"]
    seg = sample["all_pixel_seg_values"][time_index].to(auxiliary_device).float()[None, None]
    seg = F.interpolate(seg, size=(64, 64), mode="nearest")[0, 0].long()
    with torch.inference_mode():
        mask_dtype = next(slot_model.object_centric_mask_head.parameters()).dtype
        predicted_masks = slot_model.object_centric_mask_head(final_slots.to(mask_dtype)).sigmoid()[0]
    assignments = {}
    candidates = []
    for name, values in reasoning.items():
        seg_id = int(values[1])
        target = seg.eq(seg_id)
        if not bool(target.any()):
            continue
        intersection = (predicted_masks * target).sum(dim=(-2, -1))
        union = predicted_masks.sum(dim=(-2, -1)) + target.sum() - intersection
        for slot_index, score in enumerate((intersection / union.clamp_min(1e-6)).tolist()):
            candidates.append((score, name, slot_index, str(values[0])))
    used_names, used_slots = set(), set()
    for _, name, slot_index, oracle_cue in sorted(candidates, reverse=True):
        if name in used_names or slot_index in used_slots:
            continue
        assignments[slot_index] = (
            f"task: {instruction}; active object: {name}; "
            f"oracle object subgoal: {oracle_cue}"
        )
        used_names.add(name)
        used_slots.add(slot_index)
    return [assignments.get(index, "no active object subgoal") for index in range(final_slots.shape[1])]


def encode_subgoals(strings, tokenizer, embedding, embedding_device, auxiliary_device):
    tokenized = tokenizer(strings, padding=True, truncation=True, max_length=40, return_tensors="pt")
    ids = tokenized.input_ids.to(embedding_device)
    mask = tokenized.attention_mask.to(embedding_device)
    with torch.inference_mode():
        vectors = embedding(ids)
        pooled = (vectors * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
    return pooled.to(auxiliary_device).unsqueeze(0).clone()


def save_stage3(model, optimizer, step, output_dir, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.trainable_state_dict(), str(output_dir / f"stage3-step-{step:06d}.safetensors"))
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
    memory = max_memory_map(args.reserve_gib, args.free_fraction)
    print(f"OpenVLA max_memory={memory}", flush=True)
    base = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=False, device_map="balanced_low_0", max_memory=memory,
    )
    base.requires_grad_(False)
    vision_device = next(base.vision_backbone.parameters()).device
    embedding = base.get_input_embeddings()
    embedding_device = next(embedding.parameters()).device
    final_device = next(base.language_model.model.layers[-1].parameters()).device
    auxiliary_device = select_auxiliary_device()
    print(f"vision={vision_device} embedding={embedding_device} final={final_device} auxiliary={auxiliary_device}", flush=True)

    slot_model = OpenVLAForActionPrediction_SlotAtt(model=base, number_of_slots=16)
    slot_model.load_state_dict(load_file(args.slot_checkpoint), strict=False)
    slot_model.requires_grad_(False)
    slot_model.object_centric_tokenizer.to(auxiliary_device, dtype=torch.bfloat16)
    # MaskPredictionHead builds FP32 positional coordinates inside forward(), so
    # keep this small frozen oracle-matching head in FP32.
    slot_model.object_centric_mask_head.to(auxiliary_device, dtype=torch.float32)
    slot_model.eval()
    dynamics = Stage2Dynamics("mamba2", residual_next=True)
    dynamics.load_state_dict(load_file(args.stage2_checkpoint), strict=True)
    dynamics.requires_grad_(False)
    dynamics.to(auxiliary_device, dtype=torch.bfloat16).eval()

    model = Stage3ActionPath(base, args.action_chunk)
    model.subgoal_projection.to(auxiliary_device, dtype=torch.bfloat16)
    model.slot_fusion.to(auxiliary_device, dtype=torch.bfloat16)
    model.fusion_residual.to(auxiliary_device, dtype=torch.bfloat16)
    model.fusion_norm.to(auxiliary_device, dtype=torch.bfloat16)
    model.relation_encoder.to(auxiliary_device, dtype=torch.bfloat16)
    model.dynamic_projector.to(auxiliary_device, dtype=torch.bfloat16)
    model.relation_projector.to(auxiliary_device, dtype=torch.bfloat16)
    model.object_type_embedding = torch.nn.Parameter(model.object_type_embedding.to(auxiliary_device, dtype=torch.bfloat16))
    model.action_queries = torch.nn.Parameter(model.action_queries.to(embedding_device, dtype=torch.bfloat16))
    model.action_head.to(final_device, dtype=torch.bfloat16)
    trainable = [p for name, p in model.named_parameters() if not name.startswith("base_model.") and p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    transform = RLDSBatchTransformV3(
        ActionTokenizer(processor.tokenizer), processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    dataset = RLDSDatasetV3(
        args.data_root, "libero_mem", transform, resize_resolution=(224, 224),
        shuffle_buffer_size=10000, image_aug=True,
        window_size=args.horizon + args.action_chunk,
        load_camera_views=("primary", "wrist"), load_depth=True, cropping=False,
    )
    iterator = iter(dataset)
    output_dir = Path(args.output_dir)
    running = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        sample = next(iterator)
        # Causal alignment: the first `horizon` observations are context and
        # the following `action_chunk` actions are future supervision.
        pixels = sample["all_pixel_values"][:args.horizon].unsqueeze(0).to(dtype=torch.bfloat16)
        slots, patches = extract_slots(
            slot_model, base, pixels, vision_device, auxiliary_device, args.vision_microbatch
        )
        with torch.inference_mode(), torch.cuda.device(auxiliary_device), torch.autocast("cuda", dtype=torch.bfloat16):
            _, predicted_next, _ = dynamics(slots, patches, compute_window=False)
        subgoal_strings = object_subgoal_strings(
            sample, slot_model, slots[:, -1], auxiliary_device, time_index=args.horizon - 1
        )
        subgoal_llm = encode_subgoals(
            subgoal_strings, processor.tokenizer, embedding, embedding_device, auxiliary_device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            subgoals = model.subgoal_projection(subgoal_llm)
            current = slots[:, -1].detach()
            # Convert the inference-mode Stage-2 result to ordinary frozen storage
            # before a trainable fusion layer saves it for backward.
            next_slots = predicted_next[:, -1].clone()
            fused = model.slot_fusion([current, next_slots, subgoals])
            fused = model.fusion_norm(model.fusion_residual(current) + fused)
            relation = model.relation_encoder(fused, patches[:, -1].detach())
            dynamic_tokens = model.dynamic_projector(fused) + model.object_type_embedding[0]
            relation_tokens = model.relation_projector(relation) + model.object_type_embedding[1]

        instruction = sample["task"]["instruction"]
        prompt = f"What action should the robot take to {instruction}?"
        prompt_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(embedding_device)
        with torch.inference_mode():
            prompt_embeddings = embedding(prompt_ids)
        injected = torch.cat(
            (prompt_embeddings, relation_tokens.to(embedding_device),
             dynamic_tokens.to(embedding_device), model.action_queries), dim=1
        )
        attention_mask = torch.ones(injected.shape[:2], dtype=torch.long, device=embedding_device)
        output = base.language_model(
            input_ids=None, inputs_embeds=injected, attention_mask=attention_mask,
            output_hidden_states=True, use_cache=False, return_dict=True,
        )
        action_hidden = output.hidden_states[-1][:, -args.action_chunk:].to(final_device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted_actions = model.action_head(action_hidden)
            target_actions = sample["continuous_actions"][
                args.horizon:args.horizon + args.action_chunk
            ].unsqueeze(0).to(final_device)
            loss = F.l1_loss(predicted_actions.float(), target_actions.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        gradients_finite = all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in trainable
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        running += float(loss.detach())
        if step % args.log_every == 0:
            print(
                f"step={step}/{args.steps} action_l1={running / args.log_every:.6f} "
                f"gradients_finite={gradients_finite}", flush=True
            )
            running = 0.0
        if step % args.save_every == 0:
            save_stage3(model, optimizer, step, output_dir, args)
            print(f"saved checkpoint at step {step}", flush=True)
    save_stage3(model, optimizer, args.steps, output_dir, args)


if __name__ == "__main__":
    main()
