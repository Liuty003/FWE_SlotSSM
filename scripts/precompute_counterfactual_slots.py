#!/usr/bin/env python
"""Precompute Stage-1 slots for matched LIBERO counterfactual branches."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction, OpenVLAForActionPrediction_SlotAtt
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from scripts.train_slot_stage3_openvla import max_memory_map, select_auxiliary_device


BRANCHES = (
    "nominal", "translate_x_minus", "translate_x_plus",
    "translate_y_minus", "translate_y_plus", "translate_z_minus",
    "translate_z_plus", "rotate_z_minus", "rotate_z_plus", "flip_gripper",
)


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial")
    p.add_argument("--slot-checkpoint", required=True)
    p.add_argument("--input-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--episode-ids", default="0")
    p.add_argument("--branch-steps", type=int, default=4)
    p.add_argument("--vision-microbatch", type=int, default=16)
    p.add_argument("--reserve-gib", type=float, default=2.0)
    p.add_argument("--free-fraction", type=float, default=0.75)
    return p.parse_args()


def load_rows(path):
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    grouped = defaultdict(dict)
    for row in rows:
        grouped[int(row["fork_id"])][row["branch"]] = row
    return grouped


def preprocess(paths, image_transform):
    return torch.stack([
        image_transform(Image.open(path).convert("RGB")) for path in paths
    ])


@torch.inference_mode()
def encode_patches(base, pixels, vision_device, auxiliary_device, microbatch):
    chunks = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(pixels), microbatch):
            encoded = base.vision_backbone(pixels[start:start + microbatch].to(vision_device))
            chunks.append(encoded.reshape(encoded.shape[0], -1, 2176).to(auxiliary_device))
    return torch.cat(chunks)


@torch.inference_mode()
def slots_from_branch(tokenizer, before_patch, after_patches, auxiliary_device):
    with torch.cuda.device(auxiliary_device), torch.autocast("cuda", dtype=torch.bfloat16):
        before, _, _ = tokenizer(before_patch[None], past_slots=None)
        past = before[:, None]
        after = []
        for patch in after_patches:
            current, _, _ = tokenizer(patch[None], past_slots=past)
            after.append(current[0])
            past = current[:, None]
    return before[0], torch.stack(after)


def physical_labels(row, steps):
    trajectory = list(row["branch_trajectory"])
    if not trajectory:
        raise ValueError(
            f"fork={row.get('fork_id')} branch={row.get('branch')} has an empty branch_trajectory"
        )
    trajectory += [trajectory[-1]] * (steps - len(trajectory))
    trajectory = trajectory[:steps]
    before_robot = row["robot_state_before"]
    before_eef = np.asarray(before_robot["robot0_eef_pos"], dtype=np.float32)
    before_gripper = np.asarray(before_robot["robot0_gripper_qpos"], dtype=np.float32)
    initial_objects = row["privileged_object_state_before"]
    eef_delta, gripper_delta, object_delta, object_distance = [], [], [], []
    for transition in trajectory:
        robot = transition["robot_state_after"]
        eef_delta.append(np.asarray(robot["robot0_eef_pos"], dtype=np.float32) - before_eef)
        gripper_delta.append(
            np.asarray(robot["robot0_gripper_qpos"], dtype=np.float32) - before_gripper
        )
        after_objects = transition["privileged_object_state_after"]
        candidates = []
        for key, value in initial_objects.items():
            if not key.endswith("_pos") or key not in after_objects:
                continue
            delta = np.asarray(after_objects[key], dtype=np.float32) - np.asarray(value, dtype=np.float32)
            candidates.append((float(np.linalg.norm(delta)), delta))
        distance, delta = max(candidates, key=lambda item: item[0]) if candidates else (0.0, np.zeros(3, np.float32))
        object_distance.append(distance)
        object_delta.append(delta)
    return (
        torch.tensor(np.asarray(eef_delta)), torch.tensor(np.asarray(gripper_delta)),
        torch.tensor(np.asarray(object_delta)), torch.tensor(np.asarray(object_distance)),
    )


def main():
    args = arguments()
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
    result = slot_model.load_state_dict(load_file(args.slot_checkpoint), strict=False)
    slot_model.requires_grad_(False)
    slot_model.object_centric_tokenizer.to(auxiliary_device, dtype=torch.bfloat16).eval()
    print(
        f"vision={vision_device} auxiliary={auxiliary_device} "
        f"stage1_missing={len(result.missing_keys)}", flush=True,
    )

    input_root, output_dir = Path(args.input_root), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"branch_order": list(BRANCHES), "tasks": []}
    for task_id in [int(value) for value in args.task_ids.split(",") if value]:
      for episode_id in [int(value) for value in args.episode_ids.split(",") if value]:
        source = input_root / f"libero_spatial_t{task_id}_e{episode_id}_h4_dual"
        if not source.is_dir():
            raise FileNotFoundError(f"counterfactual episode directory not found: {source}")
        output = output_dir / f"libero_spatial_t{task_id}_e{episode_id}_slots.safetensors"
        if output.is_file():
            existing = load_file(str(output))
            required = {
                "before_slots", "after_slots", "after_valid", "executed_actions",
                "eef_delta_m", "gripper_qpos_delta", "max_object_delta_m",
                "max_object_displacement_m",
            }
            missing_keys = required - set(existing)
            if missing_keys:
                raise RuntimeError(f"existing shard {output} missing keys={sorted(missing_keys)}")
            manifest["tasks"].append({
                "task_id": task_id, "episode_id": episode_id,
                "source": str(source), "output": str(output),
                "num_forks": int(existing["before_slots"].shape[0]),
                "tensor_shapes": {k: list(v.shape) for k, v in existing.items()},
            })
            print(f"skip existing={output}", flush=True)
            del existing
            continue
        grouped = load_rows(source / "transitions.jsonl")
        task_before, task_after, task_valid = [], [], []
        task_actions, task_joint, task_fork_ids = [], [], []
        task_eef, task_gripper, task_object_delta, task_object_distance = [], [], [], []
        for fork_index, fork_id in enumerate(sorted(grouped)):
            branches = grouped[fork_id]
            missing = set(BRANCHES) - set(branches)
            if missing:
                raise RuntimeError(f"task={task_id} fork={fork_id} missing branches={sorted(missing)}")
            before_path = branches["nominal"]["before_image"]
            image_paths = [before_path]
            branch_lengths = []
            for name in BRANCHES:
                paths = list(branches[name]["after_images"])
                branch_lengths.append(len(paths))
                # Pad short success-terminated branches with their final frame.
                paths += [paths[-1]] * (args.branch_steps - len(paths))
                image_paths.extend(paths[:args.branch_steps])
            pixels = preprocess(image_paths, processor.image_processor.apply_transform)
            patches = encode_patches(
                base, pixels, vision_device, auxiliary_device, args.vision_microbatch
            )
            before_patch = patches[0]
            branch_slots = []
            before_slot = None
            cursor = 1
            for _ in BRANCHES:
                current_before, after = slots_from_branch(
                    slot_model.object_centric_tokenizer, before_patch,
                    patches[cursor:cursor + args.branch_steps], auxiliary_device,
                )
                cursor += args.branch_steps
                if before_slot is None:
                    before_slot = current_before
                branch_slots.append(after)
            task_before.append(before_slot.cpu())
            task_after.append(torch.stack(branch_slots).cpu())
            task_valid.append(torch.tensor([
                [step < length for step in range(args.branch_steps)] for length in branch_lengths
            ], dtype=torch.bool))
            task_actions.append(torch.tensor([
                branches[name]["executed_env_action"] for name in BRANCHES
            ], dtype=torch.float32))
            task_joint.append(torch.tensor(
                branches["nominal"]["robot_state_before"]["robot0_joint_pos"], dtype=torch.float32
            ))
            physical = [physical_labels(branches[name], args.branch_steps) for name in BRANCHES]
            task_eef.append(torch.stack([value[0] for value in physical]))
            task_gripper.append(torch.stack([value[1] for value in physical]))
            task_object_delta.append(torch.stack([value[2] for value in physical]))
            task_object_distance.append(torch.stack([value[3] for value in physical]))
            task_fork_ids.append(fork_id)
            print(
                f"task={task_id} fork={fork_index + 1}/{len(grouped)} id={fork_id}", flush=True
            )
        tensors = {
            "before_slots": torch.stack(task_before).to(torch.bfloat16).contiguous(),
            "after_slots": torch.stack(task_after).to(torch.bfloat16).contiguous(),
            "after_valid": torch.stack(task_valid).contiguous(),
            "executed_actions": torch.stack(task_actions).contiguous(),
            "joint_positions": torch.stack(task_joint).contiguous(),
            "eef_delta_m": torch.stack(task_eef).float().contiguous(),
            "gripper_qpos_delta": torch.stack(task_gripper).float().contiguous(),
            "max_object_delta_m": torch.stack(task_object_delta).float().contiguous(),
            "max_object_displacement_m": torch.stack(task_object_distance).float().contiguous(),
            "fork_ids": torch.tensor(task_fork_ids, dtype=torch.int64),
            "branch_ids": torch.arange(len(BRANCHES), dtype=torch.int64),
        }
        save_file(tensors, str(output))
        manifest["tasks"].append({
            "task_id": task_id, "episode_id": episode_id,
            "source": str(source), "output": str(output),
            "num_forks": len(grouped), "tensor_shapes": {k: list(v.shape) for k, v in tensors.items()},
        })
        print(f"saved={output}", flush=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest={output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
