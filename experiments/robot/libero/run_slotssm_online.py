#!/usr/bin/env python
"""Headless online LIBERO rollout for the reproduced Stage-1/2/3 SlotSSM path.

This is the first mechanically correct online bridge.  It recomputes a causal
32-frame window at every control step and executes only the first action in the
predicted 16-action chunk.  Recurrent Mamba cache optimization comes later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_ROOT = REPO_ROOT / "experiments" / "robot" / "libero"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LIBERO_ROOT))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_ROOT / "baseline_libero_config"))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import (
    OpenVLAForActionPrediction,
    OpenVLAForActionPrediction_SlotAtt,
)
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from scripts.evaluate_slot_stage3_openvla import build_object_tokens, decode_action, load_stage3_only
from scripts.train_slot_stage2_ssm import Stage2Dynamics
from scripts.train_slot_stage3_openvla import Stage3ActionPath, encode_subgoals
from run_openvla_baseline import (
    SUITE_MAX_STEPS,
    center_crop,
    extract_privileged_object_state,
    extract_robot_state,
    jsonable,
    write_json,
)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial")
    p.add_argument("--slot-checkpoint", required=True)
    p.add_argument("--stage2-checkpoint", required=True)
    p.add_argument("--stage3-checkpoint", required=True)
    p.add_argument("--task-suite", choices=tuple(SUITE_MAX_STEPS), default="libero_spatial")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--episode-id", type=int, default=0)
    p.add_argument("--wait-steps", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--action-chunk", type=int, default=16)
    p.add_argument("--execute-chunk", type=int, default=1)
    p.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device-map", default="balanced_low_0")
    p.add_argument("--gpu-memory-gib", type=int, default=7)
    p.add_argument("--save-features", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "stage0b-slotssm.mp4")
    p.add_argument("--rollout-dir", type=Path, default=None)
    return p.parse_args()


def register_local_openvla() -> None:
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def denormalize_actions(model, suite: str, normalized: np.ndarray) -> np.ndarray:
    key = suite
    if key not in model.norm_stats and f"{key}_no_noops" in model.norm_stats:
        key = f"{key}_no_noops"
    if key not in model.norm_stats:
        raise KeyError(f"No action statistics for {suite!r}; available={list(model.norm_stats)}")
    stats = model.get_action_stats(key)
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)))
    return np.where(mask, 0.5 * (normalized + 1.0) * (high - low) + low, normalized)


class OnlineSlotSSMPolicy:
    def __init__(self, args: argparse.Namespace):
        register_local_openvla()
        self.args = args
        self.processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
        max_memory = {i: f"{args.gpu_memory_gib}GiB" for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "64GiB"
        self.base = AutoModelForVision2Seq.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=False, device_map=args.device_map, max_memory=max_memory,
        ).eval()
        self.base.requires_grad_(False)
        self.vision_device = next(self.base.vision_backbone.parameters()).device
        self.embedding = self.base.get_input_embeddings()
        self.embedding_device = next(self.embedding.parameters()).device
        self.final_device = next(self.base.language_model.model.layers[-1].parameters()).device

        # Put the compact Slot/Mamba path beside the vision tower.  Keeping all
        # Mamba tensors on one CUDA device avoids Triton's CPU-pointer failure.
        self.auxiliary_device = self.vision_device
        self.slot_model = OpenVLAForActionPrediction_SlotAtt(model=self.base, number_of_slots=16)
        slot_result = self.slot_model.load_state_dict(load_file(args.slot_checkpoint), strict=False)
        self.slot_model.requires_grad_(False)
        self.slot_model.object_centric_tokenizer.to(self.auxiliary_device, dtype=torch.bfloat16)
        self.slot_model.object_centric_mask_head.to(self.auxiliary_device, dtype=torch.float32)
        self.slot_model.eval()

        self.stage2 = Stage2Dynamics("mamba2", residual_next=True)
        self.stage2.load_state_dict(load_file(args.stage2_checkpoint), strict=True)
        self.stage2.requires_grad_(False)
        self.stage2.to(self.auxiliary_device, dtype=torch.bfloat16).eval()

        self.stage3 = Stage3ActionPath(self.base, args.action_chunk)
        stage3_keys, stage3_params = load_stage3_only(self.stage3, args.stage3_checkpoint)
        self.stage3.requires_grad_(False)
        for module in (
            self.stage3.subgoal_projection, self.stage3.slot_fusion,
            self.stage3.fusion_residual, self.stage3.fusion_norm,
            self.stage3.relation_encoder, self.stage3.dynamic_projector,
            self.stage3.relation_projector,
        ):
            module.to(self.auxiliary_device, dtype=torch.bfloat16)
        self.stage3.object_type_embedding = torch.nn.Parameter(
            self.stage3.object_type_embedding.to(self.auxiliary_device, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.stage3.action_queries = torch.nn.Parameter(
            self.stage3.action_queries.to(self.embedding_device, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.stage3.action_head.to(self.final_device, dtype=torch.bfloat16)
        self.stage3.eval()
        self.frames: deque[torch.Tensor] = deque(maxlen=args.horizon)
        print(
            f"vision={self.vision_device} embedding={self.embedding_device} "
            f"final={self.final_device} auxiliary={self.auxiliary_device}"
        )
        print(
            f"Stage1 missing={len(slot_result.missing_keys)} unexpected={len(slot_result.unexpected_keys)}; "
            f"Stage3 keys={stage3_keys} params={stage3_params}"
        )

    def reset(self) -> None:
        self.frames.clear()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        if self.args.center_crop:
            image = center_crop(image)
        return self.processor.image_processor.apply_transform(Image.fromarray(image).convert("RGB"))

    @torch.inference_mode()
    def act(self, image: np.ndarray, instruction: str) -> tuple[np.ndarray, dict[str, np.ndarray | int]]:
        self.frames.append(self.preprocess(image))
        history_length = len(self.frames)
        # Left pad with the earliest observation: no future observation leaks in.
        history = list(self.frames)
        history = [history[0]] * (self.args.horizon - len(history)) + history
        pixels = torch.stack(history).unsqueeze(0).to(dtype=torch.bfloat16)
        b, t = pixels.shape[:2]
        flat = pixels.flatten(0, 1)
        encoded = []
        for start in range(0, flat.shape[0], 4):
            value = self.base.vision_backbone(flat[start:start + 4].to(self.vision_device))
            encoded.append(value.reshape(value.shape[0], -1, 2176).to(self.auxiliary_device))
        patches = torch.cat(encoded).reshape(b, t, -1, 2176)
        slots, past = [], None
        with torch.cuda.device(self.auxiliary_device), torch.autocast("cuda", dtype=torch.bfloat16):
            for index in range(t):
                current, _, _ = self.slot_model.object_centric_tokenizer(
                    patches[:, index], past_slots=past
                )
                past = current[:, None]
                slots.append(current)
            slots = torch.stack(slots, dim=1)
            dynamics, predicted_next, _ = self.stage2(slots, patches, compute_window=False)

        neutral = ["no active object subgoal"] * slots.shape[2]
        subgoal_llm = encode_subgoals(
            neutral, self.processor.tokenizer, self.embedding,
            self.embedding_device, self.auxiliary_device,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            relation, dynamic = build_object_tokens(
                self.stage3, slots[:, -1], predicted_next[:, -1], patches[:, -1],
                subgoal_llm, "full",
            )
            normalized = decode_action(
                self.stage3, self.base, self.embedding, self.processor.tokenizer,
                instruction, relation, dynamic, self.embedding_device, self.final_device,
            )
        normalized_np = normalized[0].float().cpu().numpy()
        planned_chunk = denormalize_actions(self.base, self.args.task_suite, normalized_np).astype(np.float32)
        # Same LIBERO conversion used by the verified OpenVLA baseline.
        planned_chunk[:, -1] = np.sign(2.0 * planned_chunk[:, -1] - 1.0) * -1.0
        features = {
            "history_length": history_length,
            "slots": slots[:, -1].float().cpu().numpy(),
            "predicted_next_slots": predicted_next[:, -1].float().cpu().numpy(),
            "dynamics_state": dynamics[:, -1].float().cpu().numpy(),
            "normalized_action_chunk": normalized_np,
            "planned_action_chunk": planned_chunk,
        }
        return planned_chunk, features


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("SlotSSM online inference requires CUDA")
    if not 1 <= args.execute_chunk <= args.action_chunk:
        raise ValueError("execute-chunk must be between 1 and action-chunk")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(initial_states[args.episode_id])

    rollout_dir = args.rollout_dir or args.output.with_suffix("")
    features_dir = rollout_dir / "features"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    if args.save_features:
        features_dir.mkdir(parents=True, exist_ok=True)
    steps_path = rollout_dir / "steps.jsonl"
    metadata_path = rollout_dir / "metadata.json"
    steps_path.write_text("", encoding="utf-8")
    metadata = {
        "schema_version": 2, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "slotssm-online", "model": args.model,
        "slot_checkpoint": args.slot_checkpoint, "stage2_checkpoint": args.stage2_checkpoint,
        "stage3_checkpoint": args.stage3_checkpoint, "task_suite": args.task_suite,
        "task_id": args.task_id, "episode_id": args.episode_id,
        "task_language": task.language, "horizon": args.horizon,
        "action_chunk": args.action_chunk, "execute_chunk": args.execute_chunk,
        "online_strategy": "causal rolling-window recomputation",
        "oracle_subgoal_available": False,
        "subgoal_fallback": "no active object subgoal",
        "completed": False, "success": False,
    }
    write_json(metadata_path, metadata)
    policy = OnlineSlotSSMPolicy(args)
    policy.reset()
    frames: list[np.ndarray] = []
    no_op = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    for _ in range(args.wait_steps):
        obs, _, done, _ = env.step(no_op)
        if done:
            break

    success = False
    policy_steps = 0
    try:
        while policy_steps < args.max_steps and not success:
            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            frames.append(image)
            chunk, features = policy.act(image, task.language)
            feature_path = None
            if args.save_features:
                feature_path = features_dir / f"step_{policy_steps:06d}.npz"
                np.savez_compressed(feature_path, **features)
            for chunk_index in range(min(args.execute_chunk, len(chunk))):
                if policy_steps >= args.max_steps:
                    break
                planned = chunk[chunk_index].copy()
                executed = planned.copy()
                before = obs
                obs, reward, done, info = env.step(executed.tolist())
                success = bool(done)
                record = {
                    "timestep": policy_steps, "chunk_index": chunk_index,
                    "history_length": int(features["history_length"]),
                    "planned_action": planned.tolist(), "executed_action": executed.tolist(),
                    "robot_state_before": extract_robot_state(before),
                    "robot_state_after": extract_robot_state(obs),
                    "privileged_object_state_before": extract_privileged_object_state(before),
                    "privileged_object_state_after": extract_privileged_object_state(obs),
                    "reward": float(reward), "done": bool(done), "success": success,
                    "info": {str(k): jsonable(v) for k, v in dict(info).items()},
                    "features_npz": str(feature_path.resolve()) if feature_path else None,
                }
                with steps_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                policy_steps += 1
                if success:
                    break
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, fps=20)
    metadata.update({
        "completed": True, "success": success, "num_policy_steps": policy_steps,
        "video": str(args.output.resolve()), "steps_jsonl": str(steps_path.resolve()),
    })
    write_json(metadata_path, metadata)
    print(f"task={task.language!r} steps={policy_steps} success={success}")
    print(f"video={args.output.resolve()}")
    print(f"rollout={rollout_dir.resolve()}")


if __name__ == "__main__":
    main()
