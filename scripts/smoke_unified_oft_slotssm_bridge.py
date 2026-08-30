#!/usr/bin/env python3
"""Verify the real OpenVLA-OFT -> Stage-1 -> Mamba2 -> action adapter path.

This is intentionally a feature-contract smoke test, not an evaluation runner:
it gathers a short no-op observation history from one LIBERO initial state and
checks that the separately versioned OFT and Embodied-SlotSSM components can be
connected without privileged online input.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFT_ROOT = Path("/data/lty/code/openvla-oft-official")
DEFAULT_OFT_CHECKPOINT = Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-10")
DEFAULT_STAGE1 = Path(
    "/data2/lty/naive-e-slotssm-runs/stage1-adapters/"
    "openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--"
    "full10tasks-h16-sync6-s20k-eb6-no-gpu3--image_aug--multiview1000--"
    "object/pretrained_slots-wo-filters_s16.safetensors"
)
DEFAULT_STAGE2 = Path(
    "/data2/lty/naive-e-slotssm-runs/stage2-full-mamba2-h32-s20k-sync6-no-gpu3/"
    "stage2-step-020000.safetensors"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oft-root", type=Path, default=DEFAULT_OFT_ROOT)
    parser.add_argument("--oft-checkpoint", type=Path, default=DEFAULT_OFT_CHECKPOINT)
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--stage2-checkpoint", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=3)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=1)
    return parser.parse_args()


def dynamic_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_torch_state(path: Path, torch):
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {key.removeprefix("module."): value for key, value in state.items()}


def center_crop_resize(image, image_cls):
    """Match the OFT evaluation's 0.9 center crop at its 224px policy input."""
    image = image_cls.fromarray(image).convert("RGB").resize((224, 224))
    size = int(224 * 0.9)
    offset = (224 - size) // 2
    return image.crop((offset, offset, offset + size, offset + size)).resize((224, 224))


def axis_angle(quaternion, np):
    quaternion = np.asarray(quaternion, dtype=np.float32).copy()
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - quaternion[3] ** 2))
    if np.isclose(denominator, 0.0):
        return np.zeros(3)
    return quaternion[:3] * 2.0 * np.arccos(quaternion[3]) / denominator


def normalized_proprio(obs, stats, np):
    raw = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            axis_angle(obs["robot0_eef_quat"], np),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    )
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)))
    return np.clip(np.where(mask, 2 * (raw - low) / (high - low + 1e-8) - 1, raw), -1.0, 1.0)


def main() -> None:
    args = parse_args()
    for path in (args.oft_root, args.oft_checkpoint, args.stage1_checkpoint, args.stage2_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    # The old project and OFT intentionally carry different ``prismatic``
    # versions. Keep the former for SlotAttention/Stage2 and dynamically load
    # only the OFT action components. This avoids silently using the old 5-step
    # action constants with OFT's 8-step head.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    libero_root = REPO_ROOT / "experiments" / "robot" / "libero"
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(libero_root / "baseline_libero_config"))
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(libero_root))

    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from prismatic.extern.hf.modeling_prismatic import SlotAttention
    import prismatic
    import prismatic.training
    import prismatic.vla

    # Allow OFT remote modeling code to find its two additions without
    # replacing the Stage-1/2 classes already imported above.
    prismatic.__path__.append(str(args.oft_root / "prismatic"))
    prismatic.training.__path__.append(str(args.oft_root / "prismatic" / "training"))
    prismatic.vla.__path__.append(str(args.oft_root / "prismatic" / "vla"))

    from scripts.train_slot_stage2_ssm import Stage2Dynamics
    from recovery.models.openvla_oft_slotssm_adapter import OpenVLAOFTSlotSSMAdapter

    action_components = dynamic_module("oft_action_heads", args.oft_root / "prismatic" / "models" / "action_heads.py")
    projector_components = dynamic_module("oft_projectors", args.oft_root / "prismatic" / "models" / "projectors.py")
    L1RegressionActionHead = action_components.L1RegressionActionHead
    ProprioProjector = projector_components.ProprioProjector

    device = torch.device("cuda")
    model = AutoModelForVision2Seq.from_pretrained(
        str(args.oft_checkpoint), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()
    model.vision_backbone.set_num_images_in_input(2)
    model.norm_stats = json.loads((args.oft_checkpoint / "dataset_statistics.json").read_text())
    norm_key = args.task_suite
    if norm_key not in model.norm_stats:
        norm_key = f"{norm_key}_no_noops"
    if norm_key not in model.norm_stats:
        raise KeyError(f"no OFT normalization stats for {args.task_suite}: {list(model.norm_stats)}")
    processor = AutoProcessor.from_pretrained(str(args.oft_checkpoint), trust_remote_code=True)

    action_head = L1RegressionActionHead(4096, 4096, 7)
    action_head.load_state_dict(load_torch_state(args.oft_checkpoint / "action_head--150000_checkpoint.pt", torch))
    action_head = action_head.to(device=device, dtype=torch.bfloat16).eval()
    proprio_projector = ProprioProjector(4096, 8)
    proprio_projector.load_state_dict(load_torch_state(args.oft_checkpoint / "proprio_projector--150000_checkpoint.pt", torch))
    proprio_projector = proprio_projector.to(device=device, dtype=torch.bfloat16).eval()

    stage1_state = load_file(args.stage1_checkpoint)
    tokenizer_state = {
        key.removeprefix("object_centric_tokenizer."): value
        for key, value in stage1_state.items()
        if key.startswith("object_centric_tokenizer.")
    }
    slot_tokenizer = SlotAttention(n_slots=16, in_dim=2176, feature_dim=512)
    slot_tokenizer.load_state_dict(tokenizer_state, strict=True)
    slot_tokenizer = slot_tokenizer.to(device=device, dtype=torch.bfloat16).eval()

    stage2 = Stage2Dynamics("mamba2", residual_next=True)
    stage2.load_state_dict(load_file(args.stage2_checkpoint), strict=True)
    stage2 = stage2.to(device=device, dtype=torch.bfloat16).eval()

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(initial_states[args.episode_id])

    observations = []
    for _ in range(args.horizon):
        observations.append(obs)
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"
    policy_inputs, patch_sequence = [], []
    with torch.inference_mode():
        for current_obs in observations:
            main = current_obs["agentview_image"][::-1, ::-1]
            wrist = current_obs["robot0_eye_in_hand_image"][::-1, ::-1]
            primary = processor(prompt, center_crop_resize(main, Image)).to(device, dtype=torch.bfloat16)
            wrist_inputs = processor(prompt, center_crop_resize(wrist, Image)).to(device, dtype=torch.bfloat16)
            primary["pixel_values"] = torch.cat((primary["pixel_values"], wrist_inputs["pixel_values"]), dim=1)
            policy_inputs.append(primary)
            patch_sequence.append(model.vision_backbone(primary["pixel_values"]).detach())

        patches = torch.stack(patch_sequence, dim=1)
        slots, past_slots = [], None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for index in range(args.horizon):
                current_slots, _, _ = slot_tokenizer(patches[:, index], past_slots=past_slots)
                past_slots = current_slots[:, None]
                slots.append(current_slots)
            slots = torch.stack(slots, dim=1)
            temporal_state, predicted_next_slots, _ = stage2(slots, patches, compute_window=False)

        final_obs, final_inputs = observations[-1], policy_inputs[-1]
        action, action_hidden = model.predict_action(
            **final_inputs,
            unnorm_key=norm_key,
            do_sample=False,
            proprio=normalized_proprio(final_obs, model.norm_stats[norm_key]["proprio"], np),
            proprio_projector=proprio_projector,
            action_head=action_head,
            use_film=False,
        )
        adapter = OpenVLAOFTSlotSSMAdapter(
            action_head, action_hidden_dim=4096, slot_dim=512, fusion_dim=512, num_heads=8
        ).to(device=device, dtype=torch.bfloat16).eval()
        output = adapter(action_hidden, temporal_state[:, -1])

    env.close()
    report = {
        "task": task.language,
        "norm_key": norm_key,
        "patches": list(patches.shape),
        "stage1_slots": list(slots.shape),
        "slotssm_state": list(temporal_state.shape),
        "predicted_next_slots": list(predicted_next_slots.shape),
        "oft_action_hidden": list(action_hidden.shape),
        "oft_action_chunk": list(action.shape),
        "normal_action_chunk": list(output["normal_actions"].shape),
        "recovery_action_chunk": list(output["recovery_actions"].shape),
        "gate_logits": list(output["gate_logits"].shape),
        "initial_gate": output["gate_logits"].argmax(dim=-1).tolist(),
        "finite": bool(torch.isfinite(temporal_state).all() and torch.isfinite(output["recovery_actions"]).all()),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
