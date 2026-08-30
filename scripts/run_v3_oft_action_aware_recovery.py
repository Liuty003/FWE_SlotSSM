#!/usr/bin/env python3
"""Run v3 frozen-OFT + action-aware SlotSSM recovery in LIBERO.

This runner intentionally has no candidate verifier, geometry controller, FSM,
object pose, contact, or simulator-state input.  At each policy decision it
chooses between the frozen OFT chunk and the v3 recovery chunk using only the
learned binary gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOCAL_LIBERO_ROOT = ROOT / "experiments" / "robot" / "libero"

from recovery.models.v3_action_aware_recovery import V3ActionAwareRecoveryPolicy, V3ModelConfig
from scripts.smoke_unified_oft_slotssm_bridge import (
    DEFAULT_OFT_ROOT, center_crop_resize, dynamic_module, load_torch_state, normalized_proprio,
)


DEFAULT_OFT = Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--oft-root", type=Path, default=DEFAULT_OFT_ROOT)
    parser.add_argument(
        "--libero-root", type=Path, default=LOCAL_LIBERO_ROOT,
        help="LIBERO or LIBERO-Plus checkout.  LIBERO-Plus is required for task ids 491+.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OFT)
    parser.add_argument("--v3-checkpoint", type=Path, required=True,
                        help="deploy_best.safetensors; auxiliary future/decoder keys are absent by design.")
    parser.add_argument("--v3-config", type=Path,
                        help="Defaults to config.json beside --v3-checkpoint.")
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--max-policy-decisions", type=int, default=65)
    parser.add_argument("--save-online-images", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def config_from_json(path: Path) -> V3ModelConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    model = raw.get("model", raw)
    return V3ModelConfig(**model)


def normalize_env_action(action: np.ndarray, stats: dict, np_module) -> np.ndarray:
    low = np_module.asarray(stats["q01"], dtype=np_module.float32)
    high = np_module.asarray(stats["q99"], dtype=np_module.float32)
    mask = np_module.asarray(stats.get("mask", np_module.ones_like(low, dtype=bool)), dtype=bool)
    return np_module.clip(np_module.where(mask, 2 * (action - low) / (high - low + 1e-8) - 1, action), -1, 1)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.gate_threshold <= 1.0 or args.max_policy_decisions < 1:
        raise ValueError("gate-threshold must be in [0,1] and max-policy-decisions must be positive")
    config_path = args.v3_config or args.v3_checkpoint.parent / "config.json"
    for path in (args.oft_root, args.libero_root / "libero" / "libero", args.checkpoint, args.v3_checkpoint, config_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    # Resolve ``libero.libero`` from Plus before OFT's vendored 10-task copy.
    # Without this order, a Plus task id such as 501 is rejected while the
    # import misleadingly appears valid.
    sys.path[:0] = [str(args.libero_root), str(args.oft_root)]
    os.chdir(args.oft_root)

    import imageio.v2 as imageio
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoModelForVision2Seq, AutoProcessor
    import prismatic
    import prismatic.training
    import prismatic.vla
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_env

    device = torch.device("cuda")
    prismatic.__path__.append(str(args.oft_root / "prismatic"))
    prismatic.training.__path__.append(str(args.oft_root / "prismatic" / "training"))
    prismatic.vla.__path__.append(str(args.oft_root / "prismatic" / "vla"))
    actions_module = dynamic_module("oft_v3_online_actions", args.oft_root / "prismatic" / "models" / "action_heads.py")
    projectors_module = dynamic_module("oft_v3_online_projectors", args.oft_root / "prismatic" / "models" / "projectors.py")
    model = AutoModelForVision2Seq.from_pretrained(
        str(args.checkpoint), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device).eval()
    model.vision_backbone.set_num_images_in_input(2)
    model.norm_stats = json.loads((args.checkpoint / "dataset_statistics.json").read_text())
    norm_key = args.task_suite if args.task_suite in model.norm_stats else f"{args.task_suite}_no_noops"
    if norm_key not in model.norm_stats:
        raise KeyError(f"OFT checkpoint has no statistics for {args.task_suite}")
    processor = AutoProcessor.from_pretrained(str(args.checkpoint), trust_remote_code=True)
    action_head = actions_module.L1RegressionActionHead(4096, 4096, 7)
    action_head.load_state_dict(load_torch_state(args.checkpoint / "action_head--150000_checkpoint.pt", torch))
    action_head = action_head.to(device=device, dtype=torch.bfloat16).eval()
    proprio_projector = projectors_module.ProprioProjector(4096, 8)
    proprio_projector.load_state_dict(load_torch_state(args.checkpoint / "proprio_projector--150000_checkpoint.pt", torch))
    proprio_projector = proprio_projector.to(device=device, dtype=torch.bfloat16).eval()
    config = config_from_json(config_path)
    if config.action_chunk != 8 or config.action_dim != 7:
        raise ValueError("LIBERO OFT runner currently requires 8x7 v3 action chunks")
    # CUDA's fused GRUCell does not implement bfloat16 on this runtime.  OFT
    # remains bfloat16; the lightweight v3 state machine deliberately runs in
    # float32 at the boundary below.
    v3 = V3ActionAwareRecoveryPolicy(config).to(device=device, dtype=torch.float32)
    loaded = v3.load_state_dict(load_file(str(args.v3_checkpoint)), strict=False)
    core_missing = [key for key in loaded.missing_keys
                    if not key.startswith(("future_predictor.", "slot_decoder."))]
    if core_missing or loaded.unexpected_keys:
        raise RuntimeError(
            f"invalid v3 deployment checkpoint; missing_core={core_missing}, "
            f"unexpected={loaded.unexpected_keys}"
        )
    v3.eval()

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.episode_id < len(initial_states):
        raise ValueError("invalid episode-id")
    env, language = get_libero_env(task, "openvla", resolution=256)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir, wrist_dir = args.output_dir / "rgb", args.output_dir / "wrist_rgb"
    if args.save_online_images:
        rgb_dir.mkdir(exist_ok=True); wrist_dir.mkdir(exist_ok=True)

    def inputs_from_obs(obs):
        prompt = f"In: What action should the robot take to {language.lower()}?\nOut:"
        primary = processor(prompt, center_crop_resize(obs["agentview_image"][::-1, ::-1], Image)).to(device, dtype=torch.bfloat16)
        wrist = processor(prompt, center_crop_resize(obs["robot0_eye_in_hand_image"][::-1, ::-1], Image)).to(device, dtype=torch.bfloat16)
        primary["pixel_values"] = torch.cat((primary["pixel_values"], wrist["pixel_values"]), dim=1)
        return primary

    frames, records, decision_records = [], [], []
    memory = previous_slots = None
    previous_chunk = torch.zeros(1, config.action_chunk, config.action_dim, device=device, dtype=torch.float32)
    success = False
    try:
        obs = env.reset()
        obs = env.set_init_state(initial_states[args.episode_id])
        for _ in range(10):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        for decision in range(args.max_policy_decisions):
            frames.append(obs["agentview_image"][::-1, ::-1])
            raw_proprio = {"robot0_eef_pos": obs["robot0_eef_pos"], "robot0_eef_quat": obs["robot0_eef_quat"],
                            "robot0_gripper_qpos": obs["robot0_gripper_qpos"]}
            normalized_proprio_value = normalized_proprio(raw_proprio, model.norm_stats[norm_key]["proprio"], np)
            inputs = inputs_from_obs(obs)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                visual_tokens = model.vision_backbone(inputs["pixel_values"])
                normal_actions, action_hidden = model.predict_action(
                    **inputs, unnorm_key=norm_key, do_sample=False, proprio=normalized_proprio_value,
                    proprio_projector=proprio_projector, action_head=action_head, use_film=False,
                )
            # Keep the recurrent v3 path *outside* OFT's bfloat16 autocast.
            # GRUCell has no bfloat16 CUDA implementation in this runtime.
            with torch.inference_mode():
                output = v3.step(
                    visual_tokens=visual_tokens.float(), action_hidden=action_hidden.float(),
                    proprio=torch.as_tensor(normalized_proprio_value[None], device=device, dtype=torch.float32),
                    previous_action=previous_chunk, previous_memory=memory, previous_slots=previous_slots,
                )
                recovery_probability = float(torch.sigmoid(output["gate_logit"])[0].float().cpu())
                use_recovery = recovery_probability >= args.gate_threshold
                if use_recovery:
                    raw_chunk = model._unnormalize_actions(output["recovery_actions"].float().cpu().numpy(), norm_key)[0]
                else:
                    raw_chunk = np.asarray(normal_actions).copy()
                memory, previous_slots = output["slots"], output["slots"]
            # Keep decision diagnostics separate from replayable action rows:
            # steps.jsonl remains consumable by the feature pipeline, while
            # this trace makes a missed/early online handoff auditable.
            decision_records.append({
                "kind": "v3_decision", "decision": decision,
                "gate_probability": recovery_probability,
                "mode": "RECOVERY" if use_recovery else "NORMAL",
            })
            executed_normalized = []
            for offset, raw in enumerate(raw_chunk):
                frame_index = decision * config.action_chunk + offset
                if args.save_online_images:
                    rgb_path, wrist_path = rgb_dir / f"step-{frame_index:04d}.png", wrist_dir / f"step-{frame_index:04d}.png"
                    imageio.imwrite(rgb_path, obs["agentview_image"][::-1, ::-1])
                    imageio.imwrite(wrist_path, obs["robot0_eye_in_hand_image"][::-1, ::-1])
                env_action = np.asarray(raw).copy()
                env_action[-1] = -np.sign(2.0 * env_action[-1] - 1.0)
                if args.save_online_images:
                    records.append({"online_observation": {"rgb": str(rgb_path.resolve()), "wrist_rgb": str(wrist_path.resolve()),
                                    "proprio": {key: np.asarray(value).tolist() for key, value in raw_proprio.items()}},
                                    "executed_action": env_action.tolist(), "gate_label": "RECOVERY" if use_recovery else "NORMAL"})
                executed_normalized.append(normalize_env_action(env_action, model.norm_stats[norm_key]["action"], np))
                obs, _, done, _ = env.step(env_action.tolist())
                frames.append(obs["agentview_image"][::-1, ::-1])
                if done:
                    # ``done`` may also mean an environment horizon.  Report
                    # task success only from the benchmark predicate.
                    success = bool(env.check_success())
                    break
            previous_chunk = torch.as_tensor(np.stack(executed_normalized), device=device, dtype=torch.float32)[None]
            if previous_chunk.shape[1] < config.action_chunk:
                previous_chunk = torch.cat((previous_chunk, previous_chunk[:, -1:].expand(-1, config.action_chunk - previous_chunk.shape[1], -1)), dim=1)
            if done:
                break
    finally:
        env.close()
    stem = f"task{args.task_id}-episode{args.episode_id}"
    video_path = args.output_dir / f"{stem}.mp4"
    imageio.mimsave(video_path, frames, fps=30)
    if args.save_online_images:
        (args.output_dir / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records if "online_observation" in row))
    (args.output_dir / "decisions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in decision_records)
    )
    probabilities = [row["gate_probability"] for row in decision_records]
    report = {"kind": "v3_frozen_oft_action_aware_recovery", "success": success, "task": language,
              "frames": len(frames), "video": str(video_path), "gate_threshold": args.gate_threshold,
              "decisions": len(decision_records),
              "recovery_decisions": sum(row["mode"] == "RECOVERY" for row in decision_records),
              "gate_probability": ({"min": min(probabilities), "mean": sum(probabilities) / len(probabilities),
                                    "max": max(probabilities)} if probabilities else None),
              "deployment_contract": ["RGB", "wrist_RGB", "language", "proprio", "previous_executed_action"],
              "forbidden_online_inputs": ["FSM", "candidate_verifier", "object_pose", "contact", "simulator_state"]}
    (args.output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
