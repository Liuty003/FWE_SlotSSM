#!/usr/bin/env python3
"""Batch V4.4.2 evaluation with one OFT/adapter load per GPU lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite", default="libero_object")
    parser.add_argument("--task-ids", required=True, help="Comma-separated task ids for this GPU lane.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", default="best.safetensors")
    parser.add_argument(
        "--family-gate-manifest", type=Path,
        help="Optional exhaustive LIBERO-10 route that swaps only the family-specific Gate head.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--resume-threshold", type=float, default=0.5)
    parser.add_argument("--trigger-confirmation-decisions", type=int, default=1)
    parser.add_argument("--release-confirmation-decisions", type=int, default=3)
    parser.add_argument("--minimum-recovery-steps", type=int, default=40)
    parser.add_argument("--maximum-recovery-steps", type=int, default=120)
    parser.add_argument("--cooldown-steps", type=int, default=16)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--max-env-steps", type=int)
    parser.add_argument("--local-translation-limit", type=float, default=0.5)
    parser.add_argument("--local-rotation-limit", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oft-root", type=Path, default=Path("/data/lty/code/openvla-oft-official"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata"))
    parser.add_argument("--libero-root", type=Path, default=Path("/data/lty/code/LIBERO-plus"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_family_gate_bundle(manifest_path: Path | None, config: dict,
                            normalization_key: str, adapter_path: Path):
    if manifest_path is None:
        return None, {}
    from safetensors.torch import load_file

    payload = json.loads(manifest_path.read_text())
    if (payload.get("kind") != "libero10_family_gate_deployment_manifest_v1"
            or payload.get("num_tasks") != 2519):
        raise ValueError("family Gate manifest must be an exhaustive LIBERO-10 deployment route")
    contract = payload.get("model_contract", {})
    if (contract.get("model") != config["model"]
            or contract.get("direct_head_version") != config.get("direct_head_version")
            or contract.get("recovery_output_horizon") != config.get("recovery_output_horizon")
            or contract.get("action_normalization_key") != normalization_key):
        raise ValueError("family Gate model contract differs from the dual-ownership adapter")
    shared_checkpoint = Path(payload["shared_non_gate_checkpoint"])
    if (not shared_checkpoint.is_file()
            or sha256(shared_checkpoint) != payload["shared_non_gate_checkpoint_sha256"]):
        raise ValueError("family Gate shared-representation checkpoint fingerprint mismatch")
    shared_state = load_file(str(shared_checkpoint), device="cpu")
    runtime_state = load_file(str(adapter_path), device="cpu")
    representation_keys = {
        key for key in shared_state
        if not key.startswith("gate.") and not key.startswith("direct_action.")
    }
    incompatible = [
        key for key in representation_keys
        if key not in runtime_state or not shared_state[key].equal(runtime_state[key])
    ]
    if incompatible:
        raise ValueError(
            "dual adapter changed the representation used by family Gates: "
            f"{incompatible[:3]}"
        )
    states = {}
    expected_keys = None
    for family, route in payload["families"].items():
        checkpoint = Path(route["checkpoint"])
        if not checkpoint.is_file() or sha256(checkpoint) != route["checkpoint_sha256"]:
            raise ValueError(f"family Gate checkpoint fingerprint mismatch: {family}")
        full_state = load_file(str(checkpoint), device="cpu")
        gate_state = {
            key.removeprefix("gate."): value.clone()
            for key, value in full_state.items() if key.startswith("gate.")
        }
        if expected_keys is None:
            expected_keys = set(gate_state)
        elif set(gate_state) != expected_keys:
            raise ValueError(f"family Gate keys differ across checkpoints: {family}")
        states[family] = gate_state
    return payload, states


def bound_local_actions(actions, translation_limit: float, rotation_limit: float, np):
    bounded = np.asarray(actions, dtype=np.float32).copy()
    if bounded.ndim != 2 or bounded.shape[1] != 7:
        raise ValueError("local actions must be [H,7]")
    for start, limit in ((0, translation_limit), (3, rotation_limit)):
        vectors = bounded[:, start:start + 3]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors *= np.minimum(1.0, limit / np.maximum(norms, 1e-12))
    return bounded


def main() -> None:
    args = arguments()
    task_ids = [int(value) for value in args.task_ids.split(",") if value.strip()]
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("task ids must be a non-empty unique list")
    if (not 0 <= args.gate_threshold <= 1 or not 0 <= args.resume_threshold <= 1
            or min(args.trigger_confirmation_decisions, args.release_confirmation_decisions,
                   args.minimum_recovery_steps, args.maximum_recovery_steps, args.fps) < 1
            or args.minimum_recovery_steps > args.maximum_recovery_steps
            or args.cooldown_steps < 0 or args.wait_steps < 0
            or min(args.local_translation_limit, args.local_rotation_limit) <= 0):
        raise ValueError("invalid ownership or rollout arguments")
    adapter_path = args.adapter_dir / args.adapter_checkpoint
    required_paths = [args.oft_root, args.checkpoint, args.libero_root,
                      args.adapter_dir / "config.json", adapter_path]
    if args.family_gate_manifest is not None:
        required_paths.append(args.family_gate_manifest)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    sys.path[:0] = [str(args.libero_root), str(args.oft_root), str(ROOT)]

    import imageio.v2 as imageio
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from experiments.robot.libero.libero_utils import get_libero_image
    from experiments.robot.libero.run_libero_eval import TASK_MAX_STEPS, process_action
    import prismatic
    import prismatic.training
    import prismatic.vla
    from recovery.data.v4_geometry import patch_median_points
    from recovery.models.oft_l1_action_head import OFTL1RegressionActionHead
    from recovery.models.v4_relational_recovery import V4ModelConfig
    from recovery.models.v43_local_recovery import RecoveryLatch, RecoveryLatchConfig
    from recovery.models.v44_dual_ownership_recovery import V44DualOwnershipRecovery
    from scripts.precompute_v4_oft_relation_features import (
        dynamic_module, exact_oft_policy_image, load_torch_state, normalized_proprio,
    )

    prismatic.__path__.append(str(args.oft_root / "prismatic"))
    prismatic.training.__path__.append(str(args.oft_root / "prismatic" / "training"))
    prismatic.vla.__path__.append(str(args.oft_root / "prismatic" / "vla"))
    projectors = dynamic_module("v44_batch_projectors", args.oft_root / "prismatic/models/projectors.py")
    device = torch.device("cuda")

    config = json.loads((args.adapter_dir / "config.json").read_text())
    if config.get("kind") != "v44_dual_ownership_recovery_v1":
        raise ValueError("batch runner requires a dual-ownership V4.4.2 checkpoint")
    provenance = config.get("feature_provenance", {})
    expected_contract = {
        "decoder_action_contract": "pre-process_action planned coordinates; executed gripper inverted by (1-g)/2",
        "image_preprocessing": "official OFT JPEG+TF-Lanczos3+sqrt(0.9)-crop_and_resize",
        "decision_grid_contract": "nominal grid from zero; recovery grid restarts at fork_timestep",
        "recovery_target_contract": "successful RECOVERY rows use canonical executed chunks when explicit target is absent",
        "native_input_only": True,
    }
    for key, value in expected_contract.items():
        if provenance.get(key) != value:
            raise ValueError(f"adapter provenance mismatch for {key}")
    action_head_path = args.checkpoint / "action_head--150000_checkpoint.pt"
    projector_path = args.checkpoint / "proprio_projector--150000_checkpoint.pt"
    if provenance.get("action_head_sha256") != sha256(action_head_path):
        raise ValueError("OFT action head does not match adapter provenance")
    if provenance.get("proprio_projector_sha256") != sha256(projector_path):
        raise ValueError("OFT proprio projector does not match adapter provenance")
    normalization_key = provenance.get("action_normalization_key")
    adapter_sha256 = sha256(adapter_path)
    family_gate_payload, family_gate_states = load_family_gate_bundle(
        args.family_gate_manifest, config, normalization_key, adapter_path,
    )

    oft = AutoModelForVision2Seq.from_pretrained(
        str(args.checkpoint), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device).eval()
    oft.vision_backbone.set_num_images_in_input(2)
    oft.norm_stats = json.loads((args.checkpoint / "dataset_statistics.json").read_text())
    if normalization_key not in oft.norm_stats:
        raise KeyError(f"checkpoint has no normalization statistics for {normalization_key}")
    processor = AutoProcessor.from_pretrained(str(args.checkpoint), trust_remote_code=True)
    action_head = OFTL1RegressionActionHead(4096, 4096, 7)
    action_head.load_state_dict(load_torch_state(action_head_path, torch))
    action_head = action_head.to(device=device, dtype=torch.bfloat16).eval()
    proprio_projector = projectors.ProprioProjector(4096, 8)
    proprio_projector.load_state_dict(load_torch_state(projector_path, torch))
    proprio_projector = proprio_projector.to(device=device, dtype=torch.bfloat16).eval()
    adapter = V44DualOwnershipRecovery(
        V4ModelConfig(**config["model"]),
        direct_head_version=config.get("direct_head_version", "pooled_chunk_mlp_v3"),
        recovery_output_horizon=config.get("recovery_output_horizon", 1),
    ).to(device).eval()
    adapter.load_state_dict(load_file(str(adapter_path), device="cuda"), strict=True)

    if family_gate_states:
        expected_gate_keys = set(adapter.gate.state_dict())
        if any(set(state) != expected_gate_keys for state in family_gate_states.values()):
            raise ValueError("family Gate keys differ from the runtime adapter")

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    maximum_steps = args.max_env_steps or TASK_MAX_STEPS[args.task_suite]
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for task_index, task_id in enumerate(task_ids, start=1):
        output = args.output_root / f"task{task_id}-episode{args.episode}"
        result_path = output / "report.json"
        family = None
        family_route = None
        task_gate_threshold = args.gate_threshold
        task_trigger_confirmations = args.trigger_confirmation_decisions
        if family_gate_payload is not None:
            family = family_gate_payload["task_to_family"].get(str(task_id))
            if family is None:
                raise KeyError(f"task {task_id} has no family Gate route")
            family_route = family_gate_payload["families"][family]
            adapter.gate.load_state_dict({
                key: value.to(device) for key, value in family_gate_states[family].items()
            }, strict=True)
            task_gate_threshold = float(family_route["gate_threshold"])
            task_trigger_confirmations = int(family_route["trigger_confirmation_decisions"])
        protocol = "v44_dual_family_gate_v1" if family_route is not None else "v44_dual_uniform_v1"
        run_contract = {
            "protocol": protocol,
            "task_suite": args.task_suite,
            "task_id": task_id,
            "episode": args.episode,
            "adapter": str(adapter_path.resolve()),
            "adapter_sha256": adapter_sha256,
            "family": family,
            "family_gate_checkpoint_sha256": (
                None if family_route is None else family_route["checkpoint_sha256"]
            ),
            "gate_threshold": task_gate_threshold,
            "resume_threshold": args.resume_threshold,
            "trigger_confirmation_decisions": task_trigger_confirmations,
            "release_confirmation_decisions": args.release_confirmation_decisions,
            "minimum_recovery_steps": args.minimum_recovery_steps,
            "maximum_recovery_steps": args.maximum_recovery_steps,
            "cooldown_steps": args.cooldown_steps,
            "wait_steps": args.wait_steps,
            "max_env_steps": maximum_steps,
            "local_translation_limit": args.local_translation_limit,
            "local_rotation_limit": args.local_rotation_limit,
            "normalization_key": normalization_key,
            "oft_checkpoint": str(args.checkpoint.resolve()),
        }
        run_fingerprint = fingerprint(run_contract)
        if result_path.is_file():
            prior = json.loads(result_path.read_text())
            if prior.get("complete") is True and prior.get("run_fingerprint") == run_fingerprint:
                records.append(prior)
                print(json.dumps({"task_id": task_id, "status": "skipped_complete",
                                  "success": prior["success"], "completed": task_index,
                                  "total": len(task_ids)}), flush=True)
                continue
        task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        if not 0 <= args.episode < len(states):
            raise ValueError(f"episode {args.episode} is invalid for task {task_id}")
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        output.mkdir(parents=True, exist_ok=True)
        try:
            env.seed(0)
            env.reset()
            observation = env.set_init_state(states[args.episode])
            for _ in range(args.wait_steps):
                observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"
            latch = RecoveryLatch(RecoveryLatchConfig(
                trigger_threshold=task_gate_threshold,
                recovery_horizon_steps=args.maximum_recovery_steps,
                minimum_recovery_steps=args.minimum_recovery_steps,
                release_threshold=args.resume_threshold,
                cooldown_steps=args.cooldown_steps,
                trigger_confirmation_decisions=task_trigger_confirmations,
                release_confirmation_decisions=args.release_confirmation_decisions,
            ))
            queue, frames, events = deque(), [], []
            env_steps = 0
            while env_steps < maximum_steps:
                if not queue:
                    main_image = get_libero_image(observation)
                    wrist_image = observation["robot0_eye_in_hand_image"][::-1, ::-1]
                    inputs = processor(prompt, exact_oft_policy_image(main_image, Image, tf)).to(
                        device, dtype=torch.bfloat16,
                    )
                    wrist = processor(prompt, exact_oft_policy_image(wrist_image, Image, tf)).to(
                        device, dtype=torch.bfloat16,
                    )
                    inputs["pixel_values"] = torch.cat((inputs["pixel_values"], wrist["pixel_values"]), dim=1)
                    proprio = normalized_proprio({
                        "robot0_eef_pos": observation["robot0_eef_pos"],
                        "robot0_eef_quat": observation["robot0_eef_quat"],
                        "robot0_gripper_qpos": observation["robot0_gripper_qpos"],
                    }, oft.norm_stats[normalization_key]["proprio"], np)
                    with torch.inference_mode():
                        _, nominal_hidden = oft.predict_action(
                            **inputs, unnorm_key=normalization_key, do_sample=False,
                            proprio=proprio, proprio_projector=proprio_projector,
                            action_head=action_head, use_film=False,
                        )
                        nominal_normalized = action_head.predict_action(
                            nominal_hidden.to(torch.bfloat16),
                        ).float().cpu().numpy()
                        visual = oft.vision_backbone(inputs["pixel_values"])[..., :256, :].float()
                        points, valid, image_xy = patch_median_points(
                            np.zeros((256, 256, 3), dtype=np.float32),
                            np.ones((256, 256), dtype=bool),
                        )
                        prediction = adapter(
                            visual_tokens=visual, action_hidden=nominal_hidden.float(),
                            token_points=torch.as_tensor(points[None], device=device),
                            token_valid=torch.as_tensor(valid[None], device=device),
                            token_image_xy=torch.as_tensor(image_xy[None], device=device),
                            gripper_position=torch.zeros((1, 3), device=device),
                            gripper_image_xy=torch.zeros((1, 2), device=device),
                        )
                    gate_probability = float(prediction["gate_probability"][0])
                    resume_probability = float(prediction["resume_probability"][0])
                    activations_before, handoffs_before = latch.activation_count, latch.handoff_count
                    recovering = latch.select(
                        gate_probability, completion_probability=resume_probability,
                    )
                    if recovering:
                        normalized = prediction["recovery_actions"].float().cpu().numpy()
                        raw = oft._unnormalize_actions(normalized, normalization_key)[0]
                        raw = bound_local_actions(
                            raw, args.local_translation_limit, args.local_rotation_limit, np,
                        )
                        execute_count = min(len(raw), latch.executable_steps(1))
                    else:
                        normalized = nominal_normalized
                        raw = oft._unnormalize_actions(normalized, normalization_key)[0]
                        execute_count = len(raw)
                    queue.extend(raw[:execute_count])
                    events.append({
                        "decision": len(events), "env_step": env_steps,
                        "gate_probability": gate_probability,
                        "resume_probability": resume_probability,
                        "recovering": recovering,
                        "new_recovery_activation": latch.activation_count > activations_before,
                        "new_handoff": latch.handoff_count > handoffs_before,
                        "recovery_activation_count": latch.activation_count,
                        "handoff_count": latch.handoff_count,
                        "executed_chunk_steps": execute_count,
                        "nominal_bypass_exact": not recovering and np.array_equal(normalized, nominal_normalized),
                        "first_planned_action": np.asarray(raw[0], np.float32).tolist(),
                    })
                if args.save_video:
                    frames.append(get_libero_image(observation))
                action = process_action(np.asarray(queue.popleft(), np.float32), "openvla")
                observation, _, done, _ = env.step(action.tolist())
                latch.advance(1)
                env_steps += 1
                if done or env.check_success():
                    break
            video = output / "hybrid.mp4"
            if args.save_video and frames:
                imageio.mimsave(video, frames, fps=args.fps)
            result = {
                "kind": "v44_dual_batch_rollout_v1", "protocol": protocol,
                "run_fingerprint": run_fingerprint, "run_contract": run_contract,
                "complete": True, "success": bool(env.check_success()),
                "task_suite": args.task_suite, "task_id": task_id, "episode": args.episode,
                "family_name": family,
                "language": task.language, "env_steps": env_steps,
                "policy_decisions": len(events), "recovery_activations": latch.activation_count,
                "adaptive_handoffs": latch.handoff_count,
                "triggered_decisions": sum(event["recovering"] for event in events),
                "normalization_key": normalization_key,
                "native_input_only": True,
                "uniform_ownership": {
                    "gate_threshold": task_gate_threshold,
                    "resume_threshold": args.resume_threshold,
                    "trigger_confirmation_decisions": task_trigger_confirmations,
                    "release_confirmation_decisions": args.release_confirmation_decisions,
                    "minimum_recovery_steps": args.minimum_recovery_steps,
                    "maximum_recovery_steps": args.maximum_recovery_steps,
                    "cooldown_steps": args.cooldown_steps,
                },
                "adapter": str(adapter_path.resolve()),
                "family_gate_checkpoint": None if family_route is None else family_route["checkpoint"],
                "video": str(video.resolve()) if args.save_video else None,
                "events": events,
            }
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            records.append(result)
            print(json.dumps({
                "task_id": task_id, "status": "evaluated", "success": result["success"],
                "env_steps": env_steps, "activations": latch.activation_count,
                "handoffs": latch.handoff_count, "completed": task_index, "total": len(task_ids),
            }), flush=True)
        finally:
            env.close()

    aggregate = {
        "kind": "v44_dual_batch_lane_report_v1", "complete": True,
        "task_suite": args.task_suite, "gpu_lane": args.gpu,
        "num_tasks": len(records), "successes": sum(record["success"] for record in records),
        "adapter": str(adapter_path.resolve()),
        "family_gate_manifest": (None if args.family_gate_manifest is None
                                 else str(args.family_gate_manifest.resolve())),
        "records": records,
    }
    (args.output_root / f"lane-gpu{args.gpu}.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps({"num_tasks": len(records), "successes": aggregate["successes"]}), flush=True)


if __name__ == "__main__":
    main()
