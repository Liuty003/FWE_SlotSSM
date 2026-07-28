#!/usr/bin/env python
"""Collect matched LIBERO transitions from identical simulator states.

The main rollout follows the successful original OpenVLA. At selected states we
restore the exact MuJoCo state and execute multiple action interventions, giving
matched (state, action, next-state) counterfactual supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_ROOT = REPO_ROOT / "experiments" / "robot" / "libero"
sys.path.insert(0, str(LIBERO_ROOT))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_ROOT / "baseline_libero_config"))

from run_openvla_baseline import (
    SUITE_MAX_STEPS,
    center_crop,
    extract_privileged_object_state,
    extract_robot_state,
    jsonable,
    load_openvla,
    write_json,
)


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial")
    p.add_argument("--task-suite", choices=tuple(SUITE_MAX_STEPS), default="libero_spatial")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--episode-id", type=int, default=0)
    p.add_argument("--wait-steps", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=220)
    p.add_argument("--collect-every", type=int, default=10)
    p.add_argument("--max-forks", type=int, default=20)
    p.add_argument("--branch-steps", type=int, default=4)
    p.add_argument("--translation-delta", type=float, default=0.35)
    p.add_argument("--rotation-delta", type=float, default=0.25)
    p.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device-map", default="balanced_low_0")
    p.add_argument("--gpu-memory-gib", type=int, default=7)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def image_from_obs(obs):
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def image_digest(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def predict_openvla(model, processor, image, instruction, suite, crop, input_device):
    if crop:
        image = center_crop(image)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, Image.fromarray(image).convert("RGB")).to(
        input_device, dtype=torch.bfloat16
    )
    with torch.inference_mode():
        physical = np.asarray(
            model.predict_action(**inputs, unnorm_key=suite, do_sample=False), dtype=np.float32
        )
    executed = physical.copy()
    executed[-1] = np.sign(2.0 * executed[-1] - 1.0) * -1.0
    return physical, executed


def intervention_actions(nominal, translation_delta, rotation_delta):
    variants = [("nominal", nominal.copy(), {"kind": "nominal", "dimension": None, "delta": 0.0})]
    labels = ("x", "y", "z")
    for index, label in enumerate(labels):
        for sign in (-1.0, 1.0):
            action = nominal.copy()
            action[index] = np.clip(action[index] + sign * translation_delta, -1.0, 1.0)
            variants.append((
                f"translate_{label}_{'plus' if sign > 0 else 'minus'}", action,
                {"kind": "translation", "dimension": index, "delta": sign * translation_delta},
            ))
    # A matched rotation intervention gives a non-translational action contrast.
    for sign in (-1.0, 1.0):
        action = nominal.copy()
        action[5] = np.clip(action[5] + sign * rotation_delta, -1.0, 1.0)
        variants.append((
            f"rotate_z_{'plus' if sign > 0 else 'minus'}", action,
            {"kind": "rotation", "dimension": 5, "delta": sign * rotation_delta},
        ))
    action = nominal.copy()
    action[-1] *= -1.0
    variants.append(("flip_gripper", action, {"kind": "gripper", "dimension": 6, "delta": None}))
    return variants


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = arguments()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(initial_states[args.episode_id])
    # A dedicated environment absorbs all state restoration and intervention
    # steps. This keeps robosuite observable buffers in the main policy rollout
    # untouched, preserving the original OpenVLA closed loop.
    fork_env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    fork_env.seed(0)
    fork_env.reset()
    fork_env.set_init_state(initial_states[args.episode_id])
    model, processor, input_device = load_openvla(
        args.checkpoint, args.device_map, args.gpu_memory_gib
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    states_dir = args.output_dir / "sim_states"
    images_dir.mkdir(exist_ok=True)
    states_dir.mkdir(exist_ok=True)
    transitions_path = args.output_dir / "transitions.jsonl"
    main_path = args.output_dir / "main_rollout.jsonl"
    transitions_path.write_text("", encoding="utf-8")
    main_path.write_text("", encoding="utf-8")
    metadata_path = args.output_dir / "metadata.json"
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_suite": args.task_suite, "task_id": args.task_id,
        "episode_id": args.episode_id, "task_language": task.language,
        "checkpoint": args.checkpoint, "collect_every": args.collect_every,
        "max_forks": args.max_forks, "translation_delta": args.translation_delta,
        "rotation_delta": args.rotation_delta, "branch_steps": args.branch_steps,
        "completed": False,
        "action_representations": {
            "policy_physical_action": "OpenVLA output after q01/q99 unnormalization; gripper remains model convention",
            "nominal_env_action": "policy action converted to LIBERO gripper convention",
            "executed_env_action": "possibly perturbed action passed to env.step",
        },
    }
    write_json(metadata_path, metadata)

    no_op = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    for _ in range(args.wait_steps):
        obs, _, done, _ = env.step(no_op)
        if done:
            break

    forks = 0
    transitions = 0
    success = False
    episode_terminated = False
    termination_reason = None
    try:
        for timestep in range(args.max_steps):
            should_fork = timestep % args.collect_every == 0 and forks < args.max_forks
            before_image = image_from_obs(obs)
            policy_physical, nominal = predict_openvla(
                model, processor, before_image, task.language, args.task_suite,
                args.center_crop, input_device,
            )
            if should_fork:
                fork_id = forks
                state = env.get_sim_state().copy()
                canonical_obs = fork_env.regenerate_obs_from_state(state)
                canonical_image = image_from_obs(canonical_obs)
                state_path = states_dir / f"fork_{fork_id:06d}.npz"
                np.savez_compressed(state_path, mujoco_state=state)
                before_path = images_dir / f"fork_{fork_id:06d}_before.png"
                policy_before_path = images_dir / f"fork_{fork_id:06d}_policy_before.png"
                imageio.imwrite(before_path, canonical_image)
                imageio.imwrite(policy_before_path, before_image)
                before_hash = image_digest(canonical_image)
                policy_before_hash = image_digest(before_image)
                before_robot = extract_robot_state(canonical_obs)
                before_objects = extract_privileged_object_state(canonical_obs)
                for branch_name, branch_action, intervention in intervention_actions(
                    nominal, args.translation_delta, args.rotation_delta
                ):
                    restored_obs = fork_env.regenerate_obs_from_state(state)
                    # Simulator flat state excludes robosuite episode counters.
                    # Each intervention is an independent short rollout.
                    fork_env.env.timestep = 0
                    fork_env.env.done = False
                    restored_image = image_from_obs(restored_obs)
                    restored_hash = image_digest(restored_image)
                    if restored_hash != before_hash:
                        raise RuntimeError(
                            f"Non-deterministic restore at fork={fork_id} branch={branch_name}: "
                            f"{restored_hash} != {before_hash}"
                        )
                    branch_trajectory = []
                    after_paths = []
                    after_obs = restored_obs
                    reward, done, info = 0.0, False, {}
                    for branch_step in range(args.branch_steps):
                        step_before = after_obs
                        after_obs, reward, done, info = fork_env.step(branch_action.tolist())
                        after_image = image_from_obs(after_obs)
                        after_path = images_dir / (
                            f"fork_{fork_id:06d}_{branch_name}_after_{branch_step + 1:02d}.png"
                        )
                        imageio.imwrite(after_path, after_image)
                        after_paths.append(str(after_path.resolve()))
                        branch_trajectory.append({
                            "branch_step": branch_step,
                            "robot_state_before": extract_robot_state(step_before),
                            "robot_state_after": extract_robot_state(after_obs),
                            "privileged_object_state_before": extract_privileged_object_state(step_before),
                            "privileged_object_state_after": extract_privileged_object_state(after_obs),
                            "reward": float(reward), "done": bool(done),
                        })
                        if done:
                            break
                    append_jsonl(transitions_path, {
                        "fork_id": fork_id, "main_timestep": timestep,
                        "branch": branch_name, "intervention": intervention,
                        "policy_physical_action": policy_physical.tolist(),
                        "nominal_env_action": nominal.tolist(),
                        "executed_env_action": branch_action.astype(np.float32).tolist(),
                        "robot_state_before": before_robot,
                        "robot_state_after": extract_robot_state(after_obs),
                        "privileged_object_state_before": before_objects,
                        "privileged_object_state_after": extract_privileged_object_state(after_obs),
                        "reward": float(reward), "done": bool(done),
                        "info": {str(k): jsonable(v) for k, v in dict(info).items()},
                        "before_image": str(before_path.resolve()),
                        "policy_before_image": str(policy_before_path.resolve()),
                        "after_image": after_paths[-1],
                        "after_images": after_paths,
                        "branch_steps_requested": args.branch_steps,
                        "branch_steps_executed": len(branch_trajectory),
                        "branch_trajectory": branch_trajectory,
                        "before_image_sha256": before_hash,
                        "policy_before_image_sha256": policy_before_hash,
                        "restored_image_sha256": restored_hash,
                        "sim_state": str(state_path.resolve()),
                    })
                    transitions += 1
                forks += 1

            main_before = obs
            try:
                obs, reward, done, info = env.step(nominal.tolist())
            except ValueError as error:
                if "terminated episode" not in str(error):
                    raise
                episode_terminated = True
                termination_reason = str(error)
                break
            success = bool(env.check_success())
            append_jsonl(main_path, {
                "timestep": timestep,
                "policy_physical_action": policy_physical.tolist(),
                "executed_env_action": nominal.tolist(),
                "robot_state_before": extract_robot_state(main_before),
                "robot_state_after": extract_robot_state(obs),
                "reward": float(reward), "done": bool(done), "success": success,
                "info": {str(k): jsonable(v) for k, v in dict(info).items()},
            })
            print(
                f"timestep={timestep} forks={forks} transitions={transitions} success={success}",
                flush=True,
            )
            if success:
                break
            if done:
                episode_terminated = True
                termination_reason = "robosuite horizon reached"
                break
    finally:
        env.close()
        fork_env.close()

    metadata.update({
        "completed": True, "success": success, "num_forks": forks,
        "num_transitions": transitions, "episode_terminated": episode_terminated,
        "termination_reason": termination_reason,
    })
    write_json(metadata_path, metadata)
    print(f"completed forks={forks} transitions={transitions} success={success}")
    print(f"output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
