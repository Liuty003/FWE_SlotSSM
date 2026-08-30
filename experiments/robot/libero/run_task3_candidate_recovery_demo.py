#!/usr/bin/env python
"""Replay a natural task-3 failure, regrasp the bowl, then resume OpenVLA.

This intentionally keeps task-specific articulated-object checks outside the
shared task-9 demo: Task 3's success requires both bowl insertion into the
bottom drawer and closing that drawer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_ROOT = REPO_ROOT / "experiments" / "robot" / "libero"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LIBERO_ROOT))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_ROOT / "baseline_libero_config"))

from recovery.teachers.sim_contact import extract_target_gripper_contact  # noqa: E402
from run_openvla_baseline import load_openvla, predict_action  # noqa: E402
from scripts.collect_task6_failure_candidate_branches import action  # noqa: E402

TASK_SUITE = "libero_10"
TASK_ID = 3
TARGET_KEY = "akita_black_bowl_1_pos"
TARGET_NAME = "akita_black_bowl_1"
TARGET_REGION = "white_cabinet_1_bottom_region"
DRAWER_REGION = "white_cabinet_1_bottom_region"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/data2/lty/models/openvla-7b-finetuned-libero-10")
    parser.add_argument("--source-rollout", type=Path, required=True)
    parser.add_argument("--candidate-branches", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--detection-timestep", type=int, required=True)
    parser.add_argument("--recovery-timestep", type=int, required=True)
    parser.add_argument("--failure-kind", required=True)
    parser.add_argument("--wait-steps", type=int, default=20)
    parser.add_argument("--max-resume-steps", type=int, default=420)
    parser.add_argument("--device-map", default="balanced_low_0")
    parser.add_argument("--gpu-memory-gib", type=int, default=7)
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def annotate(image, text, color):
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 27), fill=(0, 0, 0))
    draw.text((6, 7), text, fill=color)
    return np.asarray(canvas)


def load_candidate(path, name):
    payload = json.loads(path.read_text())
    matches = [row for row in payload["candidates"] if row["candidate_name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one candidate {name!r}, found {len(matches)}")
    return np.asarray(matches[0]["action_chunk"], np.float32), payload


def image_from(obs):
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def main():
    args = arguments()
    if args.detection_timestep > args.recovery_timestep:
        raise ValueError("detection must not occur after recovery")
    chunk, candidate_payload = load_candidate(args.candidate_branches, args.candidate_name)
    if int(candidate_payload["episode_id"]) != args.episode_id:
        raise ValueError("candidate episode does not match requested episode")
    if int(candidate_payload["failure_timestep"]) != args.recovery_timestep:
        raise ValueError("candidate state does not match requested recovery timestep")
    rows = {
        int(row["timestep"]): row
        for row in (json.loads(line) for line in (args.source_rollout / "steps.jsonl").read_text().splitlines() if line.strip())
    }

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    model, processor, input_device = load_openvla(args.checkpoint, args.device_map, args.gpu_memory_gib)
    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(TASK_ID)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(TASK_ID)[args.episode_id])

    frames, success, resume_steps = [], False, 0
    try:
        for _ in range(args.wait_steps):
            obs, _, _, _ = env.step(action().tolist())
        for timestep in range(args.recovery_timestep):
            if timestep < args.detection_timestep:
                label, color = f"OPENVLA NATURAL EXECUTION  step={timestep}", (180, 180, 180)
            else:
                label = f"SLOTSSM: {args.failure_kind} DETECTED  observe {timestep - args.detection_timestep + 1}"
                color = (255, 95, 45)
            frames.append(annotate(image_from(obs), label, color))
            obs, _, _, _ = env.step(rows[timestep]["executed_action"])

        target_before = np.asarray(obs[TARGET_KEY], np.float32)
        _, attached_before = extract_target_gripper_contact(env.env.sim, target_name=TARGET_NAME)
        # Candidate branches are evaluated from a restored simulator state
        # with a fresh OSC goal.  An online recovery must perform the same
        # explicit control hand-off; otherwise stale nominal-policy goals
        # alter the supposedly identical regrasp action chunk.
        controller = env.env.robots[0].controller
        controller.update(force=True)
        controller.reset_goal()
        env.env.timestep = 0
        env.env.done = False
        for index, candidate_action in enumerate(chunk):
            frames.append(annotate(
                image_from(obs), f"SLOT CANDIDATE {args.candidate_name}  {index + 1}/{len(chunk)}", (255, 185, 35)
            ))
            obs, _, _, _ = env.step(candidate_action.tolist())
        target_after = np.asarray(obs[TARGET_KEY], np.float32)
        _, attached_after = extract_target_gripper_contact(env.env.sim, target_name=TARGET_NAME)
        recovery_lift = float(target_after[2] - target_before[2])
        recovery_verified = bool(attached_after and recovery_lift > .015)

        for index in range(args.max_resume_steps):
            frames.append(annotate(
                image_from(obs),
                f"OPENVLA RESUMED AFTER {'VERIFIED' if recovery_verified else 'UNVERIFIED'} BOWL REGRASP  step={index}",
                (45, 215, 100) if recovery_verified else (255, 80, 80),
            ))
            policy_action = predict_action(model, processor, image_from(obs), task.language, TASK_SUITE, args.center_crop, input_device)
            obs, _, _, _ = env.step(policy_action.tolist())
            resume_steps = index + 1
            success = bool(env.check_success())
            if success:
                frames.extend([annotate(image_from(obs), "TASK SUCCESS: BOWL IN DRAWER + DRAWER CLOSED", (35, 235, 90))] * 12)
                break
        bowl_in_drawer = bool(env.env._eval_predicate(["in", TARGET_NAME, TARGET_REGION]))
        drawer_closed = bool(env.env._eval_predicate(["close", DRAWER_REGION]))
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, fps=args.fps)
    metadata = {
        "schema_version": 1, "task_suite": TASK_SUITE, "task_id": TASK_ID,
        "episode_id": args.episode_id, "failure_kind": args.failure_kind,
        "detection_timestep": args.detection_timestep, "recovery_timestep": args.recovery_timestep,
        "trigger_source": "offline_teacher_development", "candidate_name": args.candidate_name,
        "candidate_selection_source": "forced_development_candidate",
        "candidate_geometry_source": candidate_payload["before"]["candidate_geometry_source"],
        "controller_goal_reset_at_recovery": True,
        "attached_before_recovery": bool(attached_before), "attached_after_recovery": bool(attached_after),
        "recovery_lift_m": recovery_lift, "recovery_verified": recovery_verified,
        "resume_steps": resume_steps, "bowl_in_drawer_after": bowl_in_drawer,
        "drawer_closed_after": drawer_closed, "success": success, "video": str(args.output),
    }
    args.output.with_name("metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
