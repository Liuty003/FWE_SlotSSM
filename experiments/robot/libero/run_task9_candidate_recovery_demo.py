#!/usr/bin/env python
"""Replay a natural task-9 failure, regrasp, and resume OpenVLA."""

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
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", str(LIBERO_ROOT / "baseline_libero_config")
)

from recovery.teachers.sim_contact import (  # noqa: E402
    extract_target_gripper_contact,
)
from run_openvla_baseline import load_openvla, predict_action  # noqa: E402
from scripts.collect_task6_failure_candidate_branches import action  # noqa: E402

TASK_SUITE = "libero_10"
TASK_ID = 9
TARGET_KEY = "white_yellow_mug_1_pos"
TARGET_NAME = "white_yellow_mug_1"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/data2/lty/models/openvla-7b-finetuned-libero-10",
    )
    parser.add_argument("--source-rollout", type=Path, required=True)
    parser.add_argument("--candidate-branches", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--detection-timestep", type=int, required=True)
    parser.add_argument("--recovery-timestep", type=int, required=True)
    parser.add_argument("--failure-kind", required=True)
    parser.add_argument("--wait-steps", type=int, default=20)
    parser.add_argument("--max-resume-steps", type=int, default=420)
    parser.add_argument(
        "--resume-mode", choices=("full", "staged"), default="full",
        help=(
            "staged uses an insertion subgoal until the mug is inside, then "
            "switches to microwave closing; this is a development oracle."
        ),
    )
    parser.add_argument(
        "--close-instruction", choices=("subgoal", "full"), default="subgoal"
    )
    parser.add_argument(
        "--insertion-guard", action="store_true",
        help=(
            "Development oracle: if OpenVLA releases before In is true, "
            "retain the grasp and advance into the microwave."
        ),
    )
    parser.add_argument("--insertion-guard-action", type=float, default=.35)
    parser.add_argument("--insertion-guard-max-steps", type=int, default=24)
    parser.add_argument(
        "--insertion-handoff", action="store_true",
        help="After In becomes true, release the mug and retract before closing.",
    )
    parser.add_argument("--handoff-release-steps", type=int, default=20)
    parser.add_argument("--handoff-settle-steps", type=int, default=5)
    parser.add_argument("--handoff-deepen-steps", type=int, default=8)
    parser.add_argument("--handoff-deepen-action", type=float, default=.25)
    parser.add_argument("--handoff-lift-steps", type=int, default=10)
    parser.add_argument("--handoff-retract-steps", type=int, default=12)
    parser.add_argument("--device-map", default="balanced_low_0")
    parser.add_argument("--gpu-memory-gib", type=int, default=7)
    parser.add_argument(
        "--center-crop", action=argparse.BooleanOptionalAction, default=True
    )
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
    matches = [
        row for row in payload["candidates"]
        if row["candidate_name"] == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one candidate {name!r}, found {len(matches)}")
    return np.asarray(matches[0]["action_chunk"], np.float32), payload


def main():
    args = arguments()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    if args.detection_timestep > args.recovery_timestep:
        raise ValueError("detection must not occur after recovery")
    chunk, candidate_payload = load_candidate(
        args.candidate_branches, args.candidate_name
    )
    if int(candidate_payload["episode_id"]) != args.episode_id:
        raise ValueError("candidate episode does not match requested episode")
    if int(candidate_payload["failure_timestep"]) != args.recovery_timestep:
        raise ValueError("candidate state does not match recovery timestep")
    rows = {
        int(row["timestep"]): row
        for row in (
            json.loads(line)
            for line in (args.source_rollout / "steps.jsonl").read_text().splitlines()
            if line.strip()
        )
    }

    model, processor, input_device = load_openvla(
        args.checkpoint, args.device_map, args.gpu_memory_gib
    )
    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(TASK_ID)
    initial_states = suite.get_task_init_states(TASK_ID)
    bddl = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(0)
    env.reset()
    obs = env.set_init_state(initial_states[args.episode_id])

    frames = []
    success = False
    resume_steps = 0
    insertion_step = None
    insertion_guard_triggered = False
    insertion_guard_steps = 0
    insertion_handoff_done = False
    insertion_handoff_steps = 0
    try:
        for _ in range(args.wait_steps):
            obs, _, _, _ = env.step(action().tolist())
        for timestep in range(args.recovery_timestep):
            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            if timestep < args.detection_timestep:
                label = f"OPENVLA NATURAL EXECUTION  step={timestep}"
                color = (180, 180, 180)
            else:
                label = (
                    f"SLOTSSM: {args.failure_kind} DETECTED  "
                    f"observe {timestep - args.detection_timestep + 1}"
                )
                color = (255, 95, 45)
            frames.append(annotate(image, label, color))
            obs, _, _, _ = env.step(rows[timestep]["executed_action"])

        recovery_start_position = np.asarray(obs[TARGET_KEY], np.float32)
        _, attached_before = extract_target_gripper_contact(
            env.env.sim, target_name=TARGET_NAME
        )
        for index, candidate_action in enumerate(chunk):
            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            frames.append(annotate(
                image,
                f"SLOT CANDIDATE {args.candidate_name}  "
                f"{index + 1}/{len(chunk)}",
                (255, 185, 35),
            ))
            obs, _, _, _ = env.step(candidate_action.tolist())

        recovery_end_position = np.asarray(obs[TARGET_KEY], np.float32)
        _, attached_after = extract_target_gripper_contact(
            env.env.sim, target_name=TARGET_NAME
        )
        recovery_lift = float(
            recovery_end_position[2] - recovery_start_position[2]
        )
        recovery_verified = bool(attached_after and recovery_lift > 0.015)

        for index in range(args.max_resume_steps):
            mug_inside_now = bool(env.env._eval_predicate(
                ["in", TARGET_NAME, "microwave_1_heating_region"]
            ))
            if mug_inside_now and insertion_step is None:
                insertion_step = index
            if (
                args.insertion_handoff
                and mug_inside_now
                and not insertion_handoff_done
            ):
                handoff = (
                    [action(
                        y=args.handoff_deepen_action, gripper=1.
                    )] * args.handoff_deepen_steps
                    + [action(gripper=-1.)] * args.handoff_release_steps
                    + [action(gripper=-1.)] * args.handoff_settle_steps
                    + [action(z=.35, gripper=-1.)]
                    * args.handoff_lift_steps
                    + [action(y=-.4, gripper=-1.)]
                    * args.handoff_retract_steps
                )
                for handoff_index, handoff_action in enumerate(handoff):
                    handoff_image = np.ascontiguousarray(
                        obs["agentview_image"][::-1, ::-1]
                    )
                    frames.append(annotate(
                        handoff_image,
                        "INSERT VERIFIED: RELEASE + RETRACT "
                        f"{handoff_index + 1}/{len(handoff)}",
                        (80, 195, 255),
                    ))
                    obs, _, _, _ = env.step(handoff_action.tolist())
                    insertion_handoff_steps += 1
                insertion_handoff_done = True
                mug_inside_now = bool(env.env._eval_predicate(
                    ["in", TARGET_NAME, "microwave_1_heating_region"]
                ))
            instruction = (
                task.language
                if args.resume_mode == "full"
                else (
                    (
                        task.language
                        if args.close_instruction == "full"
                        else "close the microwave"
                    )
                    if mug_inside_now
                    else "put the yellow and white mug inside the microwave"
                )
            )
            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            frames.append(annotate(
                image,
                (
                    f"OPENVLA {'CLOSE' if mug_inside_now else 'INSERT'} "
                    f"AFTER VERIFIED REGRASP  step={index}"
                    if recovery_verified
                    else f"OPENVLA RESUMED; REGRASP UNVERIFIED  step={index}"
                ),
                (45, 215, 100) if recovery_verified else (255, 80, 80),
            ))
            policy_action = predict_action(
                model, processor, image, instruction, TASK_SUITE,
                args.center_crop, input_device,
            )
            if (
                args.insertion_guard
                and not mug_inside_now
                and (
                    insertion_guard_triggered
                    or float(policy_action[6]) < -.5
                )
                and insertion_guard_steps < args.insertion_guard_max_steps
            ):
                insertion_guard_triggered = True
                policy_action = action(
                    y=args.insertion_guard_action, gripper=1.
                )
                frames[-1] = annotate(
                    image,
                    "SLOTSSM: INSERTION SHORTFALL; HOLD + ADVANCE "
                    f"{insertion_guard_steps + 1}",
                    (255, 155, 30),
                )
                insertion_guard_steps += 1
            obs, _, _, _ = env.step(policy_action.tolist())
            resume_steps = index + 1
            success = bool(env.check_success())
            if success:
                final = np.ascontiguousarray(
                    obs["agentview_image"][::-1, ::-1]
                )
                frames.extend([
                    annotate(final, "TASK SUCCESS: MUG IN MICROWAVE + DOOR CLOSED",
                             (35, 235, 90))
                ] * 12)
                break
        mug_inside = bool(env.env._eval_predicate(
            ["in", TARGET_NAME, "microwave_1_heating_region"]
        ))
        microwave_closed = bool(env.env._eval_predicate(
            ["close", "microwave_1"]
        ))
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, fps=args.fps)
    metadata = {
        "schema_version": 1,
        "task_suite": TASK_SUITE,
        "task_id": TASK_ID,
        "episode_id": args.episode_id,
        "failure_kind": args.failure_kind,
        "detection_timestep": args.detection_timestep,
        "recovery_timestep": args.recovery_timestep,
        "trigger_source": "offline_teacher_development",
        "candidate_name": args.candidate_name,
        "candidate_selection_source": "forced_development_candidate",
        "candidate_geometry_source": candidate_payload["before"][
            "candidate_geometry_source"
        ],
        "attached_before_recovery": bool(attached_before),
        "attached_after_recovery": bool(attached_after),
        "recovery_lift_m": recovery_lift,
        "recovery_verified": recovery_verified,
        "resume_mode": args.resume_mode,
        "close_instruction": args.close_instruction,
        "resume_steps": resume_steps,
        "insertion_step": insertion_step,
        "insertion_guard_enabled": args.insertion_guard,
        "insertion_guard_triggered": insertion_guard_triggered,
        "insertion_guard_steps": insertion_guard_steps,
        "insertion_handoff_enabled": args.insertion_handoff,
        "insertion_handoff_done": insertion_handoff_done,
        "insertion_handoff_steps": insertion_handoff_steps,
        "mug_inside_after": mug_inside,
        "microwave_closed_after": microwave_closed,
        "success": success,
        "video": str(args.output),
    }
    args.output.with_name("metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
