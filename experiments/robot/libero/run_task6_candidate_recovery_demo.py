#!/usr/bin/env python
"""Replay task 6 to a natural stall, select a macro, then resume OpenVLA."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_ROOT = REPO_ROOT / "experiments" / "robot" / "libero"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LIBERO_ROOT))
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", str(LIBERO_ROOT / "baseline_libero_config")
)

from run_openvla_baseline import load_openvla, predict_action  # noqa: E402
from run_task6_oracle_recovery import PUDDING_SUBGOAL  # noqa: E402
from scripts.collect_task6_failure_candidate_branches import (  # noqa: E402
    action,
    candidate_macros,
)
from scripts.train_task6_execution_mamba import ExecutionMamba  # noqa: E402
from scripts.train_task6_macro_candidate_scorer import (  # noqa: E402
    MacroCandidateScorer,
)

TASK_SUITE = "libero_10"
TASK_ID = 6
MUG_SUBGOAL = "put the white mug on the plate"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/data2/lty/models/openvla-7b-finetuned-libero-10",
    )
    parser.add_argument("--source-rollout", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--execution-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, default=6)
    parser.add_argument("--failure-timestep", type=int, default=211)
    parser.add_argument("--wait-steps", type=int, default=20)
    parser.add_argument("--max-resume-steps", type=int, default=220)
    parser.add_argument(
        "--resume-mode",
        choices=("pudding", "mug-then-pudding", "full-then-pudding"),
        default="pudding",
    )
    parser.add_argument("--stage-confirm-steps", type=int, default=5)
    parser.add_argument("--device-map", default="balanced_low_0")
    parser.add_argument("--gpu-memory-gib", type=int, default=20)
    parser.add_argument(
        "--center-crop", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def annotate(image, text, color):
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 19), fill=(0, 0, 0))
    draw.text((5, 4), text, fill=color)
    return np.asarray(canvas)


@torch.inference_mode()
def select_candidate(args, device):
    sequence = load_file(
        args.sequence_dir
        / f"task6-episode{args.episode_id}-execution-roles.safetensors"
    )
    execution = ExecutionMamba().to(device).eval()
    execution.load_state_dict(load_file(args.execution_checkpoint))
    _, _, hidden = execution(
        sequence["role_slots"][None, :args.failure_timestep + 1].to(device),
        sequence["role_geometry"][None, :args.failure_timestep + 1].to(device),
        sequence["actions"][None, :args.failure_timestep + 1].to(device),
        return_hidden=True,
    )
    macros = candidate_macros()
    chunks = torch.tensor(
        np.stack(list(macros.values())), dtype=torch.float32, device=device
    )[None]
    scorer = MacroCandidateScorer(chunk_steps=chunks.shape[2]).to(device).eval()
    scorer.load_state_dict(load_file(args.candidate_checkpoint))
    values = scorer(hidden[:, -1], chunks)[0]
    selected = int(values.argmax())
    return (
        list(macros)[selected],
        list(macros.values())[selected],
        {name: float(value) for name, value in zip(macros, values)},
    )


def main():
    args = arguments()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_name, selected_chunk, predicted_values = select_candidate(
        args, device
    )
    torch.cuda.empty_cache()
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
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    env.seed(0)
    env.reset()
    obs = env.set_init_state(initial_states[args.episode_id])
    rows = {
        int(row["timestep"]): row
        for row in (
            json.loads(line)
            for line in (args.source_rollout / "steps.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    frames = []
    success = False
    stage_streak = 0
    stage_switched = args.resume_mode == "pudding"
    stage_switch_step = None
    try:
        for _ in range(args.wait_steps):
            obs, _, _, _ = env.step(action().tolist())
        for timestep in range(args.failure_timestep):
            image = np.ascontiguousarray(
                obs["agentview_image"][::-1, ::-1]
            )
            frames.append(annotate(
                image, f"REPLAY ORIGINAL  step={timestep}", (170, 170, 170)
            ))
            obs, _, _, _ = env.step(rows[timestep]["executed_action"])
        for index, candidate_action in enumerate(selected_chunk):
            image = np.ascontiguousarray(
                obs["agentview_image"][::-1, ::-1]
            )
            frames.append(annotate(
                image,
                f"RECOVERY: {selected_name}  step={index + 1}/{len(selected_chunk)}",
                (255, 180, 30),
            ))
            obs, _, _, _ = env.step(candidate_action.tolist())
        for index in range(args.max_resume_steps):
            mug_on_plate = bool(env.env._eval_predicate(
                ["on", "porcelain_mug_1", "plate_1"]
            ))
            if not stage_switched:
                stage_streak = stage_streak + 1 if mug_on_plate else 0
                if stage_streak >= args.stage_confirm_steps:
                    stage_switched = True
                    stage_switch_step = index
            instruction = PUDDING_SUBGOAL if stage_switched else (
                task.language
                if args.resume_mode == "full-then-pudding"
                else MUG_SUBGOAL
            )
            image = np.ascontiguousarray(
                obs["agentview_image"][::-1, ::-1]
            )
            frames.append(annotate(
                image,
                f"OPENVLA {'PUDDING' if stage_switched else 'MUG'}  step={index}",
                (40, 210, 100),
            ))
            policy_action = predict_action(
                model, processor, image, instruction, TASK_SUITE,
                args.center_crop, input_device,
            )
            obs, _, _, _ = env.step(policy_action.tolist())
            success = bool(env.check_success())
            if success:
                frames.extend([frames[-1]] * 10)
                break
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, fps=args.fps)
    metadata = {
        "schema_version": 1,
        "episode_id": args.episode_id,
        "failure_timestep": args.failure_timestep,
        "selected_candidate": selected_name,
        "predicted_values": predicted_values,
        "success": success,
        "original_replay_steps": args.failure_timestep,
        "recovery_steps": len(selected_chunk),
        "resume_steps": len(frames) - args.failure_timestep - len(selected_chunk)
                        - (10 if success else 0),
        "resume_mode": args.resume_mode,
        "stage_switch_step": stage_switch_step,
        "video": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
