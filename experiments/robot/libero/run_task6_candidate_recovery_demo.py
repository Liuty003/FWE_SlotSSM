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
from torchvision import transforms

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
    cup_place_macros,
    relative_regrasp_macros,
    safe_approach_macros,
)
from scripts.train_task6_execution_mamba import ExecutionMamba  # noqa: E402
from scripts.train_task6_execution_mamba import (  # noqa: E402
    FAILED_ACQUISITION_THRESHOLD,
    MOTION_GATE_CENTER,
    MOTION_GATE_SCALE,
)
from scripts.train_task6_execution_slots import HEADS  # noqa: E402
from scripts.train_task6_execution_slots import Task6ExecutionSlots  # noqa: E402
from scripts.train_task6_macro_candidate_scorer import (  # noqa: E402
    MacroCandidateScorer,
)
from scripts.train_task6_slot_relative_geometry import (  # noqa: E402
    SlotRelativeGeometry,
)
from recovery.teachers.sim_contact import (  # noqa: E402
    extract_target_gripper_contact,
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
    parser.add_argument("--execution-robot-features", action="store_true")
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument(
        "--candidate-macro-kind",
        choices=("short", "cup-place", "relative-regrasp", "safe-approach"),
        default="short",
    )
    parser.add_argument(
        "--forced-candidate",
        help=(
            "Development-only candidate override. This bypasses the learned "
            "candidate scorer and is recorded in output metadata."
        ),
    )
    parser.add_argument(
        "--candidate-name", action="append",
        help="Restrict learned scoring to named candidates (repeatable).",
    )
    parser.add_argument("--episode-id", type=int, default=6)
    parser.add_argument("--failure-timestep", type=int, default=211)
    parser.add_argument(
        "--auto-trigger", action="store_true",
        help="Derive failure timestep causally from Execution Mamba2.",
    )
    parser.add_argument(
        "--auto-geometry-trigger",
        action="store_true",
        help="Trigger from causal drift of observable slot-relative xyz.",
    )
    parser.add_argument(
        "--auto-acquisition-verify-trigger",
        action="store_true",
        help=(
            "Use post-close observable slot lift and relative drift to "
            "distinguish empty acquisition from edge-slip."
        ),
    )
    parser.add_argument("--geometry-checkpoint", type=Path)
    parser.add_argument("--slot-checkpoint", type=Path)
    parser.add_argument(
        "--reobserve-after-retract", type=int, default=0,
        help=(
            "Execute this many candidate prefix steps, re-estimate live slot "
            "geometry, then regenerate the selected relative-regrasp chunk."
        ),
    )
    parser.add_argument("--geometry-drift-threshold", type=float, default=.04)
    parser.add_argument("--geometry-lookback", type=int, default=15)
    parser.add_argument("--trigger-confirm-frames", type=int, default=3)
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
    parser.add_argument(
        "--continuous-recovery",
        action="store_true",
        help=(
            "Execute recovery in the live environment without serializing and "
            "reconstructing MuJoCo/controller state. This is the faithful "
            "online path; state forks remain useful for branch comparisons."
        ),
    )
    parser.add_argument(
        "--save-final-sim-state",
        type=Path,
        help="Optional flattened MuJoCo state for a subsequent recovery fork.",
    )
    parser.add_argument(
        "--second-recovery-candidate",
        choices=tuple(candidate_macros()),
        help="Optional second recovery after an unsuccessful OpenVLA resume.",
    )
    return parser.parse_args()


@torch.inference_mode()
def causal_failure_scores(args, device):
    sequence = load_file(
        args.sequence_dir
        / f"task6-episode{args.episode_id}-execution-roles.safetensors"
    )
    if "acquisition_assessment_active" not in sequence:
        raise ValueError(
            "auto trigger requires acquisition_assessment_active in sequence"
        )
    execution = ExecutionMamba(
        robot_feature_dim=4 if args.execution_robot_features else 0
    ).to(device).eval()
    execution.load_state_dict(load_file(args.execution_checkpoint))
    logits, _ = execution(
        sequence["role_slots"][None].to(device),
        sequence["role_geometry"][None].to(device),
        sequence["actions"][None].to(device),
        (
            sequence["robot_features"][None].to(device)
            if args.execution_robot_features else None
        ),
    )
    column = HEADS.index("failed_acquisition")
    scores = logits[0, :, column].sigmoid().cpu()
    geometry = sequence["role_geometry"][:, 2, :2]
    motion = torch.zeros(len(geometry))
    motion[15:] = (geometry[15:] - geometry[:-15]).norm(dim=-1)
    scores = torch.minimum(
        scores,
        torch.sigmoid(
            (MOTION_GATE_CENTER - motion) * MOTION_GATE_SCALE
        ),
    )
    active = sequence["acquisition_assessment_active"].bool()
    confirm = args.trigger_confirm_frames
    trigger = next((
        timestep
        for timestep in range(confirm - 1, len(scores))
        if active[timestep - confirm + 1:timestep + 1].all()
        and (
            scores[timestep - confirm + 1:timestep + 1]
            >= FAILED_ACQUISITION_THRESHOLD
        ).all()
    ), None)
    if trigger is None:
        raise RuntimeError("Execution Mamba2 did not confirm failed acquisition")
    return trigger, scores.numpy(), active.numpy()


@torch.inference_mode()
def observable_relative_geometry(args, device):
    if args.geometry_checkpoint is None:
        raise ValueError("--geometry-checkpoint is required")
    sequence = load_file(
        args.sequence_dir
        / f"task6-episode{args.episode_id}-execution-roles.safetensors"
    )
    model = SlotRelativeGeometry().to(device).eval()
    model.load_state_dict(load_file(args.geometry_checkpoint))
    relative = model(
        sequence["role_slots"].to(device),
        sequence["role_geometry"].to(device),
        sequence["robot_features"].to(device),
    ).cpu()
    return relative, sequence


@torch.inference_mode()
def causal_geometry_trigger(args, device):
    relative, sequence = observable_relative_geometry(args, device)
    lookback = args.geometry_lookback
    drift = torch.zeros(len(relative))
    drift[lookback:] = torch.linalg.vector_norm(
        relative[lookback:] - relative[:-lookback], dim=-1
    )
    active = sequence["stage2_active"].bool()
    aperture = sequence["robot_features"][:, 3]
    confirm = args.trigger_confirm_frames
    trigger = next((
        timestep
        for timestep in range(lookback + confirm - 1, len(relative))
        if active[timestep - confirm + 1:timestep + 1].all()
        and (aperture[timestep - confirm + 1:timestep + 1] < .025).all()
        and (
            drift[timestep - confirm + 1:timestep + 1]
            >= args.geometry_drift_threshold
        ).all()
    ), None)
    if trigger is None:
        raise RuntimeError("slot-relative geometry did not confirm drift")
    return trigger, relative.numpy(), drift.numpy(), active.numpy()


@torch.inference_mode()
def causal_acquisition_verify_trigger(args, device):
    relative, sequence = observable_relative_geometry(args, device)
    actions = sequence["actions"]
    aperture = sequence["robot_features"][:, 3]
    eef_z = sequence["robot_features"][:, 2]
    active = sequence["stage2_active"].bool()
    attempts = [
        timestep for timestep in range(len(actions))
        if active[timestep]
        and float(actions[timestep, 6]) > .5
        and float(aperture[timestep]) > .03
        and (timestep == 0 or float(actions[timestep - 1, 6]) <= .5)
    ]
    for attempt in attempts:
        assess = attempt + 15
        if assess >= len(relative):
            continue
        lift = float(
            relative[assess, 2] + eef_z[assess]
            - relative[attempt, 2] - eef_z[attempt]
        )
        drift = float(torch.linalg.vector_norm(
            relative[assess] - relative[attempt]
        ))
        if lift < .01 and drift < .025:
            trigger, kind = assess, "empty_acquisition"
            break
        if lift < .01 and drift >= .025:
            trigger, kind = min(attempt + 30, len(relative) - 1), "edge_slip"
            break
    else:
        raise RuntimeError("no failed acquisition was causally confirmed")
    score = torch.zeros(len(relative))
    score[attempt:trigger + 1] = torch.linalg.vector_norm(
        relative[attempt:trigger + 1] - relative[attempt], dim=-1
    )
    assessment = torch.zeros(len(relative), dtype=torch.bool)
    assessment[attempt:trigger + 1] = True
    return (
        trigger, relative.numpy(), score.numpy(), assessment.numpy(),
        kind, attempt, lift, drift,
    )


def annotate(image, text, color):
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 19), fill=(0, 0, 0))
    draw.text((5, 4), text, fill=color)
    return np.asarray(canvas)


@torch.inference_mode()
def live_relative_geometry(
    obs, last_action, slot_model, geometry_model, device
):
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
    tensor = transform(Image.fromarray(image))[None].to(device)
    action_tensor = torch.as_tensor(
        last_action, dtype=torch.float32, device=device
    )[None]
    mask_logits, role_slots, _, _ = slot_model(tensor, action_tensor)
    masks = mask_logits.sigmoid()
    height, width = masks.shape[-2:]
    y = torch.linspace(-1, 1, height, device=device)[None, None, :, None]
    x = torch.linspace(-1, 1, width, device=device)[None, None, None, :]
    mass = masks.sum((2, 3)).clamp_min(1)
    role_geometry = torch.stack((
        (masks * x).sum((2, 3)) / mass,
        (masks * y).sum((2, 3)) / mass,
        mass / (height * width),
    ), dim=-1)
    robot_features = torch.as_tensor([[
        *np.asarray(obs["robot0_eef_pos"], np.float32).tolist(),
        float(np.abs(obs["robot0_gripper_qpos"]).mean()),
    ]], dtype=torch.float32, device=device)
    return geometry_model(
        role_slots, role_geometry, robot_features
    )[0].cpu().numpy()


@torch.inference_mode()
def select_candidate(args, device, relative_prediction=None):
    if args.candidate_macro_kind == "relative-regrasp":
        if relative_prediction is not None:
            relative = relative_prediction[args.failure_timestep]
        else:
            rows = {
                int(row["timestep"]): row
                for row in (
                    json.loads(line)
                    for line in (
                        args.source_rollout / "steps.jsonl"
                    ).read_text().splitlines()
                    if line.strip()
                )
            }
            row = rows[args.failure_timestep]
            target = np.asarray(
                row["privileged_object_state_before"][
                    "chocolate_pudding_1_pos"
                ],
                dtype=np.float32,
            )
            eef = np.asarray(
                row["robot_state_before"]["robot0_eef_pos"], dtype=np.float32
            )
            relative = target - eef
        macros = relative_regrasp_macros(relative)
    else:
        macro_sets = {
            "short": candidate_macros,
            "cup-place": cup_place_macros,
            "safe-approach": safe_approach_macros,
        }
        macros = macro_sets[args.candidate_macro_kind]()
    if args.candidate_name:
        missing = set(args.candidate_name) - set(macros)
        if missing:
            raise ValueError(f"unknown candidates: {sorted(missing)}")
        macros = {name: macros[name] for name in args.candidate_name}
    if args.forced_candidate is not None:
        if args.forced_candidate not in macros:
            raise ValueError(
                f"unknown {args.candidate_macro_kind} candidate "
                f"{args.forced_candidate!r}; choices={tuple(macros)}"
            )
        return args.forced_candidate, macros[args.forced_candidate], None
    if args.candidate_checkpoint is None:
        raise ValueError(
            "--candidate-checkpoint is required without --forced-candidate"
        )
    sequence = load_file(
        args.sequence_dir
        / f"task6-episode{args.episode_id}-execution-roles.safetensors"
    )
    execution = ExecutionMamba(
        robot_feature_dim=4 if args.execution_robot_features else 0
    ).to(device).eval()
    execution.load_state_dict(load_file(args.execution_checkpoint))
    _, _, hidden = execution(
        sequence["role_slots"][None, :args.failure_timestep + 1].to(device),
        sequence["role_geometry"][None, :args.failure_timestep + 1].to(device),
        sequence["actions"][None, :args.failure_timestep + 1].to(device),
        (
            sequence["robot_features"][
                None, :args.failure_timestep + 1
            ].to(device)
            if args.execution_robot_features else None
        ),
        return_hidden=True,
    )
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
    detector_scores = None
    detector_active = None
    geometry_relative = None
    geometry_drift = None
    acquisition_kind = None
    acquisition_attempt = None
    acquisition_lift = None
    acquisition_drift = None
    if sum((
        args.auto_trigger,
        args.auto_geometry_trigger,
        args.auto_acquisition_verify_trigger,
    )) > 1:
        raise ValueError("choose only one automatic trigger")
    if args.auto_trigger:
        (
            args.failure_timestep,
            detector_scores,
            detector_active,
        ) = causal_failure_scores(args, device)
    elif args.auto_geometry_trigger:
        (
            args.failure_timestep,
            geometry_relative,
            geometry_drift,
            detector_active,
        ) = causal_geometry_trigger(args, device)
    elif args.auto_acquisition_verify_trigger:
        (
            args.failure_timestep,
            geometry_relative,
            geometry_drift,
            detector_active,
            acquisition_kind,
            acquisition_attempt,
            acquisition_lift,
            acquisition_drift,
        ) = causal_acquisition_verify_trigger(args, device)
    elif (
        args.geometry_checkpoint is not None
        and args.candidate_macro_kind == "relative-regrasp"
    ):
        geometry_relative, _ = observable_relative_geometry(args, device)
        geometry_relative = geometry_relative.numpy()
    selected_name, selected_chunk, predicted_values = select_candidate(
        args, device, geometry_relative
    )
    live_slot_model = None
    live_geometry_model = None
    reobserved_relative = None
    if args.reobserve_after_retract:
        if args.slot_checkpoint is None or args.geometry_checkpoint is None:
            raise ValueError(
                "live reobservation requires --slot-checkpoint and "
                "--geometry-checkpoint"
            )
        live_slot_model = Task6ExecutionSlots().to(device).eval()
        live_slot_model.load_state_dict(load_file(args.slot_checkpoint))
        live_geometry_model = SlotRelativeGeometry().to(device).eval()
        live_geometry_model.load_state_dict(load_file(args.geometry_checkpoint))
    torch.cuda.empty_cache()
    if args.max_resume_steps:
        model, processor, input_device = load_openvla(
            args.checkpoint, args.device_map, args.gpu_memory_gib
        )
    else:
        model = processor = input_device = None
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
    final_sim_state = None
    second_recovery_steps = 0
    second_recovery_success = False
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
                image,
                (
                    f"NORMAL  step={timestep}  "
                    f"failed_acq={detector_scores[timestep]:.3f}  "
                    f"gate={'ASSESS' if detector_active[timestep] else 'WAIT'}"
                    if detector_scores is not None
                    else (
                        f"NORMAL  step={timestep}  "
                        f"slot_drift={geometry_drift[timestep]:.3f}  "
                        f"gate={'ASSESS' if detector_active[timestep] else 'WAIT'}"
                        if geometry_drift is not None
                        else f"REPLAY ORIGINAL  step={timestep}"
                    )
                ),
                (170, 170, 170),
            ))
            obs, _, _, _ = env.step(rows[timestep]["executed_action"])
        if not args.continuous_recovery:
            # Branch comparison reconstructs a fresh environment from the
            # flattened simulator state. Actual online recovery should use
            # --continuous-recovery so unstable objects and controller state
            # are not perturbed by this reconstruction.
            sim_state = env.get_sim_state()
            env.close()
            fork_env = OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=256,
                camera_widths=256,
            )
            fork_env.seed(0)
            fork_env.reset()
            fork_env.set_init_state(initial_states[args.episode_id])
            obs = fork_env.regenerate_obs_from_state(sim_state)
            controller = fork_env.env.robots[0].controller
            controller.update(force=True)
            controller.reset_goal()
            fork_env.env.timestep = 0
            fork_env.env.done = False
            env = fork_env
        candidate_start = 0
        if args.reobserve_after_retract:
            prefix_steps = min(
                args.reobserve_after_retract, len(selected_chunk)
            )
            for index, candidate_action in enumerate(
                selected_chunk[:prefix_steps]
            ):
                image = np.ascontiguousarray(
                    obs["agentview_image"][::-1, ::-1]
                )
                frames.append(annotate(
                    image,
                    f"RETRACT BEFORE REOBSERVE  "
                    f"step={index + 1}/{prefix_steps}",
                    (255, 180, 30),
                ))
                obs, _, _, _ = env.step(candidate_action.tolist())
            reobserved_relative = live_relative_geometry(
                obs, selected_chunk[prefix_steps - 1],
                live_slot_model, live_geometry_model, device,
            )
            regenerated = relative_regrasp_macros(reobserved_relative)
            selected_chunk = np.concatenate((
                selected_chunk[:prefix_steps],
                regenerated[selected_name],
            ))
            candidate_start = prefix_steps
        for index, candidate_action in enumerate(
            selected_chunk[candidate_start:], start=candidate_start
        ):
            image = np.ascontiguousarray(
                obs["agentview_image"][::-1, ::-1]
            )
            frames.append(annotate(
                image,
                (
                    "FAILED ACQUISITION DETECTED  "
                    if index == 0 and args.auto_trigger else "REGRASP  "
                )
                + f"{selected_name}  step={index + 1}/{len(selected_chunk)}",
                (255, 180, 30),
            ))
            obs, _, _, _ = env.step(candidate_action.tolist())
        mug_completed_by_recovery = bool(env.env._eval_predicate(
            ["on", "porcelain_mug_1", "plate_1"]
        ))
        mug_position_after_recovery = np.asarray(
            obs["porcelain_mug_1_pos"], dtype=np.float32
        ).tolist()
        _, pudding_attached_after_recovery = extract_target_gripper_contact(
            env.env.sim, target_name="chocolate_pudding_1"
        )
        pudding_position_after_recovery = np.asarray(
            obs["chocolate_pudding_1_pos"], dtype=np.float32
        ).tolist()
        plate_position_after_recovery = np.asarray(
            obs["plate_1_pos"], dtype=np.float32
        ).tolist()
        if (
            args.resume_mode in ("mug-then-pudding", "full-then-pudding")
            and mug_completed_by_recovery
        ):
            stage_switched = True
            stage_switch_step = 0
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
        if not success and args.second_recovery_candidate is not None:
            second_state = env.get_sim_state()
            env.close()
            second_env = OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=256,
                camera_widths=256,
            )
            second_env.seed(0)
            second_env.reset()
            second_env.set_init_state(initial_states[args.episode_id])
            obs = second_env.regenerate_obs_from_state(second_state)
            controller = second_env.env.robots[0].controller
            controller.update(force=True)
            controller.reset_goal()
            second_env.env.timestep = 0
            second_env.env.done = False
            env = second_env
            second_chunk = candidate_macros()[args.second_recovery_candidate]
            for index, candidate_action in enumerate(second_chunk):
                image = np.ascontiguousarray(
                    obs["agentview_image"][::-1, ::-1]
                )
                frames.append(annotate(
                    image,
                    "SECOND RECOVERY: "
                    f"{args.second_recovery_candidate}  "
                    f"step={index + 1}/{len(second_chunk)}",
                    (255, 100, 220),
                ))
                obs, _, _, _ = env.step(candidate_action.tolist())
                second_recovery_steps += 1
                success = bool(env.check_success())
                if success:
                    second_recovery_success = True
                    frames.extend([frames[-1]] * 10)
                    break
        final_sim_state = env.get_sim_state()
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, fps=args.fps)
    metadata = {
        "schema_version": 1,
        "episode_id": args.episode_id,
        "failure_timestep": args.failure_timestep,
        "trigger_source": (
            "execution_mamba2_causal" if args.auto_trigger
            else "observable_slot_geometry_causal"
            if args.auto_geometry_trigger
            else "observable_slot_acquisition_verification"
            if args.auto_acquisition_verify_trigger else "fixed"
        ),
        "trigger_confirm_frames": (
            args.trigger_confirm_frames
            if args.auto_trigger or args.auto_geometry_trigger else None
        ),
        "trigger_score": (
            float(detector_scores[args.failure_timestep])
            if detector_scores is not None else None
        ),
        "selected_candidate": selected_name,
        "candidate_selection_source": (
            "forced_development_override"
            if args.forced_candidate is not None else "learned_scorer"
        ),
        "candidate_geometry_source": (
            "observable_slot_geometry_model"
            if geometry_relative is not None
            else "development_privileged_replay"
            if args.candidate_macro_kind == "relative-regrasp"
            else "fixed_macro_library"
        ),
        "geometry_drift_at_trigger": (
            float(geometry_drift[args.failure_timestep])
            if geometry_drift is not None else None
        ),
        "acquisition_failure_kind": acquisition_kind,
        "acquisition_attempt_timestep": acquisition_attempt,
        "acquisition_lift_at_assessment_m": acquisition_lift,
        "acquisition_relative_drift_at_assessment_m":
            acquisition_drift,
        "reobserve_after_retract": args.reobserve_after_retract,
        "reobserved_relative_xyz_m": (
            reobserved_relative.tolist()
            if reobserved_relative is not None else None
        ),
        "candidate_macro_kind": args.candidate_macro_kind,
        "continuous_recovery": args.continuous_recovery,
        "predicted_values": predicted_values,
        "success": success,
        "original_replay_steps": args.failure_timestep,
        "recovery_steps": len(selected_chunk),
        "resume_steps": len(frames) - args.failure_timestep - len(selected_chunk)
                        - (10 if success else 0),
        "resume_mode": args.resume_mode,
        "stage_switch_step": stage_switch_step,
        "mug_completed_by_recovery": mug_completed_by_recovery,
        "second_recovery_candidate": args.second_recovery_candidate,
        "second_recovery_steps": second_recovery_steps,
        "second_recovery_success": second_recovery_success,
        "mug_position_after_recovery": mug_position_after_recovery,
        "pudding_attached_after_recovery":
            bool(pudding_attached_after_recovery),
        "pudding_position_after_recovery":
            pudding_position_after_recovery,
        "plate_position_after_recovery": plate_position_after_recovery,
        "video": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if args.save_final_sim_state is not None:
        args.save_final_sim_state.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_final_sim_state, final_sim_state)
    print(json.dumps(metadata, indent=2))
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
