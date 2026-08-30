#!/usr/bin/env python3
"""Run one reproducible OpenVLA-OFT LIBERO episode and save its video.

This deliberately reuses the official OpenVLA-OFT inference implementation.  It
is a small, task-selectable smoke/evaluation runner; the upstream evaluator only
offers a full-suite loop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oft-root",
        type=Path,
        default=Path("/data/lty/code/openvla-oft-official"),
    )
    parser.add_argument(
        "--libero-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "robot" / "libero",
        help="Directory containing the vendored LIBERO package and local config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-10"),
    )
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=3)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--open-loop-steps", type=int, default=8)
    parser.add_argument(
        "--oracle-stage-switch",
        action="store_true",
        help=(
            "After the mug is stably on the plate, clear the OFT action queue and "
            "switch the prompt to --subgoal-instruction. This is a privileged, "
            "task-specific ceiling/overfit policy, not a learned controller."
        ),
    )
    parser.add_argument("--subgoal-confirm-steps", type=int, default=5)
    parser.add_argument(
        "--subgoal-instruction",
        default="put the chocolate pudding to the right of the plate",
    )
    parser.add_argument(
        "--first-subgoal-instruction",
        help=(
            "Optional overfit decomposition prompt used before the oracle switch; "
            "by default the original full task instruction is used."
        ),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/openvla-oft-baseline"),
    )
    parser.add_argument(
        "--save-teacher-states-dir",
        type=Path,
        help=(
            "Optional privileged replay output.  It saves MuJoCo snapshots plus "
            "policy-visible proprio/action records for offline recovery-data collection; "
            "the policy never reads these files."
        ),
    )
    parser.add_argument("--save-teacher-state-every", type=int, default=8)
    parser.add_argument(
        "--save-teacher-images",
        action="store_true",
        help=(
            "With --save-teacher-states-dir, save causal main/wrist RGB frames "
            "with the nominal action records for offline gate/recovery training."
        ),
    )
    return parser.parse_args()


def run_recorded_episode(
    *,
    cfg,
    env,
    task_description: str,
    model,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    initial_state,
    task_id: int,
    episode_id: int,
    snapshot_root: Path,
    snapshot_every: int,
    save_images: bool,
) -> tuple[bool, list, int]:
    """Exact OFT rollout with offline-only snapshots before each policy action."""
    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_image,
        get_libero_wrist_image,
    )
    from experiments.robot.libero.run_libero_eval import TASK_MAX_STEPS, prepare_observation, process_action
    from experiments.robot.robot_utils import get_action

    if snapshot_every < 1:
        raise ValueError("--save-teacher-state-every must be positive")
    snapshot_root.mkdir(parents=True, exist_ok=True)
    trajectory_path = snapshot_root / "trajectory.jsonl"
    env.reset()
    obs = env.set_init_state(initial_state)
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    frames, previous_action = [], None
    success, t, policy_step = False, 0, 0
    rgb_dir = snapshot_root / "rgb"
    wrist_dir = snapshot_root / "wrist_rgb"
    if save_images:
        rgb_dir.mkdir(parents=True, exist_ok=True)
        wrist_dir.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio
    with trajectory_path.open("w") as trajectory:
        while t < TASK_MAX_STEPS[cfg.task_suite_name] + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue
            observation, image = prepare_observation(obs, resize_size)
            frames.append(image)
            if not action_queue:
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)
            action = process_action(action_queue.popleft(), cfg.model_family)
            proprio = {
                "robot0_eef_pos": obs["robot0_eef_pos"].tolist(),
                "robot0_eef_quat": obs["robot0_eef_quat"].tolist(),
                "robot0_gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
            }
            online_observation = None
            if save_images:
                rgb_path = rgb_dir / f"step-{policy_step:04d}.png"
                wrist_path = wrist_dir / f"step-{policy_step:04d}.png"
                imageio.imwrite(rgb_path, get_libero_image(obs))
                imageio.imwrite(wrist_path, get_libero_wrist_image(obs))
                online_observation = {
                    "rgb": str(rgb_path.resolve()),
                    "wrist_rgb": str(wrist_path.resolve()),
                    "proprio": proprio,
                    "previous_executed_action": previous_action,
                }
            if policy_step % snapshot_every == 0:
                state_dir = snapshot_root / f"step{policy_step:04d}"
                state_dir.mkdir(parents=True, exist_ok=True)
                import numpy as np

                np.savez_compressed(state_dir / "mujoco_state.npz", mujoco_state=env.sim.get_state().flatten())
                (state_dir / "manifest.json").write_text(
                    json.dumps(
                        {
                            "kind": "openvla_oft_nominal_replay_fork_v1",
                            "teacher_only": True,
                            "task_suite": str(cfg.task_suite_name),
                            "task_id": task_id,
                            "episode_id": episode_id,
                            "task_language": task_description,
                            "fork_timestep": policy_step,
                            "observation_timing": "pre_action",
                            "source_steps": str(trajectory_path.resolve()),
                            "proprio": proprio,
                            "previous_executed_action": previous_action,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            row = {
                "policy_step": policy_step,
                "timestep": policy_step,
                "proprio": proprio,
                "previous_executed_action": previous_action,
                "executed_action": action.tolist(),
            }
            if online_observation is not None:
                row["online_observation"] = online_observation
            trajectory.write(json.dumps(row) + "\n")
            previous_action = action.tolist()
            obs, _, done, _ = env.step(action.tolist())
            policy_step += 1
            t += 1
            if done:
                success = True
                break
    return success, frames, policy_step


def run_oracle_stage_switch_episode(
    *,
    cfg,
    env,
    task_description: str,
    first_subgoal_instruction: str | None,
    subgoal_instruction: str,
    confirm_steps: int,
    model,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    initial_state,
) -> tuple[bool, list, int, int | None]:
    """Run current OFT with a privileged first-subgoal completion switch."""
    from experiments.robot.libero.libero_utils import get_libero_dummy_action
    from experiments.robot.libero.run_libero_eval import TASK_MAX_STEPS, prepare_observation, process_action
    from experiments.robot.robot_utils import get_action

    if confirm_steps < 1:
        raise ValueError("--subgoal-confirm-steps must be positive")
    env.reset()
    obs = env.set_init_state(initial_state)
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    frames = []
    success, t, policy_step = False, 0, 0
    positive_run = 0
    switched = False
    switch_step = None
    while t < TASK_MAX_STEPS[cfg.task_suite_name] + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        mug_on_plate = bool(
            env.env._eval_predicate(["on", "porcelain_mug_1", "plate_1"])
        )
        if not switched:
            positive_run = positive_run + 1 if mug_on_plate else 0
            if positive_run >= confirm_steps:
                switched = True
                switch_step = policy_step
                # Never execute stale actions generated for the full instruction.
                action_queue.clear()

        observation, image = prepare_observation(obs, resize_size)
        frames.append(image)
        if not action_queue:
            instruction = (
                subgoal_instruction
                if switched
                else (first_subgoal_instruction or task_description)
            )
            actions = get_action(
                cfg,
                model,
                observation,
                instruction,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=cfg.use_film,
            )
            action_queue.extend(actions)
        action = process_action(action_queue.popleft(), cfg.model_family)
        obs, _, done, _ = env.step(action.tolist())
        policy_step += 1
        t += 1
        if done:
            success = True
            break
    return success, frames, policy_step, switch_step


def main() -> None:
    args = parse_args()
    if not args.oft_root.is_dir():
        raise FileNotFoundError(f"OpenVLA-OFT source directory not found: {args.oft_root}")
    if not (args.libero_root / "libero" / "libero").is_dir():
        raise FileNotFoundError(f"vendored LIBERO package not found: {args.libero_root}")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {args.checkpoint}")
    if args.save_teacher_images and args.save_teacher_states_dir is None:
        raise ValueError("--save-teacher-images requires --save-teacher-states-dir")

    # Official utilities bind their DEVICE constant at import time.
    # An outer scheduler may have already restricted this process to one
    # physical GPU.  Preserve that mapping so that ``cuda:0`` below refers to
    # the scheduler-assigned device rather than accidentally remapping every
    # worker back onto the same physical GPU.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    # Keep each run bound to the explicitly requested LIBERO checkout.
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    sys.path.insert(0, str(args.oft_root))
    sys.path.insert(0, str(args.libero_root))
    os.chdir(args.oft_root)

    import imageio.v2 as imageio
    from libero.libero import benchmark

    from experiments.robot.libero.libero_utils import get_libero_env
    from experiments.robot.libero.run_libero_eval import (
        GenerateConfig,
        get_image_resize_size,
        initialize_model,
        run_episode,
        set_seed_everywhere,
        validate_config,
    )

    cfg = GenerateConfig(
        pretrained_checkpoint=str(args.checkpoint),
        task_suite_name=args.task_suite,
        num_trials_per_task=1,
        num_open_loop_steps=args.open_loop_steps,
        use_wandb=False,
    )
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    benchmark_cls = benchmark.get_benchmark_dict()[args.task_suite]
    task_suite = benchmark_cls()
    if not 0 <= args.task_id < task_suite.n_tasks:
        raise ValueError(f"task-id must be in [0, {task_suite.n_tasks - 1}]")

    initial_states = task_suite.get_task_init_states(args.task_id)
    if not 0 <= args.episode_id < len(initial_states):
        raise ValueError(f"episode-id must be in [0, {len(initial_states) - 1}]")

    task = task_suite.get_task(args.task_id)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    replay_steps = None
    switch_step = None
    try:
        if args.oracle_stage_switch:
            if args.save_teacher_states_dir is not None:
                raise ValueError(
                    "--save-teacher-states-dir is not supported with --oracle-stage-switch"
                )
            success, frames, replay_steps, switch_step = run_oracle_stage_switch_episode(
                cfg=cfg,
                env=env,
                task_description=task_description,
                first_subgoal_instruction=args.first_subgoal_instruction,
                subgoal_instruction=args.subgoal_instruction,
                confirm_steps=args.subgoal_confirm_steps,
                model=model,
                resize_size=resize_size,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                initial_state=initial_states[args.episode_id],
            )
        elif args.save_teacher_states_dir is None:
            success, frames = run_episode(
                cfg=cfg,
                env=env,
                task_description=task_description,
                model=model,
                resize_size=resize_size,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                initial_state=initial_states[args.episode_id],
            )
        else:
            snapshot_root = args.save_teacher_states_dir / f"task{args.task_id}-episode{args.episode_id}"
            success, frames, replay_steps = run_recorded_episode(
                cfg=cfg,
                env=env,
                task_description=task_description,
                model=model,
                resize_size=resize_size,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                initial_state=initial_states[args.episode_id],
                task_id=args.task_id,
                episode_id=args.episode_id,
                snapshot_root=snapshot_root,
                snapshot_every=args.save_teacher_state_every,
                save_images=args.save_teacher_images,
            )
    finally:
        env.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task_suite}-task{args.task_id}-episode{args.episode_id}"
    video_path = args.output_dir / f"{stem}.mp4"
    imageio.mimsave(video_path, frames, fps=30)
    metadata = {
        "policy": (
            "openvla-oft-oracle-stage-switch"
            if args.oracle_stage_switch
            else "openvla-oft"
        ),
        "checkpoint": str(args.checkpoint),
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "episode_id": args.episode_id,
        "task_description": task_description,
        "open_loop_steps": args.open_loop_steps,
        "num_frames": len(frames),
        "success": bool(success),
        "video": str(video_path),
        "teacher_state_dir": (
            str(args.save_teacher_states_dir / f"task{args.task_id}-episode{args.episode_id}")
            if args.save_teacher_states_dir is not None
            else None
        ),
        "policy_steps": replay_steps,
        "oracle_stage_switch": bool(args.oracle_stage_switch),
        "stage_switch_predicate": (
            "On(porcelain_mug_1, plate_1)" if args.oracle_stage_switch else None
        ),
        "subgoal_confirm_steps": (
            args.subgoal_confirm_steps if args.oracle_stage_switch else None
        ),
        "subgoal_instruction": (
            args.subgoal_instruction if args.oracle_stage_switch else None
        ),
        "first_subgoal_instruction": (
            args.first_subgoal_instruction if args.oracle_stage_switch else None
        ),
        "stage_switch_step": switch_step,
    }
    metadata_path = args.output_dir / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
