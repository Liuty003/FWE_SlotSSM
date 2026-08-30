#!/usr/bin/env python
"""Fork matched regrasp candidates from natural task-9 failure states."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = REPO_ROOT / "experiments" / "robot" / "libero"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LIBERO_ROOT))
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", str(LIBERO_ROOT / "baseline_libero_config")
)

from recovery.teachers.sim_contact import (  # noqa: E402
    extract_target_gripper_contact,
)
from recovery.data.branch_trajectory import BranchTrajectoryWriter  # noqa: E402
from scripts.collect_task6_failure_candidate_branches import (  # noqa: E402
    action,
    relative_regrasp_macros,
)

TASK_SUITE = "libero_10"
TASK_ID = 9
TARGET_KEY = "white_yellow_mug_1_pos"
TARGET_NAME = "white_yellow_mug_1"
# Optional task-specific diagnostic.  Task 3 reuses this collector for a
# regrasp-only branch and has no microwave region.
TARGET_REGION = "microwave_1_heating_region"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rollout", type=Path, required=True)
    # These defaults retain the task-9 collector behavior.  The same strictly
    # offline fork mechanism is reusable for another LIBERO task by supplying
    # its policy-visible observation key and simulator-only teacher name.
    parser.add_argument("--task-id", type=int, default=TASK_ID)
    parser.add_argument("--target-key", default=TARGET_KEY)
    parser.add_argument("--target-name", default=TARGET_NAME)
    parser.add_argument("--target-region", default=TARGET_REGION)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--failure-timestep", type=int, required=True)
    parser.add_argument(
        "--failure-type",
        help="Human-audited source failure category; stored as provenance only.",
    )
    parser.add_argument("--wait-steps", type=int, default=20)
    parser.add_argument(
        "--grasp-relative-reference", type=float, nargs=3,
        metavar=("DX", "DY", "DZ"),
        help=(
            "Desired mug xyz relative to EEF after a stable grasp. Candidate "
            "motion targets the residual to this relation."
        ),
    )
    parser.add_argument(
        "--stress-transport-steps", type=int, default=0,
        help="After lift verification, retain grasp and move +y for this many steps.",
    )
    parser.add_argument("--stress-transport-action", type=float, default=.30)
    parser.add_argument(
        "--candidate-name", action="append",
        help="Optionally evaluate only named candidates (repeatable).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trajectory-dir", type=Path,
        help=(
            "Optional root for aligned per-step raw recovery trajectories. "
            "OpenVLA nominal actions after the counterfactual fork remain "
            "explicitly unavailable rather than being fabricated."
        ),
    )
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def main():
    args = arguments()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv

    rows = [
        json.loads(line)
        for line in (args.source_rollout / "steps.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_timestep = {int(row["timestep"]): row for row in rows}
    if args.failure_timestep not in by_timestep:
        raise ValueError("failure timestep is absent from source rollout")

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    bddl = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder / task.bddl_file
    )

    def make_env():
        env_class = (
            SegmentationRenderEnv
            if args.trajectory_dir is not None else OffScreenRenderEnv
        )
        env = env_class(
            bddl_file_name=str(bddl),
            camera_heights=256,
            camera_widths=256,
            camera_depths=args.trajectory_dir is not None,
        )
        env.seed(0)
        env.reset()
        env.set_init_state(initial_states[args.episode_id])
        return env

    source_env = make_env()
    try:
        obs = None
        for _ in range(args.wait_steps):
            obs, _, _, _ = source_env.step(action().tolist())
        for timestep in range(args.failure_timestep):
            obs, _, _, _ = source_env.step(
                by_timestep[timestep]["executed_action"]
            )
        sim_state = source_env.get_sim_state()
        target_before = np.asarray(obs[args.target_key], np.float32)
        eef_before = np.asarray(obs["robot0_eef_pos"], np.float32)
        relative = target_before - eef_before
        _, attached_before = extract_target_gripper_contact(
            source_env.env.sim, target_name=args.target_name
        )
        target_instance_id = (
            source_env.instance_to_id.get(args.target_name)
            if args.trajectory_dir is not None else None
        )
        source_nominal_action = by_timestep[args.failure_timestep].get(
            "planned_action"
        )
    finally:
        source_env.close()

    grasp_reference = (
        np.asarray(args.grasp_relative_reference, np.float32)
        if args.grasp_relative_reference is not None else np.zeros(3, np.float32)
    )
    correction = relative - grasp_reference
    macros = relative_regrasp_macros(correction)
    if args.candidate_name:
        missing = set(args.candidate_name) - set(macros)
        if missing:
            raise ValueError(f"unknown candidates: {sorted(missing)}")
        macros = {name: macros[name] for name in args.candidate_name}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for candidate_index, (name, chunk) in enumerate(macros.items()):
        env = make_env()
        try:
            obs = env.regenerate_obs_from_state(sim_state)
            controller = env.env.robots[0].controller
            controller.update(force=True)
            controller.reset_goal()
            env.env.timestep = 0
            env.env.done = False
            restored_target = np.asarray(obs[args.target_key], np.float32)
            restore_error = float(
                np.abs(restored_target - target_before).max()
            )
            frames = [
                np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            ]
            trajectory_writer = (
                BranchTrajectoryWriter(
                    args.trajectory_dir,
                    task_suite=TASK_SUITE,
                    task_id=args.task_id,
                    source_episode_id=args.episode_id,
                    failure_timestep=args.failure_timestep,
                    candidate_name=name,
                    candidate_index=candidate_index,
                    source_rollout=str(args.source_rollout),
                    failure_type=args.failure_type,
                    target_name=args.target_name,
                    target_instance_id=target_instance_id,
                    source_nominal_action=source_nominal_action,
                )
                if args.trajectory_dir is not None else None
            )
            previous_recovery_action = None
            attached_ever = False
            min_eef_target_distance = float("inf")
            for branch_step, candidate_action in enumerate(chunk):
                pre_action_obs = obs
                target_before_step = np.asarray(
                    pre_action_obs[args.target_key], np.float32
                )
                contact_before_step, attached_before_step = (
                    extract_target_gripper_contact(
                        env.env.sim, target_name=args.target_name
                    )
                )
                obs, _, done, _ = env.step(candidate_action.tolist())
                contact_after_step, attached_after_step = (
                    extract_target_gripper_contact(
                        env.env.sim, target_name=args.target_name
                    )
                )
                if trajectory_writer is not None:
                    trajectory_writer.record(
                        step=branch_step,
                        observation=pre_action_obs,
                        recovery_action=candidate_action,
                        previous_recovery_action=previous_recovery_action,
                        teacher_contact_before=contact_before_step,
                        teacher_attached_before=attached_before_step,
                        teacher_target_position_before=target_before_step,
                        teacher_contact_after=contact_after_step,
                        teacher_attached_after=attached_after_step,
                        teacher_target_position_after=np.asarray(
                            obs[args.target_key], np.float32
                        ),
                        task_success_after=bool(env.check_success()),
                        done_after=bool(done),
                    )
                previous_recovery_action = candidate_action.copy()
                frames.append(np.ascontiguousarray(
                    obs["agentview_image"][::-1, ::-1]
                ))
                target = np.asarray(obs[args.target_key], np.float32)
                eef_position = np.asarray(obs["robot0_eef_pos"], np.float32)
                min_eef_target_distance = min(
                    min_eef_target_distance,
                    float(np.linalg.norm(target - eef_position)),
                )
                _, attached = extract_target_gripper_contact(
                    env.env.sim, target_name=args.target_name
                )
                attached_ever = attached_ever or attached
                if done:
                    break
            attached_before_stress = attached_ever and attached
            stress_start_target = np.asarray(obs[args.target_key], np.float32)
            for stress_step in range(args.stress_transport_steps):
                pre_action_obs = obs
                target_before_step = np.asarray(
                    pre_action_obs[args.target_key], np.float32
                )
                contact_before_step, attached_before_step = (
                    extract_target_gripper_contact(
                        env.env.sim, target_name=args.target_name
                    )
                )
                stress_action = action(
                    y=args.stress_transport_action, gripper=1.
                )
                obs, _, done, _ = env.step(stress_action.tolist())
                contact_after_step, attached_after_step = (
                    extract_target_gripper_contact(
                        env.env.sim, target_name=args.target_name
                    )
                )
                if trajectory_writer is not None:
                    trajectory_writer.record(
                        step=branch_step + stress_step + 1,
                        observation=pre_action_obs,
                        recovery_action=stress_action,
                        previous_recovery_action=previous_recovery_action,
                        teacher_contact_before=contact_before_step,
                        teacher_attached_before=attached_before_step,
                        teacher_target_position_before=target_before_step,
                        teacher_contact_after=contact_after_step,
                        teacher_attached_after=attached_after_step,
                        teacher_target_position_after=np.asarray(
                            obs[args.target_key], np.float32
                        ),
                        task_success_after=bool(env.check_success()),
                        done_after=bool(done),
                    )
                previous_recovery_action = stress_action.copy()
                frames.append(np.ascontiguousarray(
                    obs["agentview_image"][::-1, ::-1]
                ))
                _, attached = extract_target_gripper_contact(
                    env.env.sim, target_name=args.target_name
                )
                if done:
                    break
            target_after = np.asarray(obs[args.target_key], np.float32)
            eef_after = np.asarray(obs["robot0_eef_pos"], np.float32)
            _, attached_after = extract_target_gripper_contact(
                env.env.sim, target_name=args.target_name
            )
            mug_inside = (
                bool(env.env._eval_predicate(["on", args.target_name, args.target_region]))
                if args.target_region is not None
                else None
            )
            task_success = bool(env.check_success())
        finally:
            env.close()

        target_lift = float(target_after[2] - target_before[2])
        target_displacement = float(
            np.linalg.norm(target_after - target_before)
        )
        final_distance = float(np.linalg.norm(target_after - eef_after))
        stress_target_displacement = float(
            np.linalg.norm(target_after - stress_start_target)
        )
        transport_stable = bool(
            attached_after
            and stress_target_displacement > (
                .01 if args.stress_transport_steps else 0.
            )
        )
        # Regrasp branches end after lift verification; insertion is evaluated
        # only after the selected candidate is resumed with OpenVLA.
        teacher_value = float(
            2.0 * float(attached_after)
            + 10.0 * max(target_lift, 0.0)
            + 2.0 * min(target_displacement, 0.08)
            - 0.5 * min(final_distance, 0.25)
        )
        video = args.output_dir / f"{name}.mp4"
        imageio.mimsave(video, frames, fps=args.fps)
        outcome = {
            "candidate_name": name,
            "action_chunk": chunk.tolist(),
            "steps_executed": len(frames) - 1,
            "restore_error_m": restore_error,
            "attached_before": bool(attached_before),
            "attached_after": bool(attached_after),
            "attached_ever": bool(attached_ever),
            "attached_before_stress": bool(attached_before_stress),
            "stress_transport_steps": args.stress_transport_steps,
            "stress_target_displacement_m": stress_target_displacement,
            "transport_stable": transport_stable,
            "target_lift_m": target_lift,
            "target_displacement_m": target_displacement,
            "eef_target_distance_after_m": final_distance,
            "min_eef_target_distance_m": min_eef_target_distance,
            "mug_inside_after": mug_inside,
            "task_success_after": task_success,
            "teacher_value": teacher_value,
            "video": str(video),
            "raw_trajectory": (
                str(trajectory_writer.path)
                if trajectory_writer is not None else None
            ),
        }
        outcomes.append(outcome)
        print(json.dumps(outcome), flush=True)

    (args.output_dir / "candidate_branches.json").write_text(json.dumps({
        "schema_version": 1,
        "kind": "task9_matched_relative_regrasp_branches",
        "task_suite": TASK_SUITE,
        "task_id": args.task_id,
        "episode_id": args.episode_id,
        "failure_timestep": args.failure_timestep,
        "source_rollout": str(args.source_rollout),
        "same_sim_state_for_all_candidates": True,
        "independent_controller_per_candidate": True,
        "controller_goal_reset_after_restore": True,
        "model_input_excludes_privileged_outcomes": True,
        "before": {
            "target_position": target_before.tolist(),
            "eef_position": eef_before.tolist(),
            "target_relative_to_eef_m": relative.tolist(),
            "desired_grasp_relative_to_eef_m": grasp_reference.tolist(),
            "candidate_correction_input_m": correction.tolist(),
            "attached": bool(attached_before),
            "candidate_geometry_source": "privileged_branch_teacher",
        },
        "candidates": outcomes,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
