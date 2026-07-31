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
from scripts.collect_task6_failure_candidate_branches import (  # noqa: E402
    action,
    relative_regrasp_macros,
)

TASK_SUITE = "libero_10"
TASK_ID = 9
TARGET_KEY = "white_yellow_mug_1_pos"
TARGET_NAME = "white_yellow_mug_1"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rollout", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--failure-timestep", type=int, required=True)
    parser.add_argument("--wait-steps", type=int, default=20)
    parser.add_argument(
        "--candidate-name", action="append",
        help="Optionally evaluate only named candidates (repeatable).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def main():
    args = arguments()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    rows = [
        json.loads(line)
        for line in (args.source_rollout / "steps.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_timestep = {int(row["timestep"]): row for row in rows}
    if args.failure_timestep not in by_timestep:
        raise ValueError("failure timestep is absent from source rollout")

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(TASK_ID)
    initial_states = suite.get_task_init_states(TASK_ID)
    bddl = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder / task.bddl_file
    )

    def make_env():
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=256,
            camera_widths=256,
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
        target_before = np.asarray(obs[TARGET_KEY], np.float32)
        eef_before = np.asarray(obs["robot0_eef_pos"], np.float32)
        relative = target_before - eef_before
        _, attached_before = extract_target_gripper_contact(
            source_env.env.sim, target_name=TARGET_NAME
        )
    finally:
        source_env.close()

    macros = relative_regrasp_macros(relative)
    if args.candidate_name:
        missing = set(args.candidate_name) - set(macros)
        if missing:
            raise ValueError(f"unknown candidates: {sorted(missing)}")
        macros = {name: macros[name] for name in args.candidate_name}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for name, chunk in macros.items():
        env = make_env()
        try:
            obs = env.regenerate_obs_from_state(sim_state)
            controller = env.env.robots[0].controller
            controller.update(force=True)
            controller.reset_goal()
            env.env.timestep = 0
            env.env.done = False
            restored_target = np.asarray(obs[TARGET_KEY], np.float32)
            restore_error = float(
                np.abs(restored_target - target_before).max()
            )
            frames = [
                np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            ]
            attached_ever = False
            min_eef_target_distance = float("inf")
            for candidate_action in chunk:
                obs, _, done, _ = env.step(candidate_action.tolist())
                frames.append(np.ascontiguousarray(
                    obs["agentview_image"][::-1, ::-1]
                ))
                target = np.asarray(obs[TARGET_KEY], np.float32)
                eef_position = np.asarray(obs["robot0_eef_pos"], np.float32)
                min_eef_target_distance = min(
                    min_eef_target_distance,
                    float(np.linalg.norm(target - eef_position)),
                )
                _, attached = extract_target_gripper_contact(
                    env.env.sim, target_name=TARGET_NAME
                )
                attached_ever = attached_ever or attached
                if done:
                    break
            target_after = np.asarray(obs[TARGET_KEY], np.float32)
            eef_after = np.asarray(obs["robot0_eef_pos"], np.float32)
            _, attached_after = extract_target_gripper_contact(
                env.env.sim, target_name=TARGET_NAME
            )
            mug_inside = bool(env.env._eval_predicate(
                ["in", TARGET_NAME, "microwave_1_heating_region"]
            ))
            task_success = bool(env.check_success())
        finally:
            env.close()

        target_lift = float(target_after[2] - target_before[2])
        target_displacement = float(
            np.linalg.norm(target_after - target_before)
        )
        final_distance = float(np.linalg.norm(target_after - eef_after))
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
            "target_lift_m": target_lift,
            "target_displacement_m": target_displacement,
            "eef_target_distance_after_m": final_distance,
            "min_eef_target_distance_m": min_eef_target_distance,
            "mug_inside_after": mug_inside,
            "task_success_after": task_success,
            "teacher_value": teacher_value,
            "video": str(video),
        }
        outcomes.append(outcome)
        print(json.dumps(outcome), flush=True)

    (args.output_dir / "candidate_branches.json").write_text(json.dumps({
        "schema_version": 1,
        "kind": "task9_matched_relative_regrasp_branches",
        "task_suite": TASK_SUITE,
        "task_id": TASK_ID,
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
            "attached": bool(attached_before),
            "candidate_geometry_source": "privileged_branch_teacher",
        },
        "candidates": outcomes,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
