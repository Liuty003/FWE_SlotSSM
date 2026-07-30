#!/usr/bin/env python
"""Fork matched task-6 recovery macros from a natural empty-grasp state."""

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
from scripts.build_task6_execution_labels import (  # noqa: E402
    MUG,
    PUDDING,
    distance_to_region,
)

TASK_SUITE = "libero_10"
TASK_ID = 6
PLATE = "plate_1_pos"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rollout", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, default=6)
    parser.add_argument("--failure-timestep", type=int, default=211)
    parser.add_argument(
        "--target-role", choices=("pudding", "mug"), default="pudding"
    )
    parser.add_argument("--wait-steps", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def action(x=0., y=0., z=0., gripper=-1.):
    return np.asarray([x, y, z, 0., 0., 0., gripper], np.float32)


def recovery_macro(dx=0., dy=0.):
    """Open/retract, translate, descend, close, and verify with a lift."""
    return np.stack(
        [action(z=.35)] * 4
        + [action(x=dx, y=dy)] * 4
        + [action(z=-.35)] * 4
        + [action(gripper=1.)] * 8
        + [action(z=.4, gripper=1.)] * 8
    )


def wide_recovery_macro(dx=0., dy=0.):
    """Search farther while retaining close and lift verification in 28 steps."""
    return np.stack(
        [action(z=.35)] * 4
        + [action(x=dx, y=dy)] * 8
        + [action(z=-.35)] * 4
        + [action(gripper=1.)] * 6
        + [action(z=.4, gripper=1.)] * 6
    )


def candidate_macros():
    candidates = {
        "reclose_lift": np.stack(
            [action(gripper=1.)] * 20
            + [action(z=.4, gripper=1.)] * 8
        ),
        "regrasp_center": recovery_macro(),
        "regrasp_x_minus": recovery_macro(dx=-.30),
        "regrasp_x_plus": recovery_macro(dx=.30),
        "regrasp_y_minus": recovery_macro(dy=-.30),
        "regrasp_y_plus": recovery_macro(dy=.30),
        "regrasp_diag_minus_plus": recovery_macro(dx=-.22, dy=.22),
        "regrasp_diag_plus_minus": recovery_macro(dx=.22, dy=-.22),
        "wide_x_minus": wide_recovery_macro(dx=-.80),
        "wide_y_minus": wide_recovery_macro(dy=-.80),
        "wide_diag_minus_minus": wide_recovery_macro(dx=-.75, dy=-.30),
        "wide_diag_minus_plus": wide_recovery_macro(dx=-.75, dy=.30),
    }
    values = [chunk.tobytes() for chunk in candidates.values()]
    if len(set(values)) != len(values):
        raise ValueError("candidate macros must be distinct")
    return candidates


def object_positions(obs):
    return {
        PUDDING: np.asarray(obs[PUDDING], np.float32),
        MUG: np.asarray(obs[MUG], np.float32),
        PLATE: np.asarray(obs[PLATE], np.float32),
    }


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
    fork_env = make_env()
    target_key = PUDDING if args.target_role == "pudding" else MUG
    target_name = (
        "chocolate_pudding_1"
        if args.target_role == "pudding" else "porcelain_mug_1"
    )
    collateral_key = MUG if args.target_role == "pudding" else PUDDING
    noop = action()
    try:
        obs = None
        for _ in range(args.wait_steps):
            obs, _, _, _ = source_env.step(noop.tolist())
        for timestep in range(args.failure_timestep):
            row = by_timestep[timestep]
            obs, _, _, _ = source_env.step(row["executed_action"])
        sim_state = source_env.get_sim_state()
        before = object_positions(obs)
        before_eef = np.asarray(obs["robot0_eef_pos"], np.float32)
        before_distance = (
            distance_to_region(before[PUDDING])
            if args.target_role == "pudding"
            else float(np.linalg.norm(before[MUG][:2] - before[PLATE][:2]))
        )
        _, attached_before = extract_target_gripper_contact(
            source_env.env.sim, target_name=target_name
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        outcomes = []
        for name, chunk in candidate_macros().items():
            branch_obs = fork_env.regenerate_obs_from_state(sim_state)
            restored = object_positions(branch_obs)
            restore_error = max(
                float(np.abs(restored[key] - before[key]).max())
                for key in before
            )
            fork_env.env.timestep = 0
            fork_env.env.done = False
            frames = [
                np.ascontiguousarray(
                    branch_obs["agentview_image"][::-1, ::-1]
                )
            ]
            for candidate_action in chunk:
                branch_obs, _, done, _ = fork_env.step(
                    candidate_action.tolist()
                )
                frames.append(np.ascontiguousarray(
                    branch_obs["agentview_image"][::-1, ::-1]
                ))
                if done:
                    break
            after = object_positions(branch_obs)
            after_eef = np.asarray(
                branch_obs["robot0_eef_pos"], np.float32
            )
            _, attached_after = extract_target_gripper_contact(
                fork_env.env.sim, target_name=target_name
            )
            mug_on_plate = bool(fork_env.env._eval_predicate(
                ["on", "porcelain_mug_1", "plate_1"]
            ))
            target_displacement = float(
                np.linalg.norm(after[target_key] - before[target_key])
            )
            collateral_displacement = float(
                np.linalg.norm(
                    after[collateral_key] - before[collateral_key]
                )
            )
            after_distance = (
                distance_to_region(after[PUDDING])
                if args.target_role == "pudding"
                else float(np.linalg.norm(after[MUG][:2] - after[PLATE][:2]))
            )
            goal_progress = before_distance - after_distance
            target_lift = float(
                after[target_key][2] - before[target_key][2]
            )
            eef_target_distance = float(
                np.linalg.norm(after_eef - after[target_key])
            )
            teacher_value = float(
                8 * goal_progress
                + 1.5 * float(attached_after)
                + 10 * max(target_lift, 0)
                + 2 * min(target_displacement, .05)
                - 12 * collateral_displacement
                - float(args.target_role == "pudding" and not mug_on_plate)
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
                "target_role": args.target_role,
                "target_displacement_m": target_displacement,
                "target_lift_m": target_lift,
                "target_goal_progress_m": goal_progress,
                "collateral_displacement_m": collateral_displacement,
                # Backward-compatible task-6 pudding fields.
                "pudding_displacement_m": target_displacement,
                "pudding_lift_m": target_lift,
                "pudding_goal_progress_m": goal_progress,
                "mug_displacement_m": collateral_displacement,
                "mug_on_plate_after": mug_on_plate,
                "eef_target_distance_after_m": eef_target_distance,
                "task_success_after": bool(fork_env.check_success()),
                "teacher_value": teacher_value,
                "video": str(video),
            }
            outcomes.append(outcome)
            print(json.dumps(outcome), flush=True)
        (args.output_dir / "candidate_branches.json").write_text(
            json.dumps({
                "schema_version": 1,
                "kind": "task6_matched_macro_candidate_branches",
                "task_suite": TASK_SUITE,
                "task_id": TASK_ID,
                "episode_id": args.episode_id,
                "failure_timestep": args.failure_timestep,
                "target_role": args.target_role,
                "source_rollout": str(args.source_rollout),
                "model_input_excludes_privileged_outcomes": True,
                "same_sim_state_for_all_candidates": True,
                "before": {
                    "target_goal_distance_m": before_distance,
                    "attached": bool(attached_before),
                    "eef_position": before_eef.tolist(),
                },
                "candidates": outcomes,
            }, indent=2),
            encoding="utf-8",
        )
    finally:
        source_env.close()
        fork_env.close()


if __name__ == "__main__":
    main()
