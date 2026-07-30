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
    parser.add_argument("--source-rollout", type=Path)
    parser.add_argument(
        "--source-sim-state",
        type=Path,
        help="Flattened MuJoCo .npy state saved by an online recovery run.",
    )
    parser.add_argument("--episode-id", type=int, default=6)
    parser.add_argument("--failure-timestep", type=int, default=211)
    parser.add_argument(
        "--target-role", choices=("pudding", "mug"), default="pudding"
    )
    parser.add_argument(
        "--macro-kind",
        choices=(
            "short", "relative-regrasp", "safe-approach",
            "cup-stable-grasp", "cup-place",
        ),
        default="short",
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


def safe_approach_macros():
    """Long-range, raised approaches for prospective collision recovery."""
    parameters = {
        "safe_x070_s12_d12": (-.70, 0., 12, 12),
        "safe_x070_s12_d14": (-.70, 0., 12, 14),
        "safe_x070_s12_d16": (-.70, 0., 12, 16),
        "safe_x070_s14_d12": (-.70, 0., 14, 12),
        "safe_x070_s14_d14": (-.70, 0., 14, 14),
        "safe_x070_s14_d16": (-.70, 0., 14, 16),
        "safe_x070_ym10_s12_d12": (-.70, -.10, 12, 12),
        "safe_x070_ym10_s12_d14": (-.70, -.10, 12, 14),
        "safe_x070_ym10_s12_d16": (-.70, -.10, 12, 16),
        "safe_x085_ym10_s12_d12": (-.85, -.10, 12, 12),
        "safe_x085_ym10_s12_d14": (-.85, -.10, 12, 14),
        "safe_x085_ym10_s12_d16": (-.85, -.10, 12, 16),
    }
    candidates = {}
    for name, (dx, dy, move_steps, descend_steps) in parameters.items():
        candidates[name] = np.stack(
            [action(z=.35)] * 8
            + [action(x=dx, y=dy)] * move_steps
            + [action()] * (16 - move_steps)
            + [action(z=-.35)] * descend_steps
            + [action()] * (16 - descend_steps)
            + [action(gripper=1.)] * 12
            + [action(z=.25, gripper=1.)] * 10
        )
    return candidates


def relative_regrasp_macros(relative_target_xyz):
    """Generate matched regrasp chunks from target position relative to EEF.

    The conversion from normalized OSC action to Cartesian displacement is a
    local empirical gain, not a scene direction. Candidate diversity is
    expressed as dimensionless multipliers around the observed slot geometry.
    """
    relative = np.asarray(relative_target_xyz, dtype=np.float32)
    if relative.shape != (3,) or not np.isfinite(relative).all():
        raise ValueError("relative_target_xyz must be a finite xyz vector")
    gain_m_per_unit_step = .0105
    up_steps, move_steps, down_steps = 6, 10, 12
    up_action = .30
    raised_m = gain_m_per_unit_step * up_steps * up_action
    candidates = {}
    for xy_scale in (.8, 1., 1.2):
        for z_scale in (.85, 1.15):
            horizontal = np.clip(
                relative[:2] * xy_scale
                / (gain_m_per_unit_step * move_steps),
                -.9, .9,
            )
            descend = np.clip(
                (relative[2] - raised_m) * z_scale
                / (gain_m_per_unit_step * down_steps),
                -.7, -.12,
            )
            name = f"relative_xy{int(xy_scale * 100):03d}_z{int(z_scale * 100):03d}"
            candidates[name] = np.stack(
                [action(z=up_action)] * up_steps
                + [action(x=horizontal[0], y=horizontal[1])] * move_steps
                + [action(z=descend)] * down_steps
                + [action(gripper=1.)] * 12
                + [action(z=.25, gripper=1.)] * 10
            )
    return candidates


def cup_place_macros():
    """Long candidates that complete mug grasp, transport, and release."""
    # At task-6 episode 2's natural stall the gripper is already aligned with
    # the mug. This prefix is independently verified after controller reset.
    grasp = candidate_macros()["reclose_lift"]
    transports = {
        "place_fine_250_150": (.25, .15),
        "place_fine_260_155": (.26, .155),
        "place_fine_270_160": (.27, .16),
        "place_fine_280_170": (.28, .17),
        "place_fine_250_170": (.25, .17),
        "place_fine_270_140": (.27, .14),
        "place_fine_300_180": (.30, .18),
        "place_fine_300_150": (.30, .15),
    }
    candidates = {}
    for name, (dx, dy) in transports.items():
        candidates[name] = np.concatenate((
            grasp,
            np.stack([action(x=dx, y=dy, gripper=1.)] * 80),
            np.stack([action(z=-.35, gripper=1.)] * 8),
            np.stack([action(gripper=-1.)] * 10),
        ))
    return candidates


def stable_cup_grasp_macros():
    """Verify cup grasps with a slower, longer lift before any transport."""
    parameters = {
        "stable_center": (0., 0., 0.),
        "stable_diag_pm": (.22, -.22, 0.),
        "stable_diag_pm_rminus": (.22, -.22, -.20),
        "stable_diag_pm_rplus": (.22, -.22, .20),
        "stable_y_minus": (0., -.30, 0.),
        "stable_x_plus": (.30, 0., 0.),
        "stable_diag_strong_pm": (.35, -.30, 0.),
        "stable_diag_weak_pm": (.14, -.14, 0.),
    }
    candidates = {}
    for name, (dx, dy, rz) in parameters.items():
        candidates[name] = np.stack(
            [action(z=.25)] * 4
            + [np.asarray([dx, dy, 0., 0., 0., rz, -1.], np.float32)] * 6
            + [action(z=-.25)] * 6
            + [action(gripper=1.)] * 12
            + [action(z=.15, gripper=1.)] * 20
        )
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

    if (args.source_rollout is None) == (args.source_sim_state is None):
        raise ValueError("provide exactly one source rollout or source sim state")
    by_timestep = {}
    if args.source_rollout is not None:
        rows = [
            json.loads(line)
            for line in (
                args.source_rollout / "steps.jsonl"
            ).read_text().splitlines()
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
    target_key = PUDDING if args.target_role == "pudding" else MUG
    target_name = (
        "chocolate_pudding_1"
        if args.target_role == "pudding" else "porcelain_mug_1"
    )
    collateral_key = MUG if args.target_role == "pudding" else PUDDING
    noop = action()
    try:
        if args.source_sim_state is not None:
            sim_state = np.load(args.source_sim_state)
            obs = source_env.regenerate_obs_from_state(sim_state)
            controller = source_env.env.robots[0].controller
            controller.update(force=True)
            controller.reset_goal()
        else:
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
        source_env.close()
        source_env = None
        args.output_dir.mkdir(parents=True, exist_ok=True)
        outcomes = []
        macro_factory = {
            "short": candidate_macros,
            "safe-approach": safe_approach_macros,
            "cup-stable-grasp": stable_cup_grasp_macros,
            "cup-place": cup_place_macros,
        }.get(args.macro_kind)
        macros = (
            relative_regrasp_macros(before[target_key] - before_eef)
            if args.macro_kind == "relative-regrasp"
            else macro_factory()
        )
        if args.macro_kind.startswith("cup-") and args.target_role != "mug":
            raise ValueError("cup macros require --target-role mug")
        for name, chunk in macros.items():
            # A flattened MuJoCo state does not contain robosuite controller
            # goals. Reusing one fork across candidates leaks controller
            # history and corrupts later counterfactual labels.
            branch_env = make_env()
            try:
                branch_obs = branch_env.regenerate_obs_from_state(sim_state)
                controller = branch_env.env.robots[0].controller
                controller.update(force=True)
                controller.reset_goal()
                restored = object_positions(branch_obs)
                restore_error = max(
                    float(np.abs(restored[key] - before[key]).max())
                    for key in before
                )
                branch_env.env.timestep = 0
                branch_env.env.done = False
                frames = [np.ascontiguousarray(
                    branch_obs["agentview_image"][::-1, ::-1]
                )]
                max_collateral_displacement = 0.0
                min_eef_target_distance = float("inf")
                attached_ever = False
                for candidate_action in chunk:
                    branch_obs, _, done, _ = branch_env.step(
                        candidate_action.tolist()
                    )
                    frames.append(np.ascontiguousarray(
                        branch_obs["agentview_image"][::-1, ::-1]
                    ))
                    step_positions = object_positions(branch_obs)
                    step_eef = np.asarray(
                        branch_obs["robot0_eef_pos"], np.float32
                    )
                    min_eef_target_distance = min(
                        min_eef_target_distance,
                        float(np.linalg.norm(
                            step_eef - step_positions[target_key]
                        )),
                    )
                    _, step_attached = extract_target_gripper_contact(
                        branch_env.env.sim, target_name=target_name
                    )
                    attached_ever = attached_ever or step_attached
                    max_collateral_displacement = max(
                        max_collateral_displacement,
                        float(np.linalg.norm(
                            step_positions[collateral_key]
                            - before[collateral_key]
                        )),
                    )
                    if done:
                        break
                after = object_positions(branch_obs)
                after_eef = np.asarray(
                    branch_obs["robot0_eef_pos"], np.float32
                )
                _, attached_after = extract_target_gripper_contact(
                    branch_env.env.sim, target_name=target_name
                )
                mug_on_plate = bool(branch_env.env._eval_predicate(
                    ["on", "porcelain_mug_1", "plate_1"]
                ))
                task_success_after = bool(branch_env.check_success())
            finally:
                branch_env.close()
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
                - 20 * max_collateral_displacement
                - float(args.target_role == "pudding" and not mug_on_plate)
                + 5 * float(args.target_role == "mug" and mug_on_plate)
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
                "target_role": args.target_role,
                "macro_kind": args.macro_kind,
                "target_displacement_m": target_displacement,
                "target_lift_m": target_lift,
                "target_goal_progress_m": goal_progress,
                "collateral_displacement_m": collateral_displacement,
                "max_collateral_displacement_m":
                    max_collateral_displacement,
                # Backward-compatible task-6 pudding fields.
                "pudding_displacement_m": target_displacement,
                "pudding_lift_m": target_lift,
                "pudding_goal_progress_m": goal_progress,
                "mug_displacement_m": collateral_displacement,
                "mug_on_plate_after": mug_on_plate,
                "eef_target_distance_after_m": eef_target_distance,
                "min_eef_target_distance_m": min_eef_target_distance,
                "task_success_after": task_success_after,
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
                "source_rollout": (
                    str(args.source_rollout)
                    if args.source_rollout is not None else None
                ),
                "source_sim_state": (
                    str(args.source_sim_state)
                    if args.source_sim_state is not None else None
                ),
                "model_input_excludes_privileged_outcomes": True,
                "same_sim_state_for_all_candidates": True,
                "independent_controller_per_candidate": True,
                "controller_goal_reset_after_restore": True,
                "before": {
                    "target_goal_distance_m": before_distance,
                    "attached": bool(attached_before),
                    "eef_position": before_eef.tolist(),
                    "target_position": before[target_key].tolist(),
                    "target_relative_to_eef_m":
                        (before[target_key] - before_eef).tolist(),
                },
                "candidates": outcomes,
            }, indent=2),
            encoding="utf-8",
        )
    finally:
        if source_env is not None:
            source_env.close()


if __name__ == "__main__":
    main()
