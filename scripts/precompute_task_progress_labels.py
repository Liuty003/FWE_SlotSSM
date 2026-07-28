#!/usr/bin/env python
"""Build task-aware dense progress labels for LIBERO Spatial counterfactual forks."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


BRANCHES = (
    "nominal", "translate_x_minus", "translate_x_plus",
    "translate_y_minus", "translate_y_plus", "translate_z_minus",
    "translate_z_plus", "rotate_z_minus", "rotate_z_plus", "flip_gripper",
)
TARGET_KEY = "akita_black_bowl_1_pos"
GOAL_KEY = "plate_1_pos"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counterfactual-root", required=True)
    parser.add_argument("--slot-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--episode-ids", default="0,1,2,3,4")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--lift-threshold-m", type=float, default=0.015)
    return parser.parse_args()


def vector(state, key):
    return np.asarray(state[key], dtype=np.float32)


def main():
    args = arguments()
    counterfactual_root, slot_dir = Path(args.counterfactual_root), Path(args.slot_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"target_key": TARGET_KEY, "goal_key": GOAL_KEY, "shards": []}
    for task_id in [int(value) for value in args.task_ids.split(",") if value]:
        for episode_id in [int(value) for value in args.episode_ids.split(",") if value]:
            episode = counterfactual_root / f"libero_spatial_t{task_id}_e{episode_id}_h4_dual"
            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            grouped = defaultdict(dict)
            for line in (episode / "transitions.jsonl").open(encoding="utf-8"):
                row = json.loads(line)
                grouped[int(row["fork_id"])][row["branch"]] = row
            slots = load_file(str(slot_dir / f"libero_spatial_t{task_id}_e{episode_id}_slots.safetensors"))
            fork_ids = slots["fork_ids"].tolist()
            first = grouped[min(grouped)]["nominal"]
            initial_target_z = float(vector(first["privileged_object_state_before"], TARGET_KEY)[2])
            progress, eef_progress, goal_progress, phases, rewards = [], [], [], [], []
            for fork_id in fork_ids:
                branches = grouped[int(fork_id)]
                nominal = branches["nominal"]
                before_objects = nominal["privileged_object_state_before"]
                before_robot = nominal["robot_state_before"]
                target_before = vector(before_objects, TARGET_KEY)
                goal_before = vector(before_objects, GOAL_KEY)
                eef_before = vector(before_robot, "robot0_eef_pos")
                lifted = bool(target_before[2] > initial_target_z + args.lift_threshold_m)
                phases.append(int(lifted))
                fork_progress, fork_eef, fork_goal, fork_reward = [], [], [], []
                for name in BRANCHES:
                    row = branches[name]
                    trajectory = list(row["branch_trajectory"])
                    if not trajectory:
                        raise ValueError(f"empty trajectory task={task_id} episode={episode_id} fork={fork_id} branch={name}")
                    trajectory += [trajectory[-1]] * (args.horizon - len(trajectory))
                    trajectory = trajectory[:args.horizon]
                    branch_progress, branch_eef, branch_goal, branch_reward = [], [], [], []
                    before_eef_distance = np.linalg.norm(eef_before - target_before)
                    before_goal_distance = np.linalg.norm(target_before - goal_before)
                    for transition in trajectory:
                        after_objects = transition["privileged_object_state_after"]
                        after_robot = transition["robot_state_after"]
                        target_after = vector(after_objects, TARGET_KEY)
                        goal_after = vector(after_objects, GOAL_KEY)
                        eef_after = vector(after_robot, "robot0_eef_pos")
                        current_eef_progress = before_eef_distance - np.linalg.norm(eef_after - target_after)
                        current_goal_progress = before_goal_distance - np.linalg.norm(target_after - goal_after)
                        current_progress = current_goal_progress if lifted else current_eef_progress
                        branch_eef.append(current_eef_progress)
                        branch_goal.append(current_goal_progress)
                        branch_progress.append(current_progress)
                        branch_reward.append(float(transition["reward"]))
                    fork_eef.append(branch_eef)
                    fork_goal.append(branch_goal)
                    fork_progress.append(branch_progress)
                    fork_reward.append(branch_reward)
                eef_progress.append(fork_eef)
                goal_progress.append(fork_goal)
                progress.append(fork_progress)
                rewards.append(fork_reward)
            tensors = {
                "task_progress_m": torch.tensor(np.asarray(progress), dtype=torch.float32),
                "eef_to_target_progress_m": torch.tensor(np.asarray(eef_progress), dtype=torch.float32),
                "target_to_goal_progress_m": torch.tensor(np.asarray(goal_progress), dtype=torch.float32),
                "lifted_phase": torch.tensor(phases, dtype=torch.bool),
                "branch_reward": torch.tensor(np.asarray(rewards), dtype=torch.float32),
                "fork_ids": torch.tensor(fork_ids, dtype=torch.int64),
            }
            output = output_dir / f"libero_spatial_t{task_id}_e{episode_id}_progress.safetensors"
            save_file({key: value.contiguous() for key, value in tensors.items()}, str(output))
            manifest["shards"].append({
                "task_id": task_id, "episode_id": episode_id,
                "task_language": metadata["task_language"], "output": str(output),
                "forks": len(fork_ids), "lifted_forks": int(sum(phases)),
            })
            print(f"saved={output} forks={len(fork_ids)} lifted={sum(phases)}", flush=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest={output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
