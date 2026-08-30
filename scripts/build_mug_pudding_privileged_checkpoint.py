#!/usr/bin/env python3
"""Write the explicit parameters of the privileged mug+pudding controller."""

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = {
        "format": "mug_pudding_privileged_cartesian_v1",
        "controller": {
            "wait_steps": 10,
            "position_gain": 18.0,
            "max_translation_action": 0.7,
            "position_tolerance_m": 0.006,
            "target_eef_quaternion_xyzw": [0.9996, 0.0, -0.0283, 0.0],
            "orientation_gain": 3.0,
            "max_rotation_action": 0.5,
            "orientation_tolerance_rad": 0.02,
            "max_move_steps": 100,
            "approach_height_m": 0.10,
            "lift_height_m": 0.14,
            "retreat_height_m": 0.10,
            "close_steps": 20,
            "open_steps": 15,
            "settle_steps": 20,
            # Empirical gripper-center offset measured from successful task-6
            # traces; the cup body is approached from its handle-free side.
            "mug_grasp_offset_m": [0.0, 0.025, 0.10],
            "mug_center_above_plate_m": 0.012,
            "pudding_grasp_offset_m": [-0.003, -0.015, 0.016],
            "pudding_right_offset_xy_m": [0.0, 0.10],
        },
        "scope_warning": "privileged task-specific overfit controller; standard robot actions only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
