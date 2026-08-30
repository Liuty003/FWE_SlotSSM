#!/usr/bin/env python3
"""Collect fresh V4 nominal rollouts with RGB-D and calibration.

This collector intentionally does not accept a legacy RGB-only rollout root.
It writes policy-visible RGB-D/proprio data plus fixed agent-view calibration;
instance segmentation is stored separately as training-only Slot supervision.
The frozen OFT policy itself continues to receive exactly its standard RGB,
wrist-RGB, language, and proprio input.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-ids", required=True, help="Comma-separated task ids.")
    parser.add_argument("--episodes", default="0")
    parser.add_argument("--open-loop-steps", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--oft-root", type=Path, default=Path("/data/lty/code/openvla-oft-official"))
    parser.add_argument("--libero-root", type=Path, default=Path("/data/lty/code/LIBERO-plus"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skip-checkpoint-config-sync", action="store_true")
    return parser.parse_args()


def calibration(env, camera_name: str, height: int, width: int, np):
    """Return a camera-to-robot-base transform and raw depth conversion bounds."""
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    sim = env.env.sim
    camera_to_world = np.linalg.inv(get_camera_transform_matrix(sim, camera_name, height, width))
    root_body = env.env.robots[0].robot_model.root_body
    body_id = sim.model.body_name2id(root_body)
    robot_to_world = np.eye(4)
    robot_to_world[:3, :3] = sim.data.body_xmat[body_id].reshape(3, 3)
    robot_to_world[:3, 3] = sim.data.body_xpos[body_id]
    world_to_robot = np.linalg.inv(robot_to_world)
    extent = sim.model.stat.extent
    return {
        "camera_name": camera_name, "height": height, "width": width,
        "camera_to_robot": (world_to_robot @ camera_to_world).tolist(),
        "world_to_robot": world_to_robot.tolist(),
        "near": float(sim.model.vis.map.znear * extent),
        "far": float(sim.model.vis.map.zfar * extent),
        "saved_orientation": "RGB/depth are [::-1, ::-1]; native_x=width-1-saved_x; native_y=saved_y",
    }


def depth_and_segmentation(obs, camera_name: str, np):
    depth_key = f"{camera_name}_depth"
    if depth_key not in obs:
        raise RuntimeError(f"environment did not return {depth_key}; RGB-D collection cannot continue")
    depth = np.asarray(obs[depth_key]).squeeze()
    if depth.ndim != 2:
        raise RuntimeError(f"{depth_key} must be [H,W], got {depth.shape}")
    segmentation_key = f"{camera_name}_segmentation_instance"
    segmentation = obs.get(segmentation_key)
    if segmentation is not None:
        segmentation = np.asarray(segmentation).squeeze()
        if segmentation.shape != depth.shape:
            raise RuntimeError(f"{segmentation_key} does not align with depth")
    return depth[::-1, ::-1].copy(), None if segmentation is None else segmentation[::-1, ::-1].copy()


def main() -> None:
    args = parse_args()
    if args.open_loop_steps < 1:
        raise ValueError("--open-loop-steps must be positive")
    for path in (args.oft_root, args.libero_root / "libero" / "libero", args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    sys.path[:0] = [str(args.libero_root), str(args.oft_root)]
    os.chdir(args.oft_root)

    import imageio.v2 as imageio
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv
    import experiments.robot.openvla_utils as openvla_utils
    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_image, get_libero_wrist_image
    from experiments.robot.libero.run_libero_eval import TASK_MAX_STEPS, GenerateConfig, get_image_resize_size, initialize_model, prepare_observation, process_action, set_seed_everywhere, validate_config
    from experiments.robot.robot_utils import get_action

    if args.skip_checkpoint_config_sync:
        config = json.loads((args.checkpoint / "config.json").read_text())
        if config.get("auto_map") != {
            "AutoConfig": "configuration_prismatic.OpenVLAConfig",
            "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
        }:
            raise ValueError("checkpoint auto_map is not ready for parallel collection")
        openvla_utils.update_auto_map = lambda _checkpoint: None
    cfg = GenerateConfig(pretrained_checkpoint=str(args.checkpoint), task_suite_name=args.task_suite,
                         num_trials_per_task=1, num_open_loop_steps=args.open_loop_steps, use_wandb=False)
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    if args.skip_checkpoint_config_sync:
        model_bundle = initialize_model(cfg)
    else:
        # OpenVLA's update_auto_map() rewrites checkpoint/config.json in place.
        # Multiple collectors used to race here and one process could observe an
        # empty / partially-written JSON file.  Serialize that one-time config
        # update (and the immediately following config read during model load).
        lock_path = args.checkpoint / ".v4-collector-config-sync.lock"
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                model_bundle = initialize_model(cfg)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    model, action_head, proprio_projector, noisy_action_projector, processor = model_bundle
    resize_size = get_image_resize_size(cfg)
    task_ids = [int(value) for value in args.task_ids.split(",") if value.strip()]
    episodes = [int(value) for value in args.episodes.split(",") if value.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)

    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = SegmentationRenderEnv(
            bddl_file_name=str(bddl), camera_heights=cfg.env_img_res, camera_widths=cfg.env_img_res,
            camera_depths=True, camera_segmentations="instance",
        )
        env.seed(0)
        try:
            for episode_id in episodes:
                if not 0 <= episode_id < len(initial_states):
                    raise ValueError(f"invalid episode {episode_id} for task {task_id}")
                output = args.output_root / f"task{task_id}-episode{episode_id}"
                directories = {name: output / name for name in ("rgb", "wrist_rgb", "depth", "wrist_depth", "instance")}
                for directory in directories.values():
                    directory.mkdir(parents=True, exist_ok=True)
                env.reset()
                obs = env.set_init_state(initial_states[episode_id])
                agent_calibration = calibration(env, "agentview", cfg.env_img_res, cfg.env_img_res, np)
                records, video, queue, success = [], [], deque(), False
                for timestep in range(TASK_MAX_STEPS[cfg.task_suite_name] + cfg.num_steps_wait):
                    if timestep < cfg.num_steps_wait:
                        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                        continue
                    if not queue:
                        policy_obs, _ = prepare_observation(obs, resize_size)
                        queue.extend(get_action(cfg, model, policy_obs, task.language, processor, action_head,
                                                proprio_projector, noisy_action_projector, cfg.use_film))
                    raw_action = np.asarray(queue.popleft(), dtype=np.float32)
                    frame = f"step-{len(records):04d}"
                    main, wrist = get_libero_image(obs), get_libero_wrist_image(obs)
                    main_depth, segmentation = depth_and_segmentation(obs, "agentview", np)
                    wrist_depth, _ = depth_and_segmentation(obs, "robot0_eye_in_hand", np)
                    rgb_path, wrist_path = directories["rgb"] / f"{frame}.png", directories["wrist_rgb"] / f"{frame}.png"
                    depth_path, wrist_depth_path = directories["depth"] / f"{frame}.npz", directories["wrist_depth"] / f"{frame}.npz"
                    imageio.imwrite(rgb_path, main); imageio.imwrite(wrist_path, wrist)
                    np.savez_compressed(depth_path, depth=main_depth); np.savez_compressed(wrist_depth_path, depth=wrist_depth)
                    instance_path = None
                    if segmentation is not None:
                        instance_path = directories["instance"] / f"{frame}.npz"
                        np.savez_compressed(instance_path, instance=segmentation.astype(np.int32))
                    executed = process_action(raw_action, cfg.model_family)
                    records.append({
                        "kind": "v4_oft_rgbd_nominal_transition", "task_suite": args.task_suite,
                        "task_id": task_id, "episode_id": episode_id, "timestep": len(records),
                        "online_observation": {
                            "rgb": str(rgb_path.resolve()), "wrist_rgb": str(wrist_path.resolve()),
                            "depth": str(depth_path.resolve()), "wrist_depth": str(wrist_depth_path.resolve()),
                            "proprio": {"robot0_eef_pos": np.asarray(obs["robot0_eef_pos"]).tolist(),
                                         "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"]).tolist(),
                                         "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"]).tolist()},
                        },
                        "training_only": {"instance_segmentation": None if instance_path is None else str(instance_path.resolve())},
                        "planned_action": raw_action.tolist(), "executed_action": executed.tolist(), "gate_label": 0,
                    })
                    video.append(main)
                    obs, _, done, _ = env.step(executed.tolist())
                    if done:
                        success = True
                        break
                metadata = {
                    "kind": "v4_oft_rgbd_nominal_rollout", "schema": "v4_rgbd_rollout_v2",
                    "task_suite": args.task_suite, "task_id": task_id, "episode_id": episode_id,
                    "task_language": task.language, "success": success, "transitions": len(records),
                    "wait_steps": cfg.num_steps_wait, "open_loop_steps": cfg.num_open_loop_steps,
                    "rollout_id": f"nominal:{args.task_suite}:{task_id}:{episode_id}",
                    "libero_root": str(args.libero_root.resolve()), "oft_root": str(args.oft_root.resolve()),
                    "oft_checkpoint": str(args.checkpoint.resolve()), "env_seed": 0,
                    "geometry_view": "agentview", "agentview_calibration": agent_calibration,
                    "online_inputs_permitted": ["main_rgb", "wrist_rgb", "main_depth", "wrist_depth", "camera_calibration", "language", "proprio"],
                    "training_only_inputs": ["instance_segmentation"],
                }
                (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                with (output / "steps.jsonl").open("w") as stream:
                    for row in records:
                        stream.write(json.dumps(row) + "\n")
                if success:
                    (output / "transitions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
                else:
                    (output / "REJECTED_NOT_NORMAL.txt").write_text("Failed nominal rollout: labels are not NORMAL training examples.\n")
                imageio.mimsave(output / "rollout.mp4", video, fps=30)
                print(json.dumps(metadata), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
