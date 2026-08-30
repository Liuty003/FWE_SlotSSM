#!/usr/bin/env python3
"""Collect auditable, dual-camera OFT rollouts.

Every rollout writes a crash-safe ``steps.jsonl`` with policy-visible inputs,
the planned OFT action and the action actually passed to LIBERO.  Only a
complete success is additionally promoted to ``transitions.jsonl`` for NORMAL
gate supervision; failed rollouts remain available for failure audit and fork
selection and are never silently labelled NORMAL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import deque
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=6)
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Optional comma-separated task ids. Reuses one loaded OFT model across the audit set.",
    )
    parser.add_argument("--episodes", default="0,1,2,3")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Independent replays of every requested task/episode pair for source-stability screening.",
    )
    parser.add_argument("--open-loop-steps", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--oft-root", type=Path, default=Path("/data/lty/code/openvla-oft-official"))
    parser.add_argument("--libero-root", type=Path, default=Path(__file__).resolve().parents[1] / "experiments" / "robot" / "libero")
    parser.add_argument("--checkpoint", type=Path, default=Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-10"))
    parser.add_argument(
        "--skip-checkpoint-config-sync", action="store_true",
        help=("Do not rewrite checkpoint config.json during model initialization. Safe only when its "
              "auto_map already names OpenVLAConfig/OpenVLAForActionPrediction; use for parallel collectors."),
    )
    parser.add_argument("--output-root", type=Path, default=Path("/data2/lty/naive-e-slotssm-runs/oft-nominal-seed-v1"))
    parser.add_argument(
        "--save-frame-images",
        action="store_true",
        help="Also save per-step main/wrist PNGs.  The default audit mode keeps MP4 and state/action logs only.",
    )
    parser.add_argument(
        "--save-privileged-audit", action="store_true",
        help=("Save simulator object poses, joints, and per-goal BDDL predicate truth values. "
              "These fields are offline labels only and must never be policy inputs."),
    )
    parser.add_argument(
        "--save-dense-audit", action="store_true",
        help=(
            "Save a compressed per-rollout bundle containing lossless dual RGB, dual depth, "
            "instance segmentation, and pre-action plus terminal MuJoCo states. Implies "
            "--save-privileged-audit and is the archival schema for recovery-data mining."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip a rollout only when its metadata and steps files are both complete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.save_dense_audit:
        args.save_privileged_audit = True
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    for path in (args.oft_root, args.libero_root / "libero" / "libero", args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    # Respect a caller-provided device mask.  This makes concurrent collectors
    # safe: each process can receive one physical GPU via CUDA_VISIBLE_DEVICES
    # and use its local device 0, while the standalone CLI still honours
    # --gpu when no mask was supplied.
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    # Do not inherit an unrelated LIBERO install from the shell: it can trigger
    # interactive configuration and silently point at another benchmark copy.
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    # LIBERO-Plus must precede OFT's vendored 10-task LIBERO package.  The
    # latter has the same import name and silently accepts ``libero_10`` but
    # rejects all official Plus task ids (491+).  Keep OFT available for its
    # evaluator utilities, but resolve ``libero.libero`` from --libero-root.
    sys.path[:0] = [str(args.libero_root), str(args.oft_root)]
    os.chdir(args.oft_root)

    import imageio.v2 as imageio
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import experiments.robot.openvla_utils as openvla_utils
    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image, get_libero_wrist_image
    from experiments.robot.libero.run_libero_eval import TASK_MAX_STEPS, GenerateConfig, get_image_resize_size, initialize_model, prepare_observation, process_action, set_seed_everywhere, validate_config
    from experiments.robot.robot_utils import get_action

    def contact_pairs(sim):
        pairs = []
        for index in range(int(sim.data.ncon)):
            contact = sim.data.contact[index]
            pairs.append({
                "geom1": sim.model.geom_id2name(int(contact.geom1)),
                "geom2": sim.model.geom_id2name(int(contact.geom2)),
                "position": np.asarray(contact.pos).tolist(),
                "distance": float(contact.dist),
            })
        return pairs

    def privileged_audit(env):
        """Return task-agnostic simulator labels for automatic failure mining."""
        inner = env.env
        goal_states = [list(state) for state in inner.parsed_problem["goal_state"]]
        goal_truth = []
        for state in goal_states:
            try:
                value = bool(inner._eval_predicate(state))
            except Exception as error:  # Preserve collection; mark unsupported predicates explicitly.
                value = None
                error_text = f"{type(error).__name__}: {error}"
            else:
                error_text = None
            goal_truth.append({"predicate": state, "value": value, "error": error_text})

        entities = {}
        for entity_kind, mapping in (
            ("object", inner.objects_dict), ("fixture", inner.fixtures_dict),
        ):
            for name, entity in mapping.items():
                body_name = entity.root_body
                body_id = inner.sim.model.body_name2id(body_name)
                joints = {}
                for joint_name in getattr(entity, "joints", ()):
                    value = np.asarray(inner.sim.data.get_joint_qpos(joint_name))
                    joints[joint_name] = value.reshape(-1).tolist()
                entities[name] = {
                    "kind": entity_kind,
                    "body_name": body_name,
                    "position": np.asarray(inner.sim.data.body_xpos[body_id]).tolist(),
                    "quaternion_wxyz": np.asarray(inner.sim.data.body_xquat[body_id]).tolist(),
                    "joint_qpos": joints,
                }
        return {
            "offline_privileged_only": True,
            "goal_predicates": goal_truth,
            "task_success": bool(inner._check_success()),
            "objects_of_interest": list(inner.obj_of_interest),
            "entities": entities,
            "contacts": contact_pairs(inner.sim),
        }

    def calibration(env, camera_name: str, height: int, width: int):
        from robosuite.utils.camera_utils import get_camera_transform_matrix

        sim = env.env.sim
        camera_to_world = np.linalg.inv(
            get_camera_transform_matrix(sim, camera_name, height, width)
        )
        root_body = env.env.robots[0].robot_model.root_body
        body_id = sim.model.body_name2id(root_body)
        robot_to_world = np.eye(4)
        robot_to_world[:3, :3] = sim.data.body_xmat[body_id].reshape(3, 3)
        robot_to_world[:3, 3] = sim.data.body_xpos[body_id]
        extent = sim.model.stat.extent
        return {
            "camera_name": camera_name,
            "height": height,
            "width": width,
            "camera_to_robot": (np.linalg.inv(robot_to_world) @ camera_to_world).tolist(),
            "near": float(sim.model.vis.map.znear * extent),
            "far": float(sim.model.vis.map.zfar * extent),
            "saved_orientation": "RGB/depth/segmentation are rotated 180 degrees from native observations",
        }

    def oriented_optional(obs, key, *, squeeze=False):
        value = obs.get(key)
        if value is None:
            return None
        array = np.asarray(value)
        if squeeze:
            array = array.squeeze()
        return np.ascontiguousarray(array[::-1, ::-1])

    def simulator_state(env):
        sim = env.env.sim
        return {
            "mujoco_state": np.asarray(sim.get_state().flatten(), dtype=np.float64),
            "ctrl": np.asarray(sim.data.ctrl, dtype=np.float64).copy(),
            "mocap_pos": np.asarray(sim.data.mocap_pos, dtype=np.float64).copy(),
            "mocap_quat": np.asarray(sim.data.mocap_quat, dtype=np.float64).copy(),
            "camera_xpos": np.asarray(sim.data.cam_xpos, dtype=np.float64).copy(),
            "camera_xmat": np.asarray(sim.data.cam_xmat, dtype=np.float64).copy(),
        }

    def numpy_rng_state():
        algorithm, keys, position, has_gaussian, cached_gaussian = np.random.get_state()
        if algorithm != "MT19937":
            raise RuntimeError(f"unsupported numpy RNG algorithm: {algorithm}")
        return {
            "numpy_rng_keys": np.asarray(keys, dtype=np.uint32),
            "numpy_rng_position": np.asarray(position, dtype=np.int64),
            "numpy_rng_has_gaussian": np.asarray(has_gaussian, dtype=np.int8),
            "numpy_rng_cached_gaussian": np.asarray(cached_gaussian, dtype=np.float64),
        }

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    if args.skip_checkpoint_config_sync:
        config = json.loads((args.checkpoint / "config.json").read_text())
        expected = {
            "AutoConfig": "configuration_prismatic.OpenVLAConfig",
            "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
        }
        if config.get("auto_map") != expected:
            raise ValueError(
                "--skip-checkpoint-config-sync requires the checkpoint config.json to already contain "
                "the OpenVLA auto_map; run one non-parallel initialization first"
            )
        # get_vla resolves update_auto_map from this module at call time.  Avoid
        # concurrent backup/write races on a shared local checkpoint.
        openvla_utils.update_auto_map = lambda _checkpoint: None

    cfg = GenerateConfig(pretrained_checkpoint=str(args.checkpoint), task_suite_name=args.task_suite,
                         num_trials_per_task=1, num_open_loop_steps=args.open_loop_steps, use_wandb=False)
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    requested = [int(value) for value in args.episodes.split(",") if value.strip()]
    task_ids = [int(value) for value in (args.task_ids or str(args.task_id)).split(",") if value.strip()]
    if not task_ids:
        raise ValueError("at least one task id is required")
    args.output_root.mkdir(parents=True, exist_ok=True)

    try:
        for task_id in task_ids:
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            if args.save_dense_audit:
                task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                # Use ControlEnv's own step implementation so official sensor-noise
                # perturbations remain active. SegmentationRenderEnv overrides step
                # and would accidentally bypass LIBERO-Plus noise post-processing.
                env = OffScreenRenderEnv(
                    bddl_file_name=str(task_bddl_file),
                    camera_heights=cfg.env_img_res,
                    camera_widths=cfg.env_img_res,
                    camera_depths=True,
                    camera_segmentations="instance",
                )
                env.seed(0)
                task_text = task.language
            else:
                env, task_text = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            try:
                for episode_id in requested:
                    if not 0 <= episode_id < len(initial_states):
                        raise ValueError(f"invalid episode {episode_id}")
                    for repeat_id in range(args.repeats):
                        suffix = f"-repeat{repeat_id}" if args.repeats > 1 else ""
                        episode_dir = args.output_root / f"task{task_id}-episode{episode_id}{suffix}"
                        existing_metadata = None
                        if (episode_dir / "metadata.json").is_file():
                            try:
                                existing_metadata = json.loads((episode_dir / "metadata.json").read_text())
                            except json.JSONDecodeError:
                                existing_metadata = None
                        complete_existing = (
                            existing_metadata is not None
                            and (episode_dir / "steps.jsonl").is_file()
                            and (
                                not args.save_dense_audit
                                or (
                                    existing_metadata.get("schema") == "oft_archival_rollout_v4"
                                    and (episode_dir / "dense-audit.npz").is_file()
                                    and existing_metadata.get("complete") is True
                                )
                            )
                        )
                        if args.resume and complete_existing:
                            print(json.dumps({
                                "kind": "oft_nominal_rollout_skip", "task_suite": args.task_suite,
                                "task_id": task_id, "episode_id": episode_id,
                                "repeat_id": repeat_id, "reason": "complete_existing",
                            }), flush=True)
                            continue
                        rgb_dir, wrist_dir = episode_dir / "rgb", episode_dir / "wrist_rgb"
                        if args.save_frame_images:
                            rgb_dir.mkdir(parents=True, exist_ok=True)
                            wrist_dir.mkdir(parents=True, exist_ok=True)
                        else:
                            episode_dir.mkdir(parents=True, exist_ok=True)
                        records: list[dict] = []
                        video: list[np.ndarray] = []
                        wrist_video: list[np.ndarray] = []
                        dense: dict[str, list[np.ndarray]] = {
                            key: [] for key in (
                                "rgb", "wrist_rgb", "depth", "wrist_depth",
                                "agentview_instance", "wrist_instance", "mujoco_state", "ctrl",
                                "mocap_pos", "mocap_quat", "camera_xpos", "camera_xmat",
                                "numpy_rng_keys", "numpy_rng_position",
                                "numpy_rng_has_gaussian", "numpy_rng_cached_gaussian",
                            )
                        }
                        env.reset()
                        obs = env.set_init_state(initial_states[episode_id])
                        initial_state = np.asarray(initial_states[episode_id], dtype=np.float64).copy()
                        camera_calibration = None
                        if args.save_dense_audit:
                            camera_calibration = {
                                name: calibration(env, name, cfg.env_img_res, cfg.env_img_res)
                                for name in ("agentview", "robot0_eye_in_hand")
                            }
                        queue = deque()
                        active_chunk = None
                        chunk_id = -1
                        chunk_offset = 0
                        success = False
                        for t in range(TASK_MAX_STEPS[cfg.task_suite_name] + cfg.num_steps_wait):
                            if t < cfg.num_steps_wait:
                                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                                continue
                            if not queue:
                                policy_obs, _ = prepare_observation(obs, resize_size)
                                active_chunk = np.asarray(
                                    get_action(cfg, model, policy_obs, task_text, processor, action_head,
                                               proprio_projector, noisy_action_projector, cfg.use_film),
                                    dtype=np.float32,
                                )
                                queue.extend(active_chunk)
                                chunk_id += 1
                                chunk_offset = 0
                            raw_action = np.asarray(queue.popleft(), dtype=np.float32)
                            frame = f"step-{len(records):04d}"
                            rgb = get_libero_image(obs)
                            wrist = get_libero_wrist_image(obs)
                            if args.save_frame_images:
                                imageio.imwrite(rgb_dir / f"{frame}.png", rgb)
                                imageio.imwrite(wrist_dir / f"{frame}.png", wrist)
                            video.append(rgb)
                            wrist_video.append(wrist)
                            bundle_path = episode_dir / "dense-audit.npz"
                            if args.save_dense_audit:
                                dense["rgb"].append(np.asarray(rgb, dtype=np.uint8))
                                dense["wrist_rgb"].append(np.asarray(wrist, dtype=np.uint8))
                                dense["depth"].append(oriented_optional(obs, "agentview_depth", squeeze=True).astype(np.float32))
                                dense["wrist_depth"].append(oriented_optional(obs, "robot0_eye_in_hand_depth", squeeze=True).astype(np.float32))
                                main_instance = oriented_optional(obs, "agentview_segmentation_instance", squeeze=True)
                                wrist_instance = oriented_optional(obs, "robot0_eye_in_hand_segmentation_instance", squeeze=True)
                                if main_instance is None or wrist_instance is None:
                                    raise RuntimeError("dense archival collection requires aligned instance segmentation")
                                dense["agentview_instance"].append(main_instance.astype(np.int32))
                                dense["wrist_instance"].append(wrist_instance.astype(np.int32))
                                for key, value in simulator_state(env).items():
                                    dense[key].append(value)
                                for key, value in numpy_rng_state().items():
                                    dense[key].append(value)
                            executed_action = process_action(raw_action, cfg.model_family)
                            record = {
                                "kind": "oft_nominal_transition", "task_suite": args.task_suite,
                                "task_id": task_id, "episode_id": episode_id, "repeat_id": repeat_id, "timestep": len(records),
                                "online_observation": {
                                    "rgb": str((rgb_dir / f"{frame}.png").resolve()) if args.save_frame_images else None,
                                    "wrist_rgb": str((wrist_dir / f"{frame}.png").resolve()) if args.save_frame_images else None,
                                    "dense_bundle": str(bundle_path.resolve()) if args.save_dense_audit else None,
                                    "dense_index": len(records) if args.save_dense_audit else None,
                                    "dense_keys": {"rgb": "rgb", "wrist_rgb": "wrist_rgb", "depth": "depth", "wrist_depth": "wrist_depth"} if args.save_dense_audit else None,
                                    "proprio": {
                                        key: np.asarray(obs[key]).tolist()
                                        for key in (
                                            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                                            "robot0_gripper_qvel", "robot0_joint_pos", "robot0_joint_vel",
                                        ) if key in obs
                                    },
                                },
                                # ``raw_action`` is the OFT action produced for this step;
                                # ``executed_action`` is the exact environment command
                                # after LIBERO convention conversion (notably gripper).
                                "planned_action": raw_action.tolist(),
                                "executed_action": executed_action.tolist(),
                                "action_chunk_id": chunk_id,
                                "action_chunk_offset": chunk_offset,
                                "planned_action_chunk": active_chunk.tolist() if chunk_offset == 0 else None,
                                "gate_label": 0,
                            }
                            if args.save_privileged_audit:
                                record["privileged_audit"] = privileged_audit(env)
                            records.append(record)
                            chunk_offset += 1
                            obs, _, done, _ = env.step(executed_action.tolist())
                            if done:
                                success = True
                                break
                        terminal_audit = privileged_audit(env) if args.save_privileged_audit else None
                        dense_sha256 = None
                        if args.save_dense_audit:
                            # Dense arrays have T policy-visible pre-action entries. The terminal
                            # simulator state is stored separately to preserve the exact post-action
                            # success/failure state.
                            terminal_state = simulator_state(env)
                            terminal_state.update(numpy_rng_state())
                            payload = {
                                key: np.stack(values, axis=0) for key, values in dense.items()
                            }
                            payload.update({f"terminal_{key}": value for key, value in terminal_state.items()})
                            payload["official_initial_state"] = initial_state
                            temporary_bundle = bundle_path.with_suffix(".npz.tmp")
                            with temporary_bundle.open("wb") as stream:
                                np.savez_compressed(stream, **payload)
                            os.replace(temporary_bundle, bundle_path)
                            dense_sha256 = sha256(bundle_path)
                        metadata = {
                            "kind": "oft_nominal_rollout", "task_suite": args.task_suite,
                            "task_id": task_id, "episode_id": episode_id, "repeat_id": repeat_id,
                            "task_language": task_text, "success": success, "transitions": len(records),
                            "schema": "oft_archival_rollout_v4" if args.save_dense_audit else ("oft_dual_rollout_v3_privileged_audit" if args.save_privileged_audit else "oft_dual_rollout_v2"),
                            "complete": True, "saved_frame_images": args.save_frame_images,
                            "saved_privileged_audit": args.save_privileged_audit,
                            "saved_dense_audit": args.save_dense_audit,
                            "dense_bundle": str(bundle_path.resolve()) if args.save_dense_audit else None,
                            "dense_bundle_sha256": dense_sha256,
                            "terminal_privileged_audit": terminal_audit,
                            "camera_calibration": camera_calibration,
                            "env_seed": 0, "policy_seed": cfg.seed,
                            "wait_steps": cfg.num_steps_wait,
                            "open_loop_steps": cfg.num_open_loop_steps,
                            "checkpoint": str(args.checkpoint.resolve()),
                            "libero_root": str(args.libero_root.resolve()),
                            "oft_root": str(args.oft_root.resolve()),
                            "bddl_file": str(task.bddl_file),
                            "data_contract": {
                                "online_policy_inputs": ["rgb", "wrist_rgb", "language", "proprio", "executed_action_history"],
                                "optional_recovery_inputs": ["depth", "wrist_depth", "camera_calibration"],
                                "training_only": ["instance_segmentation"],
                                "offline_privileged_only": ["mujoco_state", "ctrl", "mocap", "contacts", "entities", "goal_predicates"],
                            },
                        }
                        (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                        with (episode_dir / "steps.jsonl").open("w") as out:
                            for record in records:
                                out.write(json.dumps(record) + "\n")
                        if success:
                            with (episode_dir / "transitions.jsonl").open("w") as out:
                                for record in records:
                                    out.write(json.dumps(record) + "\n")
                        else:
                            # Retain only metadata for audit; failed nominal steps must not
                            # become false NORMAL supervision.
                            (episode_dir / "REJECTED_NOT_NORMAL.txt").write_text("OFT rollout did not solve task; excluded from NORMAL labels.\n")
                            (episode_dir / "transitions.jsonl").unlink(missing_ok=True)
                        if success:
                            (episode_dir / "REJECTED_NOT_NORMAL.txt").unlink(missing_ok=True)
                        # Human audit requires a replay for both successful and failed
                        # natural rollouts.  Only the JSONL promotion above is conditional.
                        imageio.mimsave(episode_dir / "rollout.mp4", video, fps=30)
                        imageio.mimsave(episode_dir / "wrist-rollout.mp4", wrist_video, fps=30)
                        print(json.dumps(metadata), flush=True)
            finally:
                env.close()
    finally:
        pass


if __name__ == "__main__":
    main()
