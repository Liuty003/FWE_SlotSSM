#!/usr/bin/env python3
"""Run paired nominal/V4.2 rollouts with a frozen OpenVLA-OFT policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", default="best.safetensors")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--residual-scale-override", type=float)
    parser.add_argument("--execution-horizon", type=int, default=4,
                        help="V4.3 recovery actions executed before re-observation.")
    parser.add_argument("--local-translation-limit", type=float, default=0.5,
                        help="V4.4.2 maximum L2 norm of one recovery translation command.")
    parser.add_argument("--local-rotation-limit", type=float, default=0.5,
                        help="V4.4.2 maximum L2 norm of one recovery rotation command.")
    parser.add_argument("--recovery-horizon", type=int, default=52,
                        help="V4.3 latched recovery duration in simulator steps.")
    parser.add_argument("--minimum-recovery-steps", type=int, default=40,
                        help="After this duration V4.3 searches for handoff one simulator step at a time.")
    parser.add_argument("--release-threshold", type=float,
                        help="Gate probability below which the adaptive latch hands back to OFT.")
    parser.add_argument("--completion-threshold", type=float, default=0.8,
                        help="Completion probability required for learned recovery handoff.")
    parser.add_argument("--cooldown-steps", type=int, default=16)
    parser.add_argument("--trigger-confirmation-decisions", type=int, default=1,
                        help="Require this many consecutive high gate decisions before recovery activation.")
    parser.add_argument("--release-confirmation-decisions", type=int, default=1,
                        help="Require consecutive high resume-value decisions before handoff.")
    parser.add_argument("--handoff-on-attached", action="store_true",
                        help="Hand back to OFT after learned attached+bilateral contact confirmation.")
    parser.add_argument("--attached-threshold", type=float, default=0.8)
    parser.add_argument("--attached-confirmation-decisions", type=int, default=3)
    parser.add_argument("--grasp-only-controller", action="store_true",
                        help="Use predicted target geometry for a closed-loop grasp, then permanently hand off to OFT.")
    parser.add_argument("--gripper-obstruction-threshold", type=float, default=0.004,
                        help="Minimum summed absolute finger qpos that verifies an obstructed (non-empty) close.")
    parser.add_argument("--max-grasp-attempts", type=int, default=2,
                        help="Maximum closed-loop grasp attempts before the episode continues to its time limit.")
    parser.add_argument("--grasp-approach-threshold", type=float, default=0.012,
                        help="Position tolerance for the collision-free waypoint above the bowl.")
    parser.add_argument("--grasp-descend-threshold", type=float, default=0.012,
                        help="Strict position tolerance before closing on the bowl.")
    parser.add_argument("--max-env-steps", type=int)
    parser.add_argument("--wait-steps", type=int, default=20)
    parser.add_argument("--prefix-rollout", type=Path,
                        help="Optional canonical nominal rollout to replay before enabling either paired policy.")
    parser.add_argument("--prefix-steps", type=int, default=0)
    parser.add_argument(
        "--audit-state-dir", type=Path,
        help="V4.3 handoff audit: load step-XXXX.npy states and run Frozen OFT from each.",
    )
    parser.add_argument(
        "--audit-state-steps", type=int, nargs="*",
        help="Optional subset of teacher-relative state steps to audit.",
    )
    parser.add_argument("--paired-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nominal-only", action="store_true",
                        help="Run only frozen OFT after the optional teacher prefix (handoff audit mode).")
    parser.add_argument(
        "--nominal-normalization-key",
        help=("Override the adapter's archived normalization key in --nominal-only handoff audits. "
              "This is required when an unrelated adapter is loaded only as a runner placeholder."),
    )
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-recovery-forks-dir", type=Path,
                        help="Save each V4.3 recovery decision as a teacher-compatible exact-state fork.")
    parser.add_argument("--recovery-target-role", choices=("mug", "pudding", "black_bowl"),
                        help="Privileged target role recorded only in saved recovery-teacher forks.")
    parser.add_argument("--save-recovery-forks-activation-only", action="store_true",
                        help="Save only the first exact state of each recovery activation.")
    parser.add_argument("--oft-root", type=Path, default=Path("/data/lty/code/openvla-oft-official"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata"))
    parser.add_argument("--libero-root", type=Path, default=Path("/data/lty/code/LIBERO-plus"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def recovery_execution_count(requested: int, available: int, latch_available: int) -> int:
    """Number of recovery actions safe to place in the current queue."""
    if min(requested, available, latch_available) < 0:
        raise ValueError("execution counts must be nonnegative")
    return min(requested, available, latch_available)


def bound_local_actions(actions, translation_limit: float, rotation_limit: float):
    """L2-bound local Cartesian commands without changing gripper semantics."""
    import numpy as np

    if translation_limit <= 0 or rotation_limit <= 0:
        raise ValueError("local action limits must be positive")
    bounded = np.asarray(actions, dtype=np.float32).copy()
    if bounded.ndim != 2 or bounded.shape[1] != 7:
        raise ValueError("local actions must be [H,7]")
    original = bounded.copy()
    for start, limit in ((0, translation_limit), (3, rotation_limit)):
        vectors = bounded[:, start:start + 3]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors *= np.minimum(1.0, limit / np.maximum(norms, 1e-12))
    return bounded, bool(not np.array_equal(original, bounded))


def main() -> None:
    args = arguments()
    if (not 0 <= args.gate_threshold <= 1 or not 0 <= args.attached_threshold <= 1 or
            args.wait_steps < 0 or args.gripper_obstruction_threshold < 0 or
            min(args.grasp_approach_threshold, args.grasp_descend_threshold) <= 0 or
            min(args.execution_horizon, args.recovery_horizon, args.trigger_confirmation_decisions,
                args.release_confirmation_decisions,
                args.attached_confirmation_decisions, args.max_grasp_attempts) < 1 or args.cooldown_steps < 0 or
            min(args.local_translation_limit, args.local_rotation_limit) <= 0 or
            not 0 <= args.minimum_recovery_steps <= args.recovery_horizon):
        raise ValueError("invalid gate threshold or wait steps")
    adapter_path = args.adapter_dir / args.adapter_checkpoint
    required = (args.oft_root, args.checkpoint, args.libero_root, args.adapter_dir / "config.json", adapter_path)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.prefix_steps < 0 or (args.prefix_steps and args.prefix_rollout is None):
        raise ValueError("positive --prefix-steps requires --prefix-rollout")
    if args.nominal_only and not args.paired_baseline:
        raise ValueError("--nominal-only is incompatible with --no-paired-baseline")
    if args.audit_state_dir is not None and args.prefix_rollout is not None:
        raise ValueError("--audit-state-dir cannot be combined with a prefix rollout")
    if (args.save_recovery_forks_dir is None) != (args.recovery_target_role is None):
        raise ValueError("provide both --save-recovery-forks-dir and --recovery-target-role")
    # This must happen before importing any project module that imports torch.
    # Otherwise --gpu is silently ignored because CUDA has already enumerated devices.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    sys.path[:0] = [str(args.libero_root), str(args.oft_root)]

    from recovery.data.v4_geometry import patch_median_points, robot_point_to_saved_image_xy, saved_depth_to_robot_points
    from recovery.models.v4_relational_recovery import V4ModelConfig, V4RelationalRecovery
    from recovery.models.v43_local_recovery import RecoveryLatch, RecoveryLatchConfig, V43LocalRecovery
    from recovery.models.v44_dual_ownership_recovery import V44DualOwnershipRecovery
    from recovery.models.v43_geometry_aux_recovery import V43GeometryAuxRecovery
    from recovery.models.v43_multitask_recovery import V43MultitaskRecovery
    from recovery.models.v43_auxiliary_state_recovery import V43AuxiliaryStateRecovery
    from recovery.models.v43_auxiliary_conditioned_recovery import V43AuxiliaryConditionedRecovery
    from recovery.models.v43_completion_recovery import V43CompletionRecovery
    from recovery.models.v43_detailed_phase_recovery import V43DetailedPhaseRecovery
    from recovery.models.v43_phased_recovery import V43PhasedRecovery, PHASE_NAMES
    from recovery.models.oft_l1_action_head import OFTL1RegressionActionHead
    from scripts.collect_v4_oft_rgbd_rollouts import calibration, depth_and_segmentation
    from scripts.precompute_v4_oft_relation_features import (
        dynamic_module,
        exact_oft_policy_image,
        load_torch_state,
        normalized_proprio,
    )

    # Some of the project helpers above prepend this repository to sys.path.
    # Put the official OFT checkout back ahead of it before resolving the
    # top-level `experiments` package; the repository also has an older package
    # with that name and silently importing it breaks OFT evaluation.
    sys.path[:0] = [str(args.libero_root), str(args.oft_root)]
    os.chdir(args.oft_root)

    import imageio.v2 as imageio
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv
    from experiments.robot.libero.libero_utils import get_libero_image
    from experiments.robot.libero.run_libero_eval import TASK_MAX_STEPS, process_action
    import prismatic
    import prismatic.training
    import prismatic.vla
    from robosuite.utils.control_utils import orientation_error
    import robosuite.utils.transform_utils as T

    prismatic.__path__.append(str(args.oft_root / "prismatic"))
    prismatic.training.__path__.append(str(args.oft_root / "prismatic" / "training"))
    prismatic.vla.__path__.append(str(args.oft_root / "prismatic" / "vla"))
    projectors = dynamic_module("v4_online_projectors", args.oft_root / "prismatic/models/projectors.py")
    device = torch.device("cuda")

    adapter_config = json.loads((args.adapter_dir / "config.json").read_text())
    provenance = adapter_config.get("feature_provenance", {})
    if provenance.get("decoder_action_contract") != \
            "pre-process_action planned coordinates; executed gripper inverted by (1-g)/2":
        raise ValueError("adapter was trained with an obsolete decoder action contract")
    if provenance.get("image_preprocessing") != \
            "official OFT JPEG+TF-Lanczos3+sqrt(0.9)-crop_and_resize":
        raise ValueError("adapter was not trained with exact official OFT image preprocessing")
    if provenance.get("decision_grid_contract") != \
            "nominal grid from zero; recovery grid restarts at fork_timestep":
        raise ValueError("adapter recovery samples were not aligned to the online fork decision grid")
    if provenance.get("recovery_target_contract") != \
            "successful RECOVERY rows use canonical executed chunks when explicit target is absent":
        raise ValueError("adapter may be missing the first canonical post-fork recovery target")
    action_head_path = args.checkpoint / "action_head--150000_checkpoint.pt"
    projector_path = args.checkpoint / "proprio_projector--150000_checkpoint.pt"
    if provenance.get("action_head_sha256") != sha256(action_head_path):
        raise ValueError("online OFT action head does not match adapter feature provenance")
    if provenance.get("proprio_projector_sha256") != sha256(projector_path):
        raise ValueError("online proprio projector does not match adapter feature provenance")

    oft = AutoModelForVision2Seq.from_pretrained(
        str(args.checkpoint), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device).eval()
    oft.vision_backbone.set_num_images_in_input(2)
    oft.norm_stats = json.loads((args.checkpoint / "dataset_statistics.json").read_text())
    processor = AutoProcessor.from_pretrained(str(args.checkpoint), trust_remote_code=True)
    action_head = OFTL1RegressionActionHead(4096, 4096, 7)
    action_head.load_state_dict(load_torch_state(action_head_path, torch))
    action_head = action_head.to(device=device, dtype=torch.bfloat16).eval()
    proprio_projector = projectors.ProprioProjector(4096, 8)
    proprio_projector.load_state_dict(load_torch_state(projector_path, torch))
    proprio_projector = proprio_projector.to(device=device, dtype=torch.bfloat16).eval()
    model_values = dict(adapter_config["model"])
    if args.residual_scale_override is not None:
        if args.residual_scale_override <= 0:
            raise ValueError("residual scale override must be positive")
        model_values["residual_scale"] = args.residual_scale_override
    adapter_kind = adapter_config.get("kind", "v4_relational_recovery")
    if adapter_kind == "v44_dual_ownership_recovery_v1":
        v4 = V44DualOwnershipRecovery(
            V4ModelConfig(**model_values),
            direct_head_version=adapter_config.get("direct_head_version", "pooled_chunk_mlp_v3"),
            recovery_output_horizon=adapter_config.get("recovery_output_horizon", 1),
        ).to(device).eval()
    elif adapter_kind in {"v43_relational_local_recovery", "v44_local_closed_loop_recovery_v1"}:
        v4 = V43LocalRecovery(
            V4ModelConfig(**model_values),
            direct_head_version=adapter_config.get("direct_head_version", "context_only_v1"),
            recovery_output_horizon=adapter_config.get("recovery_output_horizon"),
        ).to(device).eval()
    elif adapter_kind == "v43_geometry_aux_recovery_v1":
        v4 = V43GeometryAuxRecovery(
            V4ModelConfig(**model_values),
            direct_head_version=adapter_config.get("direct_head_version", "pooled_chunk_mlp_v3"),
        ).to(device).eval()
    elif adapter_kind == "v43_multitask_recovery_v1":
        v4 = V43MultitaskRecovery(V4ModelConfig(**model_values)).to(device).eval()
    elif adapter_kind == "v43_auxiliary_state_recovery_v1":
        v4 = V43AuxiliaryStateRecovery(V4ModelConfig(**model_values)).to(device).eval()
    elif adapter_kind == "v43_auxiliary_conditioned_recovery_v1":
        v4 = V43AuxiliaryConditionedRecovery(V4ModelConfig(**model_values)).to(device).eval()
    elif adapter_kind == "v43_completion_recovery_v1":
        v4 = V43CompletionRecovery(V4ModelConfig(**model_values)).to(device).eval()
    elif adapter_kind == "v43_detailed_phase_recovery_v1":
        v4 = V43DetailedPhaseRecovery(V4ModelConfig(**model_values)).to(device).eval()
    elif adapter_kind == "v43_phased_recovery_v1":
        v4 = V43PhasedRecovery(
            V4ModelConfig(**model_values),
            direct_head_version=adapter_config.get("direct_head_version", "context_only_v1"),
        ).to(device).eval()
    elif adapter_kind == "v4_relational_recovery":
        v4 = V4RelationalRecovery(V4ModelConfig(**model_values)).to(device).eval()
    else:
        raise ValueError(f"unsupported adapter kind: {adapter_kind}")
    v4.load_state_dict(load_file(str(adapter_path), device="cuda"), strict=True)

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = suite.get_task(args.task_id)
    states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.episode < len(states):
        raise ValueError(f"episode {args.episode} is outside [0, {len(states)})")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    native_input_only = model_values.get("representation") in {"global", "slot"}
    env_class = (
        OffScreenRenderEnv
        if args.audit_state_dir is not None or native_input_only
        else SegmentationRenderEnv
    )
    env_kwargs = {"bddl_file_name": str(bddl), "camera_heights": 256, "camera_widths": 256}
    if env_class is SegmentationRenderEnv:
        env_kwargs.update(camera_depths=True, camera_segmentations="instance")
    env = env_class(**env_kwargs)
    if args.nominal_normalization_key is not None and not args.nominal_only:
        raise ValueError("--nominal-normalization-key is permitted only with --nominal-only")
    normalization_key = (
        args.nominal_normalization_key
        if args.nominal_normalization_key is not None
        else provenance.get("action_normalization_key")
    )
    if normalization_key not in oft.norm_stats:
        raise KeyError(f"checkpoint has no normalization statistics for {normalization_key}")
    maximum_steps = args.max_env_steps or TASK_MAX_STEPS[args.task_suite]
    prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix_actions = []
    prefix_rows = []
    if args.prefix_rollout is not None:
        prefix_metadata = json.loads((args.prefix_rollout / "metadata.json").read_text())
        identity = (prefix_metadata.get("task_suite"), prefix_metadata.get("task_id"), prefix_metadata.get("episode_id"))
        if identity != (args.task_suite, args.task_id, args.episode):
            raise ValueError(f"prefix rollout identity {identity} does not match requested episode")
        prefix_rows = [json.loads(line) for line in (args.prefix_rollout / "steps.jsonl").read_text().splitlines() if line.strip()]
        if args.prefix_steps > len(prefix_rows):
            raise ValueError("prefix rollout is shorter than --prefix-steps")
        prefix_actions = [row["executed_action"] for row in prefix_rows[:args.prefix_steps]]

    def rollout(mode: str, audit_state_path: Path | None = None) -> dict:
        env.seed(0)
        env.reset()
        observation = env.set_init_state(states[args.episode])
        camera_calibration = calibration(env, "agentview", 256, 256, np)
        camera_to_robot = np.asarray(camera_calibration["camera_to_robot"])
        world_to_robot = np.asarray(camera_calibration["world_to_robot"])
        if audit_state_path is not None:
            observation = env.regenerate_obs_from_state(np.load(audit_state_path))
            controller = env.env.robots[0].controller
            controller.update(force=True)
            controller.reset_goal()
            env.env.timestep = 0
            env.env.done = False
        else:
            for _ in range(args.wait_steps):
                observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            for action in prefix_actions:
                observation, _, done, _ = env.step(action)
                if done:
                    raise RuntimeError("task terminated while replaying the canonical prefix")
        queue, events, frames = deque(), [], []
        attached_confirmation = 0
        grasp_stage, grasp_stage_steps, grasp_stable = "approach_bowl", 0, 0
        grasp_attempts = 0
        grasp_lift_target_z, grasp_handoff_complete = None, False
        latch = RecoveryLatch(RecoveryLatchConfig(
            trigger_threshold=args.gate_threshold,
            recovery_horizon_steps=(maximum_steps if adapter_kind in {"v43_completion_recovery_v1", "v43_detailed_phase_recovery_v1"}
                                    else args.recovery_horizon),
            minimum_recovery_steps=args.minimum_recovery_steps,
            # The grasp controller has an explicit, proprioceptively verified
            # handoff boundary.  Its ownership must not change when the trigger
            # threshold is calibrated (notably for forced-activation audits).
            release_threshold=(1.0 if args.grasp_only_controller
                               else args.completion_threshold if adapter_kind in {
                                   "v43_completion_recovery_v1", "v44_dual_ownership_recovery_v1"}
                               else args.gate_threshold if args.release_threshold is None else args.release_threshold),
            cooldown_steps=args.cooldown_steps,
            trigger_confirmation_decisions=args.trigger_confirmation_decisions,
            release_confirmation_decisions=args.release_confirmation_decisions,
        ))
        env_steps = 0
        while env_steps < maximum_steps:
            if not queue:
                main_image = get_libero_image(observation)
                wrist_image = observation["robot0_eye_in_hand_image"][::-1, ::-1]
                inputs = processor(prompt, exact_oft_policy_image(main_image, Image, tf)).to(device, dtype=torch.bfloat16)
                wrist = processor(prompt, exact_oft_policy_image(wrist_image, Image, tf)).to(device, dtype=torch.bfloat16)
                inputs["pixel_values"] = torch.cat((inputs["pixel_values"], wrist["pixel_values"]), dim=1)
                proprio = normalized_proprio({
                    "robot0_eef_pos": observation["robot0_eef_pos"],
                    "robot0_eef_quat": observation["robot0_eef_quat"],
                    "robot0_gripper_qpos": observation["robot0_gripper_qpos"],
                }, oft.norm_stats[normalization_key]["proprio"], np)
                with torch.inference_mode():
                    _, nominal_hidden = oft.predict_action(
                        **inputs, unnorm_key=normalization_key, do_sample=False, proprio=proprio,
                        proprio_projector=proprio_projector, action_head=action_head, use_film=False,
                    )
                    nominal_normalized = action_head.predict_action(nominal_hidden.to(torch.bfloat16)).float().cpu().numpy()
                    if mode == "nominal":
                        probability = 0.0
                        trigger = recovering = new_activation = False
                        normalized_actions = nominal_normalized
                        execute_count = len(normalized_actions[0])
                        nominal_bypass_exact = True
                    else:
                        visual = oft.vision_backbone(inputs["pixel_values"])[..., :256, :].float()
                        if native_input_only:
                            points, valid, image_xy = patch_median_points(
                                np.zeros((256, 256, 3), dtype=np.float32),
                                np.ones((256, 256), dtype=bool),
                            )
                            eef_robot = np.zeros(3, dtype=np.float32)
                            gripper_xy = np.zeros(2, dtype=np.float32)
                        else:
                            depth, _ = depth_and_segmentation(observation, "agentview", np)
                            dense_points, dense_valid = saved_depth_to_robot_points(
                                depth, camera_to_robot, camera_calibration["near"], camera_calibration["far"],
                            )
                            points, valid, image_xy = patch_median_points(dense_points, dense_valid)
                            eef_world = np.r_[observation["robot0_eef_pos"], 1.0]
                            eef_robot = (world_to_robot @ eef_world)[:3].astype(np.float32)
                            gripper_xy, _ = robot_point_to_saved_image_xy(eef_robot, camera_to_robot, 256, 256)
                        output = v4(
                            visual_tokens=visual, action_hidden=nominal_hidden.float(),
                            token_points=torch.as_tensor(points[None], device=device),
                            token_valid=torch.as_tensor(valid[None], device=device),
                            token_image_xy=torch.as_tensor(image_xy[None], device=device),
                            gripper_position=torch.as_tensor(eef_robot[None], device=device),
                            gripper_image_xy=torch.as_tensor(gripper_xy[None], device=device),
                        )
                        probability = float(output["gate_probability"][0])
                    if mode != "nominal" and adapter_kind in {
                            "v43_relational_local_recovery", "v44_local_closed_loop_recovery_v1",
                            "v44_dual_ownership_recovery_v1",
                            "v43_phased_recovery_v1",
                            "v43_geometry_aux_recovery_v1", "v43_multitask_recovery_v1",
                            "v43_auxiliary_state_recovery_v1", "v43_auxiliary_conditioned_recovery_v1",
                            "v43_completion_recovery_v1", "v43_detailed_phase_recovery_v1"}:
                        activations_before = latch.activation_count
                        completion_probability = (
                            float(output["resume_probability"][0])
                            if adapter_kind == "v44_dual_ownership_recovery_v1"
                            else float(output["completion_probability"][0])
                            if adapter_kind == "v43_completion_recovery_v1"
                            else 0.0 if adapter_kind == "v43_detailed_phase_recovery_v1" else None
                        )
                        recovering = (mode == "v4" and not grasp_handoff_complete and latch.select(
                            probability, completion_probability=completion_probability))
                        if args.handoff_on_attached and recovering and "state_logits" in output:
                            state_probability = output["state_logits"][0].sigmoid()
                            attached_ready = (float(state_probability[0]) >= args.attached_threshold and
                                              float(state_probability[2]) >= args.attached_threshold)
                            attached_confirmation = attached_confirmation + 1 if attached_ready else 0
                            if attached_confirmation >= args.attached_confirmation_decisions:
                                latch.force_handoff(); recovering = trigger = False
                                attached_confirmation = 0
                        elif not recovering:
                            attached_confirmation = 0
                        trigger = recovering
                        new_activation = latch.activation_count > activations_before
                        normalized_actions = (output["recovery_actions"].float().cpu().numpy()
                                              if recovering else nominal_normalized)
                        execute_count = (recovery_execution_count(
                            args.execution_horizon, len(normalized_actions[0]),
                            latch.executable_steps(args.execution_horizon),
                        ) if recovering else len(normalized_actions[0]))
                        nominal_bypass_exact = (not recovering and
                                                np.array_equal(normalized_actions, nominal_normalized))
                    elif mode != "nominal":
                        trigger = mode == "v4" and probability >= args.gate_threshold
                        new_activation = trigger
                        recovering = trigger
                        routed_hidden = output["recovery_action_hidden"] if trigger else nominal_hidden.float()
                        normalized_actions = action_head.predict_action(routed_hidden.to(torch.bfloat16)).float().cpu().numpy()
                        execute_count = len(normalized_actions[0])
                        nominal_bypass_exact = not trigger and torch.equal(routed_hidden, nominal_hidden.float())
                    nominal_raw_actions = oft._unnormalize_actions(nominal_normalized, normalization_key)[0]
                    raw_actions = oft._unnormalize_actions(normalized_actions, normalization_key)[0]
                    local_action_was_clipped = False
                    if adapter_kind in {
                            "v44_local_closed_loop_recovery_v1", "v44_dual_ownership_recovery_v1"} and recovering:
                        raw_actions, local_action_was_clipped = bound_local_actions(
                            raw_actions, args.local_translation_limit, args.local_rotation_limit,
                        )
                    grasp_action_stage = grasp_stage
                    if args.grasp_only_controller and recovering:
                        predicted_delta = output["predicted_target_delta"][0].float().cpu().numpy()
                        desired_rotation = T.quat2mat(np.asarray(
                            [0.7680383, -0.6363008, 0.0477351, -0.0544048], np.float32))
                        rotation_error = orientation_error(
                            desired_rotation, T.quat2mat(np.asarray(observation["robot0_eef_quat"], np.float32)))
                        gripper = 1.0
                        if grasp_stage == "approach_bowl":
                            position_error = predicted_delta + np.asarray([-0.040, -0.012, 0.150], np.float32)
                            settled = (np.linalg.norm(position_error) < args.grasp_approach_threshold and
                                       np.linalg.norm(rotation_error) < 0.06)
                            grasp_stable = grasp_stable + 1 if settled else 0
                            if grasp_stable >= 2: grasp_stage, grasp_stable = "descend_bowl", 0
                        elif grasp_stage == "descend_bowl":
                            position_error = predicted_delta + np.asarray([-0.040, -0.012, 0.040], np.float32)
                            settled = (np.linalg.norm(position_error) < args.grasp_descend_threshold and
                                       np.linalg.norm(rotation_error) < 0.06)
                            grasp_stable = grasp_stable + 1 if settled else 0
                            if grasp_stable >= 2: grasp_stage, grasp_stage_steps, grasp_stable = "close_bowl", 0, 0
                        elif grasp_stage == "close_bowl":
                            position_error = np.zeros(3, np.float32); gripper = 0.0
                            if grasp_stage_steps >= 16:
                                aperture = float(np.abs(observation["robot0_gripper_qpos"]).sum())
                                grasp_attempts += 1
                                if aperture >= args.gripper_obstruction_threshold:
                                    grasp_stage = "lift_bowl"
                                    grasp_lift_target_z = float(observation["robot0_eef_pos"][2]) + 0.14
                                    position_error = np.asarray([0, 0, 0.14], np.float32)
                                    gripper = 0.0
                                else:
                                    grasp_stage = "retry_lift"
                                    grasp_lift_target_z = float(observation["robot0_eef_pos"][2]) + 0.10
                                    position_error = np.asarray([0, 0, 0.10], np.float32)
                                    gripper = 1.0
                        elif grasp_stage == "retry_lift":
                            position_error = np.asarray([0, 0, grasp_lift_target_z - observation["robot0_eef_pos"][2]], np.float32)
                            gripper = 1.0
                            if abs(float(position_error[2])) < 0.012 and grasp_attempts < args.max_grasp_attempts:
                                grasp_stage, grasp_stage_steps, grasp_stable = "approach_bowl", 0, 0
                        else:
                            position_error = np.asarray([0, 0, grasp_lift_target_z - observation["robot0_eef_pos"][2]], np.float32)
                            gripper = 0.0
                            lifted = abs(float(position_error[2])) < 0.018
                            aperture = float(np.abs(observation["robot0_gripper_qpos"]).sum())
                            grasp_verified = aperture >= args.gripper_obstruction_threshold
                            grasp_stable = grasp_stable + 1 if lifted and grasp_verified else 0
                            if grasp_stable >= args.attached_confirmation_decisions:
                                latch.force_handoff(); recovering = trigger = False
                                grasp_handoff_complete = True
                                raw_actions = nominal_raw_actions
                                execute_count = len(nominal_raw_actions)
                        if not grasp_handoff_complete:
                            # Match the successful fast teacher's clipped
                            # proportional controller.  The previous tanh profile
                            # spent too much of the fixed 520-step budget slowing
                            # down before every waypoint.
                            translation = np.clip(5.0 * position_error, -0.5, 0.5)
                            rotation = np.clip(rotation_error * 2.5, -0.5, 0.5)
                            planned = np.r_[translation, rotation, gripper].astype(np.float32)
                            raw_actions = np.repeat(planned[None], 8, axis=0)
                            # A decision that changes stage still executes the old
                            # stage's action.  Do not count that action toward the
                            # newly entered stage (especially the close dwell).
                            if grasp_stage == grasp_action_stage:
                                grasp_stage_steps += execute_count
                queue.extend(raw_actions[:execute_count])
                nominal_first_executed = process_action(
                    np.asarray(nominal_raw_actions[0], dtype=np.float32), "openvla",
                )
                routed_first_executed = process_action(
                    np.asarray(raw_actions[0], dtype=np.float32), "openvla",
                )
                event = {
                    "decision": len(events), "env_step": env_steps, "gate_probability": probability,
                    "triggered": trigger, "counterfactual_trigger": probability >= args.gate_threshold,
                    "new_recovery_activation": new_activation,
                    "latched_recovery_steps_remaining": latch.recovery_steps_remaining,
                    "nominal_bypass_exact": nominal_bypass_exact,
                    "nominal_first_executed_action": nominal_first_executed.tolist(),
                    "routed_first_executed_action": routed_first_executed.tolist(),
                    "routed_vs_nominal_first_action_l1": float(np.abs(
                        routed_first_executed - nominal_first_executed,
                    ).mean()),
                    "local_action_was_clipped": local_action_was_clipped,
                }
                if mode != "nominal" and adapter_kind == "v44_dual_ownership_recovery_v1":
                    event["resume_probability"] = float(output["resume_probability"][0])
                if args.grasp_only_controller:
                    event["grasp_controller_stage"] = grasp_stage
                    event["grasp_controller_action_stage"] = grasp_action_stage
                    event["grasp_handoff_complete"] = grasp_handoff_complete
                    event["grasp_attempts"] = grasp_attempts
                    event["gripper_aperture"] = float(np.abs(observation["robot0_gripper_qpos"]).sum())
                if mode != "nominal" and adapter_kind == "v43_phased_recovery_v1":
                    predicted_phase = int(output["predicted_phase"][0])
                    event["predicted_recovery_phase"] = PHASE_NAMES[predicted_phase]
                    event["phase_probability"] = output["phase_probability"][0].float().cpu().tolist()
                if mode != "nominal" and adapter_kind in {
                        "v43_multitask_recovery_v1", "v43_auxiliary_state_recovery_v1",
                        "v43_auxiliary_conditioned_recovery_v1", "v43_completion_recovery_v1",
                        "v43_detailed_phase_recovery_v1"}:
                    phase_names = adapter_config["phase_names"]
                    predicted_phase = int(output["phase_probability"][0].argmax())
                    event["predicted_recovery_phase"] = phase_names[predicted_phase]
                    event["phase_probability"] = output["phase_probability"][0].float().cpu().tolist()
                    event["predicted_attached_contact_bilateral"] = output["state_logits"][0].sigmoid().float().cpu().tolist()
                    event["predicted_target_delta_world"] = output["predicted_target_delta"][0].float().cpu().tolist()
                    if adapter_kind == "v43_completion_recovery_v1":
                        event["completion_probability"] = float(output["completion_probability"][0])
                canonical_index = args.prefix_steps + env_steps
                if canonical_index < len(prefix_rows):
                    teacher = np.asarray(prefix_rows[canonical_index]["executed_action"], dtype=np.float32)
                    event["canonical_teacher_timestep"] = canonical_index
                    event["canonical_teacher_first_executed_action"] = teacher.tolist()
                    event["routed_vs_canonical_teacher_first_action_l1"] = float(np.abs(
                        routed_first_executed - teacher,
                    ).mean())
                    event["nominal_vs_canonical_teacher_first_action_l1"] = float(np.abs(
                        nominal_first_executed - teacher,
                    ).mean())
                events.append(event)
                if (args.save_recovery_forks_dir is not None and mode == "v4" and recovering and
                        (not args.save_recovery_forks_activation_only or new_activation)):
                    fork_dir = (args.save_recovery_forks_dir /
                                f"activation-{latch.activation_count:02d}-step-{env_steps:04d}")
                    fork_dir.mkdir(parents=True, exist_ok=True)
                    np.save(fork_dir / "mujoco_state.npy", env.sim.get_state().flatten())
                    (fork_dir / "manifest.json").write_text(json.dumps({
                            "kind": ("black_bowl_bottom_drawer_exact_failure_fork_v1"
                                     if args.recovery_target_role == "black_bowl"
                                     else "mug_pudding_exact_failure_fork_v1"),
                            "teacher_only": True,
                            "task_suite": args.task_suite,
                            "task_id": args.task_id,
                            "episode_id": args.episode,
                            "language": task.language,
                            "failure_timestep": env_steps,
                            "failure_type": "learned_recovery_rollout_state",
                            "target_role": args.recovery_target_role,
                            "failure_evidence": {
                                "gate_probability": probability,
                                "activation": latch.activation_count,
                                "recovery_steps_executed": latch.executed_recovery_steps,
                            },
                            "source_snapshot_timestep": env_steps,
                            "source_snapshot": "online_v43_rollout",
                            "source_trajectory": str(args.output_dir.resolve()),
                            "actions_replayed_after_snapshot": 0,
                            "state_file": "mujoco_state.npy",
                            "observation_timing": "pre_action_at_recovery_decision",
                            "online_inputs_permitted": [
                                "main_rgb", "wrist_rgb", "language", "proprio"
                            ],
                            "teacher_only_fields": ["mujoco_state", "failure_evidence"],
                    }, indent=2) + "\n")
            if args.save_video:
                frames.append(get_libero_image(observation))
            action = process_action(np.asarray(queue.popleft(), dtype=np.float32), "openvla")
            observation, _, done, _ = env.step(action.tolist())
            if adapter_kind in {"v43_relational_local_recovery", "v44_local_closed_loop_recovery_v1",
                                "v44_dual_ownership_recovery_v1",
                                "v43_phased_recovery_v1",
                                "v43_geometry_aux_recovery_v1", "v43_multitask_recovery_v1",
                                "v43_auxiliary_state_recovery_v1",
                                "v43_auxiliary_conditioned_recovery_v1",
                                "v43_completion_recovery_v1",
                                "v43_detailed_phase_recovery_v1"} and mode == "v4":
                latch.advance(1)
            env_steps += 1
            if done or env.check_success():
                break
        result = {
            "mode": mode, "success": bool(env.check_success()), "env_steps": env_steps,
            "policy_decisions": len(events), "triggered_decisions": sum(event["triggered"] for event in events),
            "counterfactual_triggered_decisions": sum(event["counterfactual_trigger"] for event in events),
            "recovery_activations": latch.activation_count,
            "adaptive_handoffs": latch.handoff_count,
            "events": events,
        }
        if audit_state_path is not None:
            result["audit_state"] = str(audit_state_path.resolve())
            result["teacher_relative_timestep"] = int(audit_state_path.stem.split("-")[-1])
        if args.save_video and frames:
            suffix = "" if audit_state_path is None else f"-{audit_state_path.stem}"
            imageio.mimsave(args.output_dir / f"{mode}{suffix}.mp4", frames, fps=30)
        return result

    try:
        results = []
        if args.audit_state_dir is not None:
            state_paths = sorted(args.audit_state_dir.glob("step-*.npy"))
            if args.audit_state_steps is not None:
                selected_steps = set(args.audit_state_steps)
                state_paths = [path for path in state_paths
                               if int(path.stem.split("-")[-1]) in selected_steps]
            if not state_paths:
                raise FileNotFoundError(f"no selected audit states in {args.audit_state_dir}")
            for state_path in state_paths:
                if args.paired_baseline:
                    results.append(rollout("nominal", state_path))
                if not args.nominal_only:
                    results.append(rollout("v4", state_path))
        else:
            if args.paired_baseline:
                results.append(rollout("nominal"))
            if not args.nominal_only:
                results.append(rollout("v4"))
        report = {
            "kind": "v4_relational_paired_online_rollout", "task_suite": args.task_suite,
            "task_id": args.task_id, "episode": args.episode, "language": task.language,
            "adapter": str(adapter_path), "checkpoint": str(args.checkpoint),
            "nominal_normalization_key": normalization_key,
            "gate_threshold": args.gate_threshold, "wait_steps": args.wait_steps,
            "adapter_kind": adapter_kind, "residual_scale": v4.config.residual_scale,
            "native_input_only": native_input_only,
            "execution_horizon": args.execution_horizon, "recovery_horizon": args.recovery_horizon,
            "local_translation_limit": args.local_translation_limit,
            "local_rotation_limit": args.local_rotation_limit,
            "minimum_recovery_steps": args.minimum_recovery_steps,
            "release_threshold": (
                args.completion_threshold if adapter_kind in {
                    "v43_completion_recovery_v1", "v44_dual_ownership_recovery_v1"}
                else args.gate_threshold if args.release_threshold is None else args.release_threshold
            ),
            "cooldown_steps": args.cooldown_steps,
            "trigger_confirmation_decisions": args.trigger_confirmation_decisions,
            "release_confirmation_decisions": args.release_confirmation_decisions,
            "max_env_steps": maximum_steps, "paired_same_seed_and_init_state": args.paired_baseline,
            "nominal_only": args.nominal_only,
            "prefix_rollout": None if args.prefix_rollout is None else str(args.prefix_rollout),
            "prefix_steps": len(prefix_actions),
            "audit_state_dir": None if args.audit_state_dir is None else str(args.audit_state_dir),
            "audit_state_steps": args.audit_state_steps,
            "results": results,
        }
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        env.close()


if __name__ == "__main__":
    main()
