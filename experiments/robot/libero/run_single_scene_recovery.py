#!/usr/bin/env python
"""Overfit one-scene headless recovery demo with a forced reversible mistake."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_ROOT = REPO_ROOT / "experiments" / "robot" / "libero"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LIBERO_ROOT))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_ROOT / "baseline_libero_config"))

from run_openvla_baseline import (
    SUITE_MAX_STEPS, extract_privileged_object_state, extract_robot_state,
    jsonable, load_openvla, predict_action, write_json,
)
from recovery.teachers.sim_contact import extract_target_gripper_contact
from recovery.data.candidate_schema import (
    CANDIDATE_SCHEMA_VERSION,
    RecoveryCandidateGroup,
)
from recovery.models.candidate_effect_scorer import generate_recovery_candidates
from recovery.models.candidate_effect_scorer import RecoveryCandidateEffectScorer
from recovery.data.schema import RecoveryTransition, SCHEMA_VERSION
from recovery.data.tensor_dataset import (
    build_recovery_model_inputs,
    tensorize_transition,
)
from recovery.detection.failure_trigger import visual_lift_drop_evidence


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=9)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument(
        "--dataset-episode-id",
        type=int,
        help="Unique dataset identity; defaults to the LIBERO init-state episode ID.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wait-steps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--mistake-start", type=int, default=25)
    parser.add_argument("--mistake-steps", type=int, default=8)
    parser.add_argument(
        "--mistake-trigger", choices=("fixed-step", "visual-lift"),
        default="fixed-step",
    )
    parser.add_argument("--injection-lift-threshold-px", type=float, default=15.0)
    parser.add_argument("--mistake-action", default="-0.8,0.8,0.4,0,0,0,-1")
    parser.add_argument("--mistake-label", default="FORCED ERROR")
    parser.add_argument(
        "--disable-failure-injection",
        action="store_true",
        help="Collect a nominal negative episode without injecting a mistake.",
    )
    parser.add_argument(
        "--failure-detector",
        choices=("state-height", "segmentation-motion", "stage-b-tracked"),
        default="state-height",
    )
    parser.add_argument("--failure-lift-threshold-m", type=float, default=0.03)
    parser.add_argument("--visual-lift-threshold-px", type=float, default=30.0)
    parser.add_argument("--visual-drop-return-px", type=float, default=16.0)
    parser.add_argument(
        "--stage-b-config",
        type=Path,
        default=Path("experiments/robot/libero/stage_b_selector_recovery_overfit.json"),
    )
    parser.add_argument(
        "--stage-b-language-features",
        type=Path,
        default=Path(
            "/data2/lty/naive-e-slotssm-runs/stage-b-spatial-counterfactual-patches/"
            "siglip_language_features.safetensors"
        ),
    )
    parser.add_argument("--oracle-regrasp", action="store_true")
    parser.add_argument("--learned-recovery-checkpoint", type=Path)
    parser.add_argument("--learned-phase-checkpoint", type=Path)
    parser.add_argument("--visual-regrasp-checkpoint", type=Path)
    parser.add_argument("--visual-regrasp-confirm-steps", type=int, default=2)
    parser.add_argument("--visual-regrasp-settle-height-m", type=float, default=0.06)
    parser.add_argument("--visual-3d-audit", type=Path)
    parser.add_argument("--residual-clip", type=float, default=1.0)
    parser.add_argument("--regrasp-gain", type=float, default=8.0)
    parser.add_argument("--regrasp-close-steps", type=int, default=8)
    parser.add_argument("--regrasp-lift-threshold-m", type=float, default=0.06)
    parser.add_argument("--recovery-highlight-steps", type=int, default=35)
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device-map", default="balanced_low_0")
    parser.add_argument("--gpu-memory-gib", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path)
    parser.add_argument(
        "--expert-dataset-dir", type=Path,
        help="Optionally save per-step RGB/depth/mask and oracle recovery actions.",
    )
    parser.add_argument(
        "--candidate-dataset-dir",
        type=Path,
        help="Save matched short-horizon branches from recovery states.",
    )
    parser.add_argument("--candidate-collect-every", type=int, default=12)
    parser.add_argument("--candidate-max-forks", type=int, default=4)
    parser.add_argument("--candidate-branch-steps", type=int, default=4)
    parser.add_argument(
        "--candidate-split",
        choices=("train", "validation", "development_test"),
        default="train",
    )
    parser.add_argument("--candidate-translation-delta", type=float, default=0.20)
    parser.add_argument("--candidate-rotation-delta", type=float, default=0.15)
    parser.add_argument(
        "--candidate-scorer-checkpoint",
        type=Path,
        help="Score candidates in shadow mode without changing executed actions.",
    )
    return parser.parse_args()


def annotate(image, phase, step, success=False, shadow_decision=None):
    colors = {
        "NOMINAL": (35, 130, 70), "FORCED ERROR": (190, 45, 45),
        "MISSED GRASP": (205, 35, 35), "FAILURE DETECTED": (235, 95, 20),
        "LOST GRASP": (205, 35, 35),
        "RE-GRASP": (225, 140, 20),
        "RETURN TO BOWL": (225, 140, 20), "DESCEND": (225, 140, 20),
        "CLOSE GRIPPER": (225, 140, 20), "VERIFY LIFT": (225, 140, 20),
        "VISUAL SETTLE": (225, 140, 20),
        "RECOVERY": (225, 140, 20), "RESUMED TASK": (45, 90, 190),
        "SUCCESS": (25, 160, 65),
    }
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 28), fill=(0, 0, 0))
    label = "SUCCESS" if success else phase
    draw.text((8, 7), f"{label} | step {step}", fill=colors[label])
    if phase in ("FORCED ERROR", "MISSED GRASP", "LOST GRASP"):
        message = (
            "gripper opens; bowl is dropped" if phase == "LOST GRASP" else
            "gripper closes beside bowl" if phase == "MISSED GRASP" else
            "arm deliberately moved away"
        )
        draw.text((145, 7), message, fill=(255, 220, 220))
    elif phase in (
        "RECOVERY", "RE-GRASP", "RETURN TO BOWL", "DESCEND",
        "CLOSE GRIPPER", "VERIFY LIFT",
    ):
        draw.text((145, 7), "replanning from failed grasp", fill=(255, 235, 190))
    elif phase == "FAILURE DETECTED":
        draw.text((145, 7), "target fell after being lifted", fill=(255, 230, 190))
    if shadow_decision is not None:
        draw.rectangle((0, 28, canvas.width, 48), fill=(0, 0, 0))
        name = shadow_decision["selected_name"]
        value = shadow_decision["selected_value"]
        draw.text(
            (8, 32),
            f"SHADOW only: {name}  value={value:+.3f}",
            fill=(120, 210, 255),
        )
    return np.asarray(canvas)


def segmentation_key(obs):
    keys = [
        key for key in obs
        if key.startswith("agentview_segmentation") and "eye_in_hand" not in key
    ]
    if len(keys) != 1:
        raise RuntimeError(f"expected one agentview instance segmentation, got {keys}")
    return keys[0]


def mask_observation(obs, instance_id):
    """Return target mask measurements in the RGB image coordinate convention."""
    segmentation = np.asarray(obs[segmentation_key(obs)]).squeeze()[::-1, ::-1]
    mask = segmentation == instance_id
    ys, xs = np.where(mask)
    if mask.shape[0] % 64 or mask.shape[1] % 64:
        raise RuntimeError(f"cannot downsample segmentation mask {mask.shape} to 64x64")
    mask_64 = mask.reshape(
        64, mask.shape[0] // 64, 64, mask.shape[1] // 64
    ).any(axis=(1, 3)).astype(np.uint8)
    if not len(xs):
        return {
            "visible_pixels": 0,
            "centroid_xy": None,
            "_mask_64": mask_64,
        }
    return {
        "visible_pixels": int(len(xs)),
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
        "_mask_64": mask_64,
    }


class StageBMaskTracker:
    """Initial relation grounding followed by proposal-level temporal identity."""

    def __init__(self, model, processor, config, language_features, task_id, device):
        from scripts.stage_b_spatial_selector import (
            RELATION_ORDER, StageBSpatialSelector,
        )

        self.model = model
        self.processor = processor
        self.device = torch.device(device)
        self.selector = StageBSpatialSelector.from_config(config, self.device)
        language = load_file(language_features)
        index = language["task_ids"].tolist().index(task_id)
        self.instruction = language["instruction_features"][index:index + 1].to(self.device)
        self.relation = language["relation_features"][index:index + 1].to(self.device)
        self.references = language["reference_features"][index:index + 1].to(self.device)
        self.relation_id = torch.tensor(
            [RELATION_ORDER.index("on")], dtype=torch.long, device=self.device
        )
        self.presence = torch.tensor(
            [[True, True, True, True, False, True]], device=self.device
        )
        self.previous_mask = None
        self.previous_centroid = None

    @staticmethod
    def centroid(mask):
        points = mask.bool().nonzero().float()
        if not len(points):
            return None
        return torch.stack((points[:, 1].mean(), points[:, 0].mean()))

    @staticmethod
    def iou(first, second):
        intersection = (first.bool() & second.bool()).sum().float()
        union = (first.bool() | second.bool()).sum().clamp_min(1)
        return float(intersection / union)

    @torch.inference_mode()
    def __call__(self, image):
        pixel_values = self.processor.image_processor.apply_transform(
            Image.fromarray(image)
        )[None].to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = self.model.vision_backbone(pixel_values)
            patches = encoded.reshape(1, -1, 2176)
            output = self.selector(
                patches, self.relation_id, self.presence, self.instruction,
                self.relation, self.references,
            )
        candidates = output["candidate_masks"][0].cpu()
        if self.previous_mask is None:
            index = int(output["selected_indices"][0])
            mode = "initial_grounding"
        else:
            overlaps = [self.iou(candidate, self.previous_mask) for candidate in candidates]
            if max(overlaps) > 0:
                index, mode = int(np.argmax(overlaps)), "mask_iou"
            elif self.previous_centroid is None:
                # The tracked proposal disappeared in the previous frame.  A
                # centroid distance is undefined, so re-run relation grounding
                # instead of crashing or inventing a location.
                index = int(output["selected_indices"][0])
                mode = "relation_regrounding"
            else:
                centers = [self.centroid(candidate) for candidate in candidates]
                distances = [
                    float(torch.linalg.vector_norm(center - self.previous_centroid))
                    if center is not None else float("inf")
                    for center in centers
                ]
                index, mode = int(np.argmin(distances)), "centroid"
        self.previous_mask = candidates[index]
        self.previous_centroid = self.centroid(self.previous_mask)
        center = self.previous_centroid
        return {
            "visible_pixels": int(self.previous_mask.sum()),
            "centroid_xy": (
                [float(center[0] * 4), float(center[1] * 4)]
                if center is not None else None
            ),
            "candidate_index": index,
            "tracking_mode": mode,
            "selection_margin": float(output["selection_margin"][0]),
            "_mask_64": self.previous_mask.numpy().astype(np.uint8),
        }


class VisualDepthTargetEstimator:
    def __init__(self, sim, audit_path):
        from robosuite.utils.camera_utils import get_camera_transform_matrix

        audit = json.loads(Path(audit_path).read_text())
        calibration = audit["constant_bias_calibration"]
        self.bias = np.asarray(
            calibration["bias_estimate_minus_teacher_m"], np.float64
        )
        self.camera_to_world = np.linalg.inv(
            get_camera_transform_matrix(sim, "agentview", 256, 256)
        )
        extent = sim.model.stat.extent
        self.far = sim.model.vis.map.zfar * extent
        self.near = sim.model.vis.map.znear * extent

    def __call__(self, oriented_depth, mask_64):
        depth = self.near / (
            1.0 - oriented_depth * (1.0 - self.near / self.far)
        )
        mask = np.repeat(np.repeat(mask_64, 4, axis=0), 4, axis=1).astype(bool)
        ys, xs = np.where(mask)
        z = depth[ys, xs]
        camera_x = 255.0 - xs.astype(np.float64)
        camera_y = ys.astype(np.float64)
        homogeneous = np.stack(
            (camera_x * z, camera_y * z, z, np.ones_like(z)), axis=1
        )
        world = (self.camera_to_world @ homogeneous.T).T[:, :3]
        return np.median(world, axis=0) - self.bias


class LearnedRecoveryController:
    def __init__(self, checkpoint, residual_clip):
        from scripts.train_recovery_residual_smoke import (
            PHASES, RecoveryResidualMLP,
        )

        tensors = load_file(checkpoint)
        state = {
            key.removeprefix("model."): value
            for key, value in tensors.items() if key.startswith("model.")
        }
        self.model = RecoveryResidualMLP(
            state["network.0.weight"].shape[1]
        )
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.feature_mean = tensors["feature_mean"]
        self.feature_std = tensors["feature_std"]
        self.residual_mean = tensors["residual_mean"]
        self.residual_std = tensors["residual_std"]
        self.phases = PHASES
        self.residual_clip = residual_clip

    @torch.inference_mode()
    def __call__(self, target_xyz, eef_xyz, nominal, phase):
        one_hot = np.zeros(len(self.phases), np.float32)
        one_hot[self.phases.index(phase)] = 1
        features = torch.from_numpy(np.concatenate((
            np.asarray(target_xyz, np.float32) - np.asarray(eef_xyz, np.float32),
            np.asarray(nominal, np.float32),
            one_hot,
        )))[None]
        normalized = (features - self.feature_mean) / self.feature_std
        residual = (
            self.model(normalized)[0] * self.residual_std + self.residual_mean
        ).clamp(-self.residual_clip, self.residual_clip)
        action = torch.from_numpy(np.asarray(nominal, np.float32)) + residual
        action = action.clamp(-1, 1)
        action[6] = (
            -1.0 if phase in ("RETURN TO BOWL", "DESCEND") else 1.0
        )
        return action.numpy(), residual.numpy()


class LearnedPhaseController:
    INTERNAL = ("return", "descend", "close", "lift")

    def __init__(self, checkpoint):
        from scripts.train_recovery_phase_predictor import PhasePredictor

        tensors = load_file(checkpoint)
        state = {
            key.removeprefix("model."): value
            for key, value in tensors.items() if key.startswith("model.")
        }
        self.model = PhasePredictor(state["network.0.weight"].shape[1])
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.mean = tensors["feature_mean"]
        self.std = tensors["feature_std"]

    @torch.inference_mode()
    def __call__(self, target_xyz, obs, current_phase):
        current = self.INTERNAL.index(current_phase)
        previous = np.zeros(len(self.INTERNAL), np.float32)
        previous[current] = 1
        features = torch.from_numpy(np.concatenate((
            np.asarray(target_xyz, np.float32)
            - np.asarray(obs["robot0_eef_pos"], np.float32),
            np.asarray(obs["robot0_gripper_qpos"], np.float32),
            np.asarray(obs["robot0_gripper_qvel"], np.float32),
            previous,
        )))[None]
        logits = self.model((features - self.mean) / self.std)[0]
        legal = torch.full_like(logits, float("-inf"))
        legal[current:min(current + 2, len(self.INTERNAL))] = logits[
            current:min(current + 2, len(self.INTERNAL))
        ]
        selected = int(legal.argmax())
        probabilities = torch.softmax(legal, dim=0)
        return self.INTERNAL[selected], probabilities.numpy()


class VisualRegraspController:
    def __init__(self, checkpoint, confirm_steps):
        from scripts.train_visual_regrasp_verifier import (
            VisualRegraspVerifier, feature,
        )

        tensors = load_file(checkpoint)
        state = {
            key.removeprefix("model."): value
            for key, value in tensors.items() if key.startswith("model.")
        }
        self.model = VisualRegraspVerifier(state["network.0.weight"].shape[1])
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.mean = tensors["feature_mean"]
        self.std = tensors["feature_std"]
        self.threshold = float(tensors["probability_threshold"][0])
        self.feature = feature
        self.confirm_steps = confirm_steps
        self.reset()

    def reset(self):
        self.target_start = None
        self.eef_start = None
        self.elapsed = 0
        self.streak = 0

    @torch.inference_mode()
    def __call__(self, target_xyz, obs):
        if self.target_start is None:
            self.target_start = np.asarray(target_xyz, np.float32).copy()
            self.eef_start = np.asarray(obs["robot0_eef_pos"], np.float32).copy()
        values = self.feature(
            {"visual_target_xyz": target_xyz, "robot_state_before": obs},
            self.target_start, self.eef_start, self.elapsed,
        )
        inputs = torch.from_numpy(values)[None]
        probability = float(torch.sigmoid(
            self.model((inputs - self.mean) / self.std)
        )[0])
        self.elapsed += 1
        self.streak = self.streak + 1 if probability >= self.threshold else 0
        return probability, self.streak >= self.confirm_steps


class CandidateShadowScorer:
    """Read-only candidate ranking; never selects the executed action."""

    def __init__(self, checkpoint):
        state = load_file(checkpoint)
        hidden_dim = int(state["context_encoder.1.weight"].shape[0])
        context_dim = int(state["context_encoder.1.weight"].shape[1])
        self.model = RecoveryCandidateEffectScorer(
            context_dim=context_dim, hidden_dim=hidden_dim
        )
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def __call__(
        self,
        *,
        task_suite,
        task_id,
        episode_id,
        timestep,
        instruction,
        obs,
        target_mask,
        nominal,
        executed,
        previous_executed_action,
        translation_delta,
        rotation_delta,
    ):
        names, actions = generate_recovery_candidates(
            torch.from_numpy(np.asarray(nominal, np.float32)),
            torch.from_numpy(
                np.asarray(executed, np.float32)
                - np.asarray(nominal, np.float32)
            ),
            translation_delta=translation_delta,
            rotation_delta=rotation_delta,
        )
        # v1 scorer data has no enriched visual xyz yet, so shadow inference
        # deliberately matches that training distribution.
        row = RecoveryTransition(
            schema_version=SCHEMA_VERSION,
            source_format="online_candidate_shadow",
            source_path="online",
            task_suite=task_suite,
            task_id=task_id,
            episode_id=episode_id,
            timestep=timestep,
            split="development_test",
            mode="recovery",
            online_inputs={
                "rgb": None,
                "depth": None,
                "language_instruction": instruction,
                "proprio": extract_robot_state(obs),
                "previous_executed_action": (
                    None
                    if previous_executed_action is None
                    else np.asarray(
                        previous_executed_action, np.float32
                    ).tolist()
                ),
                "nominal_action": np.asarray(nominal, np.float32).tolist(),
                "target_mask": None,
                "target_mask_measurement": target_mask,
                "visual_target_xyz": None,
                "visual_target_xyz_valid": False,
                "observation_validity": {},
            },
            labels={
                "executed_action": np.asarray(executed, np.float32).tolist()
            },
        )
        row.validate()
        step_tensors = tensorize_transition(row)
        context = build_recovery_model_inputs(step_tensors)[None]
        outputs = self.model(context, torch.from_numpy(
            np.asarray(nominal, np.float32)
        )[None], actions[None])
        values = outputs["candidate_value"][0]
        selected = int(values.argmax())
        return {
            "candidate_names": names,
            "candidate_actions": actions.tolist(),
            "predicted_values": values.tolist(),
            "selected_index": selected,
            "selected_name": names[selected],
            "selected_value": float(values[selected]),
            "execution_changed": False,
        }


def collect_candidate_group(
    *,
    fork_env,
    sim_state,
    task_suite,
    task_id,
    episode_id,
    timestep,
    split,
    phase,
    nominal,
    executed,
    target_name,
    target_key,
    initial_target_z,
    attached_before,
    branch_steps,
    translation_delta,
    rotation_delta,
):
    """Roll all candidates from one identical simulator state."""

    names, actions = generate_recovery_candidates(
        torch.from_numpy(np.asarray(nominal, np.float32)),
        torch.from_numpy(
            np.asarray(executed, np.float32) - np.asarray(nominal, np.float32)
        ),
        translation_delta=translation_delta,
        rotation_delta=rotation_delta,
    )
    restored = fork_env.regenerate_obs_from_state(sim_state)
    before_eef = np.asarray(restored["robot0_eef_pos"], np.float32)
    before_target = np.asarray(restored[target_key], np.float32)
    before_distance = float(np.linalg.norm(before_eef - before_target))
    outcomes = []
    valid = []
    max_restore_error = 0.0
    for action in actions.numpy():
        try:
            branch_obs = fork_env.regenerate_obs_from_state(sim_state)
            restore_error = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(
                                branch_obs["robot0_eef_pos"], np.float32
                            )
                            - before_eef
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            np.asarray(branch_obs[target_key], np.float32)
                            - before_target
                        )
                    )
                ),
            )
            max_restore_error = max(max_restore_error, restore_error)
            if restore_error > 1e-6:
                raise RuntimeError(
                    f"candidate state restore error {restore_error:.3e}"
                )
            fork_env.env.timestep = 0
            fork_env.env.done = False
            steps_executed = 0
            for _ in range(branch_steps):
                branch_obs, _, done, _ = fork_env.step(action.tolist())
                steps_executed += 1
                if done:
                    break
            after_eef = np.asarray(branch_obs["robot0_eef_pos"], np.float32)
            after_target = np.asarray(branch_obs[target_key], np.float32)
            _, attached_after = extract_target_gripper_contact(
                fork_env.env.sim, target_name=target_name
            )
            target_lift = float(after_target[2]) - initial_target_z
            lifted_after = target_lift >= 0.03
            slipped_after = bool(attached_before and not attached_after)
            handoff_ready = bool(attached_after and lifted_after)
            failure_persistent = not handoff_ready
            distance_progress = before_distance - float(
                np.linalg.norm(after_eef - after_target)
            )
            # A task-independent recovery utility: approach progress remains
            # informative before contact, while stable lift/handoff dominates.
            teacher_value = (
                10.0 * distance_progress
                + float(attached_after)
                + float(lifted_after)
                - float(slipped_after)
                - 0.5 * float(failure_persistent)
                + float(handoff_ready)
            )
            outcomes.append(
                {
                    "eef_delta_m": (after_eef - before_eef).tolist(),
                    "target_delta_m": (after_target - before_target).tolist(),
                    "attached_after": bool(attached_after),
                    "lifted_after": bool(lifted_after),
                    "slipped_after": slipped_after,
                    "failure_persistent_after": failure_persistent,
                    "handoff_ready_after": handoff_ready,
                    "teacher_value": teacher_value,
                    "target_lift_m": target_lift,
                    "eef_target_distance_progress_m": distance_progress,
                    "branch_steps_executed": steps_executed,
                }
            )
            valid.append(True)
        except Exception as error:
            outcomes.append({"error": repr(error)})
            valid.append(False)
    group = RecoveryCandidateGroup(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        task_suite=task_suite,
        task_id=task_id,
        episode_id=episode_id,
        timestep=timestep,
        split=split,
        context_transition_path=f"transitions-v2.jsonl#{timestep}",
        nominal_action=np.asarray(nominal, np.float32).tolist(),
        candidate_names=names,
        candidate_actions=actions.numpy().tolist(),
        outcome_valid=valid,
        teacher_outcomes=outcomes,
        provenance={
            "collector": "run_single_scene_recovery.py",
            "phase": phase,
            "branch_horizon": branch_steps,
            "translation_delta": translation_delta,
            "rotation_delta": rotation_delta,
            "sim_state_restored_per_candidate": True,
            "max_restore_error_m": max_restore_error,
        },
    )
    group.validate()
    return group


def main():
    args = arguments()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv

    mistake_action = np.asarray([float(value) for value in args.mistake_action.split(",")], np.float32)
    if mistake_action.shape != (7,):
        raise ValueError("mistake-action must contain seven comma-separated values")
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    dataset_episode_id = (
        args.episode_id
        if args.dataset_episode_id is None
        else args.dataset_episode_id
    )
    if dataset_episode_id < 0:
        raise ValueError("--dataset-episode-id must be non-negative")
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_class = (
        SegmentationRenderEnv
        if (
            args.failure_detector == "segmentation-motion"
            or args.expert_dataset_dir is not None
        )
        else OffScreenRenderEnv
    )
    env = env_class(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
        camera_depths=(
            args.expert_dataset_dir is not None
            or args.learned_recovery_checkpoint is not None
        ),
    )
    env.seed(args.seed)
    env.reset()
    obs = env.set_init_state(initial_states[args.episode_id])
    fork_env = None
    if args.candidate_dataset_dir is not None:
        if args.candidate_collect_every <= 0:
            raise ValueError("--candidate-collect-every must be positive")
        if args.candidate_max_forks <= 0:
            raise ValueError("--candidate-max-forks must be positive")
        if args.candidate_branch_steps <= 0:
            raise ValueError("--candidate-branch-steps must be positive")
        fork_env = env_class(
            bddl_file_name=str(bddl),
            camera_heights=256,
            camera_widths=256,
            camera_depths=(
                args.expert_dataset_dir is not None
                or args.learned_recovery_checkpoint is not None
            ),
        )
        fork_env.seed(args.seed)
        fork_env.reset()
        fork_env.set_init_state(initial_states[args.episode_id])

    rollout = args.rollout_dir or args.output.with_suffix("")
    rollout.mkdir(parents=True, exist_ok=True)
    steps_path, metadata_path = rollout / "steps.jsonl", rollout / "metadata.json"
    steps_path.write_text("", encoding="utf-8")
    expert_path = None
    candidate_path = None
    candidates_written = 0
    if args.candidate_dataset_dir is not None:
        args.candidate_dataset_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = args.candidate_dataset_dir / "candidates.jsonl"
        candidate_path.write_text("", encoding="utf-8")
    expert_rgb_dir = expert_depth_dir = expert_mask_dir = None
    if args.expert_dataset_dir is not None:
        args.expert_dataset_dir.mkdir(parents=True, exist_ok=True)
        expert_rgb_dir = args.expert_dataset_dir / "rgb"
        expert_depth_dir = args.expert_dataset_dir / "depth"
        expert_mask_dir = args.expert_dataset_dir / "target_masks"
        for directory in (expert_rgb_dir, expert_depth_dir, expert_mask_dir):
            directory.mkdir(exist_ok=True)
        expert_path = args.expert_dataset_dir / "transitions.jsonl"
        expert_path.write_text("", encoding="utf-8")
    detector_description = (
        "Stage-B predicted bowl proposals: relation grounding on the first frame, "
        "then temporal identity tracking; target-mask centroid rise and return"
        if args.failure_detector == "stage-b-tracked"
        else
        "target-mask centroid first rose by the configured pixel threshold, then "
        "returned near its initial image row; no simulator object pose is used"
        if args.failure_detector == "segmentation-motion"
        else (
            "target was lifted, then fell below the lift threshold after forced gripper opening"
            if args.mistake_label == "LOST GRASP"
            else "gripper-close attempt followed by target lift below threshold"
        )
    )
    metadata = {
        "schema_version": 2, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "demo_kind": "single_scene_overfit_recovery", "task_suite": args.task_suite,
        "task_id": args.task_id,
        "episode_id": dataset_episode_id,
        "init_state_episode_id": args.episode_id,
        "seed": args.seed,
        "task_language": task.language, "checkpoint": args.checkpoint,
        "mistake_start": args.mistake_start, "mistake_steps": args.mistake_steps,
        "mistake_trigger": args.mistake_trigger,
        "injection_lift_threshold_px": args.injection_lift_threshold_px,
        "mistake_action": mistake_action.tolist(), "mistake_label": args.mistake_label,
        "failure_injection_enabled": not args.disable_failure_injection,
        "failure_detector_kind": args.failure_detector,
        "failure_detector": detector_description,
        "failure_lift_threshold_m": args.failure_lift_threshold_m,
        "visual_lift_threshold_px": args.visual_lift_threshold_px,
        "visual_drop_return_px": args.visual_drop_return_px,
        "stage_b_config": (
            str(args.stage_b_config) if args.failure_detector == "stage-b-tracked" else None
        ),
        "oracle_regrasp": args.oracle_regrasp,
        "learned_recovery_checkpoint": (
            str(args.learned_recovery_checkpoint)
            if args.learned_recovery_checkpoint is not None else None
        ),
        "learned_phase_checkpoint": (
            str(args.learned_phase_checkpoint)
            if args.learned_phase_checkpoint is not None else None
        ),
        "visual_regrasp_checkpoint": (
            str(args.visual_regrasp_checkpoint)
            if args.visual_regrasp_checkpoint is not None else None
        ),
        "visual_regrasp_confirm_steps": args.visual_regrasp_confirm_steps,
        "visual_regrasp_settle_height_m": args.visual_regrasp_settle_height_m,
        "expert_dataset_dir": (
            str(args.expert_dataset_dir.resolve())
            if args.expert_dataset_dir is not None else None
        ),
        "candidate_dataset_dir": (
            str(args.candidate_dataset_dir.resolve())
            if args.candidate_dataset_dir is not None else None
        ),
        "candidate_branch_steps": args.candidate_branch_steps,
        "candidate_max_forks": args.candidate_max_forks,
        "candidate_scorer_checkpoint": (
            str(args.candidate_scorer_checkpoint.resolve())
            if args.candidate_scorer_checkpoint is not None else None
        ),
        "candidate_scorer_mode": (
            "shadow_only"
            if args.candidate_scorer_checkpoint is not None else None
        ),
        "completed": False, "success": False,
    }
    write_json(metadata_path, metadata)
    model, processor, input_device = load_openvla(
        args.checkpoint, args.device_map, args.gpu_memory_gib
    )
    stage_b_tracker = (
        StageBMaskTracker(
            model, processor, args.stage_b_config, args.stage_b_language_features,
            args.task_id, input_device,
        )
        if args.failure_detector == "stage-b-tracked" else None
    )
    learned_recovery = (
        LearnedRecoveryController(
            args.learned_recovery_checkpoint, args.residual_clip
        )
        if args.learned_recovery_checkpoint is not None else None
    )
    learned_phase = (
        LearnedPhaseController(args.learned_phase_checkpoint)
        if args.learned_phase_checkpoint is not None else None
    )
    visual_regrasp = (
        VisualRegraspController(
            args.visual_regrasp_checkpoint, args.visual_regrasp_confirm_steps
        )
        if args.visual_regrasp_checkpoint is not None else None
    )
    candidate_shadow_scorer = (
        CandidateShadowScorer(args.candidate_scorer_checkpoint)
        if args.candidate_scorer_checkpoint is not None else None
    )
    if learned_recovery is not None and args.visual_3d_audit is None:
        raise ValueError("--visual-3d-audit is required for learned recovery")
    depth_target_estimator = (
        VisualDepthTargetEstimator(env.env.sim, args.visual_3d_audit)
        if learned_recovery is not None else None
    )
    noop = np.asarray([0, 0, 0, 0, 0, 0, -1], np.float32)
    for _ in range(args.wait_steps):
        obs, _, done, _ = env.step(noop.tolist())
    frames, success = [], False
    transitions_written = 0
    target_key = "akita_black_bowl_1_pos"
    initial_target_z = float(np.asarray(obs[target_key], np.float32)[2])
    target_name = "akita_black_bowl_1"
    target_instance_id = (
        env.instance_to_id[target_name]
        if isinstance(env, SegmentationRenderEnv) else None
    )
    initial_mask = (
        mask_observation(obs, target_instance_id)
        if target_instance_id is not None else None
    )
    if stage_b_tracker is not None:
        initial_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        initial_mask = stage_b_tracker(initial_image)
    if initial_mask is not None and initial_mask["centroid_xy"] is None:
        raise RuntimeError("target bowl is not visible in the initial segmentation")
    initial_target_row = (
        initial_mask["centroid_xy"][1] if initial_mask is not None else None
    )
    peak_visual_rise = 0.0
    effective_mistake_start = (
        None
        if args.disable_failure_injection
        else args.mistake_start if args.mistake_trigger == "fixed-step" else None
    )
    mistake_end = (
        effective_mistake_start + args.mistake_steps
        if effective_mistake_start is not None else None
    )
    failure_detected = False
    failure_detected_step = None
    recovery_phase = None
    recovery_phase_step = 0
    recovery_completed = False
    visual_regrasp_settling = False
    peak_target_lift = 0.0
    previous_executed_action = None
    shadow_steps = shadow_matches = shadow_matched_groups = 0
    shadow_regret_sum = 0.0
    try:
        for step in range(args.max_steps):
            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            nominal = predict_action(
                model, processor, image, task.language, args.task_suite,
                args.center_crop, input_device,
            )
            detection_event = False
            learned_residual = None
            visual_target_xyz = None
            learned_phase_probabilities = None
            visual_regrasp_probability = None
            shadow_decision = None
            candidate_group = None
            current_target_lift = (
                float(np.asarray(obs[target_key], np.float32)[2]) - initial_target_z
            )
            peak_target_lift = max(peak_target_lift, current_target_lift)
            contact_state, attached_truth = extract_target_gripper_contact(
                env.env.sim, target_name=target_name
            )
            target_mask = (
                mask_observation(obs, target_instance_id)
                if target_instance_id is not None else None
            )
            if stage_b_tracker is not None:
                target_mask = stage_b_tracker(image)
            target_mask_64 = (
                target_mask.pop("_mask_64")
                if target_mask is not None and "_mask_64" in target_mask else None
            )
            current_visual_rise = None
            if target_mask is not None and target_mask["centroid_xy"] is not None:
                current_visual_rise = initial_target_row - target_mask["centroid_xy"][1]
                peak_visual_rise = max(peak_visual_rise, current_visual_rise)
            if (
                not args.disable_failure_injection
                and
                args.mistake_trigger == "visual-lift"
                and effective_mistake_start is None
                and current_visual_rise is not None
                and current_visual_rise >= args.injection_lift_threshold_px
            ):
                effective_mistake_start = step
                mistake_end = step + args.mistake_steps
            if mistake_end is not None and step >= mistake_end and not failure_detected:
                if args.failure_detector in ("segmentation-motion", "stage-b-tracked"):
                    had_lifted, has_dropped = visual_lift_drop_evidence(
                        peak_visual_rise=peak_visual_rise,
                        current_visual_rise=current_visual_rise,
                        lift_threshold_px=args.visual_lift_threshold_px,
                        drop_return_px=args.visual_drop_return_px,
                        injection_was_visual_lift_triggered=bool(
                            args.mistake_trigger == "visual-lift"
                            and effective_mistake_start is not None
                        ),
                    )
                else:
                    had_lifted = (
                        peak_target_lift >= args.regrasp_lift_threshold_m
                        if args.mistake_label == "LOST GRASP" else True
                    )
                    has_dropped = current_target_lift < args.failure_lift_threshold_m
                if had_lifted and has_dropped:
                    failure_detected = True
                    failure_detected_step = step
                    detection_event = True
                    recovery_phase = "return"
            recovery_end = (
                failure_detected_step + args.recovery_highlight_steps
                if failure_detected_step is not None else 10**9
            )
            if (
                effective_mistake_start is not None
                and effective_mistake_start <= step < mistake_end
            ):
                phase, executed = args.mistake_label, mistake_action.copy()
            elif detection_event:
                phase, executed = "FAILURE DETECTED", nominal.copy()
            elif args.oracle_regrasp and failure_detected and not recovery_completed:
                bowl = np.asarray(obs[target_key], np.float32)
                eef = np.asarray(obs["robot0_eef_pos"], np.float32)
                if learned_phase is not None:
                    if target_mask_64 is None:
                        raise RuntimeError("learned phase requires Stage-B target mask")
                    oriented_depth = np.ascontiguousarray(
                        np.asarray(obs["agentview_depth"])[::-1, ::-1].squeeze()
                    )
                    visual_target_xyz = depth_target_estimator(
                        oriented_depth, target_mask_64
                    )
                    previous_recovery_phase = recovery_phase
                    recovery_phase, learned_phase_probabilities = learned_phase(
                        visual_target_xyz, obs, recovery_phase
                    )
                    if (
                        visual_regrasp is not None
                        and recovery_phase == "lift"
                        and previous_recovery_phase != "lift"
                    ):
                        visual_regrasp.reset()
                # Scene-specific Panda grasp offset measured from the verified
                # nominal task-9 trajectory. This is intentionally overfit.
                if recovery_phase == "return":
                    desired = bowl + np.asarray([-0.03, 0.06, 0.055], np.float32)
                    delta = desired - eef
                    executed = np.zeros(7, np.float32)
                    executed[:3] = np.clip(args.regrasp_gain * delta, -0.8, 0.8)
                    executed[-1] = -1.0
                    phase = "RETURN TO BOWL"
                    if (
                        learned_phase is None
                        and float(np.linalg.norm(delta)) < 0.018
                    ):
                        recovery_phase, recovery_phase_step = "descend", 0
                elif recovery_phase == "descend":
                    desired = bowl + np.asarray([-0.03, 0.06, 0.006], np.float32)
                    delta = desired - eef
                    executed = np.zeros(7, np.float32)
                    executed[:3] = np.clip(args.regrasp_gain * delta, -0.5, 0.5)
                    executed[-1] = -1.0
                    phase = "DESCEND"
                    if (
                        learned_phase is None
                        and float(np.linalg.norm(delta)) < 0.012
                    ):
                        recovery_phase, recovery_phase_step = "close", 0
                elif recovery_phase == "close":
                    executed = np.zeros(7, np.float32)
                    executed[-1] = 1.0
                    phase = "CLOSE GRIPPER"
                    recovery_phase_step += 1
                    if (
                        learned_phase is None
                        and recovery_phase_step >= args.regrasp_close_steps
                    ):
                        recovery_phase, recovery_phase_step = "lift", 0
                elif recovery_phase == "lift":
                    visually_verified = False
                    if visual_regrasp is not None and not visual_regrasp_settling:
                        if visual_target_xyz is None:
                            raise RuntimeError(
                                "visual regrasp verification requires visual target xyz"
                            )
                        (
                            visual_regrasp_probability,
                            visually_verified,
                        ) = visual_regrasp(visual_target_xyz, obs)
                        if visually_verified:
                            visual_regrasp_settling = True
                    oracle_verified = (
                        visual_regrasp is None
                        and float(bowl[2]) - initial_target_z
                        >= args.regrasp_lift_threshold_m
                    )
                    settle_target_z = (
                        visual_regrasp.eef_start[2]
                        + args.visual_regrasp_settle_height_m
                        if visual_regrasp_settling else None
                    )
                    settle_error = (
                        settle_target_z - eef[2]
                        if settle_target_z is not None else None
                    )
                    if (
                        visual_regrasp_settling
                        and abs(float(settle_error)) < 0.005
                    ) or oracle_verified:
                        recovery_completed = True
                        recovery_phase = None
                        phase, executed = "RESUMED TASK", nominal.copy()
                    elif visual_regrasp_settling:
                        executed = np.zeros(7, np.float32)
                        executed[2] = np.clip(8.0 * settle_error, -0.5, 0.5)
                        executed[-1] = 1.0
                        phase = "VISUAL SETTLE"
                    else:
                        executed = np.zeros(7, np.float32)
                        executed[2] = 0.65
                        executed[-1] = 1.0
                        phase = "VERIFY LIFT"
                        recovery_phase_step += 1
                else:
                    raise RuntimeError(f"unknown recovery phase {recovery_phase}")
                if learned_recovery is not None and phase in (
                    "RETURN TO BOWL", "DESCEND", "CLOSE GRIPPER", "VERIFY LIFT"
                ):
                    if target_mask_64 is None:
                        raise RuntimeError("learned recovery requires Stage-B target mask")
                    oriented_depth = np.ascontiguousarray(
                        np.asarray(obs["agentview_depth"])[::-1, ::-1].squeeze()
                    )
                    visual_target_xyz = depth_target_estimator(
                        oriented_depth, target_mask_64
                    )
                    executed, learned_residual = learned_recovery(
                        visual_target_xyz, eef, nominal, phase
                    )
            elif failure_detected and step < recovery_end:
                # OpenVLA replans from the perturbed observation every step.
                phase, executed = (
                    "RE-GRASP" if args.mistake_label in ("MISSED GRASP", "LOST GRASP")
                    else "RECOVERY"
                ), nominal.copy()
            elif failure_detected and step >= recovery_end:
                phase, executed = "RESUMED TASK", nominal.copy()
            else:
                phase, executed = "NOMINAL", nominal.copy()
            candidate_phases = (
                "RETURN TO BOWL",
                "DESCEND",
                "CLOSE GRIPPER",
                "VERIFY LIFT",
            )
            if (
                candidate_shadow_scorer is not None
                and phase in candidate_phases
            ):
                shadow_decision = candidate_shadow_scorer(
                    task_suite=args.task_suite,
                    task_id=args.task_id,
                    episode_id=dataset_episode_id,
                    timestep=step,
                    instruction=task.language,
                    obs=obs,
                    target_mask=target_mask,
                    nominal=nominal,
                    executed=executed,
                    previous_executed_action=previous_executed_action,
                    translation_delta=args.candidate_translation_delta,
                    rotation_delta=args.candidate_rotation_delta,
                )
                shadow_steps += 1
            should_collect_candidates = (
                fork_env is not None
                and candidate_path is not None
                and phase in candidate_phases
                and candidates_written < args.candidate_max_forks
                and failure_detected_step is not None
                and (
                    (step - failure_detected_step)
                    % args.candidate_collect_every
                    == 0
                )
            )
            if should_collect_candidates:
                candidate_group = collect_candidate_group(
                    fork_env=fork_env,
                    sim_state=env.get_sim_state().copy(),
                    task_suite=args.task_suite,
                    task_id=args.task_id,
                    episode_id=dataset_episode_id,
                    timestep=step,
                    split=args.candidate_split,
                    phase=phase,
                    nominal=nominal,
                    executed=executed,
                    target_name=target_name,
                    target_key=target_key,
                    initial_target_z=initial_target_z,
                    attached_before=attached_truth,
                    branch_steps=args.candidate_branch_steps,
                    translation_delta=args.candidate_translation_delta,
                    rotation_delta=args.candidate_rotation_delta,
                )
                with candidate_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            candidate_group.to_dict(), ensure_ascii=False
                        )
                        + "\n"
                    )
                candidates_written += 1
                if shadow_decision is not None:
                    teacher_values = [
                        (
                            float(outcome["teacher_value"])
                            if valid else None
                        )
                        for valid, outcome in zip(
                            candidate_group.outcome_valid,
                            candidate_group.teacher_outcomes,
                        )
                    ]
                    valid_values = [
                        value for value in teacher_values if value is not None
                    ]
                    selected = shadow_decision["selected_index"]
                    actual_best = max(
                        range(len(teacher_values)),
                        key=lambda index: (
                            float("-inf")
                            if teacher_values[index] is None
                            else teacher_values[index]
                        ),
                    )
                    selected_teacher = teacher_values[selected]
                    regret = (
                        None
                        if selected_teacher is None
                        else max(valid_values) - selected_teacher
                    )
                    shadow_decision.update(
                        {
                            "teacher_values": teacher_values,
                            "actual_best_index": actual_best,
                            "actual_best_name": candidate_group.candidate_names[
                                actual_best
                            ],
                            "matched_actual_best": selected == actual_best,
                            "teacher_regret": regret,
                        }
                    )
                    shadow_matched_groups += 1
                    shadow_matches += int(selected == actual_best)
                    if regret is not None:
                        shadow_regret_sum += regret
            before_eef = np.asarray(obs["robot0_eef_pos"], np.float32)
            next_obs, reward, done, info = env.step(executed.tolist())
            success = bool(env.check_success())
            after_eef = np.asarray(next_obs["robot0_eef_pos"], np.float32)
            frames.append(
                annotate(
                    image,
                    phase,
                    step,
                    success,
                    shadow_decision=shadow_decision,
                )
            )
            record = {
                "timestep": step, "phase": phase,
                "failure_injected": phase in ("FORCED ERROR", "MISSED GRASP", "LOST GRASP"),
                "failure_detected": detection_event,
                "target_lift_m": float(np.asarray(obs[target_key], np.float32)[2]) - initial_target_z,
                "target_mask": target_mask,
                "visual_rise_px": current_visual_rise,
                "peak_visual_rise_px": peak_visual_rise,
                "recovery_phase": recovery_phase,
                "recovery_completed": recovery_completed,
                "nominal_action": nominal.tolist(), "executed_action": executed.tolist(),
                "learned_recovery_residual": (
                    learned_residual.tolist() if learned_residual is not None else None
                ),
                "visual_target_xyz": (
                    visual_target_xyz.tolist() if visual_target_xyz is not None else None
                ),
                "learned_phase_probabilities": (
                    learned_phase_probabilities.tolist()
                    if learned_phase_probabilities is not None else None
                ),
                "visual_regrasp_probability": visual_regrasp_probability,
                "candidate_shadow": shadow_decision,
                "eef_delta_m": (after_eef - before_eef).tolist(),
                "robot_state_before": extract_robot_state(obs),
                "robot_state_after": extract_robot_state(next_obs),
                "privileged_object_state_before": extract_privileged_object_state(obs),
                "privileged_object_state_after": extract_privileged_object_state(next_obs),
                "teacher_contact_state_before": contact_state,
                "teacher_attached_truth_before": attached_truth,
                "reward": float(reward), "done": bool(done), "success": success,
                "info": {str(key): jsonable(value) for key, value in dict(info).items()},
            }
            with steps_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if expert_path is not None:
                frame_name = f"step-{step:04d}"
                rgb_path = expert_rgb_dir / f"{frame_name}.png"
                depth_path = expert_depth_dir / f"{frame_name}.npz"
                mask_path = (
                    expert_mask_dir / f"{frame_name}.npz"
                    if target_mask_64 is not None else None
                )
                imageio.imwrite(rgb_path, image)
                depth = np.ascontiguousarray(
                    np.asarray(obs["agentview_depth"])[::-1, ::-1].squeeze()
                )
                np.savez_compressed(depth_path, depth=depth.astype(np.float32))
                if mask_path is not None:
                    np.savez_compressed(mask_path, mask=target_mask_64)
                expert_record = {
                    "schema_version": 2,
                    "task_suite": args.task_suite,
                    "task_id": args.task_id,
                    "episode_id": dataset_episode_id,
                    "timestep": step,
                    "phase": phase,
                    "is_failure_injection": record["failure_injected"],
                    "is_failure_detection_event": detection_event,
                    "is_recovery_expert_action": phase in (
                        "RETURN TO BOWL", "DESCEND", "CLOSE GRIPPER", "VERIFY LIFT"
                    ),
                    "recovery_completed": recovery_completed,
                    "rgb": str(rgb_path.resolve()),
                    "depth": str(depth_path.resolve()),
                    "target_mask_64": (
                        str(mask_path.resolve()) if mask_path is not None else None
                    ),
                    "target_mask_measurement": target_mask,
                    "robot_state": extract_robot_state(obs),
                    "nominal_action": nominal.tolist(),
                    "expert_or_executed_action": executed.tolist(),
                    "teacher_privileged_object_state": extract_privileged_object_state(obs),
                    "teacher_contact_state": contact_state,
                    "teacher_attached_truth": attached_truth,
                    "success_after": success,
                }
                with expert_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(expert_record, ensure_ascii=False) + "\n")
            transitions_written += 1
            previous_executed_action = executed.copy()
            obs = next_obs
            if success:
                for _ in range(10):
                    frames.append(annotate(
                        np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
                        "SUCCESS", step, True,
                    ))
                break
    finally:
        env.close()
        if fork_env is not None:
            fork_env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, fps=args.fps)
    metadata.update({
        "completed": True,
        "success": success,
        "num_steps": transitions_written,
        "num_transitions": transitions_written,
        "num_video_frames": len(frames),
        "video": str(args.output.resolve()), "steps_jsonl": str(steps_path.resolve()),
        "failure_detected": failure_detected,
        "failure_detected_step": failure_detected_step,
        "effective_mistake_start": effective_mistake_start,
        "effective_mistake_end": mistake_end,
        "recovery_completed": recovery_completed,
        "peak_target_lift_m": peak_target_lift,
        "initial_target_mask_row_px": initial_target_row,
        "peak_visual_rise_px": peak_visual_rise,
        "expert_transitions_jsonl": (
            str(expert_path.resolve()) if expert_path is not None else None
        ),
        "candidate_groups": candidates_written,
        "candidates_jsonl": (
            str(candidate_path.resolve()) if candidate_path is not None else None
        ),
        "candidate_shadow_steps": shadow_steps,
        "candidate_shadow_matched_groups": shadow_matched_groups,
        "candidate_shadow_best_accuracy": (
            shadow_matches / shadow_matched_groups
            if shadow_matched_groups else None
        ),
        "candidate_shadow_mean_teacher_regret": (
            shadow_regret_sum / shadow_matched_groups
            if shadow_matched_groups else None
        ),
        "recovery_claim": (
            "successful nominal negative episode; no failure was injected"
            if args.disable_failure_injection
            else "successful only if env.check_success() is true after the forced error"
        ),
    })
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
