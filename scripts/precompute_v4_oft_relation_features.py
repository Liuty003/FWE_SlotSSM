#!/usr/bin/env python3
"""Freeze OFT features and RGB-D patch geometry for V4 relational recovery.

The collector schema is deliberately strict: this tool rejects every legacy
RGB-only source and makes the geometry-bearing feature file independent of the
OFT checkout afterwards.  V4 uses the 16x16 *agentview* half of the two-view
OFT visual tokens for Slot Attention; the frozen action hidden remains
conditioned on both agentview and wrist RGB as in nominal OFT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.data.v4_geometry import (
    patch_instance_labels, patch_median_points, robot_point_to_saved_image_xy, saved_depth_to_robot_points,
)
from recovery.data.v4_relation_features import V4RelationFeatures, save
from recovery.models.oft_l1_action_head import OFTL1RegressionActionHead
from scripts.smoke_unified_oft_slotssm_bridge import (
    DEFAULT_OFT_ROOT, center_crop_resize, dynamic_module, load_torch_state, normalized_proprio,
)


DEFAULT_OFT = Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oft-root", type=Path, default=DEFAULT_OFT_ROOT)
    parser.add_argument("--oft-checkpoint", type=Path, default=DEFAULT_OFT)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument(
        "--native-input-only", action="store_true",
        help=("Use RGB/OFT/proprio only: geometry placeholders are deterministic constants and "
              "depth/segmentation remain training-only supervision. Train with representation=slot/global."),
    )
    parser.add_argument("--decision-stride", type=int, default=8)
    parser.add_argument("--recovery-sampling-stride", type=int,
                        help="Post-fork observation spacing; V4.4.2 uses 1.")
    parser.add_argument(
        "--recovery-target-horizon", type=int, default=8,
        help="Number of teacher actions stored per recovery state; V4.4.2 uses 1.",
    )
    parser.add_argument("--handoff-manifest", type=Path,
                        help="V4.3 manifest: extract only certified fork-to-handoff recovery windows.")
    parser.add_argument(
        "--fixed-recovery-window-steps", type=int,
        help=("Extract successful fork-aligned recovery targets for this many simulator steps. "
              "This is the high-coverage data path when no per-source OFT handoff audit is required."),
    )
    parser.add_argument("--use-fixed-latch-handoff", action="store_true",
                        help="Use each record's audited fixed-latch handoff instead of its earliest handoff.")
    parser.add_argument("--include-handoff-gate-rows", action="store_true",
                        help="Add gate-only recovery-tail rows and an exact NORMAL handoff boundary row.")
    parser.add_argument("--include-prefork-gate-rows", action="store_true",
                        help="Also keep stride-aligned pre-fork rows as OFT-competent gate negatives.")
    parser.add_argument("--mark-all-recovery", action="store_true")
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Skip metadata.success!=true sources before OFT feature extraction (recommended for nominal negatives).",
    )
    parser.add_argument("--task-ids",
                        help="Optional comma-separated task filter applied before feature extraction.")
    parser.add_argument("--failed-prefix-gate-positive-steps", type=int,
                        help="Extract only this prefix of failed nominal sources as competence-only RECOVERY labels.")
    parser.add_argument(
        "--failed-gate-boundary-manifest", type=Path,
        help=("Label failed nominal decision rows before each manifest failure_timestep NORMAL "
              "and rows at/after it RECOVERY; no action targets are created."),
    )
    parser.add_argument(
        "--failed-gate-boundary-key", default="failure_timestep",
        choices=("failure_timestep", "event_timestep", "gate_context_timestep", "fork_timestep"),
        help="Timestamp field to use from each failed-source manifest record.",
    )
    parser.add_argument(
        "--failure-lead-steps", type=int, default=0,
        help="Move each failed-source RECOVERY boundary this many simulator steps earlier.",
    )
    parser.add_argument(
        "--gate-boundary-window-before", type=int,
        help="Keep only this many simulator steps before a failed gate boundary.",
    )
    parser.add_argument(
        "--gate-boundary-window-after", type=int,
        help="Keep only this many simulator steps after a failed gate boundary.",
    )
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--source-shard-index", type=int, default=0)
    parser.add_argument("--num-source-shards", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def split_group_key(path: Path, metadata: dict[str, Any]) -> str:
    """Identity used for train/validation isolation.

    All forks of one benchmark initial state deliberately share this key: their
    nominal prefixes overlap and must never be split across train and validation.
    """
    keys = ("task_suite", "task_id", "episode_id")
    return ("benchmark:" + ":".join(str(metadata[key]) for key in keys)
            if all(key in metadata for key in keys) else str(path.resolve()))


def stable_id(identity: str) -> int:
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") & ((1 << 63) - 1)


def source_id(path: Path, metadata: dict[str, Any]) -> int:
    return stable_id(split_group_key(path, metadata))


def source_instance_id(path: Path, metadata: dict[str, Any]) -> str:
    """Identify one concrete rollout/fork without changing its split group."""
    identity = {
        "split_group": split_group_key(path, metadata),
        "kind": metadata.get("kind"),
        "schema": metadata.get("schema"),
        "fork_timestep": metadata.get("fork_timestep"),
        "source_rollout": metadata.get("source_rollout"),
        "recovery_branch": metadata.get("recovery_branch"),
        "completion_branch": metadata.get("completion_branch"),
        "wait_steps": metadata.get("wait_steps"),
        "open_loop_steps": metadata.get("open_loop_steps"),
        "libero_root": metadata.get("libero_root"),
        "oft_checkpoint": metadata.get("oft_checkpoint"),
        "env_seed": metadata.get("env_seed"),
    }
    # Fresh collectors provide a path-independent rollout id.  The path is a
    # conservative fallback for old nominal data that has no branch provenance.
    if metadata.get("rollout_id") is not None:
        identity["rollout_id"] = metadata["rollout_id"]
    elif metadata.get("fork_timestep") is None:
        identity["legacy_path"] = str(path.resolve())
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def collection_protocol(metadata: dict[str, Any]) -> dict[str, int | None]:
    return {"wait_steps": metadata.get("wait_steps"), "open_loop_steps": metadata.get("open_loop_steps")}


def decision_starts(row_count: int, decision_stride: int, fork_timestep: int | None,
                    *, recovery_sampling_stride: int | None = None,
                    recovery_target_horizon: int | None = None,
                    recovery_handoff: int | None = None,
                    recovery_only: bool = False,
                    include_handoff_gate_rows: bool = False) -> list[int]:
    """Return the policy decision grid, restarting it at a recovery fork.

    Recovery can begin at an arbitrary simulator step (for example step 49),
    while OFT emits eight actions per decision.  A single grid anchored at zero
    silently skips the actual fork observation and trains on phase-shifted
    recovery chunks.  Keep the nominal grid for negative gate supervision and
    add a fork-anchored grid for every post-fork recovery decision.
    """
    last_start = row_count - decision_stride
    if last_start < 0:
        return []
    nominal = list(range(0, last_start + 1, decision_stride))
    if fork_timestep is None:
        return nominal
    if not isinstance(fork_timestep, int) or not 0 <= fork_timestep <= last_start:
        raise ValueError("fork_timestep must identify a complete action chunk")
    sampling_stride = recovery_sampling_stride or decision_stride
    target_horizon = recovery_target_horizon or decision_stride
    recovery_last = last_start
    if recovery_handoff is not None:
        if recovery_handoff <= fork_timestep or recovery_handoff > row_count:
            raise ValueError("recovery_handoff must be after the fork and inside the rollout")
        # A recovery target must not cross the certified OFT handoff.  This is
        # independent of the native OFT input/action-context chunk length.
        recovery_last = min(recovery_last, recovery_handoff - target_horizon)
    if include_handoff_gate_rows:
        if recovery_handoff is None:
            raise ValueError("handoff gate rows require recovery_handoff")
        # Tail states may have fewer than eight recovery actions left.  They
        # are still essential gate positives; the exact boundary is a gate
        # negative and carries no action target.
        recovery = set(range(fork_timestep, recovery_handoff, sampling_stride))
        recovery.add(recovery_handoff)
        return sorted(recovery if recovery_only else set(start for start in nominal if start < fork_timestep) | recovery)
    recovery = set(range(fork_timestep, recovery_last + 1, sampling_stride))
    return sorted(recovery if recovery_only else set(start for start in nominal if start < fork_timestep) | recovery)


def resolve_handoff(handoff_record: dict[str, Any] | None, use_fixed_latch: bool) -> int | None:
    """Resolve a recovery handoff without breaking ordinary nominal sources."""
    if handoff_record is None:
        return None
    key = "fixed_latch_handoff" if use_fixed_latch else "certified_local_handoff"
    value = handoff_record.get(key)
    if value is None:
        raise ValueError(f"handoff record is missing {key}")
    return int(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_action(action: list[float], stats: dict, np) -> Any:
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (7,):
        raise ValueError("an executed action must have shape [7]")
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)))
    return np.clip(np.where(mask, 2 * (value - low) / (high - low + 1e-8) - 1, value), -1, 1).astype(np.float32)


def executed_to_planned_action(action: list[float], np) -> Any:
    """Invert OFT's binary gripper post-processing for a simulator action.

    Translation/rotation coordinates are unchanged by ``process_action``.  Its
    gripper path maps raw policy 0 -> executed +1 (close) and raw policy 1 ->
    executed -1 (open), so the exact inverse on stored binary actions is
    ``(1 - executed) / 2``.
    """
    value = np.asarray(action, dtype=np.float32).copy()
    if value.shape != (7,):
        raise ValueError("an executed action must have shape [7]")
    if not np.isclose(abs(value[-1]), 1.0):
        raise ValueError(f"executed gripper action must be binary +/-1, got {value[-1]}")
    value[-1] = (1.0 - value[-1]) / 2.0
    return value


def decoder_target_action(row: dict[str, Any], np) -> Any:
    """Return the pre-``process_action`` action emitted by the OFT decoder."""
    if row.get("planned_action") is not None:
        value = np.asarray(row["planned_action"], dtype=np.float32)
        if value.shape != (7,):
            raise ValueError("planned_action must have shape [7]")
        return value
    if row.get("executed_action") is None:
        raise ValueError("a V4 transition requires planned_action or executed_action")
    return executed_to_planned_action(row["executed_action"], np)


def normalized_action_chunk(actions: list[list[float]], stats: dict, np,
                            expected_horizon: int = 8) -> Any:
    """Normalize a chunk already expressed in planned decoder coordinates."""
    values = np.stack([normalized_action(action, stats, np) for action in actions])
    if values.shape != (expected_horizon, 7):
        raise ValueError(f"action chunk must have shape [{expected_horizon},7], got {values.shape}")
    return values


def normalized_executed_action_chunk(actions: list[list[float]], stats: dict, np,
                                     expected_horizon: int = 8) -> Any:
    """Convert a stored simulator teacher chunk into decoder coordinates."""
    values = np.stack([
        normalized_action(executed_to_planned_action(action, np), stats, np) for action in actions
    ])
    if values.shape != (expected_horizon, 7):
        raise ValueError(
            f"recovery target must have physical shape [{expected_horizon},7], got {values.shape}"
        )
    return values


def exact_oft_policy_image(image, image_cls, tf):
    """Exactly reproduce the official OFT evaluation image pipeline.

    The JPEG roundtrip, Lanczos3 resize, and sqrt(0.9) TensorFlow crop are all
    intentional.  A PIL crop approximation changes actions enough to alter a
    long-horizon LIBERO rollout and must not be used for frozen features.
    """
    value = tf.convert_to_tensor(image)
    value = tf.image.encode_jpeg(value)
    value = tf.io.decode_image(value, expand_animations=False, dtype=tf.uint8)
    value = tf.image.resize(value, (224, 224), method="lanczos3", antialias=True)
    value = tf.cast(tf.clip_by_value(tf.round(value), 0, 255), tf.uint8)
    value = tf.image.convert_image_dtype(value, tf.float32)
    side = tf.sqrt(tf.constant(0.9, dtype=tf.float32))
    offset = (1.0 - side) / 2.0
    boxes = tf.reshape(tf.stack((offset, offset, offset + side, offset + side)), (1, 4))
    value = tf.image.crop_and_resize(value[None], boxes, tf.constant([0]), (224, 224))[0]
    value = tf.image.convert_image_dtype(tf.clip_by_value(value, 0, 1), tf.uint8, saturate=True)
    return image_cls.fromarray(value.numpy()).convert("RGB")


def observation(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("online_observation")
    if not isinstance(value, dict):
        raise ValueError("V4 extraction requires online_observation")
    if not isinstance(value.get("proprio"), dict):
        raise ValueError("V4 extraction requires proprio")
    direct = all(value.get(key) is not None for key in ("rgb", "wrist_rgb", "depth", "wrist_depth"))
    dense = value.get("dense_bundle") is not None and value.get("dense_index") is not None
    if not direct and not dense:
        raise ValueError("V4 extraction requires direct RGB-D paths or a dense archival bundle")
    if direct:
        for key in ("rgb", "wrist_rgb", "depth", "wrist_depth"):
            if not Path(value[key]).is_file():
                raise FileNotFoundError(value[key])
    elif not Path(value["dense_bundle"]).is_file():
        raise FileNotFoundError(value["dense_bundle"])
    return value


def gate_label(row: dict[str, Any], source_success: bool, mark_all_recovery: bool) -> int:
    if mark_all_recovery:
        if not source_success:
            raise ValueError("--mark-all-recovery requires metadata.success=true")
        return 1
    if not source_success:
        return -1
    value = row.get("gate_label")
    return 1 if value in (1, "RECOVERY") else 0 if value in (0, "NORMAL") else -1


def main() -> None:
    args = parse_args()
    if args.num_source_shards < 1 or not 0 <= args.source_shard_index < args.num_source_shards:
        raise ValueError("source-shard-index must be in [0,num-source-shards)")
    if args.decision_stride < 1:
        raise ValueError("--decision-stride must be positive")
    if args.recovery_sampling_stride is not None and args.recovery_sampling_stride < 1:
        raise ValueError("--recovery-sampling-stride must be positive")
    if not 1 <= args.recovery_target_horizon <= args.decision_stride:
        raise ValueError("--recovery-target-horizon must be in [1, decision-stride]")
    if args.fixed_recovery_window_steps is not None and args.fixed_recovery_window_steps < args.decision_stride:
        raise ValueError("--fixed-recovery-window-steps must contain at least one complete action chunk")
    if args.fixed_recovery_window_steps is not None and args.handoff_manifest is not None:
        raise ValueError("fixed recovery windows and handoff manifests are mutually exclusive")
    if args.fixed_recovery_window_steps is not None and not args.mark_all_recovery:
        raise ValueError("fixed recovery windows require --mark-all-recovery")
    if args.include_prefork_gate_rows and args.handoff_manifest is None:
        raise ValueError("--include-prefork-gate-rows requires --handoff-manifest")
    if args.failed_prefix_gate_positive_steps is not None:
        if args.failed_prefix_gate_positive_steps < 1:
            raise ValueError("--failed-prefix-gate-positive-steps must be positive")
        if args.handoff_manifest is not None or args.success_only or args.mark_all_recovery:
            raise ValueError("failed-prefix competence extraction is incompatible with handoff/success modes")
    if args.failure_lead_steps < 0:
        raise ValueError("--failure-lead-steps must be nonnegative")
    boundary_windows = (args.gate_boundary_window_before, args.gate_boundary_window_after)
    if any(value is not None and value < 0 for value in boundary_windows):
        raise ValueError("gate boundary windows must be nonnegative")
    if any(value is not None for value in boundary_windows) and args.failed_gate_boundary_manifest is None:
        raise ValueError("gate boundary windows require --failed-gate-boundary-manifest")
    if args.failed_gate_boundary_manifest is not None:
        if args.handoff_manifest is not None or args.success_only or args.mark_all_recovery:
            raise ValueError("failed gate-boundary extraction is incompatible with handoff/success modes")
        if args.failed_prefix_gate_positive_steps is not None:
            raise ValueError("choose only one failed-source gate labeling mode")
    task_filter = None
    if args.task_ids:
        task_filter = {int(value) for value in args.task_ids.split(",") if value.strip()}
        if not task_filter:
            raise ValueError("--task-ids did not contain any task ids")
    for path in (args.input_root, args.oft_root, args.oft_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor
    import prismatic
    import prismatic.training
    import prismatic.vla

    prismatic.__path__.append(str(args.oft_root / "prismatic"))
    prismatic.training.__path__.append(str(args.oft_root / "prismatic" / "training"))
    prismatic.vla.__path__.append(str(args.oft_root / "prismatic" / "vla"))
    projectors = dynamic_module("v4_oft_projectors", args.oft_root / "prismatic" / "models" / "projectors.py")
    device = torch.device("cuda")
    model = AutoModelForVision2Seq.from_pretrained(str(args.oft_checkpoint), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to(device).eval()
    model.vision_backbone.set_num_images_in_input(2)
    model.norm_stats = json.loads((args.oft_checkpoint / "dataset_statistics.json").read_text())
    processor = AutoProcessor.from_pretrained(str(args.oft_checkpoint), trust_remote_code=True)
    action_head = OFTL1RegressionActionHead(4096, 4096, 7)
    action_head.load_state_dict(load_torch_state(args.oft_checkpoint / "action_head--150000_checkpoint.pt", torch))
    action_head = action_head.to(device=device, dtype=torch.bfloat16).eval()
    proprio_projector = projectors.ProprioProjector(4096, 8)
    proprio_projector.load_state_dict(load_torch_state(args.oft_checkpoint / "proprio_projector--150000_checkpoint.pt", torch))
    proprio_projector = proprio_projector.to(device=device, dtype=torch.bfloat16).eval()
    handoff_records = None
    failed_gate_boundaries = None
    if args.failed_gate_boundary_manifest is not None:
        boundary_payload = json.loads(args.failed_gate_boundary_manifest.read_text())
        records = boundary_payload.get("records")
        if not isinstance(records, list):
            raise ValueError("failed gate-boundary manifest must contain records")
        failed_gate_boundaries = {
            int(record.get("task_id", record["sample_id"].split(":task")[1].split(":")[0])):
                max(0, int(record[args.failed_gate_boundary_key]) - args.failure_lead_steps)
            for record in records if record.get(args.failed_gate_boundary_key) is not None
        }
        if not failed_gate_boundaries:
            raise ValueError("failed gate-boundary manifest has no failure timesteps")
    if args.handoff_manifest is not None:
        handoff_payload = json.loads(args.handoff_manifest.read_text())
        if handoff_payload.get("kind") != "v43_local_recovery_handoff_manifest":
            raise ValueError("--handoff-manifest is not a V4.3 local-recovery manifest")
        handoff_records = {
            int(record["task_id"]): record for record in handoff_payload["records"]
            if record.get("status") == "certified_local_recovery"
        }
        source_paths = [Path(record["source"]) / "steps.jsonl" for record in handoff_records.values()]
        if not args.mark_all_recovery:
            raise ValueError("V4.3 handoff extraction requires --mark-all-recovery")
        if args.use_fixed_latch_handoff and not handoff_payload.get("fixed_latch_all_sources_certified"):
            raise ValueError("fixed-latch extraction requires successful fixed handoff audits for every source")
    else:
        source_paths = sorted(args.input_root.rglob("steps.jsonl"))
    source_paths = [
        path for index, path in enumerate(source_paths)
        if index % args.num_source_shards == args.source_shard_index
    ]
    if not source_paths:
        raise ValueError(f"no steps.jsonl below {args.input_root}")
    fields: dict[str, list] = {key: [] for key in (
        "visual_tokens", "action_hidden", "token_points", "token_valid", "token_image_xy",
        "gripper_position", "gripper_image_xy", "actions", "gate_labels", "recovery_targets",
        "recovery_mask", "source_ids", "instance_labels",
    )}
    processed = []
    seen_instances: set[str] = set()
    for steps_path in source_paths[:args.max_sources]:
        metadata_path = steps_path.with_name("metadata.json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"V4 source is missing {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        if task_filter is not None and int(metadata.get("task_id", -1)) not in task_filter:
            continue
        source_success = metadata.get("success") is True
        if args.success_only and not source_success:
            continue
        if args.failed_prefix_gate_positive_steps is not None and source_success:
            continue
        if failed_gate_boundaries is not None:
            if source_success or int(metadata.get("task_id", -1)) not in failed_gate_boundaries:
                continue
        calibration = metadata.get("agentview_calibration")
        if calibration is None:
            calibration = metadata.get("camera_calibration", {}).get("agentview")
        accepted_schemas = {"v4_rgbd_rollout_v1", "v4_rgbd_rollout_v2"}
        if args.native_input_only:
            accepted_schemas.add("oft_archival_rollout_v4")
        if metadata.get("schema") not in accepted_schemas or not isinstance(calibration, dict):
            raise ValueError(f"{steps_path}: requires fresh V4 RGB-D data, not legacy V3 data")
        if metadata.get("schema") == "v4_rgbd_rollout_v2":
            protocol = collection_protocol(metadata)
            if not isinstance(protocol["wait_steps"], int) or protocol["wait_steps"] < 0:
                raise ValueError(f"{metadata_path}: v2 requires nonnegative integer wait_steps")
            if not isinstance(protocol["open_loop_steps"], int) or protocol["open_loop_steps"] < 1:
                raise ValueError(f"{metadata_path}: v2 requires positive integer open_loop_steps")
            action_query_horizon = metadata.get("action_query_horizon", protocol["open_loop_steps"])
            if not isinstance(action_query_horizon, int) or action_query_horizon < 1:
                raise ValueError(f"{metadata_path}: action_query_horizon must be a positive integer")
            if action_query_horizon != args.decision_stride:
                raise ValueError(
                    f"{metadata_path}: action_query_horizon={action_query_horizon} does not match "
                    f"feature decision_stride={args.decision_stride}"
                )
        camera_to_robot = np.asarray(calibration["camera_to_robot"], dtype=np.float64)
        world_to_robot_value = calibration.get("world_to_robot")
        if world_to_robot_value is None and not args.native_input_only:
            raise ValueError(f"{metadata_path}: relation geometry requires world_to_robot")
        world_to_robot = (
            np.eye(4, dtype=np.float64) if world_to_robot_value is None
            else np.asarray(world_to_robot_value, dtype=np.float64)
        )
        near, far = float(calibration["near"]), float(calibration["far"])
        source_rows = rows(steps_path)
        first_observation = observation(source_rows[0])
        dense_archive = None
        if first_observation.get("dense_bundle") is not None:
            dense_archive = np.load(first_observation["dense_bundle"], allow_pickle=False)
        language = metadata.get("task_language")
        if not isinstance(language, str):
            raise ValueError(f"{metadata_path}: task_language is required")
        identity = source_id(steps_path.parent, metadata)
        instance_identity = source_instance_id(steps_path.parent, metadata)
        if instance_identity in seen_instances:
            raise ValueError(f"{steps_path}: duplicate concrete V4 rollout/fork identity")
        seen_instances.add(instance_identity)
        policy_prompt = f"In: What action should the robot take to {language.lower()}?\nOut:"
        accepted = 0
        handoff_record = None if handoff_records is None else handoff_records.get(int(metadata["task_id"]))
        if handoff_records is not None and handoff_record is None:
            continue
        fork_timestep = (metadata.get("fork_timestep") if handoff_record is None
                         else int(handoff_record["fork_timestep"]))
        selected_handoff = resolve_handoff(handoff_record, args.use_fixed_latch_handoff)
        if args.fixed_recovery_window_steps is not None:
            if not source_success or not isinstance(fork_timestep, int):
                raise ValueError("fixed recovery windows require successful forked rollouts")
            selected_handoff = min(len(source_rows), fork_timestep + args.fixed_recovery_window_steps)
        starts = decision_starts(
            len(source_rows), args.decision_stride, fork_timestep,
            recovery_sampling_stride=args.recovery_sampling_stride,
            recovery_target_horizon=args.recovery_target_horizon,
            recovery_handoff=selected_handoff,
            recovery_only=bool(
                (handoff_record is not None or args.fixed_recovery_window_steps is not None)
                and not args.include_prefork_gate_rows
            ),
            include_handoff_gate_rows=bool(handoff_record is not None and args.include_handoff_gate_rows),
        )
        if args.failed_prefix_gate_positive_steps is not None:
            starts = [start for start in starts if start < args.failed_prefix_gate_positive_steps]
        if failed_gate_boundaries is not None and any(value is not None for value in boundary_windows):
            boundary = failed_gate_boundaries[int(metadata["task_id"])]
            before = args.gate_boundary_window_before
            after = args.gate_boundary_window_after
            starts = [
                start for start in starts
                if (before is None or start >= boundary - before)
                and (after is None or start <= boundary + after)
            ]
        for start in starts:
            row = source_rows[start]
            obs = observation(row)
            raw_proprio = obs["proprio"]
            if dense_archive is None:
                main_rgb = np.asarray(Image.open(obs["rgb"]).convert("RGB"))
                wrist_rgb = np.asarray(Image.open(obs["wrist_rgb"]).convert("RGB"))
                depth = np.load(obs["depth"])["depth"]
                instance_path = row.get("training_only", {}).get("instance_segmentation")
                instance = None if instance_path is None else np.load(instance_path)["instance"]
            else:
                if Path(obs["dense_bundle"]).resolve() != Path(first_observation["dense_bundle"]).resolve():
                    raise ValueError(f"{steps_path}: one source may reference only one dense bundle")
                dense_index = int(obs["dense_index"])
                keys = obs.get("dense_keys", {})
                main_rgb = np.asarray(dense_archive[keys.get("rgb", "rgb")][dense_index])
                wrist_rgb = np.asarray(dense_archive[keys.get("wrist_rgb", "wrist_rgb")][dense_index])
                depth = np.asarray(dense_archive[keys.get("depth", "depth")][dense_index])
                instance_key = keys.get("instance_segmentation", "agentview_instance")
                instance = (np.asarray(dense_archive[instance_key][dense_index])
                            if instance_key in dense_archive.files else None)
            inputs = processor(policy_prompt, exact_oft_policy_image(main_rgb, Image, tf)).to(device, dtype=torch.bfloat16)
            wrist = processor(policy_prompt, exact_oft_policy_image(wrist_rgb, Image, tf)).to(device, dtype=torch.bfloat16)
            inputs["pixel_values"] = torch.cat((inputs["pixel_values"], wrist["pixel_values"]), dim=1)
            norm_key = args.task_suite if args.task_suite in model.norm_stats else f"{args.task_suite}_no_noops"
            if norm_key not in model.norm_stats:
                raise KeyError(f"OFT checkpoint has no statistics for {args.task_suite}")
            normalized = normalized_proprio(raw_proprio, model.norm_stats[norm_key]["proprio"], np)
            with torch.inference_mode():
                visual = model.vision_backbone(inputs["pixel_values"])
                _, action_hidden = model.predict_action(**inputs, unnorm_key=norm_key, do_sample=False, proprio=normalized,
                                                        proprio_projector=proprio_projector, action_head=action_head, use_film=False)
            # Two 16x16 views are concatenated by OFT.  Geometry is deliberately
            # agentview-only for V4-v1: wrist extrinsics move with the arm and
            # need a separate kinematic calibration before inclusion.
            visual = visual[0, :256].float().cpu()
            if args.native_input_only:
                # Slot/global representations consume visual tokens directly.
                # Constants preserve the tensor interface without smuggling RGB-D
                # geometry into deployment.
                dummy_points = np.zeros((*depth.shape, 3), dtype=np.float32)
                points, valid, image_xy = patch_median_points(
                    dummy_points, np.ones(depth.shape, dtype=bool), 16, 16,
                )
            else:
                dense_points, dense_valid = saved_depth_to_robot_points(depth, camera_to_robot, near, far)
                points, valid, image_xy = patch_median_points(dense_points, dense_valid, 16, 16)
            instance_labels = patch_instance_labels(instance, 16, 16)
            if args.native_input_only:
                eef_robot = np.zeros(3, dtype=np.float32)
                gripper_xy = np.zeros(2, dtype=np.float32)
            else:
                eef_world = np.asarray(raw_proprio["robot0_eef_pos"], dtype=np.float64)
                eef_robot = (world_to_robot @ np.concatenate((eef_world, [1.0])))[:3].astype(np.float32)
                gripper_xy, _ = robot_point_to_saved_image_xy(
                    eef_robot, camera_to_robot, depth.shape[0], depth.shape[1],
                )
            chunk = np.stack([normalized_action(decoder_target_action(item, np), model.norm_stats[norm_key]["action"], np)
                              for item in source_rows[start:start + args.decision_stride]])
            is_handoff_boundary = bool(
                handoff_record is not None and args.include_handoff_gate_rows and start == selected_handoff)
            is_prefork_negative = bool(handoff_record is not None and start < fork_timestep)
            if args.failed_prefix_gate_positive_steps is not None:
                label = 1
            elif failed_gate_boundaries is not None:
                label = int(start >= failed_gate_boundaries[int(metadata["task_id"])])
            else:
                label = 0 if is_handoff_boundary or is_prefork_negative else gate_label(
                    row, source_success, args.mark_all_recovery)
            target = row.get("recovery_target_action_chunk")
            target_allowed = label == 1 and (
                handoff_record is None or start + args.recovery_target_horizon <= selected_handoff)
            if not target_allowed:
                target = None
            if target is None and target_allowed and (args.mark_all_recovery or source_success):
                # ``chunk`` is already normalized above.
                # A successful canonical recovery trace is itself the teacher.
                # In particular, collectors may not have stored an explicit
                # target at a non-stride-aligned fork; do not discard the most
                # important first recovery chunk in that case.
                target_array = chunk[:args.recovery_target_horizon].copy()
                target_available = True
            elif target is None:
                target_array = np.zeros((args.recovery_target_horizon, chunk.shape[-1]), dtype=chunk.dtype)
                target_available = False
            else:
                # Collectors store physical executed actions.  The frozen OFT
                # head predicts normalized actions, so targets must use the
                # exact same checkpoint statistics as ordinary action chunks.
                target_array = normalized_executed_action_chunk(
                    target[:args.recovery_target_horizon], model.norm_stats[norm_key]["action"], np,
                    expected_horizon=args.recovery_target_horizon,
                )
                target_available = True
            if target_array.shape != (args.recovery_target_horizon, chunk.shape[-1]):
                raise ValueError(
                    f"{steps_path}:{start} recovery target must match "
                    f"[{args.recovery_target_horizon},{chunk.shape[-1]}]"
                )
            fields["visual_tokens"].append(visual)
            fields["action_hidden"].append(action_hidden[0].float().cpu())
            fields["token_points"].append(torch.from_numpy(points)); fields["token_valid"].append(torch.from_numpy(valid))
            fields["token_image_xy"].append(torch.from_numpy(image_xy)); fields["gripper_position"].append(torch.from_numpy(eef_robot))
            fields["gripper_image_xy"].append(torch.from_numpy(gripper_xy)); fields["actions"].append(torch.from_numpy(chunk))
            fields["gate_labels"].append(label); fields["recovery_targets"].append(torch.from_numpy(target_array))
            fields["recovery_mask"].append(target_available); fields["source_ids"].append(identity)
            fields["instance_labels"].append(torch.from_numpy(instance_labels))
            accepted += 1
        if dense_archive is not None:
            dense_archive.close()
        processed.append({"source": str(steps_path.parent), "source_id": identity,
                          "source_instance_id": instance_identity, "split_group": split_group_key(steps_path.parent, metadata),
                          "fork_timestep": fork_timestep,
                          "certified_local_handoff": None if handoff_record is None else selected_handoff,
                          "fixed_recovery_window_handoff": (
                              selected_handoff if args.fixed_recovery_window_steps is not None else None
                          ),
                          "fixed_latch_handoff": bool(handoff_record is not None and args.use_fixed_latch_handoff),
                          **collection_protocol(metadata),
                          "rows": accepted, "success": source_success})
    if not fields["visual_tokens"]:
        raise ValueError("no complete action chunks found")
    features = V4RelationFeatures(
        visual_tokens=torch.stack(fields["visual_tokens"]), action_hidden=torch.stack(fields["action_hidden"]),
        token_points=torch.stack(fields["token_points"]), token_valid=torch.stack(fields["token_valid"]),
        token_image_xy=torch.stack(fields["token_image_xy"]), gripper_position=torch.stack(fields["gripper_position"]),
        gripper_image_xy=torch.stack(fields["gripper_image_xy"]), actions=torch.stack(fields["actions"]),
        gate_labels=torch.tensor(fields["gate_labels"], dtype=torch.int64), recovery_targets=torch.stack(fields["recovery_targets"]),
        recovery_mask=torch.tensor(fields["recovery_mask"], dtype=torch.bool), source_ids=torch.tensor(fields["source_ids"], dtype=torch.int64),
        instance_labels=torch.stack(fields["instance_labels"]),
    )
    save(features, args.output)
    report = {"kind": "v4_frozen_oft_relation_features", "output": str(args.output), "rows": features.num_rows,
              "sources": processed, "oft_checkpoint": str(args.oft_checkpoint.resolve()),
              "action_head_sha256": sha256(args.oft_checkpoint / "action_head--150000_checkpoint.pt"),
              "proprio_projector_sha256": sha256(args.oft_checkpoint / "proprio_projector--150000_checkpoint.pt"),
              "action_normalization_key": norm_key,
              "decision_stride": args.decision_stride,
              "recovery_sampling_stride": args.recovery_sampling_stride or args.decision_stride,
              "recovery_target_horizon": args.recovery_target_horizon,
              "handoff_manifest": None if args.handoff_manifest is None else str(args.handoff_manifest.resolve()),
              "fixed_recovery_window_steps": args.fixed_recovery_window_steps,
              "include_handoff_gate_rows": args.include_handoff_gate_rows,
              "include_prefork_gate_rows": args.include_prefork_gate_rows,
              "success_only": args.success_only,
              "failed_prefix_gate_positive_steps": args.failed_prefix_gate_positive_steps,
              "failed_gate_boundary_manifest": (None if args.failed_gate_boundary_manifest is None
                                                else str(args.failed_gate_boundary_manifest.resolve())),
              "failed_gate_boundary_key": args.failed_gate_boundary_key,
              "failure_lead_steps": args.failure_lead_steps,
              "gate_boundary_window_before": args.gate_boundary_window_before,
              "gate_boundary_window_after": args.gate_boundary_window_after,
              "sampling_contract": (
                  f"certified fork-to-handoff rows; {args.recovery_target_horizon}-action targets never cross handoff"
                  if args.handoff_manifest is not None else
                  f"successful fixed recovery windows; {args.recovery_target_horizon}-action targets never cross the configured window"
                  if args.fixed_recovery_window_steps is not None else
                  "V4.2 nominal/fork-aligned grids"
              ),
              "recovery_target_coordinates": "checkpoint-normalized action coordinates",
              "decoder_action_contract": "pre-process_action planned coordinates; executed gripper inverted by (1-g)/2",
              "image_preprocessing": "official OFT JPEG+TF-Lanczos3+sqrt(0.9)-crop_and_resize",
              "decision_grid_contract": "nominal grid from zero; recovery grid restarts at fork_timestep",
              "recovery_target_contract": "successful RECOVERY rows use canonical executed chunks when explicit target is absent",
              "source_identity_contract": "v2: source_id is benchmark initial-state split group; source_instance_id is concrete rollout/fork",
              "source_shard_index": args.source_shard_index,
              "num_source_shards": args.num_source_shards,
              "native_input_only": args.native_input_only,
              "representation_input": (
                  "dual-view OFT visual/action tokens; deterministic geometry placeholders; depth/segmentation training-only"
                  if args.native_input_only else
                  "agentview OFT visual tokens + agentview RGB-D robot-frame geometry; action hidden remains dual-view OFT"
              )}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
