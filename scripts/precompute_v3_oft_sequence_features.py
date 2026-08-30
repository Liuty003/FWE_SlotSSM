#!/usr/bin/env python3
"""Freeze causal OFT feature sequences for v3 representation training.

The input is a directory containing recorded ``steps.jsonl`` trajectories (for
example produced by ``collect_oft_nominal_trajectories.py --save-frame-images``
or a recovery teacher collector).  It creates fixed-length *decision*
sequences; each decision is separated by an eight-action OFT chunk by default.
No simulator state, pose, contact, candidate, or FSM field is written to the
resulting feature file.
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

from recovery.data.v3_sequence_features import V3SequenceFeatures, save
from scripts.smoke_unified_oft_slotssm_bridge import (
    DEFAULT_OFT_ROOT,
    center_crop_resize,
    dynamic_module,
    load_torch_state,
    normalized_proprio,
)


DEFAULT_OFT = Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Recursively scan this root for replayable steps.jsonl files.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oft-root", type=Path, default=DEFAULT_OFT_ROOT)
    parser.add_argument("--oft-checkpoint", type=Path, default=DEFAULT_OFT)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--sequence-decisions", type=int, default=8)
    parser.add_argument("--decision-stride", type=int, default=8,
                        help="Number of low-level recorded actions in one OFT decision chunk.")
    parser.add_argument("--sequence-stride-decisions", type=int, default=8,
                        help="Start spacing in decisions; avoid overlap for strict source-isolated validation.")
    parser.add_argument("--max-sources", type=int)
    parser.add_argument(
        "--allow-legacy-planned-actions", action="store_true",
        help=("Compatibility-only: accept old oft_action_chunk_start in place of executed_action. "
              "The output is marked unsuitable for final action-effect evaluation."),
    )
    parser.add_argument(
        "--mark-all-recovery", action="store_true",
        help=("Use only on canonical successful recovery-teacher rollouts: label every decision "
              "RECOVERY and use its executed chunk as the unique recovery target."),
    )
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_id(path: Path, metadata: dict[str, Any]) -> int:
    """Return a split identity, grouping branches with their physical episode.

    A canonical recovery branch and a nominal rollout may live at different
    paths but originate from the same task initial state.  Split on that
    physical source whenever provenance is available; only legacy inputs fall
    back to their filesystem path.
    """
    keys = ("task_suite", "task_id", "episode_id")
    if all(key in metadata for key in keys):
        identity = "benchmark:" + ":".join(str(metadata[key]) for key in keys)
    else:
        identity = str(path.resolve())
    # safetensors only stores tensors; retain an unsigned stable 63-bit source id.
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") & ((1 << 63) - 1)


def action_to_normalized(action: list[float], stats: dict[str, Any], np) -> Any:
    """Normalize the *executed* 7-D action using the frozen OFT statistics."""
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (7,):
        raise ValueError("each executed_action must have seven values")
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    return np.clip(np.where(mask, 2.0 * (value - low) / (high - low + 1e-8) - 1.0, value), -1.0, 1.0)


def row_observation(row: dict[str, Any]) -> dict[str, Any]:
    observation = row.get("online_observation")
    if not isinstance(observation, dict):
        raise ValueError("v3 extraction requires policy-visible online_observation")
    for key in ("rgb", "wrist_rgb", "proprio"):
        if key not in observation:
            raise ValueError(f"online_observation missing {key}")
    if not Path(observation["rgb"]).is_file() or not Path(observation["wrist_rgb"]).is_file():
        raise FileNotFoundError("v3 extraction requires saved RGB and wrist RGB image paths")
    return observation


def gate_label(row: dict[str, Any]) -> int:
    value = row.get("gate_label")
    if value in (0, "NORMAL"):
        return 0
    if value in (1, "RECOVERY"):
        return 1
    return -1


def main() -> None:
    args = parse_args()
    if args.sequence_decisions < 2 or args.decision_stride < 1 or args.sequence_stride_decisions < 1:
        raise ValueError("sequence-decisions must be >=2 and strides must be positive")
    for path in (args.input_root, args.oft_root, args.oft_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor
    import prismatic
    import prismatic.training
    import prismatic.vla

    # A successful collector output intentionally contains both files.  They
    # are one physical source, not two independent examples: processing both
    # would duplicate it under two different source_ids and leak it across a
    # source-disjoint split.  Prefer the complete audit log (steps) and only
    # fall back to transitions for legacy teacher directories.
    paths = sorted(args.input_root.rglob("steps.jsonl"))
    paths.extend(
        path for path in sorted(args.input_root.rglob("transitions.jsonl"))
        if not (path.parent / "steps.jsonl").exists()
    )
    if args.max_sources is not None:
        paths = paths[:args.max_sources]
    if not paths:
        raise RuntimeError("no steps.jsonl files found")
    prismatic.__path__.append(str(args.oft_root / "prismatic"))
    prismatic.training.__path__.append(str(args.oft_root / "prismatic" / "training"))
    prismatic.vla.__path__.append(str(args.oft_root / "prismatic" / "vla"))
    action_components = dynamic_module("oft_v3_action_heads", args.oft_root / "prismatic" / "models" / "action_heads.py")
    projector_components = dynamic_module("oft_v3_projectors", args.oft_root / "prismatic" / "models" / "projectors.py")
    device = torch.device("cuda")
    model = AutoModelForVision2Seq.from_pretrained(
        str(args.oft_checkpoint), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device).eval()
    model.vision_backbone.set_num_images_in_input(2)
    model.norm_stats = json.loads((args.oft_checkpoint / "dataset_statistics.json").read_text())
    norm_key = args.task_suite if args.task_suite in model.norm_stats else f"{args.task_suite}_no_noops"
    if norm_key not in model.norm_stats:
        raise KeyError(f"OFT checkpoint has no stats for {args.task_suite}")
    processor = AutoProcessor.from_pretrained(str(args.oft_checkpoint), trust_remote_code=True)
    action_head = action_components.L1RegressionActionHead(4096, 4096, 7)
    action_head.load_state_dict(load_torch_state(args.oft_checkpoint / "action_head--150000_checkpoint.pt", torch))
    action_head = action_head.to(device=device, dtype=torch.bfloat16).eval()
    proprio_projector = projector_components.ProprioProjector(4096, 8)
    proprio_projector.load_state_dict(load_torch_state(args.oft_checkpoint / "proprio_projector--150000_checkpoint.pt", torch))
    proprio_projector = proprio_projector.to(device=device, dtype=torch.bfloat16).eval()

    visual_rows, hidden_rows, proprio_rows, action_rows, gate_rows, source_rows = [], [], [], [], [], []
    recovery_rows, recovery_masks = [], []
    legacy_actions_seen = False

    def policy_input(observation: dict[str, Any], language: str):
        main = imageio.imread(observation["rgb"])
        wrist = imageio.imread(observation["wrist_rgb"])
        prompt = f"In: What action should the robot take to {language.lower()}?\nOut:"
        primary = processor(prompt, center_crop_resize(main, Image)).to(device, dtype=torch.bfloat16)
        wrist_input = processor(prompt, center_crop_resize(wrist, Image)).to(device, dtype=torch.bfloat16)
        primary["pixel_values"] = torch.cat((primary["pixel_values"], wrist_input["pixel_values"]), dim=1)
        return primary

    with torch.inference_mode():
        for path in paths:
            rows = read_jsonl(path)
            if not rows:
                continue
            metadata_path = path.parent / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"missing trajectory metadata: {metadata_path}")
            metadata = json.loads(metadata_path.read_text())
            language = metadata.get("task_language")
            if not isinstance(language, str):
                raise ValueError(f"metadata lacks task_language: {metadata_path}")
            source_success = metadata.get("success")
            if args.mark_all_recovery and source_success is not True:
                raise ValueError(
                    f"--mark-all-recovery requires an explicitly successful canonical teacher: {metadata_path}"
                )
            # Validate the full causal trace before spending OFT compute.
            uses_legacy_planned_actions = False
            for row in rows:
                row_observation(row)
                if "executed_action" not in row:
                    if not args.allow_legacy_planned_actions or "oft_action_chunk_start" not in row:
                        raise ValueError(
                            f"{path} does not record executed_action; use only the explicit "
                            "--allow-legacy-planned-actions compatibility mode for old traces"
                        )
                    uses_legacy_planned_actions = True
            legacy_actions_seen = legacy_actions_seen or uses_legacy_planned_actions
            # Every complete H-action block is a valid policy decision.  The
            # old ``(len-H)//H`` expression dropped the final full block (and
            # rejected a teacher with exactly H*T actions altogether).
            decision_count = len(rows) // args.decision_stride
            # A decision's frozen OFT features do not depend on which causal
            # window later contains it.  Cache them once per physical source:
            # overlapping windows previously repeated the same expensive
            # vision/action forward pass up to ``sequence_decisions`` times.
            decision_cache = {}
            for decision_start in range(0, decision_count - args.sequence_decisions + 1, args.sequence_stride_decisions):
                visual_sequence, hidden_sequence, proprio_sequence, action_sequence, labels = [], [], [], [], []
                recovery_sequence, recovery_sequence_mask = [], []
                for decision in range(args.sequence_decisions):
                    start = (decision_start + decision) * args.decision_stride
                    if start not in decision_cache:
                        observation = row_observation(rows[start])
                        inputs = policy_input(observation, language)
                        raw_proprio = observation["proprio"]
                        normalized = normalized_proprio(raw_proprio, model.norm_stats[norm_key]["proprio"], np)
                        patches = model.vision_backbone(inputs["pixel_values"])
                        _, action_hidden = model.predict_action(
                            **inputs, unnorm_key=norm_key, do_sample=False, proprio=normalized,
                            proprio_projector=proprio_projector, action_head=action_head, use_film=False,
                        )
                        chunk = [
                            (action_to_normalized(row["executed_action"], model.norm_stats[norm_key]["action"], np)
                             if "executed_action" in row else np.asarray(row["oft_action_chunk_start"], dtype=np.float32))
                            for row in (rows[start + offset] for offset in range(args.decision_stride))
                        ]
                        # An incomplete nominal rollout is valuable for the
                        # self-supervised action-effect loss, but it is not a
                        # NORMAL example.  The collector leaves it in steps.jsonl
                        # for audit; gate supervision must ignore it.
                        label = 1 if args.mark_all_recovery else (
                            -1 if source_success is False else gate_label(rows[start])
                        )
                        target = rows[start].get("recovery_target_action_chunk")
                        if target is None and args.mark_all_recovery:
                            target = np.stack(chunk).tolist()
                        if target is None:
                            target_tensor = torch.zeros(args.decision_stride, 7, dtype=torch.float32)
                            target_mask = False
                        else:
                            target_tensor = torch.tensor(target, dtype=torch.float32)
                            if target_tensor.shape != (args.decision_stride, 7):
                                raise ValueError(
                                    "recovery_target_action_chunk must have shape "
                                    f"[{args.decision_stride},7] at {path}:{start}"
                                )
                            target_mask = True
                        decision_cache[start] = (
                            patches[0].float().cpu(), action_hidden[0].float().cpu(),
                            torch.tensor(normalized, dtype=torch.float32),
                            torch.tensor(np.stack(chunk), dtype=torch.float32),
                            label, target_tensor, target_mask,
                        )
                    visual, hidden, proprio, actions, label, target, target_mask = decision_cache[start]
                    visual_sequence.append(visual)
                    hidden_sequence.append(hidden)
                    proprio_sequence.append(proprio)
                    action_sequence.append(actions)
                    labels.append(label)
                    recovery_sequence.append(target)
                    recovery_sequence_mask.append(target_mask)
                visual_rows.append(torch.stack(visual_sequence))
                hidden_rows.append(torch.stack(hidden_sequence))
                proprio_rows.append(torch.stack(proprio_sequence))
                action_rows.append(torch.stack(action_sequence))
                gate_rows.append(torch.tensor(labels, dtype=torch.int64))
                recovery_rows.append(torch.stack(recovery_sequence))
                recovery_masks.append(torch.tensor(recovery_sequence_mask, dtype=torch.bool))
                source_rows.append(source_id(path, metadata))
    if not visual_rows:
        raise RuntimeError("no complete v3 decision sequences found; record more than H*T actions per source")
    features = V3SequenceFeatures(
        visual_tokens=torch.stack(visual_rows), action_hidden=torch.stack(hidden_rows),
        proprio=torch.stack(proprio_rows), actions=torch.stack(action_rows),
        gate_labels=torch.stack(gate_rows), recovery_targets=torch.stack(recovery_rows),
        recovery_mask=torch.stack(recovery_masks), source_ids=torch.tensor(source_rows, dtype=torch.int64),
    )
    save(features, args.output)
    metadata = {
        "kind": "v3_frozen_oft_causal_sequences", "input_root": str(args.input_root),
        "sources_scanned": len(paths), "sequences": features.num_sequences,
        "sequence_decisions": args.sequence_decisions, "decision_stride": args.decision_stride,
        "frozen_oft_checkpoint": str(args.oft_checkpoint),
        "action_representation": (
            "legacy planned OFT normalized action (compatibility-only; not valid for final action-effect claim)"
            if legacy_actions_seen else
            "executed_env_action normalized with frozen OFT q01/q99 statistics"
        ),
        "online_input_contract": ["rgb", "wrist_rgb", "language", "proprio", "previous_executed_action"],
        "excluded_online_fields": ["simulator_state", "object_pose", "contact", "FSM", "candidate"],
        "shapes": {key: list(value.shape) for key, value in features.payload().items()},
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
