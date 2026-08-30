#!/usr/bin/env python3
"""Precompute causal frozen OFT+/SlotSSM features from a v2 book manifest.

This is intentionally separate from the legacy recovery-seed feature script:
v2 manifests carry source-disjoint split assignments and store only deployment
valid observations.  Pose/contact metadata is used only to locate an offline
branch file and is never copied into the feature payload.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def bind_requested_cuda_before_imports() -> None:
    """Honor ``--gpu`` before bridge imports can initialize CUDA.

    The OFT bridge imports modules which may indirectly import torch.  Setting
    CUDA_VISIBLE_DEVICES later then leaves their cached ``cuda:0`` bound to the
    launcher's physical GPU 0, defeating sharded extraction.
    """
    # A launcher may already have selected one physical device (for example
    # ``CUDA_VISIBLE_DEVICES=2 python ... --gpu 0``).  Preserve that mapping:
    # inside the process the selected device is then cuda:0.  Overwriting it
    # here silently sends extraction back to the busy physical GPU 0.
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    for index, value in enumerate(sys.argv[:-1]):
        if value == "--gpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[index + 1]
            return


bind_requested_cuda_before_imports()

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_unified_oft_slotssm_bridge import (  # noqa: E402
    DEFAULT_OFT_ROOT,
    DEFAULT_STAGE1,
    DEFAULT_STAGE2,
    center_crop_resize,
    dynamic_module,
    load_torch_state,
    normalized_proprio,
)

DEFAULT_OFT_PLUS = Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata")
GATE_LABELS = {"NORMAL": 0, "RECOVERY": 1, "RECOVERY_COMPLETE": 2}
RECOVERY_MODE_LABELS = {
    "fsm_xyp0_p0": 0,
    "fsm_xym25_p0": 1,
    "fsm_xyp0_m25": 2,
    "fsm_xyp0_p25": 3,
    "fsm_xyp25_p0": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True,
                        help="v2 action_windows.jsonl, branch_outcomes.jsonl, or gate_windows.jsonl")
    parser.add_argument("--source-splits", type=Path,
                        help="required for gate_windows.jsonl; v2 source_splits.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oft-root", type=Path, default=DEFAULT_OFT_ROOT)
    parser.add_argument("--oft-checkpoint", type=Path, default=DEFAULT_OFT_PLUS)
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--stage2-checkpoint", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--split", choices=("train", "val", "test"),
                        help="select one source-disjoint split before optional --max-samples")
    parser.add_argument("--max-samples", type=int,
                        help="optional deterministic prefix for an integration smoke test or sharded run")
    parser.add_argument("--source-key", help="optional exact source key filter for contextual feature smoke tests")
    parser.add_argument(
        "--require-replayable-source-context", action="store_true",
        help="For action windows, exclude branch states whose preceding visual/action history was not recorded.",
    )
    parser.add_argument("--include-current-patches", action="store_true",
                        help="Persist frozen current dual-camera vision patch tokens for spatial geometry training.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def causal_robot_input(observation: dict[str, Any]) -> tuple[dict[str, Any], list[float]]:
    """Validate and retain precisely the v2 deployment-time observation fields."""
    for key in ("rgb", "wrist_rgb", "proprio"):
        if key not in observation:
            raise ValueError(f"missing online field {key}")
    for key in ("rgb", "wrist_rgb"):
        if not Path(observation[key]).is_file():
            raise FileNotFoundError(observation[key])
    previous = observation.get("previous_executed_action")
    if previous is None:
        previous = [0.0] * 7
    if len(previous) != 7:
        raise ValueError("previous_executed_action must have seven dimensions")
    return observation, [float(value) for value in previous]


def record_online_observation(record: dict[str, Any]) -> dict[str, Any]:
    """Select the causal observation field used by a manifest record."""
    if record.get("kind") == "book_regrasp_branch_outcome":
        return record["initial_online_observation"]
    return record["online_observation"]


def has_rendered_prefix(rows: list[dict[str, Any]], length: int) -> bool:
    """Whether every causal source observation has both recorded cameras."""
    return all(
        isinstance(row["online_observation"].get(camera), str)
        and Path(row["online_observation"][camera]).is_file()
        for row in rows[:length]
        for camera in ("rgb", "wrist_rgb")
    )


def nominal_transition_signature(row: dict[str, Any]) -> tuple[Any, Any]:
    """Image-free state/action identity check for a replacement source replay."""
    return row["online_observation"]["proprio"], row.get("executed_action")


def context_for_record(
    record: dict[str, Any], source_metadata: dict[str, Path],
) -> tuple[Path, int, str]:
    """Return causal rollout rows path, selected timestep and task language."""
    kind = record.get("kind")
    if kind in {"book_regrasp_action_window", "book_regrasp_branch_outcome", "book_grasp_quality_window"}:
        metadata_path = Path(record["provenance"]["branch_metadata"])
        timestep = int(record.get("branch_timestep", 0))
        return metadata_path.parent / "steps.jsonl", timestep, record["language"]
    if kind == "book_execution_gate_window":
        try:
            metadata_path = source_metadata[record["source_key"]]
        except KeyError as error:
            raise KeyError(f"no source metadata for {record['source_key']}") from error
        return metadata_path.parent / "steps.jsonl", int(record["timestep"]), record["language"]
    raise ValueError(f"unsupported v2 manifest kind: {kind}")


def main() -> None:
    args = parse_args()
    for path in (args.manifest, args.oft_root, args.oft_checkpoint, args.stage1_checkpoint, args.stage2_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    records = read_jsonl(args.manifest)
    if args.split is not None:
        records = [record for record in records if record.get("split") == args.split]
    if args.source_key is not None:
        records = [record for record in records if record.get("source_key") == args.source_key]
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        records = records[:args.max_samples]
    if not records:
        raise RuntimeError("empty manifest")
    kinds = {record.get("kind") for record in records}
    if len(kinds) != 1:
        raise ValueError(f"a feature file must contain one record kind, got {kinds}")
    kind = next(iter(kinds))
    if args.require_replayable_source_context:
        if kind not in {"book_regrasp_action_window", "book_regrasp_branch_outcome"}:
            raise ValueError("replayable source context is only defined for recovery branch records")
        replayable: dict[Path, bool] = {}

        def has_replayable_source_context(record: dict[str, Any]) -> bool:
            metadata_path = Path(record["provenance"]["branch_metadata"])
            if metadata_path not in replayable:
                fork = read_json(metadata_path)["source_fork"]
                replayable[metadata_path] = "source_steps" in read_json(Path(fork) / "manifest.json")
            return replayable[metadata_path]

        records = [record for record in records if has_replayable_source_context(record)]
        if not records:
            raise RuntimeError("no replayable recovery action windows after filtering")
    # A normal gate window from a saved online rollout is already a complete
    # causal source: it carries its RGB/proprio observations and the replay
    # supplies the actually executed action at t-1.  Legacy gate manifests
    # instead point at a source-split metadata file.  Require the latter only
    # when no direct online context was provided.
    direct_online_gate_context = (
        kind == "book_execution_gate_window"
        and all(record.get("provenance", {}).get("online_rollout") is not None for record in records)
    )
    if kind == "book_execution_gate_window" and args.source_splits is None and not direct_online_gate_context:
        raise ValueError("--source-splits is required for gate windows without online_rollout provenance")

    source_metadata: dict[str, Path] = {}
    if args.source_splits is not None:
        for row in read_json(args.source_splits):
            source_metadata[row["source_key"]] = Path(row["metadata"])

    # A context is a full causal source/branch sequence.  For a recovery
    # branch, its first rendered observation comes from a fork of a nominal
    # rollout.  Replaying only branch frames would reset SlotSSM history and
    # erase the executed action at t-1, unlike the online controller.  Prefix
    # every branch with its nominal source up to (but excluding) the fork
    # observation, then append the branch rows.
    contexts: dict[Path, dict[str, Any]] = {}
    record_context: list[tuple[Path, int]] = []
    for record in records:
        language = record["language"]
        if kind == "book_execution_gate_window" and record.get("provenance", {}).get("online_rollout") is not None:
            # This mirrors the action-window direct-rollout path below.  It
            # is deliberately limited to saved online inputs, so an
            # overfit/preservation gate still has exactly its deployment
            # contract and never reads simulator state.
            context_key = Path(record["provenance"]["online_rollout"])
            if context_key not in contexts:
                # A saved episode can continue for hundreds of frames after
                # the gate diversion.  Gate supervision at an early prefix
                # only needs causal history through the largest requested
                # timestep, not those later (and potentially off-policy)
                # observations.
                maximum_index = max(
                    int(candidate["timestep"])
                    for candidate in records
                    if candidate.get("provenance", {}).get("online_rollout") == str(context_key)
                )
                rollout_rows = read_jsonl(context_key)
                if maximum_index >= len(rollout_rows):
                    raise IndexError(f"invalid online-rollout timestep {maximum_index} for {context_key}")
                contexts[context_key] = {
                    "rows": rollout_rows[:maximum_index + 1], "language": language,
                    "source_prefix_rows": 0,
                    "context_mode": "pure_neural_online_rollout",
                }
            index = int(record["timestep"])
            rows = contexts[context_key]["rows"]
            if not 0 <= index < len(rows):
                raise IndexError(f"invalid online-rollout timestep {index} for {context_key}")
            if contexts[context_key]["language"] != language:
                raise ValueError(f"language mismatch within {context_key}")
            record_context.append((context_key, index))
            continue
        if kind in {"book_regrasp_action_window", "book_regrasp_branch_outcome"}:
            online_rollout = record["provenance"].get("online_rollout")
            if online_rollout is not None:
                # Successful pure-neural rollouts can be distilled directly:
                # their saved rows are the exact causal visual/action history
                # that produced the observed completion, with no simulator
                # target pose or teacher program involved.
                context_key = Path(online_rollout)
                if context_key not in contexts:
                    contexts[context_key] = {
                        "rows": read_jsonl(context_key), "language": language,
                        "source_prefix_rows": 0,
                        "context_mode": "pure_neural_online_rollout",
                    }
                index = int(record.get("branch_timestep", 0))
                rows = contexts[context_key]["rows"]
                if not 0 <= index < len(rows):
                    raise IndexError(f"invalid online-rollout timestep {index} for {context_key}")
                if contexts[context_key]["language"] != language:
                    raise ValueError(f"language mismatch within {context_key}")
                record_context.append((context_key, index))
                continue
            metadata_path = Path(record["provenance"]["branch_metadata"])
            steps_path = metadata_path.parent / "steps.jsonl"
            completion_prefix = record["provenance"].get("prelift_branch_metadata")
            if completion_prefix is not None:
                # A placement trajectory starts from the post-lift simulator
                # snapshot.  Replay the exact verified recovery branch before
                # it so SlotSSM and previous-action state match deployment at
                # the transport transition; the placement rows themselves
                # remain the only supervision targets.
                prelift_steps = Path(completion_prefix).parent / "steps.jsonl"
                if not prelift_steps.is_file():
                    raise FileNotFoundError(f"missing completion pre-lift rows: {prelift_steps}")
                context_key = steps_path
                if context_key not in contexts:
                    contexts[context_key] = {
                        "rows": read_jsonl(prelift_steps) + read_jsonl(steps_path),
                        "language": language,
                        "source_prefix_rows": len(read_jsonl(prelift_steps)),
                        "context_mode": "verified_lift_then_completion",
                    }
                index = contexts[context_key]["source_prefix_rows"] + int(record.get("branch_timestep", 0))
                rows = contexts[context_key]["rows"]
                if not 0 <= index < len(rows):
                    raise IndexError(f"invalid completion timestep {index} for {steps_path}")
                if contexts[context_key]["language"] != language:
                    raise ValueError(f"language mismatch within {steps_path}")
                record_context.append((context_key, index))
                continue
            fork = read_json(metadata_path.parent / "metadata.json")["source_fork"]
            # The five outcome candidates share exactly one pre-action fork
            # state.  Reusing it avoids encoding five copies of the same
            # nominal prefix and, more importantly, makes explicit that a
            # candidate value cannot see any branch-local post-action frame.
            context_key = Path(fork) if kind == "book_regrasp_branch_outcome" else steps_path
            if context_key not in contexts:
                branch_rows = read_jsonl(steps_path)
                fork_manifest = read_json(Path(fork) / "manifest.json")
                if "source_steps" in fork_manifest:
                    fork_timestep = int(fork_manifest["fork_timestep"])
                    source_path = Path(fork_manifest["source_steps"])
                    source_rows = read_jsonl(source_path)
                    if not 0 <= fork_timestep < len(source_rows):
                        raise IndexError(f"invalid source fork timestep {fork_timestep} for {steps_path}")
                    if not has_rendered_prefix(source_rows, fork_timestep):
                        run_root = Path(fork).parents[2]
                        alternative_path = run_root / "natural-v2" / (
                            f"task{fork_manifest['task_id']}-episode{fork_manifest['episode_id']}"
                        ) / "steps.jsonl"
                        if not alternative_path.is_file():
                            raise FileNotFoundError(
                                f"source {source_path} has no rendered prefix and no natural-v2 replacement"
                            )
                        alternative_rows = read_jsonl(alternative_path)
                        if len(alternative_rows) <= fork_timestep or not has_rendered_prefix(alternative_rows, fork_timestep):
                            raise ValueError(f"natural-v2 replacement lacks rendered prefix for {steps_path}")
                        if [nominal_transition_signature(row) for row in source_rows[:fork_timestep]] != [
                            nominal_transition_signature(row) for row in alternative_rows[:fork_timestep]
                        ]:
                            raise ValueError(f"natural-v2 state/action prefix does not match source fork {fork}")
                        source_rows = alternative_rows
                    contexts[context_key] = {
                        "rows": source_rows[:fork_timestep] + (
                            branch_rows[:1] if kind == "book_regrasp_branch_outcome" else branch_rows
                        ),
                        "language": language,
                        "source_prefix_rows": fork_timestep,
                        "context_mode": "nominal_prefix_then_branch",
                    }
                else:
                    # Legacy correction states lack their prior observations;
                    # retain old behavior only when the caller has not opted
                    # into replayable context exclusively.
                    contexts[context_key] = {
                        "rows": branch_rows[:1] if kind == "book_regrasp_branch_outcome" else branch_rows,
                        "language": language,
                        "source_prefix_rows": 0, "context_mode": "branch_only_unreplayable",
                    }
            # Branch-outcome records always score the shared pre-action fork
            # state, while action windows carry an explicit branch index.
            index = contexts[context_key]["source_prefix_rows"] + int(record.get("branch_timestep", 0))
            steps_path = context_key
        else:
            steps_path, index, _ = context_for_record(record, source_metadata)
            contexts.setdefault(steps_path, {"rows": read_jsonl(steps_path), "language": language})
        rows = contexts[steps_path]["rows"]
        if not 0 <= index < len(rows):
            raise IndexError(f"invalid timestep {index} for {steps_path}")
        if contexts[steps_path]["language"] != language:
            raise ValueError(f"language mismatch within {steps_path}")
        record_context.append((steps_path, index))

    # See bind_requested_cuda_before_imports(): do not override a launcher's
    # explicit physical-GPU selection after imports have begun.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.modeling_prismatic import SlotAttention
    import prismatic
    import prismatic.training
    import prismatic.vla

    # Stage-1/2 and OFT use separately versioned prismatic namespaces.
    prismatic.__path__.append(str(args.oft_root / "prismatic"))
    prismatic.training.__path__.append(str(args.oft_root / "prismatic" / "training"))
    prismatic.vla.__path__.append(str(args.oft_root / "prismatic" / "vla"))
    from scripts.train_slot_stage2_ssm import Stage2Dynamics

    action_components = dynamic_module("oft_action_heads_v2", args.oft_root / "prismatic" / "models" / "action_heads.py")
    projector_components = dynamic_module("oft_projectors_v2", args.oft_root / "prismatic" / "models" / "projectors.py")
    device = torch.device("cuda")
    model = AutoModelForVision2Seq.from_pretrained(
        str(args.oft_checkpoint), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device).eval()
    model.vision_backbone.set_num_images_in_input(2)
    model.norm_stats = read_json(args.oft_checkpoint / "dataset_statistics.json")
    processor = AutoProcessor.from_pretrained(str(args.oft_checkpoint), trust_remote_code=True)
    action_head = action_components.L1RegressionActionHead(4096, 4096, 7)
    action_head.load_state_dict(load_torch_state(args.oft_checkpoint / "action_head--150000_checkpoint.pt", torch))
    action_head = action_head.to(device=device, dtype=torch.bfloat16).eval()
    proprio_projector = projector_components.ProprioProjector(4096, 8)
    proprio_projector.load_state_dict(load_torch_state(args.oft_checkpoint / "proprio_projector--150000_checkpoint.pt", torch))
    proprio_projector = proprio_projector.to(device=device, dtype=torch.bfloat16).eval()

    stage1 = load_file(args.stage1_checkpoint)
    tokenizer_state = {key.removeprefix("object_centric_tokenizer."): value for key, value in stage1.items()
                       if key.startswith("object_centric_tokenizer.")}
    slot_tokenizer = SlotAttention(n_slots=16, in_dim=2176, feature_dim=512)
    slot_tokenizer.load_state_dict(tokenizer_state, strict=True)
    slot_tokenizer = slot_tokenizer.to(device=device, dtype=torch.bfloat16).eval()
    stage2 = Stage2Dynamics("mamba2", residual_next=True)
    stage2.load_state_dict(load_file(args.stage2_checkpoint), strict=True)
    stage2 = stage2.to(device=device, dtype=torch.bfloat16).eval()

    def policy_input(observation: dict[str, Any], language: str):
        main, wrist = imageio.imread(observation["rgb"]), imageio.imread(observation["wrist_rgb"])
        prompt = f"In: What action should the robot take to {language.lower()}?\nOut:"
        primary = processor(prompt, center_crop_resize(main, Image)).to(device, dtype=torch.bfloat16)
        wrist_inputs = processor(prompt, center_crop_resize(wrist, Image)).to(device, dtype=torch.bfloat16)
        primary["pixel_values"] = torch.cat((primary["pixel_values"], wrist_inputs["pixel_values"]), dim=1)
        return primary

    temporal_state: dict[tuple[Path, int], torch.Tensor] = {}
    requested_patch_indices: dict[Path, set[int]] = defaultdict(set)
    if args.include_current_patches:
        for steps_path, index in record_context:
            requested_patch_indices[steps_path].add(index)
    current_patch_state: dict[tuple[Path, int], torch.Tensor] = {}
    with torch.inference_mode():
        for steps_path, context in contexts.items():
            rows, language = context["rows"], context["language"]
            patches = []
            for row in rows:
                observation, _ = causal_robot_input(row["online_observation"])
                patches.append(model.vision_backbone(policy_input(observation, language)["pixel_values"]).detach())
            patch_sequence = torch.stack(patches, dim=1)
            slots, previous = [], None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for index in range(len(rows)):
                    current, _, _ = slot_tokenizer(patch_sequence[:, index], past_slots=previous)
                    previous = current[:, None]
                    slots.append(current)
                slot_sequence = torch.stack(slots, dim=1)
                states, _, _ = stage2(slot_sequence, patch_sequence, compute_window=False)
            for index in range(len(rows)):
                temporal_state[(steps_path, index)] = states[0, index].cpu().contiguous()
                if index in requested_patch_indices.get(steps_path, set()):
                    current_patch_state[(steps_path, index)] = patch_sequence[0, index].cpu().contiguous()

        hidden_rows, slot_rows, patch_rows, proprio_rows, previous_rows, labels, targets, phases, modes, geometry_targets = [], [], [], [], [], [], [], [], [], []
        for record, (steps_path, index) in zip(records, record_context):
            observation, fallback_previous_action = causal_robot_input(record_online_observation(record))
            # This is the actual action which produced the current observation
            # in the causal replay.  At a branch's first frame it comes from
            # the nominal rollout before the fork; at later frames it comes
            # from the recovery branch itself.
            previous_action = fallback_previous_action
            if index > 0:
                executed = contexts[steps_path]["rows"][index - 1].get("executed_action")
                if executed is not None:
                    if len(executed) != 7:
                        raise ValueError(f"invalid executed action before {steps_path}:{index}")
                    previous_action = [float(value) for value in executed]
            inputs = policy_input(observation, contexts[steps_path]["language"])
            norm_key = "libero_10" if "libero_10" in model.norm_stats else "libero_10_no_noops"
            if norm_key not in model.norm_stats:
                raise KeyError("OFT+ checkpoint lacks libero_10 normalization statistics")
            _, hidden = model.predict_action(
                **inputs, unnorm_key=norm_key, do_sample=False,
                proprio=normalized_proprio(observation["proprio"], model.norm_stats[norm_key]["proprio"], np),
                proprio_projector=proprio_projector, action_head=action_head, use_film=False,
            )
            hidden_rows.append(hidden[0].float().cpu().contiguous())
            slot_rows.append(temporal_state[(steps_path, index)].float().cpu().contiguous())
            if args.include_current_patches:
                patch_rows.append(current_patch_state[(steps_path, index)])
            proprio_rows.append(torch.tensor(normalized_proprio(observation["proprio"], model.norm_stats[norm_key]["proprio"], np)))
            previous_rows.append(torch.tensor(previous_action, dtype=torch.float32))
            labels.append(GATE_LABELS[record["gate_label"]] if kind == "book_execution_gate_window" else 1)
            if kind == "book_regrasp_action_window":
                targets.append(torch.tensor(record["recovery_target_action_chunk"], dtype=torch.float32))
                phase = int(record.get("recovery_phase", 0))
                if not 0 <= phase < 9:
                    raise ValueError(f"invalid recovery phase {phase}")
                # Legacy adapters have five phase embeddings.  Full-macro
                # teacher trajectories add transport/place/open/retreat;
                # when phase conditioning is disabled they need no runtime
                # phase at all, while phase-conditioned legacy consumers see
                # them as the terminal LIFT_VERIFY bucket.
                phase = min(phase, 4)
                phases.append(phase)
                try:
                    modes.append(RECOVERY_MODE_LABELS[record["candidate_name"]])
                except KeyError as error:
                    raise ValueError(f"unknown distilled recovery mode: {record.get('candidate_name')}") from error
                geometry = record.get("teacher_target_relative_position")
                if geometry is not None:
                    if len(geometry) != 3:
                        raise ValueError("teacher_target_relative_position must have three dimensions")
                    geometry_targets.append(torch.tensor(geometry, dtype=torch.float32))
                elif geometry_targets:
                    raise ValueError("mixed geometry-supervised and geometry-missing action windows")

    payload: dict[str, Any] = {
        "action_hidden": torch.stack(hidden_rows), "temporal_slots": torch.stack(slot_rows),
        "proprio": torch.stack(proprio_rows).float(), "previous_executed_action": torch.stack(previous_rows),
        "label": torch.tensor(labels, dtype=torch.int64),
    }
    if targets:
        payload["recovery_target"] = torch.stack(targets)
        payload["recovery_phase"] = torch.tensor(phases, dtype=torch.int64)
        payload["recovery_mode"] = torch.tensor(modes, dtype=torch.int64)
        if geometry_targets:
            if len(geometry_targets) != len(targets):
                raise ValueError("mixed geometry-supervised and geometry-missing action windows")
            payload["teacher_target_relative_position"] = torch.stack(geometry_targets)
    if patch_rows:
        payload["current_patches"] = torch.stack(patch_rows)
    if kind == "book_regrasp_branch_outcome":
        payload["recovery_verified_label"] = torch.tensor(
            [int(bool(record["recovery_verified_label"])) for record in records], dtype=torch.int64
        )
    if kind == "book_grasp_quality_window":
        payload["grasp_attached_label"] = torch.tensor(
            [int(bool(record["grasp_attached_label"])) for record in records], dtype=torch.int64
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(payload, str(args.output))
    metadata = {
        "kind": "frozen_oft_plus_slotssm_v2_features", "input_manifest": str(args.manifest),
        "record_kind": kind, "samples": len(records), "checkpoint": str(args.oft_checkpoint),
        "online_input_contract": ["rgb", "wrist_rgb", "language", "proprio", "previous_executed_action"],
        "action_context_contract": "nominal prefix through fork then branch; prior executed action from replay",
        "require_replayable_source_context": args.require_replayable_source_context,
        "shapes": {key: list(value.shape) for key, value in payload.items()},
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
