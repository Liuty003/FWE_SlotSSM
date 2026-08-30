#!/usr/bin/env python3
"""Precompute frozen OFT and SlotSSM features for dual-camera recovery windows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_unified_oft_slotssm_bridge import (  # noqa: E402
    DEFAULT_OFT_CHECKPOINT,
    DEFAULT_OFT_ROOT,
    DEFAULT_STAGE1,
    DEFAULT_STAGE2,
    center_crop_resize,
    dynamic_module,
    load_torch_state,
    normalized_proprio,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oft-root", type=Path, default=DEFAULT_OFT_ROOT)
    parser.add_argument("--oft-checkpoint", type=Path, default=DEFAULT_OFT_CHECKPOINT)
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--stage2-checkpoint", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--gpu", type=int, default=1)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    for path in (args.manifest, args.oft_root, args.oft_checkpoint, args.stage1_checkpoint, args.stage2_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
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

    # Keep existing Stage-1/2 code while exposing only OFT-only support modules
    # to the checkpoint's remote modeling code.
    prismatic.__path__.append(str(args.oft_root / "prismatic"))
    prismatic.training.__path__.append(str(args.oft_root / "prismatic" / "training"))
    prismatic.vla.__path__.append(str(args.oft_root / "prismatic" / "vla"))
    from scripts.train_slot_stage2_ssm import Stage2Dynamics

    action_components = dynamic_module("oft_action_heads", args.oft_root / "prismatic" / "models" / "action_heads.py")
    projector_components = dynamic_module("oft_projectors", args.oft_root / "prismatic" / "models" / "projectors.py")
    device = torch.device("cuda")
    model = AutoModelForVision2Seq.from_pretrained(
        str(args.oft_checkpoint), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()
    model.vision_backbone.set_num_images_in_input(2)
    model.norm_stats = json.loads((args.oft_checkpoint / "dataset_statistics.json").read_text())
    processor = AutoProcessor.from_pretrained(str(args.oft_checkpoint), trust_remote_code=True)
    action_head = action_components.L1RegressionActionHead(4096, 4096, 7)
    action_head.load_state_dict(load_torch_state(args.oft_checkpoint / "action_head--150000_checkpoint.pt", torch))
    action_head = action_head.to(device=device, dtype=torch.bfloat16).eval()
    proprio_projector = projector_components.ProprioProjector(4096, 8)
    proprio_projector.load_state_dict(load_torch_state(args.oft_checkpoint / "proprio_projector--150000_checkpoint.pt", torch))
    proprio_projector = proprio_projector.to(device=device, dtype=torch.bfloat16).eval()

    stage1_state = load_file(args.stage1_checkpoint)
    tokenizer_state = {key.removeprefix("object_centric_tokenizer."): value for key, value in stage1_state.items() if key.startswith("object_centric_tokenizer.")}
    slot_tokenizer = SlotAttention(n_slots=16, in_dim=2176, feature_dim=512)
    slot_tokenizer.load_state_dict(tokenizer_state, strict=True)
    slot_tokenizer = slot_tokenizer.to(device=device, dtype=torch.bfloat16).eval()
    stage2 = Stage2Dynamics("mamba2", residual_next=True)
    stage2.load_state_dict(load_file(args.stage2_checkpoint), strict=True)
    stage2 = stage2.to(device=device, dtype=torch.bfloat16).eval()

    records = read_jsonl(args.manifest)
    by_trajectory: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_trajectory[record["provenance"]["raw_trajectory"]].append(record)

    def policy_input(observation: dict, task_text: str):
        main = imageio.imread(observation["rgb"])
        wrist = imageio.imread(observation["wrist_rgb"])
        prompt = f"In: What action should the robot take to {task_text.lower()}?\nOut:"
        inputs = processor(prompt, center_crop_resize(main, Image)).to(device, dtype=torch.bfloat16)
        wrist_inputs = processor(prompt, center_crop_resize(wrist, Image)).to(device, dtype=torch.bfloat16)
        inputs["pixel_values"] = torch.cat((inputs["pixel_values"], wrist_inputs["pixel_values"]), dim=1)
        return inputs

    state_by_key: dict[tuple[str, int], torch.Tensor] = {}
    raw_rows: dict[str, list[dict]] = {}
    language_by_trajectory: dict[str, str] = {}
    with torch.inference_mode():
        for trajectory in by_trajectory:
            rows = read_jsonl(Path(trajectory))
            raw_rows[trajectory] = rows
            # Recovery branches retain their baseline rollout provenance,
            # whereas NORMAL collection stores the language in its own episode
            # metadata.  Both are deployment-valid observation sequences.
            if "source_rollout" in rows[0]:
                source_metadata = Path(rows[0]["source_rollout"]) / "metadata.json"
            else:
                source_metadata = Path(trajectory).parent / "metadata.json"
            if not source_metadata.is_file():
                raise FileNotFoundError(f"missing task language metadata: {source_metadata}")
            task_text = json.loads(source_metadata.read_text(encoding="utf-8"))["task_language"]
            language_by_trajectory[trajectory] = task_text
            patches = []
            for row in rows:
                patches.append(model.vision_backbone(policy_input(row["online_observation"], task_text)["pixel_values"]).detach())
            patch_sequence = torch.stack(patches, dim=1)
            slots, previous = [], None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for index in range(len(rows)):
                    current, _, _ = slot_tokenizer(patch_sequence[:, index], past_slots=previous)
                    previous = current[:, None]
                    slots.append(current)
                slot_sequence = torch.stack(slots, dim=1)
                temporal, _, _ = stage2(slot_sequence, patch_sequence, compute_window=False)
            for index in range(len(rows)):
                state_by_key[(trajectory, index)] = temporal[0, index].cpu().contiguous()

        hidden_rows, state_rows, action_rows, gate_rows, positive_rows = [], [], [], [], []
        for record in records:
            trajectory = record["provenance"]["raw_trajectory"]
            row = raw_rows[trajectory][record["branch_timestep"]]
            task_text = language_by_trajectory[trajectory]
            inputs = policy_input(row["online_observation"], task_text)
            norm_key = record["task_suite"]
            if norm_key not in model.norm_stats:
                norm_key = f"{norm_key}_no_noops"
            _, hidden = model.predict_action(
                **inputs,
                unnorm_key=norm_key,
                do_sample=False,
                proprio=normalized_proprio(row["online_observation"]["proprio"], model.norm_stats[norm_key]["proprio"], np),
                proprio_projector=proprio_projector,
                action_head=action_head,
                use_film=False,
            )
            hidden_rows.append(hidden[0].cpu().contiguous())
            state_rows.append(state_by_key[(trajectory, record["branch_timestep"])])
            action_rows.append(torch.tensor(record["recovery_target_oft_action_chunk"], dtype=torch.float32))
            gate_rows.append(record["gate_label"])
            positive_rows.append(record["branch_eventually_regrasped"])

    payload = {
        "action_hidden": torch.stack(hidden_rows),
        "temporal_slots": torch.stack(state_rows),
        "recovery_target": torch.stack(action_rows),
        "gate_label": torch.tensor(gate_rows, dtype=torch.int64),
        "terminal_regrasp": torch.tensor(positive_rows, dtype=torch.bool),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(payload, str(args.output))
    metadata = {
        "kind": "frozen_oft_slotssm_recovery_features",
        "manifest": str(args.manifest),
        "samples": len(records),
        "shapes": {key: list(value.shape) for key, value in payload.items()},
        "positive_terminal_regrasp": int(sum(positive_rows)),
        "note": "recovery-only seed; no NORMAL gate examples",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
