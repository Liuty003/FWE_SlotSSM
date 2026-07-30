#!/usr/bin/env python
"""Precompute continuous Stage-1 slots for the task-6 visual dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import (
    OpenVLAForActionPrediction,
    OpenVLAForActionPrediction_SlotAtt,
)
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--slot-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--vision-batch-size", type=int, default=32)
    parser.add_argument("--metadata-only", action="store_true")
    return parser.parse_args()


def register_openvla() -> None:
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def load_rows(path: Path, episode_id: int) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = [row for row in rows if int(row["episode_id"]) == episode_id]
    result.sort(key=lambda row: int(row["timestep"]))
    expected = list(range(len(result)))
    actual = [int(row["timestep"]) for row in result]
    if actual != expected:
        raise ValueError(f"episode {episode_id} has non-contiguous timesteps")
    return result


def preprocess(paths: list[str], transform) -> torch.Tensor:
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            tensors.append(transform(image.convert("RGB")))
    return torch.stack(tensors)


@torch.inference_mode()
def encode_episode(
    rows: list[dict], transform, vision_backbone, tokenizer, device, batch_size
) -> tuple[torch.Tensor, torch.Tensor]:
    patches = []
    for start in range(0, len(rows), batch_size):
        pixels = preprocess(
            [row["image"] for row in rows[start : start + batch_size]], transform
        ).to(device, dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = vision_backbone(pixels).reshape(len(pixels), -1, 2176)
        patches.append(encoded)
    patches = torch.cat(patches)
    slots = []
    past = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for patch in patches:
            current, _, _ = tokenizer(patch[None], past_slots=past)
            slots.append(current[0].float().cpu())
            past = current[:, None]
    return torch.stack(slots), patches.float().cpu()


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.data_root / "manifest.jsonl"
    episode_ids = [
        int(value) for value in args.episode_ids.split(",") if value
    ]
    if args.metadata_only:
        outputs = []
        for episode_id in episode_ids:
            rows = load_rows(manifest_path, episode_id)
            output = args.output_dir / f"task6-episode{episode_id}-slots.safetensors"
            tensors = load_file(output)
            if len(tensors["slots"]) != len(rows):
                raise RuntimeError(f"stale output: {output}")
            outputs.append(
                {"episode_id": episode_id, "frames": len(rows), "file": str(output)}
            )
        metadata = {
            "schema_version": 1,
            "model": str(args.model),
            "slot_checkpoint": str(args.slot_checkpoint),
            "slot_contract": "continuous causal slots; past_slots reset per episode",
            "privileged_state_in_input": False,
            "episodes": outputs,
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"validated episodes={len(outputs)}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Slot extraction requires CUDA")
    register_openvla()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    base = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    base.vision_backbone.requires_grad_(False)
    del base.projector
    del base.language_model
    slot_model = OpenVLAForActionPrediction_SlotAtt(model=base, number_of_slots=16)
    load_result = slot_model.load_state_dict(
        load_file(args.slot_checkpoint), strict=False
    )
    device = torch.device("cuda:0")
    slot_model = slot_model.to(device, dtype=torch.bfloat16).eval()
    slot_model.requires_grad_(False)
    print(
        f"slot_checkpoint_missing={len(load_result.missing_keys)} "
        f"unexpected={len(load_result.unexpected_keys)}",
        flush=True,
    )

    outputs = []
    for episode_id in episode_ids:
        rows = load_rows(manifest_path, episode_id)
        output = args.output_dir / f"task6-episode{episode_id}-slots.safetensors"
        if output.exists():
            tensors = load_file(output)
            if len(tensors["slots"]) != len(rows):
                raise RuntimeError(f"stale output: {output}")
            print(f"skip episode={episode_id} frames={len(rows)}", flush=True)
        else:
            slots, patches = encode_episode(
                rows,
                processor.image_processor.apply_transform,
                slot_model.model.vision_backbone,
                slot_model.object_centric_tokenizer,
                device,
                args.vision_batch_size,
            )
            save_file(
                {
                    "slots": slots.contiguous(),
                    "patch_mean": patches.mean(dim=1).contiguous(),
                    "labels": torch.tensor(
                        [row["mug_on_plate"] for row in rows], dtype=torch.float32
                    ),
                    "timesteps": torch.tensor(
                        [row["timestep"] for row in rows], dtype=torch.int64
                    ),
                },
                output,
            )
            print(
                f"episode={episode_id} frames={len(rows)} "
                f"slots={tuple(slots.shape)}",
                flush=True,
            )
        outputs.append(
            {"episode_id": episode_id, "frames": len(rows), "file": str(output)}
        )
    metadata = {
        "schema_version": 1,
        "model": str(args.model),
        "slot_checkpoint": str(args.slot_checkpoint),
        "slot_contract": "continuous causal slots; past_slots reset per episode",
        "privileged_state_in_input": False,
        "episodes": outputs,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
