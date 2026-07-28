#!/usr/bin/env python
"""Precompute vision-aligned SigLIP features for Stage-B language phrases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoProcessor, SiglipModel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.precompute_stage_b_language_features import REFERENCE_PHRASES, RELATION_PHRASES


def counterfactual_instruction(relation: str, references: list[str]) -> str:
    names = [REFERENCE_PHRASES[name] for name in references]
    if relation == "between":
        location = f"between the {names[0]} and the {names[1]}"
    elif relation == "next_to":
        location = f"next to the {names[0]}"
    elif relation == "on":
        location = f"on the {names[0]}"
    elif relation == "in":
        location = f"in the {names[0]}"
    elif relation == "table_center":
        location = "at the center of the table"
    else:
        raise ValueError(f"Unknown relation: {relation}")
    return f"pick up the black bowl {location} and place it on the plate"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("/data2/lty/models/siglip-so400m-patch14-384"),
    )
    parser.add_argument(
        "--role-schema", type=Path,
        default=Path("experiments/robot/libero/stage_b_spatial_role_schema.json"),
    )
    parser.add_argument(
        "--data-manifest", type=Path,
        default=Path(
            "/data2/lty/naive-e-slotssm-runs/"
            "stage-b-spatial-relation-data/manifest.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "/data2/lty/naive-e-slotssm-runs/stage-b-spatial-relation-patches/"
            "siglip_language_features.safetensors"
        ),
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = arguments()
    schema = json.loads(args.role_schema.read_text(encoding="utf-8"))
    data = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    instructions = {}
    for sample in data["samples"]:
        task_id = int(sample["task_id"])
        previous = instructions.setdefault(task_id, sample["language"])
        if previous != sample["language"]:
            raise RuntimeError(f"task {task_id} has inconsistent instructions")

    records, unique_texts = [], []
    for task_id in sorted(instructions):
        spec = schema["tasks"][str(task_id)]
        reference_phrases = [REFERENCE_PHRASES[name] for name in spec["references"]]
        record = {
            "task_id": task_id,
            "instruction": instructions[task_id],
            "relation_phrase": RELATION_PHRASES[spec["relation"]],
            "reference_phrases": reference_phrases,
            "counterfactual_relation_phrase": RELATION_PHRASES[spec["counterfactual"]["relation"]],
            "counterfactual_reference_phrases": [
                REFERENCE_PHRASES[name] for name in spec["counterfactual"]["references"]
            ],
            "counterfactual_instruction": counterfactual_instruction(
                spec["counterfactual"]["relation"],
                spec["counterfactual"]["references"],
            ),
        }
        records.append(record)
        unique_texts.extend((record["instruction"], record["relation_phrase"], *reference_phrases))
        unique_texts.extend((
            record["counterfactual_instruction"],
            record["counterfactual_relation_phrase"],
            *record["counterfactual_reference_phrases"],
        ))
    unique_texts = list(dict.fromkeys(unique_texts))

    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = SiglipModel.from_pretrained(
        args.model_dir, local_files_only=True, low_cpu_mem_usage=True
    ).eval()
    encoded = processor(
        text=unique_texts, padding="max_length", truncation=True,
        max_length=64, return_tensors="pt",
    )
    features = model.get_text_features(**encoded).float()
    features = torch.nn.functional.normalize(features, dim=-1)
    by_text = {text: features[index] for index, text in enumerate(unique_texts)}

    task_ids, instruction_features = [], []
    relation_features, reference_features, reference_presence = [], [], []
    counterfactual_relation_features = []
    counterfactual_instruction_features = []
    counterfactual_reference_features, counterfactual_reference_presence = [], []
    for record in records:
        padded = record["reference_phrases"] + [""] * (2 - len(record["reference_phrases"]))
        task_ids.append(record["task_id"])
        instruction_features.append(by_text[record["instruction"]])
        relation_features.append(by_text[record["relation_phrase"]])
        reference_features.append(torch.stack([
            by_text[text] if text else torch.zeros(features.shape[-1]) for text in padded
        ]))
        reference_presence.append([bool(text) for text in padded])
        counterfactual_padded = record["counterfactual_reference_phrases"] + [""] * (
            2 - len(record["counterfactual_reference_phrases"])
        )
        counterfactual_relation_features.append(
            by_text[record["counterfactual_relation_phrase"]]
        )
        counterfactual_instruction_features.append(
            by_text[record["counterfactual_instruction"]]
        )
        counterfactual_reference_features.append(torch.stack([
            by_text[text] if text else torch.zeros(features.shape[-1])
            for text in counterfactual_padded
        ]))
        counterfactual_reference_presence.append(
            [bool(text) for text in counterfactual_padded]
        )
    tensors = {
        "task_ids": torch.tensor(task_ids, dtype=torch.int64),
        "instruction_features": torch.stack(instruction_features),
        "relation_features": torch.stack(relation_features),
        "reference_features": torch.stack(reference_features),
        "reference_presence": torch.tensor(reference_presence, dtype=torch.bool),
        "counterfactual_relation_features": torch.stack(counterfactual_relation_features),
        "counterfactual_instruction_features": torch.stack(
            counterfactual_instruction_features
        ),
        "counterfactual_reference_features": torch.stack(counterfactual_reference_features),
        "counterfactual_reference_presence": torch.tensor(
            counterfactual_reference_presence, dtype=torch.bool
        ),
    }
    if not all(torch.isfinite(tensor).all() for tensor in tensors.values()):
        raise RuntimeError("Non-finite SigLIP language feature")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata={
        "model": str(args.model_dir), "pooling": "SiglipModel.get_text_features + L2 norm",
        "texts": json.dumps(records, ensure_ascii=False),
    })
    print(json.dumps({
        "output": str(args.output), "num_unique_texts": len(unique_texts),
        "feature_shape": list(tensors["instruction_features"].shape),
        "reference_shape": list(tensors["reference_features"].shape),
    }, indent=2))


if __name__ == "__main__":
    main()
