#!/usr/bin/env python
"""Extract frozen OpenVLA token-embedding features without loading the 7B model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer


EMBEDDING_KEY = "language_model.model.embed_tokens.weight"
REFERENCE_PHRASES = {
    "plate_1": "plate",
    "glazed_rim_porcelain_ramekin_1": "glazed rim porcelain ramekin",
    "cookies_1": "cookie box",
    "wooden_cabinet_1": "wooden cabinet",
    "flat_stove_1": "flat stove",
}
RELATION_PHRASES = {
    "between": "between two reference objects",
    "in": "inside the reference object",
    "next_to": "next to the reference object",
    "on": "on top of the reference object",
    "table_center": "at the center of the table",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("artifacts/checkpoints/openvla-7b-finetuned-libero-spatial"),
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
            "/data2/lty/naive-e-slotssm-runs/"
            "stage-b-spatial-relation-patches/language_features.safetensors"
        ),
    )
    return parser.parse_args()


def embedding_shard(model_dir: Path) -> Path:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    return model_dir / index["weight_map"][EMBEDDING_KEY]


def mean_embedding(text: str, tokenizer, embedding: torch.Tensor) -> torch.Tensor:
    token_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if not len(token_ids):
        raise RuntimeError(f"Text produced no tokens: {text!r}")
    return embedding[token_ids].float().mean(0)


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
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    shard = embedding_shard(args.model_dir)
    with safe_open(shard, framework="pt", device="cpu") as handle:
        embedding = handle.get_tensor(EMBEDDING_KEY)
    if embedding.ndim != 2 or embedding.shape[1] != 4096:
        raise RuntimeError(f"Unexpected embedding matrix: {tuple(embedding.shape)}")

    task_ids, instruction_features = [], []
    relation_features, reference_features, reference_presence = [], [], []
    text_records = []
    for task_id in sorted(instructions):
        spec = schema["tasks"][str(task_id)]
        references = spec["references"]
        unknown = set(references) - set(REFERENCE_PHRASES)
        if unknown:
            raise RuntimeError(f"No phrase mapping for {sorted(unknown)}")
        reference_phrases = [REFERENCE_PHRASES[name] for name in references]
        padded_phrases = reference_phrases + [""] * (2 - len(reference_phrases))
        task_ids.append(task_id)
        instruction_features.append(mean_embedding(instructions[task_id], tokenizer, embedding))
        relation_phrase = RELATION_PHRASES[spec["relation"]]
        relation_features.append(mean_embedding(relation_phrase, tokenizer, embedding))
        reference_features.append(torch.stack([
            mean_embedding(phrase, tokenizer, embedding) if phrase else torch.zeros(4096)
            for phrase in padded_phrases
        ]))
        reference_presence.append([bool(phrase) for phrase in padded_phrases])
        text_records.append({
            "task_id": task_id, "instruction": instructions[task_id],
            "relation_phrase": relation_phrase, "reference_phrases": reference_phrases,
        })
    tensors = {
        "task_ids": torch.tensor(task_ids, dtype=torch.int64),
        "instruction_features": torch.stack(instruction_features),
        "relation_features": torch.stack(relation_features),
        "reference_features": torch.stack(reference_features),
        "reference_presence": torch.tensor(reference_presence, dtype=torch.bool),
    }
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise RuntimeError("Non-finite language feature")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata={
        "embedding_source": EMBEDDING_KEY,
        "pooling": "mean token embedding; no special tokens",
        "texts": json.dumps(text_records, ensure_ascii=False),
    })
    print(json.dumps({
        "output": str(args.output),
        "embedding_shape": list(tensors["instruction_features"].shape),
        "reference_shape": list(tensors["reference_features"].shape),
        "texts": text_records,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
