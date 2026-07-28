#!/usr/bin/env python
"""Attach counterfactual masks to existing frozen vision-patch shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


RELATION_ORDER = ("between", "in", "next_to", "on", "table_center")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patch-dir", type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/stage-b-spatial-relation-patches"),
    )
    parser.add_argument(
        "--counterfactual-dir", type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/stage-b-spatial-counterfactual-data"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/stage-b-spatial-counterfactual-patches"),
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    source = json.loads((args.patch_dir / "manifest.json").read_text())
    counterfactual = json.loads((args.counterfactual_dir / "manifest.json").read_text())
    samples = {
        (int(sample["task_id"]), int(sample["state_id"])): sample
        for sample in counterfactual["samples"]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for task in source["tasks"]:
        task_id = int(task["task_id"])
        tensors = load_file(task["output"])
        target_masks, reference_masks, reference_presence, relation_ids = [], [], [], []
        for state_id in tensors["state_ids"].tolist():
            sample = samples[(task_id, int(state_id))]
            with np.load(sample["masks"]) as stored:
                target_masks.append(stored["counterfactual_target"])
                current_masks, current_presence = [], []
                for index in range(2):
                    key = f"counterfactual_reference_{index}"
                    exists = key in stored.files
                    current_presence.append(exists)
                    current_masks.append(
                        stored[key] if exists else np.zeros((256, 256), dtype=np.uint8)
                    )
                reference_masks.append(np.stack(current_masks))
                reference_presence.append(current_presence)
            relation_ids.append(RELATION_ORDER.index(sample["counterfactual"]["relation"]))
        target_masks = F.interpolate(
            torch.from_numpy(np.stack(target_masks))[:, None].float(),
            size=(64, 64), mode="nearest",
        )[:, 0].to(torch.uint8)
        reference_masks = F.interpolate(
            torch.from_numpy(np.stack(reference_masks)).float(),
            size=(64, 64), mode="nearest",
        ).to(torch.uint8)
        output_tensors = dict(tensors)
        output_tensors.update({
            "counterfactual_target_masks": target_masks.contiguous(),
            "counterfactual_reference_masks": reference_masks.contiguous(),
            "counterfactual_reference_presence": torch.tensor(
                reference_presence, dtype=torch.bool
            ),
            "counterfactual_relation_ids": torch.tensor(relation_ids, dtype=torch.int64),
        })
        output = args.output_dir / f"task{task_id}.safetensors"
        save_file(output_tensors, output)
        record = {
            "task_id": task_id, "num_samples": len(target_masks), "output": str(output),
            "counterfactual_target_shape": list(target_masks.shape),
            "counterfactual_reference_shape": list(reference_masks.shape),
            "counterfactual_reference_presence": int(
                torch.tensor(reference_presence).sum()
            ),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    manifest = dict(source)
    manifest.update({
        "schema_version": 3,
        "source_patch_manifest": str(args.patch_dir / "manifest.json"),
        "counterfactual_manifest": str(args.counterfactual_dir / "manifest.json"),
        "tasks": records,
    })
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), "tasks": len(records)}, indent=2))


if __name__ == "__main__":
    main()
