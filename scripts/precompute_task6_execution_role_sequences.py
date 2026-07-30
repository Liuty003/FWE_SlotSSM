#!/usr/bin/env python
"""Precompute three grounded task-6 role slots from RGB frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.train_task6_execution_slots import (  # noqa: E402
    HEADS,
    Task6ExecutionSlots,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-root", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-ids", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


class EpisodeFrames(Dataset):
    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda row: int(row["timestep"]))
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image"]) as image:
            image = self.transform(image.convert("RGB"))
        return (
            image,
            torch.tensor(row["executed_action"], dtype=torch.float32),
            torch.tensor([float(row[name]) for name in HEADS]),
            torch.tensor(float(row["candidate_value"])),
            torch.tensor(bool(row["stage2_active"])),
            torch.tensor(bool(row["acquisition_assessment_active"])),
            int(row["timestep"]),
        )


def main():
    args = arguments()
    visual = {}
    for root in args.visual_root:
        for line in (root / "manifest.jsonl").read_text().splitlines():
            row = json.loads(line)
            visual[(int(row["episode_id"]), int(row["timestep"]))] = row["image"]
    labels = {}
    for line in args.labels.read_text().splitlines():
        row = json.loads(line)
        labels[(int(row["episode_id"]), int(row["timestep"]))] = row
    rows = []
    for key in sorted(set(visual) & set(labels)):
        row = dict(labels[key])
        row["image"] = visual[key]
        rows.append(row)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Task6ExecutionSlots().to(device).eval()
    model.load_state_dict(load_file(args.checkpoint))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    with torch.inference_mode():
        for episode in args.episode_ids:
            dataset = EpisodeFrames([
                row for row in rows if int(row["episode_id"]) == episode
            ])
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True,
            )
            slots, actions, targets, values, active, assessment, timesteps = (
                [] for _ in range(7)
            )
            geometries = []
            for images, batch_actions, batch_targets, batch_values, batch_active, batch_assessment, batch_t in loader:
                mask_logits, role_slots, _, _ = model(
                    images.to(device), batch_actions.to(device)
                )
                masks = mask_logits.sigmoid()
                height, width = masks.shape[-2:]
                y = torch.linspace(-1, 1, height, device=device)[None, None, :, None]
                x = torch.linspace(-1, 1, width, device=device)[None, None, None, :]
                mass = masks.sum((2, 3)).clamp_min(1)
                geometry = torch.stack((
                    (masks * x).sum((2, 3)) / mass,
                    (masks * y).sum((2, 3)) / mass,
                    mass / (height * width),
                ), dim=-1)
                slots.append(role_slots.cpu())
                geometries.append(geometry.cpu())
                actions.append(batch_actions)
                targets.append(batch_targets)
                values.append(batch_values)
                active.append(batch_active)
                assessment.append(batch_assessment)
                timesteps.append(batch_t)
            tensors = {
                "role_slots": torch.cat(slots).contiguous(),
                "role_geometry": torch.cat(geometries).contiguous(),
                "actions": torch.cat(actions).contiguous(),
                "targets": torch.cat(targets).contiguous(),
                "candidate_values": torch.cat(values).contiguous(),
                "stage2_active": torch.cat(active).bool().contiguous(),
                "acquisition_assessment_active": torch.cat(
                    assessment
                ).bool().contiguous(),
                "timesteps": torch.cat(timesteps).long().contiguous(),
            }
            if not torch.equal(
                tensors["timesteps"], torch.arange(len(dataset))
            ):
                raise ValueError(f"episode {episode} timesteps are not contiguous")
            output = args.output_dir / (
                f"task6-episode{episode}-execution-roles.safetensors"
            )
            save_file(tensors, output)
            summary = {
                "episode_id": episode,
                "frames": len(dataset),
                "role_slot_shape": list(tensors["role_slots"].shape),
                "file": str(output),
            }
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "input_contract": "RGB and executed action; privileged data are labels only",
        "role_order": ["porcelain_mug", "plate", "pudding"],
        "head_order": HEADS,
        "episodes": summaries,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
