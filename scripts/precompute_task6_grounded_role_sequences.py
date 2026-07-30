#!/usr/bin/env python
"""Precompute grounded cup/plate role-slot sequences from RGB."""

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
from scripts.train_task6_grounded_role_slots import GroundedRoleSlots  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-root", type=Path, nargs="+", required=True)
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
            torch.tensor(float(row["mug_on_plate"])),
            int(row["timestep"]),
        )


def main():
    args = arguments()
    rows = []
    for root in args.visual_root:
        rows.extend(
            json.loads(line)
            for line in (root / "manifest.jsonl").read_text().splitlines()
            if line.strip()
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GroundedRoleSlots().to(device).eval()
    model.load_state_dict(load_file(args.checkpoint))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    with torch.inference_mode():
        for episode in args.episode_ids:
            episode_rows = [
                row for row in rows if int(row["episode_id"]) == episode
            ]
            dataset = EpisodeFrames(episode_rows)
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True,
            )
            slots, scores, labels, timesteps = [], [], [], []
            for images, batch_labels, batch_timesteps in loader:
                _, logits, role_slots = model(images.to(device))
                slots.append(role_slots.cpu())
                scores.append(logits.sigmoid().cpu())
                labels.append(batch_labels)
                timesteps.append(batch_timesteps)
            tensors = {
                "role_slots": torch.cat(slots).contiguous(),
                "single_frame_probability": torch.cat(scores).contiguous(),
                "labels": torch.cat(labels).float().contiguous(),
                "timesteps": torch.cat(timesteps).long().contiguous(),
            }
            expected = torch.arange(len(dataset))
            if not torch.equal(tensors["timesteps"], expected):
                raise ValueError(f"episode {episode} timesteps are not contiguous")
            output = args.output_dir / (
                f"task6-episode{episode}-grounded-roles.safetensors"
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
    (args.output_dir / "metadata.json").write_text(
        json.dumps({
            "schema_version": 1,
            "checkpoint": str(args.checkpoint),
            "input_contract": "RGB only",
            "role_order": ["porcelain_mug", "plate"],
            "episodes": summaries,
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
