#!/usr/bin/env python
"""Precompute RGB-only task-9 role-slot sequences for failure-risk learning.

Teacher labels are retained solely as training targets.  At inference the
sequence input is RGB role slots, executed actions, and gripper aperture.
"""

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
from scripts.precompute_task9_recovery_branch_slots import role_geometry  # noqa: E402
from scripts.train_task9_grounded_role_slots import (  # noqa: E402
    ROLE_ORDER,
    Task9GroundedRoleSlots,
)


def arguments() -> argparse.Namespace:
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
    def __init__(self, rows: list[dict]):
        self.rows = sorted(rows, key=lambda row: int(row["timestep"]))
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["image"]) as image:
            pixels = self.transform(image.convert("RGB"))
        return (
            pixels,
            torch.tensor(row["executed_action"], dtype=torch.float32),
            torch.tensor(float(row["gripper_aperture"]), dtype=torch.float32),
            torch.tensor(bool(row["failed_acquisition"]), dtype=torch.bool),
            torch.tensor(bool(row["transport_drop"]), dtype=torch.bool),
            torch.tensor(bool(row["acquisition_assessment_active"]), dtype=torch.bool),
            torch.tensor(int(row["timestep"]), dtype=torch.long),
        )


def read_visual_roots(roots: list[Path]) -> dict[tuple[int, int], str]:
    visual = {}
    for root in roots:
        manifest = root / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        for line in manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (int(row["episode_id"]), int(row["timestep"]))
            if key in visual and visual[key] != row["image"]:
                raise ValueError(f"duplicate visual frame for {key}")
            visual[key] = row["image"]
    return visual


def main() -> None:
    args = arguments()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    visual = read_visual_roots(args.visual_root)
    labels = {
        (int(row["episode_id"]), int(row["timestep"])): row
        for line in args.labels.read_text(encoding="utf-8").splitlines()
        if line
        for row in [json.loads(line)]
    }
    rows = []
    for key in sorted(set(visual) & set(labels)):
        row = dict(labels[key])
        row["image"] = visual[key]
        rows.append(row)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Task9GroundedRoleSlots().to(device).eval()
    model.load_state_dict(load_file(args.checkpoint))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    with torch.inference_mode():
        for episode in args.episode_ids:
            dataset = EpisodeFrames([
                row for row in rows if int(row["episode_id"]) == episode
            ])
            if not len(dataset):
                raise ValueError(f"episode {episode} has no aligned RGB/labels")
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True,
            )
            slots, geometry, actions, apertures = [], [], [], []
            failed, dropped, assessed, timesteps = [], [], [], []
            for batch in loader:
                (
                    pixels, batch_actions, batch_aperture, batch_failed,
                    batch_dropped, batch_assessed, batch_timesteps,
                ) = batch
                logits, role_slots = model(pixels.to(device))
                slots.append(role_slots.cpu())
                geometry.append(role_geometry(logits).cpu())
                actions.append(batch_actions)
                apertures.append(batch_aperture)
                failed.append(batch_failed)
                dropped.append(batch_dropped)
                assessed.append(batch_assessed)
                timesteps.append(batch_timesteps)
            tensors = {
                "role_slots": torch.cat(slots).contiguous(),
                "role_geometry": torch.cat(geometry).contiguous(),
                "executed_actions": torch.cat(actions).contiguous(),
                "gripper_aperture": torch.cat(apertures).contiguous(),
                "teacher_failed_acquisition": torch.cat(failed).bool().contiguous(),
                "teacher_transport_drop": torch.cat(dropped).bool().contiguous(),
                "teacher_assessment_active": torch.cat(assessed).bool().contiguous(),
                "timesteps": torch.cat(timesteps).long().contiguous(),
            }
            if not torch.equal(tensors["timesteps"], torch.arange(len(dataset))):
                raise ValueError(f"episode {episode} timesteps are not contiguous")
            output = args.output_dir / f"task9-episode{episode}-execution-roles.safetensors"
            save_file(tensors, output)
            summary = {
                "episode_id": episode,
                "frames": len(dataset),
                "role_slot_shape": list(tensors["role_slots"].shape),
                "failed_acquisition_frames": int(tensors["teacher_failed_acquisition"].sum()),
                "transport_drop_frames": int(tensors["teacher_transport_drop"].sum()),
                "file": str(output),
            }
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "role_order": list(ROLE_ORDER),
        "input_contract": (
            "RGB role slots, executed actions, and gripper aperture only; "
            "teacher failure flags are targets only"
        ),
        "episodes": summaries,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
