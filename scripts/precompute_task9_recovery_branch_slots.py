#!/usr/bin/env python
"""Enrich task-9 raw recovery branches with RGB-only grounded role slots."""

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
from scripts.train_task9_grounded_role_slots import ROLE_ORDER, Task9GroundedRoleSlots


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows.sort(key=lambda row: int(row["branch_timestep"]))
    if [int(row["branch_timestep"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"non-contiguous branch steps: {path}")
    return rows


class RecoveryFrames(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)), transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["online_observation"]["rgb"]) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, int(row["branch_timestep"])


def role_geometry(mask_logits: torch.Tensor) -> torch.Tensor:
    masks = mask_logits.sigmoid()
    height, width = masks.shape[-2:]
    y = torch.linspace(-1, 1, height, device=masks.device)[None, None, :, None]
    x = torch.linspace(-1, 1, width, device=masks.device)[None, None, None, :]
    mass = masks.sum((2, 3)).clamp_min(1)
    return torch.stack(((masks * x).sum((2, 3)) / mass,
                        (masks * y).sum((2, 3)) / mass,
                        mass / (height * width)), dim=-1)


def main() -> None:
    args = arguments()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    sources, seen = [], set()
    for root in args.raw_root:
        for path in sorted(root.rglob("raw-transitions.jsonl")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            rows = read_rows(path)
            if int(rows[0]["task_id"]) == 9:
                sources.append((path, rows))
                seen.add(resolved)
    if not sources:
        raise FileNotFoundError(f"no task-9 raw sequences below {args.raw_root}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Task9GroundedRoleSlots().to(device).eval()
    model.load_state_dict(load_file(args.checkpoint))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    with torch.inference_mode():
        for source, rows in sources:
            loader = DataLoader(RecoveryFrames(rows), batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
            slots, geometries, timesteps, actions = [], [], [], []
            for pixels, steps in loader:
                logits, role_slots = model(pixels.to(device))
                slots.append(role_slots.cpu())
                geometries.append(role_geometry(logits).cpu())
                timesteps.append(steps)
            actions = torch.tensor([
                row["online_observation"]["recovery_action"] for row in rows
            ], dtype=torch.float32)
            tensors = {
                "role_slots": torch.cat(slots).contiguous(),
                "role_geometry": torch.cat(geometries).contiguous(),
                "recovery_actions": actions,
                "timesteps": torch.cat(timesteps).long().contiguous(),
                "teacher_attached_after": torch.tensor(
                    [row["teacher_only"]["attached_after"] for row in rows], dtype=torch.bool
                ),
                "teacher_target_lift_after_m": torch.tensor(
                    [row["teacher_only"]["target_lift_after_m"] for row in rows], dtype=torch.float32
                ),
            }
            if not torch.equal(tensors["timesteps"], torch.arange(len(rows))):
                raise ValueError(f"timestep mismatch after loading {source}")
            identity = rows[0]
            destination = args.output_dir / (
                f"task9-source{identity['source_episode_id']}-candidate"
                f"{identity['candidate_index']}-roles.safetensors"
            )
            save_file(tensors, destination)
            summary = {
                "source": str(source.resolve()), "output": str(destination.resolve()),
                "source_episode_id": identity["source_episode_id"],
                "branch_episode_id": identity["branch_episode_id"],
                "candidate_name": identity["candidate_name"], "frames": len(rows),
                "role_slot_shape": list(tensors["role_slots"].shape),
            }
            outputs.append(summary)
            print(json.dumps(summary), flush=True)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "kind": "task9_recovery_branch_grounded_role_slots",
        "checkpoint": str(args.checkpoint.resolve()),
        "raw_roots": [str(root.resolve()) for root in args.raw_root],
        "role_order": list(ROLE_ORDER),
        "input_contract": "RGB only; teacher fields retained for evaluation only",
        "branches": outputs,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
