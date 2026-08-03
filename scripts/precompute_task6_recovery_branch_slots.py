#!/usr/bin/env python
"""Extract existing task-6 grounded role slots on recovery branch sequences.

This is an enrichment step only.  It does not train on the 8 task-6 branches,
and it keeps simulator contact/pose labels out of the slot encoder inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.train_task6_execution_slots import Task6ExecutionSlots  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        action="append",
        required=True,
        help=(
            "Root containing raw branch trajectories. Repeat this option to "
            "combine independently collected natural failure states into one "
            "slot-enrichment dataset."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def rows(path: Path) -> list[dict]:
    values = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    values.sort(key=lambda value: int(value["branch_timestep"]))
    if [value["branch_timestep"] for value in values] != list(range(len(values))):
        raise ValueError(f"non-contiguous branch steps: {path}")
    return values


class RecoveryFrames(Dataset):
    def __init__(self, sequence: list[dict]):
        self.sequence = sequence
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, index: int):
        row = self.sequence[index]
        with Image.open(row["online_observation"]["rgb"]) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return (
            image_tensor,
            torch.tensor(row["online_observation"]["recovery_action"], dtype=torch.float32),
            int(row["branch_timestep"]),
            bool(row["teacher_only"]["attached_after"]),
            float(row["teacher_only"]["target_lift_after_m"]),
        )


def role_geometry(mask_logits: torch.Tensor) -> torch.Tensor:
    masks = mask_logits.sigmoid()
    height, width = masks.shape[-2:]
    y = torch.linspace(-1, 1, height, device=masks.device)[None, None, :, None]
    x = torch.linspace(-1, 1, width, device=masks.device)[None, None, None, :]
    mass = masks.sum((2, 3)).clamp_min(1)
    return torch.stack((
        (masks * x).sum((2, 3)) / mass,
        (masks * y).sum((2, 3)) / mass,
        mass / (height * width),
    ), dim=-1)


def main() -> None:
    args = arguments()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    sources = []
    seen_sources = set()
    for raw_root in args.raw_root:
        for path in sorted(raw_root.rglob("raw-transitions.jsonl")):
            resolved = path.resolve()
            if resolved in seen_sources:
                continue
            sequence = rows(path)
            if int(sequence[0]["task_id"]) == 6:
                sources.append((path, sequence))
                seen_sources.add(resolved)
    if not sources:
        raise FileNotFoundError(f"no task-6 raw sequences below {args.raw_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Task6ExecutionSlots().to(device).eval()
    model.load_state_dict(load_file(args.checkpoint))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    with torch.inference_mode():
        for source, sequence in sources:
            loader = DataLoader(
                RecoveryFrames(sequence), batch_size=args.batch_size,
                shuffle=False, num_workers=args.workers, pin_memory=True,
            )
            role_slots, geometries, actions = [], [], []
            timesteps, attached_after, lift_after = [], [], []
            for images, batch_actions, steps, attached, lift in loader:
                logits, slots, _, _ = model(
                    images.to(device), batch_actions.to(device)
                )
                role_slots.append(slots.cpu())
                geometries.append(role_geometry(logits).cpu())
                actions.append(batch_actions)
                timesteps.append(steps)
                attached_after.append(attached)
                lift_after.append(lift)
            tensors = {
                "role_slots": torch.cat(role_slots).contiguous(),
                "role_geometry": torch.cat(geometries).contiguous(),
                "recovery_actions": torch.cat(actions).contiguous(),
                "timesteps": torch.cat(timesteps).long().contiguous(),
                "teacher_attached_after": torch.cat(attached_after).bool().contiguous(),
                "teacher_target_lift_after_m": torch.cat(lift_after).float().contiguous(),
            }
            expected = torch.arange(len(sequence))
            if not torch.equal(tensors["timesteps"], expected):
                raise ValueError(f"timestep mismatch after loading {source}")
            identity = sequence[0]
            name = (
                f"task6-source{identity['source_episode_id']}"
                f"-candidate{identity['candidate_index']}-roles.safetensors"
            )
            destination = args.output_dir / name
            save_file(tensors, destination)
            outputs.append({
                "source": str(source.resolve()),
                "output": str(destination.resolve()),
                "source_episode_id": identity["source_episode_id"],
                "branch_episode_id": identity["branch_episode_id"],
                "candidate_name": identity["candidate_name"],
                "frames": len(sequence),
                "role_slot_shape": list(tensors["role_slots"].shape),
            })
            print(json.dumps(outputs[-1]), flush=True)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "kind": "task6_recovery_branch_grounded_role_slots",
        "checkpoint": str(args.checkpoint.resolve()),
        "raw_roots": [str(path.resolve()) for path in args.raw_root],
        "input_contract": "raw RGB plus recovery action only",
        "role_order": ["porcelain_mug", "plate", "pudding"],
        "teacher_fields_retained_for_evaluation_only": [
            "attached_after", "target_lift_after_m",
        ],
        "branches": outputs,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
