#!/usr/bin/env python
"""Train RGB-only grounded role slots for current LIBERO-10 task 9.

The simulator instance masks supervise the three semantic roles during
training, but this checkpoint accepts only RGB when used on recovery branches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.train_task6_grounded_role_slots import dice_loss
from scripts.train_task6_visual_state_baseline import parse_ids


ROLE_ORDER = ("white_yellow_mug", "microwave", "porcelain_mug")
MASK_KEYS = ("target_mug_masks", "microwave_masks", "distractor_mug_masks")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-episodes", default="0,2,3,4,7,8,9")
    parser.add_argument("--validation-episodes", default="5")
    parser.add_argument("--test-episodes", default="1,6")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


class Task9GroundedRoleSlots(nn.Module):
    def __init__(self, feature_dim: int = 96):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, feature_dim, 3, padding=1), nn.BatchNorm2d(feature_dim), nn.GELU(),
        )
        self.mask_head = nn.Conv2d(feature_dim, len(ROLE_ORDER), 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(images)
        mask_logits = self.mask_head(features)
        probabilities = mask_logits.sigmoid()
        denominator = probabilities.flatten(2).sum(-1).clamp_min(1)[..., None]
        roles = torch.einsum("brhw,bdhw->brd", probabilities, features) / denominator
        return mask_logits, roles


class RoleMaskDataset(Dataset):
    def __init__(self, visual_root: Path, mask_dir: Path, episodes: set[int], augment: bool):
        visual = {}
        for line in (visual_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (int(row["episode_id"]), int(row["timestep"]))
            if key[0] in episodes:
                visual[key] = row["image"]
        masks = {}
        for episode in episodes:
            shard = load_file(mask_dir / f"task9-episode{episode}-role-masks.safetensors")
            masks[episode] = torch.stack([shard[key] for key in MASK_KEYS], dim=1).float()
        self.items = [
            (visual[key], masks[key[0]][key[1]], key)
            for key in sorted(visual)
            if key[1] < len(masks[key[0]])
        ]
        if not self.items:
            raise ValueError(f"no aligned examples for episodes {sorted(episodes)}")
        operations = [transforms.Resize((128, 128))]
        if augment:
            operations.append(transforms.ColorJitter(brightness=.12, contrast=.12, saturation=.08))
        operations += [
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ]
        self.transform = transforms.Compose(operations)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, masks, identity = self.items[index]
        with Image.open(path) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, masks, index, identity[0], identity[1]


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, count: int, device: torch.device) -> dict:
    intersections = np.zeros(len(ROLE_ORDER), np.float64)
    unions = np.zeros(len(ROLE_ORDER), np.float64)
    visible = np.zeros(len(ROLE_ORDER), np.int64)
    model.eval()
    for pixels, masks, *_ in loader:
        logits, _ = model(pixels.to(device))
        prediction = logits.sigmoid().cpu() >= .5
        truth = masks.bool()
        intersections += (prediction & truth).sum((0, 2, 3)).numpy()
        unions += (prediction | truth).sum((0, 2, 3)).numpy()
        visible += truth.flatten(2).any(2).sum(0).numpy()
    return {
        "frames": count,
        "mask_iou": {
            role: float(intersections[index] / max(unions[index], 1))
            for index, role in enumerate(ROLE_ORDER)
        },
        "visible_frames": {role: int(visible[index]) for index, role in enumerate(ROLE_ORDER)},
    }


def main() -> None:
    args = arguments()
    torch.manual_seed(args.seed)
    splits = {
        "train": set(parse_ids(args.train_episodes)),
        "validation": set(parse_ids(args.validation_episodes)),
        "test": set(parse_ids(args.test_episodes)),
    }
    if splits["train"] & splits["validation"] or splits["train"] & splits["test"]:
        raise ValueError("episode splits must be disjoint")
    datasets = {
        name: RoleMaskDataset(args.visual_root, args.mask_dir, episodes, name == "train")
        for name, episodes in splits.items()
    }
    loaders = {
        name: DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train",
                         num_workers=args.workers, pin_memory=True)
        for name, dataset in datasets.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Task9GroundedRoleSlots().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best, history = None, []
    pos_weight = torch.tensor([15., 4., 15.], device=device)[None, :, None, None]
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for pixels, masks, *_ in loaders["train"]:
            logits, _ = model(pixels.to(device))
            masks = masks.to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, masks, pos_weight=pos_weight) + dice_loss(logits, masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss))
        validation = evaluate(model, loaders["validation"], len(datasets["validation"]), device)
        score = float(np.mean(list(validation["mask_iou"].values())))
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "validation": validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        if best is None or score > best["score"]:
            best = {"score": score, **record}
            args.output_dir.mkdir(parents=True, exist_ok=True)
            save_file(model.state_dict(), args.output_dir / "task9-grounded-role-slots.safetensors")
    # Report the selected validation checkpoint, not the final optimization
    # iterate, since the latter can overfit the small set of initial states.
    model.load_state_dict(load_file(args.output_dir / "task9-grounded-role-slots.safetensors"))
    test = evaluate(model, loaders["test"], len(datasets["test"]), device)
    report = {
        "schema_version": 1,
        "kind": "task9_rgb_grounded_role_slots",
        "role_order": list(ROLE_ORDER),
        "input_contract": "RGB only; simulator masks are supervision only",
        "splits": {name: sorted(values) for name, values in splits.items()},
        "best_validation": best,
        "test": test,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
