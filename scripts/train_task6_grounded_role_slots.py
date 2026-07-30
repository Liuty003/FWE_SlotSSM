#!/usr/bin/env python
"""Train task-6 cup/plate role slots with mask-supervised grounding."""

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.train_task6_visual_state_baseline import (  # noqa: E402
    binary_metrics,
    confirmed_trigger,
    parse_ids,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-root", type=Path, nargs="+", required=True)
    parser.add_argument("--mask-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--train-episodes", default="0,1,2,3,4,5,20,21,22,23,24,25,26,27,28,29"
    )
    parser.add_argument("--validation-episodes", default="6,8")
    parser.add_argument("--test-episodes", default="7,9")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--confirm-frames", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def load_visual_rows(roots):
    rows = []
    for root in roots:
        rows.extend(
            json.loads(line)
            for line in (root / "manifest.jsonl").read_text().splitlines()
            if line.strip()
        )
    return rows


def find_one(directories, name):
    matches = [directory / name for directory in directories if (directory / name).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {name}, found {matches}")
    return matches[0]


class GroundedFrameDataset(Dataset):
    def __init__(self, rows, mask_dirs, episodes, augment=False):
        selected = [row for row in rows if int(row["episode_id"]) in episodes]
        grouped = {}
        for episode in episodes:
            tensors = load_file(
                find_one(mask_dirs, f"task6-episode{episode}-role-masks.safetensors")
            )
            grouped[episode] = torch.stack(
                (tensors["mug_masks"], tensors["plate_masks"]), dim=1
            ).float()
        self.items = []
        for row in selected:
            episode, timestep = int(row["episode_id"]), int(row["timestep"])
            self.items.append((row, grouped[episode][timestep]))
        operations = [transforms.Resize((128, 128))]
        if augment:
            operations.append(
                transforms.ColorJitter(brightness=.12, contrast=.12, saturation=.08)
            )
        operations += [
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ]
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row, masks = self.items[index]
        with Image.open(row["image"]) as image:
            image = self.transform(image.convert("RGB"))
        return (
            image, masks, torch.tensor(float(row["mug_on_plate"])),
            int(row["episode_id"]), int(row["timestep"]), index,
        )


class GroundedRoleSlots(nn.Module):
    def __init__(self, feature_dim=96):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, feature_dim, 3, padding=1), nn.BatchNorm2d(feature_dim), nn.GELU(),
        )
        self.mask_head = nn.Conv2d(feature_dim, 2, 1)
        self.relation_head = nn.Sequential(
            nn.Linear(feature_dim * 4 + 7, 192), nn.GELU(), nn.Dropout(.1),
            nn.Linear(192, 1),
        )

    def forward(self, images):
        features = self.encoder(images)
        mask_logits = self.mask_head(features)
        probabilities = mask_logits.sigmoid()
        denominator = probabilities.flatten(2).sum(-1).clamp_min(1)[..., None]
        role_slots = torch.einsum(
            "brhw,bdhw->brd", probabilities, features
        ) / denominator
        first, second = role_slots[:, 0], role_slots[:, 1]
        size = probabilities.shape[-1]
        axis = torch.linspace(-1, 1, size, device=images.device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        area = probabilities.flatten(2).mean(-1)
        cx = (probabilities * xx).flatten(2).sum(-1) / denominator[..., 0]
        cy = (probabilities * yy).flatten(2).sum(-1) / denominator[..., 0]
        overlap = (probabilities[:, 0] * probabilities[:, 1]).flatten(1).mean(-1)
        geometry = torch.stack(
            (area[:, 0], area[:, 1], cx[:, 0], cy[:, 0],
             cx[:, 1], cy[:, 1], overlap), dim=-1
        )
        relation = torch.cat(
            (first, second, first - second, first * second, geometry), dim=-1
        )
        return mask_logits, self.relation_head(relation).squeeze(-1), role_slots


def dice_loss(logits, targets):
    probabilities = logits.sigmoid().flatten(2)
    targets = targets.flatten(2)
    intersection = (probabilities * targets).sum(-1)
    return (1 - (2 * intersection + 1) / (
        probabilities.sum(-1) + targets.sum(-1) + 1
    )).mean()


@torch.inference_mode()
def evaluate(model, dataset, loader, device, threshold=.5, confirm_frames=5):
    model.eval()
    scores = np.zeros(len(dataset), np.float32)
    labels = np.zeros(len(dataset), np.int64)
    intersections = np.zeros(2, np.float64)
    unions = np.zeros(2, np.float64)
    metadata = [None] * len(dataset)
    for images, masks, relation, episodes, timesteps, indices in loader:
        mask_logits, relation_logits, _ = model(images.to(device))
        predictions = mask_logits.sigmoid().cpu() >= .5
        targets = masks.bool()
        intersections += (predictions & targets).sum((0, 2, 3)).numpy()
        unions += (predictions | targets).sum((0, 2, 3)).numpy()
        scores[indices.numpy()] = relation_logits.sigmoid().cpu().numpy()
        labels[indices.numpy()] = relation.numpy().astype(np.int64)
        for index, episode, timestep in zip(indices, episodes, timesteps):
            metadata[int(index)] = (int(episode), int(timestep))
    report = binary_metrics(labels, scores, threshold)
    report["mask_iou"] = {
        "mug": float(intersections[0] / max(unions[0], 1)),
        "plate": float(intersections[1] / max(unions[1], 1)),
    }
    episode_reports = []
    for episode in sorted({item[0] for item in metadata}):
        indices = sorted(
            (i for i, item in enumerate(metadata) if item[0] == episode),
            key=lambda i: metadata[i][1],
        )
        expected = confirmed_trigger(labels[indices].astype(bool), confirm_frames)
        predicted = confirmed_trigger(scores[indices] >= threshold, confirm_frames)
        episode_reports.append({
            "episode_id": episode,
            "expected_trigger_index": expected,
            "predicted_trigger_index": predicted,
            "false_trigger": predicted is not None and expected is None,
            "missed_trigger": predicted is None and expected is not None,
            "trigger_delay_frames": (
                predicted - expected
                if predicted is not None and expected is not None else None
            ),
        })
    report["episodes"] = episode_reports
    return report


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    rows = load_visual_rows(args.visual_root)
    split_ids = {
        "train": parse_ids(args.train_episodes),
        "validation": parse_ids(args.validation_episodes),
        "test": parse_ids(args.test_episodes),
    }
    datasets = {
        name: GroundedFrameDataset(
            rows, args.mask_dir, episodes, augment=name == "train"
        )
        for name, episodes in split_ids.items()
    }
    train_labels = np.asarray(
        [float(item[0]["mug_on_plate"]) for item in datasets["train"].items]
    )
    counts = np.bincount(train_labels.astype(int), minlength=2)
    weights = [1 / max(counts[int(label)], 1) for label in train_labels]
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            weights, len(weights), generator=generator
        ),
        num_workers=args.workers, pin_memory=True,
    )
    eval_loaders = {
        name: DataLoader(
            datasets[name], batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
        for name in datasets
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GroundedRoleSlots().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, masks, relation, *_ in train_loader:
            mask_logits, relation_logits, _ = model(images.to(device))
            masks = masks.to(device)
            relation = relation.to(device)
            mask_bce = nn.functional.binary_cross_entropy_with_logits(
                mask_logits, masks, pos_weight=torch.tensor(
                    [15., 15.], device=device
                )[None, :, None, None]
            )
            relation_loss = nn.functional.binary_cross_entropy_with_logits(
                relation_logits, relation
            )
            loss = mask_bce + dice_loss(mask_logits, masks) + relation_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
        validation = evaluate(
            model, datasets["validation"], eval_loaders["validation"], device,
            confirm_frames=args.confirm_frames,
        )
        trajectory_errors = sum(
            item["false_trigger"] or item["missed_trigger"]
            for item in validation["episodes"]
        )
        key = (
            trajectory_errors,
            -validation["balanced_accuracy"],
            -sum(validation["mask_iou"].values()),
        )
        history.append({
            "epoch": epoch, "loss": float(np.mean(losses)),
            "validation": validation,
        })
        print(
            f"epoch={epoch} loss={np.mean(losses):.4f} "
            f"val_ba={validation['balanced_accuracy']:.4f} "
            f"iou={validation['mask_iou']} errors={trajectory_errors}",
            flush=True,
        )
        if best is None or key < best["key"]:
            best = {
                "key": key, "epoch": epoch,
                "state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }
    model.load_state_dict(best["state"])
    metrics = {
        name: evaluate(
            model, datasets[name], loader, device,
            confirm_frames=args.confirm_frames,
        )
        for name, loader in eval_loaders.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "task6-grounded-role-slots.safetensors"
    save_file(best["state"], checkpoint)
    result = {
        "schema_version": 1,
        "kind": "task6_mask_supervised_rgb_role_slots",
        "input_contract": "RGB only at inference",
        "supervision_contract": "mug/plate masks and On predicate during training only",
        "episode_split": split_ids,
        "best_epoch": best["epoch"],
        "metrics": metrics,
        "history": history,
        "checkpoint": str(checkpoint),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
