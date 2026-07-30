#!/usr/bin/env python
"""Train action-conditioned task-6 execution heads over three grounded roles."""

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
from scripts.train_task6_grounded_role_slots import dice_loss  # noqa: E402
from scripts.train_task6_visual_state_baseline import (  # noqa: E402
    binary_metrics,
    parse_ids,
)

HEADS = (
    "effective_progress", "stall", "controlled", "failed_acquisition",
    "slip_risk", "collateral_risk",
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-root", type=Path, nargs="+", required=True)
    parser.add_argument("--mask-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--train-episodes", default="0,1,2,3,4,5,20,21,22,23,24,25,26,27,28,29"
    )
    parser.add_argument("--validation-episodes", default="6,8")
    parser.add_argument("--test-episodes", default="7,9")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def find_one(directories, name):
    matches = [root / name for root in directories if (root / name).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {name}, found {matches}")
    return matches[0]


class ExecutionDataset(Dataset):
    def __init__(self, visual_roots, mask_dirs, label_path, episodes, augment):
        visual = {}
        for root in visual_roots:
            for line in (root / "manifest.jsonl").read_text().splitlines():
                row = json.loads(line)
                key = (int(row["episode_id"]), int(row["timestep"]))
                if key[0] in episodes:
                    visual[key] = row["image"]
        labels = {}
        for line in label_path.read_text().splitlines():
            row = json.loads(line)
            key = (int(row["episode_id"]), int(row["timestep"]))
            if key[0] in episodes:
                labels[key] = row
        masks = {}
        for episode in episodes:
            shard = load_file(find_one(
                mask_dirs, f"task6-episode{episode}-role-masks.safetensors"
            ))
            masks[episode] = torch.stack(
                (
                    shard["mug_masks"], shard["plate_masks"],
                    shard["pudding_masks"],
                ),
                dim=1,
            ).float()
        keys = sorted(set(visual) & set(labels))
        self.items = [
            (visual[key], masks[key[0]][key[1]], labels[key]) for key in keys
        ]
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
        path, masks, row = self.items[index]
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        targets = torch.tensor([float(row[name]) for name in HEADS])
        return (
            image, masks, torch.tensor(row["executed_action"], dtype=torch.float32),
            targets, torch.tensor(float(row["candidate_value"])),
            torch.tensor(bool(row["stage2_active"])), index,
        )


class Task6ExecutionSlots(nn.Module):
    def __init__(self, feature_dim=96):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim), nn.GELU(),
        )
        self.mask_head = nn.Conv2d(feature_dim, 3, 1)
        execution_dim = feature_dim * 5 + 7
        self.execution = nn.Sequential(
            nn.LayerNorm(execution_dim), nn.Linear(execution_dim, 256),
            nn.GELU(), nn.Dropout(.1), nn.Linear(256, 128), nn.GELU(),
        )
        self.binary_heads = nn.Linear(128, len(HEADS))
        self.value_head = nn.Linear(128, 1)

    def forward(self, images, actions):
        features = self.encoder(images)
        mask_logits = self.mask_head(features)
        probabilities = mask_logits.sigmoid()
        denominator = probabilities.flatten(2).sum(-1).clamp_min(1)[..., None]
        roles = torch.einsum(
            "brhw,bdhw->brd", probabilities, features
        ) / denominator
        mug, plate, pudding = roles[:, 0], roles[:, 1], roles[:, 2]
        state_action = torch.cat(
            (mug, plate, pudding, pudding - mug, pudding - plate, actions),
            dim=-1,
        )
        hidden = self.execution(state_action)
        return (
            mask_logits, roles, self.binary_heads(hidden),
            self.value_head(hidden).squeeze(-1),
        )


@torch.inference_mode()
def evaluate(model, dataset, loader, device):
    model.eval()
    logits = np.zeros((len(dataset), len(HEADS)), np.float32)
    labels = np.zeros_like(logits)
    values = np.zeros(len(dataset), np.float32)
    value_targets = np.zeros(len(dataset), np.float32)
    active = np.zeros(len(dataset), bool)
    intersections = np.zeros(3, np.float64)
    unions = np.zeros(3, np.float64)
    for images, masks, actions, targets, target_values, stage2, indices in loader:
        mask_logits, _, binary, predicted_values = model(
            images.to(device), actions.to(device)
        )
        prediction = mask_logits.sigmoid().cpu() >= .5
        truth = masks.bool()
        intersections += (prediction & truth).sum((0, 2, 3)).numpy()
        unions += (prediction | truth).sum((0, 2, 3)).numpy()
        index = indices.numpy()
        logits[index] = binary.cpu().numpy()
        labels[index] = targets.numpy()
        values[index] = predicted_values.cpu().numpy()
        value_targets[index] = target_values.numpy()
        active[index] = stage2.numpy()
    metrics = {}
    for column, name in enumerate(HEADS):
        metrics[name] = binary_metrics(
            labels[active, column].astype(int),
            1 / (1 + np.exp(-logits[active, column])),
            .5,
        )
    metrics["candidate_value_mae_m"] = float(
        np.abs(values[active] - value_targets[active]).mean()
    )
    metrics["mask_iou"] = {
        name: float(intersections[index] / max(unions[index], 1))
        for index, name in enumerate(("mug", "plate", "pudding"))
    }
    metrics["stage2_frames"] = int(active.sum())
    return metrics


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    split_ids = {
        "train": parse_ids(args.train_episodes),
        "validation": parse_ids(args.validation_episodes),
        "test": parse_ids(args.test_episodes),
    }
    datasets = {
        name: ExecutionDataset(
            args.visual_root, args.mask_dir, args.labels,
            episodes, augment=name == "train",
        )
        for name, episodes in split_ids.items()
    }
    stage2 = np.asarray(
        [float(item[2]["stage2_active"]) for item in datasets["train"].items]
    )
    active_targets = np.asarray([
        [float(item[2][name]) for name in HEADS]
        for item in datasets["train"].items
        if item[2]["stage2_active"]
    ])
    positive_counts = active_targets.sum(axis=0)
    negative_counts = len(active_targets) - positive_counts
    head_pos_weights = torch.tensor(
        np.clip(
            negative_counts / np.maximum(positive_counts, 1),
            1,
            30,
        ),
        dtype=torch.float32,
    )
    weights = np.where(stage2 > 0, 1 / max(stage2.sum(), 1),
                       1 / max((stage2 == 0).sum(), 1))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size,
        sampler=WeightedRandomSampler(weights, len(weights), generator=generator),
        num_workers=args.workers, pin_memory=True,
    )
    eval_loaders = {
        name: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
        for name, dataset in datasets.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Task6ExecutionSlots().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, masks, actions, targets, values, stage2, _ in train_loader:
            mask_logits, _, binary, predicted_values = model(
                images.to(device), actions.to(device)
            )
            masks = masks.to(device)
            mask_loss = nn.functional.binary_cross_entropy_with_logits(
                mask_logits, masks, pos_weight=torch.tensor(
                    [15., 15., 15.], device=device
                )[None, :, None, None]
            ) + dice_loss(mask_logits, masks)
            active = stage2.to(device).bool()
            if active.any():
                binary_loss = nn.functional.binary_cross_entropy_with_logits(
                    binary[active], targets.to(device)[active],
                    pos_weight=head_pos_weights.to(device),
                )
                value_loss = nn.functional.smooth_l1_loss(
                    predicted_values[active] * 100,
                    values.to(device)[active] * 100,
                )
            else:
                binary_loss = binary.sum() * 0
                value_loss = predicted_values.sum() * 0
            loss = mask_loss + binary_loss + .1 * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss))
        validation = evaluate(
            model, datasets["validation"], eval_loaders["validation"], device
        )
        score = (
            validation["stall"]["balanced_accuracy"]
            + validation["failed_acquisition"]["balanced_accuracy"]
            + validation["effective_progress"]["balanced_accuracy"]
        )
        history.append({
            "epoch": epoch, "loss": float(np.mean(losses)),
            "validation": validation,
        })
        print(
            f"epoch={epoch} loss={np.mean(losses):.4f} score={score:.4f} "
            f"iou={validation['mask_iou']}", flush=True,
        )
        if best is None or score > best["score"]:
            best = {
                "score": score, "epoch": epoch,
                "state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }
    model.load_state_dict(best["state"])
    metrics = {
        name: evaluate(model, datasets[name], loader, device)
        for name, loader in eval_loaders.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "task6-execution-slots.safetensors"
    save_file(best["state"], checkpoint)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({
            "schema_version": 1,
            "kind": "three_grounded_roles_action_conditioned_execution_heads",
            "head_order": HEADS,
            "head_positive_counts": {
                name: int(positive_counts[index])
                for index, name in enumerate(HEADS)
            },
            "head_pos_weights": {
                name: float(head_pos_weights[index])
                for index, name in enumerate(HEADS)
            },
            "episode_split": split_ids,
            "best_epoch": best["epoch"],
            "metrics": metrics,
            "history": history,
            "checkpoint": str(checkpoint),
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
