#!/usr/bin/env python
"""Train an RGB-only development baseline for the task-6 subgoal predicate."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


def parse_ids(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate episode id in {value!r}")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-episodes", default="0,1,2,3,4,5")
    parser.add_argument("--validation-episodes", default="6,8")
    parser.add_argument("--test-episodes", default="7,9")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--confirm-frames", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def split_rows(
    rows: list[dict], split_ids: dict[str, tuple[int, ...]]
) -> dict[str, list[dict]]:
    sets = {name: set(ids) for name, ids in split_ids.items()}
    names = tuple(sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f"{left}/{right} episode leakage: {sorted(overlap)}")
    available = {int(row["episode_id"]) for row in rows}
    requested = set().union(*sets.values())
    missing = requested - available
    if missing:
        raise ValueError(f"episodes absent from manifest: {sorted(missing)}")
    return {
        name: [row for row in rows if int(row["episode_id"]) in ids]
        for name, ids in sets.items()
    }


class FrameDataset(Dataset):
    def __init__(self, rows: list[dict], image_size: int, augment: bool):
        self.rows = rows
        operations = [transforms.Resize((image_size, image_size))]
        if augment:
            operations += [
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08)
            ]
        operations += [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
        self.transform = transforms.Compose(operations)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["image"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(float(row["mug_on_plate"])), index


class SmallRgbClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (3, 32, 64, 128, 192)
        blocks = []
        for input_channels, output_channels in zip(channels, channels[1:]):
            blocks += [
                nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.GELU(),
            ]
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.15), nn.Linear(channels[-1], 1)
        )

    def forward(self, images):
        return self.head(self.encoder(images)).squeeze(-1)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = scores >= threshold
    positive = labels == 1
    negative = ~positive
    tp = int((predictions & positive).sum())
    tn = int((~predictions & negative).sum())
    fp = int((predictions & negative).sum())
    fn = int((~predictions & positive).sum())
    return {
        "frames": int(len(labels)),
        "positive_frames": int(positive.sum()),
        "accuracy": float((tp + tn) / max(len(labels), 1)),
        "balanced_accuracy": float(
            0.5
            * (
                tp / max(int(positive.sum()), 1)
                + tn / max(int(negative.sum()), 1)
            )
        ),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "false_positives": fp,
        "false_negatives": fn,
    }


def confirmed_trigger(values: np.ndarray, confirm_frames: int) -> int | None:
    streak = 0
    for index, value in enumerate(values):
        streak = streak + 1 if bool(value) else 0
        if streak >= confirm_frames:
            return index
    return None


@torch.inference_mode()
def evaluate(
    model, rows: list[dict], loader: DataLoader, device, threshold, confirm_frames
) -> dict:
    model.eval()
    scores = np.zeros(len(rows), dtype=np.float32)
    labels = np.zeros(len(rows), dtype=np.int64)
    for images, batch_labels, indices in loader:
        probabilities = torch.sigmoid(model(images.to(device))).cpu().numpy()
        scores[indices.numpy()] = probabilities
        labels[indices.numpy()] = batch_labels.numpy().astype(np.int64)
    report = binary_metrics(labels, scores, threshold)
    grouped = defaultdict(list)
    for row, label, score in zip(rows, labels, scores):
        grouped[int(row["episode_id"])].append(
            (int(row["timestep"]), bool(label), float(score))
        )
    episode_reports = []
    for episode, sequence in sorted(grouped.items()):
        sequence.sort()
        episode_labels = np.asarray([item[1] for item in sequence])
        episode_predictions = np.asarray([item[2] >= threshold for item in sequence])
        predicted = confirmed_trigger(episode_predictions, confirm_frames)
        expected = confirmed_trigger(episode_labels, confirm_frames)
        episode_reports.append(
            {
                "episode_id": episode,
                "frames": len(sequence),
                "positive_frames": int(episode_labels.sum()),
                "expected_trigger_index": expected,
                "predicted_trigger_index": predicted,
                "trigger_delay_frames": (
                    predicted - expected
                    if predicted is not None and expected is not None
                    else None
                ),
                "false_trigger": predicted is not None and expected is None,
                "missed_trigger": predicted is None and expected is not None,
            }
        )
    report["episodes"] = episode_reports
    return report


def main() -> None:
    args = arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    split_ids = {
        "train": parse_ids(args.train_episodes),
        "validation": parse_ids(args.validation_episodes),
        "test": parse_ids(args.test_episodes),
    }
    rows = load_manifest(args.data_root / "manifest.jsonl")
    splits = split_rows(rows, split_ids)
    datasets = {
        name: FrameDataset(values, args.image_size, augment=name == "train")
        for name, values in splits.items()
    }
    labels = np.asarray([float(row["mug_on_plate"]) for row in splits["train"]])
    class_counts = np.bincount(labels.astype(np.int64), minlength=2)
    weights = np.asarray([1.0 / max(class_counts[int(label)], 1) for label in labels])
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(weights, len(weights), generator=generator),
        num_workers=args.workers,
        pin_memory=True,
    )
    eval_loaders = {
        name: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
        for name, dataset in datasets.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallRgbClassifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    best_state = None
    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, batch_labels, _ in train_loader:
            logits = model(images.to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, batch_labels.to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        validation = evaluate(
            model, splits["validation"], eval_loaders["validation"], device,
            args.threshold, args.confirm_frames,
        )
        score = validation["balanced_accuracy"]
        history.append(
            {"epoch": epoch, "loss": float(np.mean(losses)), "validation": validation}
        )
        print(
            f"epoch={epoch} loss={np.mean(losses):.4f} "
            f"val_bal_acc={score:.4f} val_fp={validation['false_positives']}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    assert best_state is not None
    model.load_state_dict(best_state)
    metrics = {
        name: evaluate(
            model, splits[name], loader, device, args.threshold, args.confirm_frames
        )
        for name, loader in eval_loaders.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "task6-rgb-state-baseline.safetensors"
    save_file({key: value.contiguous() for key, value in best_state.items()}, checkpoint)
    result = {
        "schema_version": 1,
        "kind": "task6_rgb_only_subgoal_predicate_development_baseline",
        "input_contract": "raw agent-view RGB only; no simulator state",
        "label_contract": "simulator On(porcelain_mug_1, plate_1), supervision only",
        "episode_split": split_ids,
        "threshold": args.threshold,
        "confirm_frames": args.confirm_frames,
        "best_validation_balanced_accuracy": best_score,
        "metrics": metrics,
        "history": history,
        "checkpoint": str(checkpoint.resolve()),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
