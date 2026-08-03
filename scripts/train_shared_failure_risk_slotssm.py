#!/usr/bin/env python
"""Train the current-failure half of the unified SlotSSM recovery model.

The task encoders remain frozen and task-specific.  After canonical role
reordering, one Mamba2 temporal backbone and risk head are trained on both
tasks.  Teacher failure flags are targets only, never model inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from recovery.models.shared_risk_slotssm import SharedRiskSlotSSM  # noqa: E402

ROLE_PERMUTATION = {6: (2, 1, 0), 9: (0, 1, 2)}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task6-root", type=Path, required=True)
    parser.add_argument("--task9-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-task6-episodes", default="7,9")
    parser.add_argument("--validation-task9-episodes", default="1,6")
    parser.add_argument("--window-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--backend", choices=("mamba2", "gru"), default="mamba2")
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def parse_ids(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item}


def risk_label(task_id: int, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    if task_id == 6:
        # targets = progress, stall, controlled, failed-acquisition, slip, collateral.
        return (tensors["targets"][:, 3] > .5) | (tensors["targets"][:, 4] > .5)
    return tensors["teacher_failed_acquisition"] | tensors["teacher_transport_drop"]


def observable_inputs(task_id: int, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    permutation = torch.tensor(ROLE_PERMUTATION[task_id], dtype=torch.long)
    slots = tensors["role_slots"][:, permutation].reshape(len(tensors["role_slots"]), -1)
    geometry = tensors["role_geometry"][:, permutation].reshape(len(slots), -1)
    actions = tensors["actions"] if task_id == 6 else tensors["executed_actions"]
    aperture = (
        tensors["robot_features"][:, 3:4]
        if task_id == 6 else tensors["gripper_aperture"][:, None]
    )
    features = torch.cat((slots, geometry, actions, aperture), dim=-1).float()
    if features.shape[1] != 305:
        raise ValueError(f"unexpected observable feature shape {tuple(features.shape)}")
    return features


class RiskWindows(Dataset):
    def __init__(self, roots: dict[int, Path], validation: dict[int, set[int]],
                 split: str, window_length: int) -> None:
        self.windows = []
        for task_id, root in roots.items():
            for path in sorted(root.glob(f"task{task_id}-episode*-execution-roles.safetensors")):
                episode = int(path.stem.split("episode", 1)[1].split("-", 1)[0])
                is_validation = episode in validation[task_id]
                if (split == "validation") != is_validation:
                    continue
                tensors = load_file(path)
                features = observable_inputs(task_id, tensors)
                labels = risk_label(task_id, tensors).float()
                if len(features) < window_length:
                    continue
                for endpoint in range(window_length - 1, len(features)):
                    self.windows.append({
                        "context": features[endpoint - window_length + 1:endpoint + 1],
                        "task_index": 0 if task_id == 6 else 1,
                        "label": labels[endpoint],
                        "episode": episode,
                        "task_id": task_id,
                    })
        if not self.windows:
            raise ValueError(f"no {split} windows")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        return self.windows[index]


def collate(batch: list[dict]) -> dict:
    return {
        "context": torch.stack([item["context"] for item in batch]),
        "task_index": torch.tensor([item["task_index"] for item in batch]),
        "label": torch.tensor([item["label"] for item in batch]),
        "episode": [item["episode"] for item in batch],
        "task_id": [item["task_id"] for item in batch],
    }


@torch.inference_mode()
def evaluate(model, loader, device: torch.device) -> dict:
    model.eval()
    rows = []
    for batch in loader:
        logits = model(batch["context"].to(device), batch["task_index"].to(device))["current_failure_logits"][:, -1]
        rows.extend(zip(batch["task_id"], batch["episode"], batch["label"].tolist(), logits.sigmoid().cpu().tolist()))
    labels = torch.tensor([row[2] for row in rows])
    predictions = torch.tensor([row[3] >= .5 for row in rows])
    metrics = {}
    for name, mask in {
        "all": torch.ones(len(rows), dtype=torch.bool),
        "task6": torch.tensor([row[0] == 6 for row in rows]),
        "task9": torch.tensor([row[0] == 9 for row in rows]),
    }.items():
        target, predicted = labels[mask].bool(), predictions[mask]
        tp = int((target & predicted).sum())
        fp = int((~target & predicted).sum())
        fn = int((target & ~predicted).sum())
        metrics[name] = {
            "windows": int(mask.sum()),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "accuracy": float((target == predicted).float().mean()),
        }
    # An episode is a positive alert if any window crosses threshold. This
    # exposes false alarms that frame accuracy can hide.
    episodes = {}
    for task_id, episode, label, score in rows:
        key = f"task{task_id}-ep{episode}"
        item = episodes.setdefault(key, {"target": False, "alert": False, "max_score": 0.})
        item["target"] = item["target"] or bool(label)
        item["alert"] = item["alert"] or score >= .5
        item["max_score"] = max(item["max_score"], score)
    metrics["episodes"] = episodes
    return metrics


def main() -> None:
    args = arguments()
    torch.manual_seed(args.seed)
    validation = {6: parse_ids(args.validation_task6_episodes), 9: parse_ids(args.validation_task9_episodes)}
    roots = {6: args.task6_root, 9: args.task9_root}
    train = RiskWindows(roots, validation, "train", args.window_length)
    heldout = RiskWindows(roots, validation, "validation", args.window_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SharedRiskSlotSSM(backend=args.backend).to(device)
    positive = sum(float(item["label"]) for item in train.windows)
    pos_weight = torch.tensor([(len(train) - positive) / max(positive, 1)], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator, collate_fn=collate)
    validation_loader = DataLoader(heldout, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    iterator, best, history = iter(train_loader), -1., []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        logits = model(batch["context"].to(device), batch["task_index"].to(device))["current_failure_logits"][:, -1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["label"].to(device), pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            report = evaluate(model, validation_loader, device)
            row = {"step": step, "train_bce": float(loss), "validation": report}
            history.append(row)
            print(json.dumps(row), flush=True)
            score = report["all"]["recall"] + report["all"]["precision"]
            if score > best:
                best = score
                save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, args.output_dir / "shared-risk-slotssm-best.safetensors")
    (args.output_dir / "report.json").write_text(json.dumps({
        "kind": "shared_current_failure_slotssm_smoke",
        "smoke_only": True,
        "input_contract": "canonical frozen RGB role slots + observable geometry + executed actions + gripper aperture",
        "teacher_contract": "failed acquisition/slip/drop labels only",
        "task_specific_components": "frozen visual slot encoders and two input adapters",
        "shared_components": "Mamba2 temporal backbone and current-failure risk head",
        "train_windows": len(train), "validation_windows": len(heldout),
        "validation_episodes": {str(task): sorted(values) for task, values in validation.items()},
        "history": history,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
