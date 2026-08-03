#!/usr/bin/env python
"""Train a small action-conditioned next-slot predictor on recovery branches.

The default split is by *source natural episode*, not by branch frame: all
counterfactual variants of ep1 train together and ep6 is held out.  The result
is a data-contract smoke test, never a generalization claim from two states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from recovery.models.action_conditioned_slot_dynamics import (  # noqa: E402
    ActionConditionedRoleDynamics,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-source-episodes", default="1")
    parser.add_argument("--validation-source-episodes", default="6")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--horizon", type=int, default=16,
        help="Recovery action-block length and slot prediction horizon.",
    )
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def parse_ids(text: str) -> set[int]:
    return {int(value) for value in text.split(",") if value.strip()}


class DynamicsDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records
        if not records:
            raise ValueError("empty dynamics split")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.records[index]


def load_records(
    slot_dir: Path, source_episodes: set[int], horizon: int
) -> list[dict]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    metadata = json.loads((slot_dir / "metadata.json").read_text())
    records = []
    for branch in metadata["branches"]:
        if int(branch["source_episode_id"]) not in source_episodes:
            continue
        tensors = load_file(branch["output"])
        roles = tensors["role_slots"].float()
        actions = tensors["recovery_actions"].float()
        raw_rows = [
            json.loads(line)
            for line in Path(branch["source"]).read_text().splitlines()
            if line.strip()
        ]
        eef = torch.tensor([
            row["online_observation"]["proprio"]["robot0_eef_pos"]
            for row in raw_rows
        ], dtype=torch.float32)
        if not (len(roles) == len(actions) == len(eef)):
            raise ValueError(f"misaligned enrichment: {branch['output']}")
        for timestep in range(len(roles) - horizon):
            records.append({
                "roles": roles[timestep],
                "action": actions[timestep:timestep + horizon],
                "eef": eef[timestep],
                "next_roles": roles[timestep + horizon],
            })
    return records


def loss_and_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    flat_predicted = predicted.flatten(1)
    flat_target = target.flatten(1)
    cosine = 1 - F.cosine_similarity(flat_predicted, flat_target, dim=-1).mean()
    delta = F.smooth_l1_loss(predicted, target)
    return {"total": cosine + delta, "cosine": cosine, "smooth_l1": delta}


@torch.inference_mode()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "cosine": 0.0, "smooth_l1": 0.0}
    identity = {"cosine": 0.0, "smooth_l1": 0.0}
    count = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        predicted = model(batch["roles"], batch["action"], batch["eef"])
        values = loss_and_metrics(predicted, batch["next_roles"])
        baseline = loss_and_metrics(batch["roles"], batch["next_roles"])
        for key in totals:
            totals[key] += float(values[key]) * len(predicted)
        for key in identity:
            identity[key] += float(baseline[key]) * len(predicted)
        count += len(predicted)
    model.train()
    return {
        **{f"validation_{key}": value / count for key, value in totals.items()},
        **{f"identity_{key}": value / count for key, value in identity.items()},
        "validation_count": count,
    }


def main() -> None:
    args = arguments()
    torch.manual_seed(args.seed)
    train_ids = parse_ids(args.train_source_episodes)
    validation_ids = parse_ids(args.validation_source_episodes)
    if train_ids & validation_ids:
        raise ValueError("train and validation source episodes overlap")
    train = DynamicsDataset(load_records(args.slot_dir, train_ids, args.horizon))
    validation = DynamicsDataset(
        load_records(args.slot_dir, validation_ids, args.horizon)
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(validation, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionConditionedRoleDynamics(
        hidden_dim=args.hidden_dim, action_horizon=args.horizon
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    iterator = iter(train_loader)
    history = []
    best = None
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        predicted = model(batch["roles"], batch["action"], batch["eef"])
        losses = loss_and_metrics(predicted, batch["next_roles"])
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            report = {"step": step, **{f"train_{key}": float(value) for key, value in losses.items()}, **evaluate(model, validation_loader, device)}
            history.append(report)
            if best is None or report["validation_total"] < best["validation_total"]:
                best = {
                    **report,
                    "state": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                }
            print(json.dumps(report), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "action-conditioned-role-dynamics.safetensors"
    save_file(best.pop("state"), checkpoint)
    report = {
        "schema_version": 1,
        "kind": "task6_recovery_action_conditioned_role_dynamics_smoke",
        "smoke_only": True,
        "not_a_generalization_claim": True,
        "input_contract": "frozen role slots + recovery action + observable EEF position",
        "target": "frozen role slots after the action block",
        "action_horizon": args.horizon,
        "split_by_source_episode": {"train": sorted(train_ids), "validation": sorted(validation_ids)},
        "train_transitions": len(train),
        "validation_transitions": len(validation),
        "best": best,
        "history": history,
        "checkpoint": str(checkpoint),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
