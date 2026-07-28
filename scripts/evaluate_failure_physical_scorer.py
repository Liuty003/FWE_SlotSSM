#!/usr/bin/env python
"""Evaluate the failure-aware physical scorer on held-out tasks and episodes."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from scripts.train_failure_physical_scorer import FailurePhysicalScorer, evaluate, load_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-id", type=int, default=9)
    parser.add_argument("--episode-ids", default="0,1,2,3,4")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    episodes = [int(value) for value in args.episode_ids.split(",") if value]
    data = load_split(Path(args.data_dir), [args.task_id], episodes)
    state = load_file(args.checkpoint)
    model = FailurePhysicalScorer(
        torch.zeros(7), torch.ones(7), torch.zeros(7), torch.ones(7)
    )
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics = evaluate(model, data, device)
    result = {
        "task_id": args.task_id, "episode_ids": episodes,
        "num_forks": int(len(data["before_slots"])), "metrics": metrics,
        "units": {
            "physical_mae": "centimeters",
            "accuracy_and_oracle_ratio": "percent",
            "auc": "unitless, 0.5=random and 1.0=perfect",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
