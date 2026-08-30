#!/usr/bin/env python3
"""Causal acceptance test: does v3 future prediction use the action chunk?"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.data.v3_sequence_features import load
from recovery.models.v3_action_aware_recovery import V3ActionAwareRecoveryPolicy, V3ModelConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Training checkpoint with FutureSlotPredictor, not deploy checkpoint.")
    parser.add_argument("--config", type=Path, help="Defaults to config.json beside checkpoint.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("batch-size must be >=2 for an action-shuffle test")
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    config_path = args.config or args.checkpoint.parent / "config.json"
    config = V3ModelConfig(**json.loads(config_path.read_text())["model"])
    model = V3ActionAwareRecoveryPolicy(config).to(args.device).eval()
    loaded = model.load_state_dict(load_file(str(args.checkpoint)), strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("v3 training checkpoint did not load strictly")
    features = load(args.features)
    correct_total = shuffled_total = 0.0
    batches = 0
    with torch.inference_mode():
        for start in range(0, features.num_sequences - 1, args.batch_size):
            index = torch.arange(start, min(start + args.batch_size, features.num_sequences))
            if index.numel() < 2:
                continue
            batch = features.index(index).to(args.device)
            output = model.forward_sequence(
                visual_tokens=batch.visual_tokens, action_hidden=batch.action_hidden,
                proprio=batch.proprio, actions=batch.actions,
            )
            # Keep S_n fixed and only replace A_n: this isolates the exact
            # action-to-future-slot path required by the v3 claim.
            shuffled_actions = batch.actions.roll(1, dims=0)
            shuffled_future = model.future_predictor(
                output["slots"][:, :-1].reshape(-1, config.num_slots, config.slot_dim),
                shuffled_actions[:, :-1].reshape(-1, config.action_chunk, config.action_dim),
            ).view_as(output["future_slots"])
            target = output["slots"][:, 1:]
            correct_total += float(F.mse_loss(output["future_slots"].float(), target.float()))
            shuffled_total += float(F.mse_loss(shuffled_future.float(), target.float()))
            batches += 1
    if not batches:
        raise RuntimeError("need at least two feature sequences")
    correct, shuffled = correct_total / batches, shuffled_total / batches
    report = {
        "kind": "v3_action_conditioning_acceptance", "features": str(args.features),
        "checkpoint": str(args.checkpoint), "correct_action_future_mse": correct,
        "shuffled_action_future_mse": shuffled,
        "shuffle_gap": shuffled - correct,
        "passes_directional_check": shuffled > correct,
        "limitation": "This is necessary but not sufficient; report matched simulator-state counterfactual results separately.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
