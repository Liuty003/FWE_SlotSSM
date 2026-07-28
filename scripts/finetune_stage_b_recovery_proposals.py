#!/usr/bin/env python
"""Single-scene adaptation of Stage-B bowl proposals to recovery states."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_stage_b_bowl_proposals import (
    BowlProposalSegmenter, evaluate, proposal_loss,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--validation-modulus", type=int, default=5)
    parser.add_argument("--validation-remainder", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def make_samples(tensors):
    targets = torch.stack(
        (tensors["target_masks"], tensors["distractor_masks"]), dim=1
    )
    return {
        (9, int(timestep)): (tensors["patches"][index], targets[index])
        for index, timestep in enumerate(tensors["timesteps"].tolist())
    }


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("recovery proposal adaptation requires CUDA")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda:0")
    tensors = load_file(args.data)
    samples = make_samples(tensors)
    keys = sorted(samples)
    validation = [
        key for key in keys
        if key[1] % args.validation_modulus == args.validation_remainder
    ]
    train = [key for key in keys if key not in set(validation)]
    model = BowlProposalSegmenter(256, 8).to(device)
    model.load_state_dict(load_file(args.base_checkpoint), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    for step in range(1, args.steps + 1):
        selected = rng.sample(train, min(args.batch_size, len(train)))
        patches = torch.stack([samples[key][0] for key in selected]).to(device)
        targets = torch.stack([samples[key][1] for key in selected]).to(device).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(patches)
            loss, matched, overlap = proposal_loss(logits, targets, 1.0, 0.5)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step == args.steps or step % args.log_every == 0:
            record = {
                "step": step, "loss": float(loss),
                "matched_loss": float(matched), "soft_overlap": float(overlap),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
    metrics = {
        "train": evaluate(model, train, samples, device, args.batch_size),
        "validation": evaluate(model, validation, samples, device, args.batch_size),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"bowl-proposals-step-{args.steps:06d}.safetensors"
    save_file(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        checkpoint,
    )
    report = {
        "schema_version": 1,
        "base_checkpoint": str(args.base_checkpoint),
        "checkpoint": str(checkpoint),
        "split": {
            "rule": (
                f"validation iff timestep % {args.validation_modulus} "
                f"== {args.validation_remainder}"
            ),
            "num_train": len(train), "num_validation": len(validation),
            "validation_contains_57": (9, 57) in validation,
            "validation_contains_67": (9, 67) in validation,
        },
        "history": history,
        "metrics": metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
