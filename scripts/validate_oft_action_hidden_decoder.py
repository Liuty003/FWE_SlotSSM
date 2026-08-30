#!/usr/bin/env python3
"""Verify that a frozen OFT continuous action head decodes edited hidden states.

This is the architectural admission check for V4.  It does not load a VLA or
run a simulator: saved OFT action hidden states are sufficient to prove that
the exact frozen action head accepts them, that a zero residual is a strict
no-op, and that a bounded hidden perturbation changes decoded actions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--action-head", type=Path, required=True)
    parser.add_argument("--oft-root", type=Path, default=Path("/data/lty/code/openvla-oft-official"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--perturbation", type=float, default=1e-3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows < 1 or args.perturbation < 0:
        raise ValueError("rows must be positive and perturbation non-negative")
    for path in (args.features, args.action_head, args.oft_root):
        if not path.exists():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(args.oft_root))
    import torch
    from safetensors.torch import load_file
    from prismatic.models.action_heads import L1RegressionActionHead

    payload = load_file(str(args.features))
    hidden = payload["action_hidden"].reshape(-1, *payload["action_hidden"].shape[-2:])[:args.rows]
    if hidden.shape[-2] != 56:
        raise ValueError(f"expected 56 action tokens (8x7), found {hidden.shape[-2]}")
    head = L1RegressionActionHead(input_dim=hidden.shape[-1], hidden_dim=hidden.shape[-1]).to(args.device)
    state = torch.load(args.action_head, map_location="cpu", weights_only=True)
    # OFT distributed training writes action-head checkpoints with a DDP
    # ``module.`` prefix, while inference instantiates the bare head.
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    head.load_state_dict(state)
    head.eval()
    hidden = hidden.to(args.device)
    generator = torch.Generator(device=args.device).manual_seed(20260816)
    with torch.inference_mode():
        nominal = head.predict_action(hidden)
        no_op = head.predict_action(hidden + torch.zeros_like(hidden))
        perturbation = torch.randn(hidden.shape, device=args.device, dtype=hidden.dtype, generator=generator)
        perturbed = head.predict_action(hidden + args.perturbation * perturbation)
    report = {
        "kind": "oft_action_hidden_decoder_admission",
        "features": str(args.features), "action_head": str(args.action_head), "rows": int(hidden.shape[0]),
        "hidden_shape": list(hidden.shape), "perturbation": args.perturbation,
        "no_op_max_abs_action_error": float((nominal - no_op).abs().max().cpu()),
        "perturbation_mean_abs_action_change": float((nominal - perturbed).abs().mean().cpu()),
        "passed_no_op": bool(torch.equal(nominal, no_op)),
        "passed_sensitivity": bool(args.perturbation == 0 or not torch.equal(nominal, perturbed)),
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {args.output}")
    if not report["passed_no_op"] or not report["passed_sensitivity"]:
        raise SystemExit("OFT action-hidden decoder admission failed")


if __name__ == "__main__":
    main()
