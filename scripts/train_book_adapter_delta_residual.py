#!/usr/bin/env python3
"""Distil a safe, state-selective action delta between two recovery adapters.

The base and donor adapters are frozen.  The learned residual consumes only the
online recovery contract and is explicitly trained to be a no-op on base-success
states.  This avoids task-id routing and avoids unsafe parameter interpolation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.models.neural_recovery_residual import NeuralRecoveryResidual
from recovery.models.openvla_oft_slotssm_adapter import OpenVLAOFTSlotSSMAdapter
from scripts.smoke_unified_oft_slotssm_bridge import DEFAULT_OFT_ROOT, dynamic_module, load_torch_state

DEFAULT_OFT_PLUS = Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata")
REQUIRED = {"action_hidden", "temporal_slots", "proprio", "previous_executed_action", "recovery_mode"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-features", type=Path, required=True,
                        help="States where the donor adapter has verified online benefit.")
    parser.add_argument("--negative-features", type=Path, required=True,
                        help="Base-success states on which the residual must be a no-op.")
    parser.add_argument("--base-adapter", type=Path, required=True)
    parser.add_argument("--donor-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oft-root", type=Path, default=DEFAULT_OFT_ROOT)
    parser.add_argument("--oft-checkpoint", type=Path, default=DEFAULT_OFT_PLUS)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gate-weight", type=float, default=0.5)
    parser.add_argument("--negative-delta-weight", type=float, default=0.8)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.positive_features, args.negative_features, args.base_adapter, args.donor_adapter):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.oft_root.is_dir() or not args.oft_checkpoint.is_dir():
        raise FileNotFoundError("OFT root/checkpoint")
    if min(args.steps, args.batch_size) < 1 or args.lr <= 0 or args.gate_weight < 0 or args.negative_delta_weight < 0:
        raise ValueError("steps/batch-size/lr must be positive and loss weights non-negative")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    import prismatic
    import prismatic.vla

    prismatic.__path__.append(str(args.oft_root / "prismatic"))
    prismatic.vla.__path__.append(str(args.oft_root / "prismatic" / "vla"))
    torch.manual_seed(29)
    device = torch.device("cuda")
    positive, negative = load_file(str(args.positive_features)), load_file(str(args.negative_features))
    for name, payload in (("positive", positive), ("negative", negative)):
        missing = REQUIRED - set(payload)
        if missing:
            raise ValueError(f"{name} is missing {sorted(missing)}")
        if not payload["action_hidden"].shape[0]:
            raise ValueError(f"{name} is empty")

    components = dynamic_module("oft_action_heads_delta_residual", args.oft_root / "prismatic" / "models" / "action_heads.py")
    normal_head = components.L1RegressionActionHead(4096, 4096, 7)
    normal_head.load_state_dict(load_torch_state(args.oft_checkpoint / "action_head--150000_checkpoint.pt", torch))

    def adapter(checkpoint: Path) -> OpenVLAOFTSlotSSMAdapter:
        value = OpenVLAOFTSlotSSMAdapter(normal_head, action_hidden_dim=4096, slot_dim=512).to(device=device, dtype=torch.bfloat16)
        value.load_state_dict(load_file(str(checkpoint)), strict=False)
        return value.eval().requires_grad_(False)

    base, donor = adapter(args.base_adapter), adapter(args.donor_adapter)
    residual = NeuralRecoveryResidual().to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(residual.parameters(), lr=args.lr, weight_decay=1e-4)

    def move(payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
                for key, value in payload.items()}

    positive, negative = move(positive), move(negative)

    @torch.inference_mode()
    def adapter_action(model: OpenVLAOFTSlotSSMAdapter, payload: dict[str, torch.Tensor], index: torch.Tensor) -> torch.Tensor:
        output = model(payload["action_hidden"][index], payload["temporal_slots"][index],
                       proprio=payload["proprio"][index], previous_action=payload["previous_executed_action"][index],
                       recovery_mode_override=payload["recovery_mode"][index],
                       disable_phase_conditioning=True, geometry_conditioning=True)
        return output["recovery_actions"][:, 0].float()

    def residual_output(payload: dict[str, torch.Tensor], index: torch.Tensor) -> dict[str, torch.Tensor]:
        return residual(payload["action_hidden"][index], payload["temporal_slots"][index],
                        payload["proprio"][index], payload["previous_executed_action"][index],
                        payload["recovery_mode"][index])

    def objective(pos_index: torch.Tensor, neg_index: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pos_out, neg_out = residual_output(positive, pos_index), residual_output(negative, neg_index)
        # Match the *executed* bounded residual semantics of the runner.
        target_delta = (adapter_action(donor, positive, pos_index) - adapter_action(base, positive, pos_index)).clamp(-0.35, 0.35)
        delta = F.smooth_l1_loss(pos_out["delta"].float(), target_delta)
        preserve = F.smooth_l1_loss(neg_out["delta"].float(), torch.zeros_like(neg_out["delta"].float()))
        logits = torch.cat((pos_out["gate_logit"], neg_out["gate_logit"])).float()
        labels = torch.cat((torch.ones_like(pos_out["gate_logit"]), torch.zeros_like(neg_out["gate_logit"]))).float()
        gate = F.binary_cross_entropy_with_logits(logits, labels)
        return delta + args.negative_delta_weight * preserve + args.gate_weight * gate, {
            "delta": delta, "preserve": preserve, "gate": gate,
            "positive_probability": torch.sigmoid(pos_out["gate_logit"].float()).mean(),
            "negative_probability": torch.sigmoid(neg_out["gate_logit"].float()).mean(),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    full_pos = torch.arange(positive["action_hidden"].shape[0], device=device)
    full_neg = torch.arange(negative["action_hidden"].shape[0], device=device)
    for step in range(1, args.steps + 1):
        residual.train()
        pos_index = torch.randint(len(full_pos), (args.batch_size // 2,), device=device)
        neg_index = torch.randint(len(full_neg), (args.batch_size - len(pos_index),), device=device)
        loss, _ = objective(pos_index, neg_index)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(residual.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            residual.eval()
            with torch.no_grad():
                value, metrics = objective(full_pos, full_neg)
            event = {"step": step, "train_loss": float(loss.detach().cpu()), "val_loss": float(value.cpu()),
                     **{key: float(item.cpu()) for key, item in metrics.items()}}
            history.append(event)
            print(json.dumps(event), flush=True)
            if event["val_loss"] < best:
                best = event["val_loss"]
                save_file({key: value.detach().float().cpu().contiguous() for key, value in residual.state_dict().items()},
                          str(args.output_dir / "residual_best.safetensors"))
    (args.output_dir / "report.json").write_text(json.dumps({
        "kind": "book_adapter_delta_residual_v1", "base_adapter": str(args.base_adapter),
        "donor_adapter": str(args.donor_adapter), "positive_features": str(args.positive_features),
        "negative_features": str(args.negative_features), "positive_samples": int(len(full_pos)),
        "negative_samples": int(len(full_neg)), "best_val_loss": best, "history": history,
        "online_input_contract": ["OFT hidden", "SlotSSM", "proprio", "previous action", "learned candidate mode"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
