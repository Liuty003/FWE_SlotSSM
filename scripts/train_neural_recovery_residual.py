#!/usr/bin/env python3
"""Train a bounded learned residual on on-policy recovery corrections."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.smoke_unified_oft_slotssm_bridge import DEFAULT_OFT_ROOT, dynamic_module, load_torch_state
from recovery.models.openvla_oft_slotssm_adapter import OpenVLAOFTSlotSSMAdapter
from recovery.models.neural_recovery_residual import NeuralRecoveryResidual

DEFAULT_OFT_PLUS = Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata")


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-features", type=Path, required=True)
    parser.add_argument("--negative-features", type=Path, required=True)
    parser.add_argument("--base-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oft-root", type=Path, default=DEFAULT_OFT_ROOT)
    parser.add_argument("--oft-checkpoint", type=Path, default=DEFAULT_OFT_PLUS)
    parser.add_argument("--positive-mode", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gate-weight", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    config = args()
    for path in (config.positive_features, config.negative_features, config.base_adapter, config.oft_root, config.oft_checkpoint):
        if not path.is_file() and path not in (config.oft_root, config.oft_checkpoint):
            raise FileNotFoundError(path)
    if not 0 <= config.positive_mode < 5:
        raise ValueError("positive-mode must be in [0,4]")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(config.gpu))
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import load_file, save_file
    import prismatic
    import prismatic.vla
    prismatic.__path__.append(str(config.oft_root / "prismatic"))
    prismatic.vla.__path__.append(str(config.oft_root / "prismatic" / "vla"))
    device = torch.device("cuda")
    torch.manual_seed(17)
    positive, negative = load_file(str(config.positive_features)), load_file(str(config.negative_features))
    required = {"action_hidden", "temporal_slots", "proprio", "previous_executed_action", "recovery_mode", "recovery_target"}
    for name, payload in (("positive", positive), ("negative", negative)):
        missing = required - set(payload)
        if missing:
            raise ValueError(f"{name} missing {sorted(missing)}")
    keep = positive["recovery_mode"].eq(config.positive_mode)
    positive = {key: value[keep] for key, value in positive.items()}
    if not positive["action_hidden"].shape[0]:
        raise RuntimeError("no positive rows for selected mode")
    # The successful rollouts are mode xyp25, and should remain unchanged.
    negative_keep = negative["recovery_mode"].eq(config.positive_mode)
    negative = {key: value[negative_keep] for key, value in negative.items()}
    if not negative["action_hidden"].shape[0]:
        raise RuntimeError("no negative rows for selected mode")
    actions = dynamic_module("oft_action_heads_residual", config.oft_root / "prismatic" / "models" / "action_heads.py")
    head = actions.L1RegressionActionHead(4096, 4096, 7)
    head.load_state_dict(load_torch_state(config.oft_checkpoint / "action_head--150000_checkpoint.pt", torch))
    base = OpenVLAOFTSlotSSMAdapter(head, action_hidden_dim=4096, slot_dim=512).to(device=device, dtype=torch.bfloat16)
    base.load_state_dict(load_file(str(config.base_adapter)), strict=False)
    base.eval().requires_grad_(False)
    residual = NeuralRecoveryResidual().to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(residual.parameters(), lr=config.lr, weight_decay=1e-4)

    def move(payload):
        return {key: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype) for key, value in payload.items()}
    positive, negative = move(positive), move(negative)

    @torch.inference_mode()
    def base_action(payload, index):
        out = base(payload["action_hidden"][index], payload["temporal_slots"][index],
                   proprio=payload["proprio"][index], previous_action=payload["previous_executed_action"][index],
                   recovery_mode_override=payload["recovery_mode"][index],
                   disable_phase_conditioning=True, geometry_conditioning=True)
        return out["recovery_actions"][:, 0].float()

    def prediction(payload, index):
        return residual(payload["action_hidden"][index], payload["temporal_slots"][index], payload["proprio"][index],
                        payload["previous_executed_action"][index], payload["recovery_mode"][index])

    config.output_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for step in range(1, config.steps + 1):
        residual.train()
        pos_index = torch.randint(positive["action_hidden"].shape[0], (config.batch_size // 2,), device=device)
        neg_index = torch.randint(negative["action_hidden"].shape[0], (config.batch_size - len(pos_index),), device=device)
        pos_pred, neg_pred = prediction(positive, pos_index), prediction(negative, neg_index)
        pos_base, neg_base = base_action(positive, pos_index), base_action(negative, neg_index)
        pos_delta = positive["recovery_target"][pos_index, 0].float() - pos_base
        delta_loss = functional.smooth_l1_loss(pos_pred["delta"].float(), pos_delta)
        # Negative rows are successful v8 recovery states: force an explicit
        # zero correction as well as the negative classification label.
        preserve_loss = functional.smooth_l1_loss(neg_pred["delta"].float(), torch.zeros_like(neg_pred["delta"].float()))
        logits = torch.cat((pos_pred["gate_logit"], neg_pred["gate_logit"])).float()
        labels = torch.cat((torch.ones_like(pos_pred["gate_logit"]), torch.zeros_like(neg_pred["gate_logit"]))).float()
        gate_loss = functional.binary_cross_entropy_with_logits(logits, labels)
        loss = delta_loss + 0.4 * preserve_loss + config.gate_weight * gate_loss
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(residual.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 100 == 0 or step == config.steps:
            residual.eval()
            with torch.no_grad():
                all_pos, all_neg = prediction(positive, torch.arange(positive["action_hidden"].shape[0], device=device)), prediction(negative, torch.arange(negative["action_hidden"].shape[0], device=device))
                all_pos_base, all_neg_base = base_action(positive, torch.arange(positive["action_hidden"].shape[0], device=device)), base_action(negative, torch.arange(negative["action_hidden"].shape[0], device=device))
                val_delta = functional.smooth_l1_loss(all_pos["delta"].float(), positive["recovery_target"][:, 0].float() - all_pos_base)
                val_preserve = functional.smooth_l1_loss(all_neg["delta"].float(), torch.zeros_like(all_neg["delta"].float()))
                probability = torch.sigmoid(torch.cat((all_pos["gate_logit"], all_neg["gate_logit"])).float())
                labels = torch.cat((torch.ones_like(all_pos["gate_logit"]), torch.zeros_like(all_neg["gate_logit"]))).float()
                gate_bce = functional.binary_cross_entropy(probability, labels)
                value = float((val_delta + 0.4 * val_preserve + config.gate_weight * gate_bce).cpu())
                event = {"step": step, "train_loss": float(loss.detach().cpu()), "val_loss": value,
                         "val_delta": float(val_delta.cpu()), "val_preserve": float(val_preserve.cpu()), "val_gate_bce": float(gate_bce.cpu()),
                         "positive_gate_mean": float(torch.sigmoid(all_pos["gate_logit"].float()).mean().cpu()),
                         "negative_gate_mean": float(torch.sigmoid(all_neg["gate_logit"].float()).mean().cpu())}
                history.append(event); print(event, flush=True)
                if value < best:
                    best = value
                    save_file({key: value.detach().float().cpu().contiguous() for key, value in residual.state_dict().items()}, str(config.output_dir / "residual_best.safetensors"))
    report = {"positive_features": str(config.positive_features), "negative_features": str(config.negative_features), "base_adapter": str(config.base_adapter),
              "positive_mode": config.positive_mode, "positive_samples": int(positive["action_hidden"].shape[0]), "negative_samples": int(negative["action_hidden"].shape[0]), "history": history}
    (config.output_dir / "report.json").write_text(__import__("json").dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
