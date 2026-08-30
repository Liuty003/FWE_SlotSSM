#!/usr/bin/env python3
"""Held-out offline action evaluation for a v2 local-recovery adapter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_unified_oft_slotssm_bridge import DEFAULT_OFT_ROOT, dynamic_module, load_torch_state  # noqa: E402
from recovery.models.openvla_oft_slotssm_adapter import OpenVLAOFTSlotSSMAdapter  # noqa: E402

DEFAULT_OFT_PLUS = Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oft-root", type=Path, default=DEFAULT_OFT_ROOT)
    parser.add_argument("--oft-checkpoint", type=Path, default=DEFAULT_OFT_PLUS)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument(
        "--teacher-phase-override", action="store_true",
        help="Oracle-only offline diagnostic: route actions through the labeled teacher phase expert.",
    )
    return parser.parse_args()


def weighted_loss(prediction, target, functional):
    return (functional.smooth_l1_loss(prediction[..., :6].float(), target[..., :6].float())
            + 2.0 * functional.smooth_l1_loss(prediction[..., 6:].float(), target[..., 6:].float()))


def main() -> None:
    args = parse_args()
    for path in (args.features, args.adapter, args.oft_root, args.oft_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import load_file
    import prismatic
    import prismatic.vla
    prismatic.__path__.append(str(args.oft_root / "prismatic"))
    prismatic.vla.__path__.append(str(args.oft_root / "prismatic" / "vla"))
    device = torch.device("cuda")
    payload = load_file(str(args.features), device="cpu")
    needed = {"action_hidden", "temporal_slots", "proprio", "previous_executed_action", "recovery_target"}
    if needed - set(payload):
        raise ValueError(f"not a recovery-action feature file: missing {sorted(needed - set(payload))}")
    action_components = dynamic_module("oft_action_heads_eval_v2", args.oft_root / "prismatic" / "models" / "action_heads.py")
    normal_head = action_components.L1RegressionActionHead(4096, 4096, 7)
    normal_head.load_state_dict(load_torch_state(args.oft_checkpoint / "action_head--150000_checkpoint.pt", torch))
    adapter = OpenVLAOFTSlotSSMAdapter(normal_head, action_hidden_dim=4096, slot_dim=512).to(device, dtype=torch.bfloat16)
    missing, unexpected = adapter.load_state_dict(load_file(str(args.adapter)), strict=False)
    expected_missing = {f"normal_action_head.{key}" for key in adapter.normal_action_head.state_dict()}
    expected_missing.update(
        key for key in adapter.state_dict()
        if key.startswith(("phase_action_experts.", "geometry_head."))
    )
    if unexpected or not set(missing).issubset(expected_missing):
        raise RuntimeError(f"unexpected adapter state mismatch: missing={missing}, unexpected={unexpected}")
    adapter.eval()
    values = {key: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
              for key, value in payload.items()}
    with torch.inference_mode():
        if args.teacher_phase_override and "recovery_phase" not in values:
            raise ValueError("teacher phase override requires recovery_phase labels")
        output = adapter(values["action_hidden"], values["temporal_slots"],
                         proprio=values["proprio"], previous_action=values["previous_executed_action"],
                         recovery_phase_override=(values.get("recovery_phase") if args.teacher_phase_override else None))
        normal_loss = float(weighted_loss(output["normal_actions"], values["recovery_target"], functional).cpu())
        recovery_loss = float(weighted_loss(output["recovery_actions"], values["recovery_target"], functional).cpu())
    report = {"kind": "book_regrasp_v2_held_out_action_evaluation", "features": str(args.features),
              "adapter": str(args.adapter), "samples": int(values["action_hidden"].shape[0]),
              "normal_head_loss": normal_loss, "recovery_head_loss": recovery_loss,
              "relative_improvement": (normal_loss - recovery_loss) / normal_loss,
              "teacher_phase_override": args.teacher_phase_override,
              "limitation": "offline action imitation, not simulator recovery success"}
    if "recovery_phase" in values:
        phase_metrics = {}
        for phase in range(5):
            mask = values["recovery_phase"].eq(phase)
            if not mask.any():
                continue
            phase_metrics[str(phase)] = {
                "samples": int(mask.sum().item()),
                "normal_head_loss": float(weighted_loss(
                    output["normal_actions"][mask], values["recovery_target"][mask], functional
                ).cpu()),
                "recovery_head_loss": float(weighted_loss(
                    output["recovery_actions"][mask], values["recovery_target"][mask], functional
                ).cpu()),
            }
        report["phase_metrics"] = phase_metrics
        report["phase_classification_accuracy"] = float(
            output["recovery_phase_logits"].argmax(dim=-1).eq(values["recovery_phase"]).float().mean().cpu()
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
