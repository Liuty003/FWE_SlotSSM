#!/usr/bin/env python3
"""Train a teacher-supervised slot-preserving target-relative geometry head.

Target pose is an offline simulator label only.  The trained head consumes the
same frozen OFT+/SlotSSM/proprio/action-history features available online and
predicts target position minus EEF position in world coordinates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from recovery.models.openvla_oft_slotssm_adapter import OpenVLAOFTSlotSSMAdapter
from scripts.smoke_unified_oft_slotssm_bridge import DEFAULT_OFT_ROOT, dynamic_module, load_torch_state

DEFAULT_OFT_PLUS = Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata")


def args_() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--test-features", type=Path,
                        help="Optional held-out geometry features. Never used for optimization or checkpoint selection.")
    parser.add_argument("--init-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oft-root", type=Path, default=DEFAULT_OFT_ROOT)
    parser.add_argument("--oft-checkpoint", type=Path, default=DEFAULT_OFT_PLUS)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = args_()
    required_paths = (args.train_features, args.val_features, args.init_adapter, args.oft_root, args.oft_checkpoint)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.test_features is not None and not args.test_features.exists():
        raise FileNotFoundError(args.test_features)
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1:
        raise ValueError("steps, batch-size and eval-every must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import load_file, save_file
    # ``action_heads.py`` imports OFT's separately-versioned VLA constants.
    # Register that source tree before dynamically loading the head, matching
    # the recovery trainer/evaluator import contract.
    import prismatic
    import prismatic.vla
    prismatic.__path__.append(str(args.oft_root / "prismatic"))
    prismatic.vla.__path__.append(str(args.oft_root / "prismatic" / "vla"))

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    train = load_file(str(args.train_features), device="cpu")
    val = load_file(str(args.val_features), device="cpu")
    test = load_file(str(args.test_features), device="cpu") if args.test_features is not None else None
    required = {"action_hidden", "temporal_slots", "proprio", "previous_executed_action", "teacher_target_relative_position"}
    feature_sets = [("train", train), ("val", val)]
    if test is not None:
        feature_sets.append(("test", test))
    for name, payload in feature_sets:
        missing = required - set(payload)
        if missing:
            raise ValueError(f"{name} geometry feature file missing {sorted(missing)}")

    action_components = dynamic_module("oft_action_heads_book_geometry", args.oft_root / "prismatic" / "models" / "action_heads.py")
    normal_head = action_components.L1RegressionActionHead(4096, 4096, 7)
    normal_head.load_state_dict(load_torch_state(args.oft_checkpoint / "action_head--150000_checkpoint.pt", torch))
    adapter = OpenVLAOFTSlotSSMAdapter(normal_head, action_hidden_dim=4096, slot_dim=512).to(device=device, dtype=torch.bfloat16)
    missing, unexpected = adapter.load_state_dict(load_file(str(args.init_adapter)), strict=False)
    allowed_missing = {f"normal_action_head.{key}" for key in adapter.normal_action_head.state_dict()}
    allowed_missing.update(key for key in adapter.state_dict() if key.startswith(("geometry_head.", "geometry_slot_", "spatial_geometry_head.")))
    if unexpected or not set(missing).issubset(allowed_missing):
        raise RuntimeError(f"unexpected init adapter mismatch: missing={missing}, unexpected={unexpected}")
    adapter.requires_grad_(False)
    geometry_modules = (adapter.geometry_slot_projection, adapter.geometry_slot_attention, adapter.spatial_geometry_head)
    for module in geometry_modules:
        module.requires_grad_(True)
    optimizer = torch.optim.AdamW([parameter for module in geometry_modules for parameter in module.parameters()], lr=args.lr, weight_decay=1e-4)

    def move(payload):
        return {key: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
                for key, value in payload.items()}

    train, val = move(train), move(val)
    test = move(test) if test is not None else None

    @torch.inference_mode()
    def evaluate(payload):
        adapter.eval()
        prediction = adapter(
            payload["action_hidden"], payload["temporal_slots"], proprio=payload["proprio"],
            previous_action=payload["previous_executed_action"],
        )["spatial_target_relative_position"].float()
        target = payload["teacher_target_relative_position"].float()
        error = prediction - target
        metrics = {
            "smooth_l1": float(functional.smooth_l1_loss(prediction, target).cpu()),
            "mean_l2_m": float(error.norm(dim=-1).mean().cpu()),
            "mean_abs_xyz_m": error.abs().mean(dim=0).cpu().tolist(),
        }
        if "recovery_phase" in payload:
            phase_metrics = {}
            for phase in range(5):
                select = payload["recovery_phase"] == phase
                if not bool(select.any()):
                    continue
                phase_error = error[select]
                phase_metrics[str(phase)] = {
                    "samples": int(select.sum().item()),
                    "mean_l2_m": float(phase_error.norm(dim=-1).mean().cpu()),
                    "mean_abs_xyz_m": phase_error.abs().mean(dim=0).cpu().tolist(),
                }
            metrics["per_recovery_phase"] = phase_metrics
        return metrics

    best_loss, best_step, history = float("inf"), 0, []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        adapter.train()
        index = torch.randint(train["action_hidden"].shape[0], (args.batch_size,), device=device)
        prediction = adapter(
            train["action_hidden"][index], train["temporal_slots"][index], proprio=train["proprio"][index],
            previous_action=train["previous_executed_action"][index],
        )["spatial_target_relative_position"].float()
        loss = functional.smooth_l1_loss(prediction, train["teacher_target_relative_position"][index].float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameter for module in geometry_modules for parameter in module.parameters()], 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(val)
            event = {"step": step, "train_batch_loss": float(loss.detach().cpu()), **metrics}
            history.append(event)
            print(json.dumps(event), flush=True)
            if metrics["smooth_l1"] < best_loss:
                best_loss, best_step = metrics["smooth_l1"], step
                state = {key: value.detach().cpu().contiguous() for key, value in adapter.state_dict().items()
                         if not key.startswith("normal_action_head.")}
                save_file(state, str(args.output_dir / "adapter_best.safetensors"))
    test_metrics = None
    if test is not None:
        # Score the model selected exclusively by validation loss.  Reloading
        # makes this robust even when the last optimizer step is not best.
        adapter.load_state_dict(load_file(str(args.output_dir / "adapter_best.safetensors")), strict=False)
        test_metrics = evaluate(test)
        print(json.dumps({"held_out_test": test_metrics}), flush=True)

    report = {
        "kind": "book_regrasp_slot_preserving_target_relative_geometry", "train_features": str(args.train_features),
        "val_features": str(args.val_features), "init_adapter": str(args.init_adapter),
        "train_samples": int(train["action_hidden"].shape[0]), "val_samples": int(val["action_hidden"].shape[0]),
        "best_val_smooth_l1": best_loss, "best_step": best_step, "history": history,
        "held_out_test": test_metrics,
        "online_inputs": ["rgb", "wrist_rgb", "language", "proprio", "previous_executed_action"],
        "teacher_only_label": "target_position_minus_eef_position", "frozen_modules": ["OpenVLA-OFT+", "SlotSSM", "action/gate/phase heads"],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
