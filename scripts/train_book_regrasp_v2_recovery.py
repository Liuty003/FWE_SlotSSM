#!/usr/bin/env python3
"""Train only the local recovery path from source-disjoint v2 action features.

The normal OFT+ action head is loaded from the official checkpoint and remains
frozen.  This script never reads the held-out test feature file; validation is
an explicit input rather than a random split of recovery windows.
"""

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
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument(
        "--geometry-features", type=Path,
        help="Optional extra frozen features used only for geometry supervision, never action imitation.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path, help="Optional recovery adapter checkpoint to continue from.")
    parser.add_argument("--oft-root", type=Path, default=DEFAULT_OFT_ROOT)
    parser.add_argument("--oft-checkpoint", type=Path, default=DEFAULT_OFT_PLUS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--first-step-weight", type=float, default=1.0,
        help="Training-only multiplier for the first recovery action in each 8-step target chunk.",
    )
    parser.add_argument(
        "--gripper-threshold-weight", type=float, default=0.0,
        help="Training-only weight for enforcing the 0.5 OFT gripper decoding threshold.",
    )
    parser.add_argument(
        "--phase-loss-weight", type=float, default=0.0,
        help="Training-only cross-entropy weight for teacher local-regrasp phase supervision.",
    )
    parser.add_argument(
        "--phase-action-experts", action="store_true",
        help="Hard-route each training window through its teacher phase's residual action expert.",
    )
    parser.add_argument(
        "--phase-balanced-sampling", action="store_true",
        help="Sample local-regrasp phases uniformly; requires phase labels and is intended for phase experts.",
    )
    parser.add_argument(
        "--phase-experts-only", action="store_true",
        help="Freeze the shared recovery path and train only phase action experts; requires --init-adapter.",
    )
    parser.add_argument(
        "--phase-head-only", action="store_true",
        help="Freeze every adapter component except the learned phase classifier; requires --init-adapter and phase loss.",
    )
    parser.add_argument(
        "--mode-conditioned-action-head", action="store_true",
        help="Train the shared neural recovery head conditioned on the five distilled FSM approach modes.",
    )
    parser.add_argument(
        "--disable-phase-conditioning", action="store_true",
        help="Do not condition recovery actions on offline FSM phase labels or predicted phase experts.",
    )
    parser.add_argument(
        "--geometry-conditioned-action-head", action="store_true",
        help="Condition recovery actions on the adapter's learned visual target-relative position.",
    )
    parser.add_argument(
        "--geometry-loss-weight", type=float, default=0.0,
        help="Smooth-L1 supervision weight for visual target-relative position (teacher-only offline label).",
    )
    parser.add_argument("--geometry-aux-batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def action_loss(
    prediction, target, functional, first_step_weight: float = 1.0,
    gripper_threshold_weight: float = 0.0,
):
    """Translation/rotation and gripper supervision at OFT's 8-step horizon."""
    if first_step_weight < 1.0 or gripper_threshold_weight < 0.0:
        raise ValueError("first_step_weight must be at least one and gripper_threshold_weight non-negative")
    weights = prediction.float().new_ones(prediction.shape[:-1])
    weights[:, 0] = first_step_weight
    normalizer = weights.sum()
    arm = (functional.smooth_l1_loss(
        prediction[..., :6].float(), target[..., :6].float(), reduction="none"
    ) * weights[..., None]).sum() / (normalizer * 6)
    gripper = (functional.smooth_l1_loss(
        prediction[..., 6:].float(), target[..., 6:].float(), reduction="none"
    ) * weights[..., None]).sum() / normalizer
    # LIBERO decodes OFT gripper head values with a hard 0.5 threshold:
    # target 1 is environment-open and target 0 is environment-closed.
    # A small L1 error on the wrong side of this threshold can therefore
    # invert the executed grasp action.  The hinge term only activates for
    # such threshold violations (plus a small safety margin).
    desired_side = (target[..., 6].float() >= 0.5).float() * 2.0 - 1.0
    threshold_hinge = functional.relu(0.10 - desired_side * (prediction[..., 6].float() - 0.5))
    threshold = (threshold_hinge * weights).sum() / normalizer
    return arm + 2.0 * gripper + gripper_threshold_weight * threshold


def geometry_loss(prediction, target, functional):
    return functional.smooth_l1_loss(prediction.float(), target.float())


def main() -> None:
    args = parse_args()
    for path in (args.train_features, args.val_features, args.geometry_features, args.oft_root, args.oft_checkpoint, args.init_adapter):
        if path is None:
            continue
        if not path.exists():
            raise FileNotFoundError(path)
    if (args.steps < 1 or args.batch_size < 1 or args.geometry_aux_batch_size < 1 or args.eval_every < 1 or
            args.phase_loss_weight < 0 or args.geometry_loss_weight < 0):
        raise ValueError("steps, batch-size, eval-every must be positive and loss weights non-negative")
    # Let a multi-GPU launcher bind a physical device before imports; --gpu
    # remains a standalone convenience default.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    import torch
    import torch.nn.functional as functional
    from safetensors.torch import load_file, save_file
    import prismatic
    import prismatic.vla

    prismatic.__path__.append(str(args.oft_root / "prismatic"))
    prismatic.vla.__path__.append(str(args.oft_root / "prismatic" / "vla"))
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    train, val = load_file(str(args.train_features), device="cpu"), load_file(str(args.val_features), device="cpu")
    geometry_aux = load_file(str(args.geometry_features), device="cpu") if args.geometry_features else None
    required = {"action_hidden", "temporal_slots", "proprio", "previous_executed_action", "recovery_target"}
    for name, payload in (("train", train), ("val", val)):
        missing = required - set(payload)
        if missing:
            raise ValueError(f"{name} features are not v2 recovery-action features: missing {sorted(missing)}")
        if payload["action_hidden"].shape[0] == 0:
            raise ValueError(f"{name} feature set is empty")
        if args.mode_conditioned_action_head and "recovery_mode" not in payload:
            raise ValueError(f"{name} features lack recovery_mode required for --mode-conditioned-action-head")
        if (args.geometry_conditioned_action_head or args.geometry_loss_weight) and "teacher_target_relative_position" not in payload:
            raise ValueError(f"{name} features lack teacher_target_relative_position required for geometry training")
    if geometry_aux is not None:
        required_geometry = {"action_hidden", "temporal_slots", "proprio", "previous_executed_action", "teacher_target_relative_position"}
        missing = required_geometry - set(geometry_aux)
        if missing:
            raise ValueError(f"geometry features missing {sorted(missing)}")
        if not (args.geometry_conditioned_action_head or args.geometry_loss_weight):
            raise ValueError("--geometry-features requires geometry conditioning or geometry loss")
    if (args.phase_loss_weight or args.phase_action_experts or args.phase_balanced_sampling) and "recovery_phase" not in train:
        raise ValueError("phase supervision, expert routing, and balanced sampling require recovery_phase labels")
    if args.phase_action_experts and "recovery_phase" not in val:
        raise ValueError("phase action expert validation requires recovery_phase labels")
    if args.phase_balanced_sampling and not args.phase_action_experts:
        raise ValueError("phase-balanced-sampling is only supported with phase-action-experts")
    if args.phase_experts_only and (not args.phase_action_experts or args.init_adapter is None):
        raise ValueError("phase-experts-only requires phase-action-experts and init-adapter")
    if args.phase_head_only and (args.init_adapter is None or not args.phase_loss_weight):
        raise ValueError("phase-head-only requires --init-adapter and positive --phase-loss-weight")
    if args.phase_head_only and args.phase_experts_only:
        raise ValueError("phase-head-only cannot combine with phase-experts-only")
    if args.disable_phase_conditioning and (
        args.phase_loss_weight or args.phase_action_experts or args.phase_balanced_sampling or args.phase_experts_only
    ):
        raise ValueError("--disable-phase-conditioning cannot be combined with phase supervision or phase experts")

    action_components = dynamic_module(
        "oft_action_heads_book_v2", args.oft_root / "prismatic" / "models" / "action_heads.py"
    )
    normal_head = action_components.L1RegressionActionHead(4096, 4096, 7)
    normal_head.load_state_dict(load_torch_state(args.oft_checkpoint / "action_head--150000_checkpoint.pt", torch))
    adapter = OpenVLAOFTSlotSSMAdapter(
        normal_head, action_hidden_dim=4096, slot_dim=512, fusion_dim=512, num_heads=8,
    ).to(device=device, dtype=torch.bfloat16)
    if args.init_adapter is not None:
        missing, unexpected = adapter.load_state_dict(load_file(str(args.init_adapter)), strict=False)
        expected_missing = {f"normal_action_head.{key}" for key in adapter.normal_action_head.state_dict()}
        expected_missing.update(
            key for key in adapter.state_dict()
            if key.startswith((
                "phase_action_experts.", "geometry", "mode_embedding.",
                "recovery_robot_action_projection.",
            ))
        )
        if unexpected or not set(missing).issubset(expected_missing):
            raise RuntimeError(f"unexpected init adapter state mismatch: missing={missing}, unexpected={unexpected}")
    if args.phase_experts_only:
        adapter.requires_grad_(False)
        adapter.phase_action_experts.requires_grad_(True)
    if args.phase_head_only:
        adapter.requires_grad_(False)
        adapter.phase_head.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in adapter.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )

    def to_device(payload):
        return {key: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
                for key, value in payload.items()}

    train, val = to_device(train), to_device(val)
    geometry_aux = to_device(geometry_aux) if geometry_aux is not None else None

    @torch.inference_mode()
    def evaluate(payload, *, teacher_phase_override: bool):
        adapter.eval()
        output = adapter(
            payload["action_hidden"], payload["temporal_slots"],
            proprio=payload["proprio"], previous_action=payload["previous_executed_action"],
            recovery_phase_override=(
                payload["recovery_phase"]
                if teacher_phase_override and args.phase_action_experts
                else None
            ),
            recovery_mode_override=(payload["recovery_mode"] if args.mode_conditioned_action_head else None),
            disable_phase_conditioning=args.disable_phase_conditioning,
            geometry_conditioning=args.geometry_conditioned_action_head,
        )
        # Validation stays uniformly weighted, so it measures ordinary action
        # imitation rather than the phase-balanced training sampler.
        action = action_loss(output["recovery_actions"], payload["recovery_target"], functional)
        geometry = (geometry_loss(
            output["spatial_target_relative_position"], payload["teacher_target_relative_position"], functional
        ) if (args.geometry_conditioned_action_head or args.geometry_loss_weight) else action.new_zeros(()))
        return float(action.cpu()), float(geometry.cpu())

    best_val, best_step, history = float("inf"), 0, []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        adapter.train()
        if args.phase_balanced_sampling:
            sampled_phases = torch.randint(5, (args.batch_size,), device=device)
            index = torch.cat([
                torch.where(train["recovery_phase"].eq(phase))[0][
                    torch.randint(int(train["recovery_phase"].eq(phase).sum()), (1,), device=device)
                ]
                for phase in sampled_phases
            ])
        else:
            index = torch.randint(train["action_hidden"].shape[0], (args.batch_size,), device=device)
        output = adapter(
            train["action_hidden"][index], train["temporal_slots"][index],
            proprio=train["proprio"][index], previous_action=train["previous_executed_action"][index],
            recovery_phase_override=(train["recovery_phase"][index] if args.phase_action_experts else None),
            recovery_mode_override=(train["recovery_mode"][index] if args.mode_conditioned_action_head else None),
            disable_phase_conditioning=args.disable_phase_conditioning,
            geometry_conditioning=args.geometry_conditioned_action_head,
        )
        loss = action_loss(
            output["recovery_actions"], train["recovery_target"][index], functional,
            first_step_weight=args.first_step_weight,
            gripper_threshold_weight=args.gripper_threshold_weight,
        )
        if args.phase_loss_weight:
            phase_loss = functional.cross_entropy(
                output["recovery_phase_logits"], train["recovery_phase"][index]
            )
            loss = loss + args.phase_loss_weight * phase_loss
        if args.geometry_loss_weight:
            loss = loss + args.geometry_loss_weight * geometry_loss(
                output["spatial_target_relative_position"],
                train["teacher_target_relative_position"][index], functional,
            )
            if geometry_aux is not None:
                geometry_index = torch.randint(
                    geometry_aux["action_hidden"].shape[0],
                    (args.geometry_aux_batch_size,), device=device,
                )
                geometry_output = adapter(
                    geometry_aux["action_hidden"][geometry_index], geometry_aux["temporal_slots"][geometry_index],
                    proprio=geometry_aux["proprio"][geometry_index],
                    previous_action=geometry_aux["previous_executed_action"][geometry_index],
                    recovery_mode_override=(geometry_aux["recovery_mode"][geometry_index] if args.mode_conditioned_action_head else None),
                    disable_phase_conditioning=args.disable_phase_conditioning,
                    geometry_conditioning=args.geometry_conditioned_action_head,
                )
                loss = loss + args.geometry_loss_weight * geometry_loss(
                    geometry_output["spatial_target_relative_position"],
                    geometry_aux["teacher_target_relative_position"][geometry_index], functional,
                )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            # Select checkpoints with the phase router used at deployment.
            # Teacher-routed loss remains a diagnostic only; selecting by it
            # can retain a model whose phase head is inaccurate online.
            val_action_loss, val_geometry_loss = evaluate(val, teacher_phase_override=False)
            val_loss = val_action_loss + args.geometry_loss_weight * val_geometry_loss
            teacher_action_loss, teacher_geometry_loss = (
                evaluate(val, teacher_phase_override=True)
                if args.phase_action_experts
                else (val_action_loss, val_geometry_loss)
            )
            teacher_val_loss = teacher_action_loss + args.geometry_loss_weight * teacher_geometry_loss
            event = {
                "step": step,
                "train_batch_loss": float(loss.detach().cpu()),
                "val_loss": val_loss,
                "teacher_phase_val_loss": teacher_val_loss,
                "val_action_loss": val_action_loss,
                "val_geometry_loss": val_geometry_loss,
            }
            history.append(event)
            print(json.dumps(event), flush=True)
            if val_loss < best_val:
                best_val, best_step = val_loss, step
                state = {key: value.detach().cpu().contiguous() for key, value in adapter.state_dict().items()
                         if not key.startswith("normal_action_head.")}
                save_file(state, str(args.output_dir / "adapter_best.safetensors"))

    # ``adapter_best`` is selected on the supplied validation distribution.
    # Keep the final iterate as well: targeted DAgger fine-tuning may
    # intentionally validate on a different online-state distribution and
    # must not silently fall back to its initialization.
    last_state = {
        key: value.detach().cpu().contiguous()
        for key, value in adapter.state_dict().items()
        if not key.startswith("normal_action_head.")
    }
    save_file(last_state, str(args.output_dir / "adapter_last.safetensors"))
    report = {
        "kind": "book_regrasp_v2_recovery_head", "train_features": str(args.train_features),
        "val_features": str(args.val_features), "official_oft_checkpoint": str(args.oft_checkpoint),
        "init_adapter": str(args.init_adapter) if args.init_adapter is not None else None,
        "train_samples": int(train["action_hidden"].shape[0]), "val_samples": int(val["action_hidden"].shape[0]),
        "best_val_loss": best_val, "best_step": best_step, "history": history,
        "last_checkpoint": str(args.output_dir / "adapter_last.safetensors"),
        "first_step_weight": args.first_step_weight,
        "gripper_threshold_weight": args.gripper_threshold_weight,
        "phase_loss_weight": args.phase_loss_weight,
        "phase_action_experts": args.phase_action_experts,
        "phase_balanced_sampling": args.phase_balanced_sampling,
        "phase_experts_only": args.phase_experts_only,
        "phase_head_only": args.phase_head_only,
        "mode_conditioned_action_head": args.mode_conditioned_action_head,
        "disable_phase_conditioning": args.disable_phase_conditioning,
        "geometry_conditioned_action_head": args.geometry_conditioned_action_head,
        "geometry_loss_weight": args.geometry_loss_weight,
        "geometry_features": str(args.geometry_features) if args.geometry_features else None,
        "validation_phase_routing": "predicted",
        "trained_modules": [
            "slot_fusion", "recovery_action_head", "phase_head", "phase_embedding", "phase_action_experts", "mode_embedding",
            "gate projections (not gate supervised)",
        ],
        "frozen_modules": ["normal_action_head", "OpenVLA-OFT+"],
        "test_policy": "No held-out test feature file is read by this training script.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
