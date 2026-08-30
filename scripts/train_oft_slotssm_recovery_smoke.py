#!/usr/bin/env python3
"""Train the first real OFT+SlotSSM recovery-action branch on frozen features.

This intentionally trains *only* the recovery-action pathway.  The seed set
contains recovery and completion windows but no nominal-policy windows, so a
failure gate trained from it would be invalid.  The resulting checkpoint is a
smoke-test artefact, not a reported policy result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.precompute_oft_recovery_window_features import (  # noqa: E402
    DEFAULT_OFT_CHECKPOINT,
    DEFAULT_OFT_ROOT,
    dynamic_module,
    load_torch_state,
)
from recovery.models.openvla_oft_slotssm_adapter import (  # noqa: E402
    OpenVLAOFTSlotSSMAdapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/oft-dual-recovery-seed-v1/frozen-oft-slotssm-features.safetensors"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs/oft-slotssm-recovery-head-smoke-v1"),
    )
    parser.add_argument("--oft-root", type=Path, default=DEFAULT_OFT_ROOT)
    parser.add_argument("--oft-checkpoint", type=Path, default=DEFAULT_OFT_CHECKPOINT)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.features, args.oft_root, args.oft_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    # The repository's Stage-1/2 package also uses ``prismatic``.  Expose the
    # OFT-only namespace extension before loading its action-head module.
    import prismatic
    import prismatic.vla

    prismatic.__path__.append(str(args.oft_root / "prismatic"))
    prismatic.vla.__path__.append(str(args.oft_root / "prismatic" / "vla"))

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    features = load_file(str(args.features), device="cpu")
    positive = features["terminal_regrasp"].bool()
    if not positive.any():
        raise RuntimeError("no terminal-regrasp-positive recovery windows")
    # Failed branches are deliberately excluded from the action imitation
    # objective: their scripted macro is not a successful recovery teacher.
    hidden = features["action_hidden"][positive].to(device=device, dtype=torch.bfloat16)
    slots = features["temporal_slots"][positive].to(device=device, dtype=torch.bfloat16)
    targets = features["recovery_target"][positive].to(device=device, dtype=torch.float32)

    action_components = dynamic_module(
        "oft_action_heads_train", args.oft_root / "prismatic" / "models" / "action_heads.py"
    )
    action_head = action_components.L1RegressionActionHead(4096, 4096, 7)
    action_head.load_state_dict(
        load_torch_state(args.oft_checkpoint / "action_head--150000_checkpoint.pt", torch)
    )
    action_head = action_head.to(device=device, dtype=torch.bfloat16)
    adapter = OpenVLAOFTSlotSSMAdapter(
        action_head, action_hidden_dim=4096, slot_dim=512,
    ).to(device=device, dtype=torch.bfloat16)

    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    before = None
    losses: list[float] = []
    adapter.train()
    for step in range(args.steps):
        indices = torch.randint(hidden.shape[0], (args.batch_size,), device=device)
        output = adapter(hidden[indices], slots[indices])
        prediction = output["recovery_actions"].float()
        # The final gripper command is small in scale but task-critical.
        loss = F.smooth_l1_loss(prediction[..., :6], targets[indices, ..., :6])
        loss = loss + 2.0 * F.smooth_l1_loss(prediction[..., 6:], targets[indices, ..., 6:])
        if before is None:
            before = float(loss.detach().cpu())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 100 == 0 or step == 0:
            print(json.dumps({"step": step + 1, "loss": losses[-1]}), flush=True)

    adapter.eval()
    with torch.inference_mode():
        final_output = adapter(hidden, slots)
        final_loss = F.smooth_l1_loss(final_output["recovery_actions"].float()[..., :6], targets[..., :6])
        final_loss = final_loss + 2.0 * F.smooth_l1_loss(final_output["recovery_actions"].float()[..., 6:], targets[..., 6:])
        normal_gap = (final_output["normal_actions"].float() - final_output["recovery_actions"].float()).abs().mean()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Keep a compact, loadable adapter-only checkpoint; OFT's immutable normal
    # action head is reloaded from the official checkpoint at inference time.
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in adapter.state_dict().items()
        if not key.startswith("normal_action_head.")
    }
    save_file(state, str(args.output_dir / "adapter_recovery.safetensors"))
    report = {
        "kind": "oft_slotssm_recovery_head_smoke",
        "features": str(args.features),
        "official_oft_checkpoint": str(args.oft_checkpoint),
        "positive_recovery_windows": int(hidden.shape[0]),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "initial_batch_loss": before,
        "final_positive_set_loss": float(final_loss.cpu()),
        "mean_normal_recovery_action_gap": float(normal_gap.cpu()),
        "trained_modules": ["slot_fusion", "recovery_action_head"],
        "not_trained": ["failure_gate"],
        "limitation": "recovery-only seed has no NORMAL examples and is only a functional training smoke test",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
