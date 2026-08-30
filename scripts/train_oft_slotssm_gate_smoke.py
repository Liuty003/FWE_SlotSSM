#!/usr/bin/env python3
"""Fit the three-state gate on real frozen OFT+SlotSSM feature windows.

This is deliberately a window-level smoke experiment.  Its checkpoint combines
the already-trained recovery branch with a learned NORMAL/RECOVERY/DONE gate;
independent-episode gate validation follows after more collection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.precompute_oft_recovery_window_features import DEFAULT_OFT_CHECKPOINT, DEFAULT_OFT_ROOT, dynamic_module, load_torch_state  # noqa: E402
from recovery.models.openvla_oft_slotssm_adapter import OpenVLAOFTSlotSSMAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-features", type=Path, default=Path("/data2/lty/naive-e-slotssm-runs/oft-nominal-seed-v1/frozen-oft-slotssm-features.safetensors"))
    parser.add_argument("--recovery-features", type=Path, default=Path("/data2/lty/naive-e-slotssm-runs/oft-dual-recovery-seed-v1/frozen-oft-slotssm-features.safetensors"))
    parser.add_argument("--recovery-checkpoint", type=Path, default=Path("/data2/lty/naive-e-slotssm-runs/oft-slotssm-recovery-head-smoke-v1/adapter_recovery.safetensors"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data2/lty/naive-e-slotssm-runs/oft-slotssm-unified-smoke-v1"))
    parser.add_argument("--oft-root", type=Path, default=DEFAULT_OFT_ROOT)
    parser.add_argument("--oft-checkpoint", type=Path, default=DEFAULT_OFT_CHECKPOINT)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.normal_features, args.recovery_features, args.recovery_checkpoint, args.oft_root, args.oft_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    import torch.nn.functional as F
    import prismatic
    import prismatic.vla
    from safetensors.torch import load_file, save_file

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    prismatic.__path__.append(str(args.oft_root / "prismatic"))
    prismatic.vla.__path__.append(str(args.oft_root / "prismatic" / "vla"))
    action_components = dynamic_module("oft_action_heads_gate", args.oft_root / "prismatic" / "models" / "action_heads.py")
    head = action_components.L1RegressionActionHead(4096, 4096, 7)
    head.load_state_dict(load_torch_state(args.oft_checkpoint / "action_head--150000_checkpoint.pt", torch))
    adapter = OpenVLAOFTSlotSSMAdapter(head.to(device=device, dtype=torch.bfloat16), action_hidden_dim=4096, slot_dim=512).to(device=device, dtype=torch.bfloat16)
    adapter.load_state_dict(load_file(str(args.recovery_checkpoint)), strict=False)

    normal, recovery = load_file(str(args.normal_features)), load_file(str(args.recovery_features))
    hidden = torch.cat((normal["action_hidden"], recovery["action_hidden"])).to(device=device, dtype=torch.bfloat16)
    slots = torch.cat((normal["temporal_slots"], recovery["temporal_slots"])).to(device=device, dtype=torch.bfloat16)
    labels = torch.cat((normal["gate_label"], recovery["gate_label"])).to(device)
    counts = torch.bincount(labels, minlength=3).float()
    class_weights = (counts.sum() / (3 * counts.clamp_min(1))).to(device)
    gate_parameters = list(adapter.gate_action_projection.parameters()) + list(adapter.gate_slot_projection.parameters()) + list(adapter.gate_head.parameters())
    optimizer = torch.optim.AdamW(gate_parameters, lr=args.lr, weight_decay=1e-4)
    adapter.train()
    for step in range(args.steps):
        index = torch.randint(hidden.shape[0], (args.batch_size,), device=device)
        logits = adapter(hidden[index], slots[index])["gate_logits"].float()
        loss = F.cross_entropy(logits, labels[index], weight=class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate_parameters, 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach().cpu())}), flush=True)
    adapter.eval()
    with torch.inference_mode():
        predicted = adapter(hidden, slots)["gate_logits"].argmax(dim=-1)
    confusion = torch.zeros((3, 3), dtype=torch.int64)
    for truth, guess in zip(labels.cpu(), predicted.cpu()):
        confusion[truth, guess] += 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().contiguous() for key, value in adapter.state_dict().items() if not key.startswith("normal_action_head.")}
    save_file(state, str(args.output_dir / "adapter_unified.safetensors"))
    report = {"kind": "oft_slotssm_unified_gate_smoke", "samples": int(labels.numel()), "class_counts": counts.int().tolist(), "steps": args.steps, "window_level_confusion_rows_truth_cols_prediction": confusion.tolist(), "checkpoint": str(args.output_dir / "adapter_unified.safetensors"), "limitation": "window-level fit has source-episode correlation; it is not independent gate validation"}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
