#!/usr/bin/env python3
"""Train a two- or three-state execution gate on frozen OFT+/SlotSSM features."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def bind_requested_cuda_before_imports() -> None:
    """Bind the requested physical GPU before OFT bridge imports torch."""
    for index, value in enumerate(sys.argv[:-1]):
        if value == "--gpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[index + 1]
            return


bind_requested_cuda_before_imports()

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_unified_oft_slotssm_bridge import DEFAULT_OFT_ROOT, dynamic_module, load_torch_state  # noqa: E402
from recovery.models.openvla_oft_slotssm_adapter import OpenVLAOFTSlotSSMAdapter  # noqa: E402

DEFAULT_OFT_PLUS = Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oft-root", type=Path, default=DEFAULT_OFT_ROOT)
    parser.add_argument("--oft-checkpoint", type=Path, default=DEFAULT_OFT_PLUS)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-classes", type=int, choices=(2, 3), default=2)
    parser.add_argument(
        "--init-gate-checkpoint", type=Path,
        help="Optional gate-only safetensors checkpoint for conservative continuing fine-tuning.",
    )
    parser.add_argument(
        "--unweighted-classes", action="store_true",
        help="Use empirical class frequency rather than inverse-frequency class weighting."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.train_features, args.val_features, args.oft_root, args.oft_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.init_gate_checkpoint is not None and not args.init_gate_checkpoint.is_file():
        raise FileNotFoundError(args.init_gate_checkpoint)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
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
    required = {"action_hidden", "temporal_slots", "proprio", "previous_executed_action", "label"}
    if required - set(train) or required - set(val):
        raise ValueError("gate features are missing required tensors")
    if int(train["label"].max()) >= args.num_classes or int(val["label"].max()) >= args.num_classes:
        raise ValueError(f"labels exceed --num-classes={args.num_classes}")

    components = dynamic_module("oft_action_heads_gate_v2", args.oft_root / "prismatic" / "models" / "action_heads.py")
    normal_head = components.L1RegressionActionHead(4096, 4096, 7)
    normal_head.load_state_dict(load_torch_state(args.oft_checkpoint / "action_head--150000_checkpoint.pt", torch))
    adapter = OpenVLAOFTSlotSSMAdapter(normal_head, action_hidden_dim=4096, slot_dim=512).to(device, dtype=torch.bfloat16)
    if args.init_gate_checkpoint is not None:
        initial_state = load_file(str(args.init_gate_checkpoint))
        if not initial_state or any(not key.startswith("gate_") for key in initial_state):
            raise ValueError("--init-gate-checkpoint must contain only gate_* tensors")
        missing, unexpected = adapter.load_state_dict(initial_state, strict=False)
        expected = {key for key in adapter.state_dict() if key.startswith("gate_")}
        if unexpected or expected.intersection(missing):
            raise ValueError(
                f"incompatible initial gate checkpoint; missing={sorted(expected.intersection(missing))}, "
                f"unexpected={unexpected}"
            )
    adapter.requires_grad_(False)
    for module in (adapter.gate_action_projection, adapter.gate_slot_projection,
                   adapter.gate_robot_projection, adapter.gate_head):
        module.requires_grad_(True)
    optimizer = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)

    def move(payload):
        return {key: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
                for key, value in payload.items()}
    train, val = move(train), move(val)
    counts = torch.bincount(train["label"], minlength=args.num_classes).float()
    if (counts == 0).any():
        raise ValueError(f"both classes required, got counts={counts.tolist()}")
    class_weight = (
        torch.ones_like(counts, device=device)
        if args.unweighted_classes
        else (counts.sum() / (args.num_classes * counts)).to(device)
    )

    @torch.inference_mode()
    def evaluate(payload):
        adapter.eval()
        logits = adapter(payload["action_hidden"], payload["temporal_slots"],
                         proprio=payload["proprio"], previous_action=payload["previous_executed_action"])["gate_logits"][:, :args.num_classes].float()
        loss = functional.cross_entropy(logits, payload["label"], weight=class_weight)
        prediction = logits.argmax(dim=-1)
        target = payload["label"]
        tp = int(((prediction == 1) & (target == 1)).sum())
        fp = int(((prediction == 1) & (target == 0)).sum())
        fn = int(((prediction == 0) & (target == 1)).sum())
        metrics = {"tp": tp, "fp": fp, "fn": fn,
                   "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}
        if args.num_classes == 3:
            done_tp = int(((prediction == 2) & (target == 2)).sum())
            done_fp = int(((prediction == 2) & (target != 2)).sum())
            done_fn = int(((prediction != 2) & (target == 2)).sum())
            metrics.update({"done_tp": done_tp, "done_fp": done_fp, "done_fn": done_fn,
                            "done_precision": done_tp / max(1, done_tp + done_fp),
                            "done_recall": done_tp / max(1, done_tp + done_fn)})
        return float(loss.cpu()), metrics

    best, history = float("inf"), []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        adapter.train()
        index = torch.randint(train["label"].shape[0], (args.batch_size,), device=device)
        logits = adapter(train["action_hidden"][index], train["temporal_slots"][index],
                         proprio=train["proprio"][index], previous_action=train["previous_executed_action"][index])["gate_logits"][:, :args.num_classes].float()
        loss = functional.cross_entropy(logits, train["label"][index], weight=class_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val_loss, metrics = evaluate(val)
            event = {"step": step, "train_batch_loss": float(loss.detach().cpu()), "val_loss": val_loss, **metrics}
            history.append(event)
            print(json.dumps(event), flush=True)
            if val_loss < best:
                best = val_loss
                state = {key: value.detach().cpu().contiguous() for key, value in adapter.state_dict().items()
                         if key.startswith("gate_")}
                save_file(state, str(args.output_dir / "gate_best.safetensors"))
    report = {"kind": f"book_regrasp_v2_{args.num_classes}class_gate", "train_samples": int(train["label"].shape[0]),
              "val_samples": int(val["label"].shape[0]), "train_label_counts": counts.cpu().tolist(),
              "unweighted_classes": args.unweighted_classes,
              "best_val_loss": best, "history": history}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
