#!/usr/bin/env python3
"""Train a patch-preserving visual target-relative geometry head.

Unlike the SlotSSM geometry baseline, this model receives the frozen 2D patch
tokens of both cameras.  The robot/action context queries them to select the
book patch before regressing target position minus EEF position.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def args_() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = args_()
    for path in (args.train_features, args.val_features):
        if not path.is_file():
            raise FileNotFoundError(path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from safetensors.torch import load_file, save_file

    class PatchQueryGeometry(nn.Module):
        def __init__(self, patch_dim: int, action_dim: int, patch_count: int, hidden: int = 256) -> None:
            super().__init__()
            self.action = nn.Sequential(nn.LayerNorm(action_dim), nn.Linear(action_dim, hidden), nn.GELU())
            self.patch = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, hidden), nn.GELU())
            self.position = nn.Parameter(torch.zeros(1, patch_count, hidden))
            self.camera = nn.Parameter(torch.zeros(1, patch_count, hidden))
            nn.init.normal_(self.position, std=0.01)
            # Tokens are concatenated main-camera then wrist-camera by OFT.
            with torch.no_grad():
                self.camera[:, patch_count // 2:].normal_(std=0.01)
            self.attention = nn.MultiheadAttention(hidden, 8, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))

        def forward(self, patches: torch.Tensor, action_hidden: torch.Tensor) -> torch.Tensor:
            query = self.action(action_hidden.mean(dim=1))[:, None]
            tokens = self.patch(patches) + self.position + self.camera
            selected, _ = self.attention(query, tokens, tokens, need_weights=False)
            return self.head(torch.cat((query[:, 0], selected[:, 0]), dim=-1))

    torch.manual_seed(args.seed)
    train, val = load_file(str(args.train_features), device="cpu"), load_file(str(args.val_features), device="cpu")
    needed = {"current_patches", "action_hidden", "teacher_target_relative_position"}
    for name, payload in (("train", train), ("val", val)):
        missing = needed - set(payload)
        if missing:
            raise ValueError(f"{name} missing {sorted(missing)}")
    patch_count, patch_dim = train["current_patches"].shape[1:]
    device = torch.device("cuda")
    model = PatchQueryGeometry(patch_dim, train["action_hidden"].shape[-1], patch_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def batch(payload, index):
        return (payload["current_patches"][index].to(device=device, dtype=torch.bfloat16),
                payload["action_hidden"][index].to(device=device, dtype=torch.bfloat16),
                payload["teacher_target_relative_position"][index].to(device=device, dtype=torch.float32))

    @torch.inference_mode()
    def evaluate(payload):
        model.eval(); predictions, targets = [], []
        for start in range(0, payload["current_patches"].shape[0], args.batch_size):
            idx = slice(start, start + args.batch_size)
            patches, hidden, target = batch(payload, idx)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictions.append(model(patches, hidden).float().cpu())
            targets.append(target.cpu())
        error = torch.cat(predictions) - torch.cat(targets)
        return {"smooth_l1": float(functional.smooth_l1_loss(torch.cat(predictions), torch.cat(targets))),
                "mean_l2_m": float(error.norm(dim=-1).mean()), "mean_abs_xyz_m": error.abs().mean(0).tolist()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    for step in range(1, args.steps + 1):
        model.train()
        index = torch.randint(train["current_patches"].shape[0], (args.batch_size,))
        patches, hidden, target = batch(train, index)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(patches, hidden)
        loss = functional.smooth_l1_loss(prediction.float(), target)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(val); event = {"step": step, "train_batch_loss": float(loss.detach().cpu()), **metrics}; history.append(event)
            print(json.dumps(event), flush=True)
            if metrics["smooth_l1"] < best:
                best = metrics["smooth_l1"]
                save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(args.output_dir / "patch_geometry_best.safetensors"))
    report = {"kind": "book_regrasp_patch_query_geometry", "train_samples": int(train["current_patches"].shape[0]),
              "val_samples": int(val["current_patches"].shape[0]), "best_val_smooth_l1": best, "history": history,
              "online_inputs": ["rgb", "wrist_rgb", "language", "proprio", "executed_action_history"],
              "teacher_only_label": "target_position_minus_eef_position"}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
