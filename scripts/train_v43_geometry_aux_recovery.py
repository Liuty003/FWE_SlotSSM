#!/usr/bin/env python3
"""Train a geometry-supervised action residual while preserving the V4.3 gate."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.data.v4_relation_features import load
from recovery.models.v4_relational_recovery import V4ModelConfig
from recovery.models.v43_geometry_aux_recovery import V43GeometryAuxRecovery


def inputs(batch):
    return {key: getattr(batch, key) for key in (
        "visual_tokens", "action_hidden", "token_points", "token_valid", "token_image_xy",
        "gripper_position", "gripper_image_xy")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--geometry-labels", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-geometry", type=float, default=4.0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260821)
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file

    torch.manual_seed(a.seed)
    f = load(a.features)
    y = load_file(str(a.geometry_labels))["target_delta_world"]
    if len(y) != f.num_rows or not bool(f.recovery_mask.all()):
        raise ValueError("geometry labels must align with an all-recovery feature shard")
    cfg = V4ModelConfig(
        visual_dim=f.visual_tokens.shape[-1], action_hidden_dim=f.action_hidden.shape[-1],
        action_chunk=f.recovery_targets.shape[-2], action_dim=f.recovery_targets.shape[-1],
        num_slots=16, slot_dim=512, relation_dim=512, num_heads=8, representation="relation_3d")
    model = V43GeometryAuxRecovery(cfg).to(a.device)
    result = model.load_state_dict(load_file(str(a.init_checkpoint), device=a.device), strict=False)
    expected = {n for n, _ in model.named_parameters() if n.startswith(("target_delta_head.", "geometry_action_residual."))}
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise ValueError(f"incompatible V4.3 initialization: {result}")
    # Exact preservation is stronger than approximate gate distillation: the
    # complete archived representation, gate, and base action head stay frozen.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in expected)
    trainable = [x for x in model.parameters() if x.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=1e-4)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")

    def metrics() -> dict[str, float]:
        model.eval(); sums = torch.zeros(4, device=a.device); count = 0
        with torch.inference_mode():
            for start in range(0, f.num_rows, a.batch_size):
                ix = torch.arange(start, min(start + a.batch_size, f.num_rows))
                b = f.index(ix).to(a.device); target = y[ix].to(a.device)
                o = model(**inputs(b))
                action = F.l1_loss(o["recovery_actions"], b.recovery_targets)
                geo = F.smooth_l1_loss(o["predicted_target_delta"], target)
                err = (o["predicted_target_delta"] - target).norm(dim=-1)
                cosine = F.cosine_similarity(o["predicted_target_delta"], target).mean()
                n = len(ix); sums += torch.tensor([action, geo, err.mean(), cosine], device=a.device) * n; count += n
        model.train()
        z = (sums / count).tolist()
        return dict(action_l1=z[0], geometry_smooth_l1=z[1], position_error_m=z[2], cosine=z[3])

    model.train()
    for step in range(1, a.steps + 1):
        ix = torch.randint(f.num_rows, (a.batch_size,))
        b = f.index(ix).to(a.device); target = y[ix].to(a.device); o = model(**inputs(b))
        action = F.l1_loss(o["recovery_actions"], b.recovery_targets)
        geometry = F.smooth_l1_loss(o["predicted_target_delta"], target)
        loss = action + a.lambda_geometry * geometry
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        if step == 1 or step % a.eval_every == 0:
            m = metrics(); row = {"step": step, **m}; history.append(row); print(json.dumps(row), flush=True)
            score = m["action_l1"] + a.lambda_geometry * m["geometry_smooth_l1"]
            if score < best:
                best = score
                save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(a.output_dir / "best.safetensors"))
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(a.output_dir / "last.safetensors"))
    feature_report = json.loads(a.features.with_suffix(".json").read_text())
    config = {"kind": "v43_geometry_aux_recovery_v1", "model": asdict(cfg), "direct_head_version": "pooled_chunk_mlp_v3",
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              "gate_preservation": "exact: archived representation and gate frozen",
              "deployment_inputs": "RGB-D/OFT/proprio only; target delta is predicted, not supplied",
              "feature_provenance": {key: feature_report.get(key) for key in (
                  "oft_checkpoint", "action_head_sha256", "proprio_projector_sha256",
                  "action_normalization_key", "recovery_target_coordinates", "decoder_action_contract",
                  "image_preprocessing", "decision_grid_contract", "recovery_target_contract")}}
    (a.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (a.output_dir / "report.json").write_text(json.dumps({"best_objective": best, "history": history}, indent=2) + "\n")


if __name__ == "__main__":
    main()
