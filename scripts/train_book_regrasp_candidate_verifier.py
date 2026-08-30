#!/usr/bin/env python3
"""Train a visual selector over matched feedback-FSM regrasp macros.

Each group contains five branches started from the *same* saved simulator
state.  The input is frozen OFT+/SlotSSM context plus a deployable macro
descriptor (approach offset and controller constants); the sole supervision is
the offline teacher's final verified-lift bit.  Target pose/contact are never
loaded by this script.

This is deliberately an offline candidate-selection evaluation.  It measures
whether visual context can choose a better FSM macro, not an end-to-end online
recovery success rate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_descriptor(values):
    # Fixed physical scaling, not a statistic fit on validation/test labels.
    import torch
    scale = torch.tensor([0.03, 0.03, 0.105, 0.004, 12.0, 1.0, 1.0])
    return values / scale.to(values)


class MacroVerifier:  # constructed after torch import, for CPU-safe module import
    pass


def main() -> None:
    args = parse_args()
    for path in (args.train_features, args.train_manifest, args.val_features, args.val_manifest,
                 args.test_features, args.test_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.steps < 1 or args.lr <= 0:
        raise ValueError("steps and lr must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from safetensors.torch import load_file, save_file

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    class _MacroVerifier(nn.Module):
        """Small, permutation-equivariant scorer over a context and macro."""

        def __init__(self) -> None:
            super().__init__()
            # A deliberately narrow visual bottleneck is essential here: a
            # source-disjoint group split provides dozens, not millions, of
            # independent recovery states.
            self.action = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, 48), nn.GELU())
            self.slots = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 32), nn.GELU())
            self.robot = nn.Sequential(nn.LayerNorm(15), nn.Linear(15, 16), nn.GELU())
            self.candidate = nn.Sequential(nn.Linear(7, 24), nn.GELU())
            self.value = nn.Sequential(
                nn.LayerNorm(120), nn.Linear(120, 48), nn.GELU(), nn.Dropout(0.15), nn.Linear(48, 1)
            )

        def forward(self, action_hidden, temporal_slots, proprio, previous_action, candidates):
            context = torch.cat((
                self.action(action_hidden.mean(1)), self.slots(temporal_slots.mean(1)),
                self.robot(torch.cat((proprio, previous_action), dim=-1)),
            ), dim=-1)
            count = candidates.shape[1]
            context = context[:, None].expand(-1, count, -1)
            candidate = self.candidate(normalize_descriptor(candidates))
            return self.value(torch.cat((context, candidate), dim=-1)).squeeze(-1)

    def load_split(feature_path: Path, manifest_path: Path):
        payload, manifest = load_file(str(feature_path), device="cpu"), rows(manifest_path)
        required = {"action_hidden", "temporal_slots", "proprio", "previous_executed_action", "recovery_verified_label"}
        if missing := required - set(payload):
            raise ValueError(f"{feature_path} lacks {sorted(missing)}")
        if len(manifest) != payload["action_hidden"].shape[0]:
            raise ValueError(f"feature/manifest row count mismatch: {feature_path}")
        groups = defaultdict(list)
        for index, row in enumerate(manifest):
            if row.get("kind") != "book_regrasp_branch_outcome":
                raise ValueError(f"not a branch-outcome record at {index}")
            groups[str(row["provenance"]["source_fork"])].append((index, row))
        packed = []
        for source_fork, members in sorted(groups.items()):
            if len(members) != 5:
                raise ValueError(f"matched group {source_fork} has {len(members)} candidates, expected 5")
            members.sort(key=lambda item: int(item[1]["candidate_name"] != "fsm_xyp0_p0"))
            index = torch.tensor([item[0] for item in members], dtype=torch.long)
            descriptors = torch.tensor([item[1]["candidate_program_descriptor"] for item in members], dtype=torch.float32)
            labels = payload["recovery_verified_label"][index].float()
            if not torch.equal(labels, torch.tensor([float(item[1]["recovery_verified_label"]) for item in members])):
                raise ValueError(f"label mismatch for {source_fork}")
            packed.append({
                "source_fork": source_fork, "task_id": int(members[0][1]["task_id"]),
                "candidate_names": [item[1]["candidate_name"] for item in members], "index": index,
                "descriptors": descriptors, "labels": labels,
            })
        return payload, packed

    train_payload, train_groups = load_split(args.train_features, args.train_manifest)
    val_payload, val_groups = load_split(args.val_features, args.val_manifest)
    if not train_groups or not val_groups:
        raise ValueError("training and validation each need at least one matched group")

    def device_payload(payload):
        return {key: value.to(device=device, dtype=torch.float32 if value.is_floating_point() else value.dtype)
                for key, value in payload.items()}

    train_payload, val_payload = map(device_payload, (train_payload, val_payload))
    model = _MacroVerifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-3)

    def score_group(payload, group):
        index = group["index"].to(device)
        descriptor = group["descriptors"].to(device)[None]
        # All candidates fork from one state.  Use precisely one pre-action
        # visual context rather than leaking each branch's post-fork render
        # into the candidate's prediction.
        context_index = index[0]
        return model(payload["action_hidden"][context_index][None], payload["temporal_slots"][context_index][None],
                     payload["proprio"][context_index][None], payload["previous_executed_action"][context_index][None], descriptor)[0]

    @torch.inference_mode()
    def evaluate(payload, groups):
        model.eval()
        reports = []
        for group in groups:
            value = score_group(payload, group).float().cpu()
            label = group["labels"]
            choice = int(value.argmax())
            oracle = float(label.max())
            selected = float(label[choice])
            reports.append({"task_id": group["task_id"], "source_fork": group["source_fork"],
                            "selected_name": group["candidate_names"][choice], "selected_success": selected,
                            "oracle_success": oracle, "regret": oracle - selected,
                            "candidate_names": group["candidate_names"], "labels": label.tolist(),
                            "predicted_values": value.tolist()})
        return reports

    def summary(report):
        return {"groups": len(report), "selected_success_rate": sum(x["selected_success"] for x in report) / len(report),
                "oracle_success_rate": sum(x["oracle_success"] for x in report) / len(report),
                "mean_regret": sum(x["regret"] for x in report) / len(report),
                "uniform_candidate_success_rate": sum(sum(x["labels"]) / len(x["labels"]) for x in report) / len(report)}

    @torch.inference_mode()
    def supervised_objective(payload, groups):
        """Validation-only calibration/ranking criterion for checkpoint ties."""
        model.eval()
        losses = []
        for group in groups:
            value = score_group(payload, group)
            label = group["labels"].to(device)
            binary = functional.binary_cross_entropy_with_logits(value, label)
            delta = label[:, None] - label[None, :]
            ordered = delta > 0
            ranking = functional.softplus(
                -(value[:, None] - value[None, :])[ordered]
            ).mean() if bool(ordered.any()) else value.sum() * 0
            losses.append(binary + 0.5 * ranking)
        return float(torch.stack(losses).mean().cpu())

    best = None
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        group = train_groups[torch.randint(len(train_groups), ()).item()]
        value = score_group(train_payload, group)
        label = group["labels"].to(device)
        # Independent branch quality plus within-state ranking.  Equal-
        # quality candidates are deliberately not forced into an order.
        binary = functional.binary_cross_entropy_with_logits(value, label)
        delta = label[:, None] - label[None, :]
        ordered = delta > 0
        ranking = functional.softplus(-(value[:, None] - value[None, :])[ordered]).mean() if bool(ordered.any()) else value.sum() * 0
        loss = binary + 0.5 * ranking
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            validation = evaluate(val_payload, val_groups)
            metrics = summary(validation)
            val_objective = supervised_objective(val_payload, val_groups)
            event = {"step": step, "loss": float(loss.detach().cpu()),
                     "validation_objective": val_objective, **metrics}
            history.append(event); print(json.dumps(event), flush=True)
            key = (metrics["selected_success_rate"], -metrics["mean_regret"], -val_objective)
            if best is None or key > best["key"]:
                best = {"key": key, "step": step,
                        "state": {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    # Do not even deserialize held-out test features until the checkpoint is
    # selected exclusively from training/validation data.
    test_payload, test_groups = load_split(args.test_features, args.test_manifest)
    if not test_groups:
        raise ValueError("held-out test needs at least one matched group")
    test_payload = device_payload(test_payload)
    report = {"kind": "book_regrasp_visual_fsm_macro_verifier", "best_step": best["step"], "history": history,
              "train": {"summary": summary(evaluate(train_payload, train_groups)), "groups": evaluate(train_payload, train_groups)},
              "validation": {"summary": summary(evaluate(val_payload, val_groups)), "groups": evaluate(val_payload, val_groups)},
              "held_out_test": {"summary": summary(evaluate(test_payload, test_groups)), "groups": evaluate(test_payload, test_groups)},
              "online_inputs": ["frozen_OFT_action_hidden", "frozen_SlotSSM_state", "proprio", "previous_executed_action", "candidate_program"],
              "teacher_only_supervision": "final verified lift bit only",
              "limitation": "offline matched-branch selection; no claim of end-to-end recovery until selected macro is executed online"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "macro_verifier_best.safetensors"
    save_file(best["state"], str(checkpoint))
    report["checkpoint"] = str(checkpoint)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["held_out_test"]["summary"]), flush=True)


if __name__ == "__main__":
    main()
