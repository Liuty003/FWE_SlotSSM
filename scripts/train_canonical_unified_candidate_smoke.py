#!/usr/bin/env python
"""Train a source-group-held-out smoke for the canonical shared candidate head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from recovery.models.canonical_unified_candidate import CanonicalUnifiedCandidateHead


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-groups", required=True,
                        help="Comma-separated source_group_id values held out together.")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class CandidateBranches(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row = self.records[index]
        tensors = load_file(row["slot_file"])
        permutation = torch.tensor(row["source_role_permutation"], dtype=torch.long)
        roles = tensors["role_slots"][0, permutation].float()
        geometry = tensors["role_geometry"][0, permutation].float()
        actions = tensors["recovery_actions"].float()
        summary = torch.cat((actions.mean(0), actions.std(0), actions[-1], actions.abs().max(0).values))
        return {
            "role_slots": roles,
            "role_geometry": geometry,
            "action_summary": summary,
            "task_index": torch.tensor(0 if int(row["task_id"]) == 6 else 1),
            "label": torch.tensor(float(row["label_teacher_only"]["regrasp_verified"])),
            "group": row["source_group_id"],
            "candidate": row["candidate_name"],
        }


def collate(batch: list[dict]) -> dict:
    result = {
        key: torch.stack([item[key] for item in batch])
        for key in ("role_slots", "role_geometry", "action_summary", "task_index", "label")
    }
    result["group"] = [item["group"] for item in batch]
    result["candidate"] = [item["candidate"] for item in batch]
    return result


@torch.inference_mode()
def evaluate(model, loader, device) -> dict:
    model.eval()
    labels, scores, groups = [], [], []
    for batch in loader:
        logits = model(
            batch["role_slots"].to(device), batch["role_geometry"].to(device),
            batch["action_summary"].to(device), batch["task_index"].to(device),
        )
        labels.extend(batch["label"].tolist())
        scores.extend(logits.sigmoid().cpu().tolist())
        groups.extend(batch["group"])
    prediction = [score >= .5 for score in scores]
    accuracy = sum(bool(label) == value for label, value in zip(labels, prediction)) / max(len(labels), 1)
    grouped = {}
    for label, score, group in zip(labels, scores, groups):
        grouped.setdefault(group, []).append((label, score))
    group_policy_correct = {}
    group_selected_verified = {}
    for group, values in grouped.items():
        top_label, top_score = max(values, key=lambda item: item[1])
        any_verified = any(label for label, _ in values)
        execute = top_score >= .5
        # A candidate policy is correct if it selects a verified candidate when
        # one exists, or abstains when the candidate set is infeasible.
        group_policy_correct[group] = bool(top_label) if execute else not any_verified
        group_selected_verified[group] = bool(top_label) if execute else None
    return {
        "branches": len(labels),
        "accuracy_at_0_5": accuracy,
        "group_policy_accuracy_with_abstain": (
            sum(group_policy_correct.values()) / max(len(group_policy_correct), 1)
        ),
        "groups": {
            group: {
                "policy_correct": group_policy_correct[group],
                "selected_verified": group_selected_verified[group],
            }
            for group in grouped
        },
    }


def main() -> None:
    args = arguments()
    torch.manual_seed(args.seed)
    validation_groups = {value for value in args.validation_groups.split(",") if value}
    records = read_manifest(args.manifest)
    train_rows = [row for row in records if row["source_group_id"] not in validation_groups]
    validation_rows = [row for row in records if row["source_group_id"] in validation_groups]
    if not train_rows or not validation_rows:
        raise ValueError("both train and validation must contain source groups")
    if {row["source_group_id"] for row in train_rows} & {row["source_group_id"] for row in validation_rows}:
        raise ValueError("source groups leaked between train and validation")
    train = CandidateBranches(train_rows)
    validation = CandidateBranches(validation_rows)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator, collate_fn=collate)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CanonicalUnifiedCandidateHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    positives = sum(row["label_teacher_only"]["regrasp_verified"] for row in train_rows)
    pos_weight = torch.tensor([(len(train_rows) - positives) / max(positives, 1)], device=device)
    iterator, best, history = iter(train_loader), None, []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        logits = model(batch["role_slots"].to(device), batch["role_geometry"].to(device),
                       batch["action_summary"].to(device), batch["task_index"].to(device))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch["label"].to(device), pos_weight=pos_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            validation_report = evaluate(model, validation_loader, device)
            row = {"step": step, "train_bce": float(loss), "validation": validation_report}
            history.append(row)
            print(json.dumps(row), flush=True)
            score = validation_report["group_policy_accuracy_with_abstain"]
            if best is None or score > best["validation"]["group_policy_accuracy_with_abstain"]:
                best = row
                save_file(model.state_dict(), args.output_dir / "canonical-unified-candidate-best.safetensors")
    report = {
        "schema_version": 1,
        "kind": "canonical_shared_candidate_head_smoke",
        "smoke_only": True,
        "not_a_generalization_claim": True,
        "input_contract": "task-specific frozen RGB role slots -> canonical role permutation -> task adapter -> shared head",
        "label_contract": "teacher-only attached_after and final target lift threshold",
        "train_groups": sorted({row["source_group_id"] for row in train_rows}),
        "validation_groups": sorted(validation_groups),
        "train_branches": len(train_rows),
        "validation_branches": len(validation_rows),
        "history": history,
        "best": best,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
