#!/usr/bin/env python
"""Probe whether oracle mask geometry can select the queried bowl across tasks."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_stage_b_relation_queries import (
    RelationQuerySegmenter, load_dataset as load_relation_dataset,
    make_batch, paired_query_keys,
)
from scripts.train_stage_b_bowl_proposals import BowlProposalSegmenter


RELATIONS = ("between", "in", "next_to", "on", "table_center")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(
        "/data2/lty/naive-e-slotssm-runs/stage-b-spatial-counterfactual-patches"))
    parser.add_argument("--split-manifest", type=Path, default=Path(
        "/data2/lty/naive-e-slotssm-runs/stage-b-spatial-relation-data/split_manifest.json"))
    parser.add_argument("--protocol", default="compositional_reference_seen_fold_b")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--relation-checkpoint", type=Path)
    parser.add_argument("--bowl-checkpoint", type=Path)
    parser.add_argument("--language-features", type=Path, default=Path(
        "/data2/lty/naive-e-slotssm-runs/stage-b-spatial-counterfactual-patches/"
        "siglip_language_features.safetensors"))
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--cascade-test-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--component-proposals", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", type=Path, default=Path(
        "/data2/lty/naive-e-slotssm-runs/stage-b-oracle-geometry-fold-b.json"))
    parser.add_argument("--geometry-checkpoint", type=Path)
    return parser.parse_args()


def mask_stats(mask):
    mask = mask.bool()
    points = mask.nonzero().float()
    if not len(points):
        return torch.zeros(7)
    y, x = points[:, 0] / 63, points[:, 1] / 63
    return torch.tensor([
        x.mean(), y.mean(), x.min(), y.min(), x.max(), y.max(), mask.float().mean()
    ])


def geometry(candidate, references):
    candidate = candidate.bool()
    cs = mask_stats(candidate)
    features = list(cs)
    reference_stats = []
    for reference in references:
        rs = mask_stats(reference)
        reference_stats.append(rs)
        intersection = (candidate & reference.bool()).float().mean()
        union = (candidate | reference.bool()).float().mean().clamp_min(1e-6)
        features.extend((rs, torch.tensor([
            cs[0] - rs[0], cs[1] - rs[1],
            torch.linalg.vector_norm(cs[:2] - rs[:2]),
            intersection, intersection / union,
        ])))
    midpoint = (reference_stats[0][:2] + reference_stats[1][:2]) / 2
    features.extend((midpoint, torch.tensor([
        torch.linalg.vector_norm(cs[:2] - midpoint),
        torch.linalg.vector_norm(cs[:2] - torch.tensor([0.5, 0.5])),
    ])))
    return torch.cat([value.flatten() for value in features])


def load_samples(data_dir):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    samples = {}
    for task in manifest["tasks"]:
        tensors = load_file(task["output"])
        for index, state in enumerate(tensors["state_ids"].tolist()):
            primary_refs = tensors["role_masks"][index, 3:5]
            counterfactual_refs = tensors["counterfactual_reference_masks"][index]
            samples[(int(task["task_id"]), int(state))] = {
                "candidates": torch.stack((
                    tensors["role_masks"][index, 0],
                    tensors["counterfactual_target_masks"][index],
                )),
                "queries": (
                    (int(tensors["relation_ids"][index]), primary_refs, 0),
                    (int(tensors["counterfactual_relation_ids"][index]), counterfactual_refs, 1),
                ),
            }
    return samples


def query_rows(keys, samples, predicted=None, bowl_source="oracle", reference_source="oracle"):
    rows = []
    for key in keys:
        sample = samples[key]
        for variant, (relation, references, label) in enumerate(sample["queries"]):
            candidates = sample["candidates"]
            evaluation_label = label
            if bowl_source == "predicted":
                candidates = predicted[key]["candidates"]
                ious = torch.zeros(2, 2)
                for proposal_index in range(2):
                    for target_index in range(2):
                        proposal = candidates[proposal_index].bool()
                        target = sample["candidates"][target_index].bool()
                        ious[proposal_index, target_index] = (
                            (proposal & target).sum() /
                            (proposal | target).sum().clamp_min(1)
                        )
                direct = ious[0, 0] + ious[1, 1]
                swapped = ious[0, 1] + ious[1, 0]
                evaluation_label = label if direct >= swapped else 1 - label
            if reference_source == "predicted":
                references = predicted[key]["references"][variant]
            reference_list = [references[0], references[1]]
            candidate_features = []
            for candidate in candidates:
                geom = geometry(candidate, reference_list)
                one_hot = F.one_hot(torch.tensor(relation), len(RELATIONS)).float()
                candidate_features.append(torch.cat((geom, one_hot)))
            rows.append((torch.stack(candidate_features), evaluation_label, key, variant, relation))
    return rows


@torch.inference_mode()
def predict_masks(args, keys):
    if not torch.cuda.is_available():
        raise RuntimeError("predicted-mask cascade requires CUDA")
    device = torch.device("cuda:0")
    relation_samples = load_relation_dataset(args.data_dir, args.language_features)
    language_dim = relation_samples[next(iter(relation_samples))][
        "instruction_features"
    ].shape[-1]
    model = RelationQuerySegmenter(args.hidden_dim, args.heads, language_dim).to(device)
    model.load_state_dict(load_file(args.relation_checkpoint), strict=True)
    model.eval()
    bowl_model = None
    if args.bowl_checkpoint:
        bowl_model = BowlProposalSegmenter(args.hidden_dim, args.heads).to(device)
        bowl_model.load_state_dict(load_file(args.bowl_checkpoint), strict=True)
        bowl_model.eval()
    predicted = {}
    for start in range(0, len(keys), 16):
        physical_keys = keys[start:start + 16]
        batch = make_batch(paired_query_keys(physical_keys), relation_samples, device)
        patches, masks, presence, relations, instruction, relation_text, reference_text = batch
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(
                patches, relations, presence, instruction, relation_text, reference_text
            )
            bowl_logits = bowl_model(patches[::2]) if bowl_model is not None else None
        binary = logits.sigmoid() > args.mask_threshold
        binary[:, 3:5] &= presence[:, 3:5, None, None].bool()
        bowl_binary = bowl_logits.sigmoid() > args.mask_threshold if bowl_logits is not None else None
        for index, key in enumerate(physical_keys):
            primary, counterfactual = binary[2 * index], binary[2 * index + 1]
            candidates = (
                bowl_binary[index].cpu() if bowl_binary is not None else primary[0:2].cpu()
            )
            if getattr(args, "component_proposals", False):
                union = candidates.any(0).numpy()
                labels, count = ndimage.label(union)
                components = sorted(
                    ((labels == component, int((labels == component).sum()))
                     for component in range(1, count + 1)),
                    key=lambda item: item[1], reverse=True,
                )
                if len(components) >= 2:
                    candidates = torch.from_numpy(np.stack((
                        components[0][0], components[1][0]
                    )))
            predicted[key] = {
                # The primary role head defines bowl1/bowl2 without consulting the
                # counterfactual answer, so this is a strict upstream candidate test.
                "candidates": candidates,
                "references": torch.stack((primary[3:5], counterfactual[3:5])).cpu(),
            }
    return predicted


def mask_quality(keys, samples, predicted):
    candidate_ious, reference_ious, candidate_overlaps = [[], []], [[], []], []
    permutation_ious = []
    empty_candidates = 0
    for key in keys:
        oracle = samples[key]
        prediction = predicted[key]
        for index in range(2):
            pred = prediction["candidates"][index].bool()
            target = oracle["candidates"][index].bool()
            intersection = (pred & target).sum().float()
            union = (pred | target).sum().float().clamp_min(1)
            candidate_ious[index].append(float(intersection / union))
            empty_candidates += int(not pred.any())
        matrix = torch.zeros(2, 2)
        for proposal_index in range(2):
            for target_index in range(2):
                proposal = prediction["candidates"][proposal_index].bool()
                target = oracle["candidates"][target_index].bool()
                matrix[proposal_index, target_index] = (
                    (proposal & target).sum() / (proposal | target).sum().clamp_min(1)
                )
        permutation_ious.append(float(torch.maximum(
            matrix[0, 0] + matrix[1, 1], matrix[0, 1] + matrix[1, 0]
        ) / 2))
        first, second = prediction["candidates"].bool()
        candidate_overlaps.append(float(
            (first & second).sum() / (first | second).sum().clamp_min(1)
        ))
        for variant, (_, oracle_refs, _) in enumerate(oracle["queries"]):
            pred_ref = prediction["references"][variant, 0].bool()
            target_ref = oracle_refs[0].bool()
            reference_ious[variant].append(float(
                (pred_ref & target_ref).sum() / (pred_ref | target_ref).sum().clamp_min(1)
            ))
    mean = lambda values: sum(values) / max(len(values), 1)
    return {
        "candidate_0_iou": mean(candidate_ious[0]),
        "candidate_1_iou": mean(candidate_ious[1]),
        "candidate_pair_overlap_iou": mean(candidate_overlaps),
        "candidate_permutation_mean_binary_iou": mean(permutation_ious),
        "empty_candidate_rate": empty_candidates / (2 * len(keys)),
        "primary_reference_0_iou": mean(reference_ious[0]),
        "counterfactual_reference_0_iou": mean(reference_ious[1]),
    }


class GeometryScorer(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


@torch.inference_mode()
def evaluate(model, rows):
    by_task, by_variant, by_relation, correct = {}, {}, {}, 0
    margins, errors = [], []
    for features, label, key, variant, relation in rows:
        scores = model(features)
        prediction = int(scores.argmax())
        correct += prediction == label
        task = key[0]
        task_counts = by_task.setdefault(task, [0, 0])
        task_counts[0] += prediction == label
        task_counts[1] += 1
        for groups, group in ((by_variant, variant), (by_relation, RELATIONS[relation])):
            group_counts = groups.setdefault(group, [0, 0])
            group_counts[0] += prediction == label
            group_counts[1] += 1
        margins.append(float(scores[label] - scores[1 - label]))
        if prediction != label:
            errors.append({
                "task_id": key[0], "state_id": key[1],
                "variant": "primary" if variant == 0 else "counterfactual",
                "relation": RELATIONS[relation], "prediction": prediction,
                "label": label, "correct_margin": margins[-1],
                "scores": [float(value) for value in scores],
            })
    return {
        "accuracy": correct / len(rows),
        "mean_correct_margin": sum(margins) / len(margins),
        "by_task_accuracy": {str(k): a / n for k, (a, n) in by_task.items()},
        "by_variant_accuracy": {
            ("primary" if k == 0 else "counterfactual"): a / n
            for k, (a, n) in by_variant.items()
        },
        "by_relation_accuracy": {str(k): a / n for k, (a, n) in by_relation.items()},
        "num_queries": len(rows),
        "errors": errors,
    }


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    samples = load_samples(args.data_dir)
    split_data = json.loads(args.split_manifest.read_text())["protocols"][args.protocol]
    splits = {
        name: [(int(row["task_id"]), int(row["state_id"])) for row in values]
        for name, values in split_data.items()
    }
    rows = {name: query_rows(keys, samples) for name, keys in splits.items()}
    dimension = rows["train"][0][0].shape[-1]
    model = GeometryScorer(dimension)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for step in range(args.steps):
        batch = rng.sample(rows["train"], min(args.batch_size, len(rows["train"])))
        features = torch.stack([row[0] for row in batch])
        labels = torch.tensor([row[1] for row in batch])
        loss = F.cross_entropy(model(features), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    report = {
        "schema_version": 1, "protocol": args.protocol,
        "feature_dimension": dimension, "steps": args.steps,
        **{name: evaluate(model, split_rows) for name, split_rows in rows.items()},
    }
    if args.geometry_checkpoint:
        args.geometry_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {key: value.detach().cpu() for key, value in model.state_dict().items()},
            args.geometry_checkpoint,
            metadata={"protocol": args.protocol, "steps": str(args.steps),
                      "feature_dimension": str(dimension)},
        )
        reloaded = GeometryScorer(dimension)
        reloaded.load_state_dict(load_file(args.geometry_checkpoint), strict=True)
        report["geometry_checkpoint"] = str(args.geometry_checkpoint)
    if args.relation_checkpoint:
        cascade_splits = {"test": splits["test"]} if args.cascade_test_only else splits
        all_keys = list(dict.fromkeys(key for keys in cascade_splits.values() for key in keys))
        predicted = predict_masks(args, all_keys)
        report["cascade"] = {}
        report["predicted_mask_quality"] = {
            split: mask_quality(keys, samples, predicted)
            for split, keys in cascade_splits.items()
        }
        for bowl_source, reference_source in (
            ("oracle", "oracle"), ("predicted", "oracle"),
            ("oracle", "predicted"), ("predicted", "predicted"),
        ):
            name = f"{bowl_source}_bowls__{reference_source}_references"
            report["cascade"][name] = {
                split: evaluate(model, query_rows(
                    keys, samples, predicted, bowl_source, reference_source
                ))
                for split, keys in cascade_splits.items()
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
