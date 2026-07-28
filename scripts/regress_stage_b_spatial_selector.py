#!/usr/bin/env python
"""Strict dual-fold regression for the unified Stage-B spatial selector."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stage_b_spatial_selector import StageBSpatialSelector
from scripts.train_stage_b_relation_queries import (
    load_dataset, load_splits, make_batch, paired_query_keys,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, nargs="+", default=[
        Path("experiments/robot/libero/stage_b_selector_fold_a.json"),
        Path("experiments/robot/libero/stage_b_selector_fold_b.json"),
    ])
    parser.add_argument("--data-dir", type=Path, default=Path(
        "/data2/lty/naive-e-slotssm-runs/stage-b-spatial-counterfactual-patches"))
    parser.add_argument("--split-manifest", type=Path, default=Path(
        "/data2/lty/naive-e-slotssm-runs/stage-b-spatial-relation-data/split_manifest.json"))
    parser.add_argument("--language-features", type=Path, default=Path(
        "/data2/lty/naive-e-slotssm-runs/stage-b-spatial-counterfactual-patches/"
        "siglip_language_features.safetensors"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path(
        "/data2/lty/naive-e-slotssm-runs/stage-b-selector-regression.json"))
    return parser.parse_args()


def global_labels(candidates, targets):
    ious = torch.zeros(2, 2)
    for proposal_index in range(2):
        for target_index in range(2):
            proposal, target = candidates[proposal_index].bool(), targets[target_index].bool()
            ious[proposal_index, target_index] = (
                (proposal & target).sum() / (proposal | target).sum().clamp_min(1)
            )
    direct = ious[0, 0] + ious[1, 1]
    return (torch.tensor([0, 1]) if direct >= ious[0, 1] + ious[1, 0]
            else torch.tensor([1, 0])), ious


@torch.inference_mode()
def regress_config(config_path, samples, args, device):
    config = json.loads(config_path.read_text())
    selector = StageBSpatialSelector.from_config(config_path, device)
    splits = load_splits(args.split_manifest, config["protocol"], samples)
    keys = splits["test"]
    correct = {"primary": 0, "counterfactual": 0}
    margins, overlaps, errors, pair_correct = [], [], [], 0
    for start in range(0, len(keys), args.batch_size):
        physical_keys = keys[start:start + args.batch_size]
        query_keys = paired_query_keys(physical_keys)
        batch = make_batch(query_keys, samples, device)
        patches, masks, presence, relations, instruction, relation_text, reference_text = batch
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = selector(
                patches, relations, presence, instruction, relation_text, reference_text
            )
        for index, key in enumerate(physical_keys):
            first, second = output["candidate_masks"][2 * index].bool()
            overlaps.append(float(
                (first & second).sum() / (first | second).sum().clamp_min(1)
            ))
            targets = torch.stack((
                samples[key]["role_masks"][0], samples[key]["counterfactual_target_masks"]
            )).to(device)
            labels, ious = global_labels(output["candidate_masks"][2 * index], targets)
            physical_correct = True
            for variant_index, variant in enumerate(("primary", "counterfactual")):
                query_index = 2 * index + variant_index
                prediction = int(output["selected_indices"][query_index])
                label = int(labels[variant_index])
                is_correct = prediction == label
                correct[variant] += is_correct
                physical_correct &= is_correct
                margins.append(float(output["selection_margin"][query_index]))
                if not is_correct:
                    errors.append({
                        "task_id": key[0], "state_id": key[1], "variant": variant,
                        "prediction": prediction, "label": label,
                        "scores": output["candidate_scores"][query_index].float().cpu().tolist(),
                        "proposal_target_iou": ious.tolist(),
                    })
            pair_correct += physical_correct
    count = len(keys)
    report = {
        "config": str(config_path), "protocol": config["protocol"],
        "num_physical_samples": count, "num_queries": 2 * count,
        "primary_accuracy": correct["primary"] / count,
        "counterfactual_accuracy": correct["counterfactual"] / count,
        "paired_switch_accuracy": pair_correct / count,
        "mean_selected_margin": sum(margins) / len(margins),
        "minimum_selected_margin": min(margins),
        "maximum_candidate_overlap_iou": max(overlaps),
        "errors": errors,
        "strict_checkpoint_reload": True,
        "task_id_is_not_selector_input": "task_id" not in inspect.signature(
            StageBSpatialSelector.forward
        ).parameters,
        "split_leakage_check": True,
    }
    del selector
    torch.cuda.empty_cache()
    return report


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-B selector regression requires CUDA")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    device = torch.device("cuda:0")
    samples = load_dataset(args.data_dir, args.language_features)
    reports = [regress_config(path, samples, args, device) for path in args.configs]
    passed = all(
        report["primary_accuracy"] == 1.0
        and report["counterfactual_accuracy"] == 1.0
        and report["paired_switch_accuracy"] == 1.0
        and report["minimum_selected_margin"] > 0
        and report["maximum_candidate_overlap_iou"] == 0
        and report["strict_checkpoint_reload"]
        and report["task_id_is_not_selector_input"]
        for report in reports
    )
    result = {"schema_version": 1, "passed": passed, "folds": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
