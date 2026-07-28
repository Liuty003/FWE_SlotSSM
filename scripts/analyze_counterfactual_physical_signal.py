#!/usr/bin/env python
"""Analyze physical and Slot signal in matched counterfactual shards."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


BRANCHES = (
    "nominal", "translate_x_minus", "translate_x_plus",
    "translate_y_minus", "translate_y_plus", "translate_z_minus",
    "translate_z_plus", "rotate_z_minus", "rotate_z_plus", "flip_gripper",
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def ranks(values):
    """Average ranks with ties, sufficient for Spearman diagnostics."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def correlation(x, y, rank=False):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if rank:
        x, y = ranks(x), ranks(y)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def describe(values, scale=1.0):
    values = np.asarray(values, np.float64) * scale
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def main():
    args = arguments()
    paths = sorted(Path(args.input_dir).glob("libero_spatial_t*_e*_slots.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no counterfactual Slot shards in {args.input_dir}")

    all_eef, all_object, all_slot, all_effect_slot, all_effect_object = [], [], [], [], []
    all_valid, all_branch = [], []
    forks = 0
    per_branch = {name: {"eef": [], "object": [], "slot": []} for name in BRANCHES}
    per_shard = []
    for path in paths:
        data = load_file(str(path))
        before = data["before_slots"].float()
        after = data["after_slots"].float()
        valid = data["after_valid"].bool()
        eef = data["eef_delta_m"].float().norm(dim=-1)
        obj = data["max_object_displacement_m"].float()
        slot = (after - before[:, None, None]).abs().mean(dim=(-1, -2))
        # Matched counterfactual effect relative to the nominal branch at each horizon.
        effect_slot = (after - after[:, :1]).abs().mean(dim=(-1, -2))
        effect_obj = (obj - obj[:, :1]).abs()
        forks += before.shape[0]
        all_valid.append(valid.reshape(-1).numpy())
        for branch_id, name in enumerate(BRANCHES):
            mask = valid[:, branch_id]
            branch_eef = eef[:, branch_id][mask].numpy()
            branch_obj = obj[:, branch_id][mask].numpy()
            branch_slot = slot[:, branch_id][mask].numpy()
            per_branch[name]["eef"].append(branch_eef)
            per_branch[name]["object"].append(branch_obj)
            per_branch[name]["slot"].append(branch_slot)
            all_eef.append(branch_eef)
            all_object.append(branch_obj)
            all_slot.append(branch_slot)
            all_branch.append(np.full(len(branch_eef), branch_id))
            if branch_id:
                all_effect_slot.append(effect_slot[:, branch_id][mask].numpy())
                all_effect_object.append(effect_obj[:, branch_id][mask].numpy())
        per_shard.append({"file": path.name, "forks": int(before.shape[0])})

    eef = np.concatenate(all_eef)
    obj = np.concatenate(all_object)
    slot = np.concatenate(all_slot)
    effect_slot = np.concatenate(all_effect_slot)
    effect_obj = np.concatenate(all_effect_object)
    valid = np.concatenate(all_valid)
    branch = np.concatenate(all_branch)
    branch_stats = {}
    for name, values in per_branch.items():
        be = np.concatenate(values["eef"])
        bo = np.concatenate(values["object"])
        bs = np.concatenate(values["slot"])
        branch_stats[name] = {
            "valid_frames": int(len(be)),
            "eef_displacement_cm": describe(be, 100.0),
            "object_displacement_cm": describe(bo, 100.0),
            "object_moved_over_1mm_percent": float(100 * np.mean(bo > 0.001)),
            "object_moved_over_5mm_percent": float(100 * np.mean(bo > 0.005)),
            "slot_l1_per_element_unitless": describe(bs),
        }

    result = {
        "units": {
            "physical_displacement": "centimeters unless threshold explicitly says millimeters",
            "slot_l1": "unitless mean absolute difference per Slot embedding element",
            "correlation": "unitless in [-1, 1]",
        },
        "dataset": {
            "shards": len(paths), "fork_states": forks,
            "branches_per_fork": len(BRANCHES),
            "frames_total_including_padding": int(len(valid)),
            "frames_valid": int(valid.sum()),
            "valid_percent": float(100 * valid.mean()),
        },
        "overall": {
            "eef_displacement_cm": describe(eef, 100.0),
            "object_displacement_cm": describe(obj, 100.0),
            "object_moved_over_1mm_percent": float(100 * np.mean(obj > 0.001)),
            "object_moved_over_5mm_percent": float(100 * np.mean(obj > 0.005)),
            "robot_over_5mm_but_object_under_1mm_percent": float(
                100 * np.mean((eef > 0.005) & (obj < 0.001))
            ),
            "slot_l1_per_element_unitless": describe(slot),
        },
        "correlations": {
            "slot_l1_vs_object_displacement_pearson": correlation(slot, obj),
            "slot_l1_vs_object_displacement_spearman": correlation(slot, obj, rank=True),
            "slot_l1_vs_eef_displacement_pearson": correlation(slot, eef),
            "slot_l1_vs_eef_displacement_spearman": correlation(slot, eef, rank=True),
            "counterfactual_slot_effect_vs_object_effect_pearson": correlation(
                effect_slot, effect_obj
            ),
            "counterfactual_slot_effect_vs_object_effect_spearman": correlation(
                effect_slot, effect_obj, rank=True
            ),
        },
        "per_branch": branch_stats,
        "shards": per_shard,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("dataset", "overall", "correlations")}, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
