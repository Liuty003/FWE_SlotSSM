#!/usr/bin/env python3
"""Attach five-way FSM-teacher mode labels to frozen recovery features.

The feature tensors deliberately contain only deployment-time observations.
The candidate name is an offline teacher label in the action-window manifest;
this utility joins it by deterministic manifest order and writes a new tensor
file for mode-conditioned imitation.  It never exposes FSM pose/contact data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODE = {
    "fsm_xyp0_p0": 0,
    "fsm_xym25_p0": 1,
    "fsm_xyp0_m25": 2,
    "fsm_xyp0_p25": 3,
    "fsm_xyp25_p0": 4,
}


def args_() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = args_()
    from safetensors.torch import load_file, save_file
    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    records = [row for row in records if row.get("split") == args.split]
    features = load_file(str(args.features), device="cpu")
    count = int(features["action_hidden"].shape[0])
    if len(records) != count:
        raise ValueError(f"manifest/features order mismatch: {len(records)} records versus {count} tensors")
    try:
        modes = [MODE[row["candidate_name"]] for row in records]
    except KeyError as error:
        raise ValueError(f"unknown recovery teacher mode: {error}") from error
    import torch
    payload = dict(features)
    payload["recovery_mode"] = torch.tensor(modes, dtype=torch.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(payload, str(args.output))
    report = {
        "kind": "book_regrasp_mode_conditioned_features_v1",
        "features": str(args.features), "manifest": str(args.manifest),
        "split": args.split, "samples": count,
        "mode_names": list(MODE),
        "mode_counts": {name: modes.count(index) for name, index in MODE.items()},
        "online_input_contract": ["rgb", "wrist_rgb", "language", "proprio", "previous_executed_action"],
        "teacher_only": ["candidate_name"],
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
