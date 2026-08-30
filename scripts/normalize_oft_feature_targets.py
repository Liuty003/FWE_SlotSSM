#!/usr/bin/env python3
"""Convert cached OFT feature targets from environment to action-head scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-10"))
    parser.add_argument("--key", default="libero_10_no_noops")
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file, save_file

    payload = load_file(str(args.features))
    stats = json.loads((args.checkpoint / "dataset_statistics.json").read_text())[args.key]["action"]
    low = torch.tensor(stats["q01"], dtype=torch.float32)
    high = torch.tensor(stats["q99"], dtype=torch.float32)
    mask = torch.tensor(stats["mask"], dtype=torch.bool)
    target = payload["recovery_target"].float()
    normalized = 2 * (target - low) / (high - low + 1e-8) - 1
    payload["recovery_target"] = torch.where(mask, normalized, target).contiguous()
    backup = args.features.with_suffix(".unnormalized-targets.safetensors")
    if not backup.exists():
        args.features.replace(backup)
    save_file(payload, str(args.features))
    report = {"features": str(args.features), "backup": str(backup), "target_space": "OFT action-head normalized", "min": float(payload["recovery_target"].min()), "max": float(payload["recovery_target"].max())}
    args.features.with_suffix(".target-space.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
