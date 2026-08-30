#!/usr/bin/env python3
"""Concatenate compatible frozen recovery-feature safetensors files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--jsonl-inputs", type=Path, nargs="*",
        help="Optional JSONL manifests to concatenate in exactly the same order as --inputs.",
    )
    parser.add_argument(
        "--jsonl-output", type=Path,
        help="Output path for the concatenated JSONL manifest; requires --jsonl-inputs.",
    )
    parser.add_argument(
        "--common-keys-only", action="store_true",
        help=("Merge only tensor keys shared by every input. Useful when a new "
              "feature shard carries optional diagnostics such as vision patches."),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.jsonl_inputs) != bool(args.jsonl_output):
        raise ValueError("--jsonl-inputs and --jsonl-output must be specified together")
    if args.jsonl_inputs and len(args.jsonl_inputs) != len(args.inputs):
        raise ValueError("--jsonl-inputs must have the same number of paths as --inputs")
    payloads = [load_file(str(path)) for path in args.inputs]
    reference_keys = set(payloads[0])
    for path, payload in zip(args.inputs[1:], payloads[1:]):
        if not args.common_keys_only and set(payload) != reference_keys:
            raise ValueError(f"feature keys differ in {path}")
        if args.common_keys_only:
            reference_keys &= set(payload)
    if not reference_keys:
        raise ValueError("inputs have no tensor keys in common")
    merged: dict[str, torch.Tensor] = {}
    for key in sorted(reference_keys):
        tensors = [payload[key] for payload in payloads]
        tail = tensors[0].shape[1:]
        if any(tensor.shape[1:] != tail for tensor in tensors[1:]):
            raise ValueError(f"non-batch shape differs for {key}")
        merged[key] = torch.cat(tensors, dim=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(args.output))
    samples = int(next(iter(merged.values())).shape[0])
    if args.jsonl_inputs:
        rows: list[str] = []
        for feature_path, manifest_path, payload in zip(args.inputs, args.jsonl_inputs, payloads):
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            manifest_rows = [line for line in manifest_path.read_text().splitlines() if line.strip()]
            if len(manifest_rows) != next(iter(payload.values())).shape[0]:
                raise ValueError(
                    f"feature/manifest row count mismatch for {feature_path} and {manifest_path}"
                )
            # Verify that every line is valid before producing a partial output.
            [json.loads(line) for line in manifest_rows]
            rows.extend(manifest_rows)
        if len(rows) != samples:
            raise RuntimeError("merged feature/manifest row count mismatch")
        args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl_output.write_text("\n".join(rows) + "\n")
    print({"output": str(args.output), "samples": samples, "jsonl_output": str(args.jsonl_output) if args.jsonl_output else None})


if __name__ == "__main__":
    main()
