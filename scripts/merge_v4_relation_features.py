#!/usr/bin/env python3
"""Merge compatible V4 feature shards without weakening source provenance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recovery.data.v4_relation_features import REQUIRED_KEYS, V4RelationFeatures, load, save


def merged_source_records(paths: list[Path], reports: list[dict], overlap: bool) -> list[dict]:
    """Validate concrete lineage when split groups overlap across shards."""
    instance_groups = []
    source_records = []
    for path, report in zip(paths, reports):
        records = report.get("source_records", report.get("sources"))
        if isinstance(records, list):
            source_records.extend(row for row in records if isinstance(row, dict))
        if overlap:
            if not isinstance(records, list) or any(
                not isinstance(row, dict) or not row.get("source_instance_id") for row in records
            ):
                raise ValueError(f"{path}: overlapping source_ids require v2 source-instance provenance")
            instance_groups.append([row["source_instance_id"] for row in records])
    if overlap:
        flattened = [value for group in instance_groups for value in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("input shards duplicate a concrete rollout/fork source_instance_id")
    return source_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-decision-stride", type=int,
                        help="Explicitly attest the stride of legacy reports that predate this provenance field.")
    args = parser.parse_args()
    values = [load(path) for path in args.inputs]
    reports = [json.loads(path.with_suffix(".json").read_text()) for path in args.inputs]
    provenance_keys = ("oft_checkpoint", "action_head_sha256", "proprio_projector_sha256",
                       "action_normalization_key", "recovery_target_coordinates",
                       "decoder_action_contract", "image_preprocessing", "decision_grid_contract",
                       "recovery_target_contract", "native_input_only", "representation_input")
    provenance = {key: reports[0].get(key) for key in provenance_keys}
    if any(value is None for value in provenance.values()):
        raise ValueError("feature report is missing required OFT/action provenance")
    for path, report in zip(args.inputs[1:], reports[1:]):
        if any(report.get(key) != provenance[key] for key in provenance_keys):
            raise ValueError(f"{path}: incompatible OFT/action provenance")
    sampling_keys = (
        "recovery_sampling_stride", "handoff_manifest", "fixed_recovery_window_steps",
        "include_handoff_gate_rows", "include_prefork_gate_rows", "sampling_contract",
    )
    sampling_provenance = {key: reports[0].get(key) for key in sampling_keys}
    for path, report in zip(args.inputs[1:], reports[1:]):
        if any(report.get(key) != sampling_provenance[key] for key in sampling_keys):
            raise ValueError(f"{path}: incompatible recovery/handoff sampling provenance")
    if args.legacy_decision_stride is not None:
        if args.legacy_decision_stride < 1:
            raise ValueError("--legacy-decision-stride must be positive")
        for report in reports:
            report.setdefault("decision_stride", args.legacy_decision_stride)
    strides = [report.get("decision_stride") for report in reports]
    if any(value is None for value in strides) or len(set(strides)) != 1:
        raise ValueError("input shards have incompatible or incomplete decision_stride provenance; "
                         "use --legacy-decision-stride only after auditing old inputs")
    identity_contracts = {report.get("source_identity_contract") for report in reports
                          if report.get("source_identity_contract") is not None}
    if len(identity_contracts) > 1:
        raise ValueError("input shards have incompatible source_identity_contract provenance")
    reference = values[0]
    for path, current in zip(args.inputs[1:], values[1:]):
        for key in ("visual_tokens", "action_hidden", "token_points", "token_valid", "token_image_xy", "gripper_position", "gripper_image_xy", "actions", "recovery_targets"):
            if getattr(current, key).shape[1:] != getattr(reference, key).shape[1:]:
                raise ValueError(f"{path}: incompatible {key} shape")
    all_have_instances = all(item.instance_labels is not None for item in values)
    if any((item.instance_labels is not None) != all_have_instances for item in values):
        raise ValueError("either all shards must have instance_labels or none may; do not silently drop supervision")
    payload = {key: torch.cat([getattr(item, key) for item in values], dim=0) for key in REQUIRED_KEYS}
    if all_have_instances:
        payload["instance_labels"] = torch.cat([item.instance_labels for item in values], dim=0)
    merged = V4RelationFeatures(**payload)
    overlap = merged.source_ids.unique().numel() != sum(item.source_ids.unique().numel() for item in values)
    # Multiple forks from the same initial state intentionally share a
    # split-group source_id.  Distinct instance provenance lets them coexist;
    # the splitter still keeps the whole group on one side.
    source_records = merged_source_records(args.inputs, reports, overlap)
    save(merged, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({
        "kind": "v4_merged_relation_features", "output": str(args.output), "rows": merged.num_rows,
        "sources": int(merged.source_ids.unique().numel()), "inputs": [str(path) for path in args.inputs],
        "source_records": source_records,
        **provenance,
        **sampling_provenance,
        "decision_stride": strides[0],
        "recovery_target_horizon": int(reference.recovery_targets.shape[1]),
        "source_identity_contract": next(iter(identity_contracts), "legacy split-group identity only"),
        "legacy_inputs_without_source_identity": [str(path) for path, report in zip(args.inputs, reports)
                                                   if report.get("source_identity_contract") is None],
    }, indent=2) + "\n")
    print(f"Merged {len(values)} shards into {args.output}: {merged.num_rows} rows, {merged.source_ids.unique().numel()} sources")


if __name__ == "__main__":
    main()
