#!/usr/bin/env python3
"""Build an exhaustive LIBERO-10 runtime route for family-specific Gate heads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-index", type=Path, required=True)
    parser.add_argument("--family-gate-root", type=Path, required=True)
    parser.add_argument("--shared-gate-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-name", default="calibration-trajectory-fpr005.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    index = json.loads(args.family_index.read_text())
    if index.get("kind") != "libero_10_semantic_family_index_v1" or index.get("num_instances") != 2519:
        raise ValueError("family index must be the exhaustive 2519-instance LIBERO-10 index")
    shared = load_file(str(args.shared_gate_checkpoint), device="cpu")
    non_gate_keys = {key for key in shared if not key.startswith("gate.")}
    families, task_routes = {}, {}
    reference_contract = None
    for family_row in index["families"]:
        family = family_row["family_name"]
        model_dir = args.family_gate_root / family
        checkpoint = model_dir / "best.safetensors"
        config_path = model_dir / "config.json"
        calibration_path = model_dir / args.calibration_name
        for path in (checkpoint, config_path, calibration_path, Path(family_row["family_manifest"])):
            if not path.is_file():
                raise FileNotFoundError(path)
        config = json.loads(config_path.read_text())
        calibration = json.loads(calibration_path.read_text())
        if calibration.get("kind") != "libero10_gate_family_calibration_v2":
            raise ValueError(f"{family}: calibration is not trajectory-aware v2")
        value = calibration["families"].get(family, {}).get("calibrated")
        if not value or value.get("target_satisfied") is not True:
            raise ValueError(f"{family}: calibration did not satisfy the source-level FPR target")
        state = load_file(str(checkpoint), device="cpu")
        if set(state) != set(shared):
            raise ValueError(f"{family}: checkpoint keys differ from the shared Gate checkpoint")
        changed = [key for key in non_gate_keys if not state[key].equal(shared[key])]
        if changed:
            raise ValueError(f"{family}: Gate-only training changed shared tensors: {changed[:3]}")
        contract = {
            "model": config["model"],
            "direct_head_version": config["direct_head_version"],
            "recovery_output_horizon": config["recovery_output_horizon"],
            "action_normalization_key": config["feature_provenance"]["action_normalization_key"],
        }
        if reference_contract is None:
            reference_contract = contract
        elif contract != reference_contract:
            raise ValueError(f"{family}: model/feature contract differs from other family Gates")
        manifest = json.loads(Path(family_row["family_manifest"]).read_text())
        task_ids = sorted(int(row["task_id"]) for row in manifest["records"])
        if len(task_ids) != int(family_row["num_instances"]):
            raise ValueError(f"{family}: task count differs from family index")
        for task_id in task_ids:
            key = str(task_id)
            if key in task_routes:
                raise ValueError(f"task {task_id} occurs in more than one family")
            task_routes[key] = family
        families[family] = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "gate_threshold": float(value["threshold"]),
            "trigger_confirmation_decisions": int(value["trigger_confirmation_decisions"]),
            "normal_source_false_activation_rate": value["normal_source_false_activation_rate"],
            "failure_source_hit_rate": value["failure_source_hit_rate"],
            "calibration": str(calibration_path.resolve()),
            "task_ids": task_ids,
        }
    if len(task_routes) != 2519:
        raise ValueError(f"family routing covers {len(task_routes)} tasks instead of 2519")
    payload = {
        "kind": "libero10_family_gate_deployment_manifest_v1",
        "schema_version": 1,
        "family_index": str(args.family_index.resolve()),
        "shared_non_gate_checkpoint": str(args.shared_gate_checkpoint.resolve()),
        "shared_non_gate_checkpoint_sha256": sha256(args.shared_gate_checkpoint),
        "routing_contract": "task metadata selects a Gate head; task id is not a model input",
        "calibration_contract": "normal-source false activation after consecutive confirmations",
        "model_contract": reference_contract,
        "num_tasks": len(task_routes),
        "families": families,
        "task_to_family": task_routes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output.resolve()), "families": len(families),
        "tasks": len(task_routes),
    }, indent=2))


if __name__ == "__main__":
    main()
