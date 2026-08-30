#!/usr/bin/env python3
"""Build a V4.3/V4.4-compatible local handoff manifest from shared OFT audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-report", type=Path, action="append", required=True)
    parser.add_argument("--feature-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audits = [json.loads(path.read_text()) for path in args.audit_report]
    if any(audit.get("kind") != "relational_oft_handoff_audit_report_v1"
           or audit.get("complete") is not True for audit in audits):
        raise ValueError("every audit report must be a complete relational OFT handoff report")
    protocol_keys = (
        "task_suite", "normalization_key", "checkpoint", "action_head_sha256",
        "proprio_projector_sha256",
    )
    protocol = {key: audits[0].get(key) for key in protocol_keys}
    if any({key: audit.get(key) for key in protocol_keys} != protocol for audit in audits[1:]):
        raise ValueError("handoff audit reports use different OFT/normalization protocols")
    if any(
        result.get("kind") != "relational_oft_fresh_replan_handoff_audit_v1"
        or result.get("complete") is not True
        or result.get("run_fingerprint") is None
        for audit in audits for result in audit["records"]
    ):
        raise ValueError("handoff records require complete fingerprinted fresh-replan audits")
    attempts_by_source: dict[tuple[int, int], list[dict]] = {}
    for audit in audits:
        for result in audit["records"]:
            key = int(result["task_id"]), int(result["episode_id"])
            attempts_by_source.setdefault(key, []).append(result)
    records = []
    for (task_id, episode_id), results in sorted(attempts_by_source.items()):
        source = args.feature_source_root / f"task{task_id}-episode{episode_id}"
        metadata_path, steps_path = source / "metadata.json", source / "steps.jsonl"
        if not metadata_path.is_file() or not steps_path.is_file():
            raise FileNotFoundError(source)
        metadata = json.loads(metadata_path.read_text())
        if int(metadata.get("fork_timestep", -1)) != 0:
            raise ValueError(f"feature source must use branch-local fork 0: {source}")
        successful = sorted(
            (result for result in results if result["success"]),
            key=lambda result: int(result["teacher_relative_timestep"]),
        )
        certified = successful[0] if successful else None
        handoff = None if certified is None else int(certified["teacher_relative_timestep"])
        records.append({
            "task_id": task_id, "episode": episode_id,
            "source": str(source.resolve()), "fork_timestep": 0,
            "source_archival_fork_timestep": int(metadata["source_fork_timestep"]),
            "certified_local_handoff": handoff,
            "certified_recovery_steps": handoff,
            "status": "certified_local_recovery" if certified else "no_certified_oft_handoff",
            "recommended_split": "train" if certified else "excluded",
            "audit_attempts": [{
                "handoff_timestep": int(result["teacher_relative_timestep"]),
                "candidate_phase": result["candidate_phase"],
                "success": bool(result["success"]),
                "oft_suffix_env_steps": int(result["env_steps"]),
                "normalization_key": result["normalization_key"],
                "report": str((Path(result["video"]).parent / "report.json").resolve()),
                "video": result["video"], "protocol_compatible": True,
            } for result in sorted(results, key=lambda value: int(value["teacher_relative_timestep"]))],
        })
    certified = [record for record in records if record["status"] == "certified_local_recovery"]
    payload = {
        "kind": "v43_local_recovery_handoff_manifest", "schema_version": 1,
        "objective": "recover only until fresh-replan frozen OFT can resume and complete",
        "normal_policy": "frozen OpenVLA-OFT exact bypass",
        "recovery_target": "direct checkpoint-normalized single-step actions",
        "audit_rule": "restore exact teacher state, clear old chunk, fresh-replan frozen OFT",
        "candidate_phase": "earliest successful boundary across supplied audit reports",
        "certified_source_count": len(certified),
        "uncertified_source_count": len(records) - len(certified),
        "maximum_certified_recovery_steps": max(
            (record["certified_recovery_steps"] for record in certified), default=None,
        ),
        "recommended_latch_horizon_steps": None,
        "fixed_latch_all_sources_certified": False,
        "recommended_execution_horizon_steps": 1,
        "source_audit_reports": [str(path.resolve()) for path in args.audit_report],
        "audit_protocol": protocol,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output.resolve()), "certified": len(certified),
        "uncertified": len(records) - len(certified),
        "maximum_recovery_steps": payload["maximum_certified_recovery_steps"],
    }, indent=2))


if __name__ == "__main__":
    main()
