#!/usr/bin/env python3
"""Build the versioned V4.3 local-recovery contract from OFT handoff audits."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SOURCES = {
    501: (49, "prefix-teachers/task501-t0049-fsmxyp0-placement08-rgbd", "train"),
    504: (55, "prefix-teachers/task504-t0055-fsmxyp0p25-placement00-rgbd", "excluded"),
    506: (49, "action-trace-teachers-r1/task506-episode0", "train"),
    510: (49, "action-trace-teachers-r1/task510-episode0", "train"),
    511: (44, "prefix-teachers/task511-t0044-fsmxyp25-placement04-rgbd", "val"),
    521: (49, "action-trace-teachers-r1/task521-episode0", "excluded"),
    523: (49, "action-trace-teachers-r1/task523-episode0", "val"),
    494: (48, "action-trace-teachers-book34-r2/task494-episode0", "train"),
    497: (0, "action-trace-teachers-book34-r2/task497-episode0", "train"),
    518: (0, "action-trace-teachers-book34-r2/task518-episode0", "train"),
    524: (0, "action-trace-teachers-book34-r2/task524-episode0", "val"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(
        "/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book"))
    parser.add_argument(
        "--audit-dir", type=Path, action="append",
        help="Audit directory to scan; repeat to combine independently collected audit batches.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    audit_dirs = args.audit_dir or [args.root / "handoff-audit-r1"]
    output = args.output or args.root / "v43-short-recovery-r1" / "handoff-manifest.json"
    attempts: dict[int, list[dict]] = {task: [] for task in SOURCES}
    report_paths = sorted({path for audit_dir in audit_dirs for path in audit_dir.glob("*/report.json")})
    for report_path in report_paths:
        report = json.loads(report_path.read_text())
        task = int(report["task_id"])
        if task not in attempts or not report.get("nominal_only"):
            continue
        match = re.search(r"prefix(\d+)", report_path.parent.name)
        handoff = int(report.get("prefix_steps") or (match.group(1) if match else -1))
        result = report["results"][0]
        attempts[task].append({
            "handoff_timestep": handoff,
            "success": bool(result["success"]),
            "oft_suffix_env_steps": int(result["env_steps"]),
            "wait_steps": report.get("wait_steps"),
            "prefix_rollout": report.get("prefix_rollout"),
            "report": str(report_path),
        })

    records = []
    for task, (fork, relative_source, recommended_split) in SOURCES.items():
        source = args.root / relative_source
        if not (source / "metadata.json").is_file():
            raise FileNotFoundError(source / "metadata.json")
        source_metadata = json.loads((source / "metadata.json").read_text())
        source_wait = source_metadata.get("wait_steps")
        tested = sorted(attempts[task], key=lambda row: row["handoff_timestep"])
        for row in tested:
            same_source = (row["prefix_rollout"] is not None and
                           Path(row["prefix_rollout"]).resolve() == source.resolve())
            same_wait = source_wait is None or row["wait_steps"] == source_wait
            row["protocol_compatible"] = bool(same_source and same_wait)
        successful = [row for row in tested if row["success"] and row["protocol_compatible"]]
        certified = min(successful, key=lambda row: row["handoff_timestep"]) if successful else None
        records.append({
            "task_id": task,
            "episode": 0,
            "source": str(source),
            "fork_timestep": fork,
            "certified_local_handoff": None if certified is None else certified["handoff_timestep"],
            "certified_recovery_steps": None if certified is None else certified["handoff_timestep"] - fork,
            "status": "certified_local_recovery" if certified else "no_certified_oft_handoff",
            "recommended_split": recommended_split if certified else "excluded",
            "audit_attempts": tested,
        })

    certified_records = [record for record in records if record["certified_local_handoff"] is not None]
    latch_horizon = max((record["certified_recovery_steps"] for record in certified_records), default=None)
    for record in certified_records:
        fixed_handoff = record["fork_timestep"] + latch_horizon
        fixed_attempt = next((attempt for attempt in record["audit_attempts"]
                              if attempt["handoff_timestep"] == fixed_handoff), None)
        record["fixed_latch_handoff"] = fixed_handoff
        record["fixed_latch_handoff_certified"] = bool(fixed_attempt and fixed_attempt["success"])
        record["fixed_latch_audit"] = fixed_attempt
    payload = {
        "kind": "v43_local_recovery_handoff_manifest",
        "schema_version": 1,
        "objective": "recover only until the frozen OFT nominal policy can resume and complete",
        "normal_policy": "frozen OpenVLA-OFT exact bypass",
        "recovery_target": "direct checkpoint-normalized action chunks",
        "audit_rule": "replay canonical teacher through candidate handoff, then run nominal-only frozen OFT",
        "certified_source_count": len(certified_records),
        "uncertified_source_count": len(records) - len(certified_records),
        "maximum_certified_recovery_steps": latch_horizon,
        "recommended_latch_horizon_steps": latch_horizon,
        "fixed_latch_all_sources_certified": all(
            record["fixed_latch_handoff_certified"] for record in certified_records),
        "recommended_execution_horizon_steps": 4,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output), "certified": len(certified_records),
                      "uncertified": len(records) - len(certified_records),
                      "latch_horizon": payload["recommended_latch_horizon_steps"]}))


if __name__ == "__main__":
    main()
