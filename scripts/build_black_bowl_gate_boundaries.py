#!/usr/bin/env python3
"""Export only high-confidence black-bowl failure boundaries for V4.3 gate data."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.failure_manifest.read_text())
    records = []
    excluded = []
    for row in source["records"]:
        failure = row.get("selected_failure")
        if failure is None:
            if row.get("baseline_success") is False:
                excluded.append({"task_id": row["task_id"], "status": row["status"]})
            continue
        records.append({"task_id": row["task_id"], "episode_id": row["episode"],
                        "failure_timestep": failure["timestep"],
                        "failure_type": failure["failure_type"]})
    payload = {"kind": "black_bowl_drawer_gate_boundaries_v1",
               "records": records, "excluded_needs_review": excluded}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"records": len(records), "excluded": len(excluded)}))


if __name__ == "__main__":
    main()
