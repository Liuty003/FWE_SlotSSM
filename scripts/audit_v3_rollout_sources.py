#!/usr/bin/env python3
"""Audit raw rollouts before they are admitted to the v3 feature pipeline.

V3 may use only fields that are available at policy time.  In particular, a
trajectory is extractable only when every recorded decision has *both* saved
camera images, proprioception, and the exact environment action that was
executed.  This small audit is intentionally stricter than the extractor so a
large historical runs directory can be screened without loading OFT.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-recovery-labels", action="store_true",
        help="Also require at least one valid RECOVERY label/target per source.",
    )
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def valid_action(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 7 and all(isinstance(x, (int, float)) for x in value)


def row_reason(row: dict[str, Any]) -> str | None:
    observation = row.get("online_observation")
    if not isinstance(observation, dict):
        return "no_online_observation"
    for key in ("rgb", "wrist_rgb"):
        value = observation.get(key)
        if not isinstance(value, str) or not Path(value).is_file():
            return f"no_saved_{key}"
    if not isinstance(observation.get("proprio"), dict):
        return "no_proprio"
    if not valid_action(row.get("executed_action")):
        return "no_exact_executed_action"
    return None


def main() -> None:
    args = parse_args()
    if not args.input_root.is_dir():
        raise NotADirectoryError(args.input_root)
    # Prefer steps; a successful rollout often writes both files and must count
    # as one source rather than as duplicate supervision.
    sources = sorted(args.input_root.rglob("steps.jsonl"))
    sources.extend(path for path in sorted(args.input_root.rglob("transitions.jsonl")) if not (path.parent / "steps.jsonl").exists())
    report: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    admitted = 0
    for path in sources:
        try:
            sequence = rows(path)
        except (OSError, json.JSONDecodeError) as error:
            report.append({"source": str(path), "admitted": False, "reason": f"unreadable:{error}"})
            reasons["unreadable"] += 1
            continue
        reason = next((reason for row in sequence if (reason := row_reason(row)) is not None), None)
        recovery_steps = sum(row.get("gate_label") in (1, "RECOVERY") or row.get("recovery_target_action_chunk") is not None for row in sequence)
        if reason is None and args.require_recovery_labels and recovery_steps == 0:
            reason = "no_recovery_supervision"
        item = {
            "source": str(path), "rows": len(sequence), "admitted": reason is None,
            "reason": reason, "recovery_steps": recovery_steps,
        }
        report.append(item)
        if reason is None:
            admitted += 1
        else:
            reasons[reason] += 1
    payload = {
        "kind": "v3_rollout_admission_audit", "input_root": str(args.input_root),
        "sources": len(report), "admitted": admitted, "rejected": len(report) - admitted,
        "reasons": dict(sorted(reasons.items())), "require_recovery_labels": args.require_recovery_labels,
        "policy_visible_requirements": ["saved_rgb", "saved_wrist_rgb", "proprio", "executed_action"],
        "details": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("sources", "admitted", "rejected", "reasons")}, indent=2))
    if admitted == 0:
        raise RuntimeError("no v3-admissible rollout sources found")


if __name__ == "__main__":
    main()
