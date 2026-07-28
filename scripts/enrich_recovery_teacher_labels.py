#!/usr/bin/env python
"""Add conservative simulator-assisted event labels to schema-v2 episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovery.data.dataset import read_jsonl, write_jsonl
from recovery.teachers.recovery_labeler import label_episode


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    sources = sorted(args.input_dir.glob("task*-episode*/transitions-v2.jsonl"))
    if not sources:
        raise ValueError(f"no schema-v2 episodes in {args.input_dir}")
    reports = []
    explicit_contact_rows = 0
    total_rows = 0
    for source in sources:
        rows = read_jsonl(source)
        total_rows += len(rows)
        explicit_contact_rows += sum(
            row.teacher_only.get("attached_truth") is not None for row in rows
        )
        report = label_episode(rows)
        destination = args.output_dir / source.relative_to(args.input_dir)
        write_jsonl(destination, rows)
        report["source"] = str(source.resolve())
        report["output"] = str(destination.resolve())
        reports.append(report)
    totals = {}
    for report in reports:
        for key, value in report["counts"].items():
            totals[key] = totals.get(key, 0) + value
    result = {
        "schema_version": 1,
        "kind": "simulator_teacher_event_label_enrichment",
        "claim_limit": (
            "privileged state generates supervision only and is not copied into "
            "policy inputs; attachment uses simulator bilateral-finger contact "
            "when present and otherwise falls back to a conservative co-motion proxy"
        ),
        "attachment_sources": {
            "sim_contact_rows": explicit_contact_rows,
            "comotion_proxy_rows": total_rows - explicit_contact_rows,
        },
        "episodes": reports,
        "totals": totals,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "teacher-label-report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
