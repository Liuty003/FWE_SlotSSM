#!/usr/bin/env python3
"""Match existing OFT/OFT+ reports against the official global manifest."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


REPORT_PATTERN = re.compile(
    r"(?P<suite>libero_(?:spatial|object|goal|10))-task(?P<task>\d+)-episode(?P<episode>\d+)\.json$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--search-root", type=Path, action="append", required=True,
        help="Recursively scan this root. Earlier roots win when duplicate reports are equally valid.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-contains", default="openvla-7b-oft-finetuned-libero-plus-mixdata",
    )
    return parser.parse_args()


def summary(rows: list[dict], field: str) -> dict[str, dict[str, int | float | None]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = "unassigned" if row[field] is None else str(row[field])
        groups[key].append(row)
    result = {}
    for key, values in sorted(groups.items()):
        covered = [row for row in values if row["baseline"] is not None]
        successes = sum(bool(row["baseline"]["success"]) for row in covered)
        result[key] = {
            "official": len(values),
            "covered": len(covered),
            "missing": len(values) - len(covered),
            "successes": successes,
            "success_rate_on_covered": None if not covered else successes / len(covered),
        }
    return result


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = manifest["records"]
    official = {row["sample_id"]: row for row in records}
    candidates: dict[str, list[dict]] = defaultdict(list)
    rejected = Counter()
    scanned = 0
    for root_index, root in enumerate(args.search_root):
        if not root.exists():
            rejected["missing_search_root"] += 1
            continue
        for path in root.rglob("*.json"):
            match = REPORT_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            scanned += 1
            sample_id = (
                f"{match.group('suite')}:task{int(match.group('task'))}:"
                f"episode{int(match.group('episode'))}"
            )
            if sample_id not in official:
                rejected["not_in_manifest"] += 1
                continue
            try:
                report = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                rejected["invalid_json"] += 1
                continue
            if "success" not in report:
                rejected["missing_success"] += 1
                continue
            checkpoint = str(report.get("checkpoint", ""))
            if args.checkpoint_contains and args.checkpoint_contains not in checkpoint:
                rejected["checkpoint_mismatch"] += 1
                continue
            policy = str(report.get("policy", ""))
            if policy not in {"openvla-oft", "openvla-oft+"}:
                rejected["policy_mismatch"] += 1
                continue
            declared = (
                report.get("task_suite"), report.get("task_id"),
                report.get("episode_id", report.get("episode")),
            )
            expected = (match.group("suite"), int(match.group("task")), int(match.group("episode")))
            if declared != expected:
                rejected["metadata_filename_mismatch"] += 1
                continue
            video = report.get("video")
            candidates[sample_id].append({
                "success": bool(report["success"]),
                "report": str(path.resolve()),
                "video": video,
                "video_exists": bool(video and Path(video).is_file()),
                "policy": policy,
                "checkpoint": checkpoint,
                "root_priority": root_index,
            })

    joined = []
    duplicate_conflicts = []
    for record in records:
        values = sorted(
            candidates.get(record["sample_id"], []),
            key=lambda row: (not row["video_exists"], row["root_priority"], row["report"]),
        )
        if values and len({row["success"] for row in values}) > 1:
            duplicate_conflicts.append({
                "sample_id": record["sample_id"],
                "reports": values,
            })
        chosen = None if not values else {k: v for k, v in values[0].items() if k != "root_priority"}
        joined.append({**record, "baseline": chosen, "num_compatible_reports": len(values)})

    covered = [row for row in joined if row["baseline"] is not None]
    missing = [row["sample_id"] for row in joined if row["baseline"] is None]
    successes = sum(row["baseline"]["success"] for row in covered)
    payload = {
        "kind": "libero_plus_baseline_inventory_v1",
        "manifest": str(args.manifest.resolve()),
        "search_roots": [str(path.resolve()) for path in args.search_root],
        "checkpoint_filter": args.checkpoint_contains,
        "scanned_filename_matches": scanned,
        "compatible_unique_instances": len(covered),
        "missing_instances": len(missing),
        "successes_on_covered": successes,
        "success_rate_on_covered": None if not covered else successes / len(covered),
        "rejected_counts": dict(sorted(rejected.items())),
        "duplicate_instances": sum(len(values) > 1 for values in candidates.values()),
        "duplicate_conflicts": duplicate_conflicts,
        "by_suite": summary(joined, "suite"),
        "by_perturbation": summary(joined, "perturbation_category"),
        "by_difficulty": summary(joined, "difficulty_level"),
        "missing_sample_ids": missing,
        "records": joined,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "official": len(joined),
        "covered": len(covered),
        "missing": len(missing),
        "successes": successes,
        "by_suite": payload["by_suite"],
        "by_perturbation": payload["by_perturbation"],
        "duplicate_conflicts": len(duplicate_conflicts),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
