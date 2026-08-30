#!/usr/bin/env python3
"""Freeze the complete official 10,030-instance LIBERO-Plus index.

The classification JSON is the authority for suite-local task ids, official
perturbation categories, and difficulty levels.  This script deliberately does
not invent perturbations or group instances by hand-written task selectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_SUITE_COUNTS = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}
EXPECTED_CATEGORIES = {
    "Background Textures",
    "Camera Viewpoints",
    "Language Instructions",
    "Light Conditions",
    "Objects Layout",
    "Robot Initial States",
    "Sensor Noise",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classification", type=Path,
        default=Path("/data/lty/code/LIBERO-plus/libero/libero/benchmark/task_classification.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1/official-manifest.json"
        ),
    )
    parser.add_argument("--episode-id", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def grouped_counts(records: list[dict], key: str) -> dict[str, int]:
    return dict(sorted(Counter(
        "unassigned" if row[key] is None else str(row[key]) for row in records
    ).items()))


def effective_bddl_path(classification: Path, suite: str, name: str) -> Path:
    """Resolve virtual view/noise tasks to the BDDL file used by LIBERO-Plus."""
    bddl_root = classification.parent.parent / "bddl_files" / suite
    effective_name = re.sub(r"_view_.*$", "", name)
    return bddl_root / f"{effective_name}.bddl"


def main() -> None:
    args = parse_args()
    if args.episode_id < 0:
        raise ValueError("episode-id must be non-negative")
    classification = json.loads(args.classification.read_text())
    if set(classification) != set(EXPECTED_SUITE_COUNTS):
        raise RuntimeError(
            f"unexpected suites: expected {sorted(EXPECTED_SUITE_COUNTS)}, "
            f"got {sorted(classification)}"
        )

    records: list[dict] = []
    seen: set[str] = set()
    cross_counts: dict[str, Counter] = defaultdict(Counter)
    for suite, expected_count in EXPECTED_SUITE_COUNTS.items():
        entries = classification[suite]
        if len(entries) != expected_count:
            raise RuntimeError(f"expected {expected_count} {suite} entries, got {len(entries)}")
        expected_ids = list(range(1, expected_count + 1))
        actual_ids = [int(entry["id"]) for entry in entries]
        if actual_ids != expected_ids:
            raise RuntimeError(f"{suite} classification ids are not contiguous and ordered")
        for entry in entries:
            category = str(entry["category"])
            if category not in EXPECTED_CATEGORIES:
                raise RuntimeError(f"unknown perturbation category: {category}")
            task_id = int(entry["id"]) - 1
            raw_difficulty = entry["difficulty_level"]
            difficulty = None if raw_difficulty is None else int(raw_difficulty)
            if difficulty is not None and difficulty not in range(1, 6):
                raise RuntimeError(f"invalid difficulty level {difficulty} for {suite}:{task_id}")
            sample_id = f"{suite}:task{task_id}:episode{args.episode_id}"
            if sample_id in seen:
                raise RuntimeError(f"duplicate sample id: {sample_id}")
            seen.add(sample_id)
            name = str(entry["name"])
            bddl_path = effective_bddl_path(args.classification, suite, name)
            if not bddl_path.is_file():
                raise FileNotFoundError(
                    f"could not resolve BDDL for {suite}:{task_id} {name}: {bddl_path}"
                )
            records.append({
                "sample_id": sample_id,
                "suite": suite,
                "task_id": task_id,
                "classification_id": int(entry["id"]),
                "episode_id": args.episode_id,
                "official_name": name,
                "effective_bddl_file": str(bddl_path.resolve()),
                "perturbation_category": category,
                "difficulty_level": difficulty,
                "benchmark_task_key": f"{suite}:{task_id}",
            })
            cross_counts[suite][category] += 1

    if len(records) != 10030:
        raise RuntimeError(f"expected 10030 records, got {len(records)}")
    payload = {
        "kind": "libero_plus_official_global_manifest_v1",
        "schema_version": 1,
        "source": str(args.classification.resolve()),
        "source_sha256": sha256(args.classification),
        "task_id_convention": "suite-local task_id = classification_id - 1",
        "evaluation_unit": "one official task instance at the fixed episode_id",
        "episode_id": args.episode_id,
        "num_instances": len(records),
        "suite_counts": grouped_counts(records, "suite"),
        "perturbation_counts": grouped_counts(records, "perturbation_category"),
        "difficulty_counts": grouped_counts(records, "difficulty_level"),
        "suite_by_perturbation": {
            suite: dict(sorted(counts.items())) for suite, counts in sorted(cross_counts.items())
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "num_instances": len(records),
        "suite_counts": payload["suite_counts"],
        "perturbation_counts": payload["perturbation_counts"],
        "difficulty_counts": payload["difficulty_counts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
