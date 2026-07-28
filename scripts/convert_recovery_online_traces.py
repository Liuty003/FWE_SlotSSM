#!/usr/bin/env python
"""Convert selected online recovery traces to schema v2 for temporal training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovery.data.conversion import convert_online_trace
from recovery.data.dataset import write_jsonl
from recovery.data.schema import Split
from recovery.data.validation import audit_dataset


TASK9_LANGUAGE = (
    "pick up the black bowl on the wooden cabinet and place it on the plate"
)


def parse_ids(text: str) -> set[int]:
    return {int(value) for value in text.split(",") if value.strip()}


def infer_ids(path: Path) -> tuple[int, int]:
    match = re.search(r"task(\d+)-episode(\d+)", str(path))
    if match is None:
        raise ValueError(f"cannot infer task/episode IDs from {path}")
    return int(match.group(1)), int(match.group(2))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--validation-episodes", default="")
    parser.add_argument("--development-test-episodes", default="10,11,13")
    parser.add_argument("--final-test-episodes", default="")
    parser.add_argument("--task9-language", default=TASK9_LANGUAGE)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    validation = parse_ids(args.validation_episodes)
    development = parse_ids(args.development_test_episodes)
    final = parse_ids(args.final_test_episodes)
    if validation & development or validation & final or development & final:
        raise ValueError("episode split sets overlap")
    all_rows = []
    records = []
    seen = set()
    for value in args.traces:
        source = value if value.name == "steps.jsonl" else value / "steps.jsonl"
        task_id, episode_id = infer_ids(source)
        key = (task_id, episode_id)
        if key in seen:
            raise ValueError(f"duplicate online trace for task/episode {key}")
        seen.add(key)
        split = (
            Split.FINAL_TEST.value if episode_id in final
            else Split.DEVELOPMENT_TEST.value if episode_id in development
            else Split.VALIDATION.value if episode_id in validation
            else Split.TRAIN.value
        )
        rows = convert_online_trace(
            source,
            task_suite=args.task_suite,
            task_id=task_id,
            episode_id=episode_id,
            split=split,
            language_instruction=args.task9_language if task_id == 9 else None,
            source_revision=source.parents[1].name,
        )
        destination = (
            args.output_dir
            / f"task{task_id}-episode{episode_id}"
            / "transitions-v2.jsonl"
        )
        write_jsonl(destination, rows)
        all_rows.extend(rows)
        records.append(
            {
                "task_id": task_id,
                "episode_id": episode_id,
                "split": split,
                "source": str(source.resolve()),
                "output": str(destination.resolve()),
                "num_rows": len(rows),
                "aligned_media_available": False,
            }
        )
    audit = audit_dataset(all_rows, check_files=True)
    result = {
        "schema_version": 2,
        "purpose": "temporal labels and hard-negative traces; not RGB/depth training",
        "episodes": records,
        "audit": audit,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not audit["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
