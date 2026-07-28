#!/usr/bin/env python
"""Convert legacy recovery expert episodes into unified schema v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovery.data.conversion import (
    choose_latest_episode_sources,
    convert_legacy_expert_episode,
    load_legacy_rows,
)
from recovery.data.dataset import write_jsonl
from recovery.data.schema import Split
from recovery.data.validation import audit_dataset


TASK9_LANGUAGE = (
    "pick up the black bowl on the wooden cabinet and place it on the plate"
)


def parse_ids(text: str) -> set[int]:
    return {int(value) for value in text.split(",") if value.strip()}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-episodes", default="1")
    parser.add_argument("--development-test-episodes", default="9")
    parser.add_argument("--final-test-episodes", default="")
    parser.add_argument("--task9-language", default=TASK9_LANGUAGE)
    parser.add_argument("--skip-file-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    validation = parse_ids(args.validation_episodes)
    development = parse_ids(args.development_test_episodes)
    final = parse_ids(args.final_test_episodes)
    if validation & development or validation & final or development & final:
        raise ValueError("episode split sets overlap")

    sources = choose_latest_episode_sources(
        path if path.name == "transitions.jsonl" else path / "transitions.jsonl"
        for path in args.episodes
    )
    all_rows = []
    episode_records = []
    for source in sources:
        first = load_legacy_rows(source)[0]
        episode_id = int(first["episode_id"])
        split = (
            Split.FINAL_TEST.value if episode_id in final
            else Split.DEVELOPMENT_TEST.value if episode_id in development
            else Split.VALIDATION.value if episode_id in validation
            else Split.TRAIN.value
        )
        instruction = args.task9_language if int(first["task_id"]) == 9 else None
        rows = convert_legacy_expert_episode(
            source, split=split, language_instruction=instruction
        )
        destination = (
            args.output_dir
            / f"task{rows[0].task_id}-episode{rows[0].episode_id}"
            / "transitions-v2.jsonl"
        )
        write_jsonl(destination, rows)
        all_rows.extend(rows)
        episode_records.append(
            {
                "task_suite": rows[0].task_suite,
                "task_id": rows[0].task_id,
                "episode_id": rows[0].episode_id,
                "split": split,
                "source": str(source.resolve()),
                "output": str(destination.resolve()),
                "num_rows": len(rows),
            }
        )

    audit = audit_dataset(all_rows, check_files=not args.skip_file_check)
    manifest = {
        "schema_version": 2,
        "split_unit": "episode",
        "online_input_policy": "explicit_allowlist",
        "episodes": episode_records,
        "audit": audit,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not audit["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
