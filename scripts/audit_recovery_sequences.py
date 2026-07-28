#!/usr/bin/env python
"""Audit one or more schema-v2 recovery JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovery.data.dataset import read_jsonl
from recovery.data.validation import audit_dataset


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-file-check", action="store_true")
    return parser.parse_args()


def expand(value: Path) -> list[Path]:
    if value.is_file():
        return [value]
    return sorted(value.glob("task*-episode*/transitions-v2.jsonl"))


def main() -> None:
    args = arguments()
    paths = [path for value in args.inputs for path in expand(value)]
    if not paths:
        raise ValueError("no transitions-v2 JSONL files found")
    rows = []
    for path in paths:
        rows.extend(read_jsonl(path, check_files=False))
    result = {
        "inputs": [str(path.resolve()) for path in paths],
        "audit": audit_dataset(rows, check_files=not args.skip_file_check),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["audit"]["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
